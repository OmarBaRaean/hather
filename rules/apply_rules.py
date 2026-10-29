"""Phase 2 step 3: run the rules engine over message files and add `signals` and `input`.

    python rules/apply_rules.py data/out/pool.jsonl data/out/runA_messages_clean.jsonl \
        data/out/grounded_messages_clean.jsonl -o data/out/runB_pool.jsonl

Email records (source == "email") are skipped (v2). Every other record is written
unchanged plus two fields:
    signals  list of signal names
    input    the formatted [SOURCE]/[SENDER]/[SIGNALS]/[TEXT] block

It then prints how often each signal fires per group and label. Look for
shortcuts: a signal that fires on many scams but almost no legit messages in
one group (e.g. unknown_sender on English Smishtank vs UCI ham, which has no
sender) is something the model could learn as "signal = scam".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import SIGNAL_ORDER, format_input, signals  # noqa: E402

LABEL_KEYS = ("human_label", "label", "intended_label")
SHORT = {"lookalike_domain": "lookalike", "brand_link_mismatch": "mismatch",
         "link_shortener": "shortener", "link_ip": "ip",
         "foreign_number_local_brand": "foreign", "unknown_sender": "unknown"}


def label_of(rec: dict) -> str:
    for k in LABEL_KEYS:
        if rec.get(k):
            return str(rec[k])
    return "?"


def group_of(rec: dict) -> str:
    return f"{rec.get('origin') or rec.get('source') or '?'}/{rec.get('language') or '?'}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    n_out = n_email = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if (rec.get("source") or "sms") == "email":
                        n_email += 1
                        continue
                    sigs = signals(rec)
                    rec["signals"] = sigs
                    rec["input"] = format_input(rec, sigs)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    key = (group_of(rec), label_of(rec))
                    totals[key] += 1
                    counts[key].update(sigs)
                    counts[key]["(none)"] += not sigs
                    n_out += 1

    print(f"wrote {n_out} records to {args.out} (skipped {n_email} emails)\n")
    cols = [SHORT[s] for s in SIGNAL_ORDER] + ["none"]
    print(f"{'group':28} {'label':6} {'n':>6} " + " ".join(f"{c:>9}" for c in cols))
    for key in sorted(totals):
        n = totals[key]
        row = [counts[key][s] / n for s in SIGNAL_ORDER] + [counts[key]["(none)"] / n]
        print(f"{key[0]:28} {key[1]:6} {n:>6} " + " ".join(f"{r:>8.0%} " for r in row))
    print("\nRates are the share of messages in that row where the signal fired.")


if __name__ == "__main__":
    main()
