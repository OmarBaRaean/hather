"""Balance the Arabic part of train.jsonl to about 55% scam / 45% legit (decided Oct 27).

    python data/balance_train.py
    -> data/out/train_balanced.jsonl, data/out/train_balanced_report.md (counts only)

Why: the teacher drops more legit than scam messages, so Arabic training ended up about
58% scam. A model learns the mix it sees and would lean to "scam" on real phones, where
most messages are legit. 55/45 keeps it slightly cautious on purpose.

How: only teacher-written Arabic scams from the scenario list (Run A, scenario S..) are
removed, picked at random but spread evenly over the 20 scam scenarios, so every
scenario type stays. They are the least realistic scams and carry the most fake look-alike
links. Grounded scams, rewrites, translations and every real message are kept.
Only train.jsonl changes; val.jsonl stays as it is.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "data/out/train.jsonl", ROOT / "data/out/train_balanced.jsonl"
TARGET_SCAM_SHARE = 0.55


def main():
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    ar = [r for r in rows if r["language"] == "ar"]
    scam = sum(r["label"] == "scam" for r in ar); legit = len(ar) - scam
    target = round(legit * TARGET_SCAM_SHARE / (1 - TARGET_SCAM_SHARE))
    n_drop = max(0, scam - target)

    # candidates: Run A scenario scams (group "s:S..")
    by_scen = defaultdict(list)
    for r in rows:
        if r["language"] == "ar" and r["label"] == "scam" and r["origin"] == "teacher_generate" and r["group"].startswith("s:S"):
            by_scen[r["group"]].append(r["id"])
    rng = random.Random("balance-55-45")
    for ids in by_scen.values():
        rng.shuffle(ids)
    drop, scen = set(), sorted(by_scen)
    while len(drop) < n_drop and any(by_scen.values()):     # round-robin over scenarios
        for s in scen:
            if by_scen[s] and len(drop) < n_drop:
                drop.add(by_scen[s].pop())
    if len(drop) < n_drop:
        print(f"note: only {len(drop)} scenario scams available, wanted {n_drop}")

    kept = [r for r in rows if r["id"] not in drop]
    with DST.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mix(rs, lang):
        c = Counter(r["label"] for r in rs if r["language"] == lang)
        t = c["scam"] + c["legit"]
        return f"{c['scam']} scam / {c['legit']} legit ({c['scam'] / max(1, t):.0%} scam)"
    lines = ["# Balanced training set (counts only)", "",
             f"Removed {len(drop)} teacher-written Arabic scenario scams from {len(by_scen)} scenarios.", "",
             "| | Before | After |", "| --- | --- | --- |",
             f"| Arabic | {mix(rows, 'ar')} | {mix(kept, 'ar')} |",
             f"| English | {mix(rows, 'en')} | {mix(kept, 'en')} |",
             f"| Total | {len(rows)} | {len(kept)} |", "", "| Origin | After |", "| --- | ---: |"]
    lines += [f"| {o} | {n} |" for o, n in sorted(Counter(r["origin"] for r in kept).items())]
    (ROOT / "data/out/train_balanced_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
