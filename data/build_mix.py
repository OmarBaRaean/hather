"""Build the v2 training pool (decided Oct 24): fewer teacher-written Arabic legit
messages, real phone messages added, and teacher rewrites of real messages into scams.

    python data/build_mix.py                                   # without rewrites (until they are run)
    python data/build_mix.py --rewrites data/out/kaggle_rewrite/runA_messages.jsonl
    -> data/out/runB_pool_v2.jsonl, data/out/runB_pool_v2_report.md (counts only)
    python data/make_runB_jobs.py data/out/runB_pool_v2.jsonl -o data/out/runB_jobs_v2.jsonl

The v1 files (runB_pool.jsonl, runB_jobs.jsonl) are left as they are.

What changes against v1:
- Arabic translated legit: 746 -> 250 (whole translation pairs, picked by a fixed hash);
- Arabic legit from the scenario list: 734 -> about 300 (whole jobs of 5, spread evenly
  over all 20 legit scenarios, so every scenario type stays);
- Arabic legit from news seeds: 1,645 -> about 1,000 (whole seeds; the 2 official-alert
  seeds are always kept);
- every scam stays; real phone legit from data/out/phone/real_train.jsonl is added;
- rewrites (if given) are added as origin `teacher_rewrite`, label scam, with the
  language of their job and the sender group of the real message they copy;
- every record gets `parent_id` / `scenario` back from sms_pool.jsonl (apply_rules.py
  drops them, and split_runB.py needs them to keep translation pairs, scenarios and
  senders on one side of the train/val split).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rules"))
from engine import explain, format_input  # noqa: E402

KEEP_TRANSLATED_LEGIT = 250
KEEP_GENERATED_LEGIT_JOBS = 60          # x 5 messages = about 300
KEEP_GROUNDED_LEGIT_SEEDS = 200         # x about 5 messages = about 1,000
ARABIC = re.compile(r"[ء-ي]")
PLACEHOLDER = re.compile(r"1234567|2345678|3456789|4567890|9876543|7911123456|2025550|555\d{4}|(\d)\1{5,}")   # example numbers the teacher likes


def h(s: str) -> str:
    return hashlib.sha1(("mix-v2" + s).encode()).hexdigest()


def load(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def arabic_ratio(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return sum(bool(ARABIC.match(c)) for c in letters) / max(1, len(letters))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/out/runB_pool.jsonl")
    ap.add_argument("--sms-pool", type=Path, default=ROOT / "data/out/sms_pool.jsonl")
    ap.add_argument("--real", type=Path, default=ROOT / "data/out/phone/real_train.jsonl")
    ap.add_argument("--rewrites", type=Path, default=None, help="runA_messages.jsonl from the rewrite run")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data/out/runB_pool_v2.jsonl")
    a = ap.parse_args(argv)

    meta = {r["id"]: (r.get("parent_id"), r.get("scenario")) for r in load(a.sms_pool)}
    pool = load(a.pool)
    for r in pool:
        r["parent_id"], r["scenario"] = meta.get(r["id"], (None, None))
    before = Counter((r["origin"], r["language"], r["human_label"]) for r in pool)

    def part(r):
        if r["origin"] == "teacher_translate" and r["human_label"] == "legit":
            return "tr_legit"
        if r["origin"] == "teacher_generate" and r["human_label"] == "legit":
            return "gr_legit" if str(r["scenario"]).startswith("G") else "gen_legit"
        return "keep"

    groups = defaultdict(lambda: defaultdict(list))
    out = []
    for r in pool:
        p = part(r)
        if p == "keep":
            out.append(r)
        elif p == "tr_legit":
            groups[p][r["parent_id"]].append(r)
        elif p == "gen_legit":
            groups[p][r["id"].rsplit("-", 1)[0]].append(r)        # job id gen-L03-02
        else:
            groups[p][r["scenario"]].append(r)

    keys = sorted(groups["tr_legit"], key=h)[:KEEP_TRANSLATED_LEGIT]
    out += [r for k in keys for r in groups["tr_legit"][k]]

    by_scen = defaultdict(list)                                   # spread jobs over scenarios
    for job in sorted(groups["gen_legit"], key=h):
        by_scen[job.split("-")[1]].append(job)
    jobs, i = [], 0
    while len(jobs) < KEEP_GENERATED_LEGIT_JOBS and any(by_scen.values()):
        for s in sorted(by_scen):
            if by_scen[s] and len(jobs) < KEEP_GENERATED_LEGIT_JOBS:
                jobs.append(by_scen[s].pop(0))
    out += [r for j in jobs for r in groups["gen_legit"][j]]

    seeds = sorted(groups["gr_legit"], key=h)
    alerts = [s for s in seeds if any("legit_alert" in r["id"] for r in groups["gr_legit"][s])]
    rest = [s for s in seeds if s not in alerts]
    chosen = alerts + rest[:KEEP_GROUNDED_LEGIT_SEEDS - len(alerts)]
    out += [r for s in chosen for r in groups["gr_legit"][s]]

    real = load(a.real)
    out += real

    n_rw, rw_drop = 0, Counter()
    if a.rewrites:
        for r in load(a.rewrites):
            if not r["id"].startswith("rw-"):
                continue
            lang = r["id"].split("-")[1]
            text = (r.get("text") or "").strip()
            if len(text) < 5:
                rw_drop["empty"] += 1; continue
            if lang == "ar" and arabic_ratio(text) < 0.5:
                rw_drop["not Arabic"] += 1; continue
            if lang == "en" and arabic_ratio(text) > 0.2:
                rw_drop["not English"] += 1; continue
            sender = r.get("sender", "").strip()
            if PLACEHOLDER.search(sender.replace(" ", "")):
                rng = random.Random(r["id"])                       # same id -> same number every run
                s = sender.replace(" ", "")
                if s.startswith(("+966", "05")):
                    body = "5" + rng.choice("0345689") + "".join(rng.choice("0123456789") for _ in range(7))
                    sender = ("+966" if s.startswith("+966") else "0") + body
                else:                                              # foreign: keep the country code, new digits
                    m = re.match(r"(\+?\d{1,3})(\d+)$", s)
                    if m:
                        cc, rest = m.groups()
                        sender = cc + rest[0] + "".join(rng.choice("0123456789") for _ in rest[1:])
            rec = {"id": r["id"], "source": "sms", "language": lang, "sender": sender, "subject": "",
                   "text": text, "human_label": "scam", "origin": "teacher_rewrite", "eml_path": None,
                   "parent_id": None, "scenario": r.get("scenario")}
            rec["signals"] = explain(rec)["signals"]
            rec["input"] = format_input(rec)
            out.append(rec); n_rw += 1

    ids = Counter(r["id"] for r in out)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate ids: {dup[:5]}")
    with a.out.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    after = Counter((r["origin"], r["language"], r["human_label"]) for r in out)
    rows = sorted(set(before) | set(after))
    lines = ["# Training pool v2 (counts only)", "", "| Origin | Language | Label | v1 | v2 |", "| --- | --- | --- | ---: | ---: |"]
    lines += [f"| {o} | {l} | {lab} | {before[(o, l, lab)]} | {after[(o, l, lab)]} |" for o, l, lab in rows]
    tot = lambda c, lang=None, lab=None: sum(v for (o, l, b), v in c.items() if (lang is None or l == lang) and (lab is None or b == lab))
    lines += ["", f"Total: v1 {sum(before.values())}, v2 {sum(after.values())}",
              f"Arabic scam/legit: v1 {tot(before, 'ar', 'scam')}/{tot(before, 'ar', 'legit')}, "
              f"v2 {tot(after, 'ar', 'scam')}/{tot(after, 'ar', 'legit')}",
              f"English scam/legit: v1 {tot(before, 'en', 'scam')}/{tot(before, 'en', 'legit')}, "
              f"v2 {tot(after, 'en', 'scam')}/{tot(after, 'en', 'legit')}",
              f"Rewrites added: {n_rw}" + (f" (dropped: {dict(rw_drop)})" if rw_drop else "") +
              ("" if a.rewrites else " (not run yet: pass --rewrites once they are)")]
    a.out.with_name(a.out.stem + "_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
