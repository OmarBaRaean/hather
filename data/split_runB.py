"""After Run B: keep the answers the teacher got right and split them 90/10 into train/val.

    python data/split_runB.py data/out/runB_abliterated_labels.jsonl data/out/runB_pool.jsonl

Writes, next to the labels file:
    train.jsonl, val.jsonl   chat examples for the student: system / user (the Phase 2 input) / assistant (teacher JSON)
    runB_agreement.md        agreement rate per origin, language and label (for the write-up)

Rules:
- Keep an item only if the teacher answered, its verdict matches the label (human label
  for real data, intended label for teacher-written data), and its explanation is 40
  words or fewer and in the message's language.
- Split by group, never by message, so near-copies cannot sit on both sides:
  an English SMS and its Arabic translation share a group (parent_id), and all
  messages written from one scenario or grounded seed share a group (scenario).
- Real phone messages (assumed legit) that the teacher calls a scam are not trained on;
  they are listed in runB_phone_flags.jsonl for your review.
- Within each origin + language, groups go to validation (in a fixed pseudo-random
  order) until it holds 10% of that stratum's messages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_runB_jobs import SYSTEM_PROMPT  # noqa: E402

VAL_SHARE = 0.10


def group_of(rec: dict) -> str:
    if rec.get("parent_id"):
        return "p:" + rec["parent_id"]         # translation -> its English original
    if rec.get("scenario"):
        return "s:" + rec["scenario"]          # generated or grounded -> its scenario / seed
    return "p:" + rec["id"]                    # real message (an English original joins its translations)


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path, help="<jobs>_<teacher>_labels.jsonl from the notebook")
    ap.add_argument("pool", type=Path, help="runB_pool.jsonl (the records the jobs were built from)")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    out_dir = a.out_dir or a.labels.parent

    pool = {}
    for line in a.pool.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line); pool[r["id"]] = r

    total, agree, kept_n = Counter(), Counter(), Counter()
    dropped = Counter()
    kept = []
    flags = []   # real phone messages (assumed legit) that the teacher called a scam: for your review
    for line in a.labels.open(encoding="utf-8"):
        if not line.strip():
            continue
        lab = json.loads(line)
        rec = pool.get(lab["id"])
        if rec is None:
            dropped["not in pool"] += 1; continue
        label = lab["meta"].get("label")
        key = (rec.get("origin", "?"), rec.get("language", "?"), label or "?")
        total[key] += 1
        if not lab.get("ok"):
            dropped["teacher failed"] += 1; continue
        if lab["answer"]["verdict"] != label:
            dropped["teacher disagreed"] += 1
            if str(rec.get("origin", "")).startswith("phone"):     # real message the teacher calls a scam
                flags.append({"id": rec["id"], "origin": rec["origin"], "sender": rec.get("sender"),
                              "text": rec["text"], "teacher": lab["answer"]})
            continue
        agree[key] += 1
        bad = [p for p in lab.get("problems", []) if p in ("long_explanation", "wrong_language")]
        if bad:
            dropped[bad[0]] += 1; continue
        kept_n[key] += 1
        kept.append((rec, lab["answer"]))

    # group-aware split, stratified by origin + language
    groups = defaultdict(list)
    for rec, ans in kept:
        groups[group_of(rec)].append((rec, ans))
    strata = defaultdict(list)
    for g, items in groups.items():
        first = items[0][0]
        strata[(first.get("origin"), first.get("language"))].append(g)
    val_groups = set()
    for s, gs in strata.items():
        size = sum(len(groups[g]) for g in gs)
        n = 0
        for g in sorted(gs, key=h):
            if n >= VAL_SHARE * size:
                break
            val_groups.add(g); n += len(groups[g])

    files = {"train": out_dir / "train.jsonl", "val": out_dir / "val.jsonl"}
    counts = Counter()
    handles = {k: p.open("w", encoding="utf-8") for k, p in files.items()}
    for g, items in sorted(groups.items()):
        split = "val" if g in val_groups else "train"
        for rec, ans in items:
            ex = {"id": rec["id"], "group": g,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": rec["input"]},
                               {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}],
                  "origin": rec.get("origin"), "language": rec.get("language"), "label": ans["verdict"]}
            handles[split].write(json.dumps(ex, ensure_ascii=False) + "\n")
            counts[(split, rec.get("origin"), rec.get("language"), ans["verdict"])] += 1
    for f in handles.values():
        f.close()

    lines = ["# Run B: teacher agreement with the labels", "",
             f"Teacher answers: `{a.labels.name}`", "",
             "| Origin | Language | Label | Messages | Teacher agrees | Kept for training |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for k in sorted(total):
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {total[k]} | {agree[k]/total[k]:.1%} | {kept_n[k]} |")
    T = sum(total.values())
    lines.append(f"| **All** | | | **{T}** | **{sum(agree.values())/max(T,1):.1%}** | **{sum(kept_n.values())}** |")
    lines += ["", "Dropped: " + ", ".join(f"{v} {k}" for k, v in dropped.most_common()), ""]
    lines += ["| Split | Origin | Language | Label | Examples |", "| --- | --- | --- | --- | ---: |"]
    for k in sorted(counts):
        lines.append(f"| {' | '.join(map(str, k))} | {counts[k]} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "runB_agreement.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"train: {sum(v for k, v in counts.items() if k[0] == 'train')}  "
          f"val: {sum(v for k, v in counts.items() if k[0] == 'val')}  -> {out_dir}")

    if flags:
        fp = out_dir / "runB_phone_flags.jsonl"
        with fp.open("w", encoding="utf-8") as f:
            for x in flags:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{len(flags)} real phone messages the teacher called a scam -> {fp} (review them: a real scam "
              f"belongs in the test set, not in training)")

    # leakage check: no group on both sides
    tr = {json.loads(l)["group"] for l in files["train"].open(encoding="utf-8")}
    va = {json.loads(l)["group"] for l in files["val"].open(encoding="utf-8")}
    assert not tr & va, "a group is in both train and val"


if __name__ == "__main__":
    main()
