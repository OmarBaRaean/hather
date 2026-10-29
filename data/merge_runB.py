"""Merge the redo answers (v2 prompt) into the first Run B answers, after checking the control sample.

    python data/merge_runB.py data/out/runB_redo_original_labels.jsonl
    -> data/out/runB_v2b_original_labels.jsonl     all 9,737 messages, one answer each
    -> data/out/runB_redo_report.md                control check + what the redo recovered
    then: python data/split_runB.py data/out/runB_v2b_original_labels.jsonl data/out/runB_pool_v2.jsonl

- Control jobs (messages the teacher already got right with v1): their v1 answers are kept;
  the report compares how many verdicts the v2 prompt gets right. If v2 gives the right
  verdict on 5+ points fewer control scams than v1, the script stops: v2 made the teacher
  too trusting. (Fixed Oct 27: the first version also counted wrong-language explanations.)
- Redo jobs: the v2 answer replaces the v1 answer. split_runB.py then keeps it only if it
  matches the label, as before, so a wrong v2 answer is still dropped.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "data/out/runB_v2_original_labels.jsonl"
CONTROL_IDS = ROOT / "data/out/runB_redo_control_ids.txt"
OUT = ROOT / "data/out/runB_v2b_original_labels.jsonl"
REPORT = ROOT / "data/out/runB_redo_report.md"


def good(lab) -> bool:
    return (lab.get("ok") and lab["answer"]["verdict"] == lab["meta"].get("label")
            and not any(p in ("long_explanation", "wrong_language") for p in lab.get("problems", [])))


def main():
    redo_path = Path(sys.argv[1])
    first = {}
    for l in FIRST.open(encoding="utf-8"):
        lab = json.loads(l); first[lab["id"]] = lab
    redo = {}
    for l in redo_path.open(encoding="utf-8"):
        lab = json.loads(l); redo[lab["id"]] = lab
    control = set(CONTROL_IDS.read_text(encoding="utf-8").split())

    lines = ["# Run B redo with the v2 prompt", ""]
    # The control asks one thing: does v2 still give the right verdict? Language problems are
    # reported separately (they drop an answer from training, but say nothing about trusting scams).
    c1, c2, lang = Counter(), Counter(), Counter()
    for i in control & set(redo):
        lab = first[i]["meta"].get("label"); r = redo[i]
        c1[lab] += 1; c2[lab] += bool(r.get("ok") and r["answer"]["verdict"] == lab)
        lang[lab] += "wrong_language" in r.get("problems", [])
    lines += ["## Control: messages v1 already got right", "",
              "| Label | n | v1 verdict right | v2 verdict right | v2 wrong language |", "| --- | ---: | ---: | ---: | ---: |"]
    for lab in sorted(c1):
        lines.append(f"| {lab} | {c1[lab]} | 100% | {c2[lab] / c1[lab]:.1%} | {lang[lab]} |")
    scam_drop = 1 - c2["scam"] / max(1, c1["scam"])

    rec, tot = Counter(), Counter()
    merged = dict(first)
    for i, lab in redo.items():
        if i in control:
            continue
        merged[i] = lab
        k = (lab["meta"].get("origin"), lab["meta"].get("language"), lab["meta"].get("label"))
        tot[k] += 1; rec[k] += good(lab)
    lines += ["", "## Redo: messages the first run could not use", "",
              "| Origin | Language | Label | Re-asked | Now usable |", "| --- | --- | --- | ---: | ---: |"]
    for k in sorted(tot):
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {tot[k]} | {rec[k]} ({rec[k] / tot[k]:.0%}) |")
    lines.append(f"| **All** | | | **{sum(tot.values())}** | **{sum(rec.values())}** |")

    if scam_drop >= 0.05:
        lines += ["", f"**STOPPED:** v2 catches {scam_drop:.0%} fewer control scams than v1. "
                  "Do not use the redo answers; the v2 prompt needs another look."]
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines)); raise SystemExit(1)

    with OUT.open("w", encoding="utf-8") as f:
        for i in first:                                    # same order as the first run
            f.write(json.dumps(merged[i], ensure_ascii=False) + "\n")
    lines += ["", f"Merged -> `{OUT.name}` ({len(merged)} messages)."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
