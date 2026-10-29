"""Re-ask the teacher, with the v2 system prompt, only for the messages Run B could not use.

    python data/make_redo_jobs.py
    -> data/out/runB_redo_jobs.jsonl        run it with teacher_runB.ipynb (JOBS_NAME = "runB_redo_jobs.jsonl")
    -> data/out/runB_redo_control_ids.txt   which jobs are only a check (their old answers are kept)

Which messages:
- every message whose first answer was dropped: the teacher disagreed with the label,
  answered in the wrong language, or failed;
- plus a control sample of 200 messages the teacher already got right (100 scam, 100 legit,
  spread over all groups). They are not used for training again; they show whether the v2
  prompt still catches scams as well as v1 did. If it catches clearly fewer, v2 made the
  teacher too trusting and the redo answers should not be used.

Then: python data/merge_runB.py (see there).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/out/runB_v2_original_labels.jsonl"
POOL = ROOT / "data/out/runB_pool_v2.jsonl"
OUT_POOL = ROOT / "data/out/runB_redo_pool.jsonl"
OUT_JOBS = ROOT / "data/out/runB_redo_jobs.jsonl"
CONTROL_IDS = ROOT / "data/out/runB_redo_control_ids.txt"
N_CONTROL = {"scam": 100, "legit": 100}


def main():
    pool = {}
    for l in POOL.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l); pool[r["id"]] = r
    redo, good = [], defaultdict(list)
    for l in LABELS.open(encoding="utf-8"):
        lab = json.loads(l)
        label = lab["meta"].get("label")
        bad = (not lab.get("ok") or lab["answer"]["verdict"] != label
               or any(p in ("long_explanation", "wrong_language") for p in lab.get("problems", [])))
        if bad:
            redo.append(lab["id"])
        else:
            good[(label, lab["meta"].get("origin"), lab["meta"].get("language"))].append(lab["id"])

    rng = random.Random("redo-control-v1")
    control = []
    for label, n in N_CONTROL.items():
        groups = sorted(k for k in good if k[0] == label)
        per = max(1, n // len(groups))
        picked = []
        for g in groups:
            ids = sorted(good[g]); rng.shuffle(ids); picked += ids[:per]
        rest = sorted(set(i for g in groups for i in good[g]) - set(picked)); rng.shuffle(rest)
        control += (picked + rest)[:n]

    with OUT_POOL.open("w", encoding="utf-8") as f:
        for i in redo + control:
            f.write(json.dumps(pool[i], ensure_ascii=False) + "\n")
    CONTROL_IDS.write_text("\n".join(control) + "\n", encoding="utf-8")
    print(f"{len(redo)} to re-ask + {len(control)} control = {len(redo) + len(control)} jobs")
    subprocess.run([sys.executable, str(ROOT / "data/make_runB_jobs.py"), str(OUT_POOL), "-o", str(OUT_JOBS)], check=True)


if __name__ == "__main__":
    main()
