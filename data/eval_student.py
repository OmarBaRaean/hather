"""Score the student (or any GGUF) on the labelled test set, on this computer only.

    # 1. start the model server (CPU) in the background
    python data/eval_student.py serve  Models/Qwen3.5-0.8B.Q4_K_M.gguf
    # 2. answer the test jobs; resumable, so run it again until it says "all done"
    python data/eval_student.py answer eval/test_partial_jobs.jsonl --budget 160
    # 3. the scores
    python data/eval_student.py report eval/test_partial_jobs.jsonl
    # 4. stop the server
    python data/eval_student.py stop

Answers go to eval/<jobs>_answers.jsonl (gitignored). The jobs use the same system prompt and input
as training (thinking off, temperature 0), so this is what the app would see.

Report: per language, scam recall (scams caught) and false-alarm rate (legit flagged), each with a
95% Wilson interval, the app colours (red: scam with confidence 70+, amber: confidence under 70,
green: legit 70+; a lookalike_domain signal is never green), invalid JSON and wrong-language counts.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / ".wheels/llama/llama-server"
PORT = 8091
PIDFILE = ROOT / ".wheels/llama/server.pid"
LOG = ROOT / ".wheels/llama/server.log"
ARABIC = re.compile(r"[ء-ي]")


def ar(t: str) -> float:
    l = [c for c in t if c.isalpha()]
    return sum(bool(ARABIC.match(c)) for c in l) / max(1, len(l))


def healthy() -> bool:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).status == 200
    except Exception:
        return False


def serve(model: str):
    if healthy():
        print("server already running"); return
    cmd = [str(BIN), "-m", model, "-c", "4096", "--parallel", "1", "-t", str(os.cpu_count() or 2),
           "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}', "--port", str(PORT),
           "--host", "127.0.0.1", "--cache-reuse", "256"]
    p = subprocess.Popen(cmd, stdout=open(LOG, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    PIDFILE.write_text(str(p.pid))
    for i in range(60):
        if healthy():
            print(f"server ready after {i * 2}s (pid {p.pid})"); return
        if p.poll() is not None:
            print(LOG.read_text()[-2000:]); raise SystemExit("server crashed")
        time.sleep(2)
    raise SystemExit("server did not start; see " + str(LOG))


def stop():
    if PIDFILE.exists():
        try:
            os.killpg(int(PIDFILE.read_text()), signal.SIGTERM)
        except Exception:
            pass
        PIDFILE.unlink()
    print("stopped")


def out_path(jobs: Path) -> Path:
    return jobs.with_name(jobs.stem.replace("_jobs", "") + "_answers.jsonl")


def answer(jobs_file: Path, budget: float):
    if not healthy():
        raise SystemExit("start the server first: python data/eval_student.py serve <model.gguf>")
    jobs = [json.loads(l) for l in jobs_file.open(encoding="utf-8") if l.strip()]
    out = out_path(jobs_file)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()}
    todo = [j for j in jobs if j["job_id"] not in done]
    t0, n = time.time(), 0
    with out.open("a", encoding="utf-8") as f:
        for j in todo:
            if time.time() - t0 > budget:
                break
            body = {"model": "student", "messages": j["messages"], "temperature": 0, "max_tokens": 300,
                    "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": j["schema"]}},
                    "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True}
            t = time.time()
            rec = {"id": j["job_id"], "meta": j["meta"]}
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                content = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
                rec.update(ok=True, answer=json.loads(content), seconds=round(time.time() - t, 1))
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1
    left = len(todo) - n
    per = (time.time() - t0) / max(1, n)
    print(f"answered {n} in {time.time() - t0:.0f}s ({per:.1f}s each); "
          + ("all done" if left == 0 else f"{left} left (about {left * per / 60:.0f} min): run again"))


def wilson(k: int, n: int):
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def report(jobs_file: Path):
    jobs = {json.loads(l)["job_id"]: json.loads(l) for l in jobs_file.open(encoding="utf-8") if l.strip()}
    rows = [json.loads(l) for l in out_path(jobs_file).open(encoding="utf-8") if l.strip()]
    c, colours, probs = Counter(), Counter(), Counter()
    for r in rows:
        m = r["meta"]; lang, lab = m.get("language"), m.get("label")
        if not r.get("ok"):
            probs["failed or invalid JSON"] += 1; continue
        a = r["answer"]; v, conf = a.get("verdict"), a.get("confidence", 0)
        c[(lang, lab, v)] += 1
        sig = m.get("rule_signals") or []
        colour = "red" if v == "scam" and conf >= 70 else ("green" if v == "legit" and conf >= 70 else "amber")
        if colour == "green" and "lookalike_domain" in sig:
            colour = "amber"
        colours[(lang, lab, colour)] += 1
        text = jobs[r["id"]]["messages"][-1]["content"].split("[TEXT]", 1)[-1]
        e = a.get("explanation", "")
        if (ar(text) >= 0.8 and ar(e) < 0.5) or (ar(text) < 0.2 and ar(e) >= 0.5):
            probs["wrong language"] += 1
        if len(e.split()) > 40:
            probs["explanation over 40 words"] += 1
    print(f"{len(rows)} of {len(jobs)} test messages answered\n")
    print("| Language | Measure | Result | 95% range | Ship rule |")
    print("| --- | --- | --- | --- | --- |")
    for lang in ("ar", "en"):
        s_ok, s_n = c[(lang, "scam", "scam")], c[(lang, "scam", "scam")] + c[(lang, "scam", "legit")]
        l_fa, l_n = c[(lang, "legit", "scam")], c[(lang, "legit", "legit")] + c[(lang, "legit", "scam")]
        if s_n:
            lo, hi = wilson(s_ok, s_n)
            print(f"| {lang} | scams caught | {s_ok}/{s_n} = {s_ok/s_n:.0%} | {lo:.0%}-{hi:.0%} | 90%+ |")
        if l_n:
            lo, hi = wilson(l_fa, l_n)
            print(f"| {lang} | false alarms on legit | {l_fa}/{l_n} = {l_fa/l_n:.0%} | {lo:.0%}-{hi:.0%} | 10% or less |")
    print("\nApp colours:")
    for lang in ("ar", "en"):
        for lab in ("scam", "legit"):
            t = sum(colours[(lang, lab, k)] for k in ("red", "amber", "green"))
            if t:
                print(f"  {lang} {lab:5}: red {colours[(lang, lab, 'red')]}, amber {colours[(lang, lab, 'amber')]}, "
                      f"green {colours[(lang, lab, 'green')]}")
    print("\nproblems:", dict(probs) or "none")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "serve":
        serve(sys.argv[2])
    elif cmd == "stop":
        stop()
    elif cmd == "answer":
        budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e9
        answer(Path(sys.argv[2]), budget)
    elif cmd == "report":
        report(Path(sys.argv[2]))
    else:
        print(__doc__)
