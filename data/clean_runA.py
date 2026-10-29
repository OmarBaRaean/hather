"""Clean Run A output before it joins the pool.

1. Replace placeholder sender numbers (+966501234567, 0551234567, ...) with random,
   realistic Saudi numbers in the same format, so the student cannot learn
   "this exact number = scam/legit".
2. Drop near-empty messages (under 5 characters).
3. Drop messages that are mostly non-Arabic, unless they contain a link
   (link-only scam texts are realistic and kept).
4. Recompute `problems` per message (the notebook flagged all 5 messages of a
   generate job when only one of them was non-Arabic).

Usage:  python data/clean_runA.py data/out/runA_messages.jsonl data/out/runA_messages_clean.jsonl
"""
import json, random, re, sys
from collections import Counter

src, dst = sys.argv[1], sys.argv[2]
ARABIC = re.compile(r"[؀-ۿ]")
LINK = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|info|top|xyz|sa|co|ly|me|io|link|site|online|live|club|vip|cc|app|space|de|fi)\b", re.I)
PLACEHOLDER = re.compile(r"1234567|2345678|9876543|0000000|1111111|(\d)\1{5,}")

def arabic_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(bool(ARABIC.match(c)) for c in letters) / max(1, len(letters))

def new_number(old, rng):
    digits = "".join(rng.choice("0123456789") for _ in range(7))
    prefix = rng.choice(["50", "53", "54", "55", "56", "58", "59"])
    if old.replace(" ", "").startswith("+966"):
        return f"+966{prefix}{digits}"
    if old.startswith("05"):
        return f"0{prefix}{digits}"
    return old  # foreign placeholder formats are left alone (rare)

rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
kept, dropped, renumbered = [], Counter(), 0
for r in rows:
    rng = random.Random(r["id"])                      # same id -> same number on every run
    text = r["text"].strip()
    ratio = arabic_ratio(text)
    if len(text) < 5:
        dropped["too_short"] += 1; continue
    if ratio < 0.3 and not LINK.search(text):
        dropped["not_arabic"] += 1; continue
    if PLACEHOLDER.search(r["sender"]):
        new = new_number(r["sender"], rng)
        renumbered += new != r["sender"]; r["sender"] = new
    problems = [p for p in r.get("problems", []) if p != "not_arabic"]
    if ratio < 0.5:
        problems.append("not_arabic")
    r["problems"] = sorted(set(problems))
    kept.append(r)

with open(dst, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"{len(rows)} in, {len(kept)} kept, dropped: {dict(dropped)}, senders renumbered: {renumbered}")
print("labels:", dict(Counter(r["human_label"] for r in kept)))
print("problems left:", dict(Counter(p for r in kept for p in r["problems"])))
top = Counter(r["sender"] for r in kept).most_common(5)
print("most common senders now:", top)
