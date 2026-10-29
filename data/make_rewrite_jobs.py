"""Teacher jobs that turn real (scrubbed) phone messages into scams in the same style.

    python data/make_rewrite_jobs.py
    -> data/out/rewrite_jobs.jsonl      run it with notebooks/teacher_runA.ipynb (JOBS file name)

Why: the real phone messages in training are all legit. If every scam is teacher-written,
the model can learn "sounds like a real company = safe". Each job takes one real message
and asks for a scam that copies its wording, brand and layout, plus one scam trick, so
real-looking style appears on both sides. Editing real messages is fine for training,
never for the test set.

- 650 Arabic + 150 English jobs from data/out/phone/real_train.jsonl, at most 3 per sender;
- the trick rotates over 6 kinds and the sender style over 4 (same sender ID, as real
  spoofed SMS do; a look-alike sender ID; a Saudi mobile; a foreign mobile), so neither
  the trick nor the sender gives the label away;
- jobs use the Run A "generate" format with one message each, so the Run A notebook runs
  them unchanged; `scenario` is the source message's sender group, so a rewrite and its
  original always land on the same side of the train/val split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = {"ar": 650, "en": 150}
MAX_PER_SENDER = 3

SYSTEM = """You turn a real, legitimate text message into a realistic scam that imitates it. The result is used to train a scam-detection model for people in Saudi Arabia, so it must look like a real scam that copies this real message.
Rules:
- Keep the original's language, wording style, layout, brand and length as far as possible; change only what the scam needs.
- Add exactly the scam trick you are given. Any link must be a fake look-alike domain (a misspelled brand, the brand plus extra words, or an odd ending such as .top, .xyz, .info), never the brand's real website.
- Keep placeholders such as <NAME>, <PHONE>, <IBAN>, <CARD>, <ID>, <ADDRESS>, <EMAIL> exactly as written.
- Do not add warnings, labels or explanations.
- Return JSON only."""

TRICKS = [
    "a fake look-alike link the reader must open to avoid losing a service or to confirm something",
    "a request to update card or bank details through a link, because of a problem or expiry",
    "a small fee to pay through a link (delivery, customs, renewal or a fine), with a deadline",
    "an urgent threat: the account, service or document will be suspended or fined today unless the reader acts",
    "a request to send back or read out a verification code, or to call a number, to 'cancel' or 'confirm' something",
    "a prize, refund or cashback the reader must claim through a link or by sharing details",
]
SENDERS = [
    ("same", "the same sender ID as the original (scammers spoof real sender IDs)"),
    ("lookalike", "a look-alike of the original sender ID (one letter changed, extra word, or different spelling)"),
    ("mobile_sa", "a Saudi mobile number in the form 05XXXXXXXX or +9665XXXXXXXX"),
    ("mobile_foreign", "a foreign mobile number with a non-Saudi country code"),
]
SCHEMA = {"type": "object", "properties": {"messages": {"type": "array", "minItems": 1, "maxItems": 1,
          "items": {"type": "object", "properties": {"sender": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["sender", "text"]}}}, "required": ["messages"]}
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "presence_penalty": 0.5, "max_tokens": 800}


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real", type=Path, nargs="?", default=ROOT / "data/out/phone/real_train.jsonl")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data/out/rewrite_jobs.jsonl")
    a = ap.parse_args(argv)
    real = [json.loads(l) for l in a.real.open(encoding="utf-8") if l.strip()]

    jobs = []
    for lang, n in N.items():
        per_sender, k = Counter(), 0
        for r in sorted((r for r in real if r["language"] == lang), key=lambda r: h("rw" + r["id"])):
            if len(jobs) >= sum(list(N.values())[:list(N).index(lang) + 1]):
                break
            s = r["sender"].lower()
            if per_sender[s] >= MAX_PER_SENDER or len(r["text"]) < 20:
                continue
            per_sender[s] += 1
            trick = TRICKS[k % len(TRICKS)]
            skind, sdesc = SENDERS[(k // len(TRICKS)) % len(SENDERS)]
            k += 1
            user = (f"Original message ({'Arabic' if lang == 'ar' else 'English'}):\n{r['text']}\n\n"
                    f"Original sender: {r['sender']}\n\n"
                    f"Scam trick to add: {trick}.\n"
                    f"Sender of the scam: {sdesc}.\n"
                    f"Write the scam in {'Arabic' if lang == 'ar' else 'English'}, following the rules.\n"
                    'Return: {"messages": [{"sender": "...", "text": "..."}]}')
            jobs.append({
                "job_id": f"rw-{lang}-{r['id']}",
                "kind": "generate", "label": "scam", "source": "sms",
                "parent_id": None, "scenario": r["scenario"],
                "language": lang, "rewrite_of": r["id"], "trick": TRICKS.index(trick), "sender_style": skind,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "schema": SCHEMA, "sampling": SAMPLING,
            })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"{len(jobs)} rewrite jobs -> {a.out}")
    print("  by language:", dict(Counter(j["language"] for j in jobs)))
    print("  by trick:", dict(sorted(Counter(j["trick"] for j in jobs).items())))
    print("  by sender style:", dict(Counter(j["sender_style"] for j in jobs)))
    print(f"  senders: {len({j['scenario'] for j in jobs})}")


if __name__ == "__main__":
    main()
