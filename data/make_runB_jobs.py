"""Build Run B jobs: one teacher request per message (verdict, confidence, signals, explanation).

The teacher sees only the formatted input from the rules engine, never the label.
Each job also carries `meta` (label, origin, language, ...) so the notebook can
report agreement without needing the pool.

    # 1. the training pool
    python data/make_runB_jobs.py data/out/runB_pool.jsonl -o data/out/runB_jobs.jsonl
    # 2. the 100 English test SMS, to compare the abliterated and original teacher
    python data/make_runB_jobs.py eval/gold_candidates_en.jsonl -o data/out/runB_compare_jobs.jsonl
    # 3. the final test set, once it is frozen (measures the teacher; never trained on)
    python data/make_runB_jobs.py eval/gold.jsonl -o data/out/runB_test_jobs.jsonl

Records without an `input` field are run through the rules engine first.
Email records are skipped.

--no-sender blanks every sender before the rules run. Use it for a second scoring
of the English test SMS: the UCI legit items never had senders while Smishtank
scams do, so with senders shown, "has a phone number" could pass for skill.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rules"))
from engine import format_input, signals  # noqa: E402

# Shared with the student's training data (Phase 4) and the app: keep in sync.
# v1, used for the first full Run B (Oct 25): kept for reference and the write-up.
SYSTEM_PROMPT_V1 = """You are a fraud analyst for users in Saudi Arabia. Decide whether the SMS or WhatsApp message is a scam or legit.
The input has four lines: [SOURCE] sms or whatsapp; [SENDER] the sender as shown on the phone ((none) if unknown); [SIGNALS] facts checked by code; [TEXT] the message.
Use the [SIGNALS] line as verified facts, but weigh them: a legit message can come from an unsaved number or from abroad, and a scam can have no signals at all, so judge the wording too (urgency, threats, prizes, requests for codes, payments or personal data).
Signal meanings: lookalike_domain = a link imitates a known brand's domain; brand_link_mismatch = the message names a brand but links elsewhere; link_shortener = the link hides its destination; link_ip = the link is a bare IP address; foreign_number_local_brand = a Saudi brand named, but the sender is a non-Saudi number; unknown_sender = the sender is a phone number, not a saved contact or business sender ID.
Return JSON only:
{"verdict": "scam" | "legit", "confidence": 0-100,
 "signals": [the facts that drove your decision, short phrases],
 "explanation": "40 words or fewer, addressed to the user, in the language of the [TEXT]: an English message gets an English explanation and an Arabic message an Arabic one"}"""

# v2 (Oct 26): adds what normal Saudi messages look like (OTPs, alerts, official short links), after the
# teacher called 1 in 8 real company messages a scam for reasons like "an OTP with no context".
SYSTEM_PROMPT = """You are a fraud analyst for users in Saudi Arabia. Decide whether the SMS or WhatsApp message is a scam or legit.
The input has four lines: [SOURCE] sms or whatsapp; [SENDER] the sender as shown on the phone ((none) if unknown); [SIGNALS] facts checked by code; [TEXT] the message.
Use the [SIGNALS] line as verified facts, but weigh them: a legit message can come from an unsaved number or from abroad, and a scam can have no signals at all, so judge the wording too.
Signal meanings: lookalike_domain = a link imitates a known brand's domain; brand_link_mismatch = the message names a brand but links elsewhere; link_shortener = the link hides its destination; link_ip = the link is a bare IP address; foreign_number_local_brand = a Saudi brand named, but the sender is a non-Saudi number; unknown_sender = the sender is a phone number, not a saved contact or business sender ID.
Most messages people receive are legit. Call a message a scam only when it tries to get the reader to do something harmful: open a fake link, share a code, password or card details, send money, or call or reply to a stranger.
Normal, and not a sign of a scam on its own:
- a one-time code (OTP) sent to the reader, even with no other context, especially with a warning not to share it;
- alerts about purchases, transfers, balances, bills, deliveries, appointments, fines or government services;
- links on official domains (the brand's own site, .gov.sa, .edu.sa), including the brand's own short links;
- ads and offers from a company.
Signs of a scam: asking the reader to share or send back a code; asking for card, bank or login details through a link or a reply; pressure to pay or act now to avoid a penalty or a blocked account; a prize, refund or gift the reader did not ask for; a brand named but the link or sender does not belong to it. A familiar sender ID does not prove a message is real, because scammers can copy sender IDs, so check what the message asks the reader to do.
Base the explanation only on what the message says and on the [SIGNALS] line; do not invent facts.
Return JSON only:
{"verdict": "scam" | "legit", "confidence": 0-100,
 "signals": [the facts that drove your decision, short phrases],
 "explanation": "40 words or fewer, addressed to the user, in the language of the [TEXT]: an English message gets an English explanation and an Arabic message an Arabic one"}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["scam", "legit"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "signals": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "explanation": {"type": "string"},
    },
    "required": ["verdict", "confidence", "signals", "explanation"],
    "additionalProperties": False,
}

SAMPLING = {"temperature": 0.3, "top_p": 0.95, "max_tokens": 400}

META_KEYS = ("origin", "language", "source", "parent_id", "scenario", "check")


def label_of(r: dict) -> str | None:
    return r.get("human_label") or r.get("label") or r.get("intended_label")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--no-sender", action="store_true", help="blank all senders (see above)")
    a = ap.parse_args(argv)

    n, skipped, seen = 0, Counter(), set()
    stats = Counter()
    with a.out.open("w", encoding="utf-8") as out:
        for line in a.input.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("source") or "sms") == "email":
                skipped["email"] += 1; continue
            if not (r.get("text") or "").strip():
                skipped["empty"] += 1; continue
            if r["id"] in seen:
                skipped["duplicate id"] += 1; continue
            seen.add(r["id"])
            if a.no_sender:
                r = {**r, "sender": "", "input": None}
            if not r.get("input"):
                sigs = signals(r)
                r["signals"], r["input"] = sigs, format_input(r, sigs)
            job = {
                "job_id": r["id"],
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": r["input"]}],
                "schema": SCHEMA,
                "sampling": SAMPLING,
                "meta": {"label": label_of(r), "rule_signals": r.get("signals", []),
                         **{k: r.get(k) for k in META_KEYS if r.get(k) is not None}},
            }
            out.write(json.dumps(job, ensure_ascii=False) + "\n")
            n += 1
            stats[(r.get("origin") or "?", r.get("language") or "?", label_of(r) or "?")] += 1

    print(f"{n} jobs -> {a.out}" + (f"  (skipped {dict(skipped)})" if skipped else ""))
    for k, v in sorted(stats.items()):
        print(f"  {'/'.join(k):40} {v}")


if __name__ == "__main__":
    main()
