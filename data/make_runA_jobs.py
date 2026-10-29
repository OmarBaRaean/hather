"""
Hather - Phase 3, Run A: build the job list for the teacher (runs on your laptop).

Creates data/out/runA_jobs.jsonl with two kinds of jobs:
  translate  1,500 jobs: one English SMS each (750 scam from Smishtank,
             750 legit from UCI ham) -> natural Saudi Arabic.
  generate     300 jobs: 5 new Arabic messages each, from scenarios_ar.md
             (150 scam jobs + 150 legit jobs = 1,500 messages).

Each job carries the full chat messages, the JSON schema and the sampling
settings, so the Kaggle notebook just sends it and saves the answer.

Usage (from the hather folder):
  python data/make_runA_jobs.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

N_TRANSLATE_PER_CLASS = 750
N_GENERATE_PER_CLASS = 750
MSGS_PER_GEN_JOB = 5

# ------------------------------------------------------------------ prompts

TRANSLATE_SYSTEM = """You rewrite text messages into natural Arabic, exactly as people in Saudi Arabia would receive them.
Rules:
- Keep the meaning, the intent and the tone. A scam stays a scam, a normal message stays normal. Do not add warnings, notes or explanations.
- Keep every URL, email address and code exactly as written, character for character.
- Replace foreign companies, agencies and places with Saudi equivalents (for example USPS or FedEx -> Saudi Post (SPL) or Aramex; IRS -> ZATCA; a US bank -> a Saudi bank such as Al Rajhi, SNB or Riyad Bank; dollars -> riyals; US cities -> Saudi cities).
- Write it like a real message, not like a translation. Companies usually write in clear Modern Standard Arabic; friends and family write in everyday Saudi dialect.
- Return JSON only."""

GENERATE_SYSTEM = """You write realistic Arabic text messages that people in Saudi Arabia receive. They are used to train a scam-detection model, so they must look exactly like real messages, including real scams.
Rules:
- Write in Arabic. Companies and government bodies usually write in clear Modern Standard Arabic; individuals write in everyday Saudi dialect. Some messages mix in English words, brand names or numbers, as real ones do.
- Scam messages copy the tricks real scammers use: urgency, fear, rewards, small fees, requests for codes or card details. Any link in a scam must be a fake look-alike domain (for example a misspelled brand, the brand plus extra words, or a strange ending such as .top or .xyz), never the brand's real official website.
- Legitimate messages look like what the real organisation or person actually sends: calm, specific, no request for passwords, codes or card details. If a legitimate message has a link, it is the official domain.
- Do not add warnings, labels or explanations inside the messages.
- Return JSON only."""

SENDER_STYLES = {
    "contact": "a saved contact name in Arabic (for example أبو فهد or ريم الشغل)",
    "mobile_sa": "a Saudi mobile number in the form 05XXXXXXXX or +9665XXXXXXXX",
    "mobile_foreign": "a foreign mobile number with a non-Saudi country code",
    "sender_id": "an alphanumeric SMS sender ID such as a short brand name in English letters",
    "shortcode": "a short numeric code (4 to 6 digits)",
}
# Sender style must NOT give away the label, so both classes use a mix.
SENDER_MIX = {
    ("translate", "scam"): ["mobile_sa", "mobile_foreign", "sender_id", "shortcode", "mobile_sa"],
    ("translate", "legit"): ["contact", "mobile_sa", "contact", "mobile_sa", "mobile_foreign"],
}

LENGTH_HINTS = [
    "Make one message very short (one line), one long (4-6 lines), and the rest in between.",
    "Vary the length: two short, two medium and one long message.",
    "Mostly medium length, with one very short message.",
]

SCHEMA_ONE = {
    "type": "object",
    "properties": {"sender": {"type": "string"}, "text": {"type": "string"}},
    "required": ["sender", "text"],
}
SCHEMA_MANY = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "minItems": MSGS_PER_GEN_JOB,
            "maxItems": MSGS_PER_GEN_JOB,
            "items": SCHEMA_ONE,
        }
    },
    "required": ["messages"],
}

SAMPLING_TRANSLATE = {"temperature": 0.4, "top_p": 0.8, "top_k": 20, "max_tokens": 700}
SAMPLING_GENERATE = {"temperature": 0.8, "top_p": 0.9, "top_k": 20,
                     "presence_penalty": 1.0, "max_tokens": 2000}

# ------------------------------------------------------------------ helpers

SCEN_RE = re.compile(r"^-\s*([SL]\d+)\s*\|\s*(scam|legit)\s*\|\s*(sms|whatsapp)\s*\|\s*(.+?)\s*\|\s*(.+)$")


def load_scenarios(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SCEN_RE.match(line.strip())
        if m:
            sid, label, channel, sender, desc = m.groups()
            out.append({"id": sid, "label": label, "channel": channel,
                        "sender": sender, "desc": desc})
    if not out:
        raise SystemExit(f"No scenarios found in {path}; check the line format.")
    return out


def translate_job(rec: dict, rng: random.Random) -> dict:
    style = rng.choice(SENDER_MIX[("translate", rec["human_label"])])
    user = (
        f"Original message (English):\n{rec['text']}\n\n"
        f"Original sender: {rec['sender'] or '(unknown)'}\n\n"
        f"Rewrite the message in Arabic following the rules. "
        f"For the sender, use {SENDER_STYLES[style]}, consistent with the rewritten message.\n"
        'Return: {"sender": "...", "text": "..."}'
    )
    return {
        "job_id": f"tr-{rec['id']}",
        "kind": "translate",
        "label": rec["human_label"],
        "source": "sms",
        "parent_id": rec["id"],
        "scenario": None,
        "messages": [{"role": "system", "content": TRANSLATE_SYSTEM},
                     {"role": "user", "content": user}],
        "schema": SCHEMA_ONE,
        "sampling": SAMPLING_TRANSLATE,
    }


def generate_job(scen: dict, k: int, rng: random.Random) -> dict:
    kind = "scam" if scen["label"] == "scam" else "legitimate"
    channel = "SMS" if scen["channel"] == "sms" else "WhatsApp"
    user = (
        f"Write {MSGS_PER_GEN_JOB} different {kind} {channel} messages for this scenario.\n"
        f"Sender / brand: {scen['sender']}\n"
        f"Scenario: {scen['desc']}\n\n"
        f"Make the {MSGS_PER_GEN_JOB} messages clearly different from each other in wording, "
        f"details (amounts, dates, names, cities) and sender. {rng.choice(LENGTH_HINTS)}\n"
        "For each message give the sender exactly as it would appear on the phone "
        "(a sender ID, a short code, a phone number or a contact name).\n"
        'Return: {"messages": [{"sender": "...", "text": "..."}, ...]}'
    )
    return {
        "job_id": f"gen-{scen['id']}-{k:02d}",
        "kind": "generate",
        "label": scen["label"],
        "source": scen["channel"],
        "parent_id": None,
        "scenario": scen["id"],
        "messages": [{"role": "system", "content": GENERATE_SYSTEM},
                     {"role": "user", "content": user}],
        "schema": SCHEMA_MANY,
        "sampling": SAMPLING_GENERATE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/out/pool.jsonl")
    ap.add_argument("--scenarios", default="data/scenarios_ar.md")
    ap.add_argument("--out", default="data/out/runA_jobs.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    sms = [r for r in pool if r["source"] == "sms"]
    scam = [r for r in sms if r["human_label"] == "scam"]
    # Skip tiny chat lines ("ok lar...") that are useless to translate.
    legit = [r for r in sms if r["human_label"] == "legit" and len(r["text"]) >= 25]
    rng.shuffle(scam)
    rng.shuffle(legit)
    if len(scam) < N_TRANSLATE_PER_CLASS or len(legit) < N_TRANSLATE_PER_CLASS:
        print(f"! only {len(scam)} scam / {len(legit)} legit SMS available")
    jobs = [translate_job(r, rng) for r in scam[:N_TRANSLATE_PER_CLASS]]
    jobs += [translate_job(r, rng) for r in legit[:N_TRANSLATE_PER_CLASS]]

    scenarios = load_scenarios(Path(args.scenarios))
    for label in ("scam", "legit"):
        scens = [s for s in scenarios if s["label"] == label]
        n_jobs = N_GENERATE_PER_CLASS // MSGS_PER_GEN_JOB
        for i in range(n_jobs):                       # spread jobs evenly over scenarios
            scen = scens[i % len(scens)]
            jobs.append(generate_job(scen, i // len(scens), rng))

    rng.shuffle(jobs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    c = Counter((j["kind"], j["label"]) for j in jobs)
    n_msgs = sum(1 if j["kind"] == "translate" else MSGS_PER_GEN_JOB for j in jobs)
    print(f"{len(scenarios)} scenarios loaded")
    for k, v in sorted(c.items()):
        print(f"  {k[0]:<9} {k[1]:<5} {v:>5} jobs")
    print(f"  total {len(jobs)} jobs -> {n_msgs} Arabic messages -> {args.out}")


if __name__ == "__main__":
    main()
