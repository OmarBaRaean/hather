"""
Hather - grounded generation: build teacher jobs from the scrape_news extract (runs on your laptop).

Every job is seeded by something real from data/out/scrape/scam_items.jsonl:
  pattern   one scam pattern described in a news article (SMS/WhatsApp only)
  fragment  short phrases quoted from one real scam campaign
  foreign   a full scam message reported outside Saudi Arabia, adapted to Saudi context
and, to keep the classes balanced and hard:
  legit_brand  legitimate messages from the same organisation a scam pattern impersonated
  legit_alert  legitimate official alerts, seeded by real government alerts the extract found

Articles that contain the real test-set messages are excluded completely, so nothing
from the test set leaks into training.

Outputs (same job format as make_runA_jobs.py, so the Run A notebook runs them unchanged):
  data/out/grounded_jobs.jsonl    jobs for the teacher
  data/out/grounded_seeds.jsonl   one line per job: the real seed, its article URL and date

Usage (from the hather folder):
  python data/make_grounded_jobs.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_runA_jobs import (GENERATE_SYSTEM, LENGTH_HINTS, MSGS_PER_GEN_JOB,  # noqa: E402
                            SAMPLING_GENERATE, SCHEMA_MANY)

# The real, complete Arabic scam messages kept for the TEST set. Any article quoting one
# of these is excluded from seeding.
TEST_MARKERS = [
    "بشرى سارة لعملاء محفظة", "شركة سالك أن نعلمكم", "عندي مشكل فحد التحويلات",
    "عندي مشكل ف السقف", "مخالفة مواقف غير مسددة", "التدقيق في استمارات",
    "الاستمارات التي لا تحتوي", "ربحت عمرة", "أنا أرملة", "أخي مريض",
]

# Extracted "messages" that are not scam messages (headlines, victim quotes, tweets,
# reply instructions, non-SMS scams). Matched as substrings.
NOT_A_SEED = [
    "السلطات الإيرانية", "أجهزة الأمن الإيرانية", "Recently I was targeted", "منصة مالية تستخدم اسمي",
    "زافين", "قيومجيان", "كويومدجيان", "حوّلت بالخطأ", "وصلتني حوالة", "على إثر ما تم تداوله",
    "قالوا", "إن ادخل على الموقع", "خذ ثانية وتحقق", "اختر ما يناسبك", "نصب واحتيال",
    "رابط بث مباشر كأس العالم", "أرسل STOP", "أرسل NO", "Review Activity",
]
# Real legitimate official alerts the extract caught; they seed legit_alert jobs.
OFFICIAL_ALERT_MARKERS = ["تهديد صاروخي", "الوضع آمن حاليا"]

PATTERN_NOT_A_SEED = re.compile(r"zero-click|webkit|spyware|malware download|strait|مضيق|بلوكتشين", re.I)

SAUDI_RULE = (
    "Set the messages in Saudi Arabia: if the original organisation, service, currency or place "
    "is foreign, replace it with the closest Saudi equivalent (for example a Saudi bank, Absher, "
    "SPL or Aramex, ZATCA, the Saudi Electricity Company, riyals, Saudi cities)."
)
VARY_RULE = (
    f"Make the {MSGS_PER_GEN_JOB} messages clearly different from each other in wording, details "
    "(amounts, dates, names, cities) and sender. {length}\n"
    "For each message give the sender exactly as it would appear on the phone "
    "(a sender ID, a short code, a phone number or a contact name).\n"
    'Return: {{"messages": [{{"sender": "...", "text": "..."}}, ...]}}'
)


def chan(c: str) -> str:
    return "WhatsApp" if c == "whatsapp" else "SMS"


def job(job_id, label, channel, scenario_id, user):
    return {
        "job_id": job_id,
        "kind": "generate",
        "label": label,
        "source": channel if channel in ("sms", "whatsapp") else "sms",
        "parent_id": None,
        "scenario": scenario_id,
        "messages": [{"role": "system", "content": GENERATE_SYSTEM},
                     {"role": "user", "content": user}],
        "schema": SCHEMA_MANY,
        "sampling": SAMPLING_GENERATE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/out/scrape/scam_items.jsonl")
    ap.add_argument("--out", default="data/out/grounded_jobs.jsonl")
    ap.add_argument("--seeds-out", default="data/out/grounded_seeds.jsonl")
    ap.add_argument("--legit-ratio", type=float, default=1.0,
                    help="legit_brand jobs per scam pattern job (1.0 = balanced)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    rel = [r for r in recs if r.get("relevant") and "error" not in r]

    # ---- exclude test-set articles
    test_ids = {r["article_id"] for r in rel
                if any(k in m["text"] for m in r.get("messages", []) for k in TEST_MARKERS)}
    rel = [r for r in rel if r["article_id"] not in test_ids]

    jobs, seeds = [], []

    def add(kind, label, channel, seed, r, user):
        sid = f"G{len(jobs):04d}"
        jobs.append(job(f"gr-{kind}-{sid}", label, channel, sid, user))
        seeds.append({"scenario": sid, "seed_kind": kind, "label": label, "channel": channel,
                      "seed": seed, "article_id": r["article_id"], "url": r["url"],
                      "published": r.get("published", ""), "source": r.get("source", ""),
                      "country": r.get("country", "")})

    # ---- 1. pattern jobs (scam) + brand-matched legit jobs
    seen = set()
    patterns = []
    for r in rel:
        for p in r.get("patterns", []):
            if p["channel"] not in ("sms", "whatsapp"):
                continue
            if PATTERN_NOT_A_SEED.search(p["description"] + p["impersonates"]):
                continue
            key = p["description"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            patterns.append((r, p))

    for r, p in patterns:
        user = (
            f"Write {MSGS_PER_GEN_JOB} different scam {chan(p['channel'])} messages based on this real "
            f"scam reported in the news ({r.get('country') or 'unknown country'}).\n"
            f"Report: {p['description']}\n"
            f"Impersonates: {p['impersonates']}\n"
            f"Lure: {p['lure']}\n\n"
            f"{SAUDI_RULE}\n"
            + VARY_RULE.format(length=rng.choice(LENGTH_HINTS))
        )
        add("pattern", "scam", p["channel"], p, r, user)

    n_legit = round(len(patterns) * args.legit_ratio)
    for r, p in rng.sample(patterns, min(n_legit, len(patterns))):
        who = p["impersonates"]
        if who.lower() in ("unknown", "غير محدد", "جهات غير محددة"):
            who = "a real Saudi bank, government service or delivery company that fits this topic"
        user = (
            f"Write {MSGS_PER_GEN_JOB} different legitimate {chan(p['channel'])} messages that the real "
            f"organisation below actually sends to its customers in Saudi Arabia about the same topic "
            f"or a related routine service. Scammers copy this organisation, so the real messages "
            f"should look official and some may be time-sensitive, but they never ask for passwords, "
            f"verification codes or card details, and never ask for payment through a link in the message.\n"
            f"Organisation: {who}\n"
            f"Topic: {p['lure']}\n\n"
            f"{SAUDI_RULE}\n"
            + VARY_RULE.format(length=rng.choice(LENGTH_HINTS))
        )
        add("legit_brand", "legit", p["channel"], p, r, user)

    # ---- 2. fragment jobs (short phrases from one campaign) and foreign full messages
    frag_by_article = defaultdict(list)
    long_by_article = defaultdict(list)
    alerts, seen_text = [], set()
    for r in rel:
        for m in r.get("messages", []):
            t = m["text"].strip()
            if t in seen_text:
                continue
            seen_text.add(t)
            if any(k in t for k in OFFICIAL_ALERT_MARKERS):
                alerts.append((r, m)); continue
            if any(k in t or k in m.get("impersonates", "") for k in NOT_A_SEED) or len(t) < 6:
                continue
            (frag_by_article if len(t) < 40 else long_by_article)[r["article_id"]].append((r, m))

    # one "foreign" job per article: its quoted message, or the parts of it the article quoted
    for aid, items in long_by_article.items():
        r = items[0][0]
        channel = items[0][1]["channel"]
        quoted = "\n".join(f"- {m['text']}" for _, m in items)
        user = (
            f"A real scam reported in the news ({r.get('country') or 'unknown country'}) was quoted "
            f"like this, possibly translated into Arabic (one message, or parts of it):\n{quoted}\n\n"
            f"Write {MSGS_PER_GEN_JOB} different scam {chan(channel)} messages that use the same "
            f"trick, adapted to people in Saudi Arabia. Do not copy the original wording.\n"
            f"{SAUDI_RULE}\n"
            + VARY_RULE.format(length=rng.choice(LENGTH_HINTS))
        )
        add("foreign", "scam", channel, {"quoted": [m["text"] for _, m in items],
                                          "impersonates": items[0][1]["impersonates"]}, r, user)

    for aid, items in frag_by_article.items():
        r = items[0][0]
        phrases = "\n".join(f"- {m['text']}" for _, m in items)
        who = next((m["impersonates"] for _, m in items if m["impersonates"].lower() != "unknown"), "unknown")
        channel = items[0][1]["channel"]
        user = (
            f"These short phrases were quoted from a real scam campaign reported in the news "
            f"({r.get('country') or 'unknown country'}):\n{phrases}\n"
            f"Impersonates: {who}\n\n"
            f"Write {MSGS_PER_GEN_JOB} different complete scam {chan(channel)} messages from this kind of "
            f"campaign, aimed at people in Saudi Arabia. Use these phrases or close variants naturally "
            f"inside full messages.\n"
            f"{SAUDI_RULE}\n"
            + VARY_RULE.format(length=rng.choice(LENGTH_HINTS))
        )
        add("fragment", "scam", channel, {"phrases": [m["text"] for _, m in items],
                                           "impersonates": who}, r, user)

    # ---- 3. legit official alerts (hard negatives), a few jobs from the real alerts found
    for i, (r, m) in enumerate(alerts):
        user = (
            "This is a real, legitimate official alert sent to residents by a Gulf government:\n"
            f"{m['text']}\n\n"
            f"Write {MSGS_PER_GEN_JOB} different legitimate official SMS alerts that Saudi authorities "
            "really send (for example civil defense, the National Center for Meteorology, the Ministry "
            "of Interior, Absher). They are urgent and official but never contain a link to pay, never "
            "ask for personal data and never ask the reader to reply with codes.\n"
            + VARY_RULE.format(length=rng.choice(LENGTH_HINTS))
        )
        add("legit_alert", "legit", "sms", m, r, user)

    # ---- write
    order = list(range(len(jobs)))
    rng.shuffle(order)
    out, seeds_out = Path(args.out), Path(args.seeds_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(jobs[i], ensure_ascii=False) + "\n")
    with seeds_out.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    c = Counter((s["seed_kind"], s["label"]) for s in seeds)
    print(f"{len(test_ids)} test-set articles excluded")
    for (k, l), v in sorted(c.items()):
        print(f"  {k:<12} {l:<5} {v:>4} jobs")
    lab = Counter(s["label"] for s in seeds)
    print(f"  total {len(jobs)} jobs -> {len(jobs) * MSGS_PER_GEN_JOB} messages "
          f"({lab['scam'] * MSGS_PER_GEN_JOB} scam / {lab['legit'] * MSGS_PER_GEN_JOB} legit)")
    print(f"  -> {out}\n  -> {seeds_out}")


if __name__ == "__main__":
    main()
