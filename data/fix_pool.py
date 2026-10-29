"""Fix two label shortcuts in the SMS/WhatsApp pool before Run B.

1. UCI ham has no senders, while 68% of Smishtank scams come from phone numbers,
   so `unknown_sender` alone would separate English scam from legit. UCI ham gets
   senders drawn from the same mix as Smishtank: about 68% phone numbers (copying
   the format and country of a random Smishtank number, with new digits), 22%
   contact names and 10% none. Seeded by message id, so it is reproducible.

2. Some teacher-written legit messages use domains the teacher invented
   (stc.sa, aramex.sa, spl.com.sa...), so `lookalike_domain` fires on "legit"
   messages and the model would learn to ignore it. Each such link host is
   replaced by the brand's real domain (from rules/brands.json); if a message
   still fires `lookalike_domain` after that, it is dropped.

Usage:
    python data/fix_pool.py data/out/pool.jsonl data/out/runA_messages_clean.jsonl \
        data/out/grounded_messages_clean.jsonl -o data/out/sms_pool.jsonl
    python rules/apply_rules.py data/out/sms_pool.jsonl -o data/out/runB_pool.jsonl

Email records are skipped. Each changed record gets a `fixes` list.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rules"))
from engine import classify_sender, explain, extract_links, load_brands, lookalike_reason  # noqa: E402

CONTACT_NAMES = [
    "Mum", "Dad", "Mom", "Sis", "Bro", "John", "Sarah", "Mike", "Emma", "Dave", "Chris",
    "Laura", "James", "Kate", "Tom", "Anna", "Ben", "Lucy", "Dan", "Rachel", "Alex",
    "Jess", "Sam", "Nick", "Hannah", "Matt", "Sophie", "Rob", "Amy", "Jack", "Grandma",
    "Uncle Pete", "Aunt Mary", "Work - Steve", "Gym buddy", "Neighbour", "Babe", "Wei Ming",
    "Priya", "Raj", "Hui Min", "Ahmad", "Kumar", "Li Ting", "Jason", "Mel", "Karen", "Paul",
]


def fake_number_like(template: str, rng: random.Random) -> str:
    """Same format and country code as the template, new subscriber digits."""
    digits = [i for i, c in enumerate(template) if c.isdigit()]
    keep = set(digits[: max(1, len(digits) - 7)])  # keep country/area prefix
    return "".join(c if (i in keep or not c.isdigit()) else rng.choice("0123456789")
                   for i, c in enumerate(template))


def fix_uci_senders(records: list[dict]) -> int:
    phone_templates = sorted({r["sender"] for r in records
                              if r.get("origin") == "smishtank" and classify_sender(r["sender"]) == "phone"})
    if not phone_templates:
        return 0
    n = 0
    for r in records:
        if r.get("origin") != "uci_ham" or (r.get("sender") or "").strip():
            continue
        rng = random.Random("uci-sender-" + r["id"])
        x = rng.random()
        if x < 0.68:
            r["sender"] = fake_number_like(rng.choice(phone_templates), rng)
        elif x < 0.90:
            r["sender"] = rng.choice(CONTACT_NAMES)
        else:
            continue
        r.setdefault("fixes", []).append("uci_sender_added")
        n += 1
    return n


def official_host(brand_id: str, old_host: str) -> str:
    brand = next(b for b in load_brands() if b.id == brand_id)
    new = brand.domains[0]
    return ("www." + new) if old_host.startswith("www.") and not new.startswith("www.") else new


def fix_invented_domains(records: list[dict]) -> tuple[int, list[dict]]:
    fixed, dropped, kept = 0, [], []
    for r in records:
        if r.get("human_label") != "legit":
            kept.append(r); continue
        text, changed = r["text"], False
        named = explain(r)["brands"]  # brands named in the text or by the sender
        for link in extract_links(text)[0]:
            reason = lookalike_reason(link)
            if not reason or reason == "punycode":
                continue
            # Prefer the brand the message names: sender "SE" + saudi-e.com means
            # Saudi Electricity, even though "saudie" is one letter from "saudia".
            brand_id = reason.split(":")[0]
            if named and brand_id not in named:
                brand_id = named[0]
            new = official_host(brand_id, link.host)
            text = re.sub(re.escape(link.host), new, text, count=1, flags=re.I)
            changed = True
        if changed:
            r = {**r, "text": text, "fixes": r.get("fixes", []) + ["legit_domain_made_official"]}
            if "lookalike_domain" in explain(r)["signals"]:
                dropped.append(r); continue
            fixed += 1
        kept.append(r)
    return fixed, kept, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args(argv)

    records, emails = [], 0
    for p in a.inputs:
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("source") or "sms") == "email":
                emails += 1; continue
            r.pop("signals", None); r.pop("input", None)  # recomputed by apply_rules.py
            records.append(r)

    n_senders = fix_uci_senders(records)
    n_domains, records, dropped = fix_invented_domains(records)

    with a.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} records -> {a.out} (skipped {emails} emails)")
    print(f"UCI ham senders added: {n_senders}")
    kinds = Counter(classify_sender(r.get("sender")) for r in records if r.get("origin") == "uci_ham")
    print(f"  UCI ham sender kinds now: {dict(kinds)}")
    print(f"legit messages with invented domains fixed: {n_domains}, dropped: {len(dropped)}")
    for r in dropped:
        print("  dropped:", r["id"], explain(r)["reasons"].get("lookalike_domain"))


if __name__ == "__main__":
    main()
