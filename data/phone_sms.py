"""Turn an SMS Backup & Restore export into test-set candidates. Runs on your laptop only.

    python data/phone_sms.py "data/raw/phones/sms-<name>.xml" -o data/out/phone/candidates.jsonl \
        [--yemen-from 2025-07-01 --yemen-to 2025-08-15]

What it keeps (received messages only; your sent messages are skipped):
- business messages: the sender is a name/ID (AlRajhiBank, STC) or a short code (902)
  and is not a saved contact;
- scam suspects from unsaved phone numbers: only if they have a link, or ask for
  money, a code or personal data. Everything else from phone numbers or saved
  contacts is personal and is skipped without being written anywhere.

Then it:
- groups messages by template (digits ignored) and keeps at most 3 per template,
  so "تم خصم مبلغ" alerts do not fill the test set;
- hides phone numbers, IBANs and long account/card numbers inside the text
  (names are not detected: check for them when you label);
- marks each message's country: ye (Yemeni brand/currency, or received inside the
  --yemen dates), sa (Saudi brand/currency/number) or unknown;
- runs the rules engine, and flags near-copies of the training pool (Rule 2);
- prints counts only, never message text.

Each candidate has `human_label: ""`: fill in scam / legit / drop by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rules"))
from engine import classify_sender, explain, extract_links, is_saudi_number, normalize  # noqa: E402

MAX_PER_TEMPLATE = 3

YEMEN_SENDERS = re.compile(
    r"yemen\s*mobile|yemenmobile|sabafon|saba\s*fon|^you$|you\s*telecom|mtn|y\s*telecom|kuraimi|alkuraimi|"
    r"tadhamon|cac\s*bank|yemen\s*kuwait|ykb|jawali|jaib|floosak|one\s*cash|mfloos|cash\s*plus|shamil|"
    r"bank\s*of\s*aden|بنك\s*عدن|يمن\s*موبايل|سبأفون|الكريمي|التضامن|جوالي|جيب|فلوسك", re.I)
YEMEN_TEXT = re.compile(r"ريال\s*يمني|ر\.ي|\bYER\b|\bYR\b|صنعاء|عدن|تعز|حضرموت|المكلا|\+?967\d{6,}", re.I)
SAUDI_TEXT = re.compile(r"ر\.س|ريال\s*سعودي|\bSAR\b|\bSR\b|الرياض|جدة|الدمام|مكة|المدينة|أبشر|ابشر|توكلنا|نفاذ|مدى", re.I)
SCAMMY = re.compile(
    r"رمز|كود|التحقق|تحقق|حدث|تحديث|بيانات|بطاقت|حساب|حوّ?ل|تحويل|ارسل|أرسل|مبلغ|جائز|ربحت|فزت|مبروك|"
    r"عاجل|فوراً|فورا|موقوف|إيقاف|ايقاف|تعليق|مخالف|شحن|طرد|سداد|ادفع|"
    r"code|otp|verify|update|account|suspend|locked|prize|won|winner|urgent|payment|pay now|parcel|delivery|bank",
    re.I)

PHONE_IN_TEXT = re.compile(r"(?<!\d)(?:(?:\+|00)?(?:966|967|971|20|44|1)[\s-]?)?0?[57]\d(?:[\s-]?\d){7}(?!\d)|(?<!\d)(?:\+|00)\d{9,14}(?!\d)")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[0-9A-Z]{4}){3,7}(?:\s?[0-9A-Z]{1,4})?\b")
LONG_NUM = re.compile(r"(?<![\d.,])\d{8,}(?![\d.,])")
MASKED_CARD = re.compile(r"[*xX]{2,}\s?\d{3,4}")


def hide_numbers(text: str) -> str:
    links, _ = extract_links(text)
    spans = [l.raw for l in links]
    protected = {f"\x00{i}\x00": s for i, s in enumerate(spans)}
    for k, s in protected.items():      # never touch numbers inside links
        text = text.replace(s, k, 1)
    text = IBAN.sub("<IBAN>", text)
    text = PHONE_IN_TEXT.sub("<PHONE>", text)
    text = MASKED_CARD.sub("**<CARD>", text)
    text = LONG_NUM.sub("<NUMBER>", text)
    for k, s in protected.items():
        text = text.replace(k, s)
    return text


def template_key(text: str) -> str:
    t = normalize(text)
    t = re.sub(r"\d+([.,]\d+)?", "#", t)
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def country(sender: str, text: str, when: datetime, ye_from, ye_to) -> str:
    if YEMEN_SENDERS.search(sender) or YEMEN_TEXT.search(text):
        return "ye"
    if ye_from and ye_to and ye_from <= when.date() <= ye_to:
        return "ye"
    kind = classify_sender(sender)
    if (kind == "phone" and is_saudi_number(sender)) or SAUDI_TEXT.search(text):
        return "sa"
    e = explain({"sender": sender, "text": text})
    return "sa" if e["brands"] else "unknown"


def arabic(text: str) -> bool:
    text = extract_links(text)[1]               # links are Latin even in Arabic messages
    letters = [c for c in text if c.isalpha()]
    return sum("؀" <= c <= "ۿ" for c in letters) / max(1, len(letters)) >= 0.5


def load_pool_keys(paths):
    keys = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            keys.setdefault(template_key(r.get("text", "")), r["id"])
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data/out/phone/candidates.jsonl")
    ap.add_argument("--yemen-from", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--yemen-to", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--pool", type=Path, nargs="*", default=[
        ROOT / "data/out/runB_pool.jsonl", ROOT / "data/out/pool.jsonl"])
    a = ap.parse_args(argv)

    stats = Counter()
    by_template: dict[str, list] = {}
    years = Counter()
    for _, el in ET.iterparse(a.xml, events=("end",)):
        if el.tag != "sms":
            if el.tag == "mms":
                stats["mms skipped"] += 1
            continue
        stats["sms"] += 1
        if el.get("type") != "1":                 # 1 = received; 2 = sent, others = drafts/outbox
            stats["sent/other skipped"] += 1; el.clear(); continue
        sender = (el.get("address") or "").strip()
        body = (el.get("body") or "").strip()
        contact = (el.get("contact_name") or "").strip()
        when = datetime.fromtimestamp(int(el.get("date", "0")) / 1000, tz=timezone.utc)
        el.clear()
        years[when.year] += 1
        if not body:
            stats["empty"] += 1; continue
        kind = classify_sender(sender)
        saved = contact and contact != "(Unknown)"
        if saved:
            stats["saved contact: personal, skipped"] += 1; continue
        if kind == "phone":
            links = extract_links(body)[0]
            if not links and not SCAMMY.search(body):
                stats["unsaved number, no link or money/code words: skipped"] += 1; continue
            group = "suspect_number"
        else:
            group = "business"
        stats[group] += 1
        rec = {"sender": sender, "text": body, "date": when.strftime("%Y-%m-%d"), "group": group,
               "country": country(sender, body, when, a.yemen_from, a.yemen_to)}
        by_template.setdefault(template_key(body), []).append(rec)

    pool_keys = load_pool_keys(a.pool)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out_stats = Counter()
    senders = Counter()
    with a.out.open("w", encoding="utf-8") as f:
        for key, recs in by_template.items():
            recs.sort(key=lambda r: r["date"], reverse=True)      # newest first
            for i, r in enumerate(recs[:MAX_PER_TEMPLATE]):
                text = hide_numbers(r["text"])
                e = explain({"sender": r["sender"], "text": text})
                cand = {"id": f"phone-{key}-{i}", "source": "sms",
                        "language": "ar" if arabic(text) else "en",
                        "sender": r["sender"],   # scammer numbers are kept: the foreign/unknown signals need them
                        "text": text, "date": r["date"], "country": r["country"], "group": r["group"],
                        "template": key, "template_count": len(recs),
                        "rule_signals": e["signals"], "brands": e["brands"],
                        "near_copy_of": pool_keys.get(key), "human_label": "", "origin": "phone"}
                f.write(json.dumps(cand, ensure_ascii=False) + "\n")
                out_stats[(r["group"], r["country"], cand["language"])] += 1
                out_stats["near-copy of training"] += bool(cand["near_copy_of"])
                if r["group"] == "business":
                    senders[(r["country"], r["sender"])] += 1

    print("received/skipped:", dict(stats))
    print("messages per year:", dict(sorted(years.items())))
    print(f"templates: {len(by_template)}; candidates written (max {MAX_PER_TEMPLATE}/template) -> {a.out}")
    for k, v in sorted((k, v) for k, v in out_stats.items() if isinstance(k, tuple)):
        print(f"  {'/'.join(k):32} {v}")
    print(f"  near-copies of training data: {out_stats['near-copy of training']}")
    print("\nbusiness senders (country, sender ID: candidates):")
    for (c, s), v in senders.most_common(60):
        print(f"  {c:8} {s:24} {v}")


if __name__ == "__main__":
    main()
