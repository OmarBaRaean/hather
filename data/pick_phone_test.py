"""Pick a small, varied set of phone SMS for your label check (instead of all ~4,000).

    python data/pick_phone_test.py                       # reads data/out/phone/candidates.jsonl
    -> eval/phone_review.jsonl                            # open it in eval/review.html

Picks by metadata only (sender, signals, keywords), never by reading for you:
- one message per template (the newest), near-copies of training data excluded;
- groups and targets (a few extra, since you will drop some):
    sa_ar  Saudi Arabic legit      280  (test needs 200, decided Oct 23)
    sa_en  Saudi English legit      60  (replaces part of the NUS chat)
  Yemen messages are not used.
- inside each group, hard messages first: flagged links, codes, government,
  banks/wallets, deliveries, telecom, then the rest; ads are capped at 15%;
  at most 3 messages per sender, so no single company dominates (loosened only
  when a group cannot reach its target otherwise);
- `human_label` is pre-filled "legit" (all are business senders and none looked like
  a scam), `check` says what to look at. You confirm, change or drop each one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rules"))
from engine import load_brands  # noqa: E402

TARGETS = {"sa_ar": 280, "sa_en": 60}   # Yemen messages are left out of the test set (decided Oct 23)
MAX_PER_SENDER = 3
AD_SHARE = 0.15

OTP = re.compile(r"رمز|كود|التحقق|\bOTP\b|\bcode\b|verification", re.I)
GOV_SENDERS = re.compile(r"^(MOI|MOH|CivilDef|Absher|NAZAHA|NCA|MOE|MOJ|Najiz|Qiwa|GOSI|ZATCA|ZAKATY|Tawakkalna|"
                         r"Nafath|Balady|Ejar|Muroor|Saher|SDAIA|SFDA|MHRSD|Musaned|Sakani|HRSD|MOMAH|MCI)\b", re.I)
FINANCE = re.compile(r"bank|pay|tiqmo|urpay|neo|tabby|tamara|emkan|stc\s*bank|rajhi|alinma|snb|ahli|riyad|sab\b|"
                     r"fransi|jazira|albilad|d360|meem|kuraimi|jawali|floosak|بنك|مصرف|حوال|خصم|رصيد|إيداع|ايداع", re.I)
DELIVERY = re.compile(r"spl|aramex|smsa|naqel|zajil|dhl|fedex|ups|imile|j&t|noon|amazon|salasa|شحن|شحنة|طرد|توصيل", re.I)
TELECOM = re.compile(r"^(stc|900|902|mobily|1100|zain|959|jawwy|virgin|lebara|YOU|Sabafon|YemenMobile)", re.I)
AD = re.compile(r"-AD$|campaign", re.I)
NAME_HINT = re.compile(r"(عزيزي|عزيزتي|أهلا|اهلا|مرحبا|مرحباً|الأستاذ|الاستاذ|السيد|السيدة|\bDear\b|\bHi\b|\bHello\b)\s+\S+", re.I)


def category(c: dict) -> str:
    s, t = c["sender"], c["text"]
    if set(c.get("rule_signals", [])) - {"unknown_sender"} or c.get("group") == "suspect_number":
        return "1 flagged"
    if OTP.search(t):
        return "2 code"
    if GOV_SENDERS.search(s):
        return "3 government"
    if AD.search(s):
        return "8 ad"
    if FINANCE.search(s) or FINANCE.search(t):
        return "4 bank/wallet"
    if DELIVERY.search(s) or DELIVERY.search(t):
        return "5 delivery"
    if TELECOM.search(s):
        return "6 telecom"
    return "7 other"


def group_of(c: dict) -> str | None:
    if c["country"] == "ye":
        return None
    return "sa_ar" if c["language"] == "ar" else "sa_en"


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", type=Path, nargs="?", default=ROOT / "data/out/phone/candidates.jsonl")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "eval/phone_review.jsonl")
    a = ap.parse_args(argv)

    newest = {}
    for line in a.candidates.open(encoding="utf-8"):
        c = json.loads(line)
        if c.get("near_copy_of"):
            continue
        if c["template"] not in newest or c["date"] > newest[c["template"]]["date"]:
            newest[c["template"]] = c

    pools = defaultdict(lambda: defaultdict(list))
    for c in newest.values():
        g = group_of(c)
        if g:
            pools[g][category(c)].append(c)

    picked = []
    for g, target in TARGETS.items():
        per_sender, chosen, used = Counter(), [], set()
        ad_cap = int(target * AD_SHARE)
        cats = sorted(pools[g])
        # round-robin over categories (hard ones first); start with at most 3 per sender and
        # loosen the cap only if the group cannot reach its target otherwise (Yemen: mostly YOU)
        for cap in (MAX_PER_SENDER, MAX_PER_SENDER * 3, 10**9):
            queues = {k: [c for c in sorted(v, key=lambda c: h(c["id"])) if c["id"] not in used]
                      for k, v in pools[g].items()}
            progress = True
            while len(chosen) < target and progress:
                progress = False
                for k in cats:
                    if len(chosen) >= target:
                        break
                    if k == "8 ad" and sum(c["category"] == "8 ad" for c in chosen) >= ad_cap:
                        continue
                    while queues[k]:
                        c = queues[k].pop(0)
                        if per_sender[c["sender"].lower()] < cap:
                            per_sender[c["sender"].lower()] += 1
                            chosen.append({**c, "category": k}); used.add(c["id"])
                            progress = True
                            break
            if len(chosen) >= target:
                break
        for c in chosen:
            check = []
            if c["category"] == "1 flagged":
                check.append("signals fired: look closely, could be a real scam")
            if NAME_HINT.search(c["text"]):
                check.append("may contain a name: replace it with <NAME>")
            picked.append({**c, "test_group": g, "human_label": "legit", "check": check})

    picked.sort(key=lambda c: (list(TARGETS).index(c["test_group"]), c["category"], h(c["id"])))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for c in picked:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"{len(picked)} messages to review -> {a.out}")
    for g in TARGETS:
        rows = [c for c in picked if c["test_group"] == g]
        print(f"  {g}: {len(rows)} ({len({c['sender'] for c in rows})} senders)  "
              + ", ".join(f"{k[2:]} {v}" for k, v in sorted(Counter(c['category'] for c in rows).items())))
    print(f"  may contain a name: {sum(any('name' in x for x in c['check']) for c in picked)}")


if __name__ == "__main__":
    main()
