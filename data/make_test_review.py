"""Fresh review files for the full 450-message test set (relabelled from scratch).

    python data/make_test_review.py
    -> eval/v2/1_phone_review.jsonl     340  your phone: Arabic legit (need 200) + English legit (need 50)
    -> eval/v2/2_en_review.jsonl        160  English: Smishtank + IMC 2025 scams (need 100), UCI legit (need 50)
    -> eval/v2/3_scam_hunt_review.jsonl      real scam candidates, Arabic first (need 50 Arabic scams)
    Open each in eval/review.html. New file names, so the browser starts with no old labels.

3_scam_hunt mixes four sources (each message says which in its yellow note):
- second phone, unknown numbers;
- phone messages the teacher called scams (Run B). These are in training now: any you label
  scam moves to the test set and is removed from training before the next training run;
- complete scam messages quoted in Saudi news articles (SMS/WhatsApp only);
- the Arabic messages of the IMC 2025 smishing dataset (public; links and numbers already hidden
  as <URL>, <PHONE_NUMBER>). Label scam only if it could reach someone in Saudi Arabia; drop it if
  it is clearly about another country's bank, service or currency.
Any candidate that is a near-copy of a training message (edit similarity 80+) is left out.
Nothing is pre-labelled: human_label is empty everywhere.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/v2"
AR = re.compile(r"[ء-ي]")
RULES = ("Never change the wording. Only hide a person's name (<NAME>), ID or iqama (<ID>), car plate (<PLATE>) "
         "or home address (<ADDRESS>). Keep links, dates, times and amounts as they are.")


def ar(t: str) -> float:
    l = [c for c in t if c.isalpha()]
    return sum(bool(AR.match(c)) for c in l) / max(1, len(l))


def load(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d", "0", re.sub(r"<[A-Z_]+>", " ", t))).strip().lower()


def fresh(r: dict, notes: list[str]) -> dict:
    keep = {k: v for k, v in r.items() if k not in ("human_label", "check", "edited", "restored_from_original")}
    return {**keep, "human_label": "", "check": notes + [RULES]}


def write(name: str, rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}  {dict(Counter(r.get('test_group') for r in rows))}")


def main():
    # 1. your phone
    phone = [fresh(r, [n for n in r.get("check", []) if "signals fired" in n]) for r in load(ROOT / "eval/phone_review.jsonl")]
    write("1_phone_review.jsonl", phone)

    # 2. English
    write("2_en_review.jsonl", [fresh(r, r.get("check", [])) for r in load(ROOT / "eval/en_review.jsonl")])

    # 3. scam hunt
    cands = []
    for r in load(ROOT / "eval/phone2_review.jsonl"):
        cands.append(fresh(r, ["second phone, unknown number"]))
    for r in load(ROOT / "data/out/phone_flags_review.jsonl"):
        teacher = [n for n in r.get("check", []) if n.startswith("teacher says")]
        cands.append(fresh(r, teacher + ["from training: the teacher called this real phone message a scam. "
                                         "Scam = a real scam you received (moves to the test set); legit = the teacher was wrong"]))
    for a in load(ROOT / "data/out/scrape/scam_items.jsonl"):
        if not a.get("relevant"):
            continue
        for i, m in enumerate(a.get("messages") or []):
            if m.get("channel") not in ("sms", "whatsapp") or len(m.get("text", "")) < 20:
                continue
            t = m["text"]
            cands.append({"id": f"scrape-{hashlib.sha1(a['url'].encode()).hexdigest()[:10]}-{i}", "source": m["channel"],
                          "language": "ar" if ar(t) >= 0.5 else "en", "sender": "", "text": t, "origin": "news_scrape",
                          "test_group": "ar_scam" if ar(t) >= 0.5 else "en_scam", "category": "1 flagged", "human_label": "",
                          "check": [f"quoted in a news article ({a.get('source')}, {str(a.get('published'))[:10]}). "
                                    "Scam only if it is a complete scam message; a fragment or a headline is drop", RULES]})
    for i, m in enumerate(json.load((ROOT / "data/raw/imc_arabic.json").open(encoding="utf-8"))):
        t = m["text"]
        cands.append({"id": f"imc25-ar-{i:03d}", "source": "sms", "language": "ar", "sender": "", "text": t,
                      "origin": "imc25", "test_group": "ar_scam", "category": "1 flagged", "human_label": "",
                      "check": [f"IMC 2025 dataset (reported as a scam: {m['scam_type']}, country {m['original_network_country'] or 'unknown'}). "
                                "Scam if it could reach someone in Saudi Arabia; drop if it is clearly about another country", RULES]})

    pool = load(ROOT / "data/out/runB_pool_v2.jsonl")
    ptxt = [norm(r["text"]) for r in pool]
    pid = [r["id"] for r in pool]
    keep, dropped = [], Counter()
    sims = process.cdist([norm(c["text"]) for c in cands], ptxt, scorer=fuzz.ratio, workers=-1)
    for c, row in zip(cands, sims):
        best = int(row.argmax())
        if row[best] >= 80 and pid[best] != c["id"]:
            dropped[c["origin"]] += 1; continue
        keep.append(c)
    keep.sort(key=lambda c: (c["test_group"] != "ar_scam", c["origin"]))
    if dropped:
        print("left out as near-copies of training messages:", dict(dropped))
    write("3_scam_hunt_review.jsonl", keep)
    print("scam hunt by source:", dict(Counter(c["origin"] for c in keep)))


if __name__ == "__main__":
    main()
