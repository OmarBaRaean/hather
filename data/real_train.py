"""Real phone SMS for training: legit messages from the exported phones,
scrubbed of private data and kept apart from the test set. Runs on your laptop only.

    python data/real_train.py
    -> data/out/phone/real_train.jsonl        training records (pool format, human_label "legit")
    -> data/out/phone/real_train_report.md    counts only, never message text

Which messages (counts are printed, never text):
- received business messages (sender ID or short code), Saudi or unknown country
  (Yemen left out), Arabic or English, with no rule signal (so no odd links);
- not a test message: any template used in a review file is skipped, and any message
  with token-set similarity of 70 or more (digits hidden) to a test candidate is skipped;
- one message per template (the newest), at most 15 per sender, ads at most 15%.

Privacy (on top of what phone_sms.py already hides: phone numbers, IBANs, card and
long account numbers):
- every name listed in data/raw/private_names.txt (the phone owners and their family) -> <NAME>;
- people's names after greetings, titles and transfer/beneficiary fields -> <NAME>,
  and every word caught that way is also hidden wherever else it appears;
- national ID / iqama numbers -> <ID>, car plates -> <PLATE>, streets, districts,
  addresses and national address codes -> <ADDRESS>, e-mail addresses -> <EMAIL>;
- every number of 3+ digits (amounts, balances, codes, references) gets new random
  digits of the same length; years (1990-2035) are kept;
- link paths and booking/reference codes get random letters and digits in the same
  shape; the domain is kept, so the rules engine sees the same signals;
- a message is dropped if, after all that, a greeting is still followed by a word that
  is not a known generic word (عميلنا, customer...), or if it mentions a personal
  document field we cannot hide reliably (date of birth, passport, mother's name).
The same seed always gives the same output, so the file can be rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rules"))
from engine import explain, format_input  # noqa: E402

PHONES = {  # candidates file -> origin
    ROOT / "data/out/phone/candidates.jsonl": "phone_1",
    ROOT / "data/out/phone/candidates-phone2.jsonl": "phone_2",
}
TEST_FILES = [ROOT / "eval/phone_review.jsonl", ROOT / "eval/phone2_review.jsonl",
              ROOT / "eval/en_review.jsonl", ROOT / "eval/gold_candidates_en_extra.jsonl"]
SCRAPE = ROOT / "data/out/scrape/scam_items.jsonl"
# Names that must never appear, one per line (the phone owners and their family).
# It lives in data/raw/, so it never goes to git, Kaggle or anywhere else.
PRIVATE_NAMES = ROOT / "data/raw/private_names.txt"
MAX_PER_SENDER = 15
AD_SHARE = 0.15
NEAR = 70

AR = "ء-غـ-ٰٟ-ۓ"   # Arabic letters and marks, not Arabic punctuation (، ؛ ؟)
WORD = rf"[{AR}A-Za-z][{AR}A-Za-z'\-]*"
# words that may follow a greeting or field label without being a name
GENERIC = set("""
عميل عميلنا العميل العملاء عميلتنا عزيزنا المشترك مشترك مشتركنا المستفيد العزيز الكريم الكريمة الغالي الغالية
عضو عضونا العضو شريكنا الشريك المواطن المقيم المستخدم مستخدم المريض المراجع الطالب الطالبة ولي أمر الأمر
بك بكم فيك فيكم وسهلا وسهلاً سهلا هلا يا من في على إلى الى عن مع تم لقد نود نشكرك شكرا شكراً نرحب
customer customers valued member members user client subscriber guest friend there sir madam team all you your
dear hi hello welcome to from the a an our
""".split())
GREETING = re.compile(rf"(?<![{AR}A-Za-z])(?:عزيزي|عزيزتي|أهلا|اهلا|أهلاً|اهلاً|مرحبا|مرحباً|هلا|السلام عليكم|الأستاذ|الاستاذ|الأستاذة|الاستاذة|"
                      rf"السيد|السيدة|السيد/ة|الأخ|الاخ|الأخت|الاخت|يا|"
                      rf"\b[Dd]ear\b|\bDEAR\b|\bH[Ii]\b|\bH[Ee][Ll][Ll][Oo]\b|\bH[Ee][Yy]\b|\bMr\.?|\bMrs\.?|\bMs\.)"
                      rf"[\s,،:]+((?:{WORD}[\s]*){{1,3}})", re.I)
FIELD = re.compile(rf"(?<![{AR}A-Za-z])(?:من|إلى|الى|المستفيد|المرسل|المحول|المحول له|اسم المستفيد|اسم العميل|الاسم|باسم|لصالح|"
                   rf"\bBeneficiary\b|\bSender\b|\bFrom\b|\bTo\b|\bName\b|\bCustomer name\b)\s*[:：]\s*((?:{WORD}\s*){{1,4}})", re.I)
TRANSFER_FROM = re.compile(rf"((?<![{AR}A-Za-z])(?:حوالة|حواله|تحويل|حوالة واردة|حوالة صادرة|transfer)[^\n.،,]{{0,30}}?\b(?:من|إلى|الى|to|from)\s+)"
                           rf"((?:{WORD}\s*){{1,4}})", re.I)
UNSAFE = re.compile(r"تاريخ الميلاد|تاريخ ميلاد|date of birth|\bDOB\b|جواز السفر|رقم الجواز|passport|اسم الأم|اسم الام|mother'?s name", re.I)
ID_NUM = re.compile(r"((?:رقم\s*)?(?:الهوية|هوية|الإقامة|الاقامة|إقامة|اقامة|السجل المدني|\biqama\b|\bnational id\b|\bID\b)[^\d\n]{0,12})[\d*xX•]{4,}", re.I)
_L1 = rf"[{AR}A-Za-z](?![{AR}A-Za-z])"       # one plate letter standing alone
PLATE = re.compile(rf"((?:لوحة|اللوحة|لوحه|اللوحه|رقم المركبة|المركبة رقم|\bplate\b)\s*(?:رقم)?\s*[:：]?\s*)"
                   rf"(?:{_L1}\s*){{0,4}}\d{{1,4}}(?:\s+{_L1}){{0,4}}", re.I)
ADDRESS = re.compile(rf"((?<![{AR}A-Za-z])(?:العنوان|عنوانك|عنوان التوصيل|حي|شارع|address|district|street)(?![{AR}A-Za-z])"
                     rf"\s*[:：]?\s*)[^\n.،,؛]{{2,40}}", re.I)
NAT_ADDRESS = re.compile(r"\b[A-Z]{4}\d{4}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(https?://[^\s/]+|www\.[^\s/]+|(?<![@\w])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|sa|net|org|link|ly|ee|gl|me|io|co|app|page)(?=/))(/[^\s]*)", re.I)
MIXED_CODE = re.compile(r"(?<![\w<])(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{6,}(?![\w>])")
DIGITS = re.compile(r"\d{3,}")
WS = re.compile(r"\s+")


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def shape_random(s: str, rng: random.Random) -> str:
    out = []
    for c in s:
        if c.isdigit():
            out.append(str(rng.randint(0, 9)))
        elif c.isupper():
            out.append(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))
        elif c.islower():
            out.append(rng.choice("abcdefghijkmnopqrstuvwxyz"))
        else:
            out.append(c)
    return "".join(out)


def new_digits(m: re.Match, rng: random.Random) -> str:
    s = m.group(0)
    if len(s) == 4 and 1990 <= int(s) <= 2035:
        return s
    first = str(rng.randint(1, 9)) if s[0] != "0" else "0"
    return first + "".join(str(rng.randint(0, 9)) for _ in s[1:])


VOCAB: Counter = Counter()   # word counts in the training pool, filled in main()


def name_words(span: str) -> list[str]:
    """The words of a captured span up to the first word that is clearly not a name:
    a generic word, a placeholder, a lower-case English word, an English word used 20+
    times in the training pool, or an Arabic word used 150+ times (أو، رمز، الساعة).
    Names the teacher also used (خالد، سعيد) stay below that, so they are still hidden."""
    words = []
    for w in span.split():
        w = w.strip(",،.:!؟?()")
        lw = w.lower()
        if not lw or lw in GENERIC or lw.startswith("<"):
            break
        if re.match(r"[A-Za-z]", w):
            if not w[0].isupper() or VOCAB[lw] >= 20:
                break
        elif VOCAB[lw] >= 150:
            break
        words.append(w)
    return words


def find_names(text: str) -> set[str]:
    found = set()
    for rx, g in ((GREETING, 1), (FIELD, 1), (TRANSFER_FROM, 2)):
        for m in rx.finditer(text):
            for w in name_words(m.group(g)):
                if len(w) >= 2 and not w.isdigit():
                    found.add(w)
    return found


def scrub(text: str, names: set[str], rng: random.Random) -> str:
    t = EMAIL.sub("<EMAIL>", text)
    t = URL.sub(lambda m: m.group(1) + shape_random(m.group(2), rng), t)
    t = ID_NUM.sub(lambda m: m.group(1) + "<ID>", t)
    t = PLATE.sub(lambda m: m.group(1) + "<PLATE>", t)
    t = NAT_ADDRESS.sub("<ADDRESS>", t)
    t = ADDRESS.sub(lambda m: m.group(1) + "<ADDRESS>", t)
    if names:
        # longest first, whole words only (Arabic has no \b, so use letter boundaries)
        for w in sorted(names, key=len, reverse=True):
            t = re.sub(rf"(?<![{AR}A-Za-z]){re.escape(w)}(?![{AR}A-Za-z])", "<NAME>", t, flags=re.I)
        t = re.sub(r"<NAME>(?:\s*<NAME>)+", "<NAME>", t)
    parts = re.split(r"(<[A-Z]+>)", t)            # keep placeholders untouched
    for i, p in enumerate(parts):
        if not p.startswith("<"):
            p = MIXED_CODE.sub(lambda m: shape_random(m.group(0), rng), p)
            parts[i] = DIGITS.sub(lambda m: new_digits(m, rng), p)
    return "".join(parts)


def leftover_name(text: str) -> bool:
    """A greeting still followed by a non-generic word after scrubbing."""
    for m in GREETING.finditer(text):
        if name_words(m.group(1)):
            return True
    return False


def norm(t: str) -> str:
    return WS.sub(" ", re.sub(r"\d", "0", re.sub(r"<[A-Z]+>", " ", t))).strip().lower()


def load(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data/out/phone/real_train.jsonl")
    ap.add_argument("--seed", default="hather-real-train-v1")
    a = ap.parse_args(argv)
    stats = Counter()

    # --- test candidates: templates and texts to keep away from
    test_texts, test_tpl = [], set()
    for f in TEST_FILES:
        for r in load(f):
            test_texts.append(norm(r["text"]))
            if r.get("template"):
                test_tpl.add(r["template"])
    for art in load(SCRAPE):
        for m in art.get("messages") or []:
            txt = m.get("text") if isinstance(m, dict) else m
            if txt:
                test_texts.append(norm(txt))
    stats["test texts checked against"] = len(test_texts)

    # --- candidates from both phones
    newest = {}
    for path, origin in PHONES.items():
        for r in load(path):
            stats[f"{origin}: candidates"] += 1
            if r["group"] != "business" or r["country"] == "ye" or r["language"] not in ("ar", "en"):
                continue
            if r.get("rule_signals"):
                stats["skipped: rule signal fired"] += 1; continue
            if r["template"] in test_tpl:
                stats["skipped: same template as a test message"] += 1; continue
            r = {**r, "origin": origin}
            k = r["template"]
            if k not in newest or r["date"] > newest[k]["date"]:
                newest[k] = r
    rows = sorted(newest.values(), key=lambda r: h(a.seed + r["template"]))
    stats["distinct templates"] = len(rows)

    # --- near-copies of test messages
    if test_texts:
        keep = []
        texts = [norm(r["text"]) for r in rows]
        sims = process.cdist(texts, test_texts, scorer=fuzz.token_set_ratio, score_cutoff=NEAR, workers=-1)
        for r, row in zip(rows, sims):
            if row.max() >= NEAR:
                stats["skipped: near-copy of a test message"] += 1
            else:
                keep.append(r)
        rows = keep

    # --- per-sender and ad caps, per language
    picked = []
    for lang in ("ar", "en"):
        per_sender, ads, chosen = Counter(), 0, []
        pool = [r for r in rows if r["language"] == lang]
        ad_cap = int(min(len(pool), 10**9) * AD_SHARE)
        for r in pool:
            s = r["sender"].lower()
            is_ad = s.endswith("-ad")
            if per_sender[s] >= MAX_PER_SENDER:
                stats[f"{lang}: skipped, sender cap"] += 1; continue
            per_sender[s] += 1
            chosen.append((r, is_ad))
        n_ok = sum(1 for _, ad in chosen if not ad)
        ad_cap = int(n_ok * AD_SHARE / (1 - AD_SHARE))
        for r, is_ad in chosen:
            if is_ad:
                if ads >= ad_cap:
                    stats[f"{lang}: skipped, ad cap"] += 1; continue
                ads += 1
            picked.append(r)

    # --- privacy: collect names everywhere first, then scrub.
    # A caught word is hidden in its own message always; it is hidden in every other
    # message too unless it is a common word (5+ uses in the training pool), so that
    # "من البنك" does not turn every "البنك" into <NAME>.
    vocab = VOCAB
    for r in load(ROOT / "data/out/runB_pool.jsonl"):
        vocab.update(w.lower() for w in re.findall(WORD, r["text"]))
    local = {r["id"] + r["origin"]: find_names(r["text"]) for r in picked}
    all_names = {w for ws in local.values() for w in ws if vocab[w.lower()] < 5}
    private = set()
    if PRIVATE_NAMES.exists():
        private = {l.strip() for l in PRIVATE_NAMES.read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.startswith("#")}
    all_names |= private
    stats["private names from data/raw/private_names.txt"] = len(private)
    stats["name words found"] = len({w for ws in local.values() for w in ws})
    stats["of which hidden in every message"] = len(all_names)

    out = []
    for r in picked:
        rng = random.Random(a.seed + r["id"] + r["origin"])
        if UNSAFE.search(r["text"]):
            stats["dropped: personal document field"] += 1; continue
        names_here = local[r["id"] + r["origin"]] | {
            w for w in all_names
            if re.search(rf"(?<![{AR}A-Za-z]){re.escape(w)}(?![{AR}A-Za-z])", r["text"], re.I)}
        text = scrub(r["text"], names_here, rng)
        if leftover_name(text):
            stats["dropped: possible name left"] += 1; continue
        for tag in ("NAME", "ID", "PLATE", "ADDRESS", "EMAIL"):
            if f"<{tag}>" in text:
                stats[f"hidden: <{tag}> (messages)"] += 1
        rec = {"id": f"{r['origin']}-{h(r['origin'] + r['id'])[:12]}", "source": "sms", "language": r["language"],
               "sender": r["sender"], "subject": "", "text": text, "human_label": "legit", "origin": r["origin"],
               "eml_path": None, "scenario": "phone:" + h(r["sender"].lower())[:10],
               "template_count": r.get("template_count")}
        rec["signals"] = explain(rec)["signals"]
        if rec["signals"] and set(rec["signals"]) - {"unknown_sender"}:
            stats["dropped: a signal fired after scrubbing"] += 1; continue
        rec["input"] = format_input(rec)
        out.append(rec)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = Counter((r["origin"], r["language"]) for r in out)
    lines = ["# Real phone SMS for training (counts only)", "",
             f"Written: {len(out)} messages -> `{a.out.relative_to(ROOT)}`", "",
             "| Phone | Arabic | English |", "| --- | ---: | ---: |"]
    for o in PHONES.values():
        lines.append(f"| {o} | {by[(o, 'ar')]} | {by[(o, 'en')]} |")
    lines += ["", f"Senders: {len({r['sender'].lower() for r in out})}", "", "| Step | Count |", "| --- | ---: |"]
    lines += [f"| {k} | {v} |" for k, v in stats.items()]
    rep = a.out.with_name("real_train_report.md")
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
