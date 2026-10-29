"""
Hather - Phase 1, step 1: download-and-clean -> pool.jsonl

Reads the raw English datasets you downloaded, cleans them, removes
duplicates, holds out English test candidates FIRST, then samples the
training pool.

Outputs
  data/out/pool.jsonl                 training pool (the teacher labels this in Phase 3)
  eval/gold_candidates_en.jsonl       200 English test candidates for you to review
  data/out/stats.json                 counts per part, for your write-up

Each record:
  id, source (email|sms|whatsapp), language, sender, subject, text,
  human_label (scam|legit), origin (which dataset), eml_path (or null)

Only the Python standard library is needed.

Example
  python data/build_pool.py \
      --kaggle-dir   raw/phishing-email-dataset \
      --pot-dir      raw/phishing_pot \
      --smishtank    raw/smishtank.csv \
      --uci          raw/SMSSpamCollection \
      --takeout-mbox raw/takeout/Inbox.mbox

  Add --inspect first to print the columns each file has, without writing anything.
"""

from __future__ import annotations

import argparse
import csv
import email
import email.header
import email.policy
import hashlib
import html
import json
import mailbox
import random
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

csv.field_size_limit(sys.maxsize)

MAX_CHARS = 4000

# Targets from the Part 1 plan
N_EMAIL_PER_CLASS = 1500   # Kaggle emails: 1,500 scam + 1,500 legit
N_POT = 300                # phishing_pot .eml files (scam, with headers)
N_TAKEOUT = 300            # your own legit emails (with headers)
N_SMS_TOTAL = 2000         # Smishtank scams + UCI ham

# English test candidates held out BEFORE sampling (you review them)
N_TEST = {"email_scam": 50, "email_legit": 50, "sms_scam": 50, "sms_legit": 50}

# Kaggle phishing-email-dataset files.
# Scam side: only true phishing/fraud corpora. CEAS, SpamAssassin and Ling
# mark advertising spam as 1, which is not a scam, so their 1s are skipped.
KAGGLE_SCAM_FILES = ["Nazario.csv", "Nigerian_Fraud.csv"]
# Enron and Ling are NOT used: their text was pre-lowercased and tokenized
# ("haven , t"), so the model could learn "odd lowercase text = legit".
KAGGLE_LEGIT_FILES = ["CEAS_08.csv", "SpamAssasin.csv"]

# Common English words, for a cheap "is this email English?" check.
_EN_WORDS = set("""the and you your to of is for in this please account be have on
with we are click a it that not will from at as by our can if or has an all now""".split())

# phishing_pot also contains plain advertising spam. Keep only emails with
# a phishing cue (credentials, payment, delivery, account trouble).
POT_PHISH_CUES = re.compile(
    r"account|password|passcode|verif|log ?in|sign ?in|suspend|locked|limited|"
    r"unusual|invoice|payment|pay now|billing|bank|card|package|parcel|deliver|"
    r"shipment|refund|security|confirm|wallet|credential|expire|mailbox|storage|"
    r"tax|irs|reactivat|unlock|validate", re.I)


# ---------------------------------------------------------------- cleaning

class _HTMLToText(HTMLParser):
    """Strip tags but keep link targets, so URLs survive for the rules engine."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._href: str | None = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self.parts.append(" ")
        elif tag in ("br", "p", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag == "a" and self._href:
            if self._href.startswith("http"):
                self.parts.append(f" [{self._href}]")
            self._href = None

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    p = _HTMLToText()
    try:
        p.feed(raw)
        p.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html.unescape(raw))
    return "".join(p.parts)


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    if re.search(r"<(html|body|div|p|a|table|br)\b", text, re.I):
        text = html_to_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def is_english(text: str, min_ratio: float = 0.08) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 8:
        return True          # too short to judge; keep
    return sum(w in _EN_WORDS for w in words) / len(words) >= min_ratio


def decode_mime(value: str | None) -> str:
    """'=?iso-8859-1?Q?...?=' -> readable text."""
    if not value or "=?" not in value:
        return value or ""
    try:
        return str(email.header.make_header(email.header.decode_header(value)))
    except Exception:
        return value


def dedup_key(text: str) -> str:
    """Catches near-identical template messages (same text, different link or number)."""
    t = text.lower()
    t = re.sub(r"https?://\S+|www\.\S+", "URL", t)
    t = re.sub(r"\d", "0", t)
    t = re.sub(r"\W+", " ", t).strip()
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def make_record(*, source, sender, subject, text, label, origin, eml_path=None):
    return {
        "source": source,
        "language": "en",
        "sender": decode_mime(str(sender or "")).strip()[:300],
        "subject": clean_text(decode_mime(str(subject or "")))[:300],
        "text": clean_text(text),
        "human_label": label,
        "origin": origin,
        "eml_path": eml_path,
    }


# ---------------------------------------------------------------- loaders

def read_csv_rows(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        yield from csv.DictReader(f)


def pick(row: dict, *names: str) -> str:
    """Return the first matching column, case-insensitive."""
    lower = {k.lower().strip(): v for k, v in row.items() if k}
    for n in names:
        if lower.get(n.lower()):
            return lower[n.lower()]
    return ""


def load_kaggle(kaggle_dir: Path) -> list[dict]:
    out = []
    for fname, wanted in [(f, "scam") for f in KAGGLE_SCAM_FILES] + \
                         [(f, "legit") for f in KAGGLE_LEGIT_FILES]:
        path = kaggle_dir / fname
        if not path.exists():
            print(f"  ! missing {path}, skipped")
            continue
        n = 0
        for row in read_csv_rows(path):
            label = pick(row, "label").strip()
            # scam files: keep label 1; legit files: keep label 0 only
            if wanted == "scam" and label != "1":
                continue
            if wanted == "legit" and label != "0":
                continue
            out.append(make_record(
                source="email",
                sender=pick(row, "sender", "from"),
                subject=pick(row, "subject"),
                text=pick(row, "body", "text", "content"),
                label=wanted,
                origin=f"kaggle/{path.stem}",
            ))
            n += 1
        print(f"  kaggle/{fname}: {n} {wanted}")
    return out


def eml_body(msg) -> str:
    """Prefer text/plain, fall back to text/html."""
    plain, htm = None, None
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_filename():          # attachment
            continue
        ctype = part.get_content_type()
        try:
            content = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True) or b""
            content = payload.decode("utf-8", errors="replace")
        if not isinstance(content, str):
            continue
        if ctype == "text/plain" and plain is None:
            plain = content
        elif ctype == "text/html" and htm is None:
            htm = content
    return plain if plain and plain.strip() else (htm or "")


def parse_eml_bytes(raw: bytes):
    return email.message_from_bytes(raw, policy=email.policy.default)


def load_pot(pot_dir: Path) -> list[dict]:
    files = sorted(list(pot_dir.rglob("*.eml")))
    out = []
    for p in files:
        try:
            msg = parse_eml_bytes(p.read_bytes())
            out.append(make_record(
                source="email",
                sender=str(msg.get("From", "")),
                subject=str(msg.get("Subject", "")),
                text=eml_body(msg),
                label="scam",
                origin="phishing_pot",
                eml_path=str(p),
            ))
        except Exception as e:
            print(f"  ! could not parse {p.name}: {e}")
    print(f"  phishing_pot: {len(out)} parsed from {len(files)} .eml files")
    return out


def load_takeout(mbox_path: Path, save_dir: Path) -> list[dict]:
    """Your own legit emails. Keeps only inbox mail that has an
    Authentication-Results header (needed for spf/dkim/dmarc signals).
    Each kept message is saved as .eml so the rules engine can read headers."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, m in enumerate(mailbox.mbox(str(mbox_path))):
        labels = (m.get("X-Gmail-Labels") or "").lower()
        if any(x in labels for x in ("spam", "sent", "draft", "trash")):
            continue
        if not m.get("Authentication-Results"):
            continue
        raw = m.as_bytes()
        msg = parse_eml_bytes(raw)
        p = save_dir / f"takeout_{i:06d}.eml"
        p.write_bytes(raw)
        out.append(make_record(
            source="email",
            sender=str(msg.get("From", "")),
            subject=str(msg.get("Subject", "")),
            text=eml_body(msg),
            label="legit",
            origin="takeout",
            eml_path=str(p),
        ))
    print(f"  takeout: {len(out)} usable legit emails")
    return out


def load_smishtank(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        rows = data if isinstance(data, list) else data.get("data", [])
    else:
        rows = list(read_csv_rows(path))
    out = []
    for row in rows:
        text = pick(row, "MainText", "text", "message", "body", "sms", "content")
        out.append(make_record(
            source="sms",
            sender=pick(row, "Sender", "sender", "from", "phone", "number"),
            subject="",
            text=text,
            label="scam",
            origin="smishtank",
        ))
    print(f"  smishtank: {len(out)} scam SMS")
    return out


def load_uci_ham(path: Path) -> list[dict]:
    """UCI SMS Spam Collection: 'label<TAB>text'. Keep ham only;
    its 'spam' is old advertising, not scams."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label.strip().lower() == "ham":
            out.append(make_record(source="sms", sender="", subject="", text=text,
                                   label="legit", origin="uci_ham"))
    print(f"  uci: {len(out)} ham SMS")
    return out


# ---------------------------------------------------------------- pipeline

def clean_and_dedup(records: list[dict], seen: set[str]) -> list[dict]:
    """Drop empty / too long / non-English emails / ad-only phishing_pot,
    then exact and near-duplicates (across ALL sources)."""
    kept, stats = [], Counter()
    for r in records:
        if r["source"] == "email" and not is_english(r["subject"] + " " + r["text"]):
            stats["not_english"] += 1
            continue
        if r["origin"] == "phishing_pot" and not POT_PHISH_CUES.search(r["subject"] + " " + r["text"]):
            stats["pot_ad_not_phishing"] += 1
            continue
        if len(r["text"]) < 5:
            stats["empty"] += 1
            continue
        if len(r["text"]) > MAX_CHARS:
            stats["too_long"] += 1
            continue
        k = dedup_key(r["subject"] + " " + r["text"])
        if k in seen:
            stats["duplicate"] += 1
            continue
        seen.add(k)
        r["_key"] = k
        r["id"] = f'{r["origin"].replace("/", "-")}-{k[:10]}'
        kept.append(r)
    return kept, stats


def balanced_sample(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin across origins, so one big corpus (e.g. Enron) can't dominate."""
    by_origin = defaultdict(list)
    for r in records:
        by_origin[r["origin"]].append(r)
    for lst in by_origin.values():
        rng.shuffle(lst)
    out, origins = [], sorted(by_origin)
    while len(out) < n and any(by_origin.values()):
        for o in origins:
            if by_origin[o] and len(out) < n:
                out.append(by_origin[o].pop())
    if len(out) < n:
        print(f"  ! wanted {n}, only {len(out)} available")
    return out


def take(records: list[dict], n: int, rng: random.Random):
    """Remove n balanced items from records; return (taken, remaining)."""
    taken = balanced_sample(records, n, rng)
    ids = {r["id"] for r in taken}
    return taken, [r for r in records if r["id"] not in ids]


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "source", "language", "sender", "subject", "text",
              "human_label", "origin", "eml_path"]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")


def inspect(args):
    """Print the columns of each input so you can confirm the loaders match."""
    if args.kaggle_dir:
        for p in sorted(Path(args.kaggle_dir).glob("*.csv")):
            row = next(read_csv_rows(p), {})
            print(f"{p.name}: {list(row.keys())}")
    if args.smishtank:
        p = Path(args.smishtank)
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            first = (data if isinstance(data, list) else data.get("data", [{}]))[0]
            print(f"{p.name}: {list(first.keys())}")
        else:
            print(f"{p.name}: {list(next(read_csv_rows(p), {}).keys())}")
    if args.pot_dir:
        print(f"phishing_pot: {len(list(Path(args.pot_dir).rglob('*.eml')))} .eml files")
    if args.uci:
        print("UCI first lines:", Path(args.uci).read_text(errors="replace").splitlines()[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-dir", required=True, help="folder with the Kaggle CSVs")
    ap.add_argument("--pot-dir", required=True, help="cloned phishing_pot repo")
    ap.add_argument("--smishtank", required=True, help="Smishtank CSV or JSON")
    ap.add_argument("--uci", required=True, help="UCI SMSSpamCollection file")
    ap.add_argument("--takeout-mbox", help="Google Takeout Inbox.mbox (optional)")
    ap.add_argument("--out-dir", default="data/out")
    ap.add_argument("--eval-dir", default="eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true", help="print columns and exit")
    args = ap.parse_args()

    if args.inspect:
        inspect(args)
        return

    rng = random.Random(args.seed)
    out_dir, eval_dir = Path(args.out_dir), Path(args.eval_dir)

    print("Loading...")
    raw = {
        "kaggle": load_kaggle(Path(args.kaggle_dir)),
        "pot": load_pot(Path(args.pot_dir)),
        "smish": load_smishtank(Path(args.smishtank)),
        "uci": load_uci_ham(Path(args.uci)),
        "takeout": load_takeout(Path(args.takeout_mbox), out_dir / "takeout_eml")
                   if args.takeout_mbox else [],
    }

    print("\nCleaning and removing duplicates across all sources...")
    seen: set[str] = set()
    clean, drop_stats = {}, Counter()
    for name, recs in raw.items():
        clean[name], s = clean_and_dedup(recs, seen)
        drop_stats.update(s)
    print(f"  dropped: {dict(drop_stats)}")

    email_scam = [r for r in clean["kaggle"] if r["human_label"] == "scam"]
    email_legit = [r for r in clean["kaggle"] if r["human_label"] == "legit"]
    sms_scam, sms_legit = clean["smish"], clean["uci"]

    # 1) Hold out English test candidates FIRST, so they can never leak into training.
    test_es, email_scam = take(email_scam, N_TEST["email_scam"], rng)
    test_el, email_legit = take(email_legit, N_TEST["email_legit"], rng)
    test_ss, sms_scam = take(sms_scam, N_TEST["sms_scam"], rng)
    test_sl, sms_legit = take(sms_legit, N_TEST["sms_legit"], rng)
    test = test_es + test_el + test_ss + test_sl

    # 2) Sample the training pool from what's left.
    pool_es = balanced_sample(email_scam, N_EMAIL_PER_CLASS, rng)
    pool_el = balanced_sample(email_legit, N_EMAIL_PER_CLASS, rng)
    pool_pot = balanced_sample(clean["pot"], N_POT, rng)
    pool_tk = balanced_sample(clean["takeout"], N_TAKEOUT, rng)
    rng.shuffle(sms_scam)
    pool_ss = sms_scam[: N_SMS_TOTAL // 2]                # up to 1,000 scams (Smishtank has ~1,062)
    pool_sl = balanced_sample(sms_legit, N_SMS_TOTAL - len(pool_ss), rng)

    pool = pool_es + pool_el + pool_pot + pool_tk + pool_ss + pool_sl
    rng.shuffle(pool)

    # Safety check: no overlap between pool and test.
    assert not ({r["_key"] for r in pool} & {r["_key"] for r in test}), "test leaked into pool"

    write_jsonl(out_dir / "pool.jsonl", pool)
    write_jsonl(eval_dir / "gold_candidates_en.jsonl", test)

    stats = {
        "seed": args.seed,
        "dropped": dict(drop_stats),
        "pool_total": len(pool),
        "pool_parts": {
            "email_scam (kaggle)": len(pool_es),
            "email_legit (kaggle)": len(pool_el),
            "email_scam_headers (phishing_pot)": len(pool_pot),
            "email_legit_headers (takeout)": len(pool_tk),
            "sms_scam (smishtank)": len(pool_ss),
            "sms_legit (uci ham)": len(pool_sl),
        },
        "pool_by_origin": dict(Counter(r["origin"] for r in pool)),
        "pool_by_label": dict(Counter(r["human_label"] for r in pool)),
        "test_candidates": len(test),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print("\nDone.")
    for k, v in stats["pool_parts"].items():
        print(f"  {k:<36} {v:>5}")
    print(f"  {'POOL TOTAL':<36} {len(pool):>5}   -> {out_dir / 'pool.jsonl'}")
    print(f"  {'English test candidates':<36} {len(test):>5}   -> {eval_dir / 'gold_candidates_en.jsonl'}")
    if not raw["takeout"]:
        print("\n  Note: no --takeout-mbox given, so the 300 legit emails with headers are missing.")


if __name__ == "__main__":
    main()