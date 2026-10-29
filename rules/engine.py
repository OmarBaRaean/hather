"""Hather rules engine.

Plain code, not AI: reads one SMS/WhatsApp message and its sender and lists the
tricks a computer can check for certain. The same signals go to the teacher
(Run B), the student, and the app (ported to Dart), so every rule here is
deterministic and documented in rules/README.md.

    from engine import signals, format_input
    msg = {"source": "sms", "sender": "AlRajhiBank", "text": "..."}
    signals(msg)       -> ["lookalike_domain", "brand_link_mismatch"]
    format_input(msg)  -> "[SOURCE] sms\n[SENDER] AlRajhiBank\n[SIGNALS] ...\n[TEXT] ..."
"""
from __future__ import annotations

import ipaddress
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import idna
import tldextract
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent

# Fixed order: signals are always reported in this order.
SIGNAL_ORDER = [
    "lookalike_domain",
    "brand_link_mismatch",
    "link_shortener",
    "link_ip",
    "foreign_number_local_brand",
    "unknown_sender",
]

TEXT_LIMIT = 1000  # characters of [TEXT] in the formatted input

SHORTENERS = {
    "bit.ly", "tinyurl.com", "cutt.ly", "t.co", "goo.gl", "ow.ly", "is.gd", "v.gd",
    "buff.ly", "rebrand.ly", "shorturl.at", "rb.gy", "t.ly", "tiny.cc", "bl.ink",
    "s.id", "lnkd.in", "shorte.st", "adf.ly", "bit.do", "short.gy", "qrco.de",
    "u.to", "x.co", "snip.ly", "surl.li", "urlz.fr", "gg.gg", "shrtco.de",
    "tny.im", "clck.ru", "linktr.ee", "2u.pw", "cutt.us", "shorturl.asia",
}

# Bare domains (no http://, no www., no path) count as links only with one of
# these TLDs, so chat like "ok.so" or "app.py" is not read as a link.
COMMON_TLDS = {
    "com", "net", "org", "info", "biz", "sa", "ae", "eg", "uk", "co", "io", "me",
    "ly", "cc", "ws", "pw", "xyz", "top", "icu", "online", "site", "live", "shop",
    "store", "app", "vip", "club", "link", "click", "win", "tk", "ml", "ga", "cf",
    "gq", "buzz", "cyou", "sbs", "cfd", "bond", "support", "help", "tech", "fun",
    "space", "website", "today", "world", "pro", "mobi", "ru", "cn",
}

# Everyday words never treated as a 1-2 letter misspelling of a brand
# (mobile ~ mobily, saudi ~ saudia). Exact brand matches still count.
COMMON_WORDS = {
    "mobile", "saudi", "online", "group", "store", "shop", "bank", "home", "news",
    "info", "media", "cloud", "global", "express", "career", "cargo", "sales",
    "stock", "nation", "national", "service", "services", "secure", "support",
}

EN_CONTEXT = {
    "account", "app", "order", "card", "customer", "bank", "delivery", "deliver",
    "shipment", "package", "parcel", "login", "verify", "verification", "update",
    "store", "prime", "pay", "payment", "bill", "support", "service",
    "subscription", "id", "code", "otp", "purchase", "refund", "wallet",
    "points", "reward", "gift", "membership", "team", "locked", "suspended",
}
# Arabic context roots: a token containing one of these counts as context.
AR_CONTEXT = [
    "حساب", "تطبيق", "منص", "بطاق", "عميل", "عملاء", "بنك", "مصرف", "شحن", "طرد",
    "خدم", "رابط", "دفع", "سداد", "فاتور", "اشتراك", "توصيل", "موقع", "تحديث",
    "مخالف", "تحقق", "رمز", "دخول", "مركب", "حادث", "محفظ", "نقاط", "جائز",
]
CONTEXT_WINDOW = 3  # tokens on each side

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                                 include_psl_private_domains=False)

# ---------------------------------------------------------------- normalizing
_TASHKEEL = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي",
                         "ة": "ه", "ؤ": "و", "ئ": "ي"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize(s: str) -> str:
    """Lowercase, unify Arabic letter variants, drop diacritics, ASCII digits."""
    s = unicodedata.normalize("NFKC", s or "")
    s = _TASHKEEL.sub("", s)
    return s.translate(_AR_MAP).translate(_DIGITS).lower()


def _is_arabic(s: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in s)


# --------------------------------------------------------------------- brands
@dataclass
class Brand:
    id: str
    domains: list[str]
    sender_keys: set[str]
    keywords: set[str]
    saudi: bool
    gov: bool
    patterns: list[tuple[re.Pattern, bool]] = field(default_factory=list)  # (regex, needs_context)


def _sender_key(s: str) -> str:
    return re.sub(r"[^a-z0-9ء-ي]", "", normalize(s))


def _name_pattern(name: str) -> re.Pattern:
    name = normalize(name)
    toks = name.split()
    if _is_arabic(name):
        first, rest = toks[0], toks[1:]
        if first.startswith("ال") and len(first) > 3:
            # و/ف/ب/ك + ال..., or ل + ال... written as لل...
            head = f"(?:[وفبك]?{re.escape(first)}|[وف]?لل{re.escape(first[2:])})"
        else:
            head = f"[وفبكل]{{0,2}}{re.escape(first)}"
        body = head + "".join(r"\s+" + re.escape(t) for t in rest)
        return re.compile(rf"(?<!\w){body}(?!\w)")
    body = r"\s+".join(re.escape(t) for t in toks)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


@lru_cache(maxsize=1)
def load_brands(path: str = str(HERE / "brands.json")) -> tuple[Brand, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for b in data["brands"]:
        domains = [d.lower() for d in b.get("domains", [])]
        kws = {k.lower() for k in b.get("keywords", [])}
        for d in domains:  # the domain label itself is a keyword (alrajhibank)
            ext = _EXTRACT(d)
            label = ext.domain
            if ext.subdomain:  # maps.app.goo.gl: "goo" is not Google's name
                continue
            if len(label) >= 3:
                kws.add(label)
        brand = Brand(
            id=b["id"], domains=domains,
            sender_keys={_sender_key(s) for s in b.get("sender_ids", []) if _sender_key(s)},
            keywords={k for k in kws if len(k) >= 3},
            saudi=bool(b.get("saudi")), gov=bool(b.get("gov")),
        )
        for key, ctx in (("en", False), ("ar", False), ("en_ctx", True), ("ar_ctx", True)):
            for name in b.get(key, []):
                brand.patterns.append((_name_pattern(name), ctx))
        out.append(brand)
    return tuple(out)


def _has_context(text: str, start: int, end: int) -> bool:
    # Header use: the name is followed by a colon ("نجم: ..." / "Apple: ...").
    if re.match(r"\s*[:：]", text[end:]):
        return True
    tokens = [(m.start(), m.group()) for m in re.finditer(r"\w+", text)]
    before = [t for s, t in tokens if s < start][-CONTEXT_WINDOW:]
    after = [t for s, t in tokens if s >= end][:CONTEXT_WINDOW]
    for t in before + after:
        if _is_arabic(t):
            if any(root in t for root in AR_CONTEXT):
                return True
        elif t in EN_CONTEXT or any(len(w) >= 4 and t.startswith(w) for w in EN_CONTEXT):
            return True
    return False


def brands_in_text(text_norm: str) -> set[str]:
    found = set()
    for b in load_brands():
        for pat, needs_ctx in b.patterns:
            for m in pat.finditer(text_norm):
                if not needs_ctx or _has_context(text_norm, m.start(), m.end()):
                    found.add(b.id)
                    break
            if b.id in found:
                break
    return found


def brand_from_sender(sender: str) -> set[str]:
    key = _sender_key(sender or "")
    return {b.id for b in load_brands() if key and key in b.sender_keys}


# ---------------------------------------------------------------------- links
_URLC = "[^\\s<>\"'؀-ۿ]"
_RE_SCHEME = re.compile(rf"(?i)\bhttps?://{_URLC}+")
_RE_WWW = re.compile(rf"(?i)(?<![\w.@/-])www\.{_URLC}+")
_RE_IP = re.compile(rf"(?<![\w.])(\d{{1,3}}(?:\.\d{{1,3}}){{3}})(:\d{{2,5}})?(/{_URLC}*)?")
_RE_BARE = re.compile(
    rf"(?i)(?<![\w@./:-])((?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)+[a-z]{{2,24}})"
    rf"(?![a-z0-9-])(:\d{{2,5}})?(/{_URLC}*)?")
_TRAIL = ".,;:!?)]}'\"…—–"


@dataclass
class Link:
    raw: str
    host: str          # as written (may be Unicode), lowercase
    ascii_host: str    # IDNA/Punycode form ("" if it could not be encoded)
    registered: str    # domain + public suffix, e.g. alrajhibank.com.sa
    subdomain: str
    domain: str
    is_ip: bool


def _make_link(raw: str) -> Link | None:
    url = raw if re.match(r"(?i)https?://", raw) else "http://" + raw
    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return None
    if not host:
        return None
    try:
        ipaddress.ip_address(host)
        return Link(raw, host, host, host, "", host, True)
    except ValueError:
        pass
    if host.isascii():
        ascii_host = host
    else:
        try:
            ascii_host = idna.encode(host, uts46=True).decode()
        except idna.IDNAError:
            ascii_host = ""
    ext = _EXTRACT(ascii_host or host)
    registered = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else host
    return Link(raw, host, ascii_host, registered, ext.subdomain, ext.domain, False)


def extract_links(text: str) -> tuple[list[Link], str]:
    """Return the links in the text and the text with link spans blanked out."""
    links: list[Link] = []
    masked = text

    def take(m: re.Match, kind: str):
        nonlocal masked
        raw = m.group(0).rstrip(_TRAIL)
        if not raw:
            return
        if kind == "ip" and not (m.group(2) or m.group(3)):
            return  # 2.1.3.4 alone is a version number, not a link
        if kind == "bare":
            ext = _EXTRACT(m.group(1).lower())
            if not (ext.domain and ext.suffix):
                return
            if not (m.group(2) or (m.group(3) and len(m.group(3)) > 1)):
                if ext.suffix.split(".")[-1] not in COMMON_TLDS or len(ext.domain) < 2:
                    return
        link = _make_link(raw)
        if link:
            links.append(link)
            s, e = m.start(), m.start() + len(raw)
            masked = masked[:s] + " " * (e - s) + masked[e:]

    for kind, rx in (("scheme", _RE_SCHEME), ("www", _RE_WWW), ("ip", _RE_IP), ("bare", _RE_BARE)):
        for m in list(rx.finditer(masked)):
            take(m, kind)
    return links, masked


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def is_official_for(link: Link, brand: Brand) -> bool:
    h = link.ascii_host or link.host
    if any(_host_matches(h, d) for d in brand.domains):
        return True
    return brand.gov and _host_matches(h, "gov.sa")


# Second-level domains only Saudi government bodies / schools can register.
RESTRICTED_SA = ("gov.sa", "edu.sa")


def is_known_official(link: Link) -> bool:
    """Real domain of any listed brand, or any *.gov.sa / *.edu.sa domain."""
    h = link.ascii_host or link.host
    return (any(_host_matches(h, d) for d in RESTRICTED_SA)
            or any(is_official_for(link, b) for b in load_brands()))


def _keyword_hit(token: str, kw: str) -> bool:
    if token == kw:
        return True
    if len(kw) >= 6 and kw in token:
        return True
    if len(kw) == 5 and token.startswith(kw):
        return True
    if token in COMMON_WORDS or len(token) < 4 or abs(len(token) - len(kw)) > 2:
        return False
    # 5-letter keywords only match exactly or as a prefix: one edit turned "sahem"
    # (a real KSrelief platform) into "saher".
    limit = 2 if len(kw) >= 7 else 1 if len(kw) >= 6 else 0
    return limit > 0 and Levenshtein.distance(token, kw, score_cutoff=limit) <= limit


def lookalike_reason(link: Link) -> str | None:
    if link.is_ip or is_known_official(link):
        return None
    if not link.ascii_host or "xn--" in link.ascii_host:
        return "punycode"
    labels = [l for l in f"{link.subdomain}.{link.domain}".split(".") if l]
    tokens = [t for l in labels for t in l.split("-") if t]
    tokens += [l.replace("-", "") for l in labels if "-" in l]  # al-rajhi -> alrajhi
    for b in load_brands():
        for kw in b.keywords:
            for t in tokens:
                if _keyword_hit(t, kw):
                    return f"{b.id}:{kw}~{t}"
    return None


# --------------------------------------------------------------------- sender
def classify_sender(sender: str | None) -> str:
    """'none', 'phone', 'shortcode' or 'name' (contact name or sender ID)."""
    s = re.sub(r"[\s\-().]", "", (sender or "").translate(_DIGITS))
    if not s:
        return "none"
    if re.fullmatch(r"(\+|00)?\d{7,15}", s):
        return "phone"
    if re.fullmatch(r"\d{3,6}", s):
        return "shortcode"
    return "name"


def is_saudi_number(sender: str) -> bool:
    s = re.sub(r"[\s\-().]", "", sender.translate(_DIGITS))
    if s.startswith("+"):
        return s[1:].startswith("966")
    if s.startswith("00"):
        return s[2:].startswith("966")
    return (s.startswith(("966", "05", "01", "800", "920"))
            or (len(s) == 9 and s.startswith("5")))


# ----------------------------------------------------------------- main entry
def _get(msg, key, default=""):
    if isinstance(msg, dict):
        return msg.get(key) or default
    return getattr(msg, key, None) or default


def explain(message) -> dict:
    """Signals plus the facts behind them (for debugging and review)."""
    text = _get(message, "text")
    sender = _get(message, "sender")
    links, masked = extract_links(text)
    named = brands_in_text(normalize(masked)) | brand_from_sender(sender)
    by_id = {b.id: b for b in load_brands()}
    kind = classify_sender(sender)
    fired: dict[str, str] = {}

    for link in links:
        r = lookalike_reason(link)
        if r and "lookalike_domain" not in fired:
            fired["lookalike_domain"] = f"{link.host} ({r})"
        if named and "brand_link_mismatch" not in fired:
            ok = is_known_official(link) or any(is_official_for(link, by_id[i]) for i in named)
            if not ok:
                fired["brand_link_mismatch"] = f"{sorted(named)} -> {link.host}"
        if (not link.is_ip and not is_known_official(link)
                and any(_host_matches(link.ascii_host or link.host, s) for s in SHORTENERS)):
            fired.setdefault("link_shortener", link.host)
        if link.is_ip:
            fired.setdefault("link_ip", link.host)

    saudi_named = sorted(i for i in named if by_id[i].saudi)
    if kind == "phone" and saudi_named and not is_saudi_number(sender):
        fired["foreign_number_local_brand"] = f"{sender} with {saudi_named}"
    if kind == "phone":
        fired["unknown_sender"] = sender

    return {
        "signals": [s for s in SIGNAL_ORDER if s in fired],
        "reasons": fired,
        "links": [l.host for l in links],
        "brands": sorted(named),
        "sender_kind": kind,
    }


def signals(message) -> list[str]:
    """The signal names for one message, in SIGNAL_ORDER."""
    return explain(message)["signals"]


def format_input(message, sigs: list[str] | None = None) -> str:
    """The one input format used for the teacher, the student and the app."""
    if sigs is None:
        sigs = signals(message)
    source = (_get(message, "source", "sms") or "sms").lower()
    sender = (_get(message, "sender") or "").strip() or "(none)"
    text = (_get(message, "text") or "").strip()[:TEXT_LIMIT]
    return (f"[SOURCE] {source}\n[SENDER] {sender}\n"
            f"[SIGNALS] {'; '.join(sigs) if sigs else 'none'}\n[TEXT] {text}")


if __name__ == "__main__":
    import sys
    msg = {"source": "sms", "sender": sys.argv[1] if len(sys.argv) > 1 else "",
           "text": sys.stdin.read()}
    print(json.dumps(explain(msg), ensure_ascii=False, indent=2))
    print(format_input(msg))
