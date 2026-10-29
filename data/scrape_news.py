"""
Hather: collect real Saudi scam examples from public news and warnings,
then have an LLM extract and clean them.

Stage 1 (collect): Google News RSS search (Arabic, Saudi edition) for scam-warning
  phrases -> fetch each article -> extract main text -> data/out/scrape/articles.jsonl
Stage 2 (extract): send each article to an OpenAI-compatible LLM server
  (llama-server on Kaggle, or any local server) -> data/out/scrape/scam_items.jsonl

Usage:
  pip install httpx feedparser trafilatura googlenewsdecoder "selectolax<1.0"
  python data/scrape_news.py collect --days 365   (works from any folder)
  python data/scrape_news.py extract --base-url http://localhost:8080/v1 --model qwen

Both stages are resumable: they skip URLs already done.
"""

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

# Always data/out/scrape next to this script, whatever folder you run it from.
OUT = Path(__file__).resolve().parent / "out" / "scrape"
ARTICLES = OUT / "articles.jsonl"
ITEMS = OUT / "scam_items.jsonl"

# Search phrases. Each one becomes a Google News query restricted to Saudi results.
QUERIES = [
    "رسائل احتيالية",
    "رسالة احتيال",
    "رسائل نصية احتيالية",
    "تحذير من رسائل",
    "روابط احتيالية",
    "انتحال صفة البنك",
    "احتيال واتساب",
    "رسائل وهمية شحنة",
    "سبل رسالة احتيال",
    "أبشر رسالة احتيال",
    "البنك المركزي السعودي يحذر",
    "لجنة الإعلام والتوعية المصرفية تحذر",
    "الأمن العام يحذر احتيال",
    "هيئة الاتصالات رسائل احتيال",
    "تصيد إلكتروني السعودية",
]

# Extra sites known to publish warnings; added as site: filters.
SITES = ["sabq.org", "okaz.com.sa", "alwatan.com.sa", "ajel.sa", "aleqt.com",
         "sama.gov.sa", "spa.gov.sa", "alarabiya.net", "aawsat.com", "argaam.com"]

UA = "Mozilla/5.0 (research crawler for Hather scam-detection project; contact via GitHub)"


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def load_ids(path: Path, key: str) -> set:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)[key] for line in f if line.strip()}


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- stage 1
def rss_url(q: str, days: int) -> str:
    q = f"{q} when:{days}d"
    return f"https://news.google.com/rss/search?q={quote(q)}&hl=ar&gl=SA&ceid=SA:ar"


def real_url(u: str) -> str:
    """Turn a news.google.com/rss/articles/... link into the publisher's real URL."""
    if "news.google.com" not in u:
        return u
    from googlenewsdecoder import gnewsdecoder
    res = gnewsdecoder(u, interval=1)
    # newer versions report "success", older ones "status"
    if not (res.get("success") or res.get("status")):
        raise RuntimeError(f"could not decode Google News link: {res.get('message')}")
    return res["decoded_url"]


def collect(days: int, delay: float, limit: int | None) -> None:
    import feedparser
    import httpx
    import trafilatura

    # Only articles that actually got text count as done; empty ones are retried.
    done = set()
    if ARTICLES.exists():
        with ARTICLES.open(encoding="utf-8") as f:
            done = {r["id"] for r in map(json.loads, filter(str.strip, f)) if r.get("chars", 0) > 0}
    queries = QUERIES + [f"احتيال site:{s}" for s in SITES]
    links = {}
    for q in queries:
        feed = feedparser.parse(rss_url(q, days), agent=UA)
        for e in feed.entries:
            links.setdefault(e.link, {"title": e.title, "published": e.get("published", ""),
                                      "source": e.get("source", {}).get("title", ""), "query": q})
        print(f"[rss] {q!r}: {len(feed.entries)} results ({len(links)} unique so far)")
        time.sleep(delay)

    todo = [(u, m) for u, m in links.items() if url_id(u) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[fetch] {len(todo)} new articles to fetch")

    seen_text = set()
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=30) as client:
        for i, (u, meta) in enumerate(todo, 1):
            try:
                # Google News RSS links are not real redirects; decode them to the news site's URL.
                real = real_url(u)
                r = client.get(real)
                r.raise_for_status()
                text = trafilatura.extract(r.text, include_comments=False) or ""
            except Exception as ex:  # keep going; record the failure
                print(f"  [{i}] fail {u[:80]}: {ex}")
                append(ARTICLES, {"id": url_id(u), "url": u, **meta, "error": str(ex), "text": ""})
                time.sleep(delay)
                continue
            h = hashlib.sha1(re.sub(r"\s+", " ", text).encode()).hexdigest()
            dup = h in seen_text
            seen_text.add(h)
            append(ARTICLES, {"id": url_id(u), "url": u, "final_url": str(r.url), **meta,
                              "chars": len(text), "duplicate": dup, "text": text[:12000]})
            print(f"  [{i}/{len(todo)}] {len(text):>6} chars  {meta['title'][:70]}")
            time.sleep(delay)


# ---------------------------------------------------------------- stage 2
SYSTEM = """You extract real scam messages from Arabic or English news articles about fraud in Saudi Arabia and the Gulf.
Return JSON only, matching the schema. Rules:
- "messages": ONLY scam messages quoted word-for-word in the article (SMS, WhatsApp, email, or call script). Copy the exact text; never invent or complete a message. If the article only describes a scam without quoting it, return an empty list.
- "patterns": short descriptions of each scam the article describes (quoted or not).
- Replace personal names, phone numbers and account numbers with <NAME>, <PHONE>, <ACCOUNT>. Keep links and brand names as written.
- "country": the country the warning is about (SA, AE, EG, ...), or "unknown".
- "relevant": false if the article is not about scam messages at all."""

SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "country": {"type": "string"},
        "messages": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "channel": {"type": "string", "enum": ["sms", "whatsapp", "email", "call", "other"]},
            "impersonates": {"type": "string"}},
            "required": ["text", "channel", "impersonates"]}},
        "patterns": {"type": "array", "items": {"type": "object", "properties": {
            "description": {"type": "string"},
            "channel": {"type": "string", "enum": ["sms", "whatsapp", "email", "call", "other"]},
            "impersonates": {"type": "string"},
            "lure": {"type": "string"}},
            "required": ["description", "channel", "impersonates", "lure"]}},
    },
    "required": ["relevant", "country", "messages", "patterns"],
}


def verbatim(msg: str, article: str) -> bool:
    """True if the extracted message really appears in the article (guards against invention)."""
    norm = lambda s: re.sub(r"[\s\u064B-\u0652«»\"'“”.,،:]+", "", s)
    m, a = norm(msg), norm(article)
    if not m:
        return False
    if m in a:
        return True
    # allow placeholder substitutions: every long chunk between placeholders must appear
    chunks = [c for c in re.split(r"<(?:NAME|PHONE|ACCOUNT)>", m) if len(c) >= 8]
    return bool(chunks) and all(c in a for c in chunks)


async def extract(base_url: str, model: str, parallel: int, api_key: str) -> None:
    import httpx

    done = load_ids(ITEMS, "article_id")
    with ARTICLES.open(encoding="utf-8") as f:
        arts = [json.loads(l) for l in f if l.strip()]
    arts = [a for a in arts if a["id"] not in done and a.get("chars", 0) >= 200 and not a.get("duplicate")]
    print(f"[extract] {len(arts)} articles to process")
    sem = asyncio.Semaphore(parallel)

    async def one(client, a):
        body = {
            "model": model, "temperature": 0.1,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"TITLE: {a['title']}\nSOURCE: {a['source']}\n\n{a['text'][:8000]}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.post(f"{base_url}/chat/completions", json=body)
                    r.raise_for_status()
                    out = json.loads(r.json()["choices"][0]["message"]["content"])
                    break
                except Exception as ex:
                    err = str(ex)
                    await asyncio.sleep(2 * (attempt + 1))
            else:
                append(ITEMS, {"article_id": a["id"], "url": a["url"], "error": err})
                return
        for m in out.get("messages", []):
            m["verbatim"] = verbatim(m["text"], a["text"])
        append(ITEMS, {"article_id": a["id"], "url": a.get("final_url", a["url"]), "title": a["title"],
                       "source": a["source"], "published": a["published"], **out})
        n = len(out.get("messages", []))
        print(f"  {a['id']}  relevant={out.get('relevant')}  msgs={n}  patterns={len(out.get('patterns', []))}")

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=300, headers=headers) as client:
        await asyncio.gather(*(one(client, a) for a in arts))
    report()


def report() -> None:
    if not ITEMS.exists():
        return
    recs = [json.loads(l) for l in ITEMS.open(encoding="utf-8") if l.strip()]
    ok = [r for r in recs if "error" not in r]
    rel = [r for r in ok if r.get("relevant")]
    msgs = [m for r in rel for m in r.get("messages", [])]
    real = [m for m in msgs if m.get("verbatim")]
    print("\n=== report ===")
    print(f"articles processed: {len(recs)}  errors: {len(recs) - len(ok)}  relevant: {len(rel)}")
    print(f"patterns: {sum(len(r.get('patterns', [])) for r in rel)}")
    print(f"quoted messages: {len(msgs)}  verified verbatim: {len(real)}")
    by_c = {}
    for r in rel:
        by_c[r.get("country", "?")] = by_c.get(r.get("country", "?"), 0) + 1
    print(f"by country: {by_c}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--days", type=int, default=365)
    c.add_argument("--delay", type=float, default=2.0, help="seconds between requests (be polite)")
    c.add_argument("--limit", type=int)
    e = sub.add_parser("extract")
    e.add_argument("--base-url", default="http://localhost:8080/v1")
    e.add_argument("--model", default="qwen")
    e.add_argument("--parallel", type=int, default=4)
    e.add_argument("--api-key", default="")
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.days, a.delay, a.limit)
    elif a.cmd == "extract":
        asyncio.run(extract(a.base_url, a.model, a.parallel, a.api_key))
    else:
        report()


if __name__ == "__main__":
    main()
