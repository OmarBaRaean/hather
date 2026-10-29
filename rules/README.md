# Hather rules engine

Plain Python (no AI) that reads one SMS/WhatsApp message and its sender and lists
the tricks a computer can check for certain. The teacher (Run B), the student and
the app all see the same signals. Signals are hints the model weighs, not verdicts;
only `lookalike_domain` forces anything in the app (never green).

```
pip install -r rules/requirements.txt
pytest rules/                                  # 50 fixtures + unit tests
echo "نص الرسالة" | python rules/engine.py AlRajhiBank   # explain one message
python rules/apply_rules.py data/out/pool.jsonl data/out/runA_messages_clean.jsonl \
    data/out/grounded_messages_clean.jsonl -o data/out/runB_pool.jsonl
```

| File | What it is |
| --- | --- |
| `engine.py` | `signals(msg)`, `format_input(msg)`, `explain(msg)`. `msg` has `source`, `sender`, `text` |
| `brands.json` | 91 brands: Arabic/English names, official domains, sender IDs, Saudi/government flags |
| `fixtures.json` | 50 test messages (25 scam, 25 legit, Arabic and English) with hand-written expected signals. The Dart port must pass the same file |
| `test_engine.py` | pytest suite |
| `apply_rules.py` | Adds `signals` and `input` to every non-email record and prints a signal-rate table per group and label |

## Input format

```
[SOURCE] sms | whatsapp
[SENDER] AlRajhiBank            ((none) when there is no sender)
[SIGNALS] lookalike_domain; brand_link_mismatch      (none when empty; always in the order below)
[TEXT] message, stripped, first 1,000 characters
```

Signals are computed on the full text, before trimming.

## The rules, exactly (spec for the Dart port)

**1. Normalize** (for brand matching): NFKC; remove Arabic diacritics and tatweel;
أ إ آ ٱ → ا, ى → ي, ة → ه, ؤ → و, ئ → ي; Arabic-Indic digits → 0-9; lowercase.

**2. Find links**, in this order, blanking each one out before the next pass:
`http(s)://…`; `www.…`; bare IPv4 (only with a port or path, so `2.1.3.4` is a version
number); bare domains (`absher-sa.info/login`). A bare domain with no path or port
counts only if its TLD is in `COMMON_TLDS` and its name is 2+ characters (so `ok.so`,
`app.py`, `S.A.` are not links). Links stop at spaces and Arabic letters; trailing
`.,;:!?)]}'"…—–` is dropped. Placeholders like `<URL>` are not links.
Domains are split with the Public Suffix List **without private domains**, so
`zatca-gov.sa.com` belongs to `sa.com`. Unicode hosts are converted to Punycode.

**3. Find brands named**, from the text with links blanked out (a brand word inside a
link is the look-alike rule's job), plus the sender ID:
- `en`/`ar` names match as whole words. Arabic names may carry the prefixes و ف ب ك ل
  (and `ال` + ل written as `لل`: للراجحي).
- `en_ctx`/`ar_ctx` names are also everyday words (ابشر, زين, نجم, توكلنا, noon, Apple).
  They count only when followed by a colon (`نجم:`) or when a context word is within 3
  words either side (`EN_CONTEXT` whole word or prefix; `AR_CONTEXT` roots as substrings).
- Sender: letters and digits of the sender, normalized, equal a brand's sender ID.

**4. Signals**

| Signal | Fires when |
| --- | --- |
| `lookalike_domain` | A non-IP link that is not official for any brand (and not `*.gov.sa` or `*.edu.sa`, which only government bodies and schools can register) either has a Unicode/`xn--` host, or one of its host words (labels split on `.` and `-`, plus each hyphenated label joined: `al-rajhi` → `alrajhi`; public suffix excluded) matches a brand keyword: equal; contains it (keyword 6+ letters); starts with it (5 letters); or is a misspelling of it: up to 2 edits for keywords of 7+ letters, 1 edit for 6 letters, none for shorter ones, and never for everyday words in `COMMON_WORDS` (mobile, saudi, …) |
| `brand_link_mismatch` | At least one brand is named and a link is not official for any named brand, not official for any other listed brand, and not `*.gov.sa` or `*.edu.sa` |
| `link_shortener` | A link host is in `SHORTENERS` (or a subdomain of one) and is not a listed brand's own domain (`maps.app.goo.gl` is Google's; bare `goo.gl` still counts) |
| `link_ip` | A link host is an IP address |
| `foreign_number_local_brand` | A Saudi brand is named and the sender is a phone number that is not Saudi |
| `unknown_sender` | The sender is a phone number (7-15 digits, optional + or 00), not a contact name, sender ID or short code |

Brand keywords are the `keywords` list plus the name label (3+ letters) of each official domain that has no subdomain (so `maps.app.goo.gl` does not make "goo" a Google keyword).
A link is official for a brand when its host equals or is a subdomain of one of the brand's
domains, or ends in `.gov.sa` for a brand marked `gov`.
Saudi numbers: `+966…`, `00966…`, `966…`, `05…`, `01…` (landline), `800…`, `920…`, or 9 digits starting with 5.
Short codes are 3-6 digits and are treated as local business senders.

## Where this differs from the plan, and why

- **Look-alike edit distance depends on length.** "Within 2 letter edits" for every brand
  flagged `etc.com` (1 edit from `stc`) and `usher.com` (2 from `absher`). Short brand
  names (3-4 letters: stc, spl, snb, najm) now match only as a whole word in the domain.
- **A link to any listed brand's real domain is never a mismatch.** A real SNB OTP says
  "purchase at NOON.COM"; the strict rule would flag it.
- **Brand words inside links do not count as "naming" the brand.** Otherwise every
  look-alike link would also fire `brand_link_mismatch` by itself.
- **`unknown_sender` also fires on text input**, not only screenshots: whenever the
  sender field is a phone number. In training, the sender field plays the role of what
  the screenshot shows.

## Things to check by hand

1. **Fixtures** (`fixtures.json`): I wrote the 50 messages and their expected signals;
   read them and change any expectation you disagree with, then fix the engine.
2. **Sender IDs** in `brands.json` are starting guesses. When the phone exports are in,
   list the real business sender IDs and correct them.
3. **Domains**: verify the official domains of brands you know less well (insurers,
   couriers, apps) before Run B.
4. **Shortcuts**: `apply_rules.py`'s table shows signal rates per group and label. If UCI
   ham has no sender while Smishtank scams have phone numbers, `unknown_sender` becomes a
   perfect English label shortcut. Same for `link_*` signals if the teacher wrote fake
   links into almost every generated scam.

## Known limits

- IMC 2025 messages have links replaced by `<URL>`, so link signals cannot fire on them.
- Unknown domains with no brand named (`allrecipes.com`) fire nothing: the model decides.
- Arabic-script domains written without `http://` or `www.` are not detected as links.
- The Dart port needs the same Public Suffix List behaviour (private domains off);
  the fixtures `F10` (`sa.com`) and `F31` (`com.sa`) catch differences.
