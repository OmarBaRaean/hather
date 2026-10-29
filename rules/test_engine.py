"""Run with:  pytest rules/   (the Dart port runs the same fixtures.json)."""
import json
from pathlib import Path

import pytest

from engine import (SIGNAL_ORDER, classify_sender, explain, extract_links,
                    format_input, is_saudi_number, load_brands, signals)

FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("fx", FIXTURES, ids=[f["id"] for f in FIXTURES])
def test_fixture(fx):
    got = explain(fx)
    assert got["signals"] == fx["expected"], got


def test_fixture_mix():
    assert len(FIXTURES) == 50
    assert sum(f["label"] == "scam" for f in FIXTURES) == 25
    assert {f["lang"] for f in FIXTURES} == {"ar", "en"}
    assert len({f["id"] for f in FIXTURES}) == 50
    for f in FIXTURES:  # expected lists use the canonical names and order
        assert f["expected"] == [s for s in SIGNAL_ORDER if s in f["expected"]]


def test_brand_list():
    brands = load_brands()
    assert len(brands) >= 80
    assert len({b.id for b in brands}) == len(brands)
    assert all(b.domains for b in brands)


def test_format_input():
    msg = {"source": "whatsapp", "sender": "", "text": "  hello  "}
    assert format_input(msg) == "[SOURCE] whatsapp\n[SENDER] (none)\n[SIGNALS] none\n[TEXT] hello"
    long = {"source": "sms", "sender": "STC", "text": "a" * 1500}
    assert format_input(long).endswith("a" * 1000)
    assert format_input(FIXTURES[0]).splitlines()[2] == "[SIGNALS] lookalike_domain; brand_link_mismatch"


@pytest.mark.parametrize("sender,kind", [
    ("+966 50 123 4567", "phone"), ("٠٥٥١٢٣٤٥٦٧", "phone"), ("00447700900123", "phone"),
    ("959", "shortcode"), ("88888", "shortcode"), ("AlRajhiBank", "name"),
    ("أمي", "name"), ("", "none"), (None, "none"),
])
def test_classify_sender(sender, kind):
    assert classify_sender(sender) == kind


@pytest.mark.parametrize("number,saudi", [
    ("+966501234567", True), ("00966501234567", True), ("0501234567", True),
    ("966501234567", True), ("920012345", True), ("8001234567", True),
    ("+447700900123", False), ("+12025550143", False), ("01001234567", True),
])
def test_saudi_number(number, saudi):
    assert is_saudi_number(number) == saudi


@pytest.mark.parametrize("text,hosts", [
    ("go to https://Example.com/a?b=1, now", ["example.com"]),
    ("visit www.alrajhibank.com.sa.", ["www.alrajhibank.com.sa"]),
    ("bit.ly/abc and t.co/x", ["bit.ly", "t.co"]),
    ("email me at ali@gmail.com", []),
    ("price 2.5 SAR, v1.2.3.4, at 10.30, e.g. ok.so app.py", []),
    ("الرابط:https://x-pay.top/الدفع", ["x-pay.top"]),
    ("http://10.0.0.1:8080/login", ["10.0.0.1"]),
    ("placeholder <URL> only", []),
])
def test_extract_links(text, hosts):
    assert [l.host for l in extract_links(text)[0]] == hosts


def test_links_not_counted_as_brand_names():
    # The brand word inside a link is the look-alike's job, not a brand mention.
    e = explain({"sender": "", "text": "check alrajhi-gift.com"})
    assert e["brands"] == []
    assert e["signals"] == ["lookalike_domain"]


def test_arabic_prefixes_and_variants():
    for t in ["حولت عبر بالراجحي", "للراجحي فرع", "والراجحي", "مصرف الانماء", "مصرف الإنماء"]:
        assert explain({"sender": "", "text": t})["brands"], t


def test_restricted_saudi_domains_are_official():
    for t in ["التسجيل عبر sab.edu.sa/apply", "اضغط https://www.absher.gov.sa"]:
        assert explain({"sender": "", "text": t})["signals"] == [], t


def test_real_business_domains_from_phone_export():
    # Seen in real business SMS (Oct 23 phone export): none of these may fire.
    for sender, text in [("900", "باقتك على my.stc و stc.sa"), ("Tabby-AD", "تسوق الآن tabby.sa"),
                         ("JARIR-AD", "عروض جرير jarir.sa"), ("Jahez-AD", "اطلب من جاهز jahez.link/x"),
                         ("MOH", "صحتي sehhaty.sa"), ("ksrelief", "تبرع sahem.ksrelief.org"),
                         ("GHE", "الموقع maps.app.goo.gl/abc")]:
        sig = explain({"sender": sender, "text": text})["signals"]
        assert "lookalike_domain" not in sig and "link_shortener" not in sig, (text, sig)
    assert explain({"sender": "", "text": "goo.gl/abc"})["signals"] == ["link_shortener"]
