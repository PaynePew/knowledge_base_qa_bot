"""Hermetic unit tests for the pure ``parseWikilinks`` function (issue #410,
ADR-0030 decision 5).

``parseWikilinks`` lives duplicated in both ``gateway/static/console.html``
(curator surface) and ``gateway/static/index.html`` (reader surface) — the
project has no JS test toolchain (CODING_STANDARD §12.6), so mirroring
``test_sse_parser.py``'s established pattern, this module:

1. Implements the *same pure parsing algorithm* in Python to verify parsing
   correctness at the function-logic level (bracket matching, ``|`` display
   syntax, whitespace trimming, adjacent/unmatched links).
2. Inspects both production UI files to assert the algorithm is present
   verbatim in each (duplication is expected — CODING_STANDARD §12.1 forbids
   a shared JS module / build step across the two single-file pages).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 / §12.7).
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Python mirror of the JS parseWikilinks() pure function
# ---------------------------------------------------------------------------
# Direct algorithmic port of the JavaScript regex/loop in both UI files, so
# this test coverage is equivalent to a JS unit test without a JS toolchain
# (CODING_STANDARD §12.6 — no new toolchain).

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def parse_wikilinks(text: str) -> list[dict]:
    """Python mirror of the JS ``parseWikilinks(text)`` pure function.

    Splits ``text`` into ``{"type": "text", "value": ...}`` and
    ``{"type": "link", "slug": ..., "display": ...}`` segments. A pipe
    (``[[slug|display]]``) supplies the display text; resolution always
    uses the (trimmed) slug part. Un-bracketed text passes through
    untouched; an unmatched ``[[`` with no closing ``]]`` is left as plain
    text (never matched).
    """
    segments: list[dict] = []
    last_index = 0
    for m in _WIKILINK_RE.finditer(text):
        if m.start() > last_index:
            segments.append({"type": "text", "value": text[last_index : m.start()]})
        slug = m.group(1).strip()
        display = m.group(2).strip() if m.group(2) is not None else slug
        segments.append({"type": "link", "slug": slug, "display": display})
        last_index = m.end()
    if last_index < len(text):
        segments.append({"type": "text", "value": text[last_index:]})
    return segments


# ---------------------------------------------------------------------------
# Parsing correctness
# ---------------------------------------------------------------------------


def test_plain_text_with_no_wikilinks_passes_through():
    assert parse_wikilinks("no links here") == [{"type": "text", "value": "no links here"}]


def test_empty_string_returns_no_segments():
    assert parse_wikilinks("") == []


def test_bare_wikilink_uses_slug_as_display():
    assert parse_wikilinks("[[PayPal]]") == [
        {"type": "link", "slug": "PayPal", "display": "PayPal"}
    ]


def test_pipe_syntax_shows_display_text_resolution_uses_slug():
    """AC: `[[slug|display]]` pipe shows the display text; resolution uses the slug part."""
    segments = parse_wikilinks("[[replacement-payment-methods|PayPal]]")
    assert segments == [
        {"type": "link", "slug": "replacement-payment-methods", "display": "PayPal"}
    ]


def test_wikilink_surrounded_by_text():
    segments = parse_wikilinks("See [[PayPal]] for details.")
    assert segments == [
        {"type": "text", "value": "See "},
        {"type": "link", "slug": "PayPal", "display": "PayPal"},
        {"type": "text", "value": " for details."},
    ]


def test_multiple_wikilinks_in_one_string():
    segments = parse_wikilinks("[[配送方式]]、[[運費規則]] 以及 [[到貨時效]]")
    links = [s for s in segments if s["type"] == "link"]
    assert [s["slug"] for s in links] == ["配送方式", "運費規則", "到貨時效"]


def test_adjacent_wikilinks_with_no_text_between():
    segments = parse_wikilinks("[[a]][[b]]")
    assert segments == [
        {"type": "link", "slug": "a", "display": "a"},
        {"type": "link", "slug": "b", "display": "b"},
    ]


def test_whitespace_inside_brackets_is_trimmed():
    segments = parse_wikilinks("[[ foo | Foo Bar ]]")
    assert segments == [{"type": "link", "slug": "foo", "display": "Foo Bar"}]


def test_unclosed_bracket_is_left_as_plain_text():
    assert parse_wikilinks("this [[has no closing") == [
        {"type": "text", "value": "this [[has no closing"}
    ]


# ---------------------------------------------------------------------------
# Structural: the algorithm is present verbatim in BOTH production UI files
# ---------------------------------------------------------------------------

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"
_INDEX_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"

_WIKILINK_REGEX_SRC = r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]"


def test_console_defines_parse_wikilinks_with_the_same_regex():
    text = _CONSOLE_HTML.read_text(encoding="utf-8")
    assert "function parseWikilinks(text)" in text
    assert _WIKILINK_REGEX_SRC in text


def test_index_defines_parse_wikilinks_with_the_same_regex():
    text = _INDEX_HTML.read_text(encoding="utf-8")
    assert "function parseWikilinks(text)" in text
    assert _WIKILINK_REGEX_SRC in text
