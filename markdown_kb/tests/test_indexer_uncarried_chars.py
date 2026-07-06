"""Tests for issue #511 / ADR-0033 "Ingest observability": indexer.count_uncarried_chars.

Companion diagnostic to parse_markdown: the 63-page-book incident (ADR-0033)
showed a degenerate parse can report plain success while most of a Source's
text never reached a Section, with nothing flagging it. Post issue #509
(preamble becomes a Section) the two remaining "no Section emitted" branches —
a whitespace-only preamble, and a non-leaf heading with a whitespace-only body
— by their own qualifying condition already carry zero non-whitespace
characters, so a healthy Source should always compute to 0 here. These tests
pin that invariant across the same shapes issue #509 exercised, plus the
multi-section / single-section shapes issue #511's route tests use.

Tests:
- test_multi_section_source_is_zero: 3-heading source (mirrors
  fixtures/docs/refund_policy.md) -> 0.
- test_zero_heading_source_is_zero: rule-7 single Section -> 0.
- test_preamble_plus_headings_is_zero: rescued preamble (#509) -> 0.
- test_whitespace_only_preamble_is_zero: dropped preamble is pure whitespace -> 0.
- test_non_leaf_heading_no_body_is_zero: h1 title with only h2 children -> 0.
- test_fenced_code_block_with_hash_lines_is_zero: fenced `#` lines are body,
  not headings, and are still fully carried.
"""

from __future__ import annotations

from pathlib import Path

from app.indexer import count_uncarried_chars, parse_markdown


def _write(tmp_dir: Path, filename: str, content: str) -> Path:
    p = tmp_dir / filename
    p.write_text(content, encoding="utf-8")
    return p


def test_multi_section_source_is_zero(tmp_path):
    md = (
        "# Refund Policy\n\n"
        "## Cancellation Window\n\nCustomers can cancel within 24 hours.\n\n"
        "## Refund Timeline\n\nProcessed within 5-7 business days.\n\n"
        "## Non-Refundable Items\n\nGift cards are not refundable.\n"
    )
    p = _write(tmp_path, "refund_policy.md", md)
    sections = parse_markdown(p)

    assert len(sections) == 3, f"Expected 3 sections, got {len(sections)}"
    assert count_uncarried_chars(p, sections) == 0


def test_zero_heading_source_is_zero(tmp_path):
    md = "Just a plain file with no headings at all.\n"
    p = _write(tmp_path, "plain.md", md)
    sections = parse_markdown(p)

    assert len(sections) == 1
    assert count_uncarried_chars(p, sections) == 0


def test_preamble_plus_headings_is_zero(tmp_path):
    md = "Some intro text before any heading.\n\n# First Heading\nBody one.\n"
    p = _write(tmp_path, "intro.md", md)
    sections = parse_markdown(p)

    assert len(sections) == 2, "preamble must be rescued as its own Section (#509)"
    assert count_uncarried_chars(p, sections) == 0


def test_whitespace_only_preamble_is_zero(tmp_path):
    md = "\n\n   \n\n# Heading\nBody.\n"
    p = _write(tmp_path, "blank_preamble.md", md)
    sections = parse_markdown(p)

    assert len(sections) == 1, "whitespace-only preamble creates no Section"
    assert count_uncarried_chars(p, sections) == 0


def test_non_leaf_heading_no_body_is_zero(tmp_path):
    md = "# Title\n\n## Sub\n\nBody.\n"
    p = _write(tmp_path, "title_only.md", md)
    sections = parse_markdown(p)

    assert len(sections) == 1, "the whitespace-only h1 title yields no Section"
    assert count_uncarried_chars(p, sections) == 0


def test_fenced_code_block_with_hash_lines_is_zero(tmp_path):
    md = "```\n# not a heading\ncode line\n```\n\n## Real Heading\n\nBody.\n"
    p = _write(tmp_path, "fenced.md", md)
    sections = parse_markdown(p)

    assert len(sections) == 2, "fenced `#` lines are preamble content, not a heading"
    assert count_uncarried_chars(p, sections) == 0
