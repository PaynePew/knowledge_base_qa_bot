"""Unit tests for ``app.structure_enrichment`` (ADR-0033 decision 2, issue #512).

AC coverage:
  - Longform predicate: table-driven over heading count / preamble share /
    oversized-section cases; filename-, size-, and page-count-blind.
  - Well-headed Sources bypass enrichment byte-identically (no LLM call, no
    frontmatter change).
  - Mocked-LLM: headings materialized at proposed boundaries; every
    resulting Section <= the per-section cap (mechanical re-split fallback
    covered).
  - Page-furniture fixture mirroring the real transcript (34 repeated
    lines) is stripped; non-repeating content is untouched.
  - Enrichment LLM failure fails soft to the un-enriched transcript.

The LLM is mocked at the lazy-singleton getter (``get_enrichment_llm``), per
CODING_STANDARD §6.3 — never the deep-module entry points themselves.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import app.indexer as indexer_module
import app.structure_enrichment as se

FILLER = "Lorem ipsum filler text about nothing in particular. "


def _padded(prefix: str, repeats: int = 40) -> str:
    """A paragraph starting with ``prefix`` verbatim, padded past the min-chars floor share."""
    return prefix + " " + (FILLER * repeats)


def _make_fake_llm(chapters: list[SimpleNamespace]) -> MagicMock:
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = SimpleNamespace(chapters=chapters)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


# ---------------------------------------------------------------------------
# Longform predicate — table-driven
# ---------------------------------------------------------------------------


def test_is_longform_false_below_min_chars_floor(monkeypatch):
    """A tiny zero-heading doc never counts as longform (nothing to segment)."""
    monkeypatch.delenv("KB_LONGFORM_MIN_CHARS", raising=False)
    assert se.is_longform("hello world, no headings here.") is False


def test_is_longform_true_zero_headings_long_body():
    body = "\n\n".join(_padded(f"Paragraph {i} starts here.", 20) for i in range(5))
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is True


def test_is_longform_true_one_heading_long_body():
    body = "# Stray Page Header\n\n" + "\n\n".join(
        _padded(f"Paragraph {i} starts here.", 20) for i in range(5)
    )
    assert se.is_longform(body) is True


def test_is_longform_true_dominant_preamble():
    preamble = _padded("The preamble dominates this document.", 60)
    body = preamble + "\n\n## Heading A\n\nShort body A.\n\n## Heading B\n\nShort body B.\n"
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is True


def test_is_longform_false_well_headed_handbook():
    """Multiple headings, small preamble, no oversized Section => NOT longform."""
    sections = "\n\n".join(
        f"## Chapter {i}\n\n{_padded(f'Chapter {i} content.', 15)}" for i in range(4)
    )
    body = sections
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is False


def test_is_longform_true_oversized_section(monkeypatch):
    """Well-headed on the surface, but one Section blows the per-section cap."""
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "40")  # cap ~= 120 chars
    body = (
        "## Short Chapter\n\nJust a little text.\n\n"
        f"## Huge Chapter\n\n{_padded('This chapter is way too big.', 40)}\n"
    )
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is True


# ---------------------------------------------------------------------------
# Page-furniture stripping
# ---------------------------------------------------------------------------


def test_strip_page_furniture_removes_34_repeated_lines():
    """Mirrors the real transcript: 34 identical running-header lines stripped."""
    furniture = "Sample Corp — Internal Report — 2024-01-01 — www.example.com — Page"
    blocks = [
        f"{furniture}\n\nParagraph {i} carries unique content about topic {i}." for i in range(34)
    ]
    body = "\n\n".join(blocks)

    stripped, removed = se._strip_page_furniture(body)

    assert removed == 34
    assert furniture not in stripped
    for i in range(34):
        assert f"Paragraph {i} carries unique content about topic {i}." in stripped


def test_strip_page_furniture_leaves_headings_and_rules_alone():
    """A repeated heading or horizontal rule is never treated as furniture."""
    body = "\n\n".join(["## Same Title", "body one", "---", "## Same Title", "body two", "---"] * 2)
    stripped, removed = se._strip_page_furniture(body)
    assert removed == 0
    assert stripped == body


def test_strip_page_furniture_untouched_when_no_repeats():
    body = "\n\n".join(f"Unique paragraph number {i}." for i in range(10))
    stripped, removed = se._strip_page_furniture(body)
    assert removed == 0
    assert stripped == body


# ---------------------------------------------------------------------------
# enrich_structure — bypass (well-headed), mocked-LLM success, fail-soft
# ---------------------------------------------------------------------------


def test_enrich_structure_bypasses_well_headed_source_byte_identical(monkeypatch):
    """A well-headed Source is untouched: no LLM call, enriched=False, body unchanged."""

    def _boom() -> None:
        raise AssertionError("get_enrichment_llm must not be called for a well-headed Source")

    monkeypatch.setattr(se, "get_enrichment_llm", _boom)

    body = "\n\n".join(
        f"## Chapter {i}\n\n{_padded(f'Chapter {i} content.', 15)}" for i in range(4)
    )
    result = se.enrich_structure(body, filename="handbook.md")

    assert result.enriched is False
    assert result.reason is None
    assert result.body == body


def test_enrich_structure_materializes_headings_at_proposed_boundaries(monkeypatch):
    para1 = _padded("Chapter One begins here.", 30)
    para2 = _padded("Chapter Two starts now.", 30)
    para3 = _padded("Chapter Three opens here.", 30)
    body = "\n\n".join([para1, para2, para3])
    assert se.is_longform(body) is True  # zero headings, well past the min-chars floor

    chapters = [
        SimpleNamespace(title="Chapter One", boundary_anchor="Chapter One begins here."),
        SimpleNamespace(title="Chapter Two", boundary_anchor="Chapter Two starts now."),
        SimpleNamespace(title="Chapter Three", boundary_anchor="Chapter Three opens here."),
    ]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    result = se.enrich_structure(body, filename="book.md")

    assert result.enriched is True
    assert result.reason is None

    # Headings appear in the proposed order.
    idx1 = result.body.index("## Chapter One")
    idx2 = result.body.index("## Chapter Two")
    idx3 = result.body.index("## Chapter Three")
    assert idx1 < idx2 < idx3

    sections = indexer_module.parse_markdown_body(result.body, source_prefix="book.md")
    headings = [s.heading for s in sections]
    assert headings == ["Chapter One", "Chapter Two", "Chapter Three"]
    assert "Chapter One begins here." in sections[0].content
    assert "Chapter Two starts now." in sections[1].content
    assert "Chapter Three opens here." in sections[2].content


def _fixed_len_paragraph(prefix: str, target_len: int) -> str:
    """A paragraph of EXACTLY ``target_len`` chars, starting with ``prefix`` verbatim."""
    body = prefix + " "
    pad = "filler word "
    while len(body) < target_len:
        body += pad
    return body[:target_len]


def test_enrich_structure_oversized_chapter_mechanically_resplit(monkeypatch):
    """A single proposed chapter that blows the cap gets paragraph-boundary re-split.

    Each individual paragraph (150 chars => 50 estimated tokens) stays safely
    UNDER the cap on its own; only the combined chapter (15 paragraphs) blows
    it, so the mechanical re-split fallback must pack multiple whole
    paragraphs per emitted Section, never exceeding the cap.
    """
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "100")  # cap ~= 300 chars/section

    paragraphs = [
        _fixed_len_paragraph(f"Paragraph {i} of the big chapter.", 150) for i in range(15)
    ]
    body = "\n\n".join(paragraphs)
    assert len(body.strip()) >= 2000

    chapters = [SimpleNamespace(title="Big Chapter", boundary_anchor="Paragraph 0 of")]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    result = se.enrich_structure(body, filename="big.md")

    assert result.enriched is True
    sections = indexer_module.parse_markdown_body(result.body, source_prefix="big.md")
    assert len(sections) > 1, "Expected the oversized chapter to be re-split into >1 Section"
    cap = 100
    for sec in sections:
        assert sec.heading.startswith("Big Chapter")
        assert se.ingest_module.estimate_tokens(sec.content) <= cap


def test_enrich_structure_fails_soft_on_llm_error(monkeypatch):
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("simulated model failure")
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    body = "\n\n".join(_padded(f"Paragraph {i} starts here.", 20) for i in range(5))
    assert se.is_longform(body) is True

    result = se.enrich_structure(body, filename="broken.md")

    assert result.enriched is False
    assert result.body == body, "Un-enriched transcript must be returned byte-identical on failure"
    assert result.reason is not None
    assert "simulated model failure" in result.reason


def test_enrich_structure_fails_soft_on_unfindable_boundary_anchor(monkeypatch):
    """A proposal whose anchor text is not in the document degrades gracefully."""
    chapters = [SimpleNamespace(title="Ghost Chapter", boundary_anchor="text that never appears")]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    body = "\n\n".join(_padded(f"Paragraph {i} starts here.", 20) for i in range(5))

    result = se.enrich_structure(body, filename="ghost.md")

    assert result.enriched is False
    assert result.body == body
    assert result.reason is not None
