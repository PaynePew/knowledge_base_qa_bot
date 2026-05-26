"""Unit tests for indexer.expand_to_pages() — Slice 4-4 (issue #49).

Tests cover all acceptance criteria for the pure-function contract:
  - empty hits → empty output
  - single-page hits → that page's full Section set returned
  - cross-page hits → both pages expanded, correct page ordering
  - dedup: two BM25 hits in the same page → page included once
  - section ordering within a page: document order (file-top to file-bottom)
  - page ordering: page whose top hit ranks highest comes first

All tests are hermetic (no OPENAI_API_KEY, no I/O, no real index build).
"""

from __future__ import annotations

import app.indexer as indexer
from app.indexer import Section, expand_to_pages

# ---------------------------------------------------------------------------
# Helpers — build lightweight Section fixtures without calling parse_markdown
# ---------------------------------------------------------------------------


def _sec(file: str, heading: str, order: int = 0) -> Section:
    """Create a bare Section with the given file slug and heading.

    ``order`` is a monotone integer that embeds document position into the
    heading_path so tests can assert section ordering without a real file.
    The tokens list is empty because BM25 scoring is not exercised here.
    """
    return Section(
        id=f"{file}#{heading.lower().replace(' ', '-')}",
        file=file,
        heading=heading,
        heading_path=[heading],
        content=f"Content of {heading}",
        tokens=[],
    )


# ---------------------------------------------------------------------------
# Fixtures — three pages with two Sections each
# ---------------------------------------------------------------------------

# Page A: sections in document order (top → bottom)
A1 = _sec("page-a", "Introduction", order=0)
A2 = _sec("page-a", "Details", order=1)
A3 = _sec("page-a", "Summary", order=2)

# Page B: two sections
B1 = _sec("page-b", "Overview", order=0)
B2 = _sec("page-b", "Examples", order=1)

# Page C: one section
C1 = _sec("page-c", "Only Section", order=0)

ALL_SECTIONS = [A1, A2, A3, B1, B2, C1]


# ---------------------------------------------------------------------------
# Test: empty hits → empty output
# ---------------------------------------------------------------------------


def test_expand_empty_hits_returns_empty(monkeypatch):
    """expand_to_pages([]) == []."""
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)
    result = expand_to_pages([])
    assert result == []


# ---------------------------------------------------------------------------
# Test: single hit from page-a → full page-a returned in document order
# ---------------------------------------------------------------------------


def test_expand_single_hit_returns_full_page(monkeypatch):
    """A single hit from page-a expands to all sections of page-a in document order."""
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)
    result = expand_to_pages([A2])  # hit is the middle section

    # All three sections of page-a should be returned
    assert len(result) == 3
    assert A1 in result
    assert A2 in result
    assert A3 in result

    # Document order: A1 < A2 < A3
    assert result.index(A1) < result.index(A2) < result.index(A3)


# ---------------------------------------------------------------------------
# Test: hits from two pages → both pages expanded
# ---------------------------------------------------------------------------


def test_expand_cross_page_hits_returns_both_pages(monkeypatch):
    """Hits from page-a and page-b → both pages fully expanded."""
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)
    result = expand_to_pages([A1, B2])

    # Should contain all 3 sections from page-a and all 2 from page-b
    result_ids = {s.id for s in result}
    expected_ids = {A1.id, A2.id, A3.id, B1.id, B2.id}
    assert result_ids == expected_ids, f"Expected sections {expected_ids}, got {result_ids}"


# ---------------------------------------------------------------------------
# Test: page ordering — page whose top hit ranks highest comes first
# ---------------------------------------------------------------------------


def test_expand_page_order_highest_hit_rank_first(monkeypatch):
    """The page whose top hit is ranked first in `hits` comes first in output.

    hits = [B1, A2] → page-b has rank-0 hit, page-a has rank-1 hit.
    So page-b sections come before page-a sections.
    """
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)
    result = expand_to_pages([B1, A2])  # page-b ranks higher (rank 0)

    # First section must belong to page-b
    assert result[0].file == "page-b", (
        f"Expected first section from page-b (highest-ranked hit), got file={result[0].file!r}"
    )
    # All page-b sections appear before any page-a section
    page_b_positions = [i for i, s in enumerate(result) if s.file == "page-b"]
    page_a_positions = [i for i, s in enumerate(result) if s.file == "page-a"]
    assert max(page_b_positions) < min(page_a_positions), (
        f"All page-b sections should precede all page-a sections. "
        f"page-b positions: {page_b_positions}, page-a positions: {page_a_positions}"
    )


# ---------------------------------------------------------------------------
# Test: dedup — two BM25 hits in the same page → page included once
# ---------------------------------------------------------------------------


def test_expand_dedup_two_hits_same_page(monkeypatch):
    """When two BM25 hits land in the same page, that page appears once.

    Input: [A1, A3] (both from page-a).
    Output: A1, A2, A3 — exactly once each.
    """
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)
    result = expand_to_pages([A1, A3])

    # page-a appears exactly once (no duplication)
    page_a_sections = [s for s in result if s.file == "page-a"]
    assert len(page_a_sections) == 3, (
        f"Expected 3 sections from page-a (all unique), got {len(page_a_sections)}: {page_a_sections}"
    )

    # No duplicates at all
    assert len(result) == len(set(id(s) for s in result)), "No section should appear twice"


# ---------------------------------------------------------------------------
# Test: section ordering within a page follows document order
# ---------------------------------------------------------------------------


def test_expand_section_order_within_page_is_document_order(monkeypatch):
    """Sections within an expanded page appear in document order (indexer.sections order)."""
    # Arrange: sections list has A1, A2, A3 in order (document order = order in sections list)
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)

    # Hit is A3 (last section), which should NOT cause page to appear in reverse
    result = expand_to_pages([A3])

    page_a_in_result = [s for s in result if s.file == "page-a"]
    assert page_a_in_result == [A1, A2, A3], (
        f"Expected [A1, A2, A3] in document order, got {[s.heading for s in page_a_in_result]}"
    )


# ---------------------------------------------------------------------------
# Test: hit from page with single section → returns that one section
# ---------------------------------------------------------------------------


def test_expand_single_section_page(monkeypatch):
    """A page with only one section is expanded to just that one section."""
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)
    result = expand_to_pages([C1])

    assert result == [C1]


# ---------------------------------------------------------------------------
# Test: hits order determines page order (first-hit rank determines page rank)
# ---------------------------------------------------------------------------


def test_expand_page_order_respects_hit_rank(monkeypatch):
    """Page ordering is determined by the position of the first/best hit for each page.

    hits = [A2, B1, C1] → page-a first, page-b second, page-c third.
    """
    monkeypatch.setattr(indexer, "sections", ALL_SECTIONS)
    result = expand_to_pages([A2, B1, C1])

    # Groups: page-a (hit rank 0), page-b (hit rank 1), page-c (hit rank 2)
    files_in_order = []
    for s in result:
        if not files_in_order or files_in_order[-1] != s.file:
            files_in_order.append(s.file)

    assert files_in_order == ["page-a", "page-b", "page-c"], (
        f"Expected page order [page-a, page-b, page-c], got {files_in_order}"
    )


# ---------------------------------------------------------------------------
# Test: pure function — does not mutate indexer.sections
# ---------------------------------------------------------------------------


def test_expand_does_not_mutate_sections(monkeypatch):
    """expand_to_pages is pure: it does not mutate indexer.sections."""
    monkeypatch.setattr(indexer, "sections", list(ALL_SECTIONS))  # copy
    original_ids = [id(s) for s in indexer.sections]

    expand_to_pages([A1, B2])

    # sections list identity unchanged
    assert [id(s) for s in indexer.sections] == original_ids, (
        "expand_to_pages must not mutate indexer.sections"
    )
