"""Unit tests for ``indexer.search(exclude_qa=True)`` (tier-B S4, issue #380,
ADR-0026 decision 1).

The C9 Re-file remediation re-synthesizes a stale Filed Answer's question
through the chat pipeline. Without exclusion, a live qa page can retrieve —
and re-cite — itself: its own body was written to directly answer the
question, so it often out-scores the entity/concept page under plain BM25
term-frequency scoring. ``exclude_qa`` closes that trap by dropping every
``metadata["type"] == "qa"`` Section from candidate scoring before ranking.

Hand-built Section fixtures (mirrors ``test_expand_to_pages.py``'s pattern) —
no real files, no ``build_index()`` — so the BM25 corpus is fully controlled
and the "qa would otherwise be the top hit" premise is proven, not assumed.
"""

from __future__ import annotations

import app.indexer as indexer
from app.indexer import Section, search, tokenize


def _sec(file: str, section_type: str, content: str) -> Section:
    return Section(
        id=f"{file}#body",
        file=file,
        heading="Body",
        heading_path=["Body"],
        content=content,
        tokens=tokenize(content),
        metadata={"type": section_type, "lang": "en"},
    )


# A stale Filed Answer whose body densely repeats the question's own terms —
# under plain BM25 term-frequency scoring this out-ranks the sparser entity
# page for the identical question (the exact trap ADR-0026 names).
QA_SECTION = _sec(
    "cancel-window-abc123",
    "qa",
    "You can cancel your order within the cancellation window. To cancel "
    "your order within the cancellation window, contact support before the "
    "cancellation window closes. Cancel order cancellation window.",
)

# The current entity page — sparser term overlap, but the actual source of
# truth an entity/concept-derived answer must re-derive from.
ENTITY_SECTION = _sec(
    "acme-shop",
    "entity",
    "Orders may be cancelled within the cancellation window for a full refund.",
)

QUESTION = "how do i cancel my order within the cancellation window"


def _install(monkeypatch) -> None:
    monkeypatch.setattr(indexer, "sections", [QA_SECTION, ENTITY_SECTION])
    indexer.rebuild_stats()


def test_qa_page_would_otherwise_be_top_hit(monkeypatch):
    """Premise check: without exclusion, the stale qa page out-ranks the
    entity page for its own question (proves the fixture is load-bearing,
    not merely plausible)."""
    _install(monkeypatch)

    ranked = search(QUESTION, k=3)

    assert ranked, "expected at least one BM25 hit"
    assert ranked[0][0].file == "cancel-window-abc123", (
        f"expected the qa page to out-rank the entity page absent exclusion, "
        f"got: {[(s.file, round(sc, 3)) for s, sc in ranked]}"
    )


def test_exclude_qa_drops_qa_sections_from_ranking(monkeypatch):
    """With exclude_qa=True, the same query never returns the qa Section at
    any rank — the entity page becomes the (only) top hit."""
    _install(monkeypatch)

    ranked = search(QUESTION, k=3, exclude_qa=True)

    files = [s.file for s, _score in ranked]
    assert "cancel-window-abc123" not in files, (
        f"qa Section must never appear when exclude_qa=True, got: {files}"
    )
    assert ranked, "the entity page must still be retrievable"
    assert ranked[0][0].file == "acme-shop"


def test_exclude_qa_default_false_preserves_existing_behaviour(monkeypatch):
    """Default call shape (no exclude_qa arg) is unchanged — every existing
    caller (``/chat``, ``stream_query``, CLI/MCP/hybrid) keeps seeing qa
    Sections in candidate scoring."""
    _install(monkeypatch)

    ranked = search(QUESTION, k=3)

    files = [s.file for s, _score in ranked]
    assert "cancel-window-abc123" in files
