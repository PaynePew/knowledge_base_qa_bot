"""Unit tests for kb_mcp.normalizer — pure-function result normalizer.

Tests verify that each stack's native search output maps correctly to the
neutral shape: {id, content, score|null}.

No I/O, no LLM calls, no mocking needed — the normalizer is a pure function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kb_mcp.normalizer import normalize_rag_results, normalize_wiki_results

# ---------------------------------------------------------------------------
# Minimal stubs for the stack-specific types
# (avoids importing heavy stack modules just for unit testing the normalizer)
# ---------------------------------------------------------------------------


@dataclass
class _StubSection:
    """Minimal stub mimicking markdown_kb.app.indexer.Section."""

    id: str
    content: str


@dataclass
class _StubChunk:
    """Minimal stub mimicking vector_rag.app.indexer.Chunk."""

    id: str
    content: str


# ---------------------------------------------------------------------------
# normalize_wiki_results
# ---------------------------------------------------------------------------


def test_normalize_wiki_results_maps_id_content_score() -> None:
    """Each (Section, float) pair maps to {id, content, score: float}."""
    hits: list[tuple[Any, float]] = [
        (_StubSection(id="refund-policy#returns", content="You may return within 30 days."), 2.5),
        (_StubSection(id="shipping-faq#cost", content="Shipping is free over $50."), 1.1),
    ]
    result = normalize_wiki_results(hits)
    assert result == [
        {"id": "refund-policy#returns", "content": "You may return within 30 days.", "score": 2.5},
        {"id": "shipping-faq#cost", "content": "Shipping is free over $50.", "score": 1.1},
    ]


def test_normalize_wiki_results_empty() -> None:
    """Empty hit list produces an empty result list."""
    assert normalize_wiki_results([]) == []


def test_normalize_wiki_results_preserves_score_value() -> None:
    """Score is forwarded exactly, including 0.0."""
    hits = [(_StubSection(id="a#b", content="text"), 0.0)]
    result = normalize_wiki_results(hits)
    assert result[0]["score"] == 0.0


# ---------------------------------------------------------------------------
# normalize_rag_results
# ---------------------------------------------------------------------------


def test_normalize_rag_results_maps_id_content_null_score() -> None:
    """Each Chunk maps to {id, content, score: None}."""
    chunks = [
        _StubChunk(id="refund-policy#returns", content="Return info."),
        _StubChunk(id="shipping-faq#cost", content="Shipping info."),
    ]
    result = normalize_rag_results(chunks)
    assert result == [
        {"id": "refund-policy#returns", "content": "Return info.", "score": None},
        {"id": "shipping-faq#cost", "content": "Shipping info.", "score": None},
    ]


def test_normalize_rag_results_empty() -> None:
    """Empty chunk list produces an empty result list."""
    assert normalize_rag_results([]) == []


def test_normalize_rag_results_score_is_always_none() -> None:
    """score is always None regardless of chunk content."""
    chunks = [_StubChunk(id="x#y", content="abc")]
    result = normalize_rag_results(chunks)
    assert result[0]["score"] is None
