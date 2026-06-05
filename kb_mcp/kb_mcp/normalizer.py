"""Result normalizer — pure function mapping stack search output to the MCP neutral shape.

The normalizer is consumed by the MCP tool (JSON) and will be consumed by the
CLI renderer (human display) in Slice 3.  Keeping it as a pure function (no I/O,
no side-effects) makes it trivially testable and reusable.

Neutral shape per ADR-0016 / PRD #198:

    kb_search_v1:
        {
            "stack": "wiki" | "rag",
            "results": [
                {"id": str, "content": str, "score": float | null}
            ]
        }

Stack-specific notes:
- ``wiki`` stack: :func:`markdown_kb.app.indexer.search` returns
  ``list[tuple[Section, float]]`` — the BM25 score is available and is
  forwarded as ``score``.
- ``rag`` stack: :func:`vector_rag.app.indexer.search` returns
  ``list[Chunk]`` (no score exposed) — ``score`` is ``null``.
"""

from __future__ import annotations

from typing import Any


def normalize_wiki_results(
    hits: list[tuple[Any, float]],
) -> list[dict[str, Any]]:
    """Map ``[(Section, float), ...]`` (markdown_kb BM25 results) to the neutral shape.

    Args:
        hits: Output of ``markdown_kb.app.indexer.search`` —
              a list of ``(Section, bm25_score)`` tuples.

    Returns:
        List of ``{"id": str, "content": str, "score": float}`` dicts.
        Never raises; an empty input produces an empty list.
    """
    return [
        {
            "id": section.id,
            "content": section.content,
            "score": score,
        }
        for section, score in hits
    ]


def normalize_rag_results(
    chunks: list[Any],
) -> list[dict[str, Any]]:
    """Map ``[Chunk, ...]`` (vector_rag search results) to the neutral shape.

    Args:
        chunks: Output of ``vector_rag.app.indexer.search`` —
                a list of :class:`vector_rag.app.indexer.Chunk` objects.
                ``score`` is not exposed by the RAG indexer, so it is
                normalised to ``null`` (``None`` in Python).

    Returns:
        List of ``{"id": str, "content": str, "score": None}`` dicts.
        Never raises; an empty input produces an empty list.
    """
    return [
        {
            "id": chunk.id,
            "content": chunk.content,
            "score": None,
        }
        for chunk in chunks
    ]
