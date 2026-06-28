"""Live real-embedding smoke for the dense-over-wiki build (#311 AC6).

The ONE authorized live test for the hybrid_kb dense surface (CODING_STANDARD
§6.4 / ADR-0005 surface enumeration; ADR-0018 grants the dense build its single
``@pytest.mark.live``). Because the committed seed bake already requires real
OpenAI embeddings, this smoke is authorized for S1: it builds the dense index
over the real ``wiki/`` corpus with ``text-embedding-3-small`` and asserts the
SHAPE of the result — not specific neighbours — so it stays robust across model
updates.

Run with ``uv run pytest -m live`` (needs OPENAI_API_KEY; loaded from .env by
conftest). Skipped by default. The autouse path redirect persists the built
index to tmp, so this never clobbers the committed ``.kb/hybrid_dense/`` seed.

All other dense-index behaviour (persist/load, fail-fast, language filter, id
alignment) is covered hermetically with the offline fake embeddings — this live
test only proves the real embedding + FAISS build path actually runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hybrid_kb.app.dense_index as dense_index
from markdown_kb.app.indexer import Section

_COMMITTED_BM25_INDEX = Path(__file__).resolve().parents[2] / ".kb" / "index.json"


@pytest.mark.live
def test_dense_build_over_real_wiki_smoke():
    """Real text-embedding-3-small build over wiki/ yields a usable dense index."""
    expected_ids = {s.id for s in dense_index.filtered_wiki_sections()}
    assert expected_ids, "the real wiki corpus must yield a non-empty filtered list"

    n = dense_index.build_index()
    assert n == len(expected_ids), "every filtered wiki Section must be embedded once"

    # The built index reloads from the (tmp-redirected) seed and reports the count.
    dense_index.vectorstore = None
    assert dense_index.load_dense_index() == n

    # Real-embedding retrieval returns ranked (Section, distance) for an English
    # query, filtered to English Sections — the shape S2 fuses over.
    results = dense_index.search_with_distance("How long do refunds take?", k=3)
    assert results, "a refund query must return real dense hits"
    for section, distance in results:
        assert isinstance(section, Section)
        assert isinstance(distance, float)
    assert {s.metadata.get("lang") for s, _ in results} == {"en"}, (
        "an English query must return only English Sections"
    )

    # Built ids are a subset of the committed BM25 wiki ids (same-corpus invariant
    # holds for the freshly real-embedded index too).
    bm25_ids = {
        s["id"]
        for s in json.loads(_COMMITTED_BM25_INDEX.read_text("utf-8"))["sections"]
    }
    assert {s.id for s, _ in results} <= bm25_ids
