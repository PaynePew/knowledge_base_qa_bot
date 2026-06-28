"""Live real-embedding + real-LLM smoke for ``hybrid_kb.query()`` (#313 AC4).

The ONE authorized live test for the hybrid_kb.query() surface (CODING_STANDARD
§6.4 / ADR-0005 surface enumeration; ADR-0018 grants the new query() surface its
single ``@pytest.mark.live``). It exercises the full end-to-end path with the
REAL stack — real ``text-embedding-3-small`` dense build over ``wiki/``, real BM25
seed, and a real ``ChatOpenAI`` synthesis + grounding call — for a clearly
in-scope question, asserting the SHAPE of a grounded, cited answer rather than
exact wording (robust across model updates).

Run with ``uv run pytest -c pyproject.toml hybrid_kb/tests -m live`` (needs
OPENAI_API_KEY; loaded from .env by conftest). Skipped by default. The autouse
path redirect persists the freshly built dense index to tmp, so this never
clobbers the committed ``.kb/hybrid_dense/`` seed; the real committed BM25
``.kb/index.json`` is loaded read-only.

All deterministic behaviour (RRF fusion, the OR-gate, the query() composition,
Cannot Confirm parity) is covered hermetically in ``test_query.py`` /
``test_retrieval.py`` — this live test only proves the real embedding + LLM path
actually runs end to end.
"""

from __future__ import annotations

import pytest

import hybrid_kb.app.dense_index as dense_index
import hybrid_kb.app.query as query_module
import markdown_kb.app.indexer as bm25_indexer
from markdown_kb.app.grounding import GroundingOutcome


@pytest.mark.live
def test_hybrid_query_grounded_smoke():
    """A real refund query returns a grounded, cited answer fused from both arms."""
    # Real committed BM25 seed (read-only) + a real-embedding dense build over the
    # live wiki/ corpus (persisted to the tmp-redirected DENSE_INDEX_DIR).
    bm25_indexer.load_index_json()
    dense_index.build_index()

    result = query_module.query("How long do refunds take?")

    # Response shape parity with the Wiki / RAG stacks.
    assert set(result) >= {"answer", "sources", "grounding_outcome"}
    assert isinstance(result["grounding_outcome"], GroundingOutcome)
    assert result["sources"], "an in-scope refund query must retrieve wiki Sections"
    assert all({"source", "heading", "content"} <= set(s) for s in result["sources"])

    # A clearly in-scope question over the real wiki should ground and cite.
    assert result["grounding_outcome"].passed is True, result["answer"]
    assert "[Source:" in result["answer"]
