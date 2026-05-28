"""Live smoke test for vector_rag's /chat surface — the ONE per-surface live test.

This is the single PRD-authorised ``@pytest.mark.live`` test for the LLM-facing
surface added by issue #103 (vector_rag /chat + embeddings), enumerated in
ADR-0005 § Consequences. It makes real OpenAI embedding + chat + verifier calls
to confirm the end-to-end grounded path works. Opt-in only; skipped by default:

    uv run pytest -m live vector_rag/tests

Assertions are SHAPE-only (§6.4) — never specific prose words — so the test
outlives model updates.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import vector_rag.app.indexer as indexer

from .conftest import REAL_DOCS


@pytest.mark.live
def test_chat_refund_query_live(monkeypatch):
    """POST /chat with a real refund query against the real OpenAI API.

    Shape assertions only:
      - HTTP 200
      - "[Source:" appears in the answer (the model cited a chunk)
      - sources is a non-empty list
      - the grounding field is present
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail(
            "OPENAI_API_KEY is not set. Export your key before running live tests: "
            "export OPENAI_API_KEY=sk-..."
        )

    # The autouse _redirect_paths_to_tmp fixture already points FAISS_INDEX_DIR
    # and the log paths at tmp, so this build does not pollute production .kb/.
    # Reset the real LLM singleton so it picks up the current OPENAI_API_KEY.
    import vector_rag.app.retrieval as retrieval

    monkeypatch.setattr(retrieval, "_llm", None)
    indexer._embeddings = None

    indexer.build_index(REAL_DOCS)

    from vector_rag.app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, (
        f"Expected HTTP 200, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()

    assert "[Source:" in body.get("answer", ""), (
        f"Expected '[Source:' citation in answer, got: {body.get('answer')!r}"
    )

    sources = body.get("sources", [])
    assert isinstance(sources, list) and len(sources) > 0, (
        f"Expected non-empty sources list, got: {sources!r}"
    )

    grounding = body.get("grounding")
    assert grounding is not None, "ChatResponse must have a 'grounding' field"
    assert "passed" in grounding and "reason" in grounding

    indexer.vectorstore = None
