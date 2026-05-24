"""Live integration smoke test for the /chat endpoint — Slice 6.

Makes a real OpenAI API call to confirm the model follows the SYSTEM_PROMPT's
cite-and-fallback rules. Opt-in only: skipped by default; run with:

    pytest -m live   (from markdown_kb/)

Requirements:
    OPENAI_API_KEY must be set in the environment; the test fails with a clear
    message if it is absent rather than silently passing or skipping.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer
import app.logger as logger_module

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_chat_refund_query_live(tmp_path, monkeypatch):
    """POST /chat with a real refund query against the real OpenAI API.

    Assertions are SHAPE-only:
      - HTTP 200
      - "[Source:" appears in the answer (model cited a section)
      - sources is a non-empty list
      - at least one source's 'source' field contains 'refund_policy.md#'

    No specific prose words are asserted so the test stays robust across
    model updates.
    """
    # Fail with a clear message if the API key is absent — do NOT silently pass.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail(
            "OPENAI_API_KEY is not set. "
            "Export your key before running live tests: "
            "export OPENAI_API_KEY=sk-..."
        )

    # Redirect index and log to tmp so we don't pollute the real .kb / wiki/
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"
    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)

    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    # Reset any cached LLM singleton so we get a fresh real ChatOpenAI instance
    # with the current OPENAI_API_KEY from the environment.
    import app.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "_llm", None)
    monkeypatch.setattr(retrieval_module, "_retry_llm", None)

    # Build the real section index from docs/
    indexer.build_index(REAL_DOCS)

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    # Shape assertion 1: HTTP 200
    assert resp.status_code == 200, (
        f"Expected HTTP 200, got {resp.status_code}: {resp.text}"
    )

    body = resp.json()

    # Shape assertion 2: "[Source:" appears in the answer
    assert "[Source:" in body.get("answer", ""), (
        f"Expected '[Source:' citation in answer, got: {body.get('answer')!r}"
    )

    # Shape assertion 3: sources is a non-empty list
    sources = body.get("sources", [])
    assert isinstance(sources, list) and len(sources) > 0, (
        f"Expected non-empty sources list, got: {sources!r}"
    )

    # Shape assertion 4: at least one source references refund_policy.md#
    refund_sources = [s for s in sources if "refund_policy.md#" in s.get("source", "")]
    assert refund_sources, (
        f"Expected at least one source with 'refund_policy.md#', got sources: "
        f"{[s.get('source') for s in sources]}"
    )

    # Clean up in-memory state
    indexer.sections.clear()
