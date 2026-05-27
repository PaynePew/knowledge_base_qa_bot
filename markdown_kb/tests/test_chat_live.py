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

# Canonical path to the production wiki/ directory (seed content from #51).
# The live test must index the real wiki/, not the tmp dir that the autouse
# _redirect_paths_to_tmp fixture sets up, because the autouse tmp wiki is empty.
_REAL_WIKI = Path(__file__).resolve().parents[2] / "wiki"

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
      - at least one source's 'source' field contains 'refund-timeline#'
        (A2 bare-slug form; seed wiki page: wiki/concepts/refund-timeline.md)

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

    # Point WIKI_DIR at the real production wiki/ so SOURCE_DIRS resolves to
    # [wiki/entities, wiki/concepts] with the seed content from issue #51.
    # The autouse _redirect_paths_to_tmp sets WIKI_DIR to an empty tmp dir,
    # so we explicitly override it here for the live smoke test.
    monkeypatch.setattr(indexer, "WIKI_DIR", _REAL_WIKI)
    entities_dir = _REAL_WIKI / "entities"
    concepts_dir = _REAL_WIKI / "concepts"
    monkeypatch.setattr(indexer, "SOURCE_DIRS", [entities_dir, concepts_dir])

    # Reset any cached LLM singleton so we get a fresh real ChatOpenAI instance
    # with the current OPENAI_API_KEY from the environment.
    import app.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "_llm", None)

    # Build the real section index from the production wiki/ (default SOURCE_DIRS).
    indexer.build_index()

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    # Shape assertion 1: HTTP 200
    assert resp.status_code == 200, f"Expected HTTP 200, got {resp.status_code}: {resp.text}"

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

    # Shape assertion 4: at least one source references refund-timeline# (A2 bare-slug form).
    # The seed wiki page wiki/concepts/refund-timeline.md produces Section IDs of the form
    # refund-timeline#<heading-slug> (e.g. refund-timeline#refund-timeline).
    refund_sources = [s for s in sources if "refund-timeline#" in s.get("source", "")]
    assert refund_sources, (
        f"Expected at least one source with 'refund-timeline#', got sources: "
        f"{[s.get('source') for s in sources]}"
    )

    # Shape assertion 5: grounding field is present and shows claim_supported
    grounding = body.get("grounding")
    assert grounding is not None, "ChatResponse must have 'grounding' field"
    assert grounding.get("passed") is True, (
        f"Expected grounding.passed=True for supported query, got: {grounding}"
    )
    assert grounding.get("reason") == "claim_supported", (
        f"Expected grounding.reason=claim_supported, got: {grounding.get('reason')!r}"
    )
    assert grounding.get("claims") is not None and len(grounding["claims"]) > 0, (
        f"Expected non-empty grounding.claims, got: {grounding.get('claims')}"
    )

    # Clean up in-memory state
    indexer.sections.clear()
