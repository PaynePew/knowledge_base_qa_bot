"""Hermetic Gateway test: POST /chat/stream?stack=wiki with lazy-loaded index.

Issue #148 AC: Gateway POST /chat/stream?stack=wiki returns a grounded stream
(sources → status → token(s) → done{passed:true}) when a persisted .kb/index.json
exists on disk but sections=[] (fresh Gateway process).

Setup:
  - Build + persist index using real docs/ (BM25 only, no embeddings needed).
  - Clear sections to simulate a fresh process.
  - Mock only the LLM getter (no network calls needed — BM25 is purely offline).
  - Assert the full grounded-stream SSE contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger
import markdown_kb.app.retrieval as mk_retrieval
import pytest
from fastapi.testclient import TestClient
from markdown_kb.app.grounding import GroundingOutcome

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    CANNED_ANSWER = (
        "Approved refunds are processed within 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


def _approved_outcome() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported", result=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_wiki_paths(tmp_path, monkeypatch):
    """Redirect INDEX_PATH, LOG_PATH, WIKI_DIR to tmp for isolation."""
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(mk_indexer, "WIKI_DIR", tmp_path / "wiki")
    yield
    mk_indexer.sections.clear()


@pytest.fixture()
def persisted_index_fresh_process(monkeypatch):
    """Persist a wiki index to disk, then simulate a fresh process (sections=[]).

    Step 1: build + persist the index to tmp INDEX_PATH.
    Step 2: clear in-memory sections to mimic a fresh Gateway process that has
            not called POST /wiki/index — the lazy-load logic must pick it up.
    """
    mk_indexer.build_index(REAL_DOCS)

    # Verify index is persisted before simulating fresh process.
    assert mk_indexer.INDEX_PATH.exists(), "index must be persisted after build"

    # Simulate fresh Gateway process.
    mk_indexer.sections.clear()
    assert not mk_indexer.sections


@pytest.fixture()
def lazy_wiki_gateway_client(persisted_index_fresh_process, monkeypatch):
    """TestClient for the Gateway app — wiki index on disk, sections=[], LLM mocked."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(mk_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(mk_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        mk_retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse_response(content: str) -> list[dict]:
    events = []
    for frame in content.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        lines = frame.split("\n")
        event_type = "message"
        data_str = ""
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:]
        if data_str:
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"raw": data_str}
            events.append({"type": event_type, "data": data})
    return events


# ---------------------------------------------------------------------------
# AC 4 — Gateway grounded stream with lazy-loaded wiki index
# ---------------------------------------------------------------------------


def test_gateway_wiki_stream_lazy_loads_and_returns_200(lazy_wiki_gateway_client):
    """Gateway POST /chat/stream?stack=wiki returns 200 when index is on disk only.

    Simulates a fresh Gateway process (sections=[], index on disk) —
    the lazy-load fix must make the endpoint serve a grounded stream, not a 200
    with done{passed:false, reason:index_missing}.
    """
    resp = lazy_wiki_gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


def test_gateway_wiki_stream_lazy_loads_grounded_event_order(lazy_wiki_gateway_client):
    """Gateway wiki stream lazy-loads and produces: sources, status, token(s), done."""
    resp = lazy_wiki_gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    assert types[0] == "sources", f"First event must be 'sources', got {types[0]!r}"
    assert types[-1] == "done", f"Last event must be 'done', got {types[-1]!r}"
    middle = types[1:-1]
    assert middle and middle[0] == "status", f"'status' event expected after sources: {types}"
    assert all(t == "token" for t in middle[1:]), (
        f"Events after status must all be 'token': {types}"
    )


def test_gateway_wiki_stream_lazy_loads_done_passed_true(lazy_wiki_gateway_client):
    """Gateway wiki stream done event has passed=True (grounded) after lazy-load."""
    resp = lazy_wiki_gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    done = events[-1]

    assert done["type"] == "done"
    assert done["data"]["grounding"]["passed"] is True, (
        f"done.grounding.passed must be True after lazy-load, got: {done['data']['grounding']}"
    )
    assert done["data"]["grounding"]["reason"] != "index_missing", (
        f"done.grounding.reason must NOT be index_missing after lazy-load, "
        f"got: {done['data']['grounding']['reason']}"
    )


def test_gateway_wiki_stream_lazy_loads_sources_non_empty(lazy_wiki_gateway_client):
    """Gateway wiki stream sources event is non-empty after lazy-load (not index_missing)."""
    resp = lazy_wiki_gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    sources_event = next((e for e in events if e["type"] == "sources"), None)

    assert sources_event is not None, "sources event must be present"
    sources = sources_event["data"]["sources"]
    assert len(sources) >= 1, f"sources must be non-empty after lazy-load, got {sources}"
