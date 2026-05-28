"""Hermetic Gateway test: POST /chat/stream?stack=rag with lazy-loaded FAISS index.

Issue #133 AC: Gateway POST /chat/stream?stack=rag returns a grounded stream
(sources → status → token(s) → done{passed:true}) when a persisted FAISS index
exists on disk but vectorstore is None (fresh Gateway process).

Setup:
  - Build + persist FAISS index using fake embeddings (no OPENAI_API_KEY).
  - Clear vectorstore to simulate a fresh process.
  - Mock only the LLM getter (no embedding call on lazy-load; FAISS.load_local
    needs get_embeddings() to embed the *query*, so fake embeddings stay active).
  - Assert the full grounded-stream SSE contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
import vector_rag.app.retrieval as vr_retrieval
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from markdown_kb.app.grounding import GroundingOutcome

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# Fake helpers (mirrors test_chat_stream_rag.py to stay offline)
# ---------------------------------------------------------------------------


class _FakeEmbeddings(Embeddings):
    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


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
def _redirect_rag_paths(tmp_path, monkeypatch):
    """Redirect FAISS_INDEX_DIR + LOG_PATH to tmp for isolation."""
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    yield
    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0


@pytest.fixture()
def persisted_index_fresh_process(tmp_path, monkeypatch):
    """Persist a FAISS index to disk, then simulate a fresh process (vectorstore=None).

    Step 1: build + persist the FAISS index using fake embeddings.
    Step 2: set vectorstore=None to mimic a fresh Gateway process that has not
            called POST /index — the lazy-load logic must pick it up from disk.
    The fake embeddings fixture stays active so FAISS.load_local can embed the
    query (embed_documents is NOT called on load, only embed_query at search time).
    """
    fake = _FakeEmbeddings()
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: fake)
    vr_indexer.build_index(REAL_DOCS)

    # Verify index is persisted before simulating fresh process.
    assert vr_indexer.FAISS_INDEX_DIR.exists(), "index must be persisted after build"

    # Simulate fresh Gateway process.
    vr_indexer.vectorstore = None


@pytest.fixture()
def lazy_gateway_client(persisted_index_fresh_process, monkeypatch):
    """TestClient for the Gateway app — RAG index on disk, vectorstore=None, LLM mocked."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(vr_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        vr_retrieval.grounding_module,
        "verify",
        lambda draft, chunks: _approved_outcome(),
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
# AC 4 — Gateway grounded stream with lazy-loaded index
# ---------------------------------------------------------------------------


def test_gateway_rag_stream_lazy_loads_and_returns_200(lazy_gateway_client):
    """Gateway POST /chat/stream?stack=rag returns 200 when index is on disk only.

    Simulates a fresh Gateway process (vectorstore=None, index on disk) —
    the lazy-load fix must make the endpoint serve a grounded stream, not a 200
    with done{passed:false, reason:index_missing}.
    """
    resp = lazy_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


def test_gateway_rag_stream_lazy_loads_grounded_event_order(lazy_gateway_client):
    """Gateway RAG stream lazy-loads and produces: sources, status, token(s), done."""
    resp = lazy_gateway_client.post(
        "/chat/stream?stack=rag",
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


def test_gateway_rag_stream_lazy_loads_done_passed_true(lazy_gateway_client):
    """Gateway RAG stream done event has passed=True (grounded) after lazy-load."""
    resp = lazy_gateway_client.post(
        "/chat/stream?stack=rag",
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


def test_gateway_rag_stream_lazy_loads_sources_non_empty(lazy_gateway_client):
    """Gateway RAG stream sources event is non-empty after lazy-load (not index_missing)."""
    resp = lazy_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    sources_event = next((e for e in events if e["type"] == "sources"), None)

    assert sources_event is not None, "sources event must be present"
    sources = sources_event["data"]["sources"]
    assert len(sources) >= 1, f"sources must be non-empty after lazy-load, got {sources}"
