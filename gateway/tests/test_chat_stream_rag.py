"""Gateway endpoint tests for POST /chat/stream?stack=rag (Phase 9 Slice 3 / issue #120).

Tests assert the RAG SSE event sequence and the gateway dispatch:
  sources (RAG source shape: id + heading + content, NO score, NO derived_from)
    → token(s) → done{passed, reason, filed=null}

The LLM (via vector_rag.app.retrieval) and grounding verifier are mocked;
fake embeddings (via vector_rag.app.indexer) are used. No OPENAI_API_KEY.
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
# Fake embeddings (mirrors vector_rag conftest._FakeEmbeddings)
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


# ---------------------------------------------------------------------------
# Fake LLM stub
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_rag_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect vector_rag paths to tmp for all rag-stream gateway tests."""
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")


@pytest.fixture()
def indexed_rag_corpus(tmp_path, monkeypatch):
    """Build the FAISS index from REAL_DOCS with fake embeddings."""
    fake = _FakeEmbeddings()
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: fake)
    vr_indexer.build_index(REAL_DOCS)
    yield
    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0


def _approved_outcome() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported", result=None)


@pytest.fixture()
def rag_gateway_client(indexed_rag_corpus, monkeypatch):
    """TestClient for the Gateway app with mocked RAG LLM."""
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
    """Parse a multi-frame SSE response into a list of {type, data} dicts."""
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
# RAG dispatch: 501 is gone, real dispatch lands
# ---------------------------------------------------------------------------


def test_chat_stream_rag_no_longer_returns_501(rag_gateway_client):
    """stack=rag now returns 200 (real dispatch), not 501."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# RAG happy-path: correct SSE event order
# ---------------------------------------------------------------------------


def test_chat_stream_rag_event_order(rag_gateway_client):
    """POST /chat/stream?stack=rag emits: sources, then token(s), then done."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    assert types[0] == "sources", f"Expected first event 'sources', got {types[0]!r}"
    assert types[-1] == "done", f"Expected last event 'done', got {types[-1]!r}"
    assert all(t == "token" for t in types[1:-1]), f"Middle events must all be 'token': {types}"


def test_chat_stream_rag_sources_event_non_empty(rag_gateway_client):
    """RAG sources event carries at least one source with required fields."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    sources_event = events[0]
    assert sources_event["type"] == "sources"
    sources = sources_event["data"]["sources"]
    assert len(sources) >= 1
    item = sources[0]
    assert "source" in item
    assert "heading" in item
    assert "content" in item


def test_chat_stream_rag_sources_have_no_score(rag_gateway_client):
    """RAG source objects in SSE carry NO 'score' field (issue #120 spec)."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    sources_event = events[0]
    for src in sources_event["data"]["sources"]:
        assert "score" not in src, f"RAG source must not carry 'score': {src}"


def test_chat_stream_rag_sources_have_no_derived_from(rag_gateway_client):
    """RAG source objects in SSE carry NO 'derived_from' field (issue #120 spec)."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    sources_event = events[0]
    for src in sources_event["data"]["sources"]:
        assert "derived_from" not in src, f"RAG source must not carry 'derived_from': {src}"


def test_chat_stream_rag_token_events_form_verified_answer(rag_gateway_client):
    """Joining all token texts reconstructs the LLM-generated (verified) answer."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    tokens = [e["data"]["text"] for e in events if e["type"] == "token"]
    answer = "".join(tokens)
    assert "5-7 business days" in answer


def test_chat_stream_rag_done_event_passed_true(rag_gateway_client):
    """done event carries passed=True for a grounded RAG query."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    assert done["data"]["passed"] is True
    assert done["data"]["reason"] == "claim_supported"


def test_chat_stream_rag_done_filed_always_null(rag_gateway_client):
    """RAG stream done event always has filed=null (RAG never files)."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    assert done["data"].get("filed") is None, (
        f"RAG done.filed must always be null, got: {done['data'].get('filed')}"
    )


# ---------------------------------------------------------------------------
# Unknown / missing stack: still 400 after rag dispatch is live
# ---------------------------------------------------------------------------


def test_chat_stream_unknown_stack_still_400(rag_gateway_client):
    """Unknown stack still returns HTTP 400 after rag dispatch is live."""
    resp = rag_gateway_client.post(
        "/chat/stream?stack=unknown",
        json={"query": "test"},
    )
    assert resp.status_code == 400
