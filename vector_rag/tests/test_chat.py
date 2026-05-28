"""Integration tests for vector_rag's /chat end-to-end path (issue #103).

Covers the happy path (grounded answer + citation + grounding field), the
pre-LLM Cannot Confirm gates (index missing, retrieval empty), the post-LLM
grounding rejection (claim_unsupported → Cannot Confirm), and the OpenAI
exception → HTTP status mapping (§4.2). All offline: the embedding leaf is the
``fake_embeddings`` fixture; the answer LLM and the verifier are stubbed.

Run with:
    uv run pytest vector_rag/tests   (skips the one live test)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import openai
from fastapi.testclient import TestClient

import vector_rag.app.indexer as indexer
import vector_rag.app.retrieval as retrieval
from markdown_kb.app.grounding import GroundingOutcome

from .conftest import FakeLLMResponse

REFUND_SOURCE = "refund_policy.md#refund-timeline"


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------
class FakeLLM:
    """Returns a canned grounded answer with a [Source: ...] token; records calls."""

    def __init__(self, content: str):
        self._content = content
        self.call_count = 0
        self.last_messages: list = []

    def invoke(self, messages: list):
        self.call_count += 1
        self.last_messages = messages
        return FakeLLMResponse(content=self._content)


class ErrorLLM:
    """Raises the given exception on every invoke()."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def invoke(self, messages: list):
        raise self._exc


def _approved() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported", result=None)


def _rejected() -> GroundingOutcome:
    return GroundingOutcome(passed=False, reason="claim_unsupported", result=None)


def _make_client(monkeypatch, llm) -> TestClient:
    monkeypatch.setattr(retrieval, "_llm", llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: llm)
    from vector_rag.app.main import app

    return TestClient(app)


def _make_auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError(
        "Incorrect API key provided",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        ),
        body={},
    )


def _make_rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limit exceeded",
        response=httpx.Response(
            429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        ),
        body={},
    )


def _make_api_error() -> openai.APIError:
    return openai.APIError(
        "internal api error",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        body={},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_returns_ok(fake_embeddings):
    from vector_rag.app.main import app

    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Happy path — grounded answer with citation + grounding field
# ---------------------------------------------------------------------------
def test_chat_grounded_answer_with_citation(indexed_corpus, monkeypatch):
    fake_llm = FakeLLM(f"Refunds take 5-7 business days. [Source: {REFUND_SOURCE}]")
    client = _make_client(monkeypatch, fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "[Source:" in body["answer"]
    assert isinstance(body["sources"], list) and len(body["sources"]) > 0
    assert all({"source", "heading", "content"} <= set(s) for s in body["sources"])
    assert body["grounding"]["passed"] is True
    assert body["grounding"]["reason"] == "claim_supported"
    # The LLM was actually invoked with [SystemMessage, HumanMessage].
    assert fake_llm.call_count == 1
    assert len(fake_llm.last_messages) == 2


def test_chat_uses_only_vector_rag_chunks_in_prompt(indexed_corpus, monkeypatch):
    """The prompt CONTEXT is filled exclusively with Stack B's retrieved Chunks."""
    fake_llm = FakeLLM(f"Refunds take 5-7 business days. [Source: {REFUND_SOURCE}]")
    client = _make_client(monkeypatch, fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        client.post("/chat", json={"query": "How long do refunds take?"})

    human_msg = fake_llm.last_messages[1]
    prompt_text = human_msg.content
    assert "CONTEXT:" in prompt_text and "QUESTION:" in prompt_text
    assert "[Source: refund_policy.md#" in prompt_text


# ---------------------------------------------------------------------------
# Pre-LLM gate — index missing (no POST /index yet)
# ---------------------------------------------------------------------------
def test_chat_before_index_says_not_indexed(fake_embeddings, monkeypatch):
    indexer.vectorstore = None
    sentinel = FakeLLM("should not be called")
    client = _make_client(monkeypatch, sentinel)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "not been indexed" in body["answer"].lower()
    assert body["sources"] == []
    assert body["grounding"]["reason"] == "index_missing"
    assert sentinel.call_count == 0, "LLM must not be called before indexing"


# ---------------------------------------------------------------------------
# Pre-LLM gate — retrieval empty
# ---------------------------------------------------------------------------
def test_chat_empty_retrieval_returns_cannot_confirm(indexed_corpus, monkeypatch):
    sentinel = FakeLLM("should not be called")
    monkeypatch.setattr(indexer, "search", lambda q, k=3: [])
    client = _make_client(monkeypatch, sentinel)

    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert body["grounding"]["reason"] == "retrieval_empty"
    assert sentinel.call_count == 0, "LLM must not be called when retrieval is empty"


# ---------------------------------------------------------------------------
# Post-LLM grounding rejection → Cannot Confirm
# ---------------------------------------------------------------------------
def test_chat_grounding_rejection_replaces_with_cannot_confirm(indexed_corpus, monkeypatch):
    fake_llm = FakeLLM("Refunds take 3 days and we ship to Mars for free.")
    client = _make_client(monkeypatch, fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_rejected()):
        resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] == "claim_unsupported"
    # The main LLM is still invoked exactly once (the verifier is the gate).
    assert fake_llm.call_count == 1


# ---------------------------------------------------------------------------
# OpenAI exception mapping (§4.2)
# ---------------------------------------------------------------------------
def test_chat_rate_limit_returns_503(indexed_corpus, monkeypatch):
    client = _make_client(monkeypatch, ErrorLLM(_make_rate_limit_error()))
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"].lower()


def test_chat_auth_error_returns_500(indexed_corpus, monkeypatch):
    client = _make_client(monkeypatch, ErrorLLM(_make_auth_error()))
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 500
    assert "auth" in resp.json()["detail"].lower()


def test_chat_generic_api_error_returns_500(indexed_corpus, monkeypatch):
    client = _make_client(monkeypatch, ErrorLLM(_make_api_error()))
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 500


def test_chat_error_logged_to_vector_rag_channel(indexed_corpus, monkeypatch):
    """chat_error is written to vector_rag's own log channel, not markdown_kb's."""
    import vector_rag.app.logger as vr_logger

    client = _make_client(monkeypatch, ErrorLLM(_make_rate_limit_error()))
    client.post("/chat", json={"query": "How long do refunds take?"})

    log_path: Path = vr_logger.LOG_PATH
    assert log_path.exists(), "vector_rag/log.md must exist after a chat_error"
    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content
    assert "kind=openai_transient" in content
