"""Integration tests for error handling (Slice 4 — issue #5).

Covers:
  AC 1: APITimeoutError / RateLimitError → HTTP 503, log kind=openai_transient
  AC 2: AuthenticationError → HTTP 500, log kind=openai_auth
  AC 3: generic APIError → HTTP 500, log kind=openai_api
  AC 4: ungrounded LLM response (no [Source:) → grounding retry → replaced with
        Cannot Confirm phrase and sources==[], LLM invoked exactly twice
  AC 5: exact Cannot Confirm phrase on first call → passed through unchanged,
        LLM invoked exactly once

All tests run without live OpenAI calls.

Run with:
    pytest -m "not live"   (from markdown_kb/)
"""
from __future__ import annotations

from pathlib import Path

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer
import app.logger as logger_module
import app.retrieval as retrieval_module

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"

CANNOT_CONFIRM_PHRASE = "I cannot confirm from the knowledge base."
REFUND_SECTION_ID = "refund_policy.md#refund-timeline"

# ---------------------------------------------------------------------------
# Helpers for constructing OpenAI exceptions
# ---------------------------------------------------------------------------


def _make_api_timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )


def _make_rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limit exceeded",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body={},
    )


def _make_auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError(
        "Incorrect API key provided",
        response=httpx.Response(
            401,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body={},
    )


def _make_api_error() -> openai.APIError:
    return openai.APIError(
        "some internal api error",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        body={},
    )


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------


class ErrorLLM:
    """Raises the given exception on every invoke() call."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def invoke(self, messages: list):
        raise self._exc


class CountingLLM:
    """Returns configurable responses, counts calls."""

    def __init__(self, responses: list[str]):
        """responses: list of content strings to return in order (last repeated)."""
        self._responses = responses
        self.call_count = 0

    def invoke(self, messages: list):
        self.call_count += 1
        idx = min(self.call_count - 1, len(self._responses) - 1)
        content = self._responses[idx]

        class _Resp:
            pass

        r = _Resp()
        r.content = content
        return r


# ---------------------------------------------------------------------------
# Shared fixture: indexed corpus + log redirection
# ---------------------------------------------------------------------------


@pytest.fixture()
def indexed_corpus(tmp_path, monkeypatch):
    """Index real docs/ into a tmp location; clean up after each test."""
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"
    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)

    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    indexer.build_index(REAL_DOCS)

    yield {"log_path": log_path}

    indexer.sections.clear()


# ---------------------------------------------------------------------------
# Helpers for building a TestClient with a given fake LLM
# ---------------------------------------------------------------------------


def _make_client(fake_llm, monkeypatch, retry_llm=None) -> TestClient:
    """Create a TestClient with the given fake LLM injected for both primary and retry calls."""
    effective_retry = retry_llm if retry_llm is not None else fake_llm
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(retrieval_module, "_retry_llm", effective_retry)
    monkeypatch.setattr(retrieval_module, "get_retry_llm", lambda: effective_retry)
    from app.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# AC 1a: APITimeoutError → HTTP 503 with correct detail + openai_transient log
# ---------------------------------------------------------------------------


def test_api_timeout_returns_503(indexed_corpus, monkeypatch):
    """APITimeoutError → HTTP 503 with 'LLM service temporarily unavailable' detail."""
    fake_llm = ErrorLLM(_make_api_timeout_error())
    client = _make_client(fake_llm, monkeypatch)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 503, (
        f"Expected 503, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "detail" in body, f"Expected 'detail' in error response, got: {body}"
    assert "temporarily unavailable" in body["detail"].lower(), (
        f"Expected 'temporarily unavailable' in detail, got: {body['detail']!r}"
    )
    assert "retry" in body["detail"].lower(), (
        f"Expected 'retry' in detail, got: {body['detail']!r}"
    )


def test_api_timeout_logs_openai_transient(indexed_corpus, monkeypatch):
    """APITimeoutError → wiki/log.md contains chat_error | … kind=openai_transient."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_make_api_timeout_error())
    client = _make_client(fake_llm, monkeypatch)

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content, (
        f"Expected 'chat_error |' in log, got:\n{content}"
    )
    assert "kind=openai_transient" in content, (
        f"Expected 'kind=openai_transient' in log, got:\n{content}"
    )


# ---------------------------------------------------------------------------
# AC 1b: RateLimitError → HTTP 503 with correct detail + openai_transient log
# ---------------------------------------------------------------------------


def test_rate_limit_returns_503(indexed_corpus, monkeypatch):
    """RateLimitError → HTTP 503 with 'LLM service temporarily unavailable' detail."""
    fake_llm = ErrorLLM(_make_rate_limit_error())
    client = _make_client(fake_llm, monkeypatch)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 503, (
        f"Expected 503, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "temporarily unavailable" in body["detail"].lower(), (
        f"Expected 'temporarily unavailable' in detail, got: {body['detail']!r}"
    )


def test_rate_limit_logs_openai_transient(indexed_corpus, monkeypatch):
    """RateLimitError → wiki/log.md contains chat_error | … kind=openai_transient."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_make_rate_limit_error())
    client = _make_client(fake_llm, monkeypatch)

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content, (
        f"Expected 'chat_error |' in log, got:\n{content}"
    )
    assert "kind=openai_transient" in content, (
        f"Expected 'kind=openai_transient' in log, got:\n{content}"
    )


# ---------------------------------------------------------------------------
# AC 2: AuthenticationError → HTTP 500 with auth detail + openai_auth log
# ---------------------------------------------------------------------------


def test_auth_error_returns_500(indexed_corpus, monkeypatch):
    """AuthenticationError → HTTP 500 with auth-failure detail."""
    fake_llm = ErrorLLM(_make_auth_error())
    client = _make_client(fake_llm, monkeypatch)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 500, (
        f"Expected 500, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "detail" in body, f"Expected 'detail' in error response, got: {body}"
    assert "auth" in body["detail"].lower(), (
        f"Expected 'auth' in detail, got: {body['detail']!r}"
    )
    assert "openai_api_key" in body["detail"].lower() or "api_key" in body["detail"].lower(), (
        f"Expected 'OPENAI_API_KEY' reference in detail, got: {body['detail']!r}"
    )


def test_auth_error_logs_openai_auth(indexed_corpus, monkeypatch):
    """AuthenticationError → wiki/log.md contains chat_error | … kind=openai_auth."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_make_auth_error())
    client = _make_client(fake_llm, monkeypatch)

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content, (
        f"Expected 'chat_error |' in log, got:\n{content}"
    )
    assert "kind=openai_auth" in content, (
        f"Expected 'kind=openai_auth' in log, got:\n{content}"
    )


# ---------------------------------------------------------------------------
# AC 3: generic APIError → HTTP 500 + openai_api log
# ---------------------------------------------------------------------------


def test_generic_api_error_returns_500(indexed_corpus, monkeypatch):
    """Generic APIError → HTTP 500 with error detail."""
    fake_llm = ErrorLLM(_make_api_error())
    client = _make_client(fake_llm, monkeypatch)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 500, (
        f"Expected 500, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "detail" in body, f"Expected 'detail' in error response, got: {body}"


def test_generic_api_error_logs_openai_api(indexed_corpus, monkeypatch):
    """Generic APIError → wiki/log.md contains chat_error | … kind=openai_api."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_make_api_error())
    client = _make_client(fake_llm, monkeypatch)

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content, (
        f"Expected 'chat_error |' in log, got:\n{content}"
    )
    assert "kind=openai_api" in content, (
        f"Expected 'kind=openai_api' in log, got:\n{content}"
    )


# ---------------------------------------------------------------------------
# AC 4: ungrounded response (no [Source:) → retry once → replace with Cannot
#        Confirm when second response is also ungrounded; LLM invoked exactly 2x
# ---------------------------------------------------------------------------


def test_ungrounded_response_replaced_with_cannot_confirm(indexed_corpus, monkeypatch):
    """When LLM returns text without [Source: → retry → replace with Cannot Confirm."""
    # Both responses lack [Source: and are not the exact Cannot Confirm phrase
    fake_llm = CountingLLM(["This is an answer without any citation."])
    client = _make_client(fake_llm, monkeypatch)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["answer"] == CANNOT_CONFIRM_PHRASE, (
        f"Expected exact Cannot Confirm phrase, got: {body['answer']!r}"
    )
    assert body["sources"] == [], (
        f"Expected sources == [], got: {body['sources']}"
    )


def test_ungrounded_response_invokes_llm_twice(indexed_corpus, monkeypatch):
    """When LLM response is ungrounded → retry → LLM must be invoked exactly twice."""
    fake_llm = CountingLLM(["No citations here at all."])
    client = _make_client(fake_llm, monkeypatch)

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert fake_llm.call_count == 2, (
        f"Expected LLM to be invoked exactly 2 times, got: {fake_llm.call_count}"
    )


# ---------------------------------------------------------------------------
# AC 5: exact Cannot Confirm phrase on first call → passed through, LLM called once
# ---------------------------------------------------------------------------


def test_cannot_confirm_phrase_passed_through_unchanged(indexed_corpus, monkeypatch):
    """When LLM returns exact Cannot Confirm phrase → pass through unchanged."""
    fake_llm = CountingLLM([CANNOT_CONFIRM_PHRASE])
    client = _make_client(fake_llm, monkeypatch)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["answer"] == CANNOT_CONFIRM_PHRASE, (
        f"Expected Cannot Confirm phrase passed through, got: {body['answer']!r}"
    )


def test_cannot_confirm_phrase_invokes_llm_once(indexed_corpus, monkeypatch):
    """When LLM returns exact Cannot Confirm phrase → LLM invoked exactly once."""
    fake_llm = CountingLLM([CANNOT_CONFIRM_PHRASE])
    client = _make_client(fake_llm, monkeypatch)

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert fake_llm.call_count == 1, (
        f"Expected LLM to be invoked exactly 1 time, got: {fake_llm.call_count}"
    )
