"""Unit + integration tests for the transport-agnostic LLM-error contract (ADR-0015).

Covers (markdown_kb side):
  AC-wrapper:  _call_llm_with_error_handling maps each OpenAI exception group to
               the correct LLMError(retryable=...).
  AC-logs:     chat_error with the right kind= is still logged before raising.
  AC-route-mk: markdown_kb /chat route maps LLMError → HTTPException 503/500.

No live OpenAI calls in this file.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import app.retrieval as retrieval_module
from app.errors import LLMError

# ---------------------------------------------------------------------------
# Helpers — OpenAI exception constructors
# ---------------------------------------------------------------------------


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limit exceeded",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body={},
    )


def _auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError(
        "Incorrect API key provided",
        response=httpx.Response(
            401,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body={},
    )


def _api_error() -> openai.APIError:
    return openai.APIError(
        "some internal api error",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        body={},
    )


# ---------------------------------------------------------------------------
# Error LLM stub
# ---------------------------------------------------------------------------


class ErrorLLM:
    """Raises the given exception on every invoke() call."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def invoke(self, messages):  # noqa: ANN001
        raise self._exc


# ---------------------------------------------------------------------------
# AC-wrapper: wrapper raises LLMError (not HTTPException) per exception group
# ---------------------------------------------------------------------------


def test_wrapper_timeout_raises_llmerror_retryable_true(indexed_corpus, monkeypatch):
    """APITimeoutError → LLMError(retryable=True)."""
    fake_llm = ErrorLLM(_timeout_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is True, "APITimeoutError must be retryable=True"


def test_wrapper_rate_limit_raises_llmerror_retryable_true(indexed_corpus, monkeypatch):
    """RateLimitError → LLMError(retryable=True)."""
    fake_llm = ErrorLLM(_rate_limit_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is True, "RateLimitError must be retryable=True"


def test_wrapper_auth_error_raises_llmerror_retryable_false(indexed_corpus, monkeypatch):
    """AuthenticationError → LLMError(retryable=False)."""
    fake_llm = ErrorLLM(_auth_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is False, "AuthenticationError must be retryable=False"


def test_wrapper_api_error_raises_llmerror_retryable_false(indexed_corpus, monkeypatch):
    """Generic APIError → LLMError(retryable=False)."""
    fake_llm = ErrorLLM(_api_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is False, "Generic APIError must be retryable=False"


# ---------------------------------------------------------------------------
# AC-logs: chat_error log is still emitted before raising
# ---------------------------------------------------------------------------


def test_wrapper_timeout_logs_openai_transient(indexed_corpus, monkeypatch):
    """APITimeoutError → chat_error | kind=openai_transient still logged."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_timeout_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError):
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content
    assert "kind=openai_transient" in content


def test_wrapper_auth_logs_openai_auth(indexed_corpus, monkeypatch):
    """AuthenticationError → chat_error | kind=openai_auth still logged."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_auth_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError):
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content
    assert "kind=openai_auth" in content


def test_wrapper_api_error_logs_openai_api(indexed_corpus, monkeypatch):
    """Generic APIError → chat_error | kind=openai_api still logged."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_api_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError):
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content
    assert "kind=openai_api" in content


# ---------------------------------------------------------------------------
# AC-route-mk: markdown_kb /chat route maps LLMError → 503 / 500
# ---------------------------------------------------------------------------


def _mk_client(fake_llm, monkeypatch) -> TestClient:
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    from app.main import app

    return TestClient(app)


def test_mk_route_transient_returns_503(indexed_corpus, monkeypatch):
    """markdown_kb /chat: LLMError(retryable=True) → HTTP 503."""
    client = _mk_client(ErrorLLM(_timeout_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 503


def test_mk_route_rate_limit_returns_503(indexed_corpus, monkeypatch):
    """markdown_kb /chat: LLMError(retryable=True, RateLimitError) → HTTP 503."""
    client = _mk_client(ErrorLLM(_rate_limit_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 503


def test_mk_route_auth_returns_500(indexed_corpus, monkeypatch):
    """markdown_kb /chat: LLMError(retryable=False, auth) → HTTP 500."""
    client = _mk_client(ErrorLLM(_auth_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 500


def test_mk_route_api_error_returns_500(indexed_corpus, monkeypatch):
    """markdown_kb /chat: LLMError(retryable=False) → HTTP 500."""
    client = _mk_client(ErrorLLM(_api_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 500
