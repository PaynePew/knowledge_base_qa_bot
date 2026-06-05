"""Tests for transport-agnostic LLM-error contract in vector_rag stack (ADR-0015).

Covers:
  AC-wrapper-vr: _call_llm_with_error_handling maps each OpenAI exception group to
                 the correct LLMError(retryable=...).
  AC-route-vr:   vector_rag /chat route maps LLMError → HTTPException 503/500.

No live OpenAI calls in this file.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import vector_rag.app.retrieval as retrieval_module
from markdown_kb.app.errors import LLMError

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


class ErrorLLM:
    """Raises the given exception on every invoke() call."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def invoke(self, messages):  # noqa: ANN001
        raise self._exc


# ---------------------------------------------------------------------------
# AC-wrapper-vr: vector_rag wrapper raises LLMError per exception group
# ---------------------------------------------------------------------------


def test_vr_wrapper_timeout_raises_llmerror_retryable_true(indexed_corpus, monkeypatch):
    """APITimeoutError → LLMError(retryable=True) in vector_rag wrapper."""
    fake_llm = ErrorLLM(_timeout_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is True


def test_vr_wrapper_rate_limit_raises_llmerror_retryable_true(
    indexed_corpus, monkeypatch
):
    """RateLimitError → LLMError(retryable=True) in vector_rag wrapper."""
    fake_llm = ErrorLLM(_rate_limit_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is True


def test_vr_wrapper_auth_error_raises_llmerror_retryable_false(
    indexed_corpus, monkeypatch
):
    """AuthenticationError → LLMError(retryable=False) in vector_rag wrapper."""
    fake_llm = ErrorLLM(_auth_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is False


def test_vr_wrapper_api_error_raises_llmerror_retryable_false(
    indexed_corpus, monkeypatch
):
    """Generic APIError → LLMError(retryable=False) in vector_rag wrapper."""
    fake_llm = ErrorLLM(_api_error())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)

    with pytest.raises(LLMError) as exc_info:
        retrieval_module._call_llm_with_error_handling("test q", "prompt")

    assert exc_info.value.retryable is False


# ---------------------------------------------------------------------------
# AC-route-vr: vector_rag /chat route maps LLMError → 503 / 500
# ---------------------------------------------------------------------------


def _vr_client(fake_llm, monkeypatch) -> TestClient:
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    from vector_rag.app.main import app

    return TestClient(app)


def test_vr_route_transient_returns_503(indexed_corpus, monkeypatch):
    """vector_rag /chat: LLMError(retryable=True) → HTTP 503."""
    client = _vr_client(ErrorLLM(_timeout_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 503


def test_vr_route_rate_limit_returns_503(indexed_corpus, monkeypatch):
    """vector_rag /chat: LLMError(retryable=True, RateLimitError) → HTTP 503."""
    client = _vr_client(ErrorLLM(_rate_limit_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 503


def test_vr_route_auth_returns_500(indexed_corpus, monkeypatch):
    """vector_rag /chat: LLMError(retryable=False, auth) → HTTP 500."""
    client = _vr_client(ErrorLLM(_auth_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 500


def test_vr_route_api_error_returns_500(indexed_corpus, monkeypatch):
    """vector_rag /chat: LLMError(retryable=False) → HTTP 500."""
    client = _vr_client(ErrorLLM(_api_error()), monkeypatch)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 500
