"""Integration tests for error handling (issue #5, updated for Slice 4 / issue #12).

Covers:
  AC 1: APITimeoutError / RateLimitError → HTTP 503, log kind=openai_transient
  AC 2: AuthenticationError → HTTP 500, log kind=openai_auth
  AC 3: generic APIError → HTTP 500, log kind=openai_api
  AC 4: grounding.verify() returns claim_unsupported → answer replaced with the
        Cannot Confirm phrase; main LLM invoked exactly once (the old
        light-heuristic temperature=0 retry is gone — ADR-0004 layer 3).
  AC 5: grounding.verify() returns claim_supported → draft is returned unchanged;
        main LLM invoked exactly once.

All tests run without live OpenAI calls.

Run with:
    pytest -m "not live"   (from markdown_kb/)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import app.retrieval as retrieval_module
from app.grounding import GroundingOutcome
from app.retrieval import CANNOT_CONFIRM_PHRASE

from .conftest import FakeLLMResponse

REFUND_SECTION_ID = "refund-timeline#refund-timeline"

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
        return FakeLLMResponse(content=self._responses[idx])


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

    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "detail" in body, f"Expected 'detail' in error response, got: {body}"
    assert "temporarily unavailable" in body["detail"].lower(), (
        f"Expected 'temporarily unavailable' in detail, got: {body['detail']!r}"
    )
    assert "retry" in body["detail"].lower(), f"Expected 'retry' in detail, got: {body['detail']!r}"


def test_api_timeout_logs_openai_transient(indexed_corpus, monkeypatch):
    """APITimeoutError → wiki/log.md contains chat_error | … kind=openai_transient."""
    log_path: Path = indexed_corpus["log_path"]
    fake_llm = ErrorLLM(_make_api_timeout_error())
    client = _make_client(fake_llm, monkeypatch)

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")
    assert "chat_error |" in content, f"Expected 'chat_error |' in log, got:\n{content}"
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

    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"
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
    assert "chat_error |" in content, f"Expected 'chat_error |' in log, got:\n{content}"
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

    assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "detail" in body, f"Expected 'detail' in error response, got: {body}"
    assert "auth" in body["detail"].lower(), f"Expected 'auth' in detail, got: {body['detail']!r}"
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
    assert "chat_error |" in content, f"Expected 'chat_error |' in log, got:\n{content}"
    assert "kind=openai_auth" in content, f"Expected 'kind=openai_auth' in log, got:\n{content}"


# ---------------------------------------------------------------------------
# AC 3: generic APIError → HTTP 500 + openai_api log
# ---------------------------------------------------------------------------


def test_generic_api_error_returns_500(indexed_corpus, monkeypatch):
    """Generic APIError → HTTP 500 with error detail."""
    fake_llm = ErrorLLM(_make_api_error())
    client = _make_client(fake_llm, monkeypatch)

    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text}"
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
    assert "chat_error |" in content, f"Expected 'chat_error |' in log, got:\n{content}"
    assert "kind=openai_api" in content, f"Expected 'kind=openai_api' in log, got:\n{content}"


# ---------------------------------------------------------------------------
# AC 4: grounding.verify() returns claim_unsupported → Cannot Confirm;
#        LLM invoked exactly once (grounding.verify() replaces the old retry heuristic)
# ---------------------------------------------------------------------------


def test_verifier_rejects_answer_replaced_with_cannot_confirm(indexed_corpus, monkeypatch):
    """When grounding.verify() returns claim_unsupported → answer becomes Cannot Confirm.

    The old light heuristic used a temperature=0 retry LLM; the new ADR-0004 layer 3
    grounding check uses grounding.verify() with a separate structured-output call.
    """
    fake_llm = CountingLLM(["Refunds take 3 days. Also, we offer free shipping worldwide."])
    client = _make_client(fake_llm, monkeypatch)

    rejected_outcome = GroundingOutcome(
        passed=False,
        reason="claim_unsupported",
        result=None,
        retries_attempted=0,
    )
    with patch("app.retrieval.grounding_module.verify", return_value=rejected_outcome):
        resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["answer"] == CANNOT_CONFIRM_PHRASE, (
        f"Expected exact Cannot Confirm phrase, got: {body['answer']!r}"
    )
    assert body["grounding"]["reason"] == "claim_unsupported", (
        f"Expected reason=claim_unsupported, got: {body['grounding']['reason']!r}"
    )


def test_verifier_rejects_answer_invokes_llm_once(indexed_corpus, monkeypatch):
    """When grounding.verify() rejects → main LLM invoked exactly once (no retry LLM)."""
    fake_llm = CountingLLM(["Some claim without proof."])
    client = _make_client(fake_llm, monkeypatch)

    rejected_outcome = GroundingOutcome(
        passed=False,
        reason="claim_unsupported",
        result=None,
        retries_attempted=0,
    )
    with patch("app.retrieval.grounding_module.verify", return_value=rejected_outcome):
        client.post("/chat", json={"query": "How long do refunds take?"})

    assert fake_llm.call_count == 1, (
        f"Expected LLM to be invoked exactly 1 time, got: {fake_llm.call_count}"
    )


# ---------------------------------------------------------------------------
# AC 5: grounding.verify() returns claim_supported → draft returned as-is;
#        LLM invoked exactly once
# ---------------------------------------------------------------------------


def test_verifier_approves_answer_returned_as_is(indexed_corpus, monkeypatch):
    """When grounding.verify() returns claim_supported → draft is returned unchanged."""
    canned_answer = "Refunds take 5-7 business days. [Source: refund-timeline#refund-timeline]"
    fake_llm = CountingLLM([canned_answer])
    client = _make_client(fake_llm, monkeypatch)

    approved_outcome = GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=None,
        retries_attempted=0,
    )
    with patch("app.retrieval.grounding_module.verify", return_value=approved_outcome):
        resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["answer"] == canned_answer, (
        f"Expected draft passed through, got: {body['answer']!r}"
    )
    assert body["grounding"]["passed"] is True


def test_verifier_approves_answer_invokes_llm_once(indexed_corpus, monkeypatch):
    """When grounding.verify() approves → main LLM invoked exactly once."""
    canned_answer = "Refunds take 5-7 business days. [Source: refund-timeline#refund-timeline]"
    fake_llm = CountingLLM([canned_answer])
    client = _make_client(fake_llm, monkeypatch)

    approved_outcome = GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=None,
        retries_attempted=0,
    )
    with patch("app.retrieval.grounding_module.verify", return_value=approved_outcome):
        client.post("/chat", json={"query": "How long do refunds take?"})

    assert fake_llm.call_count == 1, (
        f"Expected LLM to be invoked exactly 1 time, got: {fake_llm.call_count}"
    )
