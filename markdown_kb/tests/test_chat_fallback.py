"""Integration tests for the Cannot Confirm fallback paths — Slice 3.

Tests translate every acceptance criterion from issue #4 directly into
executable assertions. All tests use a sentinel / fake LLM stub (no live
OpenAI calls) and assert the LLM is NEVER invoked in fallback scenarios.

Run with:
    pytest -m "not live"   (from markdown_kb/)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer
import app.retrieval as retrieval_module
from app.grounding import GroundingOutcome
from app.retrieval import CANNOT_CONFIRM_PHRASE, NOT_INDEXED_MESSAGE

from .conftest import FakeLLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SentinelLLM:
    """Raises if invoke() is ever called — proves the LLM gate fired."""

    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        raise AssertionError("LLM must NOT be invoked when the pre-LLM Cannot Confirm gate fires.")


class CaptureLLM:
    """Records calls so tests can verify the LLM WAS reached when expected."""

    def __init__(self):
        self.call_count = 0
        self.last_messages: list = []

    def invoke(self, messages):
        self.call_count += 1
        self.last_messages = messages
        return FakeLLMResponse(content="Some canned answer from the LLM.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_corpus():
    """Ensure no sections are indexed — simulates a pre-index state.

    Path redirection is handled by conftest's autouse `_redirect_paths_to_tmp`.
    """
    import app.logger as _logger

    indexer.sections.clear()
    yield {"log_path": _logger.LOG_PATH}
    indexer.sections.clear()


# ---------------------------------------------------------------------------
# AC 1 & 2: out-of-scope query returns 200 with exact Cannot Confirm phrase
#            and sources == []
# ---------------------------------------------------------------------------


def test_out_of_scope_query_returns_cannot_confirm_exact_phrase(indexed_corpus, monkeypatch):
    """POST /chat with an out-of-scope query returns 200 and the exact Cannot
    Confirm phrase — no trailing punctuation, no apology, no explanation."""
    sentinel = SentinelLLM()
    monkeypatch.setattr(retrieval_module, "_llm", sentinel)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: sentinel)

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()

    # AC 1: exact phrase
    assert body["answer"] == CANNOT_CONFIRM_PHRASE, (
        f"Expected exact phrase '{CANNOT_CONFIRM_PHRASE}', got: {body['answer']!r}"
    )

    # AC 2: grounding.reason reflects the pre-LLM gate (below_threshold or retrieval_empty)
    assert "grounding" in body, "Response must have 'grounding' field"
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] in ("below_threshold", "retrieval_empty"), (
        f"Expected pre-LLM gate reason, got: {body['grounding']['reason']!r}"
    )


# ---------------------------------------------------------------------------
# AC 3: LLM is NEVER invoked for an out-of-scope query
# ---------------------------------------------------------------------------


def test_out_of_scope_query_does_not_invoke_llm(indexed_corpus, monkeypatch):
    """The mocked LLM's invoke method must never be called for an out-of-scope query."""
    sentinel = SentinelLLM()
    monkeypatch.setattr(retrieval_module, "_llm", sentinel)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: sentinel)

    from app.main import app

    client = TestClient(app)
    # If the sentinel's invoke() fires, it raises AssertionError and the test fails
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})
    assert resp.status_code == 200
    assert sentinel.call_count == 0, f"LLM was invoked {sentinel.call_count} time(s); expected 0."


# ---------------------------------------------------------------------------
# AC 4: wiki/log.md contains chat_fallback | … reason=below_threshold entry
# ---------------------------------------------------------------------------


def test_out_of_scope_query_logs_chat_fallback_below_threshold(indexed_corpus, monkeypatch):
    """wiki/log.md must contain a chat_fallback entry with reason=below_threshold
    and top_score=<score> for an out-of-scope query."""
    sentinel = SentinelLLM()
    monkeypatch.setattr(retrieval_module, "_llm", sentinel)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: sentinel)

    log_path: Path = indexed_corpus["log_path"]

    from app.main import app

    client = TestClient(app)
    client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")

    assert "chat_fallback |" in content, f"Expected 'chat_fallback |' in log, got:\n{content}"
    assert "reason=below_threshold" in content, (
        f"Expected 'reason=below_threshold' in log, got:\n{content}"
    )
    assert "top_score=" in content, f"Expected 'top_score=<score>' in log, got:\n{content}"


# ---------------------------------------------------------------------------
# AC 5: KB_SCORE_THRESHOLD=0.0 allows the same query to reach the LLM
# ---------------------------------------------------------------------------


def test_low_threshold_allows_out_of_scope_query_to_reach_llm(indexed_corpus, monkeypatch):
    """When KB_SCORE_THRESHOLD=0.0 is set, an out-of-scope query that would
    otherwise be gated DOES reach the (mocked) LLM — proving the env var works."""
    # Set threshold to 0.0 so ANY positive BM25 score passes
    monkeypatch.setattr(retrieval_module, "_SCORE_THRESHOLD", 0.0)

    capture = CaptureLLM()
    monkeypatch.setattr(retrieval_module, "_llm", capture)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: capture)

    # Mock grounding.verify() so the post-LLM check doesn't attempt a real API call.
    approved_outcome = GroundingOutcome(passed=True, reason="claim_supported")
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: approved_outcome,
    )

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200
    # With threshold=0.0 the LLM must have been invoked (primary call only — no retry LLM)
    assert capture.call_count == 1, (
        f"Expected LLM to be invoked exactly once when KB_SCORE_THRESHOLD=0.0, "
        f"but call_count={capture.call_count}"
    )


# ---------------------------------------------------------------------------
# AC 6a: POST /chat before any POST /index returns the not-indexed response
# ---------------------------------------------------------------------------


def test_pre_index_chat_returns_not_indexed_message(empty_corpus, monkeypatch):
    """POST /chat before any POST /index returns the 'not indexed yet' response."""
    sentinel = SentinelLLM()
    monkeypatch.setattr(retrieval_module, "_llm", sentinel)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: sentinel)

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()

    assert body["answer"] == NOT_INDEXED_MESSAGE, (
        f"Expected NOT_INDEXED_MESSAGE, got: {body['answer']!r}"
    )
    assert body["sources"] == [], f"Expected sources == [], got: {body['sources']}"

    # Grounding field: index_missing (AC #2 from issue #12)
    assert "grounding" in body, "Response must have 'grounding' field"
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] == "index_missing", (
        f"Expected grounding.reason=index_missing, got: {body['grounding']['reason']!r}"
    )


# ---------------------------------------------------------------------------
# AC 6b: pre-index call logs chat_fallback | … reason=not_indexed
# ---------------------------------------------------------------------------


def test_pre_index_chat_logs_chat_fallback_not_indexed(empty_corpus, monkeypatch):
    """POST /chat before indexing must write a chat_fallback entry with
    reason=not_indexed and must NOT call the LLM."""
    sentinel = SentinelLLM()
    monkeypatch.setattr(retrieval_module, "_llm", sentinel)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: sentinel)

    log_path: Path = empty_corpus["log_path"]

    from app.main import app

    client = TestClient(app)
    client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")

    assert "chat_fallback |" in content, f"Expected 'chat_fallback |' in log, got:\n{content}"
    assert "reason=not_indexed" in content, f"Expected 'reason=not_indexed' in log, got:\n{content}"
    # LLM must not be invoked
    assert sentinel.call_count == 0, (
        f"LLM was invoked {sentinel.call_count} time(s) before indexing; expected 0."
    )
