"""Gateway SSE tests for transport-agnostic LLM-error contract (ADR-0015).

Covers:
  AC-sse: Gateway SSE generator catches LLMError (not HTTPException) and emits
          terminal error{detail, retryable} event.

No live OpenAI calls in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _logger
import markdown_kb.app.retrieval as _retrieval
import pytest
from fastapi.testclient import TestClient
from markdown_kb.app.errors import LLMError

_FIXTURE_DOCS = Path(__file__).resolve().parents[2] / "markdown_kb" / "tests" / "fixtures" / "docs"


# ---------------------------------------------------------------------------
# Fixtures (mirroring test_chat_stream.py autouse + indexed_wiki_corpus pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect INDEX_PATH, LOG_PATH, WIKI_DIR to tmp for all tests in this file."""
    monkeypatch.setattr(_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")


@pytest.fixture()
def indexed_wiki_corpus(monkeypatch):
    """Build the Section Index from the 3-Source hermetic fixture."""
    _indexer.build_index(_FIXTURE_DOCS)
    yield
    _indexer.sections.clear()


# ---------------------------------------------------------------------------
# SSE event parsing helper
# ---------------------------------------------------------------------------


def _parse_sse_events(content: str) -> list[dict]:
    """Parse raw SSE text into a list of {type, data} dicts."""
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
# AC-sse: Gateway catches LLMError → terminal error{detail, retryable} event
# ---------------------------------------------------------------------------


def test_sse_llmerror_retryable_true_emits_error_event(indexed_wiki_corpus, monkeypatch):
    """Gateway SSE: LLMError(retryable=True) → terminal error{retryable:true} event."""

    def _raise_retryable(question, prompt_text):  # noqa: ANN001
        raise LLMError(
            retryable=True,
            message="LLM service temporarily unavailable, please retry.",
        )

    monkeypatch.setattr(_retrieval, "_call_llm_with_error_handling", _raise_retryable)

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/chat/stream", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, "SSE always returns 200 (HTTP already committed)"
    events = _parse_sse_events(resp.text)
    error_events = [e for e in events if e["type"] == "error"]

    assert error_events, f"Expected at least one error event; got: {[e['type'] for e in events]}"
    err = error_events[0]["data"]
    assert err.get("retryable") is True, f"Expected retryable=true; got: {err}"
    assert "detail" in err, f"Expected 'detail' key; got: {err}"

    # After an error event, no 'done' event must follow (ADR-0015 / ADR-0009).
    done_events = [e for e in events if e["type"] == "done"]
    assert not done_events, f"Expected no done event after error; got: {done_events}"


def test_sse_llmerror_retryable_false_emits_error_event(indexed_wiki_corpus, monkeypatch):
    """Gateway SSE: LLMError(retryable=False) → terminal error{retryable:false} event."""

    def _raise_not_retryable(question, prompt_text):  # noqa: ANN001
        raise LLMError(
            retryable=False,
            message="LLM service auth failed (check OPENAI_API_KEY).",
        )

    monkeypatch.setattr(_retrieval, "_call_llm_with_error_handling", _raise_not_retryable)

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/chat/stream", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    error_events = [e for e in events if e["type"] == "error"]

    assert error_events, f"Expected at least one error event; got: {[e['type'] for e in events]}"
    err = error_events[0]["data"]
    assert err.get("retryable") is False, f"Expected retryable=false; got: {err}"
    assert "detail" in err, f"Expected 'detail' key; got: {err}"

    done_events = [e for e in events if e["type"] == "done"]
    assert not done_events, f"Expected no done event after error; got: {done_events}"


def test_sse_llmerror_detail_carried_through(indexed_wiki_corpus, monkeypatch):
    """Gateway SSE: LLMError.message is carried through to error event detail."""
    expected_message = "LLM service temporarily unavailable, please retry."

    def _raise_with_message(question, prompt_text):  # noqa: ANN001
        raise LLMError(retryable=True, message=expected_message)

    monkeypatch.setattr(_retrieval, "_call_llm_with_error_handling", _raise_with_message)

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/chat/stream", json={"query": "How long do refunds take?"})

    events = _parse_sse_events(resp.text)
    error_events = [e for e in events if e["type"] == "error"]

    assert error_events
    err = error_events[0]["data"]
    assert err.get("detail") == expected_message, (
        f"Expected detail={expected_message!r}; got: {err.get('detail')!r}"
    )
