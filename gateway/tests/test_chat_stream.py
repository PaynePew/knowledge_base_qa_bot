"""Gateway endpoint tests for POST /chat/stream?stack=wiki.

Phase 9 Slice 1 — AC: "The SSE serializer is a pure function covered by
hermetic unit tests; the event sequence is covered by an endpoint test with
a mocked LLM (no OPENAI_API_KEY)."

Tests assert the SSE event sequence:
  sources (non-empty, Wiki source shape) → token(s) → done{passed, reason}

The LLM and grounding verifier are mocked; no OPENAI_API_KEY is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Import markdown_kb modules via their workspace namespace.
import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _logger
import markdown_kb.app.retrieval as _retrieval
import pytest
from fastapi.testclient import TestClient
from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult
from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# Fake LLM stub (mirrors markdown_kb/tests/conftest.py FakeLLMResponse)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    """Minimal LLM stub for gateway tests."""

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
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect INDEX_PATH, LOG_PATH, WIKI_DIR to tmp for all gateway tests."""
    monkeypatch.setattr(_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")


@pytest.fixture()
def indexed_wiki_corpus(tmp_path, monkeypatch):
    """Build the Section Index from REAL_DOCS into the tmp paths."""
    _indexer.build_index(REAL_DOCS)
    yield
    _indexer.sections.clear()


def _approved_outcome() -> GroundingOutcome:
    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims trace to the cited section.",
            claims=[
                GroundingClaim(
                    text="Approved refunds are processed within 5-7 business days.",
                    supported=True,
                    citing_section_ids=["refund_policy.md#refund-timeline"],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


@pytest.fixture()
def gateway_client(indexed_wiki_corpus, monkeypatch):
    """TestClient for the Gateway app with a mocked LLM."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
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
# Happy-path endpoint tests
# ---------------------------------------------------------------------------


def test_chat_stream_wiki_event_order(gateway_client):
    """POST /chat/stream?stack=wiki emits: sources, then token(s), then done."""
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    assert types[0] == "sources", f"Expected first event to be 'sources', got {types[0]!r}"
    assert types[-1] == "done", f"Expected last event to be 'done', got {types[-1]!r}"
    assert all(t == "token" for t in types[1:-1]), f"Middle events should all be 'token': {types}"


def test_chat_stream_sources_event_non_empty(gateway_client):
    """sources event carries at least one source with required fields."""
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    sources_event = events[0]
    assert sources_event["type"] == "sources"
    sources = sources_event["data"]["sources"]
    assert len(sources) >= 1, "Expected at least one source"
    item = sources[0]
    # Wiki source shape: citation id, heading, content snippet, derived_from.
    assert "source" in item, "source field missing"
    assert "heading" in item, "heading field missing"
    assert "content" in item, "content field missing"
    assert "derived_from" in item, "derived_from field missing"


def test_chat_stream_sources_emitted_before_answer_tokens(gateway_client):
    """sources event index < first token event index (sources-first invariant)."""
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]
    sources_idx = types.index("sources")
    token_idxs = [i for i, t in enumerate(types) if t == "token"]
    assert token_idxs, "Expected at least one token event"
    assert sources_idx < token_idxs[0], "sources event must precede first token event"


def test_chat_stream_token_events_form_verified_answer(gateway_client):
    """Joining all token texts reconstructs the LLM-generated (verified) answer."""
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    tokens = [e["data"]["text"] for e in events if e["type"] == "token"]
    answer = " ".join(tokens)
    assert "5-7 business days" in answer


def test_chat_stream_done_event_passed_true(gateway_client):
    """done event carries passed=True for a grounded query."""
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    assert done["data"]["passed"] is True
    assert done["data"]["reason"] == "claim_supported"


def test_chat_stream_debug_page_loads(gateway_client):
    """GET / returns the debug HTML page."""
    resp = gateway_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Minimal content checks — textContent usage (§12.4), fetch (§12.2).
    assert "fetch(" in resp.text
    assert "textContent" in resp.text
    assert "EventSource" not in resp.text, "Must not use EventSource (GET-only; §12.2)"


# ---------------------------------------------------------------------------
# Cannot Confirm streaming
# ---------------------------------------------------------------------------


def test_chat_stream_cannot_confirm_emits_tokens(gateway_client):
    """Cannot Confirm answer streams as token events (uniform representation).

    Sends a query that the corpus cannot answer (score below threshold).
    The done event must carry passed=False.
    """
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "xyz completely unrelated gibberish banana orbit"},
    )
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]
    # Still emits sources + token(s) + done, not a special event type.
    assert types[0] == "sources"
    assert types[-1] == "done"
    done_data = events[-1]["data"]
    assert done_data["passed"] is False


# ---------------------------------------------------------------------------
# stack= parameter validation
# ---------------------------------------------------------------------------


def test_chat_stream_unknown_stack_returns_400(gateway_client):
    """Unknown stack value returns HTTP 400."""
    resp = gateway_client.post(
        "/chat/stream?stack=unknown",
        json={"query": "test"},
    )
    assert resp.status_code == 400


def test_chat_stream_rag_stack_returns_501(gateway_client):
    """stack=rag returns HTTP 501 (not yet implemented in this slice)."""
    resp = gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "test"},
    )
    assert resp.status_code == 501
