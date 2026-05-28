"""Gateway endpoint tests for POST /chat/stream?stack=wiki.

Phase 9 Slice 1 — AC: "The SSE serializer is a pure function covered by
hermetic unit tests; the event sequence is covered by an endpoint test with
a mocked LLM (no OPENAI_API_KEY)."

Phase 9 Slice 2 — AC: status event (between sources and first token), terminal
error event (post-sources LLM failure, no done follows), and uniform Cannot
Confirm for all five reasons (sources → token(phrase) → done{passed:false}).

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
from fastapi import HTTPException
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
    """POST /chat/stream?stack=wiki emits: sources → status → token(s) → done.

    Phase 9 Slice 2 adds a ``status`` event between sources and the first token
    (liveness signal during the draft+verify gap). The event order is:
      sources, status, token(s)..., done.
    """
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
    # Middle events: one status followed by token(s).
    middle = types[1:-1]
    assert middle[0] == "status", f"First middle event must be 'status': {types}"
    assert all(t == "token" for t in middle[1:]), (
        f"All events after status must be 'token': {types}"
    )


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
    answer = "".join(tokens)
    assert "5-7 business days" in answer


def test_chat_stream_done_event_passed_true(gateway_client):
    """done event carries nested grounding.passed=True for a grounded query (PRD #116)."""
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    # PRD-locked shape: done.grounding.{passed,reason}
    assert done["data"]["grounding"]["passed"] is True
    assert done["data"]["grounding"]["reason"] == "claim_supported"
    # stack is populated by the Gateway dispatcher
    assert done["data"]["stack"] == "wiki"


def test_ui_page_loads(gateway_client):
    """GET / returns the Gateway UI (production UI — Slice 6 #122).

    Asserts §12 invariants: fetch+ReadableStream (§12.2), textContent-only
    rendering (§12.4), no EventSource (GET-only — §12.2).
    """
    resp = gateway_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # §12.2: must use fetch(), not EventSource instantiation.
    # 'EventSource' may appear in comments (explaining why it is NOT used)
    # but `new EventSource` must not appear.
    assert "fetch(" in resp.text
    assert "new EventSource" not in resp.text, "Must not instantiate EventSource (GET-only; §12.2)"
    # §12.4: LLM/server content via textContent (never innerHTML assignment)
    # 'innerHTML' may appear in comments/guard strings but must never be an assignment target.
    assert "textContent" in resp.text
    assert ".innerHTML =" not in resp.text and ".innerHTML=" not in resp.text, (
        "innerHTML assignment is banned per §12.4"
    )
    # §12.2: SSE parser function present (the unit-testable pure fn)
    assert "createSSEParser" in resp.text


# ---------------------------------------------------------------------------
# Cannot Confirm streaming
# ---------------------------------------------------------------------------


def test_chat_stream_cannot_confirm_emits_tokens(gateway_client):
    """Cannot Confirm answer streams as token events (uniform representation).

    Sends a query that the corpus cannot answer (score below threshold).
    The done event must carry grounding.passed=False (PRD #116 shape).
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
    # PRD-locked shape: done.grounding.passed
    assert done_data["grounding"]["passed"] is False


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


def test_chat_stream_rag_stack_no_longer_returns_501(gateway_client):
    """stack=rag is now dispatched (Phase 9 Slice 3); 501 is replaced by real dispatch.

    This test confirms the gateway no longer rejects stack=rag with 501.
    The RAG dispatch uses the vector_rag stack; since gateway_client only
    indexes the wiki corpus (not the FAISS index), RAG returns a grounded
    fallback (200 with done.passed=False) or a real answer — either way, not 501.
    """
    resp = gateway_client.post(
        "/chat/stream?stack=rag",
        json={"query": "test"},
    )
    assert resp.status_code != 501, "stack=rag must not return 501 after Phase 9 Slice 3"


# ---------------------------------------------------------------------------
# Phase 9 Slice 2: status event (AC #1)
# ---------------------------------------------------------------------------


def test_chat_stream_emits_status_event_between_sources_and_tokens(gateway_client):
    """A status event is emitted between sources and the first token event.

    ADR-0009: the draft+verify gap (4-7s) must emit a liveness signal so the
    user knows the system has not stalled. The status event carries
    ``{phase: "verifying"}`` or similar — tests assert position only.
    """
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    assert "status" in types, f"Expected a 'status' event in {types}"
    status_idx = types.index("status")
    sources_idx = types.index("sources")
    first_token_idx = next((i for i, t in enumerate(types) if t == "token"), None)

    assert sources_idx < status_idx, "sources must precede status"
    assert first_token_idx is not None, "Expected at least one token event"
    assert status_idx < first_token_idx, "status must precede first token"


def test_chat_stream_status_event_has_phase_field(gateway_client):
    """The status event payload carries a ``phase`` field (liveness descriptor)."""
    resp = gateway_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    status_events = [e for e in events if e["type"] == "status"]
    assert status_events, "Expected at least one status event"
    assert "phase" in status_events[0]["data"], (
        f"status event data must contain 'phase': {status_events[0]['data']}"
    )


# ---------------------------------------------------------------------------
# Phase 9 Slice 2: terminal error event after sources (AC #2)
# ---------------------------------------------------------------------------


@pytest.fixture()
def gateway_client_llm_error(indexed_wiki_corpus, monkeypatch):
    """TestClient where the LLM raises a 503 HTTPException (post-sources error).

    Simulates an LLM/infra failure that occurs after the sources event has
    already been emitted (HTTP 200 committed). The stream must terminate with
    a terminal ``error`` event instead of a ``done`` event.
    """

    def _raise_503(*_args, **_kwargs):
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable, please retry.",
        )

    fake_llm_error = type("_ErrLLM", (), {"invoke": staticmethod(_raise_503)})()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm_error)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm_error)

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


def test_post_sources_error_emits_terminal_error_event(
    gateway_client_llm_error,
):
    """An LLM error after sources emits a terminal SSE ``error`` event.

    HTTP 200 has already been committed (sources event sent), so the failure
    cannot be expressed as an HTTP status code.  The stream must end with an
    SSE ``error`` event carrying ``{detail, retryable}``.  No ``done`` event
    follows (ADR-0009).
    """
    resp = gateway_client_llm_error.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200, "HTTP 200 is committed before sources"
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    # Must start with sources
    assert types[0] == "sources", f"Expected sources first, got {types[0]!r}"
    # Must end with error
    assert types[-1] == "error", f"Expected terminal 'error' event, got {types[-1]!r}"
    # No done event after error
    assert "done" not in types, f"'done' must not follow an error event: {types}"


def test_post_sources_error_event_has_detail_and_retryable(
    gateway_client_llm_error,
):
    """The terminal error event payload carries ``detail`` and ``retryable`` fields."""
    resp = gateway_client_llm_error.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    error_event = next((e for e in events if e["type"] == "error"), None)
    assert error_event is not None, "Expected an error event"
    data = error_event["data"]
    assert "detail" in data, f"error event must have 'detail': {data}"
    assert "retryable" in data, f"error event must have 'retryable': {data}"


def test_post_sources_503_error_is_retryable(gateway_client_llm_error):
    """A 503 (transient) LLM error produces retryable=True in the error event."""
    resp = gateway_client_llm_error.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    error_event = next((e for e in events if e["type"] == "error"), None)
    assert error_event is not None
    assert error_event["data"]["retryable"] is True


# ---------------------------------------------------------------------------
# Phase 9 Slice 2: uniform Cannot Confirm for all 5 reasons (AC #3 + #4)
# ---------------------------------------------------------------------------


@pytest.fixture()
def gateway_client_cc(indexed_wiki_corpus, monkeypatch, request):
    """Parametrised fixture: returns a (TestClient, query_text) tuple primed for
    the requested Cannot Confirm reason.

    Uses request.param to receive the reason string from the parametrize marker.
    """
    reason = request.param

    # Import once at fixture setup time (markdown_kb is in sys.modules by then).
    from gateway.app.main import app as _gateway_app

    if reason == "index_missing":
        # Clear the section index so _retrieve_and_gate fires index_missing.
        _indexer.sections.clear()
        query_text = "any question"
    elif reason == "retrieval_empty":
        # Single stop word; BM25 returns [] for stop-word-only queries.
        query_text = "a"
    elif reason == "below_threshold":
        # Gibberish ensures all BM25 scores fall below _SCORE_THRESHOLD (0.5).
        query_text = "zzzzxxxxxqqqqqwwwwwvvvvv"
    elif reason == "claim_unsupported":
        fake_llm = _FakeLLM()
        monkeypatch.setattr(_retrieval, "_llm", fake_llm)
        monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
        monkeypatch.setattr(
            _retrieval.grounding_module,
            "verify",
            lambda draft, sections: GroundingOutcome(passed=False, reason="claim_unsupported"),
        )
        query_text = "What is the refund policy?"
    elif reason == "verifier_unavailable":
        fake_llm = _FakeLLM()
        monkeypatch.setattr(_retrieval, "_llm", fake_llm)
        monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
        monkeypatch.setattr(
            _retrieval.grounding_module,
            "verify",
            lambda draft, sections: GroundingOutcome(passed=False, reason="verifier_unavailable"),
        )
        query_text = "What is the refund policy?"
    else:
        raise ValueError(f"Unknown CC reason: {reason!r}")

    return TestClient(_gateway_app), query_text


@pytest.mark.parametrize(
    "gateway_client_cc",
    [
        "index_missing",
        "retrieval_empty",
        "below_threshold",
        "claim_unsupported",
        "verifier_unavailable",
    ],
    indirect=True,
)
def test_cannot_confirm_uniform_event_sequence(gateway_client_cc):
    """Each CC reason streams: sources → token(phrase) → done{passed:false, reason}.

    No special ``cannot_confirm`` event type is ever emitted (ADR-0009).
    No filing occurs on any CC path (no ``done.filed`` populated).
    """
    client, query_text = gateway_client_cc
    # Recover the reason from the fixture param via the client — the fixture
    # stores the reason in the query_text side-band, so we extract it from
    # the done event instead.
    resp = client.post("/chat/stream?stack=wiki", json={"query": query_text})
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]
    # done.grounding.reason carries the specific CC gate (PRD #116 shape)
    reason = (
        events[-1]["data"].get("grounding", {}).get("reason", "unknown") if events else "no events"
    )

    # Sequence: sources first, done last, tokens in between
    assert types[0] == "sources", f"[{reason}] expected sources first: {types}"
    assert types[-1] == "done", f"[{reason}] expected done last: {types}"
    # No special event type
    assert "cannot_confirm" not in types, (
        f"[{reason}] no cannot_confirm event type (ADR-0009): {types}"
    )
    # Token events reconstruct CANNOT_CONFIRM_PHRASE
    token_texts = [e["data"]["text"] for e in events if e["type"] == "token"]
    reconstructed = "".join(token_texts)
    assert reconstructed == CANNOT_CONFIRM_PHRASE, (
        f"[{reason}] tokens must spell CANNOT_CONFIRM_PHRASE; got: {reconstructed!r}"
    )
    # done carries grounding.passed=False (PRD-locked shape)
    done_data = events[-1]["data"]
    assert done_data["grounding"]["passed"] is False, (
        f"[{reason}] done.grounding.passed must be False"
    )
    # No filing on CC paths
    assert done_data.get("filed") is None, (
        f"[{reason}] done.filed must be None on CC paths: {done_data.get('filed')}"
    )
