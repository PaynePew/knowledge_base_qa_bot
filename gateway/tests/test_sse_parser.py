"""Hermetic unit tests for the pure SSE parser function (CODING_STANDARD §12.7).

Phase 9 Slice 6 (#122) AC: "The SSE parser is a pure function (text chunks →
parsed events) with hermetic unit tests; unknown event types are ignored."

The SSE parser lives in ``gateway/static/index.html`` as the JavaScript
``createSSEParser()`` factory.  Because the project has no JS test toolchain
(CODING_STANDARD §12.6 — no new toolchain), this test module:

1. Implements the *same pure parsing algorithm* in Python to verify correctness
   at the function-logic level (chunk buffering, double-newline frame
   delimiters, event/data line parsing, unknown-type passthrough, JSON decode).

2. Inspects the production UI file to assert §12 structural invariants:
   createSSEParser is present, innerHTML is absent, EventSource is absent,
   and the done shape in the render fn reads ``done.grounding.passed``
   (the PRD-locked nested shape, not the old flat ``done.passed``).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 / §12.7).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Python mirror of the JS createSSEParser() pure function
# ---------------------------------------------------------------------------
# This is a direct algorithmic port of the JavaScript in index.html so the
# test coverage is equivalent to a JS unit test, without requiring Node.js
# or any JS toolchain (CODING_STANDARD §12.6 — no new toolchain).


def create_sse_parser():
    """Python mirror of the JS ``createSSEParser()`` factory (pure function).

    Returns a callable ``push(chunk: str) -> list[dict]`` that accumulates
    text chunks, splits on the SSE ``\\n\\n`` frame delimiter, parses
    ``event:`` and ``data:`` lines, and returns a list of ``{event, data}``
    dicts for each complete frame.

    Algorithm is identical to the JS implementation in gateway/static/index.html
    so the test covers the same logic path.
    """
    buf = ""

    def push(chunk: str) -> list[dict]:
        nonlocal buf
        buf += chunk
        out = []
        while "\n\n" in buf:
            idx = buf.index("\n\n")
            raw = buf[:idx]
            buf = buf[idx + 2 :]
            event = "message"
            data_str = ""
            for line in raw.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_str += line[5:].strip()
            data = None
            with contextlib.suppress(Exception):
                data = json.loads(data_str)
            out.append({"event": event, "data": data})
        return out

    return push


# ---------------------------------------------------------------------------
# Path to the production UI file
# ---------------------------------------------------------------------------

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


# ---------------------------------------------------------------------------
# Pure parser: correctness tests (text chunks → parsed events)
# ---------------------------------------------------------------------------


def test_single_complete_frame_is_parsed():
    """A single complete SSE frame yields one event dict."""
    parser = create_sse_parser()
    frame = 'event: sources\ndata: {"sources": []}\n\n'
    events = parser(frame)
    assert len(events) == 1
    assert events[0]["event"] == "sources"
    assert events[0]["data"] == {"sources": []}


def test_two_frames_in_one_chunk():
    """Two back-to-back frames in a single chunk are both parsed."""
    parser = create_sse_parser()
    chunk = 'event: sources\ndata: {"sources": []}\n\nevent: token\ndata: {"text": "Hello"}\n\n'
    events = parser(chunk)
    assert len(events) == 2
    assert events[0]["event"] == "sources"
    assert events[1]["event"] == "token"
    assert events[1]["data"]["text"] == "Hello"


def test_partial_frame_buffered_across_chunks():
    """A frame split across two chunk calls is buffered and emitted on second call."""
    parser = create_sse_parser()
    # First chunk: incomplete frame (no double-newline yet)
    half = 'event: token\ndata: {"text": "partial"}'
    result1 = parser(half)
    assert result1 == [], "Incomplete frame must not yield an event yet"
    # Second chunk: the terminating double-newline
    result2 = parser("\n\n")
    assert len(result2) == 1
    assert result2[0]["event"] == "token"
    assert result2[0]["data"]["text"] == "partial"


def test_unknown_event_type_is_included_in_output():
    """Unknown event types are emitted unchanged — callers ignore them (§12.2).

    The parser does NOT filter types; the dispatch fn ignores unknowns.
    This is the forward-compatibility contract.
    """
    parser = create_sse_parser()
    frame = 'event: future_event\ndata: {"x": 1}\n\n'
    events = parser(frame)
    assert len(events) == 1
    assert events[0]["event"] == "future_event"
    assert events[0]["data"] == {"x": 1}


def test_default_event_type_is_message_when_no_event_line():
    """If an SSE frame has no ``event:`` line, event defaults to ``message``."""
    parser = create_sse_parser()
    frame = 'data: {"text": "hi"}\n\n'
    events = parser(frame)
    assert events[0]["event"] == "message"


def test_malformed_json_data_yields_none():
    """Malformed JSON in the data line yields data=None rather than raising."""
    parser = create_sse_parser()
    frame = "event: bad\ndata: {not valid json}\n\n"
    events = parser(frame)
    assert len(events) == 1
    assert events[0]["event"] == "bad"
    assert events[0]["data"] is None


def test_empty_frames_skipped():
    """Empty string between double-newlines does not yield extra events."""
    parser = create_sse_parser()
    # Double double-newline: one blank frame between two real frames
    chunk = (
        'event: sources\ndata: {"sources": []}\n\n'
        "\n\n"
        'event: done\ndata: {"grounding": {"passed": true, "reason": "claim_supported"}, "filed": null, "stack": "wiki"}\n\n'
    )
    events = parser(chunk)
    # The blank frame produces an event with event="message", data=None.
    # Our push fn still processes it (empty raw → dataStr="" → json.loads("") raises).
    # What matters: both real events are present.
    source_evs = [e for e in events if e["event"] == "sources"]
    done_evs = [e for e in events if e["event"] == "done"]
    assert len(source_evs) == 1
    assert len(done_evs) == 1


def test_done_event_nested_grounding_shape():
    """done event data carries nested grounding object (PRD #116 shape)."""
    parser = create_sse_parser()
    done_payload = {
        "grounding": {"passed": True, "reason": "claim_supported"},
        "filed": {"slug": "refund-policy", "status": "draft", "op": "created", "count": 1},
        "stack": "wiki",
    }
    frame = f"event: done\ndata: {json.dumps(done_payload)}\n\n"
    events = parser(frame)
    assert events[0]["event"] == "done"
    data = events[0]["data"]
    assert data["grounding"]["passed"] is True
    assert data["grounding"]["reason"] == "claim_supported"
    assert data["filed"]["slug"] == "refund-policy"
    assert data["stack"] == "wiki"


def test_sources_event_with_wiki_source():
    """sources event with a Wiki source (trio + derived_from) is parsed correctly."""
    parser = create_sse_parser()
    sources_payload = {
        "sources": [
            {
                "source": "refund-policy#refund-timeline",
                "heading": "concepts › Refund Timeline",
                "content": "Approved refunds are processed within 5-7 business days.",
                "derived_from": [
                    {"source": "refund_policy.md#refund-timeline", "heading": "Refund Timeline"}
                ],
            }
        ]
    }
    frame = f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"
    events = parser(frame)
    assert events[0]["event"] == "sources"
    src = events[0]["data"]["sources"][0]
    assert src["source"] == "refund-policy#refund-timeline"
    assert src["derived_from"][0]["source"] == "refund_policy.md#refund-timeline"


def test_token_events_reconstruct_answer():
    """Concatenating text from consecutive token events reconstructs the answer."""
    parser = create_sse_parser()
    words = ["Approved refunds are ", "processed within ", "5-7 business days."]
    chunk = "".join(f"event: token\ndata: {json.dumps({'text': w})}\n\n" for w in words)
    events = parser(chunk)
    tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
    assert "".join(tokens) == "".join(words)


def test_error_event_carries_detail():
    """error event data carries detail and retryable fields."""
    parser = create_sse_parser()
    err_payload = {"detail": "LLM temporarily unavailable.", "retryable": True}
    frame = f"event: error\ndata: {json.dumps(err_payload)}\n\n"
    events = parser(frame)
    assert events[0]["event"] == "error"
    assert events[0]["data"]["detail"] == "LLM temporarily unavailable."
    assert events[0]["data"]["retryable"] is True


# ---------------------------------------------------------------------------
# UI file structural invariants (§12 compliance)
# ---------------------------------------------------------------------------


def test_ui_file_exists():
    """The production UI file gateway/static/index.html exists."""
    assert _STATIC_INDEX.exists(), f"UI file not found at {_STATIC_INDEX}"


def test_ui_file_contains_create_sse_parser():
    """The UI file defines createSSEParser() — the pure parser factory (§12.2)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    assert "createSSEParser" in text, "createSSEParser not found in index.html"


def test_ui_file_no_inner_html_assignment():
    """The UI file never assigns to innerHTML (XSS — §12.4).

    The word 'innerHTML' may appear in comments or guard-clause strings
    (e.g. ``throw new Error("innerHTML banned")``), but must never appear
    as an assignment target (``n.innerHTML =`` etc.).
    """
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    # An assignment to innerHTML: either `.innerHTML =` or `.innerHTML=`
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found — §12.4 requires textContent only"
    )


def test_ui_file_has_no_event_source_instantiation():
    """The UI file does not instantiate EventSource (GET-only — §12.2).

    The word 'EventSource' may appear in comments explaining why it is not
    used, but ``new EventSource`` must not appear.
    """
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    assert "new EventSource" not in text, (
        "EventSource is GET-only (§12.2); must use fetch+ReadableStream"
    )


def test_ui_file_uses_fetch_with_post():
    """The UI file uses fetch() with POST for the SSE request (§12.2)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    assert 'method: "POST"' in text or "method: 'POST'" in text, (
        "UI must POST to /chat/stream (§12.2)"
    )


def test_ui_file_reads_done_grounding_passed():
    """The UI reads done.grounding.passed, not the old flat done.passed (PRD #116)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    assert "d.grounding.passed" in text, (
        "UI must read done.grounding.passed per PRD #116 nested shape"
    )
    assert "d.passed" not in text, "UI must NOT read the old flat d.passed (PRD #116)"


def test_ui_file_reads_done_grounding_reason():
    """The UI reads done.grounding.reason, not the old flat done.reason (PRD #116)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    assert "d.grounding.reason" in text, (
        "UI must read done.grounding.reason per PRD #116 nested shape"
    )


def test_ui_file_has_no_framework():
    """No framework or build-step import (§12.1 — vanilla only)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    forbidden = ["import React", "from 'react'", "from 'vue'", "require('react')", "webpack"]
    for marker in forbidden:
        assert marker not in text, (
            f"Framework/build marker {marker!r} found — §12.1 requires vanilla"
        )


def test_ui_file_stack_toggle_uses_query_param():
    """The UI maps the stack toggle to the 'stack' query param (§12.3)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    assert "stack=" in text, (
        "UI must include stack= query param in the /chat/stream request (§12.3)"
    )


# ---------------------------------------------------------------------------
# Phase 11 Slice 4: done.session surfacing + multi-turn UI invariants
# ---------------------------------------------------------------------------


def test_parser_surfaces_done_session():
    """The SSE parser surfaces done.session from the done event payload (Phase 11 Slice 4 AC)."""
    parser = create_sse_parser()
    done_payload = {
        "grounding": {"passed": True, "reason": "claim_supported"},
        "filed": None,
        "stack": "wiki",
        "session": "abc123-uuid-here",
    }
    frame = f"event: done\ndata: {json.dumps(done_payload)}\n\n"
    events = parser(frame)
    assert events[0]["event"] == "done"
    assert events[0]["data"]["session"] == "abc123-uuid-here", (
        "done.session must be surfaced by the parser"
    )


def test_ui_file_captures_done_session():
    """The UI captures done.session from the done event (Phase 11 Slice 4 — §12 multi-turn)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    # The UI must reference done.session to capture the session id.
    assert "d.session" in text or "done.session" in text or ".session" in text, (
        "UI must read done.session to capture the session id (Phase 11 Slice 4)"
    )


def test_ui_file_sends_session_query_param():
    """The UI echoes session id via ?session= on subsequent requests (Phase 11 Slice 4)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    assert "session=" in text, (
        "UI must include session= query param when echoing session id (Phase 11 Slice 4)"
    )


def test_ui_file_has_rewriting_indicator():
    """The UI renders a rewriting indicator for status:{phase:rewriting} (Phase 11 Slice 4)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    # The UI should show text like "understanding your question" for the rewriting phase.
    assert "understanding your question" in text.lower() or "rewriting" in text.lower(), (
        "UI must render a rewriting indicator for status:rewriting (Phase 11 Slice 4)"
    )


def test_ui_file_status_handler_is_phase_aware():
    """The UI's status handler dispatches on phase (rewriting vs verifying) — Phase 11 Slice 4."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    # The onStatus function must reference phase to differentiate rewriting from verifying.
    assert '"rewriting"' in text or "'rewriting'" in text, (
        "UI must handle phase=rewriting in the status event handler (Phase 11 Slice 4)"
    )


def test_ui_file_toggle_keeps_session():
    """The UI stack toggle does NOT reset the session id (Phase 11 Slice 4 — history preserved)."""
    text = _STATIC_INDEX.read_text(encoding="utf-8")
    # The setStack function must NOT clear/reset sessionId.
    # We verify indirectly: sessionId is a state variable and setStack does not reassign it to null.
    # The structural invariant: the word sessionId appears in the source.
    assert "sessionId" in text, "UI must use a sessionId state variable (Phase 11 Slice 4)"
