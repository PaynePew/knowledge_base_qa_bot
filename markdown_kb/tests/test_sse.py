"""Hermetic unit tests for the SSE serializer (app.sse).

Phase 9 Slice 1 — AC: "The SSE serializer is a pure function (query-result dict
→ ordered event list) covered by hermetic unit tests."

Tests do NOT import fastapi, retrieval, or any LLM module — pure logic only.
"""

from __future__ import annotations

import json

import pytest

from app.sse import encode_event, events_for_result

# ---------------------------------------------------------------------------
# Minimal stubs for GroundingOutcome (avoids importing from grounding.py
# and thus importing langchain_openai at test collection time)
# ---------------------------------------------------------------------------


class _FakeOutcome:
    """Minimal GroundingOutcome shape needed by events_for_result."""

    def __init__(self, passed: bool, reason: str) -> None:
        self.passed = passed
        self.reason = reason


class _FakeFiled:
    """Minimal FiledStatus shape."""

    def __init__(self) -> None:
        self.slug = "test-slug-abc123"
        self.status = "draft"
        self.op = "created"
        self.count = 1


# ---------------------------------------------------------------------------
# encode_event
# ---------------------------------------------------------------------------


def test_encode_event_format():
    """encode_event produces valid SSE frame: event:, data:, double newline."""
    frame = encode_event("sources", {"sources": []})
    lines = frame.split("\n")
    assert lines[0] == "event: sources"
    assert lines[1].startswith("data: ")
    assert frame.endswith("\n\n")


def test_encode_event_data_is_json():
    """encode_event data field is valid JSON."""
    payload = {"sources": [{"source": "a.md#b", "heading": "B"}]}
    frame = encode_event("sources", payload)
    data_line = [ln for ln in frame.split("\n") if ln.startswith("data: ")][0]
    parsed = json.loads(data_line[len("data: ") :])
    assert parsed == payload


def test_encode_event_unicode():
    """encode_event preserves non-ASCII characters (ensure_ascii=False)."""
    frame = encode_event("token", {"text": "退款"})
    assert "退款" in frame


# ---------------------------------------------------------------------------
# events_for_result — structure
# ---------------------------------------------------------------------------


def _make_result(
    answer: str = "The answer.",
    sources: list | None = None,
    passed: bool = True,
    reason: str = "claim_supported",
    filed=None,
) -> dict:
    return {
        "answer": answer,
        "sources": sources if sources is not None else [],
        "grounding_outcome": _FakeOutcome(passed=passed, reason=reason),
        "filed": filed,
    }


def _parse_frames(frames: list[str]) -> list[dict]:
    """Parse a list of SSE frame strings into (event_type, data) dicts."""
    out = []
    for frame in frames:
        lines = frame.strip().split("\n")
        event_type = lines[0].split(": ", 1)[1]
        data = json.loads(lines[1].split(": ", 1)[1])
        out.append({"type": event_type, "data": data})
    return out


def test_events_order_is_sources_tokens_done():
    """events_for_result emits sources first, then token(s), then done."""
    result = _make_result(answer="Hello world")
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    types = [p["type"] for p in parsed]
    # sources first
    assert types[0] == "sources"
    # done last
    assert types[-1] == "done"
    # every middle frame is a token
    assert all(t == "token" for t in types[1:-1])


def test_sources_event_carries_source_fields():
    """sources event payload contains source id, heading, content, derived_from."""
    sources = [
        {
            "source": "refund_policy.md#refund-timeline",
            "heading": "Refund Timeline",
            "score": 0.8,
            "content": "Refunds processed in 5-7 days.",
            "derived_from": None,
        }
    ]
    result = _make_result(sources=sources)
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    src_event = parsed[0]
    assert src_event["type"] == "sources"
    src_list = src_event["data"]["sources"]
    assert len(src_list) == 1
    item = src_list[0]
    assert item["source"] == "refund_policy.md#refund-timeline"
    assert item["heading"] == "Refund Timeline"
    assert item["content"] == "Refunds processed in 5-7 days."
    assert item["derived_from"] is None
    # score is NOT in the sources event (not in the SSE wire format)
    assert "score" not in item


def test_sources_event_excludes_score():
    """score is an internal retrieval detail; not sent to the client."""
    sources = [
        {
            "source": "a.md#b",
            "heading": "B",
            "score": 0.99,
            "content": "...",
            "derived_from": None,
        }
    ]
    result = _make_result(sources=sources)
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    src_item = parsed[0]["data"]["sources"][0]
    assert "score" not in src_item


def test_token_events_cover_full_answer():
    """Joining all token texts (space-separated) reconstructs the answer."""
    answer = "Refunds are processed within five days."
    result = _make_result(answer=answer)
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    tokens = [p["data"]["text"] for p in parsed if p["type"] == "token"]
    reconstructed = " ".join(tokens)
    assert reconstructed == answer


def test_cannot_confirm_emits_token_frames():
    """Cannot Confirm phrase is streamed as token event(s), not a special event."""
    from app.retrieval import CANNOT_CONFIRM_PHRASE

    result = _make_result(
        answer=CANNOT_CONFIRM_PHRASE,
        passed=False,
        reason="below_threshold",
    )
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    token_texts = [p["data"]["text"] for p in parsed if p["type"] == "token"]
    reconstructed = " ".join(token_texts)
    assert reconstructed == CANNOT_CONFIRM_PHRASE
    # No special event type introduced (ADR-0009 uniformity)
    event_types = {p["type"] for p in parsed}
    assert "cannot_confirm" not in event_types


def test_done_event_carries_grounding_outcome():
    """done event carries passed and reason from the grounding outcome."""
    result = _make_result(passed=False, reason="claim_unsupported")
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    done = parsed[-1]
    assert done["type"] == "done"
    assert done["data"]["passed"] is False
    assert done["data"]["reason"] == "claim_unsupported"


def test_done_event_filed_none_when_not_filed():
    """done.filed is null when no filing happened (Cannot Confirm paths)."""
    result = _make_result(passed=False, reason="retrieval_empty", filed=None)
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    done_data = parsed[-1]["data"]
    assert done_data["filed"] is None


def test_done_event_filed_populated_when_filed():
    """done.filed carries slug/status/op/count when Answer Filing ran."""
    filed = _FakeFiled()
    result = _make_result(passed=True, reason="claim_supported", filed=filed)
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    done_data = parsed[-1]["data"]
    assert done_data["filed"] is not None
    assert done_data["filed"]["slug"] == "test-slug-abc123"
    assert done_data["filed"]["status"] == "draft"
    assert done_data["filed"]["op"] == "created"
    assert done_data["filed"]["count"] == 1


def test_empty_sources_list():
    """Empty sources list produces a sources event with an empty array."""
    result = _make_result(sources=[])
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    assert parsed[0]["type"] == "sources"
    assert parsed[0]["data"]["sources"] == []


def test_minimum_three_frames():
    """Every result produces at least 3 frames: sources + at least 1 token + done."""
    result = _make_result(answer="Hello")
    frames = events_for_result(result)
    assert len(frames) >= 3


def test_no_token_frames_for_empty_answer():
    """An empty answer string produces only sources + done (no token frames)."""
    result = _make_result(answer="")
    frames = events_for_result(result)
    parsed = _parse_frames(frames)
    types = [p["type"] for p in parsed]
    assert "token" not in types
    assert types[0] == "sources"
    assert types[-1] == "done"
