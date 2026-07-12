"""Tests for gateway/app/sse_capacity.py — SSE slow-client capacity guard (issue #599).

Covers the three ACs, split from #580's ops availability hardening:
  1. Per-server concurrent SSE cap (KB_SSE_MAX_CONCURRENT) — over-limit sheds
     /chat/stream with the same clean busy-retry 503 shape the general
     read/admin semaphores already use (no 500, no hang).
  2. Heartbeat — a ": ping" SSE comment frame is injected while the
     downstream generator is idle (e.g. during the draft+verify LLM gap),
     without disturbing the existing sources/status/token/done event
     contract (ADR-0009) — the reader UI already ignores unknown event
     types (§12.2, gateway/static/index.html `dispatch()`), and a comment
     frame parses to exactly that shape (event="message", data=null).
  3. Idle-read timeout (KB_SSE_IDLE_TIMEOUT_SEC) — a client that stops
     draining its socket (send() never completes) is closed server-side
     instead of held open forever.

Per project-docs/agents/implement.md's brief for this issue ("mock
slow/stalled clients at the ASGI level; no wall-clock sleeps beyond minimal
timeouts — drive via injected clock/config where possible"):
  - The heartbeat/idle-timeout mechanics are exercised directly against
    `run_with_heartbeat()`'s ASGI-shaped `send` callable, using hand-rolled
    fake `send`/`call` coroutines (no real client, no real socket).
  - Every interval is a keyword-argument override (hundredths of a second),
    not a real env-var wall-clock wait.
  - No assertion depends on an exact elapsed duration or exact ping COUNT
    (CODING_STANDARD §6.2 — "don't assert wall-clock timing"); each test
    asserts that the right KIND of frame/behaviour occurred, wrapped in a
    generous outer `asyncio.wait_for` so a regression that drops a timeout
    fails the test via an actual timeout rather than hanging the suite.

All hermetic — no OPENAI_API_KEY, no real network.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _wiki_logger
import markdown_kb.app.retrieval as _retrieval
import pytest
import vector_rag.app.indexer as _rag_indexer
from fastapi.testclient import TestClient
from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

import gateway.app.logger as _gw_logger
import gateway.app.sse_capacity as _sse_capacity

_FIXTURE_DOCS = Path(__file__).resolve().parents[2] / "markdown_kb" / "tests" / "fixtures" / "docs"


# ---------------------------------------------------------------------------
# encode_comment — pure function
# ---------------------------------------------------------------------------


def test_encode_comment_is_a_valid_sse_comment_frame():
    """A comment frame starts with ':' and terminates with the SSE blank line
    (RFC 8895 §3.1) — the exact shape a heartbeat must have."""
    assert _sse_capacity.encode_comment("ping") == b": ping\n\n"


def test_encode_comment_never_starts_an_event_or_data_line():
    """A heartbeat must never collide with the event:/data: contract (ADR-0009) —
    every conforming SSE parser (including gateway/static/index.html's) treats
    a ':'-prefixed line as an ignorable comment."""
    for line in _sse_capacity.encode_comment("ping").decode().splitlines():
        assert not line.startswith("event:")
        assert not line.startswith("data:")


# ---------------------------------------------------------------------------
# Config defaults (issue #599 scope decision)
# ---------------------------------------------------------------------------


def test_default_config_values_and_ordering(monkeypatch):
    """Defaults land where the issue's scope decision baked them, and the
    heartbeat cadence stays under the idle timeout so at least two heartbeats
    can land before an idle-read timeout would otherwise fire."""
    monkeypatch.delenv("KB_SSE_IDLE_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("KB_SSE_HEARTBEAT_INTERVAL_SEC", raising=False)
    monkeypatch.delenv("KB_SSE_MAX_CONCURRENT", raising=False)
    importlib.reload(_sse_capacity)
    try:
        assert _sse_capacity.IDLE_TIMEOUT_SEC == 45.0
        assert _sse_capacity.HEARTBEAT_INTERVAL_SEC > 0
        assert _sse_capacity.HEARTBEAT_INTERVAL_SEC < _sse_capacity.IDLE_TIMEOUT_SEC
        assert _sse_capacity.MAX_CONCURRENT > 0
    finally:
        importlib.reload(_sse_capacity)  # restore for later tests in this module


def test_config_reads_env_overrides_and_ignores_invalid_values(monkeypatch):
    """A set, valid override wins; an unset or invalid value falls back safely
    (mirrors gateway.app.middleware._read_int / gateway.app.budget._read_cap)."""
    monkeypatch.setenv("KB_SSE_IDLE_TIMEOUT_SEC", "12.5")
    monkeypatch.setenv("KB_SSE_HEARTBEAT_INTERVAL_SEC", "not-a-number")
    monkeypatch.setenv("KB_SSE_MAX_CONCURRENT", "-3")
    importlib.reload(_sse_capacity)
    try:
        assert _sse_capacity.IDLE_TIMEOUT_SEC == 12.5
        assert _sse_capacity.HEARTBEAT_INTERVAL_SEC == 15.0  # invalid -> default
        assert _sse_capacity.MAX_CONCURRENT == 6  # negative -> default
    finally:
        monkeypatch.delenv("KB_SSE_IDLE_TIMEOUT_SEC", raising=False)
        monkeypatch.delenv("KB_SSE_HEARTBEAT_INTERVAL_SEC", raising=False)
        monkeypatch.delenv("KB_SSE_MAX_CONCURRENT", raising=False)
        importlib.reload(_sse_capacity)  # restore for later tests in this module


# ---------------------------------------------------------------------------
# run_with_heartbeat — direct ASGI-level tests (AC1 heartbeat, AC2 idle-timeout)
# ---------------------------------------------------------------------------


def test_run_with_heartbeat_emits_ping_while_call_is_slow():
    """A slow `call` (standing in for the several-second draft+verify LLM gap)
    gets at least one heartbeat comment frame injected before its own real
    messages resume."""

    async def _scenario() -> list[dict]:
        sent: list[dict] = []

        async def fake_send(message):
            sent.append(message)

        async def slow_call(queued_send):
            await queued_send({"type": "http.response.start", "status": 200, "headers": []})
            await asyncio.sleep(0.08)  # spans several heartbeat intervals below
            await queued_send({"type": "http.response.body", "body": b"real", "more_body": False})

        await _sse_capacity.run_with_heartbeat(
            slow_call, fake_send, heartbeat_interval=0.02, idle_timeout=5.0
        )
        return sent

    sent = asyncio.run(asyncio.wait_for(_scenario(), timeout=5.0))

    ping_frames = [
        m
        for m in sent
        if m["type"] == "http.response.body" and m["body"] == _sse_capacity.encode_comment("ping")
    ]
    assert len(ping_frames) >= 1, f"expected at least one heartbeat frame, got: {sent}"
    # Real frames are still delivered, unmolested, on either side of the gap.
    assert sent[0] == {"type": "http.response.start", "status": 200, "headers": []}
    assert sent[-1]["body"] == b"real"
    assert sent[-1]["more_body"] is False


def test_run_with_heartbeat_no_ping_before_headers_sent():
    """A heartbeat never fires before `http.response.start` — there is nothing
    to comment onto yet (SSE framing requires headers first)."""

    async def _scenario() -> list[dict]:
        sent: list[dict] = []

        async def fake_send(message):
            sent.append(message)

        async def call(queued_send):
            await asyncio.sleep(0.06)  # idle gap BEFORE any message is queued
            await queued_send({"type": "http.response.start", "status": 200, "headers": []})
            await queued_send({"type": "http.response.body", "body": b"x", "more_body": False})

        await _sse_capacity.run_with_heartbeat(
            call, fake_send, heartbeat_interval=0.02, idle_timeout=5.0
        )
        return sent

    sent = asyncio.run(asyncio.wait_for(_scenario(), timeout=5.0))
    assert sent[0]["type"] == "http.response.start", (
        f"expected no heartbeat before headers, got first message: {sent[0]!r}"
    )


def test_run_with_heartbeat_closes_on_stalled_send(monkeypatch, tmp_path):
    """A `send` that never completes (a stalled TCP peer that stopped reading)
    trips the idle-read timeout and returns instead of hanging forever."""
    log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(_gw_logger, "LOG_PATH", log_path)

    async def _scenario() -> None:
        async def stalled_send(message):
            await asyncio.Event().wait()  # never completes — simulates a full send buffer

        async def call(queued_send):
            await queued_send({"type": "http.response.start", "status": 200, "headers": []})

        await _sse_capacity.run_with_heartbeat(
            call, stalled_send, heartbeat_interval=10.0, idle_timeout=0.02
        )

    # The outer timeout is the real assertion: a regression that drops the
    # idle-read wrap would hang here and fail via TimeoutError, never a false
    # green.
    asyncio.run(asyncio.wait_for(_scenario(), timeout=5.0))

    assert log_path.exists(), "expected an sse_idle_timeout log line to be written"
    assert "sse_idle_timeout" in log_path.read_text(encoding="utf-8")


def test_run_with_heartbeat_stalled_heartbeat_send_also_times_out(monkeypatch, tmp_path):
    """A stalled `send` for a HEARTBEAT frame (not just a real frame) also
    trips the idle-read timeout — the client made no read progress either way."""
    log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(_gw_logger, "LOG_PATH", log_path)

    async def _scenario() -> None:
        async def stalled_after_start_send(message):
            if message["type"] == "http.response.start":
                return  # headers go through fine; only later sends stall
            await asyncio.Event().wait()  # every later send (the heartbeat) hangs

        async def call(queued_send):
            await queued_send({"type": "http.response.start", "status": 200, "headers": []})
            await asyncio.sleep(1.0)  # never resumes with a real frame in time

        await _sse_capacity.run_with_heartbeat(
            call, stalled_after_start_send, heartbeat_interval=0.02, idle_timeout=0.03
        )

    asyncio.run(asyncio.wait_for(_scenario(), timeout=5.0))
    assert "sse_idle_timeout" in log_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Full-stack tests through ProdMiddleware + POST /chat/stream
# ---------------------------------------------------------------------------


def _fresh_app():
    """Reload the gateway app so module-level middleware/sse state is pristine
    (mirrors gateway/tests/test_prod_middleware.py's `_fresh_app` pattern)."""
    import gateway.app.budget as budget_mod
    import gateway.app.main as main_mod
    import gateway.app.middleware as mw_mod

    importlib.reload(budget_mod)
    importlib.reload(_sse_capacity)
    importlib.reload(mw_mod)
    importlib.reload(main_mod)
    return main_mod.app


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect INDEX_PATH, LOG_PATH, WIKI_DIR to tmp for every test in this file."""
    monkeypatch.setattr(_wiki_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(_rag_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(_rag_indexer, "vectorstore", None)


@pytest.fixture()
def indexed_wiki_corpus():
    _indexer.build_index(_FIXTURE_DOCS)
    yield
    _indexer.sections.clear()


def _clear_env(monkeypatch):
    for name in (
        "KB_SSE_MAX_CONCURRENT",
        "KB_SSE_HEARTBEAT_INTERVAL_SEC",
        "KB_SSE_IDLE_TIMEOUT_SEC",
        "KB_MAX_INFLIGHT",
        "KB_MAX_ADMIN",
        "KB_DAILY_USD_CAP",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sse_cap_sheds_with_busy_retry_once_full(monkeypatch):
    """Draining the SSE-specific semaphore sheds /chat/stream with the same
    clean 503 busy-retry shape as the general read/admin semaphores — no
    500, no hang (AC3)."""
    _clear_env(monkeypatch)
    client = TestClient(_fresh_app())

    acquired = []
    while _sse_capacity.sse_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.post("/chat/stream", json={"query": "anything"})
        assert resp.status_code == 503
        assert resp.json() == {"detail": "server busy, please retry"}
    finally:
        for _ in acquired:
            _sse_capacity.sse_sem.release()


def test_sse_cap_releases_slot_after_a_normal_request(monkeypatch, indexed_wiki_corpus):
    """The SSE semaphore slot is released once a real stream completes, so a
    saturated cap is transient, not a stuck/leaked slot."""
    _clear_env(monkeypatch)
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module, "verify", lambda draft, sections: _approved_outcome()
    )
    client = TestClient(_fresh_app())

    resp = client.post("/chat/stream?stack=wiki", json={"query": "What is the refund policy?"})
    assert resp.status_code == 200

    # If the slot leaked, this drain would find fewer than MAX_CONCURRENT free.
    acquired = []
    while _sse_capacity.sse_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        assert len(acquired) == _sse_capacity.MAX_CONCURRENT
    finally:
        for _ in acquired:
            _sse_capacity.sse_sem.release()


def test_heartbeat_frames_appear_without_disturbing_the_sse_event_contract(
    monkeypatch, indexed_wiki_corpus
):
    """End-to-end: a slow draft+verify LLM call gets heartbeat comment frames
    injected, and the sources -> status -> token(s) -> done contract (ADR-0009)
    is still intact once they are filtered out — a heartbeat is invisible to
    any event:/data: consumer (AC1)."""
    monkeypatch.setenv("KB_SSE_HEARTBEAT_INTERVAL_SEC", "0.03")
    monkeypatch.setenv("KB_SSE_IDLE_TIMEOUT_SEC", "5.0")
    monkeypatch.delenv("KB_SSE_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("KB_MAX_INFLIGHT", raising=False)
    monkeypatch.delenv("KB_MAX_ADMIN", raising=False)
    monkeypatch.delenv("KB_DAILY_USD_CAP", raising=False)

    fake_llm = _SlowFakeLLM(delay_sec=0.12)
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module, "verify", lambda draft, sections: _approved_outcome()
    )
    client = TestClient(_fresh_app())

    resp = client.post("/chat/stream?stack=wiki", json={"query": "What is the refund policy?"})
    assert resp.status_code == 200
    assert resp.text.count(": ping\n\n") >= 1, (
        "expected at least one heartbeat during the slow draft+verify gap"
    )

    events = _parse_sse_events(resp.text)  # comment-only frames are dropped by this parser
    types = [e["type"] for e in events]
    assert types[0] == "sources"
    assert types[-1] == "done"
    assert "status" in types[1:-1]
    assert all(t in ("status", "token") for t in types[1:-1])


# ---------------------------------------------------------------------------
# Fixtures / helpers shared by the full-stack tests above
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    """Minimal LLM stub (mirrors gateway/tests/test_chat_stream.py's)."""

    CANNED_ANSWER = (
        "Approved refunds are processed within 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


class _SlowFakeLLM(_FakeLLM):
    """Like _FakeLLM, but blocks for `delay_sec` before answering.

    `_sse_generator` (gateway/app/routes.py) is a SYNC generator, driven by
    Starlette's StreamingResponse via anyio's threadpool — this sleep runs on
    that worker thread, not the event loop, so it does not block the
    heartbeat pump running concurrently on the event loop (see the
    sse_capacity module docstring).
    """

    def __init__(self, *, delay_sec: float) -> None:
        self._delay_sec = delay_sec

    def invoke(self, messages):
        time.sleep(self._delay_sec)
        return super().invoke(messages)


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


def _parse_sse_events(content: str) -> list[dict]:
    """Parse raw SSE text into {type, data} dicts (mirrors
    gateway/tests/test_chat_stream.py's `_parse_sse_response`). A comment-only
    frame (no `data:` line, e.g. a heartbeat) is silently dropped — the same
    behaviour a real event:/data: consumer relies on."""
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
