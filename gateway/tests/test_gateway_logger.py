"""Hermetic tests for gateway/app/logger.py — log_event format and chat_rewrite emission.

Covers:
- gateway logger writes correctly-formatted lines to gateway/log.md
- turn-2+ request writes exactly one chat_rewrite entry to gateway/log.md
- turn-1 request writes no chat_rewrite entry
- bounded raw/rewritten fields (60-char truncation per §5.3)
- Phase 5 /lint reads wiki/log.md only — chat_rewrite (in gateway/log.md) never reaches lint
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _wiki_logger
import markdown_kb.app.retrieval as _retrieval
import pytest
from fastapi.testclient import TestClient
from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

import gateway.app.conversation_store as _store_module
import gateway.app.logger as _gateway_logger_module
import gateway.app.query_rewriting as _rewrite_module
from gateway.app.logger import log_event

_FIXTURE_DOCS = Path(__file__).resolve().parents[2] / "markdown_kb" / "tests" / "fixtures" / "docs"

LOG_LINE_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] \S+ \| .+\n$")


# ---------------------------------------------------------------------------
# Unit tests: gateway logger module
# ---------------------------------------------------------------------------


def test_gateway_logger_format_and_append(tmp_path, monkeypatch):
    """log_event writes ISO-8601 UTC line; successive calls append in order."""
    log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(_gateway_logger_module, "LOG_PATH", log_path)

    # First call — directory does not yet exist, must be created
    log_event("chat_rewrite", 'session=abc raw="hello" rewritten="hello there"')
    assert log_path.exists(), "gateway/log.md must be created if missing"

    lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == 1
    assert re.match(LOG_LINE_RE, lines[0]), f"Line format mismatch: {repr(lines[0])}"
    assert "chat_rewrite" in lines[0]

    # Second call must append, not overwrite
    log_event("chat_rewrite", 'session=xyz raw="foo" rewritten="bar"')
    lines2 = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines2) == 2
    assert "abc" in lines2[0]
    assert "xyz" in lines2[1]


def test_gateway_logger_creates_parent_dir(tmp_path, monkeypatch):
    """log_event creates the parent directory if it doesn't exist."""
    log_path = tmp_path / "gateway" / "log.md"
    assert not (tmp_path / "gateway").exists()
    monkeypatch.setattr(_gateway_logger_module, "LOG_PATH", log_path)

    log_event("chat_rewrite", 'session=x raw="q" rewritten="q"')
    assert (tmp_path / "gateway").exists()
    assert log_path.exists()


def test_gateway_log_path_is_under_gateway_package():
    """LOG_PATH must be under the gateway/ package directory, NOT wiki/."""
    log_path = _gateway_logger_module.LOG_PATH
    # Must resolve to something like .../gateway/log.md
    assert log_path.name == "log.md"
    assert "gateway" in str(log_path).lower(), (
        f"Gateway LOG_PATH must be under gateway/, got: {log_path}"
    )
    # Must NOT point at wiki/log.md (would write into markdown_kb's channel)
    assert "wiki" not in str(log_path).lower(), (
        f"Gateway LOG_PATH must NOT point at wiki/log.md, got: {log_path}"
    )


# ---------------------------------------------------------------------------
# Integration tests: chat_rewrite emitted on turn 2+, not on turn 1
# ---------------------------------------------------------------------------


@pytest.fixture()
def _approved_outcome():
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
def _indexed_wiki(tmp_path, monkeypatch):
    """Build Section Index from hermetic fixture docs."""
    monkeypatch.setattr(_wiki_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")
    _indexer.build_index(_FIXTURE_DOCS)
    yield
    _indexer.sections.clear()


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    """Fresh ConversationStore per test."""
    from gateway.app.conversation_store import ConversationStore

    monkeypatch.setattr(_store_module, "store", ConversationStore())


@pytest.fixture(autouse=True)
def _reset_rewrite_llm(monkeypatch):
    monkeypatch.setattr(_rewrite_module, "_rewrite_llm", None)


class _FakeLLMResponse:
    content = (
        "Approved refunds are processed within 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )


class _FakeLLM:
    def invoke(self, messages):
        return _FakeLLMResponse()


class _FakeRewriteLLM:
    def __init__(self, rewritten: str = "how long do exchanges take?") -> None:
        self._rewritten = rewritten

    def with_structured_output(self, schema):
        chain = MagicMock()
        result = MagicMock()
        result.rewritten_query = self._rewritten
        chain.invoke.return_value = result
        return chain


def _parse_sse_events(content: str) -> list[dict]:
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


@pytest.fixture()
def _logger_client(_indexed_wiki, tmp_path, monkeypatch, _approved_outcome):
    """TestClient with mocked LLM/grounding/rewrite + gateway log redirected to tmp."""
    # Redirect gateway log to tmp
    gw_log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(_gateway_logger_module, "LOG_PATH", gw_log_path)

    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome,
    )
    fake_rewrite = _FakeRewriteLLM("how long do exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app), gw_log_path


def test_turn1_writes_no_chat_rewrite_entry(_logger_client):
    """Turn 1 (no prior history) MUST NOT write a chat_rewrite entry to gateway/log.md."""
    client, gw_log_path = _logger_client
    resp = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200

    # No gateway log written at all on turn 1
    if gw_log_path.exists():
        content = gw_log_path.read_text(encoding="utf-8")
        assert "chat_rewrite" not in content, (
            f"Turn 1 must NOT write chat_rewrite to gateway/log.md; found: {content!r}"
        )


def test_turn2_writes_exactly_one_chat_rewrite_entry(_logger_client):
    """Turn 2+ MUST write exactly one chat_rewrite entry to gateway/log.md."""
    client, gw_log_path = _logger_client

    # First turn to create a session
    resp1 = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp1.status_code == 200
    events1 = _parse_sse_events(resp1.text)
    session_id = next(e for e in events1 if e["type"] == "done")["data"]["session"]

    # Reset the log so we're only checking turn 2's write
    if gw_log_path.exists():
        gw_log_path.write_text("", encoding="utf-8")

    # Second turn
    resp2 = client.post(
        f"/chat/stream?stack=wiki&session={session_id}",
        json={"query": "and exchanges?"},
    )
    assert resp2.status_code == 200

    assert gw_log_path.exists(), "gateway/log.md must exist after turn 2"
    content = gw_log_path.read_text(encoding="utf-8")
    lines = [ln for ln in content.splitlines() if "chat_rewrite" in ln]
    assert len(lines) == 1, (
        f"Turn 2 must write exactly one chat_rewrite entry; found {len(lines)}: {lines}"
    )


def test_chat_rewrite_entry_contains_session_raw_rewritten(_logger_client):
    """chat_rewrite entry must include session=, raw=, and rewritten= fields."""
    client, gw_log_path = _logger_client

    resp1 = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    session_id = next(e for e in _parse_sse_events(resp1.text) if e["type"] == "done")["data"][
        "session"
    ]

    resp2 = client.post(
        f"/chat/stream?stack=wiki&session={session_id}",
        json={"query": "and exchanges?"},
    )
    assert resp2.status_code == 200

    content = gw_log_path.read_text(encoding="utf-8")
    rewrite_lines = [ln for ln in content.splitlines() if "chat_rewrite" in ln]
    assert rewrite_lines, "Expected at least one chat_rewrite log line"
    line = rewrite_lines[-1]

    assert "session=" in line, f"chat_rewrite line must have session=: {line!r}"
    assert "raw=" in line, f"chat_rewrite line must have raw=: {line!r}"
    assert "rewritten=" in line, f"chat_rewrite line must have rewritten=: {line!r}"
    # Must match the standard log line format
    assert re.match(LOG_LINE_RE, line + "\n"), f"Format mismatch: {repr(line)}"


def test_chat_rewrite_raw_bounded_to_60_chars(_logger_client):
    """raw= field in chat_rewrite must be bounded to 60 chars per §5.3."""
    client, gw_log_path = _logger_client

    long_query = "a" * 80  # 80 chars — exceeds the 60-char bound

    resp1 = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    session_id = next(e for e in _parse_sse_events(resp1.text) if e["type"] == "done")["data"][
        "session"
    ]

    resp2 = client.post(
        f"/chat/stream?stack=wiki&session={session_id}",
        json={"query": long_query},
    )
    assert resp2.status_code == 200

    content = gw_log_path.read_text(encoding="utf-8")
    rewrite_lines = [ln for ln in content.splitlines() if "chat_rewrite" in ln]
    assert rewrite_lines
    line = rewrite_lines[-1]

    # Extract the raw= value (between raw=" and " rewritten=)
    raw_match = re.search(r'raw="([^"]*)"', line)
    assert raw_match, f'Could not find raw="..." in: {line!r}'
    raw_val = raw_match.group(1)
    assert len(raw_val) <= 60, f"raw= value must be ≤60 chars, got {len(raw_val)}: {raw_val!r}"


def test_chat_rewrite_rewritten_bounded_to_60_chars(
    _indexed_wiki, tmp_path, monkeypatch, _approved_outcome
):
    """rewritten= field in chat_rewrite must be bounded to 60 chars per §5.3."""
    gw_log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(_gateway_logger_module, "LOG_PATH", gw_log_path)

    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome,
    )

    long_rewritten = "b" * 80  # rewrite LLM returns an 80-char string
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: _FakeRewriteLLM(long_rewritten))

    session_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        session_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={session_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200

    content = gw_log_path.read_text(encoding="utf-8")
    rewrite_lines = [ln for ln in content.splitlines() if "chat_rewrite" in ln]
    assert rewrite_lines
    line = rewrite_lines[-1]

    rewritten_match = re.search(r'rewritten="([^"]*)"', line)
    assert rewritten_match, f'Could not find rewritten="..." in: {line!r}'
    rewritten_val = rewritten_match.group(1)
    assert len(rewritten_val) <= 60, (
        f"rewritten= value must be ≤60 chars, got {len(rewritten_val)}: {rewritten_val!r}"
    )


# ---------------------------------------------------------------------------
# AC: Phase 5 /lint unaffected — chat_rewrite never reaches wiki/log.md
# ---------------------------------------------------------------------------


def test_chat_rewrite_does_not_write_to_wiki_log(
    _indexed_wiki, tmp_path, monkeypatch, _approved_outcome
):
    """chat_rewrite entries go ONLY to gateway/log.md, never to wiki/log.md.

    Phase 5 /lint reads wiki/log.md and matches kinds chat_fallback /
    chat_grounding_fallback only.  A chat_rewrite entry landing in wiki/log.md
    would be a cross-package log-channel violation (CODING_STANDARD §5.1).
    """
    wiki_log_path = tmp_path / "wiki" / "log.md"
    gw_log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(_wiki_logger, "LOG_PATH", wiki_log_path)
    monkeypatch.setattr(_gateway_logger_module, "LOG_PATH", gw_log_path)

    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome,
    )
    monkeypatch.setattr(
        _rewrite_module, "get_rewrite_llm", lambda: _FakeRewriteLLM("how long do exchanges take?")
    )

    session_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        session_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={session_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200

    # gateway/log.md must contain chat_rewrite
    assert gw_log_path.exists(), "gateway/log.md must exist after turn 2"
    gw_content = gw_log_path.read_text(encoding="utf-8")
    assert "chat_rewrite" in gw_content, "chat_rewrite must be in gateway/log.md"

    # wiki/log.md must NOT contain chat_rewrite
    if wiki_log_path.exists():
        wiki_content = wiki_log_path.read_text(encoding="utf-8")
        assert "chat_rewrite" not in wiki_content, (
            f"chat_rewrite must NOT appear in wiki/log.md; found: {wiki_content!r}"
        )
