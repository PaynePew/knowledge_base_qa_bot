"""Integration tests for kb_transcribe_v1 using the FastMCP in-process harness.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.  This exercises FastMCP's argument
validation and the full dispatch path.

Mocking strategy
-----------------
``transcribe_path`` from ``markdown_kb.app.transcriber`` is the single deep
entry point this tool wraps (ADR-0017 / issue #427, reusing the #426 deep
module).  Tests redirect ``markdown_kb.app.transcriber.RAW_DIR`` and
``DOCS_DIR`` to tmp dirs so writes land in hermetic locations, and fake the
model ONLY at the lazy-singleton getter ``get_transcribe_llm`` (CODING_STANDARD
§6.3) — the real rasterization / hash-skip / provenance-rendering pipeline
runs unmocked, exactly as T1's own unit tests (``test_transcriber.py``) and
the HTTP route tests (``test_transcribe_route.py``) do.

CRITICAL: do NOT mock ``transcribe_path`` / ``transcribe_source`` /
``transcribe_pdf_bytes``.  Mocking a deep-module entry point is a FAIL
(CODING_STANDARD §6.3/§11, implement.md §3.1).

No new live test is added here — this tool is a second caller of the single
``get_transcribe_llm`` surface already carrying its one live smoke test
(``markdown_kb/tests/test_transcribe_live.py``, ADR-0005 §6.4). §6.4
accounting is unchanged by this slice.

AC reference (issue #427)
--------------------------
AC-1  kb_transcribe_v1(path) force-transcribes a digital-native fixture
      through the real deep module (faked model) and returns docs path +
      origin: transcribed provenance.
AC-2  Typed failures (TranscribeUnavailable, TranscribePageLimitExceeded,
      TranscribeError, plus the shared validation types) surface as
      structured tool errors whose `code` is the same reason string the
      CLI / HTTP surfaces use.
AC-3  Hash-skip result (`status='skipped'`) round-trips through the tool
      envelope with no second model call.
AC-4  No second live test is added (ADR-0005 / §6.4 accounting unchanged).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "markdown_kb" / "tests" / "fixtures" / "raw_import"


@dataclass(frozen=True)
class _FakeLLMResponse:
    """Minimal local stand-in for a langchain_core message (`.content` only).

    Not imported cross-package from ``markdown_kb.tests.conftest`` — that
    conftest is scoped to markdown_kb's own pytest rootdir (``app.*`` imports),
    not a public symbol kb_mcp is blessed to reach into (CODING_STANDARD §2.4).
    """

    content: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result."""
    from mcp.types import CallToolResult

    if isinstance(raw, CallToolResult):
        return json.loads(raw.content[0].text)
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


def _is_error_result(raw: Any) -> bool:
    """Return True when the MCP call result has isError=True."""
    from mcp.types import CallToolResult

    return isinstance(raw, CallToolResult) and raw.isError is True


class FakeTranscribeLLM:
    """Records calls; returns a canned page body (mirrors test_transcriber.py's fake)."""

    def __init__(self, body: str = "# forced\ntranscribed via kb_transcribe_v1."):
        self.call_count = 0
        self.body = body

    def invoke(self, messages):
        self.call_count += 1
        return _FakeLLMResponse(content=self.body)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_transcribe_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Redirect transcriber RAW_DIR/DOCS_DIR to tmp and fake the vision model.

    Mirrors ``markdown_kb/tests/test_transcribe_route.py``'s
    ``transcribe_route_env`` fixture, adapted for the MCP in-process harness.
    """
    import markdown_kb.app.transcriber as transcriber_mod

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(transcriber_mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(transcriber_mod, "DOCS_DIR", docs_dir)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")
    monkeypatch.setenv("OPENAI_TRANSCRIBE_MODEL", "gpt-5-mini-test")

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_mod, "get_transcribe_llm", lambda: fake_llm)

    return {"raw_dir": raw_dir, "docs_dir": docs_dir, "fake_llm": fake_llm}


# ---------------------------------------------------------------------------
# AC: tool registration and strict schema
# ---------------------------------------------------------------------------


def test_kb_transcribe_v1_tool_registered():
    """kb_transcribe_v1 is registered as a tool in the MCP server."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = [t.name for t in tools]
    assert "kb_transcribe_v1" in names, f"kb_transcribe_v1 not in tool list: {names}"


def test_kb_transcribe_v1_schema_additional_properties_false():
    """kb_transcribe_v1 schema has additionalProperties:false (ADR-0016)."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_transcribe_v1"), None)
    assert tool is not None, "kb_transcribe_v1 not registered"

    schema = tool.inputSchema
    assert schema.get("additionalProperties") is False, (
        f"additionalProperties:false missing from schema: {schema}"
    )


def test_kb_transcribe_v1_schema_required_fields():
    """kb_transcribe_v1 schema requires 'path'."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_transcribe_v1"), None)
    assert tool is not None, "kb_transcribe_v1 not registered"

    required = tool.inputSchema.get("required", [])
    assert "path" in required, f"'path' not in required: {required}"


def test_kb_transcribe_v1_schema_properties_present():
    """kb_transcribe_v1 schema exposes a 'path' property."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_transcribe_v1"), None)
    assert tool is not None, "kb_transcribe_v1 not registered"

    props = tool.inputSchema.get("properties", {})
    assert "path" in props, f"'path' not in properties: {props}"


# ---------------------------------------------------------------------------
# AC-1: successful forced transcription — real deep module, faked model
# ---------------------------------------------------------------------------


def test_kb_transcribe_v1_success_returns_ok(tmp_transcribe_env, tmp_path):
    """kb_transcribe_v1 returns {ok: true, ...} for a digital-native PDF."""
    import asyncio
    import shutil

    from kb_mcp.server import mcp

    src = tmp_path / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", src)

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))

    assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
    result = _parse_result(raw)
    assert result.get("ok") is True, f"Expected ok=true, got: {result}"
    assert tmp_transcribe_env["fake_llm"].call_count == 1, (
        "The force entry must call the model even for a digital-native PDF"
    )


def test_kb_transcribe_v1_success_writes_docs_source(tmp_transcribe_env, tmp_path):
    """kb_transcribe_v1 writes a real docs/ Source via the Transcribe deep module."""
    import asyncio
    import shutil

    from kb_mcp.server import mcp

    docs_dir = tmp_transcribe_env["docs_dir"]
    src = tmp_path / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", src)

    asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))

    docs_file = docs_dir / "sample_english.md"
    assert docs_file.exists(), (
        f"Expected docs/ Source at {docs_file}; docs/ contains: {list(docs_dir.iterdir())}"
    )
    text = docs_file.read_text(encoding="utf-8")
    assert "transcribed via kb_transcribe_v1" in text, f"Fake body not found: {text[:300]}"


def test_kb_transcribe_v1_success_payload_includes_origin_provenance(tmp_transcribe_env, tmp_path):
    """Success payload carries origin: transcribed + transcribe_model provenance (AC-1)."""
    import asyncio
    import shutil

    from kb_mcp.server import mcp

    src = tmp_path / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", src)

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))

    result = _parse_result(raw)
    assert result.get("origin") == "transcribed", f"Expected origin='transcribed': {result}"
    assert result.get("transcribe_model") == "gpt-5-mini-test", (
        f"Expected configured model name: {result}"
    )


def test_kb_transcribe_v1_success_payload_includes_source_and_status(tmp_transcribe_env, tmp_path):
    """Success payload includes 'source' (docs/ basename) and 'status' (created)."""
    import asyncio
    import shutil

    from kb_mcp.server import mcp

    src = tmp_path / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", src)

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))

    result = _parse_result(raw)
    assert result.get("source") == "sample_english.md", f"Unexpected source: {result}"
    assert result.get("status") == "created", f"Expected status='created': {result}"


# ---------------------------------------------------------------------------
# AC-3: hash-skip round-trips through the tool envelope
# ---------------------------------------------------------------------------


def test_kb_transcribe_v1_hash_skip_no_second_model_call(tmp_transcribe_env, tmp_path):
    """Re-transcribing unchanged bytes returns status='skipped' with no second model call."""
    import asyncio
    import shutil

    from kb_mcp.server import mcp

    src = tmp_path / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", src)

    first = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))
    assert _parse_result(first)["status"] == "created"

    second = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))
    assert not _is_error_result(second)
    result = _parse_result(second)
    assert result["status"] == "skipped", f"Expected status='skipped': {result}"
    assert tmp_transcribe_env["fake_llm"].call_count == 1, (
        "Hash-skip must not trigger a second model call"
    )


# ---------------------------------------------------------------------------
# AC-2: typed failures surface as structured tool errors with CLI/HTTP reason strings
# ---------------------------------------------------------------------------


def test_kb_transcribe_v1_rejects_nonexistent_path(tmp_transcribe_env, tmp_path):
    """A nonexistent path is rejected with isError=True and code='FileNotFoundError'."""
    import asyncio

    from kb_mcp.server import mcp

    nonexistent = tmp_path / "does_not_exist.pdf"
    assert not nonexistent.exists()

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(nonexistent)}))

    assert _is_error_result(raw), f"Expected isError=True for nonexistent path, got: {raw}"
    payload = _parse_result(raw)
    assert payload["code"] == "FileNotFoundError", f"Unexpected code: {payload}"
    assert list(tmp_transcribe_env["docs_dir"].rglob("*.md")) == [], (
        "Nothing should be written to docs/ on rejection"
    )


def test_kb_transcribe_v1_rejects_unsupported_extension(tmp_transcribe_env, tmp_path):
    """A non-.pdf file is rejected with code='UnsupportedExtension' (CLI/HTTP parity)."""
    import asyncio

    from kb_mcp.server import mcp

    src = tmp_path / "notes.txt"
    src.write_text("hello", encoding="utf-8")

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))

    assert _is_error_result(raw), f"Expected isError=True for .txt, got: {raw}"
    payload = _parse_result(raw)
    assert payload["code"] == "UnsupportedExtension", f"Unexpected code: {payload}"


def test_kb_transcribe_v1_rejects_when_unavailable(tmp_transcribe_env, tmp_path, monkeypatch):
    """Missing KB_TRANSCRIBE_ENABLED yields code='TranscribeUnavailable' (503 on HTTP)."""
    import asyncio
    import shutil

    from kb_mcp.server import mcp

    monkeypatch.delenv("KB_TRANSCRIBE_ENABLED", raising=False)
    src = tmp_path / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", src)

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))

    assert _is_error_result(raw), f"Expected isError=True when unavailable, got: {raw}"
    payload = _parse_result(raw)
    assert payload["code"] == "TranscribeUnavailable", f"Unexpected code: {payload}"
    assert tmp_transcribe_env["fake_llm"].call_count == 0, (
        "TranscribeUnavailable must be raised before any model call"
    )


def test_kb_transcribe_v1_rejects_page_limit_exceeded(tmp_transcribe_env, tmp_path, monkeypatch):
    """Page count above KB_TRANSCRIBE_MAX_PAGES yields code='TranscribePageLimitExceeded'."""
    import asyncio
    import shutil

    from kb_mcp.server import mcp

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "0")
    src = tmp_path / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", src)

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(src)}))

    assert _is_error_result(raw), f"Expected isError=True for page-limit, got: {raw}"
    payload = _parse_result(raw)
    assert payload["code"] == "TranscribePageLimitExceeded", f"Unexpected code: {payload}"
    assert tmp_transcribe_env["fake_llm"].call_count == 0, (
        "Page-limit guard must fire before any model call"
    )


def test_kb_transcribe_v1_error_payload_has_code_and_message(tmp_transcribe_env, tmp_path):
    """Error payload always carries both 'code' and 'message' keys."""
    import asyncio

    from kb_mcp.server import mcp

    nonexistent = tmp_path / "missing.pdf"

    raw = asyncio.run(mcp.call_tool("kb_transcribe_v1", {"path": str(nonexistent)}))

    payload = _parse_result(raw)
    assert "code" in payload, f"Missing 'code' in error payload: {payload}"
    assert "message" in payload, f"Missing 'message' in error payload: {payload}"
