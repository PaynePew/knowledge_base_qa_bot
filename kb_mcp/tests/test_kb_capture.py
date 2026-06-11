"""Integration tests for kb_capture_v1 using the FastMCP in-process harness.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.  This exercises FastMCP's argument
validation and the full dispatch path.

Mocking strategy
----------------
``capture_source`` from ``markdown_kb.app.capture`` is the single deep entry
point.  Tests monkeypatch ``markdown_kb.app.capture.DOCS_DIR`` to a tmp dir
so writes land in a hermetic location, then inspect the written file directly.
No LLM is involved — Capture skips Import entirely.

AC reference (issue #230)
--------------------------
AC-4  kb_capture_v1 tool: strict schema (additionalProperties:false, ADR-0016),
      _v1 suffix, neutral success payload
AC-5  hermetic tests cover the tool via the FastMCP in-process harness
AC-6  .kb/index.json / wiki/ byte-stability; tests write only to tmp
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the capture module's DOCS_DIR to a hermetic tmp directory.

    This ensures ``kb_capture_v1`` writes only to tmp, never to the real
    ``docs/`` directory (AC-6).
    """
    import markdown_kb.app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)
    return docs_dir


# ---------------------------------------------------------------------------
# AC-4a: tool is registered with _v1 suffix
# ---------------------------------------------------------------------------


def test_kb_capture_v1_tool_registered():
    """kb_capture_v1 is registered as a tool in the MCP server."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = [t.name for t in tools]
    assert "kb_capture_v1" in names, f"kb_capture_v1 not in tool list: {names}"


# ---------------------------------------------------------------------------
# AC-4b: strict schema — additionalProperties:false and required fields
# ---------------------------------------------------------------------------


def test_kb_capture_v1_schema_additional_properties_false():
    """kb_capture_v1 schema has additionalProperties:false (ADR-0016)."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_capture_v1"), None)
    assert tool is not None, "kb_capture_v1 not registered"

    schema = tool.inputSchema
    assert schema.get("additionalProperties") is False, (
        f"additionalProperties:false missing from schema: {schema}"
    )


def test_kb_capture_v1_schema_required_fields():
    """kb_capture_v1 schema requires both filename and content."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_capture_v1"), None)
    assert tool is not None, "kb_capture_v1 not registered"

    schema = tool.inputSchema
    required = schema.get("required", [])
    assert "filename" in required, f"'filename' not in required: {required}"
    assert "content" in required, f"'content' not in required: {required}"


def test_kb_capture_v1_schema_properties_present():
    """kb_capture_v1 schema exposes filename and content properties."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_capture_v1"), None)
    assert tool is not None, "kb_capture_v1 not registered"

    props = tool.inputSchema.get("properties", {})
    assert "filename" in props, f"'filename' not in properties: {props}"
    assert "content" in props, f"'content' not in properties: {props}"


# ---------------------------------------------------------------------------
# AC-4c: neutral success payload
# ---------------------------------------------------------------------------


def test_kb_capture_v1_success_returns_ok(tmp_docs):
    """kb_capture_v1 returns a neutral {ok: true, path: str} payload on success."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(
        mcp.call_tool(
            "kb_capture_v1",
            {"filename": "test_note.md", "content": "# Test\n\nSome content.\n"},
        )
    )

    assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
    result = _parse_result(raw)
    assert result.get("ok") is True, f"Expected ok=true, got: {result}"


def test_kb_capture_v1_success_returns_path(tmp_docs):
    """kb_capture_v1 success payload includes the path of the written file."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(
        mcp.call_tool(
            "kb_capture_v1",
            {"filename": "my_capture.md", "content": "# Capture\n\nContent.\n"},
        )
    )

    result = _parse_result(raw)
    assert "path" in result, f"Expected 'path' in result: {result}"
    assert "my_capture.md" in result["path"], f"Expected filename in path: {result['path']!r}"


# ---------------------------------------------------------------------------
# AC-1 (via MCP): provenance frontmatter stamped in the written file
# ---------------------------------------------------------------------------


def test_kb_capture_v1_writes_file_to_docs(tmp_docs):
    """kb_capture_v1 writes the file into docs_dir."""
    import asyncio

    from kb_mcp.server import mcp

    asyncio.run(
        mcp.call_tool(
            "kb_capture_v1",
            {"filename": "written_note.md", "content": "# Written\n\nHello.\n"},
        )
    )

    assert (tmp_docs / "written_note.md").exists(), (
        f"File not found in {tmp_docs}: {list(tmp_docs.iterdir())}"
    )


def test_kb_capture_v1_stamps_provenance_frontmatter(tmp_docs):
    """kb_capture_v1 stamps all three mandatory provenance frontmatter keys."""
    import asyncio
    import re

    from kb_mcp.server import mcp

    asyncio.run(
        mcp.call_tool(
            "kb_capture_v1",
            {"filename": "prov_note.md", "content": "# Provenance\n\nBody.\n"},
        )
    )

    text = (tmp_docs / "prov_note.md").read_text(encoding="utf-8")
    assert "origin: mcp-conversation" in text, f"origin missing: {text[:200]}"
    assert "authored_by: agent" in text, f"authored_by missing: {text[:200]}"
    assert re.search(r"created_at: \d{4}-\d{2}-\d{2}T", text), f"created_at missing: {text[:200]}"


# ---------------------------------------------------------------------------
# AC-2 (via MCP): unsafe filenames rejected
# ---------------------------------------------------------------------------


def test_kb_capture_v1_rejects_traversal_filename(tmp_docs):
    """kb_capture_v1 rejects a path-traversal filename; nothing is written."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(
        mcp.call_tool(
            "kb_capture_v1",
            {"filename": "../evil.md", "content": "malicious content"},
        )
    )

    assert _is_error_result(raw), f"Expected isError=True for traversal filename, got: {raw}"
    # Nothing should have been written
    written = list(tmp_docs.rglob("*.md"))
    assert written == [], f"Files written despite traversal attempt: {written}"


def test_kb_capture_v1_rejects_slash_filename(tmp_docs):
    """kb_capture_v1 rejects a filename with path separators."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(
        mcp.call_tool(
            "kb_capture_v1",
            {"filename": "sub/dir/note.md", "content": "some content"},
        )
    )

    assert _is_error_result(raw), f"Expected isError=True for slash filename, got: {raw}"


def test_kb_capture_v1_reject_error_payload_shape(tmp_docs):
    """Rejection returns isError=True with a {code, message} payload."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(
        mcp.call_tool(
            "kb_capture_v1",
            {"filename": "../evil.md", "content": "bad"},
        )
    )

    assert _is_error_result(raw)
    payload = _parse_result(raw)
    assert "code" in payload, f"Missing 'code' in error payload: {payload}"
    assert "message" in payload, f"Missing 'message' in error payload: {payload}"
    assert payload["code"] == "CAPTURE_REJECTED", (
        f"Expected code='CAPTURE_REJECTED', got: {payload.get('code')!r}"
    )
