"""Integration tests for kb_source_trash_v1 (issue #606, ADR-0041).

Following the pattern in ``test_kb_index_lint.py``: tests call the tool via
``mcp.call_tool()`` (the same path as a real MCP host), driving the REAL
``markdown_kb.app.source_lifecycle.list_trash`` — no LLM involved, nothing to
mock (§6.3).

Path isolation: ``source_lifecycle.TRASH_DIR`` is a module-level binding
imported from ``._paths`` and is NOT redirected by this suite's autouse
``_isolate_module_state`` conftest fixture (which only covers indexer/
freshness/hot-cache/lint paths — issue #200/#202/#231). Each test that
exercises a non-empty trash therefore monkeypatches ``source_lifecycle.
TRASH_DIR`` directly to ``tmp_path``, mirroring ``gateway/tests/
test_source_lifecycle_routes.py``'s ``sources_env`` fixture.

Also pins ADR-0041's own MCP Invariant ("MCP exposes no Source-lifecycle
write verb") mechanically, mirroring ``test_no_gate_resolving_tools.py``'s
ADR-0026 guard.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result."""
    if isinstance(raw, CallToolResult):
        return json.loads(raw.content[0].text)
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


def _is_error_result(raw: Any) -> bool:
    return isinstance(raw, CallToolResult) and raw.isError is True


# ---------------------------------------------------------------------------
# Schema / registration
# ---------------------------------------------------------------------------


class TestKbSourceTrashV1Schema:
    def test_tool_registered(self):
        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        names = [t.name for t in tools]
        assert "kb_source_trash_v1" in names, f"kb_source_trash_v1 not in registered tools: {names}"

    def test_strict_schema(self):
        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool = next((t for t in tools if t.name == "kb_source_trash_v1"), None)
        assert tool is not None
        schema = tool.inputSchema
        assert schema.get("additionalProperties") is False, (
            "additionalProperties:false missing from kb_source_trash_v1 schema"
        )

    def test_no_required_parameters(self):
        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool = next((t for t in tools if t.name == "kb_source_trash_v1"), None)
        assert tool is not None
        required = tool.inputSchema.get("required", [])
        assert required == [], f"Expected no required params, got: {required}"


# ---------------------------------------------------------------------------
# Functional — real list_trash, TRASH_DIR monkeypatched per test
# ---------------------------------------------------------------------------


class TestKbSourceTrashV1Success:
    def test_empty_trash_returns_empty_list(self, tmp_path: Path, monkeypatch):
        import markdown_kb.app.source_lifecycle as sl_module

        from kb_mcp.server import mcp

        monkeypatch.setattr(sl_module, "TRASH_DIR", tmp_path / ".trash")

        raw = asyncio.run(mcp.call_tool("kb_source_trash_v1", {}))
        assert not _is_error_result(raw), f"Expected success, got isError: {raw}"
        result = _parse_result(raw)
        assert result == {"entries": []}

    def test_missing_trash_dir_returns_empty_list(self, tmp_path: Path, monkeypatch):
        """No Source has ever been retired: TRASH_DIR does not exist on disk."""
        import markdown_kb.app.source_lifecycle as sl_module

        from kb_mcp.server import mcp

        monkeypatch.setattr(sl_module, "TRASH_DIR", tmp_path / "never-created" / ".trash")

        raw = asyncio.run(mcp.call_tool("kb_source_trash_v1", {}))
        assert not _is_error_result(raw)
        assert _parse_result(raw) == {"entries": []}

    def test_returns_real_trash_entry(self, tmp_path: Path, monkeypatch):
        """A real retired Source on disk is reported with its timestamp + relpath."""
        import markdown_kb.app.source_lifecycle as sl_module

        from kb_mcp.server import mcp

        trash_dir = tmp_path / ".trash"
        entry_dir = trash_dir / "20260101T000000000000Z" / "docs"
        entry_dir.mkdir(parents=True)
        (entry_dir / "policy.md").write_text("# Policy\n\nBody.\n", encoding="utf-8")
        monkeypatch.setattr(sl_module, "TRASH_DIR", trash_dir)

        raw = asyncio.run(mcp.call_tool("kb_source_trash_v1", {}))
        result = _parse_result(raw)
        assert result == {
            "entries": [{"timestamp": "20260101T000000000000Z", "relpath": "policy.md"}]
        }

    def test_multiple_entries_sorted_by_timestamp_then_relpath(self, tmp_path: Path, monkeypatch):
        import markdown_kb.app.source_lifecycle as sl_module

        from kb_mcp.server import mcp

        trash_dir = tmp_path / ".trash"
        for ts, relpath in (
            ("20260102T000000000000Z", "b.md"),
            ("20260101T000000000000Z", "a.md"),
        ):
            entry_dir = trash_dir / ts / "docs"
            entry_dir.mkdir(parents=True)
            (entry_dir / relpath).write_text("# X\n\nBody.\n", encoding="utf-8")
        monkeypatch.setattr(sl_module, "TRASH_DIR", trash_dir)

        raw = asyncio.run(mcp.call_tool("kb_source_trash_v1", {}))
        result = _parse_result(raw)
        assert result["entries"] == [
            {"timestamp": "20260101T000000000000Z", "relpath": "a.md"},
            {"timestamp": "20260102T000000000000Z", "relpath": "b.md"},
        ]


# ---------------------------------------------------------------------------
# ADR-0041 Invariant: MCP exposes no Source-lifecycle write verb
# ---------------------------------------------------------------------------


def test_no_source_lifecycle_write_tool_registered():
    """No MCP tool name contains a Source-lifecycle write verb (retire/restore/
    rename) — ADR-0041's own Invariant, distinct from (and additional to)
    ADR-0026's gate-resolving-verb guard in test_no_gate_resolving_tools.py."""
    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    normalized = [t.name.lower().replace("_", "") for t in tools]
    write_verbs = ("retire", "restore", "rename")
    offending = [
        t.name
        for t, norm in zip(tools, normalized, strict=True)
        if any(v in norm for v in write_verbs)
    ]
    assert offending == [], (
        f"Found MCP tool(s) that appear to write a Source-lifecycle verb: {offending}. "
        "ADR-0041 Invariant: MCP exposes no Source-lifecycle write verb — "
        "retire/restore/rename are Console/CLI-only."
    )
