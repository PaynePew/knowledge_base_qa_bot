"""Tests for the hot_cache deep module and the kb_read_hot_v1 / kb_save_hot_v1 MCP tools.

Unit layer: pure-Python read/write, atomic write, absent-file → empty.
Integration layer: round-trip via the FastMCP in-process harness (same dispatch
path as a real MCP host), mirroring the pattern in test_kb_search.py.

Path isolation: the conftest autouse fixture redirects HOT_PATH to tmp_path so
no test touches the real wiki/hot.md.  Each test body only needs to redirect
HOT_PATH on the hot_cache module for the write-path tests.
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
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Unit tests — hot_cache module
# ---------------------------------------------------------------------------


class TestReadHot:
    def test_returns_empty_string_when_file_absent(self, tmp_path: Path) -> None:
        """A missing hot.md returns '' — first-session empty SUCCESS, not an error."""
        import kb_mcp.hot_cache as hc

        monkeypatch_path = tmp_path / "wiki" / "hot.md"
        # File does not exist — read_hot should return ""
        result = hc.read_hot(hot_path=monkeypatch_path)
        assert result == ""

    def test_returns_file_contents_when_present(self, tmp_path: Path) -> None:
        """read_hot returns the exact bytes written by save_hot."""
        import kb_mcp.hot_cache as hc

        hot_file = tmp_path / "wiki" / "hot.md"
        hot_file.parent.mkdir(parents=True, exist_ok=True)
        hot_file.write_text("session summary text", encoding="utf-8")

        result = hc.read_hot(hot_path=hot_file)
        assert result == "session summary text"


class TestSaveHot:
    def test_write_round_trip(self, tmp_path: Path) -> None:
        """save_hot then read_hot returns the original summary."""
        import kb_mcp.hot_cache as hc

        hot_file = tmp_path / "wiki" / "hot.md"
        summary = "# Hot Cache\n\nWe were discussing refund policy.\n"

        hc.save_hot(summary, hot_path=hot_file)
        result = hc.read_hot(hot_path=hot_file)

        assert result == summary

    def test_atomic_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_hot creates parent directories if they do not exist."""
        import kb_mcp.hot_cache as hc

        hot_file = tmp_path / "deep" / "nested" / "wiki" / "hot.md"
        # Parent does not exist yet
        assert not hot_file.parent.exists()

        hc.save_hot("content", hot_path=hot_file)

        assert hot_file.exists()
        assert hot_file.read_text(encoding="utf-8") == "content"

    def test_atomic_write_overwrites_existing(self, tmp_path: Path) -> None:
        """save_hot atomically replaces an existing hot.md."""
        import kb_mcp.hot_cache as hc

        hot_file = tmp_path / "wiki" / "hot.md"
        hot_file.parent.mkdir(parents=True, exist_ok=True)
        hot_file.write_text("old content", encoding="utf-8")

        hc.save_hot("new content", hot_path=hot_file)

        assert hot_file.read_text(encoding="utf-8") == "new content"

    def test_atomic_write_uses_tmp_then_replace(self, tmp_path: Path, monkeypatch) -> None:
        """save_hot writes to a .tmp file then atomically renames — no partial writes."""
        import os

        import kb_mcp.hot_cache as hc

        hot_file = tmp_path / "wiki" / "hot.md"
        tmp_files_seen: list[str] = []
        original_replace = os.replace

        def spy_replace(src: str, dst) -> None:
            tmp_files_seen.append(str(src))
            original_replace(src, dst)

        monkeypatch.setattr(hc._loader.os, "replace", spy_replace)  # type: ignore[attr-defined]

        hc.save_hot("atomic content", hot_path=hot_file)

        assert len(tmp_files_seen) == 1
        assert tmp_files_seen[0].endswith(".tmp")
        assert hot_file.read_text(encoding="utf-8") == "atomic content"


# ---------------------------------------------------------------------------
# Integration tests — MCP harness round-trip
# ---------------------------------------------------------------------------


class TestMCPHotCacheRoundTrip:
    """Round-trip via FastMCP in-process call — same dispatch path as a real host."""

    @pytest.fixture(autouse=True)
    def _redirect_hot_path(self, tmp_path: Path, monkeypatch) -> None:
        """Redirect HOT_PATH on the hot_cache module so the MCP tools use tmp_path."""
        import kb_mcp.hot_cache as hc

        tmp_hot = tmp_path / "wiki" / "hot.md"
        monkeypatch.setattr(hc, "HOT_PATH", tmp_hot)

    def test_kb_read_hot_absent_returns_empty_success(self) -> None:
        """kb_read_hot_v1 on a fresh KB (no hot.md) returns empty string, not an error."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_read_hot_v1", {}))
        result = _parse_result(raw)

        assert result["content"] == ""
        assert result.get("error") is None

    def test_kb_save_and_read_round_trip(self) -> None:
        """kb_save_hot_v1 persists; subsequent kb_read_hot_v1 returns the same summary."""
        import asyncio

        from kb_mcp.server import mcp

        summary = "# Session\n\nWe discussed the return policy. Next: pricing FAQ.\n"
        save_raw = asyncio.run(mcp.call_tool("kb_save_hot_v1", {"summary": summary}))
        save_result = _parse_result(save_raw)
        assert save_result.get("ok") is True

        read_raw = asyncio.run(mcp.call_tool("kb_read_hot_v1", {}))
        read_result = _parse_result(read_raw)
        assert read_result["content"] == summary

    def test_kb_save_overwrites_previous(self) -> None:
        """A second kb_save_hot_v1 replaces the first; kb_read_hot_v1 returns latest."""
        import asyncio

        from kb_mcp.server import mcp

        asyncio.run(mcp.call_tool("kb_save_hot_v1", {"summary": "first version"}))
        asyncio.run(mcp.call_tool("kb_save_hot_v1", {"summary": "second version"}))

        raw = asyncio.run(mcp.call_tool("kb_read_hot_v1", {}))
        result = _parse_result(raw)
        assert result["content"] == "second version"

    def test_kb_read_hot_schema_shape(self) -> None:
        """kb_read_hot_v1 and kb_save_hot_v1 are registered tools (not resources)."""
        import asyncio

        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}

        assert "kb_read_hot_v1" in tool_names, "kb_read_hot_v1 must be a tool, not a resource"
        assert "kb_save_hot_v1" in tool_names, "kb_save_hot_v1 must be registered"
