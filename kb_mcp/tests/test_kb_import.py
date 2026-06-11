"""Integration tests for kb_import_v1 using the FastMCP in-process harness.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.  This exercises FastMCP's argument
validation and the full dispatch path.

Mocking strategy
----------------
``import_path`` from ``markdown_kb.app.importer`` is the single deep entry
point (ADR-0017 / issue #227).  Tests redirect ``markdown_kb.app.importer.RAW_DIR``
and ``markdown_kb.app.importer.DOCS_DIR`` to tmp dirs so writes land in hermetic
locations, then inspect the written files directly.

CRITICAL: do NOT mock ``import_path`` or the converters.  Import has no LLM;
exercise the REAL conversion and assert the REAL ``docs/`` Source result.
Mocking the deep-module entry point is a FAIL (CODING_STANDARD §6.3/§11,
implement.md §3.1).

AC reference (issue #233)
--------------------------
AC-1  kb_import_v1(path) reads a local file and converts it to a docs/ Source
      via the Import deep module from #227.
AC-2  A traversal-unsafe / nonexistent path is rejected with a structured error;
      nothing is written outside raw/ / docs/.
AC-3  The tool carries a strict schema (additionalProperties:false, ADR-0016)
      and the _v1 suffix, and is added to the strict-schema registry.
AC-4  Hermetic tests via the FastMCP in-process harness cover a successful
      import and the unsafe-path rejection (prior art: kb_mcp/tests/test_kb_ask.py).
AC-5  .kb/index.json / wiki/ byte-stability respected; tests write only to tmp.
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
def tmp_import_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Redirect importer RAW_DIR and DOCS_DIR to hermetic tmp directories.

    This ensures kb_import_v1 writes only to tmp, never to the real
    raw/ or docs/ directories (AC-5).
    """
    import markdown_kb.app.importer as importer_mod

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(importer_mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_mod, "DOCS_DIR", docs_dir)
    return raw_dir, docs_dir


# ---------------------------------------------------------------------------
# AC-3: tool registration and strict schema
# ---------------------------------------------------------------------------


def test_kb_import_v1_tool_registered():
    """kb_import_v1 is registered as a tool in the MCP server."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = [t.name for t in tools]
    assert "kb_import_v1" in names, f"kb_import_v1 not in tool list: {names}"


def test_kb_import_v1_schema_additional_properties_false():
    """kb_import_v1 schema has additionalProperties:false (ADR-0016)."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_import_v1"), None)
    assert tool is not None, "kb_import_v1 not registered"

    schema = tool.inputSchema
    assert schema.get("additionalProperties") is False, (
        f"additionalProperties:false missing from schema: {schema}"
    )


def test_kb_import_v1_schema_required_fields():
    """kb_import_v1 schema requires 'path'."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_import_v1"), None)
    assert tool is not None, "kb_import_v1 not registered"

    schema = tool.inputSchema
    required = schema.get("required", [])
    assert "path" in required, f"'path' not in required: {required}"


def test_kb_import_v1_schema_properties_present():
    """kb_import_v1 schema exposes a 'path' property."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool = next((t for t in tools if t.name == "kb_import_v1"), None)
    assert tool is not None, "kb_import_v1 not registered"

    props = tool.inputSchema.get("properties", {})
    assert "path" in props, f"'path' not in properties: {props}"


# ---------------------------------------------------------------------------
# AC-1: successful import — real conversion via import_path deep module
# ---------------------------------------------------------------------------


def test_kb_import_v1_success_txt_returns_ok(tmp_import_dirs, tmp_path):
    """kb_import_v1 returns a neutral {ok, source, status} payload on success (.txt)."""
    import asyncio

    from kb_mcp.server import mcp

    src = tmp_path / "hello.txt"
    src.write_text("Hello world.\n", encoding="utf-8")

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
    result = _parse_result(raw)
    assert result.get("ok") is True, f"Expected ok=true, got: {result}"


def test_kb_import_v1_success_txt_writes_docs_source(tmp_import_dirs, tmp_path):
    """kb_import_v1 writes a real docs/ Source for a .txt file (REAL conversion)."""
    import asyncio

    from kb_mcp.server import mcp

    _raw_dir, docs_dir = tmp_import_dirs
    src = tmp_path / "sample.txt"
    src.write_text("Sample content for the KB.\n", encoding="utf-8")

    asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    docs_file = docs_dir / "sample.md"
    assert docs_file.exists(), (
        f"Expected docs/ Source at {docs_file}; docs/ contains: {list(docs_dir.iterdir())}"
    )


def test_kb_import_v1_success_txt_source_has_provenance(tmp_import_dirs, tmp_path):
    """Real import_path stamps imported_from / original_format / imported_at frontmatter."""
    import asyncio

    from kb_mcp.server import mcp

    _raw_dir, docs_dir = tmp_import_dirs
    src = tmp_path / "prov.txt"
    src.write_text("Provenance test content.\n", encoding="utf-8")

    asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    text = (docs_dir / "prov.md").read_text(encoding="utf-8")
    assert "imported_from:" in text, f"imported_from missing: {text[:300]}"
    assert "original_format: txt" in text, f"original_format missing: {text[:300]}"
    assert "imported_at:" in text, f"imported_at missing: {text[:300]}"
    assert "content_sha256:" in text, f"content_sha256 missing: {text[:300]}"


def test_kb_import_v1_success_html_converts_to_markdown(tmp_import_dirs, tmp_path):
    """kb_import_v1 for .html runs REAL markdownify conversion and writes a docs/ Source."""
    import asyncio

    from kb_mcp.server import mcp

    _raw_dir, docs_dir = tmp_import_dirs
    src = tmp_path / "page.html"
    src.write_text(
        "<html><body><h1>Title</h1><p>Some paragraph.</p></body></html>",
        encoding="utf-8",
    )

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    assert not _is_error_result(raw), f"Expected success, got isError: {raw}"
    docs_file = docs_dir / "page.md"
    assert docs_file.exists(), f"docs/page.md not written; docs/: {list(docs_dir.iterdir())}"
    text = docs_file.read_text(encoding="utf-8")
    # Real markdownify should produce heading and paragraph content
    assert "Title" in text, f"Heading not found in converted output: {text[:300]}"


def test_kb_import_v1_success_payload_includes_source_key(tmp_import_dirs, tmp_path):
    """kb_import_v1 success payload includes 'source' (the docs/ filename)."""
    import asyncio

    from kb_mcp.server import mcp

    src = tmp_path / "meta.txt"
    src.write_text("Meta content.\n", encoding="utf-8")

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    result = _parse_result(raw)
    assert "source" in result, f"Expected 'source' in result: {result}"


def test_kb_import_v1_success_payload_includes_status_key(tmp_import_dirs, tmp_path):
    """kb_import_v1 success payload includes 'status' (created/updated/skipped)."""
    import asyncio

    from kb_mcp.server import mcp

    src = tmp_path / "status_test.txt"
    src.write_text("Status test.\n", encoding="utf-8")

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    result = _parse_result(raw)
    assert "status" in result, f"Expected 'status' in result: {result}"
    assert result["status"] in ("created", "updated", "skipped"), (
        f"Unexpected status value: {result['status']!r}"
    )


def test_kb_import_v1_success_status_is_created_for_new_file(tmp_import_dirs, tmp_path):
    """Status is 'created' when no docs/ Source existed before."""
    import asyncio

    from kb_mcp.server import mcp

    src = tmp_path / "brand_new.txt"
    src.write_text("Brand new content.\n", encoding="utf-8")

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    result = _parse_result(raw)
    assert result["status"] == "created", f"Expected status='created', got: {result['status']!r}"


# ---------------------------------------------------------------------------
# AC-2: unsafe / nonexistent path rejected — structured error, nothing written
# ---------------------------------------------------------------------------


def test_kb_import_v1_rejects_nonexistent_path(tmp_import_dirs, tmp_path):
    """kb_import_v1 rejects a path that does not exist; returns isError."""
    import asyncio

    from kb_mcp.server import mcp

    nonexistent = tmp_path / "does_not_exist.txt"
    # Ensure it really does not exist
    assert not nonexistent.exists()

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(nonexistent)},
        )
    )

    assert _is_error_result(raw), f"Expected isError=True for nonexistent path, got: {raw}"


def test_kb_import_v1_rejects_nonexistent_path_error_payload(tmp_import_dirs, tmp_path):
    """Nonexistent path rejection returns {code, message} payload."""
    import asyncio

    from kb_mcp.server import mcp

    nonexistent = tmp_path / "missing.txt"

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(nonexistent)},
        )
    )

    payload = _parse_result(raw)
    assert "code" in payload, f"Missing 'code' in error payload: {payload}"
    assert "message" in payload, f"Missing 'message' in error payload: {payload}"
    assert payload["code"] == "IMPORT_REJECTED", (
        f"Expected code='IMPORT_REJECTED', got: {payload.get('code')!r}"
    )


def test_kb_import_v1_rejects_traversal_basename(tmp_import_dirs, tmp_path):
    """kb_import_v1 rejects a path whose basename contains '#' (traversal-unsafe per importer)."""
    import asyncio

    from kb_mcp.server import mcp

    # Create a file whose basename has a '#' which importer._validate_filename rejects
    # We can't actually create a file named 'bad#name.txt' on Windows, so simulate
    # by testing with a path containing ':' (Windows reserved)
    _raw_dir, docs_dir = tmp_import_dirs

    # Use a path with backslash separator in the effective name passed — the
    # real guard is in import_path's _validate_filename. To exercise it, create a
    # file whose name contains a colon (rejected by _validate_filename) using
    # a monkeypatched name, or simply pass a non-existent path with bad chars.
    # On Windows we cannot create 'bad:name.txt', so we pass as string directly.
    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            # absolute path with a colon in the filename part (rejected by importer)
            {"path": str(tmp_path / "bad:name.txt")},
        )
    )

    assert _is_error_result(raw), f"Expected isError=True for basename with ':', got: {raw}"
    # Nothing written to docs/
    written = list(docs_dir.rglob("*.md"))
    assert written == [], f"Files written despite rejection: {written}"


def test_kb_import_v1_rejects_unsupported_extension(tmp_import_dirs, tmp_path):
    """kb_import_v1 rejects files with unsupported extensions (e.g., .docx)."""
    import asyncio

    from kb_mcp.server import mcp

    _raw_dir, docs_dir = tmp_import_dirs
    src = tmp_path / "document.docx"
    src.write_bytes(b"fake docx content")

    raw = asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(src)},
        )
    )

    assert _is_error_result(raw), (
        f"Expected isError=True for unsupported .docx extension, got: {raw}"
    )
    written = list(docs_dir.rglob("*.md"))
    assert written == [], f"Files written despite rejection: {written}"


def test_kb_import_v1_nothing_written_on_nonexistent_path(tmp_import_dirs, tmp_path):
    """Nothing is written to docs/ when the import is rejected for a missing file."""
    import asyncio

    from kb_mcp.server import mcp

    _raw_dir, docs_dir = tmp_import_dirs
    nonexistent = tmp_path / "ghost.txt"

    asyncio.run(
        mcp.call_tool(
            "kb_import_v1",
            {"path": str(nonexistent)},
        )
    )

    written = list(docs_dir.rglob("*.md"))
    assert written == [], f"Files written despite missing-file rejection: {written}"
