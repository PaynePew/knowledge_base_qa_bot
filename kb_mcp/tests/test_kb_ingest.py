"""Integration tests for kb_ingest_v1 using the FastMCP in-process harness.

Issue #232 — Slice 7: MCP kb_ingest_v1 — single-source sync + progress notifications.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.

Scenarios covered:
  AC-success:          single-source ingest returns neutral result with
                       {source, pages_created, pages_overwritten, ...}.
  AC-overwrite:        pages_overwritten is populated (and non-empty) when a page
                       already exists, making cross-call slug collision visible
                       (not silent) — #54 / CODING_STANDARD §12.8.
  AC-cannot-confirm:   a grounding-failed outcome is a SUCCESS result (not
                       isError); the grounding_failed_pages list is non-empty.
  AC-llm-error:        LLMError from the ingest deep module renders as
                       isError=True with code ∈ {LLM_UNAVAILABLE, LLM_ERROR}
                       (ADR-0015).
  AC-schema:           additionalProperties:false applied (strict schema,
                       ADR-0016); no batch parameter; _v1 suffix.

Mocking follows CODING_STANDARD §11:
  - LLM mocked at ``markdown_kb.app.templates.get_ingest_llm``
    (lazy-singleton getter, not a deep entry point).
  - Grounding verifier mocked at ``markdown_kb.app.ingest.verify``
    (the name in the ingest module's own namespace after ``from .grounding import verify``).
  - ``ingest_sources`` itself is NOT mocked; we exercise the real pipeline
    with tmp_path-redirected docs/wiki dirs.

The ``_isolate_module_state`` autouse fixture in conftest.py keeps tests off
the real .kb/index.json and wiki/log.md.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers: result parsing (mirrors test_kb_ask.py pattern)
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
# Stubs: LLM / grounding
# ---------------------------------------------------------------------------


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Stub LLM that returns a minimal parseable SourceType + wiki page body."""

    # For classify_source: returns a JSON string with {"type": "concept"}
    # For generate_page / generate_entity_page: returns wiki-page-like prose.
    _type_json = '{"type": "concept"}'
    _page_body = "A stub concept page body.\n\n[Source: stub.md#stub-section]"

    def __init__(self) -> None:
        self._call_count = 0

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "_FakeLLMStructured":  # noqa: ANN401
        return _FakeLLMStructured(schema=schema)

    def invoke(self, messages: Any) -> _FakeLLMResponse:
        self._call_count += 1
        return _FakeLLMResponse(content=self._page_body)


class _FakeLLMStructured:
    """Stub returned by .with_structured_output(); produces structured Pydantic output."""

    def __init__(self, schema: Any) -> None:
        self._schema = schema

    def invoke(self, messages: Any) -> Any:
        # classify_source uses SourceTypeClassification Pydantic model
        # generate_page uses WikiPageDraft-compatible structured output
        # We always return type=concept for simplicity
        try:
            return self._schema(type="concept")  # type: ignore[call-arg]
        except Exception:
            # If schema doesn't accept type kwarg, fall back to mock
            return type("_Stub", (), {"type": "concept"})()


def _grounding_passed():
    from markdown_kb.app.grounding import GroundingOutcome

    return GroundingOutcome(passed=True, reason="claim_supported", retries_attempted=0)


def _grounding_failed():
    from markdown_kb.app.grounding import GroundingOutcome

    return GroundingOutcome(passed=False, reason="claim_unsupported", retries_attempted=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    """A tmp docs/ directory with one minimal Source."""
    d = tmp_path / "docs"
    d.mkdir(parents=True)
    source = d / "stub_source.md"
    source.write_text(
        "# Stub Section\n\nThis is stub content for the test Source.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def wiki_dir(tmp_path: Path) -> Path:
    """A tmp wiki/ directory (empty initially)."""
    w = tmp_path / "wiki"
    w.mkdir(parents=True)
    (w / "concepts").mkdir(parents=True)
    (w / "entities").mkdir(parents=True)
    return w


@pytest.fixture(autouse=True)
def _patch_ingest_paths(monkeypatch: pytest.MonkeyPatch, docs_dir: Path, wiki_dir: Path) -> None:
    """Redirect ingest deep module paths to tmp dirs so no real disk I/O occurs."""
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.wiki_writer as wiki_writer_mod

    monkeypatch.setattr(ingest_mod, "_resolve_docs_files", lambda d: [])
    # Redirect DOCS_DIR so single-source lookups go to tmp
    import markdown_kb.app._paths as paths_mod

    monkeypatch.setattr(paths_mod, "DOCS_DIR", docs_dir)

    # Redirect indexer WIKI_DIR to tmp
    import markdown_kb.app.indexer as indexer_mod

    monkeypatch.setattr(indexer_mod, "WIKI_DIR", wiki_dir)


# ---------------------------------------------------------------------------
# AC-schema: tool exists with _v1 suffix, additionalProperties:false,
#            no batch parameter, and the correct description keywords.
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_tool_registered():
    """kb_ingest_v1 is registered in the FastMCP tool manager."""
    from kb_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("kb_ingest_v1")
    assert tool is not None, "kb_ingest_v1 tool not found in server"


def test_kb_ingest_v1_strict_schema():
    """kb_ingest_v1 schema has additionalProperties:false (ADR-0016)."""
    from kb_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("kb_ingest_v1")
    assert tool is not None
    assert tool.parameters.get("additionalProperties") is False, (
        f"Expected additionalProperties:false, got: {tool.parameters}"
    )


def test_kb_ingest_v1_no_batch_param():
    """kb_ingest_v1 does not expose a 'sources' batch parameter (single-source only)."""
    from kb_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("kb_ingest_v1")
    assert tool is not None
    props = tool.parameters.get("properties", {})
    assert "sources" not in props, (
        "Batch 'sources' parameter must NOT be exposed on kb_ingest_v1"
    )


def test_kb_ingest_v1_has_source_param():
    """kb_ingest_v1 has exactly a 'source' parameter (single Source name)."""
    from kb_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("kb_ingest_v1")
    assert tool is not None
    props = tool.parameters.get("properties", {})
    assert "source" in props, f"Expected 'source' param, got props: {props}"


# ---------------------------------------------------------------------------
# AC-success: single-source ingest returns neutral result with
#             {source, pages_created, pages_overwritten, ...}
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_success_result_shape(docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """kb_ingest_v1 returns a success result with required keys."""
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda draft, sections: _grounding_passed())

    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))

    assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
    result = _parse_result(raw)

    assert "source" in result, f"Missing 'source' key: {result}"
    assert "pages_created" in result, f"Missing 'pages_created' key: {result}"
    assert "pages_overwritten" in result, f"Missing 'pages_overwritten' key: {result}"


def test_kb_ingest_v1_success_source_name_matches(docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Result.source matches the requested source filename."""
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda draft, sections: _grounding_passed())

    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))

    result = _parse_result(raw)
    assert result["source"] == "stub_source.md", (
        f"Expected source='stub_source.md', got: {result['source']}"
    )


# ---------------------------------------------------------------------------
# AC-overwrite: pages_overwritten surfaces cross-call slug collision (#54)
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_overwrite_surfaced(docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """pages_overwritten is non-empty when a page already exists on disk.

    This test pre-creates a wiki page at the expected slug location to simulate
    a cross-call slug collision, then re-ingests the same Source. The tool
    result must surface the overwrite (pages_overwritten non-empty) so the
    collision is visible, not silent (#54 / CODING_STANDARD §12.8).
    """
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda draft, sections: _grounding_passed())

    # First ingest — creates the page
    raw1 = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))
    assert not _is_error_result(raw1), f"First ingest failed: {raw1}"
    result1 = _parse_result(raw1)
    assert result1["pages_created"], f"Expected pages_created to be non-empty: {result1}"

    # Second ingest of the same source — must overwrite, not silently collide
    raw2 = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))
    assert not _is_error_result(raw2), f"Second ingest failed: {raw2}"
    result2 = _parse_result(raw2)
    # pages_overwritten must be non-empty on the second run (page already exists)
    assert result2["pages_overwritten"], (
        f"Expected pages_overwritten to be non-empty on re-ingest: {result2}"
    )


# ---------------------------------------------------------------------------
# AC-cannot-confirm: grounding failure is a success result, NOT isError
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_grounding_failed_is_success(docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """A grounding-failed outcome is a SUCCESS result (not isError) — ADR-0015."""
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda draft, sections: _grounding_failed())

    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))

    assert not _is_error_result(raw), (
        "Grounding-failed ingest must be a SUCCESS result (isError=False), "
        f"got isError result: {raw}"
    )


def test_kb_ingest_v1_grounding_failed_pages_surfaced(docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Grounding-failed pages appear in grounding_failed_pages list."""
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda draft, sections: _grounding_failed())

    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))
    result = _parse_result(raw)

    assert "grounding_failed_pages" in result, f"Missing 'grounding_failed_pages' key: {result}"
    assert result["grounding_failed_pages"], (
        f"Expected grounding_failed_pages to be non-empty: {result}"
    )


# ---------------------------------------------------------------------------
# AC-llm-error: LLMError → isError=True with code ∈ {LLM_UNAVAILABLE, LLM_ERROR}
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_llm_unavailable_iserror(docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """LLMError(retryable=True) from ingest → isError=True, code=LLM_UNAVAILABLE."""
    import markdown_kb.app.templates as templates_mod
    from markdown_kb.app.errors import LLMError

    from kb_mcp.server import mcp

    def _raise_unavailable() -> None:
        raise LLMError(retryable=True, message="LLM service temporarily unavailable.")

    monkeypatch.setattr(templates_mod, "get_ingest_llm", _raise_unavailable)

    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))

    assert _is_error_result(raw), f"Expected isError=True for LLMError, got: {raw}"
    payload = _parse_result(raw)
    assert payload.get("code") == "LLM_UNAVAILABLE", (
        f"Expected code=LLM_UNAVAILABLE, got: {payload}"
    )
    assert "message" in payload, f"Missing 'message' key: {payload}"


def test_kb_ingest_v1_llm_error_non_retryable_iserror(docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """LLMError(retryable=False) from ingest → isError=True, code=LLM_ERROR."""
    import markdown_kb.app.templates as templates_mod
    from markdown_kb.app.errors import LLMError

    from kb_mcp.server import mcp

    def _raise_error() -> None:
        raise LLMError(retryable=False, message="LLM authentication failed.")

    monkeypatch.setattr(templates_mod, "get_ingest_llm", _raise_error)

    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))

    assert _is_error_result(raw), f"Expected isError=True for LLMError, got: {raw}"
    payload = _parse_result(raw)
    assert payload.get("code") == "LLM_ERROR", (
        f"Expected code=LLM_ERROR, got: {payload}"
    )
    assert "message" in payload, f"Missing 'message' key: {payload}"
