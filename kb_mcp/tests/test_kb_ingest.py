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


class _FakeLLMStructured:
    """Stub returned by .with_structured_output(); produces structured Pydantic output.

    Handles both _ClassifierOutput (needs type='concept') and
    _PageSynthesisOutput (needs body+open_questions).
    Identifies the schema by its field names.
    """

    def __init__(self, schema: Any) -> None:
        self._schema = schema

    def invoke(self, messages: Any) -> Any:
        schema = self._schema
        # Determine which schema by checking the model_fields attribute
        fields = set(getattr(schema, "model_fields", {}).keys())
        if "type" in fields and "body" not in fields:
            # _ClassifierOutput
            return schema(type="concept")
        elif "body" in fields:
            # _PageSynthesisOutput
            return schema(
                body="A stub concept page body. This is the synthesised content.",
                open_questions=[],
            )
        else:
            # Fallback: try type= first
            try:
                return schema(type="concept")
            except Exception:
                return schema()


class _FakeLLM:
    """Stub LLM that returns minimal valid structured output for ingest pipeline.

    Supports:
      - with_structured_output(schema): for classify_source, generate_page,
        generate_entity_page — returns a _FakeLLMStructured chain.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _FakeLLMStructured:  # noqa: ANN401
        return _FakeLLMStructured(schema=schema)


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
    """Redirect ingest deep module paths to tmp dirs so no real disk I/O occurs.

    Redirects DOCS_DIR (used by ingest_sources for source lookup) and WIKI_DIR
    (used by indexer / wiki_writer for output pages) to tmp_path subdirs.
    Also redirects the logger LOG_PATH to tmp to avoid touching wiki/log.md.

    Note: the conftest _isolate_module_state autouse fixture already redirects
    indexer.WIKI_DIR; we redirect it again here to point to our specific wiki_dir
    fixture (conftest uses tmp_path / 'wiki' which is the same path since we
    build wiki_dir from tmp_path too, but we do it explicitly for clarity).
    """
    import markdown_kb.app._paths as paths_mod
    import markdown_kb.app.indexer as indexer_mod
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.logger as logger_mod

    # DOCS_DIR: ingest_sources builds source_path as docs_dir / source_name
    monkeypatch.setattr(paths_mod, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(ingest_mod, "DOCS_DIR", docs_dir)

    # WIKI_DIR: used by ingest to look up existing pages + write output
    monkeypatch.setattr(indexer_mod, "WIKI_DIR", wiki_dir)

    # LOG_PATH: log_event appends to wiki/log.md; redirect to tmp
    monkeypatch.setattr(logger_mod, "LOG_PATH", wiki_dir / "log.md")


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
    assert "sources" not in props, "Batch 'sources' parameter must NOT be exposed on kb_ingest_v1"


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


def test_kb_ingest_v1_success_result_shape(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
):
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


def test_kb_ingest_v1_success_source_name_matches(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
):
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


def test_kb_ingest_v1_overwrite_surfaced(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """pages_overwritten is non-empty when a page already exists on disk.

    Simulates a cross-call slug collision by pre-creating a wiki concept page
    at the slug the ingest pipeline will produce for the stub Source's section.
    Because the pre-existing page has a DIFFERENT docs_body_hash (it carries
    the hash of a different earlier Source), the hash-skip logic does NOT fire —
    the page is overwritten and pages_overwritten is populated (#54 / §12.8).
    """
    import markdown_kb.app.indexer as indexer_mod
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda draft, sections: _grounding_passed())

    # Pre-create a concept page at the slug the pipeline will produce.
    # The stub source has heading "Stub Section" → slug "stub-section".
    # Write it with a SOURCE hash that does NOT match stub_source.md's hash,
    # so the hash-skip logic treats it as "unknown drift" and proceeds to overwrite.
    slug = "stub-section"
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    preexisting_page = concepts_dir / f"{slug}.md"
    preexisting_page.write_text(
        "---\n"
        f"id: {slug}\n"
        "type: concept\n"
        "created: '2020-01-01T00:00:00Z'\n"
        "updated: '2020-01-01T00:00:00Z'\n"
        "sources:\n"
        "  - other_source.md#stub-section\n"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes:\n"
        "  other_source.md:\n"
        "    raw: null\n"
        "    docs_body: aaaa1111bbbb2222cccc3333\n"
        "---\n\n"
        "# Stub Section\n\nPre-existing page from a different source.\n\n"
        "[Source: other_source.md#stub-section]\n",
        encoding="utf-8",
    )

    # Ingest stub_source.md — the pipeline will see the pre-existing page
    # (different source hash → NOT skipped) and overwrite it.
    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_source.md"}))
    assert not _is_error_result(raw), f"Ingest failed: {raw}"
    result = _parse_result(raw)
    # The page was pre-existing → pages_overwritten must be non-empty
    assert result["pages_overwritten"], (
        f"Expected pages_overwritten to be non-empty when page pre-existed: {result}"
    )


# ---------------------------------------------------------------------------
# AC-cannot-confirm: grounding failure is a success result, NOT isError
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_grounding_failed_is_success(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
):
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


def test_kb_ingest_v1_grounding_failed_pages_surfaced(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
):
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


def test_kb_ingest_v1_llm_unavailable_iserror(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
):
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


def test_kb_ingest_v1_llm_error_non_retryable_iserror(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
):
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
    assert payload.get("code") == "LLM_ERROR", f"Expected code=LLM_ERROR, got: {payload}"
    assert "message" in payload, f"Missing 'message' key: {payload}"
