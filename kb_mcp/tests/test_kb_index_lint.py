"""Integration tests for kb_index_v1 and kb_lint_v1 using FastMCP in-process harness.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.  This exercises FastMCP's argument
validation and the full dispatch path.

Three scenarios:
  kb_index_v1:  rebuilds the Section Index; returns {files_indexed, sections_indexed}.
  kb_lint_v1 success: runs lint; returns the structured LintResponse payload as a
                neutral JSON dict.
  kb_lint_v1 LLMError (C5): LLMError from the C5 check renders as isError=True
                with code in {LLM_UNAVAILABLE, LLM_ERROR} (ADR-0015).

Mocking follows the project pattern (CODING_STANDARD section 11 / implement.md section 3.1):
  - build_index mocked at ``markdown_kb.app.indexer.build_index`` leaf.
  - run_lint mocked at ``markdown_kb.app.lint.run_lint`` leaf.

The ``_isolate_module_state`` autouse fixture in conftest.py provides module-state
isolation; no duplication of path-redirect logic here.
"""

from __future__ import annotations

import json
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
# Stub LintResponse
# ---------------------------------------------------------------------------


def _stub_lint_response():
    """Return a minimal LintResponse with empty findings."""
    from markdown_kb.app.schemas import (
        LintFindings,
        LintResponse,
        LintSummary,
    )

    summary = LintSummary(
        total_findings=0,
        findings_by_check={
            "c1": 0,
            "c2": 0,
            "c3": 0,
            "c4a": 0,
            "c5": 0,
            "c6": 0,
            "c8": 0,
            "c9": 0,
            "c10": 0,
            "c11": 0,
        },
        llm_calls=0,
        cost_usd=0.0,
        c5_pairs_capped=0,
        generated_at="2025-01-01T00:00:00Z",
    )
    findings = LintFindings(
        orphans=[],
        failed_grounding=[],
        slug_collisions=[],
        stale_pages=[],
        red_links=[],
        coverage_gaps=[],
        page_pairs=[],
        promotion_candidates=[],
        stale_filed_answers=[],
        invalid_qa_schemas=[],
    )
    return LintResponse(
        report_path="wiki/lint-report.md",
        findings=findings,
        summary=summary,
        check_errors={},
    )


# ===========================================================================
# kb_index_v1 tests
# ===========================================================================


class TestKbIndexV1Schema:
    """Schema / registration tests for kb_index_v1."""

    def test_tool_registered(self):
        """kb_index_v1 is registered in the server."""
        import asyncio

        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        names = [t.name for t in tools]
        assert "kb_index_v1" in names, f"kb_index_v1 not in registered tools: {names}"

    def test_strict_schema(self):
        """kb_index_v1 schema has additionalProperties:false (ADR-0016)."""
        import asyncio

        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool = next((t for t in tools if t.name == "kb_index_v1"), None)
        assert tool is not None
        schema = tool.inputSchema
        assert schema.get("additionalProperties") is False, (
            "additionalProperties:false missing from kb_index_v1 schema"
        )

    def test_no_required_parameters(self):
        """kb_index_v1 takes no required parameters (zero-arg tool)."""
        import asyncio

        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool = next((t for t in tools if t.name == "kb_index_v1"), None)
        assert tool is not None
        schema = tool.inputSchema
        required = schema.get("required", [])
        assert required == [], f"Expected no required params, got: {required}"


class TestKbIndexV1Success:
    """Functional tests for kb_index_v1 success shape."""

    @pytest.fixture(autouse=True)
    def _patch_build_index(self, monkeypatch):
        """Patch build_index to avoid real I/O."""
        import markdown_kb.app.indexer as indexer_mod

        monkeypatch.setattr(
            indexer_mod,
            "build_index",
            lambda *_a, **_kw: (2, 7),
        )

    def test_returns_files_and_sections_indexed(self):
        """kb_index_v1 returns {files_indexed, sections_indexed}."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        assert not _is_error_result(raw), f"Expected success, got isError: {raw}"
        result = _parse_result(raw)
        assert "files_indexed" in result, f"Missing files_indexed: {result}"
        assert "sections_indexed" in result, f"Missing sections_indexed: {result}"

    def test_files_indexed_from_build_index(self):
        """files_indexed reflects what build_index returned."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        result = _parse_result(raw)
        assert result["files_indexed"] == 2

    def test_sections_indexed_from_build_index(self):
        """sections_indexed reflects what build_index returned."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        result = _parse_result(raw)
        assert result["sections_indexed"] == 7


# ===========================================================================
# kb_lint_v1 tests
# ===========================================================================


class TestKbLintV1Schema:
    """Schema / registration tests for kb_lint_v1."""

    def test_tool_registered(self):
        """kb_lint_v1 is registered in the server."""
        import asyncio

        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        names = [t.name for t in tools]
        assert "kb_lint_v1" in names, f"kb_lint_v1 not in registered tools: {names}"

    def test_strict_schema(self):
        """kb_lint_v1 schema has additionalProperties:false (ADR-0016)."""
        import asyncio

        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool = next((t for t in tools if t.name == "kb_lint_v1"), None)
        assert tool is not None
        schema = tool.inputSchema
        assert schema.get("additionalProperties") is False, (
            "additionalProperties:false missing from kb_lint_v1 schema"
        )

    def test_include_c5_param_is_optional(self):
        """include_c5 is an optional boolean parameter."""
        import asyncio

        from kb_mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool = next((t for t in tools if t.name == "kb_lint_v1"), None)
        assert tool is not None
        schema = tool.inputSchema
        required = schema.get("required", [])
        assert "include_c5" not in required, (
            f"include_c5 should be optional but is in required: {required}"
        )


class TestKbLintV1Success:
    """Functional tests for kb_lint_v1 success shape."""

    @pytest.fixture(autouse=True)
    def _patch_run_lint(self, monkeypatch):
        """Patch run_lint to avoid real I/O and LLM calls."""
        import markdown_kb.app.lint as lint_mod

        monkeypatch.setattr(
            lint_mod,
            "run_lint",
            lambda **_kw: _stub_lint_response(),
        )

    def test_returns_success_payload(self):
        """kb_lint_v1 returns a success payload (not isError)."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))
        assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"

    def test_payload_has_summary(self):
        """kb_lint_v1 result contains a 'summary' key."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))
        result = _parse_result(raw)
        assert "summary" in result, f"Missing 'summary' key: {result}"

    def test_payload_has_findings(self):
        """kb_lint_v1 result contains a 'findings' key."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))
        result = _parse_result(raw)
        assert "findings" in result, f"Missing 'findings' key: {result}"

    def test_payload_has_report_path(self):
        """kb_lint_v1 result contains a 'report_path' key."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))
        result = _parse_result(raw)
        assert "report_path" in result, f"Missing 'report_path' key: {result}"

    def test_summary_total_findings_is_int(self):
        """summary.total_findings is an integer."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))
        result = _parse_result(raw)
        total = result["summary"]["total_findings"]
        assert isinstance(total, int), f"Expected int, got {type(total)}: {total}"

    def test_include_c5_false_forwarded(self):
        """include_c5=False is forwarded to run_lint."""
        import asyncio

        import markdown_kb.app.lint as lint_mod

        from kb_mcp.server import mcp

        recorded: list[bool] = []

        def stub_lint(**kwargs):
            recorded.append(kwargs.get("include_c5", True))
            return _stub_lint_response()

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(lint_mod, "run_lint", stub_lint)
            asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))

        assert recorded == [False], f"Expected include_c5=False forwarded, got: {recorded}"


# ===========================================================================
# kb_lint_v1 LLMError -> isError (ADR-0015)
# ===========================================================================


class TestKbLintV1LLMError:
    """LLMError from C5 renders as isError=True with structured payload (ADR-0015)."""

    def test_retryable_llm_error_renders_as_iserror(self):
        """LLMError(retryable=True) -> isError=True with code='LLM_UNAVAILABLE'."""
        import asyncio

        import markdown_kb.app.lint as lint_mod
        from markdown_kb.app.errors import LLMError

        from kb_mcp.server import mcp

        def _failing_run_lint(**kwargs):
            raise LLMError(retryable=True, message="OpenAI service timeout")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(lint_mod, "run_lint", _failing_run_lint)
            raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))

        assert _is_error_result(raw), (
            f"Expected isError=True for retryable LLMError, got success: {raw}"
        )
        payload = _parse_result(raw)
        assert payload.get("code") == "LLM_UNAVAILABLE", (
            f"Expected code='LLM_UNAVAILABLE', got: {payload}"
        )
        assert "message" in payload, f"Missing 'message' key: {payload}"

    def test_non_retryable_llm_error_renders_as_iserror(self):
        """LLMError(retryable=False) -> isError=True with code='LLM_ERROR'."""
        import asyncio

        import markdown_kb.app.lint as lint_mod
        from markdown_kb.app.errors import LLMError

        from kb_mcp.server import mcp

        def _failing_run_lint(**kwargs):
            raise LLMError(retryable=False, message="Authentication error")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(lint_mod, "run_lint", _failing_run_lint)
            raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))

        assert _is_error_result(raw), (
            f"Expected isError=True for non-retryable LLMError, got success: {raw}"
        )
        payload = _parse_result(raw)
        assert payload.get("code") == "LLM_ERROR", (
            f"Expected code='LLM_ERROR', got: {payload}"
        )

    def test_llm_error_message_carried(self):
        """LLMError.message is carried through to the isError payload."""
        import asyncio

        import markdown_kb.app.lint as lint_mod
        from markdown_kb.app.errors import LLMError

        from kb_mcp.server import mcp

        expected_message = "LLM service temporarily unavailable"

        def _failing_run_lint(**kwargs):
            raise LLMError(retryable=True, message=expected_message)

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(lint_mod, "run_lint", _failing_run_lint)
            raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))

        payload = _parse_result(raw)
        assert payload.get("message") == expected_message, (
            f"Expected message={expected_message!r}, got: {payload.get('message')!r}"
        )
