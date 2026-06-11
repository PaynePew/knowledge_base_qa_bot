"""Integration tests for kb_index_v1 and kb_lint_v1 using FastMCP in-process harness.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.  This exercises FastMCP's argument
validation and the full dispatch path.

Three scenarios:
  kb_index_v1:  rebuilds the Section Index; returns {files_indexed, sections_indexed}.
  kb_lint_v1 success: runs the REAL run_lint against tmp wiki/docs fixtures (no LLM
                needed — include_c5=False); returns the structured LintResponse payload
                as a neutral JSON dict.
  kb_lint_v1 C5 LLM error: drives the REAL run_lint with include_c5=True against
                tmp wiki pages that produce a candidate pair.  get_lint_llm is
                monkeypatched to a fake that raises LLMError so no real API call is
                made.  Because run_lint's C5 block uses continue-on-error semantics
                (except Exception swallows all per-check exceptions including
                LLMError), the response is a SUCCESS payload with check_errors["c5"]
                populated — proving the real C5 LLM failure path is captured.

Mocking strategy (CODING_STANDARD §6.3 / §11):
  - build_index and run_lint are NOT mocked — they are the deep-module entry
    points; mocking them would mask integration drift.
  - The LLM is the ONLY thing replaced with a stub, via the sanctioned lazy-
    singleton getter: ``markdown_kb.app.lint.get_lint_llm`` (CODING_STANDARD §6.3).
  - Path isolation is provided by the autouse ``_isolate_module_state`` fixture in
    conftest.py, which redirects WIKI_DIR / DOCS_DIR / LOG_PATH / SOURCE_DIRS to
    tmp so no test touches the real wiki/, docs/, or .kb/ trees (issue #231).

Prior art for this pattern:
  - markdown_kb/tests/lint/test_c5_unit.py — monkeypatches get_lint_llm
  - markdown_kb/tests/lint/test_c1_coverage.py — calls real run_lint(**lint_env)
  - kb_mcp/tests/test_kb_ask.py — mocks get_llm (not the deep query entry point)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

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
# Wiki page fixture helpers (mirrors markdown_kb/tests/lint/test_c5_unit.py)
# ---------------------------------------------------------------------------


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    sources: list[str],
    body: str = "",
    *,
    subdir: str = "concepts",
) -> Path:
    """Write a minimal wiki page with YAML frontmatter to tmp_wiki_dir."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": sources,
        "status": "live",
        "open_questions": [],
    }
    body_content = body or f"# {slug}\n\nContent about {slug.replace('-', ' ')}."
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body_content}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


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
    """Functional tests for kb_index_v1 driving the REAL build_index.

    The autouse ``_isolate_module_state`` fixture in conftest.py redirects
    SOURCE_DIRS to tmp_wiki subdirs (entities / concepts / qa) and INDEX_PATH to
    tmp so build_index never reads/writes the real on-disk trees.  These tests
    call build_index through the full MCP dispatch path without mocking it.
    """

    def test_returns_files_and_sections_indexed(self, tmp_path):
        """kb_index_v1 returns {files_indexed, sections_indexed} from real build_index."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        assert not _is_error_result(raw), f"Expected success, got isError: {raw}"
        result = _parse_result(raw)
        assert "files_indexed" in result, f"Missing files_indexed: {result}"
        assert "sections_indexed" in result, f"Missing sections_indexed: {result}"

    def test_files_indexed_is_integer(self, tmp_path):
        """files_indexed is an integer (real return from build_index)."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        result = _parse_result(raw)
        assert isinstance(result["files_indexed"], int), (
            f"Expected int, got {type(result['files_indexed'])}"
        )

    def test_sections_indexed_is_integer(self, tmp_path):
        """sections_indexed is an integer (real return from build_index)."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        result = _parse_result(raw)
        assert isinstance(result["sections_indexed"], int), (
            f"Expected int, got {type(result['sections_indexed'])}"
        )

    def test_empty_wiki_returns_zero_counts(self, tmp_path):
        """With no wiki pages in tmp, build_index reports zero files and sections."""
        import asyncio

        from kb_mcp.server import mcp

        # tmp SOURCE_DIRS exist but are empty (created by conftest redirect)
        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        result = _parse_result(raw)
        assert result["files_indexed"] == 0, (
            f"Expected 0 files for empty wiki, got {result['files_indexed']}"
        )
        assert result["sections_indexed"] == 0, (
            f"Expected 0 sections for empty wiki, got {result['sections_indexed']}"
        )

    def test_real_wiki_page_is_indexed(self, tmp_path):
        """A real .md page written to the tmp wiki subdir is indexed by build_index.

        This test exercises the full real build_index scan pipeline — parse_markdown,
        BM25 inverted index construction, and JSON persistence — driven through the
        MCP dispatch path.  It proves the wrapper correctly unpacks the real return
        tuple from the deep module.
        """
        import asyncio

        import markdown_kb.app.indexer as current_indexer

        from kb_mcp.server import mcp

        # Write a single page into the redirected entities/ subdir
        entities_dir = current_indexer.SOURCE_DIRS[0]  # tmp_wiki/entities
        entities_dir.mkdir(parents=True, exist_ok=True)
        page = entities_dir / "test-page.md"
        page.write_text(
            "# Test Page\n\nThis section has some content for indexing.\n\n"
            "## Another Section\n\nMore content here.\n",
            encoding="utf-8",
        )

        raw = asyncio.run(mcp.call_tool("kb_index_v1", {}))
        result = _parse_result(raw)
        assert result["files_indexed"] >= 1, (
            f"Expected at least 1 file indexed, got {result['files_indexed']}"
        )
        assert result["sections_indexed"] >= 1, (
            f"Expected at least 1 section indexed, got {result['sections_indexed']}"
        )


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
    """Functional tests for kb_lint_v1 driving the REAL run_lint (include_c5=False).

    The autouse ``_isolate_module_state`` fixture redirects
    markdown_kb.app.lint.WIKI_DIR / DOCS_DIR / LOG_PATH to tmp so run_lint never
    reads or writes the real wiki/ tree.  No LLM is involved because include_c5
    defaults to True in the server but the tests pass include_c5=False to skip C5
    (the only LLM-backed check).  The LintResponse shape and JSON serialisation
    are verified against the real Pydantic model output.
    """

    def test_returns_success_payload(self):
        """kb_lint_v1 returns a success payload (not isError) from real run_lint."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
        assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"

    def test_payload_has_summary(self):
        """kb_lint_v1 result contains a 'summary' key from real LintResponse."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
        result = _parse_result(raw)
        assert "summary" in result, f"Missing 'summary' key: {result}"

    def test_payload_has_findings(self):
        """kb_lint_v1 result contains a 'findings' key from real LintResponse."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
        result = _parse_result(raw)
        assert "findings" in result, f"Missing 'findings' key: {result}"

    def test_payload_has_report_path(self):
        """kb_lint_v1 result contains a 'report_path' key from real LintResponse."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
        result = _parse_result(raw)
        assert "report_path" in result, f"Missing 'report_path' key: {result}"

    def test_payload_has_check_errors(self):
        """kb_lint_v1 result contains a 'check_errors' key from real LintResponse."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
        result = _parse_result(raw)
        assert "check_errors" in result, f"Missing 'check_errors' key: {result}"

    def test_summary_total_findings_is_int(self):
        """summary.total_findings is an integer from the real LintSummary model."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
        result = _parse_result(raw)
        total = result["summary"]["total_findings"]
        assert isinstance(total, int), f"Expected int, got {type(total)}: {total}"

    def test_include_c5_false_skips_llm(self):
        """include_c5=False produces page_pairs=[] and llm_calls=0 in the real response."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
        result = _parse_result(raw)
        assert result["findings"]["page_pairs"] == [], (
            f"Expected page_pairs=[] with include_c5=False, got: {result['findings']['page_pairs']}"
        )
        assert result["summary"]["llm_calls"] == 0, (
            f"Expected llm_calls=0 with include_c5=False, got: {result['summary']['llm_calls']}"
        )

    def test_include_c5_true_is_default(self):
        """Calling kb_lint_v1 with no args runs the full lint suite (include_c5 defaults True).

        Without pages in the wiki, C5 finds no candidate pairs and returns an
        empty list without calling the LLM.
        """
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {}))
        assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
        result = _parse_result(raw)
        assert "findings" in result, f"Missing 'findings' key: {result}"


# ===========================================================================
# kb_lint_v1 C5 LLM error captured in check_errors (real run_lint + mocked LLM)
# ===========================================================================


class TestKbLintV1C5LLMError:
    """Drives the REAL run_lint with mocked get_lint_llm to prove C5 LLM errors are captured.

    Design note (CODING_STANDARD §6.3):
      run_lint's C5 block uses continue-on-error semantics: all exceptions raised
      inside _check_c5_page_pair — including LLMError from the thread-pool judge —
      are caught by ``except Exception`` and stored in check_errors["c5"].  The
      response is therefore always a SUCCESS payload; LLMError never propagates out
      of run_lint to trigger the server's isError path when the failure originates
      inside the C5 judge threads.

    What we test here (real integration):
      - Set up two wiki pages sharing a source so C5 generates a candidate pair.
      - Monkeypatch get_lint_llm (the sanctioned §6.3 mock seam) to return a fake
        LLM whose with_structured_output chain raises LLMError on invoke.
      - Run the real run_lint via the MCP dispatch path.
      - Assert the response is a SUCCESS with check_errors["c5"] populated,
        proving the real C5 failure path is captured and surfaced correctly.
    """

    @pytest.fixture()
    def _two_page_wiki(self, tmp_path):
        """Write two wiki pages sharing a source so C5 builds a candidate pair."""
        import markdown_kb.app.lint as lint_mod

        wiki_dir = lint_mod.WIKI_DIR  # already redirected to tmp by conftest
        _write_wiki_page(
            wiki_dir,
            "page-alpha",
            ["shared-source.md#section"],
            "Refunds take 5 business days.",
        )
        _write_wiki_page(
            wiki_dir,
            "page-beta",
            ["shared-source.md#section"],
            "Refunds take 14 business days.",
        )

    @pytest.fixture()
    def _failing_lint_llm(self, monkeypatch):
        """Monkeypatch get_lint_llm to a fake that raises LLMError on chain.invoke."""
        import markdown_kb.app.lint as lint_mod
        from markdown_kb.app.errors import LLMError

        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = LLMError(
            retryable=True, message="LLM service unavailable (test stub)"
        )
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_chain
        monkeypatch.setattr(lint_mod, "get_lint_llm", lambda: fake_llm)

    def test_c5_llm_error_captured_in_check_errors(self, _two_page_wiki, _failing_lint_llm):
        """Real run_lint with failing get_lint_llm: C5 error stored in check_errors['c5'].

        Proves the continue-on-error path: LLMError from the C5 judge is caught
        by run_lint's except-Exception block and recorded in check_errors, not
        raised to the server's LLMError handler.  The MCP response is a SUCCESS.
        """
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": True}))
        # run_lint's continue-on-error semantics: C5 LLMError is caught, not raised
        assert not _is_error_result(raw), (
            "Expected SUCCESS payload (run_lint catches C5 exceptions via continue-on-error), "
            f"got isError: {raw}"
        )
        result = _parse_result(raw)
        assert "check_errors" in result, f"Missing 'check_errors' key: {result}"
        assert "c5" in result["check_errors"], (
            f"Expected 'c5' in check_errors (C5 LLMError should be captured), "
            f"got check_errors={result['check_errors']}"
        )

    def test_c5_llm_error_message_in_check_errors(self, _two_page_wiki, _failing_lint_llm):
        """check_errors['c5'] contains the LLMError class name from the real C5 path."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": True}))
        result = _parse_result(raw)
        c5_error = result["check_errors"].get("c5", "")
        assert "LLMError" in c5_error, (
            f"Expected 'LLMError' in check_errors['c5'], got: {c5_error!r}"
        )

    def test_c5_error_does_not_suppress_other_findings_keys(
        self, _two_page_wiki, _failing_lint_llm
    ):
        """When C5 errors, the success payload still has all top-level keys."""
        import asyncio

        from kb_mcp.server import mcp

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": True}))
        result = _parse_result(raw)
        for key in ("report_path", "findings", "summary", "check_errors"):
            assert key in result, f"Missing '{key}' key when C5 errored: {result}"

    def test_non_retryable_llm_error_also_captured(self, _two_page_wiki, monkeypatch):
        """Non-retryable LLMError from get_lint_llm is also captured in check_errors['c5']."""
        import asyncio

        import markdown_kb.app.lint as lint_mod
        from markdown_kb.app.errors import LLMError

        from kb_mcp.server import mcp

        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = LLMError(retryable=False, message="Auth error (test stub)")
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_chain
        monkeypatch.setattr(lint_mod, "get_lint_llm", lambda: fake_llm)

        raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": True}))
        assert not _is_error_result(raw), "Expected SUCCESS payload for non-retryable LLMError too"
        result = _parse_result(raw)
        assert "c5" in result["check_errors"], (
            f"Expected 'c5' in check_errors for non-retryable error: {result['check_errors']}"
        )
