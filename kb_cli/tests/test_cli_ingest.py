"""Tests for ``kb ingest [source]`` subcommand (issue #228, slice/228-cli-ingest).

All tests are hermetic: LLM / verifier are mocked at the module level per the
project pattern (CODING_STANDARD §11).  The ingest deep-module entry point
``ingest_sources`` is mocked at ``markdown_kb.app.ingest.ingest_sources`` so no
real LLM calls or file I/O reach real wiki/ or .kb/index.json.

Coverage:
  AC-1: single-source ingest (``kb ingest refund_policy.md``)
  AC-1: batch ingest (``kb ingest`` with no argument)
  AC-2: per-source progress lines on stdout
  AC-3: grounding-failed / Cannot-Confirm outcome surfaced in output
  AC-4: LLMError exits non-zero with stderr message and no traceback
  AC-6: tests write only to tmp_path (autouse conftest guarantees this)
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------


def _make_batch_result(
    *,
    sources: list[str] | None = None,
    failed_sources: list[str] | None = None,
    pages_with_failed_grounding: list[str] | None = None,
):
    """Build a minimal IngestBatchResult for use as a mock return value."""
    from markdown_kb.app.ingest import IngestBatchResult
    from markdown_kb.app.schemas import IngestSourceResult

    result = IngestBatchResult()
    result.failed_sources = list(failed_sources or [])
    result.pages_with_failed_grounding = list(pages_with_failed_grounding or [])

    for src in sources or []:
        result.results.append(
            IngestSourceResult(
                source=src,
                pages_written=[f"wiki/concepts/{src.replace('.md', '')}.md"],
                pages_created=[f"wiki/concepts/{src.replace('.md', '')}.md"],
                status="created",
            )
        )

    return result


def _patch_ingest(monkeypatch, *, return_value=None, side_effect=None):
    """Patch ``ingest_sources`` to avoid real LLM/file I/O.

    Per the project pattern (CODING_STANDARD §11): mock at the deep-module
    entry point in ``markdown_kb.app.ingest``, not at a private helper.
    """
    import markdown_kb.app.ingest as ingest_mod

    if side_effect is not None:
        monkeypatch.setattr(ingest_mod, "ingest_sources", side_effect)
    else:
        if return_value is None:
            return_value = _make_batch_result()
        monkeypatch.setattr(ingest_mod, "ingest_sources", lambda *a, **kw: return_value)


# ---------------------------------------------------------------------------
# AC-1: single-source ingest
# ---------------------------------------------------------------------------


def test_kb_ingest_single_source_exits_zero(monkeypatch):
    """``kb ingest refund_policy.md`` exits with code 0 on success."""
    from kb_cli.main import app

    _patch_ingest(monkeypatch, return_value=_make_batch_result(sources=["refund_policy.md"]))
    result = runner.invoke(app, ["ingest", "refund_policy.md"])
    assert result.exit_code == 0, (
        f"Expected exit 0 for single-source ingest, got {result.exit_code}\n{result.output}"
    )


def test_kb_ingest_single_source_calls_ingest_with_filename(monkeypatch):
    """``kb ingest refund_policy.md`` passes the filename to ``ingest_sources``."""
    import markdown_kb.app.ingest as ingest_mod

    from kb_cli.main import app

    calls: list[tuple] = []

    def _recording_ingest(source_filenames, **kw):
        calls.append((source_filenames,))
        return _make_batch_result(sources=source_filenames or [])

    monkeypatch.setattr(ingest_mod, "ingest_sources", _recording_ingest)
    runner.invoke(app, ["ingest", "refund_policy.md"])

    assert len(calls) == 1, f"Expected ingest_sources called once, got {len(calls)}"
    filenames = calls[0][0]
    assert filenames is not None, "Expected source_filenames to be a list, got None"
    assert "refund_policy.md" in filenames, (
        f"Expected 'refund_policy.md' in source_filenames, got {filenames}"
    )


# ---------------------------------------------------------------------------
# AC-1: batch ingest (no argument)
# ---------------------------------------------------------------------------


def test_kb_ingest_batch_exits_zero(monkeypatch):
    """``kb ingest`` with no argument exits with code 0 on success."""
    from kb_cli.main import app

    _patch_ingest(
        monkeypatch,
        return_value=_make_batch_result(sources=["refund_policy.md", "shipping_policy.md"]),
    )
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0, (
        f"Expected exit 0 for batch ingest, got {result.exit_code}\n{result.output}"
    )


def test_kb_ingest_batch_calls_ingest_with_none(monkeypatch):
    """``kb ingest`` (no arg) passes ``source_filenames=None`` to ``ingest_sources``."""
    import markdown_kb.app.ingest as ingest_mod

    from kb_cli.main import app

    calls: list[tuple] = []

    def _recording_ingest(source_filenames, **kw):
        calls.append((source_filenames,))
        return _make_batch_result()

    monkeypatch.setattr(ingest_mod, "ingest_sources", _recording_ingest)
    runner.invoke(app, ["ingest"])

    assert len(calls) == 1, f"Expected ingest_sources called once, got {len(calls)}"
    assert calls[0][0] is None, f"Expected source_filenames=None for batch mode, got {calls[0][0]}"


# ---------------------------------------------------------------------------
# AC-2: per-source progress lines on stdout
# ---------------------------------------------------------------------------


def test_kb_ingest_single_source_prints_source_name(monkeypatch):
    """``kb ingest <source>`` prints the source name in its progress output."""
    import markdown_kb.app.ingest as ingest_mod

    from kb_cli.main import app

    monkeypatch.setattr(
        ingest_mod,
        "ingest_sources",
        lambda *a, **kw: _make_batch_result(sources=["refund_policy.md"]),
    )
    result = runner.invoke(app, ["ingest", "refund_policy.md"])

    output = result.output
    assert "refund_policy" in output, f"Expected source name in progress output, got:\n{output}"


def test_kb_ingest_single_source_prints_done_with_page_count(monkeypatch):
    """``kb ingest <source>`` prints a done/completion line with the page count."""
    from kb_cli.main import app

    _patch_ingest(monkeypatch, return_value=_make_batch_result(sources=["refund_policy.md"]))
    result = runner.invoke(app, ["ingest", "refund_policy.md"])

    output = result.output
    # Must contain a count or "page" to indicate completion with page count
    assert "page" in output.lower() or "1" in output, (
        f"Expected page count in output, got:\n{output}"
    )


def test_kb_ingest_batch_prints_progress_for_each_source(monkeypatch):
    """Batch ``kb ingest`` prints a progress line per source."""
    from kb_cli.main import app

    _patch_ingest(
        monkeypatch,
        return_value=_make_batch_result(sources=["refund_policy.md", "shipping_policy.md"]),
    )
    result = runner.invoke(app, ["ingest"])

    output = result.output
    # Each source name must appear in the progress output
    assert "refund_policy" in output, f"Expected 'refund_policy' in output:\n{output}"
    assert "shipping_policy" in output, f"Expected 'shipping_policy' in output:\n{output}"


# ---------------------------------------------------------------------------
# AC-3: grounding-failed / Cannot-Confirm surfaced, not silently skipped
# ---------------------------------------------------------------------------


def test_kb_ingest_reports_grounding_failed_pages(monkeypatch):
    """Pages with failed grounding are reported in the output, not silently skipped."""
    from kb_cli.main import app

    batch = _make_batch_result(
        sources=["refund_policy.md"],
        pages_with_failed_grounding=["cancellation-window"],
    )
    _patch_ingest(monkeypatch, return_value=batch)

    result = runner.invoke(app, ["ingest", "refund_policy.md"])

    output = result.output
    # The failed grounding page slug must appear in the output
    assert "cancellation-window" in output or "grounding" in output.lower(), (
        f"Expected grounding failure info in output, got:\n{output}"
    )


def test_kb_ingest_exits_zero_when_grounding_fails_soft(monkeypatch):
    """Grounding failure is fail-soft: command exits 0 (the page was still written)."""
    from kb_cli.main import app

    batch = _make_batch_result(
        sources=["refund_policy.md"],
        pages_with_failed_grounding=["cancellation-window"],
    )
    _patch_ingest(monkeypatch, return_value=batch)

    result = runner.invoke(app, ["ingest", "refund_policy.md"])
    assert result.exit_code == 0, (
        f"Grounding failure should be fail-soft (exit 0), got {result.exit_code}\n{result.output}"
    )


def test_kb_ingest_reports_failed_sources(monkeypatch):
    """Sources that failed processing are reported in the output."""
    from kb_cli.main import app

    batch = _make_batch_result(failed_sources=["broken_source.md"])
    _patch_ingest(monkeypatch, return_value=batch)

    result = runner.invoke(app, ["ingest"])
    output = result.output
    assert "broken_source" in output or "fail" in output.lower(), (
        f"Expected failed source name in output, got:\n{output}"
    )


# ---------------------------------------------------------------------------
# AC-4: LLMError exits non-zero + clear stderr message + no traceback
# ---------------------------------------------------------------------------


def test_kb_ingest_llm_error_nonzero_exit(monkeypatch):
    """``LLMError`` from ingest_sources renders as non-zero exit."""
    import markdown_kb.app.ingest as ingest_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise_llm_error(source_filenames, **kw):
        raise LLMError(retryable=True, message="LLM service temporarily unavailable.")

    monkeypatch.setattr(ingest_mod, "ingest_sources", _raise_llm_error)

    result = runner.invoke(app, ["ingest", "refund_policy.md"])
    assert result.exit_code != 0, (
        f"Expected non-zero exit for LLMError, got {result.exit_code}\n{result.output}"
    )


def test_kb_ingest_llm_error_message_in_output(monkeypatch):
    """``LLMError`` message appears in CLI output (stderr mixed by CliRunner)."""
    import markdown_kb.app.ingest as ingest_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise_llm_error(source_filenames, **kw):
        raise LLMError(retryable=False, message="LLM auth failed (check OPENAI_API_KEY).")

    monkeypatch.setattr(ingest_mod, "ingest_sources", _raise_llm_error)

    result = runner.invoke(app, ["ingest", "refund_policy.md"])
    combined = result.output or ""
    assert "LLM" in combined or "Error" in combined, (
        f"Expected LLM error message in output, got:\n{combined}"
    )


def test_kb_ingest_llm_error_no_traceback(monkeypatch):
    """``LLMError`` renders without a Python traceback in the output."""
    import markdown_kb.app.ingest as ingest_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise_llm_error(source_filenames, **kw):
        raise LLMError(retryable=True, message="LLM timeout.")

    monkeypatch.setattr(ingest_mod, "ingest_sources", _raise_llm_error)

    result = runner.invoke(app, ["ingest", "refund_policy.md"])
    combined = result.output or ""
    # No traceback frames in output
    assert "Traceback" not in combined and 'File "' not in combined, (
        f"Expected no traceback in LLMError output, got:\n{combined}"
    )


def test_kb_ingest_llm_error_retryable_label(monkeypatch):
    """Retryable LLMError uses the LLM_UNAVAILABLE code label."""
    import markdown_kb.app.ingest as ingest_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise_llm_error(source_filenames, **kw):
        raise LLMError(retryable=True, message="Rate limit exceeded.")

    monkeypatch.setattr(ingest_mod, "ingest_sources", _raise_llm_error)

    result = runner.invoke(app, ["ingest", "refund_policy.md"])
    combined = result.output or ""
    assert "LLM_UNAVAILABLE" in combined or "unavailable" in combined.lower(), (
        f"Expected LLM_UNAVAILABLE label in output, got:\n{combined}"
    )


def test_kb_ingest_llm_error_nonretryable_label(monkeypatch):
    """Non-retryable LLMError uses the LLM_ERROR code label."""
    import markdown_kb.app.ingest as ingest_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise_llm_error(source_filenames, **kw):
        raise LLMError(retryable=False, message="Bad API key.")

    monkeypatch.setattr(ingest_mod, "ingest_sources", _raise_llm_error)

    result = runner.invoke(app, ["ingest", "refund_policy.md"])
    combined = result.output or ""
    assert "LLM_ERROR" in combined or "Error" in combined, (
        f"Expected LLM_ERROR label in output, got:\n{combined}"
    )
