"""Integration tests for ``kb lint`` CLI subcommand (issue #229).

Tests use typer's CliRunner to invoke the CLI in-process without spawning a
subprocess.

Mocking follows the project pattern (CODING_STANDARD §11):
  - ``run_lint`` is mocked at ``markdown_kb.app.lint.run_lint`` (deep module
    public surface) returning a stubbed LintResponse — we do NOT mock sub-checks.
  - ``LLMError`` is raised inside the mock to test the error exit path.
  - LLM is NOT called directly; no real OpenAI calls are made.

The autouse ``_isolate_module_state`` fixture in conftest.py redirects all
real on-disk paths to ``tmp_path``.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _make_lint_response(
    *,
    orphan_slugs: list[str] | None = None,
    total_findings: int | None = None,
    check_errors: dict | None = None,
):
    """Return a minimal LintResponse for testing."""
    from markdown_kb.app.schemas import (
        LintFindings,
        LintResponse,
        LintSummary,
        OrphanPageFinding,
    )

    orphans = [
        OrphanPageFinding(
            page_slug=slug,
            missing_sources=["missing.md"],
            suggested_action=f"Review {slug}.",
        )
        for slug in (orphan_slugs or [])
    ]
    findings = LintFindings(orphans=orphans)
    total = total_findings if total_findings is not None else len(orphans)
    summary = LintSummary(
        total_findings=total,
        findings_by_check={"c11": len(orphans)},
        generated_at="2026-06-11T00:00:00Z",
    )
    return LintResponse(
        report_path="wiki/lint-report.md",
        findings=findings,
        summary=summary,
        check_errors=check_errors or {},
    )


def _patch_run_lint(monkeypatch, response):
    """Patch run_lint to return a fixed LintResponse without real I/O."""
    import markdown_kb.app.lint as lint_mod

    monkeypatch.setattr(lint_mod, "run_lint", lambda **_kw: response)


# ---------------------------------------------------------------------------
# AC-1: kb lint runs and renders findings to stdout
# ---------------------------------------------------------------------------


def test_kb_lint_exits_zero_with_findings(monkeypatch):
    """``kb lint`` exits 0 when run_lint returns findings."""
    from kb_cli.main import app

    resp = _make_lint_response(orphan_slugs=["orphan-page"])
    _patch_run_lint(monkeypatch, resp)

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}\n{result.output}"
    )


def test_kb_lint_renders_findings(monkeypatch):
    """``kb lint`` prints finding details to stdout."""
    from kb_cli.main import app

    resp = _make_lint_response(orphan_slugs=["orphan-page"])
    _patch_run_lint(monkeypatch, resp)

    result = runner.invoke(app, ["lint"])
    assert "orphan-page" in result.output, (
        f"Expected orphan-page slug in output:\n{result.output}"
    )


def test_kb_lint_shows_summary_counts(monkeypatch):
    """``kb lint`` prints a summary with finding counts."""
    from kb_cli.main import app

    resp = _make_lint_response(orphan_slugs=["page-a", "page-b"], total_findings=2)
    _patch_run_lint(monkeypatch, resp)

    result = runner.invoke(app, ["lint"])
    # Must mention total findings count in some form
    assert "2" in result.output or "finding" in result.output.lower(), (
        f"Expected findings count in output:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# AC-2: Empty / clean result reported clearly (success, not error)
# ---------------------------------------------------------------------------


def test_kb_lint_clean_run_exits_zero(monkeypatch):
    """``kb lint`` exits 0 when there are no findings (clean KB)."""
    from kb_cli.main import app

    resp = _make_lint_response(total_findings=0)
    _patch_run_lint(monkeypatch, resp)

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0, (
        f"Expected exit 0 on clean run, got {result.exit_code}\n{result.output}"
    )


def test_kb_lint_clean_run_reports_no_findings(monkeypatch):
    """``kb lint`` prints a success message when there are no findings."""
    from kb_cli.main import app

    resp = _make_lint_response(total_findings=0)
    _patch_run_lint(monkeypatch, resp)

    result = runner.invoke(app, ["lint"])
    # Must contain some clear positive indicator (not just silence)
    lower = result.output.lower()
    assert (
        "no finding" in lower
        or "0 finding" in lower
        or "clean" in lower
        or "pass" in lower
        or "ok" in lower
        or "0" in result.output
    ), f"Expected a clean-run success message:\n{result.output}"


# ---------------------------------------------------------------------------
# AC-3: LLMError exits non-zero with stderr, no traceback
# ---------------------------------------------------------------------------


def test_kb_lint_llm_error_exits_nonzero(monkeypatch):
    """``kb lint`` exits non-zero when run_lint raises LLMError."""
    import markdown_kb.app.lint as lint_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise(*_a, **_kw):
        raise LLMError(retryable=True, message="LLM service temporarily unavailable.")

    monkeypatch.setattr(lint_mod, "run_lint", _raise)

    result = runner.invoke(app, ["lint"])
    assert result.exit_code != 0, (
        f"Expected non-zero exit on LLMError, got {result.exit_code}\n{result.output}"
    )


def test_kb_lint_llm_error_message_in_output(monkeypatch):
    """``kb lint`` prints the LLMError message to stderr (CliRunner mixes it in)."""
    import markdown_kb.app.lint as lint_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise(*_a, **_kw):
        raise LLMError(retryable=False, message="LLM auth failed (check OPENAI_API_KEY).")

    monkeypatch.setattr(lint_mod, "run_lint", _raise)

    result = runner.invoke(app, ["lint"])
    combined = result.output or ""
    assert "LLM" in combined or "Error" in combined, (
        f"Expected LLM error message in output:\n{combined}"
    )


def test_kb_lint_llm_error_no_traceback(monkeypatch):
    """``kb lint`` does NOT print a Python traceback on LLMError."""
    import markdown_kb.app.lint as lint_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    def _raise(*_a, **_kw):
        raise LLMError(retryable=True, message="LLM timeout.")

    monkeypatch.setattr(lint_mod, "run_lint", _raise)

    result = runner.invoke(app, ["lint"])
    combined = result.output or ""
    # If a traceback were present, it would contain "Traceback (most recent call last)"
    assert "Traceback" not in combined, (
        f"LLMError must not produce a traceback:\n{combined}"
    )


# ---------------------------------------------------------------------------
# AC-4 & AC-5: Hermetic — tests write only to tmp, not real .kb/ / wiki/
# ---------------------------------------------------------------------------


def test_kb_lint_does_not_write_real_wiki(monkeypatch, tmp_path):
    """``kb lint`` (mocked) does not write to the real wiki directory."""
    # The _isolate_module_state autouse fixture already redirects wiki/ and
    # .kb/index.json to tmp_path — so this test simply verifies the lint
    # command completes without touching paths outside tmp_path.
    from kb_cli.main import app

    resp = _make_lint_response(total_findings=0)
    _patch_run_lint(monkeypatch, resp)

    result = runner.invoke(app, ["lint"])
    # Completing without error is the invariant — the autouse fixture ensures
    # real wiki/ was not written
    assert result.exit_code == 0
