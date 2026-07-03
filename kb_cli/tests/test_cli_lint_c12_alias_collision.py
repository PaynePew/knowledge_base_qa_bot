"""Tests for ``kb lint``'s C12 alias-collision rendering (issue #406, ADR-0030).

Mirrors ``test_cli_lint.py``'s pattern: ``run_lint`` is mocked at
``markdown_kb.app.lint.run_lint`` returning a stubbed ``LintResponse``; no
real I/O, no LLM calls.
"""

from __future__ import annotations

from typer.testing import CliRunner

runner = CliRunner()


def _make_c12_lint_response():
    from markdown_kb.app.schemas import (
        AliasCollisionFinding,
        LintFindings,
        LintResponse,
        LintSummary,
    )

    alias_collisions = [
        AliasCollisionFinding(
            kind="alias_vs_slug",
            alias="pricing",
            claimed_by=["other-page"],
            slug_owner="pricing",
            resolved_to="pricing",
            suggested_action="Edit frontmatter to remove or rename this alias.",
        )
    ]
    findings = LintFindings(alias_collisions=alias_collisions)
    summary = LintSummary(
        total_findings=1,
        findings_by_check={"c12": 1},
        generated_at="2026-07-03T00:00:00Z",
    )
    return LintResponse(
        report_path="wiki/lint-report.md",
        findings=findings,
        summary=summary,
        check_errors={},
    )


def _patch_run_lint(monkeypatch, response):
    import markdown_kb.app.lint as lint_mod

    monkeypatch.setattr(lint_mod, "run_lint", lambda **_kw: response)


def test_kb_lint_renders_c12_alias_collision(monkeypatch):
    """``kb lint`` prints the alias-collision finding without raising —
    regression guard for the KeyError a missing formatter entry would cause
    once C12 is a LINT_CHECK_TAXONOMY member (group_findings_by_axis walks
    every taxonomy code)."""
    from kb_cli.main import app

    _patch_run_lint(monkeypatch, _make_c12_lint_response())

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"
    assert "C12 Alias collisions (1)" in result.output
    assert "pricing" in result.output
    assert "== Coherence ==" in result.output


def test_kb_lint_omits_c12_section_when_no_alias_collisions(monkeypatch):
    """A clean run's C12 section is silent, matching every other check's
    zero-findings convention (the always-render axis body simply has
    nothing to contribute for C12)."""
    from markdown_kb.app.schemas import LintFindings, LintResponse, LintSummary

    from kb_cli.main import app

    response = LintResponse(
        report_path="wiki/lint-report.md",
        findings=LintFindings(),
        summary=LintSummary(
            total_findings=0, findings_by_check={}, generated_at="2026-07-03T00:00:00Z"
        ),
        check_errors={},
    )
    _patch_run_lint(monkeypatch, response)

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0
    assert "C12 Alias collisions" not in result.output
