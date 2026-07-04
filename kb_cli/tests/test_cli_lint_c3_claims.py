"""Tests for ``kb lint``'s C3 unsupported_claims rendering (issue #407, ADR-0017
four-surface parity).

Mirrors ``test_cli_lint_c12_alias_collision.py``'s pattern: ``run_lint`` is
mocked at ``markdown_kb.app.lint.run_lint`` returning a stubbed
``LintResponse``; no real I/O, no LLM calls.
"""

from __future__ import annotations

from typer.testing import CliRunner

runner = CliRunner()


def _make_c3_lint_response(
    *, unsupported_claims: list[str] | None, reason: str = "claim_unsupported"
):
    from markdown_kb.app.schemas import (
        FailedGroundingFinding,
        LintFindings,
        LintResponse,
        LintSummary,
    )

    failed_grounding = [
        FailedGroundingFinding(
            page_slug="broken-page",
            source="policy.md",
            reason=reason,  # type: ignore[arg-type]
            unsupported_claims=unsupported_claims or [],
            suggested_action="Amend the Source and force re-ingest.",
        )
    ]
    findings = LintFindings(failed_grounding=failed_grounding)
    summary = LintSummary(
        total_findings=1,
        findings_by_check={"c3": 1},
        generated_at="2026-07-04T00:00:00Z",
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


def test_kb_lint_renders_c3_unsupported_claims(monkeypatch):
    """``kb lint`` prints the recorded unsupported claims for a
    claim_unsupported finding (issue #407 AC — CLI was previously dropping
    them, only the Markdown report rendered claims)."""
    from kb_cli.main import app

    _patch_run_lint(
        monkeypatch,
        _make_c3_lint_response(unsupported_claims=["The refund window is 90 days."]),
    )

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"
    assert "broken-page" in result.output
    assert "The refund window is 90 days." in result.output


def test_kb_lint_c3_empty_claims_degrades_gracefully(monkeypatch):
    """An empty unsupported_claims list (e.g. verifier_unavailable) must not
    crash and must render an honest note rather than a blank claims cell."""
    from kb_cli.main import app

    _patch_run_lint(
        monkeypatch,
        _make_c3_lint_response(unsupported_claims=[], reason="verifier_unavailable"),
    )

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"
    assert "not recorded" in result.output.lower()
