"""``kb lint`` renders C3's added Routed fix-the-Source hint (issue #408,
ADR-0029 decisions 2-4).

C3 is the first check carrying two remediation classes: the existing Direct
"Re-ingest (retry)" line stays (issue #407 pinned it in
``test_cli_lint_c3_claims.py``), and this adds a Routed navigation hint —
plain text, mirroring C1/C2's "fill via: kb import ..." (ADR-0027).

Mocking follows ``test_cli_lint_routed_coverage.py``'s pattern: ``run_lint``
is patched at ``markdown_kb.app.lint.run_lint`` with a stubbed LintResponse —
no real I/O, no LLM.
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


def test_c3_renders_fix_source_hint(monkeypatch):
    from kb_cli.main import app

    _patch_run_lint(
        monkeypatch,
        _make_c3_lint_response(unsupported_claims=["The refund window is 90 days."]),
    )

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"
    assert "fix via:" in result.output, f"C3 output must render the Routed hint:\n{result.output}"


def test_c3_hint_does_not_replace_the_existing_reingest_reference(monkeypatch):
    """C3 keeps both remediation classes visible: the per-finding
    suggested_action text (Direct-tier evidence, issue #407) AND the new
    Routed hint line — this is a strict addition, not a replacement."""
    from kb_cli.main import app

    _patch_run_lint(
        monkeypatch,
        _make_c3_lint_response(unsupported_claims=["fake claim"]),
    )

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0
    assert "fix via:" in result.output
    assert "Amend the Source and force re-ingest." in result.output


def test_c3_hint_is_driven_by_the_shared_taxonomy(monkeypatch):
    """Not a re-derived string — reads remediation_for("C3").secondary_route."""
    from markdown_kb.app.lint import remediation_for

    assert remediation_for("C3").tier == "direct"
    assert remediation_for("C3").secondary_route == "fix-source"
