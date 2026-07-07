"""``kb lint`` renders C5's added Routed fix-the-Source hint (issue #534,
ADR-0036 decisions 1-2).

C5 is the second check (after C3, issue #408) carrying two remediation
classes: the existing Authored Reconcile tier stays unaffected (a
wiki-rooted contradiction still converges there), and this adds a Routed
navigation hint — plain text, mirroring C1/C2/C3's "fix via: ..." pattern.

Mocking follows ``test_cli_lint_routed_coverage.py``'s pattern: ``run_lint``
is patched at ``markdown_kb.app.lint.run_lint`` with a stubbed LintResponse —
no real I/O, no LLM.
"""

from __future__ import annotations

from typer.testing import CliRunner

runner = CliRunner()


def _make_c5_lint_response():
    from markdown_kb.app.schemas import LintFindings, LintResponse, LintSummary, PagePairFinding

    page_pairs = [
        PagePairFinding(
            severity="direct",
            page_a="refund-policy",
            page_b="return-window-reminder",
            page_a_claim="Refunds must be requested within 14 days.",
            page_b_claim="Refunds must be requested within 30 days.",
            summary="The two pages disagree about the refund window.",
            suggested_action="Reconcile the two pages.",
        )
    ]
    findings = LintFindings(page_pairs=page_pairs)
    summary = LintSummary(
        total_findings=1,
        findings_by_check={"c5": 1},
        generated_at="2026-07-07T00:00:00Z",
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


def test_c5_renders_fix_source_hint(monkeypatch):
    from kb_cli.main import app

    _patch_run_lint(monkeypatch, _make_c5_lint_response())

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"
    assert "fix via:" in result.output, f"C5 output must render the Routed hint:\n{result.output}"


def test_c5_hint_does_not_replace_the_contradiction_lines(monkeypatch):
    """C5 keeps both remediation classes visible: the per-pair contradiction
    line (Authored-tier evidence) AND the new Routed hint line — a strict
    addition, not a replacement."""
    from kb_cli.main import app

    _patch_run_lint(monkeypatch, _make_c5_lint_response())

    result = runner.invoke(app, ["lint"])

    assert result.exit_code == 0
    assert "fix via:" in result.output
    assert "refund-policy ↔ return-window-reminder" in result.output


def test_c5_hint_is_driven_by_the_shared_taxonomy(monkeypatch):
    """Not a re-derived string — reads remediation_for("C5").secondary_route,
    the SAME value C3's hint (and the future Console Source view) reads."""
    from markdown_kb.app.lint import remediation_for

    assert remediation_for("C5").tier == "authored"
    assert remediation_for("C5").secondary_route == "fix-source"
