"""``kb lint`` renders C1/C2 as Routed with a text navigation hint.

tier-B S7 (issue #383, ADR-0027): C1 coverage gaps and C2 red links flip from
Authored to Routed — fill routes through the existing Upload -> Import ->
Ingest pipeline, so there is no draft to approve. The Console gets a real
"Fill via Import" control (gateway/tests/test_ui_console_routed_coverage_fill.py);
the CLI has nothing to click, so it renders the SAME shared-taxonomy route as
plain text instead (ADR-0027 Consequences: "CLI/MCP render the route as
text").

Mocking follows the existing ``test_cli_lint.py`` pattern: ``run_lint`` is
patched at ``markdown_kb.app.lint.run_lint`` with a stubbed LintResponse — no
real I/O, no LLM.
"""

from __future__ import annotations

from typer.testing import CliRunner

runner = CliRunner()


def _make_coverage_and_red_link_response():
    """A LintResponse carrying one C1 and one C2 finding."""
    from markdown_kb.app.schemas import (
        CoverageGapFinding,
        LintFindings,
        LintResponse,
        LintSummary,
        RedLinkFinding,
    )

    findings = LintFindings(
        coverage_gaps=[
            CoverageGapFinding(
                reason="retrieval_empty",
                query_canonical="how do refunds work",
                sample_raw_queries=["how do refunds work"],
                hit_count=3,
                first_seen="2026-07-01T00:00:00Z",
                last_seen="2026-07-02T00:00:00Z",
                suggested_action="Create a new wiki page covering how do refunds work",
            )
        ],
        red_links=[
            RedLinkFinding(
                slug="missing-target",
                mention_count=2,
                referenced_by=["some-page"],
                sample_context="...see [[missing-target]] for details...",
            )
        ],
    )
    summary = LintSummary(
        total_findings=2,
        findings_by_check={"c1": 1, "c2": 1},
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


def test_c1_coverage_gap_renders_fill_via_import_hint(monkeypatch):
    from kb_cli.main import app

    _patch_run_lint(monkeypatch, _make_coverage_and_red_link_response())

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0
    assert "fill via: kb import" in result.output, (
        f"C1 coverage-gap output must render the Routed navigation hint:\n{result.output}"
    )


def test_c2_red_link_renders_fill_via_import_hint(monkeypatch):
    from kb_cli.main import app

    _patch_run_lint(monkeypatch, _make_coverage_and_red_link_response())

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0
    # Both C1 and C2 route through the same "import" route — assert it
    # appears at least twice (once per Routed check section).
    assert result.output.count("fill via: kb import") == 2, (
        f"Expected the Routed hint once per C1 and once per C2:\n{result.output}"
    )


def test_c1_and_c2_headers_are_unchanged_by_the_new_hint_line(monkeypatch):
    """The Routed hint is a new line, not a rewrite of the existing header
    line the pre-existing test_kb_lint_each_check_is_labelled anchors on."""
    from kb_cli.main import app

    _patch_run_lint(monkeypatch, _make_coverage_and_red_link_response())

    result = runner.invoke(app, ["lint"])
    assert "C1 Coverage gaps (1) — coverage-gap:" in result.output
    assert "C2 Red links (1) — red-link:" in result.output


def test_routed_tier_taxonomy_matches_cli_hint(monkeypatch):
    """The CLI hint is driven by the shared taxonomy, not a re-derived string."""
    from markdown_kb.app.lint import remediation_for

    assert remediation_for("C1").tier == "routed"
    assert remediation_for("C1").route == "import"
    assert remediation_for("C2").tier == "routed"
    assert remediation_for("C2").route == "import"
