"""``kb lint``'s C8/C9 text output gains question + path (issue #377 AC2, CLI half).

Companion to ``kb_mcp/tests/test_kb_lint_qa_visibility.py`` (the MCP half).
Mirrors ``test_cli_lint.py``'s established pattern of mocking
``markdown_kb.app.lint.run_lint`` with a stubbed ``LintResponse`` — this file
is intentionally separate from ``test_cli_lint.py`` (an existing test file
this slice must not modify) rather than appended to it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

runner = CliRunner()


def _patch_run_lint(monkeypatch, response):
    import markdown_kb.app.lint as lint_mod

    monkeypatch.setattr(lint_mod, "run_lint", lambda **_kw: response)


def _write_qa_page(wiki_dir: Path, slug: str, *, question: str, status: str) -> Path:
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "id": slug,
        "type": "qa",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": [],
        "status": status,
        "open_questions": [],
        "question": question,
        "count": 1,
    }
    fm_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    page_path = qa_dir / f"{slug}.md"
    page_path.write_text(f"---\n{fm_yaml}---\n\nBody.\n", encoding="utf-8")
    return page_path


def _lint_response_with_c8(candidate):
    from markdown_kb.app.schemas import LintFindings, LintResponse, LintSummary

    findings = LintFindings(promotion_candidates=[candidate])
    summary = LintSummary(
        total_findings=1, findings_by_check={"c8": 1}, generated_at="2026-07-03T00:00:00Z"
    )
    return LintResponse(
        report_path="wiki/lint-report.md", findings=findings, summary=summary, check_errors={}
    )


def _lint_response_with_c9(finding):
    from markdown_kb.app.schemas import LintFindings, LintResponse, LintSummary

    findings = LintFindings(stale_filed_answers=[finding])
    summary = LintSummary(
        total_findings=1, findings_by_check={"c9": 1}, generated_at="2026-07-03T00:00:00Z"
    )
    return LintResponse(
        report_path="wiki/lint-report.md", findings=findings, summary=summary, check_errors={}
    )


def test_kb_lint_c8_output_includes_question_and_path(monkeypatch):
    from markdown_kb.app.schemas import PromotionCandidateFinding

    from kb_cli.main import app

    candidate = PromotionCandidateFinding(
        slug="popular-question",
        question="How do refunds work?",
        count=5,
        age_days=3.0,
        cited_count=1,
    )
    _patch_run_lint(monkeypatch, _lint_response_with_c8(candidate))

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0, result.output
    # Header line unchanged (test_kb_lint_each_check_is_labelled pins this
    # exact string in test_cli_lint.py — this file only extends the body).
    assert "C8 Promotion candidates (1) — promotion:" in result.output
    assert "How do refunds work?" in result.output
    assert "wiki/qa/popular-question.md" in result.output


def test_kb_lint_c9_output_includes_backfilled_question_and_path(monkeypatch, tmp_path):
    import markdown_kb.app.indexer as indexer_mod
    from markdown_kb.app.schemas import QaStalenessFinding

    from kb_cli.main import app

    _write_qa_page(indexer_mod.WIKI_DIR, "stale-answer", question="What is the SLA?", status="live")
    finding = QaStalenessFinding(
        page_slug="stale-answer",
        stale_citations=["refund-policy#cancellation-window"],
        max_drift_days=10.0,
    )
    _patch_run_lint(monkeypatch, _lint_response_with_c9(finding))

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0, result.output
    assert "C9 Stale filed answers (1) — stale-qa:" in result.output
    assert "What is the SLA?" in result.output
    assert "wiki/qa/stale-answer.md" in result.output


def test_kb_lint_c9_output_falls_back_when_page_missing(monkeypatch):
    """A C9 finding whose page is not on disk still renders without crashing."""
    from markdown_kb.app.schemas import QaStalenessFinding

    from kb_cli.main import app

    finding = QaStalenessFinding(page_slug="vanished", stale_citations=["x#y"], max_drift_days=1.0)
    _patch_run_lint(monkeypatch, _lint_response_with_c9(finding))

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0, result.output
    assert "vanished" in result.output
    assert "question unavailable" in result.output.lower()
