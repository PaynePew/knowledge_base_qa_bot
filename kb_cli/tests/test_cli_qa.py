"""Integration tests for the ``kb qa`` command group (issue #377).

``kb qa list / show / promote / discard`` wrap ``markdown_kb.app.qa``'s
public ``promote`` / ``delete`` and the read-only ``kb_mcp.qa_view`` helper
directly (ADR-0026 decision 3: gates resolve on human surfaces only — the
CLI's human is real, so its gate is real).

Tests use typer's CliRunner in-process, against real ``wiki/qa/*.md``
fixtures planted under the conftest-redirected ``tmp_path``/wiki dir (the
autouse ``_isolate_module_state`` fixture in conftest.py redirects
``markdown_kb.app.indexer.WIKI_DIR``, which both ``qa.py`` and ``qa_view.py``
resolve at call time).

``kb qa list`` mocks ``run_lint`` (matching the established pattern in
``test_cli_lint.py`` — no existing kb_cli test drives the real ``run_lint``,
since doing so would need ``markdown_kb.app.lint``'s own WIKI_DIR/DOCS_DIR
redirected, which this suite's conftest does not provide) while still
exercising the REAL C9 question-backfill file read against a planted
fixture.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _write_qa_page(
    wiki_dir: Path,
    slug: str,
    *,
    question: str,
    status: str,
    count: int = 1,
    sources: list[str] | None = None,
    body: str = "Refunds are processed within 5 business days.",
) -> Path:
    """Write a real wiki/qa/<slug>.md fixture mirroring qa._render_qa_page's shape."""
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "id": slug,
        "type": "qa",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": sources or ["refund-policy#cancellation-window"],
        "status": status,
        "open_questions": [],
        "question": question,
        "count": count,
    }
    fm_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    content = f"<!-- Auto-filed by POST /chat. -->\n\n---\n{fm_yaml}---\n\n{body}\n"
    page_path = qa_dir / f"{slug}.md"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _wiki_dir() -> Path:
    import markdown_kb.app.indexer as indexer_mod

    return indexer_mod.WIKI_DIR


# ---------------------------------------------------------------------------
# kb qa show
# ---------------------------------------------------------------------------


def test_qa_show_not_found_exits_nonzero():
    from kb_cli.main import app

    result = runner.invoke(app, ["qa", "show", "never-filed"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_qa_show_prints_question_body_sources_status():
    from kb_cli.main import app

    _write_qa_page(
        _wiki_dir(),
        "popular-question",
        question="How do refunds work?",
        status="draft",
        sources=["refund-policy#cancellation-window"],
        body="Refunds are processed within 5 business days.",
    )

    result = runner.invoke(app, ["qa", "show", "popular-question"])
    assert result.exit_code == 0, result.output
    assert "How do refunds work?" in result.output
    assert "draft" in result.output
    assert "refund-policy#cancellation-window" in result.output
    assert "Refunds are processed within 5 business days." in result.output
    assert "wiki/qa/popular-question.md" in result.output


# ---------------------------------------------------------------------------
# kb qa promote
# ---------------------------------------------------------------------------


def test_qa_promote_not_found_exits_nonzero():
    from kb_cli.main import app

    result = runner.invoke(app, ["qa", "promote", "never-filed"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_qa_promote_flips_status_and_reindexes(monkeypatch):
    """promote() flips draft->live on disk and the command triggers a reindex."""
    import markdown_kb.app.indexer as wiki_indexer

    from kb_cli.main import app

    _write_qa_page(_wiki_dir(), "popular-question", question="How do refunds work?", status="draft")

    # Mirrors test_cli.py's kb-index pattern: mock build_index to avoid the
    # real SOURCE_DIRS scan (SOURCE_DIRS is bound at indexer import time to
    # the production wiki/ subdirs and is not redirected by this suite's
    # conftest — only WIKI_DIR is).
    monkeypatch.setattr(wiki_indexer, "build_index", lambda: (2, 7))

    result = runner.invoke(app, ["qa", "promote", "popular-question"])
    assert result.exit_code == 0, result.output
    assert "status=live" in result.output
    assert "Reindexed 2 file(s), 7 section(s)." in result.output

    # The real qa.promote() ran against the planted file — verify on disk via
    # the same PUBLIC split_frontmatter helper qa_view uses (no reach-in to
    # qa.py's private readers, per CODING_STANDARD §2.4).
    from markdown_kb.app.indexer import split_frontmatter

    raw = (_wiki_dir() / "qa" / "popular-question.md").read_text(encoding="utf-8")
    metadata, _body = split_frontmatter(raw)
    assert metadata["status"] == "live"


def test_qa_promote_corrupt_frontmatter_exits_nonzero():
    from kb_cli.main import app

    qa_dir = _wiki_dir() / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "broken.md").write_text(
        "---\nstatus: not-a-real-status\n---\n\nBody.\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["qa", "promote", "broken"])
    assert result.exit_code != 0
    assert "corrupt" in result.output.lower()


# ---------------------------------------------------------------------------
# kb qa discard
# ---------------------------------------------------------------------------


def test_qa_discard_not_found_exits_nonzero():
    from kb_cli.main import app

    result = runner.invoke(app, ["qa", "discard", "never-filed"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_qa_discard_removes_draft_page():
    from kb_cli.main import app

    page_path = _write_qa_page(
        _wiki_dir(), "throwaway-draft", question="A bad draft?", status="draft"
    )

    result = runner.invoke(app, ["qa", "discard", "throwaway-draft"])
    assert result.exit_code == 0, result.output
    assert "Discarded throwaway-draft" in result.output
    assert not page_path.exists()


def test_qa_discard_refuses_live_page_with_clear_message():
    """AC: discard refuses a live page with a clear message."""
    from kb_cli.main import app

    page_path = _write_qa_page(_wiki_dir(), "live-answer", question="A live answer?", status="live")

    result = runner.invoke(app, ["qa", "discard", "live-answer"])
    assert result.exit_code != 0
    assert "live" in result.output.lower()
    assert "refused" in result.output.lower()
    # The clear message names the remediation path instead of just failing silently.
    assert "re-ingest" in result.output.lower()
    assert page_path.exists(), "a refused discard must not delete the file"


# ---------------------------------------------------------------------------
# kb qa list
# ---------------------------------------------------------------------------


def _make_qa_lint_response(*, candidates=None, stale=None):
    from markdown_kb.app.schemas import LintFindings, LintResponse, LintSummary

    findings = LintFindings(
        promotion_candidates=candidates or [],
        stale_filed_answers=stale or [],
    )
    total = len(findings.promotion_candidates) + len(findings.stale_filed_answers)
    summary = LintSummary(
        total_findings=total,
        findings_by_check={
            "c8": len(findings.promotion_candidates),
            "c9": len(findings.stale_filed_answers),
        },
        generated_at="2026-07-03T00:00:00Z",
    )
    return LintResponse(
        report_path="wiki/lint-report.md",
        findings=findings,
        summary=summary,
        check_errors={},
    )


def test_qa_list_reports_empty_queue(monkeypatch):
    import markdown_kb.app.lint as lint_mod

    from kb_cli.main import app

    monkeypatch.setattr(lint_mod, "run_lint", lambda **_kw: _make_qa_lint_response())

    result = runner.invoke(app, ["qa", "list"])
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower() or "no filed answers" in result.output.lower()


def test_qa_list_shows_c8_candidate_with_question_and_path(monkeypatch):
    import markdown_kb.app.lint as lint_mod
    from markdown_kb.app.schemas import PromotionCandidateFinding

    from kb_cli.main import app

    candidate = PromotionCandidateFinding(
        slug="popular-question",
        question="How do refunds work?",
        count=5,
        age_days=3.0,
        cited_count=1,
    )
    monkeypatch.setattr(
        lint_mod, "run_lint", lambda **_kw: _make_qa_lint_response(candidates=[candidate])
    )

    result = runner.invoke(app, ["qa", "list"])
    assert result.exit_code == 0, result.output
    assert "popular-question" in result.output
    assert "How do refunds work?" in result.output
    assert "wiki/qa/popular-question.md" in result.output
    assert "draft" in result.output


def test_qa_list_shows_c9_finding_with_backfilled_question_and_path(monkeypatch):
    """C9's question is backfilled by reading the real planted page."""
    import markdown_kb.app.lint as lint_mod
    from markdown_kb.app.schemas import QaStalenessFinding

    from kb_cli.main import app

    _write_qa_page(
        _wiki_dir(),
        "stale-answer",
        question="What is the SLA?",
        status="live",
    )
    stale = QaStalenessFinding(
        page_slug="stale-answer",
        stale_citations=["refund-policy#cancellation-window"],
        max_drift_days=10.0,
    )
    monkeypatch.setattr(lint_mod, "run_lint", lambda **_kw: _make_qa_lint_response(stale=[stale]))

    result = runner.invoke(app, ["qa", "list"])
    assert result.exit_code == 0, result.output
    assert "stale-answer" in result.output
    assert "What is the SLA?" in result.output
    assert "wiki/qa/stale-answer.md" in result.output
    assert "live" in result.output


def test_qa_list_c9_falls_back_when_page_unreadable(monkeypatch):
    """A C9 finding whose page is no longer on disk still renders (no crash)."""
    import markdown_kb.app.lint as lint_mod
    from markdown_kb.app.schemas import QaStalenessFinding

    from kb_cli.main import app

    stale = QaStalenessFinding(
        page_slug="vanished-page",
        stale_citations=["refund-policy#cancellation-window"],
        max_drift_days=2.0,
    )
    monkeypatch.setattr(lint_mod, "run_lint", lambda **_kw: _make_qa_lint_response(stale=[stale]))

    result = runner.invoke(app, ["qa", "list"])
    assert result.exit_code == 0, result.output
    assert "vanished-page" in result.output
    assert "question unavailable" in result.output.lower()
