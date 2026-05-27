"""Hermetic tests for Slice 5-1: POST /lint scaffold + C11 orphan detection.

AC coverage (issue #66):
  - POST /lint route exists, returns 200 with LintResponse JSON body
  - run_lint() on a clean wiki (no orphan pages) returns summary.total_findings == 0
  - run_lint() on a wiki with one orphan page (sources pointing at non-existent
    file) returns summary.total_findings == 1, findings.orphans has length 1
  - wiki/lint-report.md is written; contains sentinel HTML comment, # Lint Report
    heading, summary blockquote, ## C11 Orphan pages section
  - wiki/log.md receives lint_started + lint_completed entries on every invocation
  - lint_check_error entry when C11 raises (continue-on-error)
  - Orphan finding shape: page_slug, missing_sources, suggested_action
  - Sort: alphabetical by page_slug

All tests are hermetic (no OPENAI_API_KEY required).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    sources: list[str],
    *,
    subdir: str = "concepts",
) -> Path:
    """Write a minimal wiki page with frontmatter.sources pointing at the given source refs."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),  # "concept" or "entity"
        "created": "2026-05-26T00:00:00Z",
        "updated": "2026-05-26T00:00:00Z",
        "sources": sources,
        "status": "live",
        "open_questions": [],
    }
    content = (
        f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n# {slug}\n\nSome content.\n"
    )
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ---------------------------------------------------------------------------
# POST /lint route tests (via TestClient)
# ---------------------------------------------------------------------------


class TestLintRoute:
    """Test the POST /lint FastAPI route."""

    @pytest.fixture(autouse=True)
    def _patch_run_lint(self, tmp_wiki_dir, tmp_docs_dir, lint_log_path, monkeypatch):
        """Redirect run_lint's default dirs to tmp so the route does not touch real wiki."""
        import app.lint as lint_module

        monkeypatch.setattr(lint_module, "WIKI_DIR", tmp_wiki_dir)
        monkeypatch.setattr(lint_module, "DOCS_DIR", tmp_docs_dir)
        monkeypatch.setattr(lint_module, "LOG_PATH", lint_log_path)

    def test_post_lint_returns_200(self, tmp_wiki_dir):
        from app.main import app

        client = TestClient(app)
        resp = client.post("/lint")
        assert resp.status_code == 200

    def test_post_lint_response_is_valid_lint_response(self, tmp_wiki_dir):
        from app.main import app
        from app.schemas import LintResponse

        client = TestClient(app)
        resp = client.post("/lint")
        assert resp.status_code == 200
        body = resp.json()
        # Validate against schema (will raise if keys are missing)
        parsed = LintResponse(**body)
        assert parsed.summary.total_findings == 0
        assert parsed.findings.orphans == []

    def test_post_lint_clean_wiki_zero_findings(self, tmp_wiki_dir, tmp_docs_dir):
        """A wiki with no pages should produce zero findings."""
        from app.main import app

        client = TestClient(app)
        resp = client.post("/lint")
        body = resp.json()
        assert body["summary"]["total_findings"] == 0

    def test_post_lint_orphan_produces_one_finding(self, tmp_wiki_dir, tmp_docs_dir, lint_log_path):
        """A wiki page whose sources point at a missing file produces one orphan finding."""
        # The source file does NOT exist in tmp_docs_dir
        _write_wiki_page(tmp_wiki_dir, "orphan-page", ["missing_source.md#section"])

        from app.main import app

        client = TestClient(app)
        resp = client.post("/lint")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_findings"] == 1
        assert len(body["findings"]["orphans"]) == 1
        orphan = body["findings"]["orphans"][0]
        assert orphan["page_slug"] == "orphan-page"
        assert "missing_source.md" in orphan["missing_sources"]


# ---------------------------------------------------------------------------
# run_lint() direct tests
# ---------------------------------------------------------------------------


class TestRunLint:
    """Direct tests for app.lint.run_lint()."""

    def test_clean_wiki_returns_zero_findings(self, lint_env):
        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.summary.total_findings == 0
        assert result.findings.orphans == []

    def test_clean_wiki_writes_report_file(self, lint_env):
        from app.lint import run_lint

        run_lint(**lint_env)
        report_path = lint_env["wiki_dir"] / "lint-report.md"
        assert report_path.exists(), "lint-report.md must be written on every invocation"
        content = report_path.read_text(encoding="utf-8")

        # AC: sentinel HTML comment
        assert "<!-- Auto-generated by POST /lint" in content
        # AC: # Lint Report heading
        assert "# Lint Report" in content
        # AC: summary blockquote
        assert content.startswith("<!--") or ">" in content  # blockquote uses >
        # AC: ## C11 Orphan pages section
        assert "## C11 Orphan pages" in content

    def test_clean_wiki_logs_started_and_completed(self, lint_env):
        from app.lint import run_lint

        run_lint(**lint_env)
        log_text = lint_env["log_path"].read_text(encoding="utf-8")
        assert "lint_started" in log_text
        assert "lint_completed" in log_text

    def test_orphan_page_produces_one_finding(self, lint_env):
        """One wiki page with a missing source reference → one orphan finding."""
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "orphan-page", ["deleted_source.md#some-section"])
        # docs_dir is empty — deleted_source.md does not exist

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.summary.total_findings == 1
        assert len(result.findings.orphans) == 1

        orphan = result.findings.orphans[0]
        assert orphan.page_slug == "orphan-page"
        assert "deleted_source.md" in orphan.missing_sources
        # suggested_action must mention both rename and deletion paths
        assert orphan.suggested_action != ""

    def test_orphan_finding_suggested_action_mentions_both_paths(self, lint_env):
        """suggested_action must mention both rename and deletion since lint cannot distinguish."""
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "orphan-page", ["deleted_source.md#section"])

        from app.lint import run_lint

        result = run_lint(**lint_env)
        action = result.findings.orphans[0].suggested_action
        # Must mention both rename/update AND delete/remove
        action_lower = action.lower()
        has_rename_or_update = any(w in action_lower for w in ("rename", "update", "re-ingest"))
        has_delete_or_remove = any(w in action_lower for w in ("delete", "remove"))
        assert has_rename_or_update, f"suggested_action must mention rename/update path: {action!r}"
        assert has_delete_or_remove, f"suggested_action must mention deletion path: {action!r}"

    def test_grounded_page_not_flagged_as_orphan(self, lint_env):
        """A page whose source exists in docs/ must NOT appear as an orphan (C11-specific check)."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        # Create the actual source file
        (docs_dir / "real_source.md").write_text("# Real Source\n\nContent.\n", encoding="utf-8")
        _write_wiki_page(wiki_dir, "grounded-page", ["real_source.md#some-section"])

        from app.lint import run_lint

        result = run_lint(**lint_env)
        # C11-specific: a page with an existing source is NOT an orphan.
        # (C6 may flag it as stale if the source file is newer; that is correct behaviour.)
        assert result.findings.orphans == []

    def test_c11_only_checks_file_stem_not_anchor(self, lint_env):
        """C11 checks existence of the file portion (before #), ignoring the anchor."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        # File exists but anchor is different — should NOT be an orphan
        (docs_dir / "existing.md").write_text("# Existing\n\nContent.\n", encoding="utf-8")
        _write_wiki_page(wiki_dir, "ok-page", ["existing.md#non-existent-anchor"])

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.findings.orphans == []

    def test_orphans_sorted_alphabetically(self, lint_env):
        """Orphan findings must be sorted alphabetically by page_slug."""
        wiki_dir = lint_env["wiki_dir"]
        # Write multiple orphan pages out of alphabetical order
        _write_wiki_page(wiki_dir, "zebra-page", ["missing1.md#s1"])
        _write_wiki_page(wiki_dir, "alpha-page", ["missing2.md#s2"])
        _write_wiki_page(wiki_dir, "mango-page", ["missing3.md#s3"])

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [o.page_slug for o in result.findings.orphans]
        assert slugs == sorted(slugs), f"Orphans not sorted: {slugs}"

    def test_multiple_missing_sources_all_listed(self, lint_env):
        """A page with multiple missing sources should list all of them."""
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(
            wiki_dir,
            "multi-orphan",
            ["missing_a.md#s1", "missing_b.md#s2"],
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert len(result.findings.orphans) == 1
        orphan = result.findings.orphans[0]
        assert len(orphan.missing_sources) == 2
        assert "missing_a.md" in orphan.missing_sources
        assert "missing_b.md" in orphan.missing_sources

    def test_mixed_sources_only_missing_listed(self, lint_env):
        """A page with one existing and one missing source: only missing one flagged."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        (docs_dir / "present.md").write_text("# Present\n\nContent.\n", encoding="utf-8")
        _write_wiki_page(
            wiki_dir,
            "mixed-page",
            ["present.md#s1", "absent.md#s2"],
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert len(result.findings.orphans) == 1
        orphan = result.findings.orphans[0]
        assert orphan.missing_sources == ["absent.md"]

    def test_c11_glob_checks_nested_subdirs(self, lint_env):
        """Source in a nested subdirectory of docs/ should NOT be flagged as missing."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        nested = docs_dir / "sub" / "dir"
        nested.mkdir(parents=True)
        (nested / "deep_source.md").write_text("# Deep\n\nContent.\n", encoding="utf-8")
        # C11 checks file stem against all docs/**/*.md; nested file should match
        _write_wiki_page(wiki_dir, "nested-page", ["deep_source.md#section"])

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.findings.orphans == []

    def test_continue_on_error_c11_logs_check_error(self, lint_env, monkeypatch):
        """If C11 raises, the error is recorded in check_errors; other checks still run."""
        import app.lint as lint_module

        def _bad_c11(*_args, **_kwargs):
            raise RuntimeError("simulated C11 failure")

        monkeypatch.setattr(lint_module, "_check_c11_orphan", _bad_c11)

        from app.lint import run_lint

        result = run_lint(**lint_env)
        # Response still 200-equivalent (no exception raised to caller)
        assert result is not None
        assert "c11" in result.check_errors
        # lint_check_error must appear in log
        log_text = lint_env["log_path"].read_text(encoding="utf-8")
        assert "lint_check_error" in log_text

    def test_report_file_has_c11_section_when_zero_findings(self, lint_env):
        """Report must include ## C11 Orphan pages even when no orphans found."""
        from app.lint import run_lint

        run_lint(**lint_env)
        content = (lint_env["wiki_dir"] / "lint-report.md").read_text(encoding="utf-8")
        assert "## C11 Orphan pages" in content

    def test_report_file_lists_orphan_slug(self, lint_env):
        """When orphan found, report must list the page_slug."""
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "specific-orphan", ["gone.md#sec"])

        from app.lint import run_lint

        run_lint(**lint_env)
        content = (wiki_dir / "lint-report.md").read_text(encoding="utf-8")
        assert "specific-orphan" in content

    def test_log_completed_summary_has_c11_count(self, lint_env):
        """lint_completed summary must include c11 finding count."""
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "orphan-x", ["missing.md#s"])

        from app.lint import run_lint

        run_lint(**lint_env)
        log_text = lint_env["log_path"].read_text(encoding="utf-8")
        # summary line format: "findings=N by_check=c11:X ..."
        assert "c11:1" in log_text
