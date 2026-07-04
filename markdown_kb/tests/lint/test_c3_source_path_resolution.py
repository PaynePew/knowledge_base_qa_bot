"""Hermetic tests for C3's ``source_path`` / ``source_resolution`` resolution
(issue #445, ADR-0029 follow-on).

AC coverage:
  - A Source basename that exists directly under ``docs_dir`` resolves to
    ``"docs/<basename>"`` with ``source_resolution == "resolved"``.
  - A Source basename that exists only in a NESTED ``docs/`` subdirectory
    (the live-reported case: a wiki page whose Source is
    ``docs/fake-docs/product_care.md``) resolves to the full nested
    repo-relative path — never the flat ``"docs/<basename>"`` guess that
    404'd for the Console's View Source button.
  - A Source basename matching NO file under ``docs_dir`` resolves to
    ``source_path=None`` / ``source_resolution="missing"``.
  - A Source basename matching 2+ files in different subdirectories
    resolves to ``source_path=None`` / ``source_resolution="ambiguous"`` —
    never guessed silently.
  - A finding with no source citation at all is treated as "missing".

All tests are hermetic (no OPENAI_API_KEY required).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_failed_grounding_page(
    wiki_dir: Path,
    slug: str,
    source: str,
    *,
    subdir: str = "concepts",
) -> Path:
    """Write a minimal wiki page with status=failed_grounding citing ``source``."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-07-04T00:00:00Z",
        "updated": "2026-07-04T00:00:00Z",
        "sources": [source] if source else [],
        "status": "failed_grounding",
        "open_questions": [],
        "grounding_failure": {"reason": "claim_unsupported", "unsupported_claims": []},
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n# {slug}\n\nFailed content.\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


class TestC3SourcePathResolution:
    """Tests for _check_c3_failed_grounding's source_path / source_resolution."""

    def test_flat_docs_file_resolves(self, lint_env):
        """A Source living directly under docs/ (no subdirectory) resolves."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "refund_policy.md").write_text("# Refund Policy\n", encoding="utf-8")
        _write_failed_grounding_page(wiki_dir, "flat-source", "refund_policy.md#section")

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir, docs_dir)
        assert len(findings) == 1
        assert findings[0].source_resolution == "resolved"
        assert findings[0].source_path == "docs/refund_policy.md"

    def test_nested_docs_file_resolves_to_full_path(self, lint_env):
        """The live-reported case (issue #445): a Source nested under a
        docs/ subdirectory must resolve to the FULL nested path, not the
        flat "docs/<basename>" guess that 404'd."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        nested = docs_dir / "fake-docs"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "product_care.md").write_text("# Product Care\n", encoding="utf-8")
        _write_failed_grounding_page(
            wiki_dir, "cleaning-instructions", "product_care.md#cleaning-instructions"
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir, docs_dir)
        assert len(findings) == 1
        assert findings[0].source_resolution == "resolved"
        assert findings[0].source_path == "docs/fake-docs/product_care.md"

    def test_missing_source_reports_distinct_state(self, lint_env):
        """A basename matching no file under docs/ is "missing", not a
        guessed path."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        _write_failed_grounding_page(wiki_dir, "ghost-source", "does_not_exist.md#section")

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir, docs_dir)
        assert len(findings) == 1
        assert findings[0].source_resolution == "missing"
        assert findings[0].source_path is None

    def test_ambiguous_basename_in_two_subdirs_reports_distinct_state(self, lint_env):
        """Two files sharing a basename in different subdirectories must
        never be silently resolved to either one — the finding reports
        "ambiguous" instead of guessing."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        (docs_dir / "alpha").mkdir(parents=True, exist_ok=True)
        (docs_dir / "beta").mkdir(parents=True, exist_ok=True)
        (docs_dir / "alpha" / "policy.md").write_text("# Alpha\n", encoding="utf-8")
        (docs_dir / "beta" / "policy.md").write_text("# Beta\n", encoding="utf-8")
        _write_failed_grounding_page(wiki_dir, "ambiguous-source", "policy.md#section")

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir, docs_dir)
        assert len(findings) == 1
        assert findings[0].source_resolution == "ambiguous"
        assert findings[0].source_path is None

    def test_no_source_citation_is_missing(self, lint_env):
        """A finding with no source citation at all (empty sources list)
        degrades to "missing" rather than raising."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]
        _write_failed_grounding_page(wiki_dir, "no-source", "")

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir, docs_dir)
        assert len(findings) == 1
        assert findings[0].source == ""
        assert findings[0].source_resolution == "missing"
        assert findings[0].source_path is None
