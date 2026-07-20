"""Unit tests for the C9 check: qa-staleness via entity frontmatter timestamp.

Phase 6 PRD #78 §"Phase 5 lint amendment" — C9 (updated in #349).

For each ``wiki/qa/*.md`` page with ``status: live``, C9 inspects each citation
in ``frontmatter.sources``, extracts the bare entity slug, and locates the file
under ``wiki/entities/`` (preferred) then ``wiki/concepts/``. If any entity
page's ``frontmatter.updated`` timestamp exceeds the qa page's
``frontmatter.updated`` timestamp, the qa page is flagged stale;
``max_drift_days`` reports the worst (largest) drift in days across all stale
citations.

Closes the "entity re-ingested, qa stranded" failure mode (PRD #78 Q6b).
Detection is content-stable: timestamps come from file content, not filesystem
mtime, so a fresh git clone with unchanged content does NOT trigger false
positives (AC1 of issue #349).

Hermetic — no LLM invocation; the check is a pure file-content scan.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_entity_page(
    wiki_dir: Path,
    slug: str,
    *,
    subdir: str = "entities",
    updated: str = "2026-01-01T00:00:00Z",
) -> Path:
    """Write a minimal entity (or concept) page used as a citation target."""
    sub = wiki_dir / subdir
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-01-01T00:00:00Z",
        "updated": updated,
        "sources": [f"{slug}.md#main"],
        "status": "live",
        "open_questions": [],
    }
    body = f"# {slug}\n\nEntity body for `{slug}`."
    path.write_text(
        f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def _write_qa_page(
    wiki_dir: Path,
    slug: str,
    *,
    sources: list[str],
    status: str = "live",
    updated: str = "2026-02-01T00:00:00Z",
) -> Path:
    """Write a qa page with the requested status / updated / citations."""
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": "qa",
        "created": updated,
        "updated": updated,
        "sources": sources,
        "status": status,
        "question": f"Question for {slug}?",
        "count": 1,
        "open_questions": [],
    }
    body = f"# Question for {slug}?\n\nAnswer for `{slug}`."
    path.write_text(
        f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


class TestC9QaStaleness:
    """C9 flags live qa pages whose cited entity frontmatter.updated > qa.frontmatter.updated."""

    # ------------------------------------------------------------------
    # AC1 / AC2 — content-stable detection (issue #349)
    # ------------------------------------------------------------------

    def test_c9_no_false_positive_on_fresh_clone(self, tmp_wiki_dir):
        """AC1: entity.frontmatter.updated <= qa.frontmatter.updated → no finding.

        Simulates a fresh git clone: mtime is 'now' for all files but the
        entity was last ingested BEFORE the qa was filed, so no staleness.
        """
        _write_entity_page(tmp_wiki_dir, "stable-entity", updated="2026-01-01T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-stable",
            sources=["stable-entity#main"],
            status="live",
            updated="2026-02-01T00:00:00Z",  # qa filed after entity last ingest
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert findings == []

    def test_c9_drift_within_grace_period_not_flagged(self, tmp_wiki_dir):
        """issue #639: a routine re-ingest 2 days after filing stays quiet —
        only drift beyond the 3.0-day default grace flags."""
        _write_entity_page(tmp_wiki_dir, "fresh-entity", updated="2026-02-03T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-fresh",
            sources=["fresh-entity#main"],
            status="live",
            updated="2026-02-01T00:00:00Z",  # entity newer by 2d < 3d grace
        )

        from app.lint import _check_c9_qa_staleness

        assert _check_c9_qa_staleness(tmp_wiki_dir) == []

    def test_c9_boundary_just_under_grace_not_flagged(self, tmp_wiki_dir):
        """issue #639 AC: 2.9d drift (just under the 3.0d default grace) stays quiet."""
        # 2.9d after 2026-02-01T00:00:00Z = 2026-02-03T21:36:00Z
        _write_entity_page(tmp_wiki_dir, "fresh-entity", updated="2026-02-03T21:36:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-fresh",
            sources=["fresh-entity#main"],
            status="live",
            updated="2026-02-01T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        assert _check_c9_qa_staleness(tmp_wiki_dir) == []

    def test_c9_exact_grace_boundary_not_flagged(self, tmp_wiki_dir):
        """issue #639: drift of exactly 3.0d does NOT flag (spec: '<= 3 days
        should NOT flag' — the predicate is strict >)."""
        _write_entity_page(tmp_wiki_dir, "fresh-entity", updated="2026-02-04T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-fresh",
            sources=["fresh-entity#main"],
            status="live",
            updated="2026-02-01T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        assert _check_c9_qa_staleness(tmp_wiki_dir) == []

    def test_c9_boundary_just_over_grace_flags(self, tmp_wiki_dir):
        """issue #639 AC: 3.1d drift (just over the 3.0d default grace) flags."""
        # 3.1d after 2026-02-01T00:00:00Z = 2026-02-04T02:24:00Z
        _write_entity_page(tmp_wiki_dir, "fresh-entity", updated="2026-02-04T02:24:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-fresh",
            sources=["fresh-entity#main"],
            status="live",
            updated="2026-02-01T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert [f.page_slug for f in findings] == ["qa-fresh"]
        assert findings[0].max_drift_days == 3.1

    def test_c9_negative_grace_env_falls_back_to_default(self, tmp_wiki_dir, monkeypatch):
        """issue #639: a negative KB_LINT_C9_GRACE_DAYS clamps to the 3.0d
        default (mirrors the KB_LINT_C5_* negative-value guard)."""
        monkeypatch.setenv("KB_LINT_C9_GRACE_DAYS", "-1")
        _write_entity_page(tmp_wiki_dir, "fresh-entity", updated="2026-02-03T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-fresh",
            sources=["fresh-entity#main"],
            status="live",
            updated="2026-02-01T00:00:00Z",  # 2d drift < default grace
        )

        from app.lint import _check_c9_qa_staleness

        assert _check_c9_qa_staleness(tmp_wiki_dir) == []

    def test_c9_grace_env_override(self, tmp_wiki_dir, monkeypatch):
        """issue #639: KB_LINT_C9_GRACE_DAYS=0 restores flag-any-drift, read
        per request (no restart)."""
        monkeypatch.setenv("KB_LINT_C9_GRACE_DAYS", "0")
        _write_entity_page(tmp_wiki_dir, "fresh-entity", updated="2026-02-03T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-fresh",
            sources=["fresh-entity#main"],
            status="live",
            updated="2026-02-01T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert [f.page_slug for f in findings] == ["qa-fresh"]

    def test_c9_newer_entity_flags_qa(self, tmp_wiki_dir):
        """AC2: entity.frontmatter.updated > qa.frontmatter.updated → finding emitted."""
        _write_entity_page(tmp_wiki_dir, "vip-membership", updated="2026-03-01T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-vip-fee",
            sources=["vip-membership#main"],
            status="live",
            updated="2026-01-15T00:00:00Z",  # qa filed before entity was re-ingested
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert len(findings) == 1
        assert findings[0].page_slug == "qa-vip-fee"
        assert "vip-membership#main" in findings[0].stale_citations
        assert findings[0].max_drift_days > 0

    # ------------------------------------------------------------------
    # Existing behaviour (updated to use frontmatter timestamps)
    # ------------------------------------------------------------------

    def test_newer_entity_flags_qa_as_stale(self, tmp_wiki_dir):
        """entity.frontmatter.updated > qa.frontmatter.updated → the qa page is flagged."""
        _write_entity_page(tmp_wiki_dir, "vip-membership", updated="2026-06-01T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-vip-fee",
            sources=["vip-membership#main"],
            status="live",
            updated="2026-01-15T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert len(findings) == 1
        assert findings[0].page_slug == "qa-vip-fee"
        assert "vip-membership#main" in findings[0].stale_citations
        assert findings[0].max_drift_days > 0

    def test_older_entity_does_not_flag(self, tmp_wiki_dir):
        """entity.frontmatter.updated <= qa.frontmatter.updated → no finding."""
        _write_entity_page(tmp_wiki_dir, "stable-entity", updated="2020-01-01T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-stable",
            sources=["stable-entity#main"],
            status="live",
            updated="2030-01-01T00:00:00Z",  # far future qa.updated
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert findings == []

    def test_multiple_citations_only_newer_ones_contribute(self, tmp_wiki_dir):
        """A page citing several entities reports only the newer-updated ones."""
        _write_entity_page(tmp_wiki_dir, "old-entity", updated="2020-01-01T00:00:00Z")
        _write_entity_page(tmp_wiki_dir, "new-entity", updated="2026-06-01T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-mixed",
            sources=["old-entity#main", "new-entity#main"],
            status="live",
            updated="2026-01-15T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert len(findings) == 1
        f = findings[0]
        assert f.page_slug == "qa-mixed"
        # Only the newer-updated citation contributes.
        assert "new-entity#main" in f.stale_citations
        assert "old-entity#main" not in f.stale_citations

    def test_draft_page_is_not_checked(self, tmp_wiki_dir):
        """Only ``status: live`` qa pages are inspected by C9."""
        _write_entity_page(tmp_wiki_dir, "vip-membership", updated="2026-06-01T00:00:00Z")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-draft-only",
            sources=["vip-membership#main"],
            status="draft",
            updated="2026-01-15T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert findings == []

    def test_concept_path_is_consulted_when_entity_missing(self, tmp_wiki_dir):
        """If entity is missing, the concept variant is checked."""
        _write_entity_page(
            tmp_wiki_dir, "shipping-info", subdir="concepts", updated="2026-06-01T00:00:00Z"
        )
        _write_qa_page(
            tmp_wiki_dir,
            "qa-ship",
            sources=["shipping-info#delivery"],
            status="live",
            updated="2026-01-15T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert len(findings) == 1
        assert findings[0].page_slug == "qa-ship"

    def test_missing_entity_file_does_not_flag(self, tmp_wiki_dir):
        """Citation pointing at a nonexistent entity is silently skipped (not C9's concern)."""
        _write_qa_page(
            tmp_wiki_dir,
            "qa-dangling",
            sources=["nonexistent-entity#x"],
            status="live",
            updated="2026-01-15T00:00:00Z",
        )

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert findings == []
