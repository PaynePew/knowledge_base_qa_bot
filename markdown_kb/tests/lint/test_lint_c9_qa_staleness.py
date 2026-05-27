"""Unit tests for the Slice 6-5 C9 check: qa-staleness via entity-mtime drift.

Phase 6 PRD #78 §"Phase 5 lint amendment" — C9.

For each ``wiki/qa/*.md`` page with ``status: live``, C9 inspects each citation
in ``frontmatter.sources``, extracts the bare entity slug, and locates the file
under ``wiki/entities/`` (preferred) then ``wiki/concepts/``. If any entity
file's filesystem mtime exceeds the qa page's ``frontmatter.updated`` timestamp,
the qa page is flagged stale; ``max_drift_days`` reports the worst (largest)
drift in days across all stale citations.

Closes the "entity re-ingested, qa stranded" failure mode (PRD #78 Q6b).

Hermetic — no LLM invocation; the check is a pure filesystem scan.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml


def _write_entity_page(wiki_dir: Path, slug: str, *, subdir: str = "entities") -> Path:
    """Write a minimal entity (or concept) page used as a citation target."""
    sub = wiki_dir / subdir
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
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


def _set_mtime(path: Path, when: float) -> None:
    os.utime(str(path), (when, when))


class TestC9QaStaleness:
    """C9 flags live qa pages whose cited entity is newer than the qa frontmatter."""

    def test_newer_entity_flags_qa_as_stale(self, tmp_wiki_dir):
        """Entity mtime > qa.frontmatter.updated -> the qa page is flagged."""
        entity_path = _write_entity_page(tmp_wiki_dir, "vip-membership")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-vip-fee",
            sources=["vip-membership#main"],
            status="live",
            updated="2026-01-15T00:00:00Z",
        )
        # Touch the entity to "now" so it is unambiguously newer.
        _set_mtime(entity_path, time.time())

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert len(findings) == 1
        assert findings[0].page_slug == "qa-vip-fee"
        assert "vip-membership#main" in findings[0].stale_citations
        assert findings[0].max_drift_days > 0

    def test_older_entity_does_not_flag(self, tmp_wiki_dir):
        """Entity mtime <= qa.frontmatter.updated -> no finding."""
        entity_path = _write_entity_page(tmp_wiki_dir, "stable-entity")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-stable",
            sources=["stable-entity#main"],
            status="live",
            updated="2030-01-01T00:00:00Z",  # far future qa.updated
        )
        # Force entity mtime well in the past.
        _set_mtime(entity_path, 1577836800.0)  # 2020-01-01

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert findings == []

    def test_multiple_citations_only_newer_ones_contribute(self, tmp_wiki_dir):
        """A page citing several entities reports only the newer-mtime ones."""
        old_entity = _write_entity_page(tmp_wiki_dir, "old-entity")
        new_entity = _write_entity_page(tmp_wiki_dir, "new-entity")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-mixed",
            sources=["old-entity#main", "new-entity#main"],
            status="live",
            updated="2026-01-15T00:00:00Z",
        )
        _set_mtime(old_entity, 1577836800.0)  # 2020-01-01 — older than qa
        _set_mtime(new_entity, time.time())  # now — newer than qa

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert len(findings) == 1
        f = findings[0]
        assert f.page_slug == "qa-mixed"
        # Only the newer-mtime citation contributes.
        assert "new-entity#main" in f.stale_citations
        assert "old-entity#main" not in f.stale_citations

    def test_draft_page_is_not_checked(self, tmp_wiki_dir):
        """Only ``status: live`` qa pages are inspected by C9."""
        entity_path = _write_entity_page(tmp_wiki_dir, "vip-membership")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-draft-only",
            sources=["vip-membership#main"],
            status="draft",
            updated="2026-01-15T00:00:00Z",
        )
        _set_mtime(entity_path, time.time())

        from app.lint import _check_c9_qa_staleness

        findings = _check_c9_qa_staleness(tmp_wiki_dir)
        assert findings == []

    def test_concept_path_is_consulted_when_entity_missing(self, tmp_wiki_dir):
        """If entity is missing, the concept variant is checked."""
        concept_path = _write_entity_page(tmp_wiki_dir, "shipping-info", subdir="concepts")
        _write_qa_page(
            tmp_wiki_dir,
            "qa-ship",
            sources=["shipping-info#delivery"],
            status="live",
            updated="2026-01-15T00:00:00Z",
        )
        _set_mtime(concept_path, time.time())

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
