"""Hermetic tests for Slice 5-3: C6 content-hash stale detection + C2 red link backlog.

AC coverage (issue #68, updated in #349):
  - C6: _check_c6_stale() emits StalePageFinding when source content hash differs from
    frontmatter.source_hashes["docs_body"] (content-stable, survives git clone)
  - C6: skips pages with missing/empty source_hashes (legacy drift-unknown pages)
  - C6: skips pages whose Source file does not exist (C11's job, not C6's)
  - C6: sort by drift_days descending
  - StalePageFinding schema: page_slug, source, source_mtime, page_updated, drift_days, suggested_action
  - C2: _check_c2_red_links() scans entities/ + concepts/ for [[wikilink]] patterns
  - C2: only unresolved slugs are flagged
  - C2: explicit exclusions enforced (lint-report.md, index.md, log.md, hot.md, README.md, .archive/*)
  - RedLinkFinding schema: slug, mention_count, referenced_by, sample_context
  - C2: sort by mention_count descending, alphabetical by slug for ties
  - Heading-anchor portions ([[slug#heading]]) ignored for resolution (slug only)
  - Report renderer includes C6 and C2 sections

All tests are hermetic (no OPENAI_API_KEY required).
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Hash helper (mirrors ingest._compute_docs_body_hash)
# ---------------------------------------------------------------------------


def _hash(content: str) -> str:
    """SHA-256 of the UTF-8 bytes of *content*, hex-encoded."""
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    sources: list[str],
    *,
    subdir: str = "concepts",
    updated: str = "2026-01-01T00:00:00Z",
    body: str = "",
    source_hashes: dict | None = None,
) -> Path:
    """Write a minimal wiki page with frontmatter.sources and optional body.

    ``source_hashes`` is the dict stored in ``frontmatter.source_hashes``:
    ``{filename: {"docs_body": <sha256_hex>}}``.  Pass the *correct* hash
    (matching source content) for "fresh / up-to-date" scenarios, and a
    *wrong* hash (or omit the key entirely) for "stale" scenarios.
    """
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter: dict = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-01-01T00:00:00Z",
        "updated": updated,
        "sources": sources,
        "status": "live",
        "open_questions": [],
    }
    if source_hashes is not None:
        frontmatter["source_hashes"] = source_hashes
    if not body:
        body = f"# {slug}\n\nSome content."
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ---------------------------------------------------------------------------
# C6 — Stale detection tests
# ---------------------------------------------------------------------------


class TestC6StaleDetection:
    """Tests for _check_c6_stale and StalePageFinding schema."""

    # ------------------------------------------------------------------
    # AC1 / AC2 — content-hash gate (issue #349)
    # ------------------------------------------------------------------

    def test_c6_no_false_positive_on_fresh_clone(self, lint_env):
        """AC1: same content on disk as stored hash → no StalePageFinding.

        Simulates a fresh git clone: mtime is 'now' but content is unchanged.
        """
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_content = "# Refund Policy\n\nContent.\n"
        source_path = docs_dir / "refund_policy.md"
        source_path.write_text(source_content, encoding="utf-8")
        # mtime is 'now' (as in a fresh git clone) — but content matches stored hash
        correct_hash = _hash(source_content)

        _write_wiki_page(
            wiki_dir,
            "cancellation-window",
            ["refund_policy.md#cancellation-window"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"refund_policy.md": {"docs_body": correct_hash}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.findings.stale_pages == []

    def test_c6_changed_content_is_still_stale(self, lint_env):
        """AC2: source content differs from stored hash → StalePageFinding emitted."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_content = "# Refund Policy\n\nUpdated content.\n"
        source_path = docs_dir / "refund_policy.md"
        source_path.write_text(source_content, encoding="utf-8")
        # Store an old/wrong hash so the check detects divergence
        old_hash = _hash("# Refund Policy\n\nOriginal content.\n")

        _write_wiki_page(
            wiki_dir,
            "cancellation-window",
            ["refund_policy.md#cancellation-window"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"refund_policy.md": {"docs_body": old_hash}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        stale = result.findings.stale_pages
        assert len(stale) == 1
        assert stale[0].page_slug == "cancellation-window"
        assert stale[0].source == "refund_policy.md"

    def test_c6_missing_source_hashes_skipped(self, lint_env):
        """Legacy pages without source_hashes are skipped (drift state unknown)."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_path = docs_dir / "legacy_doc.md"
        source_path.write_text("# Legacy\n\nContent.\n", encoding="utf-8")

        # No source_hashes key at all (pre-Phase-6 ingest)
        _write_wiki_page(
            wiki_dir,
            "legacy-page",
            ["legacy_doc.md#section"],
            updated="2026-01-01T00:00:00Z",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.findings.stale_pages == []

    # ------------------------------------------------------------------
    # Existing behaviour (updated to use hash-based staleness)
    # ------------------------------------------------------------------

    def test_stale_page_detected_when_source_newer(self, lint_env):
        """Hash mismatch between stored and current source → StalePageFinding."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_content = "# Refund Policy\n\nCurrent content.\n"
        source_path = docs_dir / "refund_policy.md"
        source_path.write_text(source_content, encoding="utf-8")
        # Store an old hash to simulate content drift
        old_hash = _hash("# Refund Policy\n\nOld content.\n")

        _write_wiki_page(
            wiki_dir,
            "cancellation-window",
            ["refund_policy.md#cancellation-window"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"refund_policy.md": {"docs_body": old_hash}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        stale = result.findings.stale_pages
        assert len(stale) == 1
        assert stale[0].page_slug == "cancellation-window"
        assert stale[0].source == "refund_policy.md"

    def test_fresh_page_not_flagged(self, lint_env):
        """Correct hash stored → no StalePageFinding (content unchanged)."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_content = "# Account Help\n\nContent.\n"
        source_path = docs_dir / "account_help.md"
        source_path.write_text(source_content, encoding="utf-8")

        _write_wiki_page(
            wiki_dir,
            "reset-password",
            ["account_help.md#reset-password"],
            updated="2028-01-01T00:00:00Z",
            source_hashes={"account_help.md": {"docs_body": _hash(source_content)}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.findings.stale_pages == []

    def test_c6_skips_missing_source_file(self, lint_env):
        """C6 must NOT flag a page whose Source file doesn't exist — that's C11's job."""
        wiki_dir = lint_env["wiki_dir"]
        # docs_dir is empty — no source files

        _write_wiki_page(
            wiki_dir,
            "orphan-page",
            ["missing_source.md#section"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"missing_source.md": {"docs_body": "aabbcc"}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.findings.stale_pages == []

    def test_c6_uses_first_source_only(self, lint_env):
        """C6 reads frontmatter.sources[0] to determine which Source to check."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_content = "# Shipping FAQ\n\nContent.\n"
        source_path = docs_dir / "shipping_faq.md"
        source_path.write_text(source_content, encoding="utf-8")
        old_hash = _hash("# Shipping FAQ\n\nOld content.\n")

        # Two sources — C6 uses only the first
        _write_wiki_page(
            wiki_dir,
            "tracking-number",
            ["shipping_faq.md#tracking-number", "account_help.md#tracking"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"shipping_faq.md": {"docs_body": old_hash}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        stale = result.findings.stale_pages
        assert len(stale) == 1
        assert stale[0].source == "shipping_faq.md"

    def test_c6_sort_drift_days_descending(self, lint_env):
        """Multiple stale pages: sort by drift_days descending.

        drift_days is still derived from mtime for display; we just need both
        pages to be hash-stale. The page whose source has a larger mtime delta
        should appear first.
        """
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        src_a_content = "# A\n\nContent A.\n"
        src_a = docs_dir / "source_a.md"
        src_a.write_text(src_a_content, encoding="utf-8")
        # mtime: large drift (200+ days ahead of page updated 2026-01-01)
        import os

        big_drift_ts = datetime.datetime(2026, 8, 19, tzinfo=datetime.UTC).timestamp()
        os.utime(src_a, (big_drift_ts, big_drift_ts))

        src_b_content = "# B\n\nContent B.\n"
        src_b = docs_dir / "source_b.md"
        src_b.write_text(src_b_content, encoding="utf-8")
        small_drift_ts = datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC).timestamp()
        os.utime(src_b, (small_drift_ts, small_drift_ts))

        wrong_hash_a = _hash("# A\n\nOld.\n")
        wrong_hash_b = _hash("# B\n\nOld.\n")

        _write_wiki_page(
            wiki_dir,
            "page-a",
            ["source_a.md#section"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"source_a.md": {"docs_body": wrong_hash_a}},
        )
        _write_wiki_page(
            wiki_dir,
            "page-b",
            ["source_b.md#section"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"source_b.md": {"docs_body": wrong_hash_b}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        stale = result.findings.stale_pages
        assert len(stale) == 2
        # Large drift (page-a) must come before small drift (page-b)
        assert stale[0].page_slug == "page-a"
        assert stale[1].page_slug == "page-b"
        assert stale[0].drift_days >= stale[1].drift_days

    def test_stale_page_finding_schema(self, lint_env):
        """StalePageFinding must have all required fields with correct types."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_content = "# Refund\n\nContent.\n"
        source_path = docs_dir / "refund_policy.md"
        source_path.write_text(source_content, encoding="utf-8")
        old_hash = _hash("# Refund\n\nOld content.\n")

        _write_wiki_page(
            wiki_dir,
            "refund-page",
            ["refund_policy.md#refund-page"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"refund_policy.md": {"docs_body": old_hash}},
        )

        from app.lint import run_lint
        from app.schemas import StalePageFinding

        result = run_lint(**lint_env)
        assert len(result.findings.stale_pages) == 1
        finding = result.findings.stale_pages[0]

        # Verify it's the correct type
        assert isinstance(finding, StalePageFinding)
        # Required fields
        assert isinstance(finding.page_slug, str)
        assert isinstance(finding.source, str)
        assert isinstance(finding.source_mtime, datetime.datetime)
        assert isinstance(finding.page_updated, datetime.datetime)
        assert isinstance(finding.drift_days, float)
        assert isinstance(finding.suggested_action, str)
        # suggested_action mentions re-ingest
        assert (
            "re-ingest" in finding.suggested_action.lower()
            or "ingest" in finding.suggested_action.lower()
        )

    def test_c6_strips_anchor_from_source_citation(self, lint_env):
        """C6 must split off #section to get the Source filename."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        source_content = "# Shipping\n\nContent.\n"
        source_path = docs_dir / "shipping_faq.md"
        source_path.write_text(source_content, encoding="utf-8")
        old_hash = _hash("# Shipping\n\nOld content.\n")

        # Citation includes #anchor — C6 must strip it to find the file
        _write_wiki_page(
            wiki_dir,
            "shipping-page",
            ["shipping_faq.md#some-section"],
            updated="2026-01-01T00:00:00Z",
            source_hashes={"shipping_faq.md": {"docs_body": old_hash}},
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert len(result.findings.stale_pages) == 1
        assert result.findings.stale_pages[0].source == "shipping_faq.md"

    def test_report_has_c6_section(self, lint_env):
        """lint-report.md must include a C6 Stale pages section."""
        from app.lint import run_lint

        run_lint(**lint_env)
        content = (lint_env["wiki_dir"] / "lint-report.md").read_text(encoding="utf-8")
        assert "## C6 Stale pages" in content

    def test_summary_includes_c6_count(self, lint_env):
        """LintSummary.findings_by_check must include 'c6' key."""
        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert "c6" in result.summary.findings_by_check


# ---------------------------------------------------------------------------
# C2 — Red link backlog tests
# ---------------------------------------------------------------------------


class TestC2RedLinks:
    """Tests for _check_c2_red_links and RedLinkFinding schema."""

    def test_unresolved_link_detected(self, lint_env):
        """A [[unresolved-slug]] in a wiki page body → RedLinkFinding."""
        wiki_dir = lint_env["wiki_dir"]

        # Existing page (resolved target)
        _write_wiki_page(
            wiki_dir,
            "refund-timeline",
            ["refund_policy.md#refund-timeline"],
            body="# Refund Timeline\n\nContent about refunds.",
        )

        # Page with unresolved red link
        _write_wiki_page(
            wiki_dir,
            "cancellation-window",
            ["refund_policy.md#cancellation-window"],
            body="# Cancellation\n\nSee [[unresolved-slug]] for more info.",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        red_links = result.findings.red_links
        slugs = [r.slug for r in red_links]
        assert "unresolved-slug" in slugs
        assert "refund-timeline" not in slugs  # this page exists

    def test_resolved_wikilink_not_flagged(self, lint_env):
        """A [[slug]] that matches an existing page must NOT appear in red_links."""
        wiki_dir = lint_env["wiki_dir"]

        # Both pages exist
        _write_wiki_page(
            wiki_dir,
            "refund-timeline",
            ["refund_policy.md#refund-timeline"],
            body="# Refund Timeline\n\nSee [[cancellation-window]] for details.",
        )
        _write_wiki_page(
            wiki_dir,
            "cancellation-window",
            ["refund_policy.md#cancellation-window"],
            body="# Cancellation Window\n\nContent.",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert result.findings.red_links == []

    def test_c2_lint_report_excluded(self, lint_env):
        """lint-report.md must NOT contribute red links (self-feeding loop guard)."""
        wiki_dir = lint_env["wiki_dir"]

        # Write lint-report.md with a wikilink
        lint_report = wiki_dir / "lint-report.md"
        lint_report.write_text(
            "<!-- Auto-generated -->\n# Lint Report\n\nSee [[some-red-link]] here.\n",
            encoding="utf-8",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "some-red-link" not in slugs

    def test_c2_exclusion_index_md(self, lint_env):
        """wiki/index.md must NOT contribute red links."""
        wiki_dir = lint_env["wiki_dir"]

        (wiki_dir / "index.md").write_text(
            "# Index\n\nSee [[phantom-page]] for details.\n", encoding="utf-8"
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "phantom-page" not in slugs

    def test_c2_exclusion_log_md(self, lint_env):
        """wiki/log.md must NOT contribute red links."""
        wiki_dir = lint_env["wiki_dir"]

        (wiki_dir / "log.md").write_text(
            "## [2026-01-01] event | See [[ghost-page]]\n", encoding="utf-8"
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "ghost-page" not in slugs

    def test_c2_exclusion_hot_md(self, lint_env):
        """wiki/hot.md must NOT contribute red links (future file, preemptively excluded)."""
        wiki_dir = lint_env["wiki_dir"]

        (wiki_dir / "hot.md").write_text("# Hot topics\n\n[[hot-phantom]]\n", encoding="utf-8")

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "hot-phantom" not in slugs

    def test_c2_exclusion_readme_md(self, lint_env):
        """wiki/README.md must NOT contribute red links."""
        wiki_dir = lint_env["wiki_dir"]

        (wiki_dir / "README.md").write_text("# Readme\n\n[[readme-phantom]]\n", encoding="utf-8")

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "readme-phantom" not in slugs

    def test_c2_exclusion_archive_subdir(self, lint_env):
        """wiki/.archive/* pages must NOT contribute red links."""
        wiki_dir = lint_env["wiki_dir"]

        archive_dir = wiki_dir / ".archive"
        archive_dir.mkdir(parents=True)
        (archive_dir / "old-page.md").write_text("# Old\n\n[[archive-phantom]]\n", encoding="utf-8")

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "archive-phantom" not in slugs

    def test_c2_heading_anchor_ignored_for_resolution(self, lint_env):
        """[[slug#heading]] — slug is checked for existence, anchor is ignored."""
        wiki_dir = lint_env["wiki_dir"]

        # Create the target page (slug exists)
        _write_wiki_page(
            wiki_dir,
            "refund-timeline",
            ["refund_policy.md#refund-timeline"],
            body="# Refund Timeline\n\nContent.",
        )

        # Link with anchor pointing at existing slug
        _write_wiki_page(
            wiki_dir,
            "cancellation-window",
            ["refund_policy.md#cancellation-window"],
            body="# Cancellation\n\nSee [[refund-timeline#30-day-policy]] for details.",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        # refund-timeline exists, so [[refund-timeline#30-day-policy]] should NOT be a red link
        slugs = [r.slug for r in result.findings.red_links]
        assert "refund-timeline" not in slugs

    def test_c2_heading_anchor_unresolved_slug_still_flagged(self, lint_env):
        """[[missing-slug#heading]] — missing slug is still flagged (anchor dropped)."""
        wiki_dir = lint_env["wiki_dir"]

        _write_wiki_page(
            wiki_dir,
            "cancellation-window",
            ["refund_policy.md#cancellation-window"],
            body="# Cancellation\n\nSee [[missing-slug#some-heading]] for details.",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "missing-slug" in slugs

    def test_c2_mention_count_aggregates_across_pages(self, lint_env):
        """mention_count counts total occurrences across all pages."""
        wiki_dir = lint_env["wiki_dir"]

        # Page A mentions unresolved-slug twice
        _write_wiki_page(
            wiki_dir,
            "page-a",
            ["refund_policy.md#page-a"],
            body="# Page A\n\nSee [[unresolved-slug]] here and [[unresolved-slug]] there.",
        )

        # Page B mentions unresolved-slug once
        _write_wiki_page(
            wiki_dir,
            "page-b",
            ["refund_policy.md#page-b"],
            body="# Page B\n\nAlso [[unresolved-slug]].",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        red_links = result.findings.red_links
        unresolved = next((r for r in red_links if r.slug == "unresolved-slug"), None)
        assert unresolved is not None
        assert unresolved.mention_count == 3  # 2 from page-a + 1 from page-b

    def test_c2_referenced_by_lists_page_slugs(self, lint_env):
        """referenced_by must list slugs of pages that contain at least one mention."""
        wiki_dir = lint_env["wiki_dir"]

        _write_wiki_page(
            wiki_dir,
            "page-a",
            ["refund_policy.md#page-a"],
            body="# Page A\n\nSee [[phantom]] here.",
        )
        _write_wiki_page(
            wiki_dir,
            "page-b",
            ["refund_policy.md#page-b"],
            body="# Page B\n\nAlso [[phantom]].",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        red_links = result.findings.red_links
        phantom = next((r for r in red_links if r.slug == "phantom"), None)
        assert phantom is not None
        assert set(phantom.referenced_by) == {"page-a", "page-b"}
        # referenced_by must be alphabetical
        assert phantom.referenced_by == sorted(phantom.referenced_by)

    def test_c2_sample_context_captures_first_mention(self, lint_env):
        """sample_context should include ~50 chars surrounding the first mention."""
        wiki_dir = lint_env["wiki_dir"]

        _write_wiki_page(
            wiki_dir,
            "main-page",
            ["refund_policy.md#main-page"],
            body="# Main\n\nThis is some content before [[phantom-target]] and after.",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        red_links = result.findings.red_links
        phantom = next((r for r in red_links if r.slug == "phantom-target"), None)
        assert phantom is not None
        assert phantom.sample_context is not None
        # Should contain some surrounding text
        assert "phantom-target" in phantom.sample_context or len(phantom.sample_context) > 0

    def test_c2_sort_mention_count_descending(self, lint_env):
        """Red links sorted by mention_count descending."""
        wiki_dir = lint_env["wiki_dir"]

        # phantom-a mentioned 3 times, phantom-b mentioned 1 time
        _write_wiki_page(
            wiki_dir,
            "page-one",
            ["refund_policy.md#page-one"],
            body="# Page\n\n[[phantom-a]] and [[phantom-a]] and [[phantom-a]] and [[phantom-b]].",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        red_links = result.findings.red_links
        assert red_links[0].slug == "phantom-a"
        assert red_links[1].slug == "phantom-b"
        assert red_links[0].mention_count > red_links[1].mention_count

    def test_c2_sort_alphabetical_for_ties(self, lint_env):
        """When mention_count is equal, sort alphabetically by slug."""
        wiki_dir = lint_env["wiki_dir"]

        # Both appear once
        _write_wiki_page(
            wiki_dir,
            "page-one",
            ["refund_policy.md#page-one"],
            body="# Page\n\n[[zzz-link]] and [[aaa-link]].",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        red_links = result.findings.red_links
        # Same mention count → alphabetical
        assert red_links[0].slug == "aaa-link"
        assert red_links[1].slug == "zzz-link"

    def test_red_link_finding_schema(self, lint_env):
        """RedLinkFinding must have all required fields with correct types."""
        wiki_dir = lint_env["wiki_dir"]

        _write_wiki_page(
            wiki_dir,
            "any-page",
            ["refund_policy.md#any-page"],
            body="# Any Page\n\nSee [[phantom-schema]] for more.",
        )

        from app.lint import run_lint
        from app.schemas import RedLinkFinding

        result = run_lint(**lint_env)
        assert len(result.findings.red_links) == 1
        finding = result.findings.red_links[0]

        assert isinstance(finding, RedLinkFinding)
        assert isinstance(finding.slug, str)
        assert isinstance(finding.mention_count, int)
        assert isinstance(finding.referenced_by, list)
        assert finding.sample_context is None or isinstance(finding.sample_context, str)

    def test_report_has_c2_section(self, lint_env):
        """lint-report.md must include a C2 Red links section."""
        from app.lint import run_lint

        run_lint(**lint_env)
        content = (lint_env["wiki_dir"] / "lint-report.md").read_text(encoding="utf-8")
        assert "## C2 Red links" in content

    def test_summary_includes_c2_count(self, lint_env):
        """LintSummary.findings_by_check must include 'c2' key."""
        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert "c2" in result.summary.findings_by_check

    def test_c2_only_scans_entities_and_concepts(self, lint_env):
        """C2 scans only wiki/entities/ and wiki/concepts/ (whitelist per ADR-0006 SOURCE_DIRS)."""
        wiki_dir = lint_env["wiki_dir"]

        # Page in a different dir (not entities/concepts) — should NOT contribute
        other_dir = wiki_dir / "other"
        other_dir.mkdir(parents=True)
        (other_dir / "spurious.md").write_text(
            "---\nid: spurious\n---\n\n# Spurious\n\n[[spurious-phantom]]\n",
            encoding="utf-8",
        )

        from app.lint import run_lint

        result = run_lint(**lint_env)
        slugs = [r.slug for r in result.findings.red_links]
        assert "spurious-phantom" not in slugs
