"""Hermetic tests for Slice 5-2: C3 (failed-grounding sweep) + C4-a (slug collision groups).

AC coverage (issue #67):
  - _check_c3_failed_grounding yields FailedGroundingFinding for each page with
    status == "failed_grounding" in frontmatter
  - FailedGroundingFinding fields: page_slug, source, reason, unsupported_claims,
    suggested_action
  - C3 sort: alphabetical by page_slug
  - _check_c4a_slug_collision collects slugs, groups by base slug (strip -N suffix
    where N >= 2), emits SlugCollisionFinding per group with len >= 2
  - SlugCollisionFinding fields: base_slug, pages_in_group, suggested_action
  - C4-a sort: collision group size descending, alphabetical by base_slug for ties
  - Report renderer: ## C3 Failed grounding + ## C4 Slug collision groups sections
  - Combined run with C3 + C4-a + C11 all firing verifies cross-check independence
  - Continue-on-error: if C3 raises on one page, C4-a still runs; error in
    LintResponse.check_errors["c3"]
  - No changes to env vars or log kinds

All tests are hermetic (no OPENAI_API_KEY required).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_live_page(
    wiki_dir: Path, slug: str, sources: list[str], *, subdir: str = "concepts"
) -> Path:
    """Write a minimal wiki page with status=live."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-05-26T00:00:00Z",
        "updated": "2026-05-26T00:00:00Z",
        "sources": sources,
        "status": "live",
        "open_questions": [],
    }
    content = (
        f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n# {slug}\n\nContent.\n"
    )
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _write_failed_grounding_page(
    wiki_dir: Path,
    slug: str,
    source: str,
    *,
    reason: str = "claim_unsupported",
    unsupported_claims: list[str] | None = None,
    subdir: str = "concepts",
) -> Path:
    """Write a wiki page with status=failed_grounding and grounding_failure block."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    grounding_failure: dict = {"reason": reason, "unsupported_claims": unsupported_claims or []}

    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-05-26T00:00:00Z",
        "updated": "2026-05-26T00:00:00Z",
        "sources": [source],
        "status": "failed_grounding",
        "open_questions": [],
        "grounding_failure": grounding_failure,
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n# {slug}\n\nFailed content.\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ---------------------------------------------------------------------------
# C3 — Failed-grounding sweep
# ---------------------------------------------------------------------------


class TestC3FailedGrounding:
    """Tests for _check_c3_failed_grounding."""

    def test_clean_wiki_no_c3_findings(self, lint_env):
        """A wiki with only live pages produces no C3 findings."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "refund-policy", ["refund_policy.md#section"])

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        assert findings == []

    def test_failed_grounding_page_produces_finding(self, lint_env):
        """A page with status=failed_grounding produces one FailedGroundingFinding."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "broken-page",
            "refund_policy.md#cancellation-window",
            reason="claim_unsupported",
            unsupported_claims=["fake claim"],
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        assert len(findings) == 1
        f = findings[0]
        assert f.page_slug == "broken-page"
        assert f.source == "refund_policy.md#cancellation-window"
        assert f.reason == "claim_unsupported"
        assert f.unsupported_claims == ["fake claim"]
        assert f.suggested_action != ""

    def test_verifier_unavailable_reason(self, lint_env):
        """A page with reason=verifier_unavailable is captured with empty unsupported_claims."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "unavailable-page",
            "policy.md#section",
            reason="verifier_unavailable",
            unsupported_claims=[],
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        assert len(findings) == 1
        assert findings[0].reason == "verifier_unavailable"
        assert findings[0].unsupported_claims == []

    def test_c3_sorted_alphabetically(self, lint_env):
        """C3 findings must be sorted alphabetically by page_slug."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(wiki_dir, "zebra-failed", "s.md#z")
        _write_failed_grounding_page(wiki_dir, "alpha-failed", "s.md#a")
        _write_failed_grounding_page(wiki_dir, "mango-failed", "s.md#m")

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        slugs = [f.page_slug for f in findings]
        assert slugs == sorted(slugs)

    def test_c3_source_from_first_sources_entry(self, lint_env):
        """source field comes from frontmatter.sources[0]."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "multi-source-failed",
            "first_source.md#section",
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        assert findings[0].source == "first_source.md#section"

    def test_live_pages_not_included_in_c3(self, lint_env):
        """live pages are ignored; only failed_grounding pages produce findings."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "live-page", ["existing.md#sec"])
        _write_failed_grounding_page(wiki_dir, "failed-page", "missing.md#sec")

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        assert len(findings) == 1
        assert findings[0].page_slug == "failed-page"

    def test_c3_scans_both_entities_and_concepts(self, lint_env):
        """C3 must scan wiki/entities/ as well as wiki/concepts/."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(wiki_dir, "entity-failed", "s.md#e", subdir="entities")
        _write_failed_grounding_page(wiki_dir, "concept-failed", "s.md#c", subdir="concepts")

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        slugs = [f.page_slug for f in findings]
        assert "entity-failed" in slugs
        assert "concept-failed" in slugs

    def test_c3_suggested_action_mentions_review_or_delete(self, lint_env):
        """suggested_action must reference Source review + re-ingest or page deletion."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir, "broken", "source.md#sec", reason="claim_unsupported"
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        action_lower = findings[0].suggested_action.lower()
        has_review_or_reingest = any(w in action_lower for w in ("review", "re-ingest", "reingest"))
        has_delete_or_delete = any(w in action_lower for w in ("delete", "remove"))
        assert has_review_or_reingest, (
            f"suggested_action must mention review/re-ingest: {findings[0].suggested_action!r}"
        )
        assert has_delete_or_delete, (
            f"suggested_action must mention deletion: {findings[0].suggested_action!r}"
        )


# ---------------------------------------------------------------------------
# C4-a — Slug collision groups
# ---------------------------------------------------------------------------


class TestC4aSlugCollision:
    """Tests for _check_c4a_slug_collision."""

    def test_clean_wiki_no_c4a_findings(self, lint_env):
        """A wiki with unique slugs produces no C4-a findings."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "refund", ["refund.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert findings == []

    def test_pricing_and_pricing_2_produce_one_group(self, lint_env):
        """pricing.md + pricing-2.md → one SlugCollisionFinding with both pages."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["pricing2.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert len(findings) == 1
        f = findings[0]
        assert f.base_slug == "pricing"
        assert "pricing" in f.pages_in_group
        assert "pricing-2" in f.pages_in_group
        assert len(f.pages_in_group) == 2

    def test_three_collisions_in_one_group(self, lint_env):
        """pricing + pricing-2 + pricing-3 → one group with 3 members."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-3", ["pricing.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert len(findings) == 1
        assert len(findings[0].pages_in_group) == 3

    def test_c4a_only_groups_n_gte_2_suffix(self, lint_env):
        """Suffix -1 should NOT trigger grouping; only -2, -3, ... trigger."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-1", ["pricing.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        # pricing-1 does not trigger grouping (N must be >= 2)
        assert findings == []

    def test_c4a_groups_multiple_independent_collisions(self, lint_env):
        """Two independent collision groups produce two separate findings."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["pricing.md#section"])
        _write_live_page(wiki_dir, "shipping", ["shipping.md#section"])
        _write_live_page(wiki_dir, "shipping-2", ["shipping.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert len(findings) == 2
        base_slugs = {f.base_slug for f in findings}
        assert base_slugs == {"pricing", "shipping"}

    def test_c4a_sort_size_desc_then_alpha(self, lint_env):
        """C4-a sort: collision group size descending, alphabetical by base_slug for ties."""
        wiki_dir = lint_env["wiki_dir"]
        # "aardvark" group: 2 members
        _write_live_page(wiki_dir, "aardvark", ["a.md#section"])
        _write_live_page(wiki_dir, "aardvark-2", ["a.md#section"])
        # "zebra" group: 3 members
        _write_live_page(wiki_dir, "zebra", ["z.md#section"])
        _write_live_page(wiki_dir, "zebra-2", ["z.md#section"])
        _write_live_page(wiki_dir, "zebra-3", ["z.md#section"])
        # "mango" group: 2 members
        _write_live_page(wiki_dir, "mango", ["m.md#section"])
        _write_live_page(wiki_dir, "mango-2", ["m.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert len(findings) == 3
        # zebra (3 members) comes first
        assert findings[0].base_slug == "zebra"
        assert len(findings[0].pages_in_group) == 3
        # aardvark and mango (2 members each) sorted alpha
        assert findings[1].base_slug == "aardvark"
        assert findings[2].base_slug == "mango"

    def test_c4a_scans_entities_and_concepts(self, lint_env):
        """C4-a must scan across both wiki/entities/ and wiki/concepts/."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "refund", ["r.md#section"], subdir="concepts")
        _write_live_page(wiki_dir, "refund-2", ["r.md#section"], subdir="entities")

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert len(findings) == 1
        assert findings[0].base_slug == "refund"
        assert len(findings[0].pages_in_group) == 2

    def test_c4a_suggested_action_mentions_merge_or_rename(self, lint_env):
        """suggested_action must mention review + merge or heading rename."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["pricing.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        action_lower = findings[0].suggested_action.lower()
        has_merge_or_rename = any(w in action_lower for w in ("merge", "rename", "review"))
        assert has_merge_or_rename, (
            f"suggested_action must mention merge/rename/review: {findings[0].suggested_action!r}"
        )

    def test_c4a_non_numeric_suffix_not_grouped(self, lint_env):
        """Slugs like 'pricing-special' must NOT be grouped with 'pricing'."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-special", ["pricing.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert findings == [], f"Non-numeric suffix should not trigger collision: {findings}"

    def test_c4a_large_numeric_suffix(self, lint_env):
        """Suffix like -10 or -100 should be captured."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-10", ["pricing.md#section"])

        from app.lint import _check_c4a_slug_collision

        findings = _check_c4a_slug_collision(wiki_dir)
        assert len(findings) == 1
        assert "pricing-10" in findings[0].pages_in_group


# ---------------------------------------------------------------------------
# C3 + C4-a + C11 combined run via run_lint()
# ---------------------------------------------------------------------------


class TestCombinedC3C4aC11:
    """Tests for cross-check independence when C3 + C4-a + C11 all fire."""

    def test_combined_run_all_checks_fire(self, lint_env):
        """C3, C4-a, and C11 all fire independently in a single run_lint() call."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        # C11: orphan page (source missing from docs/)
        from tests.lint.test_lint_scaffold import _write_wiki_page as _write_orphan_page

        _write_orphan_page(wiki_dir, "orphan-page", ["deleted_source.md#section"])

        # C3: failed grounding page
        _write_failed_grounding_page(
            wiki_dir,
            "broken-page",
            "refund_policy.md#cancellation-window",
            reason="claim_unsupported",
            unsupported_claims=["fake claim"],
        )

        # C4-a: slug collision
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["pricing.md#section"])

        # docs_dir has refund_policy.md but not deleted_source.md
        (docs_dir / "refund_policy.md").write_text(
            "# Refund Policy\n\nContent.\n", encoding="utf-8"
        )
        (docs_dir / "pricing.md").write_text("# Pricing\n\nContent.\n", encoding="utf-8")

        from app.lint import run_lint

        result = run_lint(**lint_env)

        # All three checks must fire
        assert len(result.findings.orphans) >= 1, "C11 must fire"
        assert len(result.findings.failed_grounding) >= 1, "C3 must fire"
        assert len(result.findings.slug_collisions) >= 1, "C4-a must fire"

        # No errors — all checks ran cleanly
        assert result.check_errors == {}

    def test_combined_run_check_independence(self, lint_env):
        """Findings from C3 + C4-a + C11 are independent; counts are consistent."""
        wiki_dir = lint_env["wiki_dir"]
        docs_dir = lint_env["docs_dir"]

        # Plant exactly 1 of each type
        from tests.lint.test_lint_scaffold import _write_wiki_page as _write_orphan_page

        _write_orphan_page(wiki_dir, "orphan-a", ["missing.md#section"])
        _write_failed_grounding_page(wiki_dir, "broken-b", "policy.md#section")
        # slug-c and slug-c-2 both reference s.md — create s.md so C11 does not fire for them
        _write_live_page(wiki_dir, "slug-c", ["s.md#section"])
        _write_live_page(wiki_dir, "slug-c-2", ["s.md#section"])

        # Make policy.md and s.md exist (orphan-a's source "missing.md" still missing)
        (docs_dir / "policy.md").write_text("# Policy\n\nContent.\n", encoding="utf-8")
        (docs_dir / "s.md").write_text("# S\n\nContent.\n", encoding="utf-8")

        from app.lint import run_lint

        result = run_lint(**lint_env)

        assert len(result.findings.orphans) == 1
        assert len(result.findings.failed_grounding) == 1
        assert len(result.findings.slug_collisions) == 1

        total = result.summary.total_findings
        assert total == 3

        # by_check includes all three
        assert result.summary.findings_by_check.get("c11") == 1
        assert result.summary.findings_by_check.get("c3") == 1
        assert result.summary.findings_by_check.get("c4a") == 1

    def test_c3_continue_on_error_c4a_still_runs(self, lint_env, monkeypatch):
        """If C3 raises, C4-a still runs and its findings appear in the response."""
        wiki_dir = lint_env["wiki_dir"]

        # C4-a: plant a collision
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["pricing.md#section"])

        # Patch C3 to raise
        import app.lint as lint_module

        def _bad_c3(*_args, **_kwargs):
            raise RuntimeError("simulated C3 failure")

        monkeypatch.setattr(lint_module, "_check_c3_failed_grounding", _bad_c3)

        from app.lint import run_lint

        result = run_lint(**lint_env)

        # C3 error recorded
        assert "c3" in result.check_errors

        # C4-a still ran
        assert len(result.findings.slug_collisions) == 1

    def test_report_contains_c3_section(self, lint_env):
        """The rendered report must contain ## C3 Failed grounding section."""
        from app.lint import run_lint

        run_lint(**lint_env)
        content = (lint_env["wiki_dir"] / "lint-report.md").read_text(encoding="utf-8")
        assert "## C3 Failed grounding" in content

    def test_report_contains_c4_section(self, lint_env):
        """The rendered report must contain ## C4 Slug collision groups section."""
        from app.lint import run_lint

        run_lint(**lint_env)
        content = (lint_env["wiki_dir"] / "lint-report.md").read_text(encoding="utf-8")
        assert "## C4 Slug collision groups" in content

    def test_report_c3_section_shows_finding(self, lint_env):
        """When a C3 finding exists, its page_slug appears in the C3 report section."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "broken-report-page",
            "policy.md#section",
            reason="claim_unsupported",
            unsupported_claims=["claim x"],
        )

        from app.lint import run_lint

        run_lint(**lint_env)
        content = (wiki_dir / "lint-report.md").read_text(encoding="utf-8")
        assert "broken-report-page" in content

    def test_report_c4_section_shows_base_slug(self, lint_env):
        """When a C4-a finding exists, its base_slug appears in the C4 report section."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["pricing.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["pricing.md#section"])

        from app.lint import run_lint

        run_lint(**lint_env)
        content = (wiki_dir / "lint-report.md").read_text(encoding="utf-8")
        assert "pricing" in content

    def test_report_c3_count_in_heading(self, lint_env):
        """## C3 Failed grounding heading must contain the finding count."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(wiki_dir, "b1", "s.md#sec")
        _write_failed_grounding_page(wiki_dir, "b2", "s.md#sec")

        from app.lint import run_lint

        run_lint(**lint_env)
        content = (wiki_dir / "lint-report.md").read_text(encoding="utf-8")
        assert "## C3 Failed grounding (2 pages)" in content

    def test_report_c4_count_in_heading(self, lint_env):
        """## C4 Slug collision groups heading must contain the group count."""
        wiki_dir = lint_env["wiki_dir"]
        _write_live_page(wiki_dir, "pricing", ["p.md#section"])
        _write_live_page(wiki_dir, "pricing-2", ["p.md#section"])

        from app.lint import run_lint

        run_lint(**lint_env)
        content = (wiki_dir / "lint-report.md").read_text(encoding="utf-8")
        assert "## C4 Slug collision groups (1 groups)" in content
