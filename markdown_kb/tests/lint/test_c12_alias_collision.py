"""Tests for C12 alias-collision (issue #406, ADR-0030, Coherence axis).

AC coverage (issue #406):
  - C12 fires on both collision shapes:
      (a) an alias colliding with an existing page slug
      (b) two pages claiming the same alias
  - Resolution stays deterministic while the finding stands (mirrors
    ``slugs.build_alias_resolution_map``'s tie-break rule).
  - ``LINT_CHECK_TAXONOMY["C12"]`` / ``_REMEDIATION_TAXONOMY["C12"]`` /
    ``group_findings_by_axis`` / the report renderer all recognise C12
    (ADR-0017 shared-taxonomy interface parity).
  - C7 stays unassigned (never in ``LINT_CHECK_TAXONOMY``).

All tests are hermetic (no OPENAI_API_KEY required).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.lint import (
    LINT_CHECK_TAXONOMY,
    RemediationDescriptor,
    _check_c12_alias_collision,
    _render_report_markdown,
    group_findings_by_axis,
    remediation_for,
    run_lint,
)
from app.schemas import LintFindings, LintSummary


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    *,
    subdir: str = "concepts",
    aliases: list[str] | None = None,
) -> Path:
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter: dict = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": [f"source.md#{slug}"],
        "status": "live",
        "open_questions": [],
    }
    if aliases is not None:
        frontmatter["aliases"] = aliases
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n# {slug}\n\nBody.\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ---------------------------------------------------------------------------
# Collision shape (a): alias vs. real slug
# ---------------------------------------------------------------------------


class TestAliasVsSlugCollision:
    def test_alias_colliding_with_a_real_slug_fires(self, tmp_wiki_dir):
        _write_wiki_page(tmp_wiki_dir, "pricing")
        _write_wiki_page(tmp_wiki_dir, "other-page", aliases=["pricing"])

        findings = _check_c12_alias_collision(tmp_wiki_dir)

        assert len(findings) == 1
        f = findings[0]
        assert f.kind == "alias_vs_slug"
        assert f.alias == "pricing"
        assert f.claimed_by == ["other-page"]
        assert f.slug_owner == "pricing"
        assert f.resolved_to == "pricing"

    def test_no_collision_when_alias_is_unique_and_no_real_slug_conflict(self, tmp_wiki_dir):
        _write_wiki_page(tmp_wiki_dir, "replacement-payment-methods", aliases=["paypal"])
        _write_wiki_page(tmp_wiki_dir, "other-page")

        findings = _check_c12_alias_collision(tmp_wiki_dir)

        assert findings == []


# ---------------------------------------------------------------------------
# Collision shape (b): alias vs. alias
# ---------------------------------------------------------------------------


class TestAliasVsAliasCollision:
    def test_two_pages_claiming_the_same_alias_fires(self, tmp_wiki_dir):
        _write_wiki_page(tmp_wiki_dir, "zeta-page", aliases=["shared-alias"])
        _write_wiki_page(tmp_wiki_dir, "alpha-page", aliases=["shared-alias"])

        findings = _check_c12_alias_collision(tmp_wiki_dir)

        assert len(findings) == 1
        f = findings[0]
        assert f.kind == "alias_vs_alias"
        assert f.alias == "shared-alias"
        assert f.claimed_by == ["alpha-page", "zeta-page"]
        assert f.slug_owner is None
        # Deterministic tie-break: lexicographically-first canonical slug —
        # mirrors slugs.build_alias_resolution_map exactly.
        assert f.resolved_to == "alpha-page"

    def test_findings_sorted_alphabetically_by_alias(self, tmp_wiki_dir):
        _write_wiki_page(tmp_wiki_dir, "page-1", aliases=["zeta-alias"])
        _write_wiki_page(tmp_wiki_dir, "page-2", aliases=["zeta-alias"])
        _write_wiki_page(tmp_wiki_dir, "page-3", aliases=["alpha-alias"])
        _write_wiki_page(tmp_wiki_dir, "page-4", aliases=["alpha-alias"])

        findings = _check_c12_alias_collision(tmp_wiki_dir)

        assert [f.alias for f in findings] == ["alpha-alias", "zeta-alias"]


# ---------------------------------------------------------------------------
# Non-collision cases degrade cleanly
# ---------------------------------------------------------------------------


class TestNoCollision:
    def test_empty_wiki_produces_no_findings(self, tmp_wiki_dir):
        assert _check_c12_alias_collision(tmp_wiki_dir) == []

    def test_pages_with_no_aliases_field_produce_no_findings(self, tmp_wiki_dir):
        _write_wiki_page(tmp_wiki_dir, "page-a")
        _write_wiki_page(tmp_wiki_dir, "page-b")

        assert _check_c12_alias_collision(tmp_wiki_dir) == []


# ---------------------------------------------------------------------------
# resolved_to stays in lockstep with the shared resolver (ADR-0030 decision 4:
# "resolution stays deterministic while the finding stands" — C12 and
# slugs.build_alias_resolution_map must never disagree on the live outcome).
# ---------------------------------------------------------------------------


class TestResolvedToMatchesSharedResolver:
    def test_alias_vs_slug_resolved_to_matches_the_shared_resolver(self, tmp_wiki_dir):
        from app.slugs import build_alias_resolution_map

        _write_wiki_page(tmp_wiki_dir, "pricing")
        _write_wiki_page(tmp_wiki_dir, "other-page", aliases=["pricing"])

        findings = _check_c12_alias_collision(tmp_wiki_dir)
        resolution = build_alias_resolution_map(tmp_wiki_dir)

        assert findings[0].resolved_to == resolution["pricing"]

    def test_alias_vs_alias_resolved_to_matches_the_shared_resolver(self, tmp_wiki_dir):
        from app.slugs import build_alias_resolution_map

        _write_wiki_page(tmp_wiki_dir, "zeta-page", aliases=["shared-alias"])
        _write_wiki_page(tmp_wiki_dir, "alpha-page", aliases=["shared-alias"])

        findings = _check_c12_alias_collision(tmp_wiki_dir)
        resolution = build_alias_resolution_map(tmp_wiki_dir)

        assert findings[0].resolved_to == resolution["shared-alias"]


# ---------------------------------------------------------------------------
# Shared taxonomy wiring (ADR-0017 interface parity)
# ---------------------------------------------------------------------------


class TestC12SharedTaxonomyWiring:
    def test_c12_is_in_the_taxonomy_under_coherence(self):
        meta = LINT_CHECK_TAXONOMY["C12"]
        assert meta.axis == "Coherence"
        assert meta.code == "C12"
        assert meta.label == "alias-collision"
        assert meta.label_zh == "別名衝突"

    def test_c7_is_never_assigned(self):
        assert "C7" not in LINT_CHECK_TAXONOMY

    def test_c12_group_findings_by_axis(self):
        groups = group_findings_by_axis(LintFindings())
        coherence = next(g for g in groups if g.axis == "Coherence")
        codes = [meta.code for meta, _findings in coherence.checks]
        assert "C12" in codes

    def test_c12_remediation_is_direct_tier(self):
        descriptor = remediation_for("C12")
        assert isinstance(descriptor, RemediationDescriptor)
        assert descriptor.tier == "direct"
        # Foundation slice (issue #406): no assign-alias endpoint exists yet
        # to wire an executable action to.
        assert descriptor.actions == ()
        assert descriptor.route is None

    def test_c12_renders_in_the_report_when_findings_exist(self, tmp_wiki_dir):
        _write_wiki_page(tmp_wiki_dir, "pricing")
        _write_wiki_page(tmp_wiki_dir, "other-page", aliases=["pricing"])
        findings = LintFindings(alias_collisions=_check_c12_alias_collision(tmp_wiki_dir))
        summary = LintSummary(
            total_findings=1, findings_by_check={"c12": 1}, generated_at="2026-07-03T00:00:00Z"
        )

        content = _render_report_markdown(findings, summary, {})

        assert "### C12" in content
        assert "`pricing`" in content

    def test_c12_report_section_always_present_even_when_empty(self):
        """Coherence axis convention: C12 renders a placeholder, not silence."""
        summary = LintSummary(
            total_findings=0, findings_by_check={}, generated_at="2026-07-03T00:00:00Z"
        )
        content = _render_report_markdown(LintFindings(), summary, {})

        assert "### C12" in content
        assert "_No alias collisions found._" in content


# ---------------------------------------------------------------------------
# run_lint() end-to-end wiring
# ---------------------------------------------------------------------------


class TestC12WiredIntoRunLint:
    def test_run_lint_surfaces_c12_findings(self, lint_env):
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "pricing")
        _write_wiki_page(wiki_dir, "other-page", aliases=["pricing"])

        result = run_lint(**lint_env, include_c5=False)

        assert len(result.findings.alias_collisions) == 1
        assert result.summary.findings_by_check["c12"] == 1
        assert result.check_errors == {}
