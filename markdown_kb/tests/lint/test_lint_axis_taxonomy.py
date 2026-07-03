"""Unit tests for the Lint Axis taxonomy + group_findings_by_axis (issue #361).

Slice S1 scope
--------------
Introduces the single-source-of-truth taxonomy mapping each of the ten wired
checks (C1, C2, C3, C4, C5, C6, C8, C9, C10, C11) to ``{code, label, axis}``,
plus the stable axis order Freshness -> Coherence -> Coverage -> Lifecycle
(CONTEXT.md "Lint Axis" / ADR-0023). ``group_findings_by_axis`` is a pure data
transform — it does not run any check, so these tests build ``LintFindings``
directly rather than exercising ``run_lint``.

Issue #406 (ADR-0030) amendment
--------------------------------
Adds the eleventh wired check, C12 alias-collision (Coherence axis; C7 is
skipped and stays unassigned) — CONTEXT.md's "Lint Axis" entry already
documents Coherence as "C5 contradiction, C4 collision, C12 alias-collision"
(2026-07-03 grill), so the fixtures below are updated to match the taxonomy's
now-permanent eleven-entry shape rather than the ten it had at S1.
"""

from __future__ import annotations

import datetime

from app.lint import (
    LINT_AXIS_ORDER,
    LINT_CHECK_TAXONOMY,
    _render_report_markdown,
    group_findings_by_axis,
)
from app.schemas import (
    FailedGroundingFinding,
    LintFindings,
    LintSummary,
    OrphanPageFinding,
    StalePageFinding,
)

# CONTEXT.md "Lint Axis" enumeration — the expected code -> axis mapping.
_EXPECTED_AXIS_BY_CODE = {
    "C6": "Freshness",
    "C3": "Freshness",
    "C11": "Freshness",
    "C5": "Coherence",
    "C4": "Coherence",
    "C12": "Coherence",
    "C1": "Coverage",
    "C2": "Coverage",
    "C8": "Lifecycle",
    "C10": "Lifecycle",
    "C9": "Lifecycle",
}

# CONTEXT.md "Lint Axis" short labels (English; zh strings are a later slice).
_EXPECTED_LABEL_BY_CODE = {
    "C6": "stale",
    "C3": "failed-grounding",
    "C11": "orphan",
    "C5": "contradiction",
    "C4": "collision",
    "C12": "alias-collision",
    "C1": "coverage-gap",
    "C2": "red-link",
    "C8": "promotion",
    "C10": "invalid-schema",
    "C9": "stale-qa",
}


class TestLintAxisTaxonomy:
    """Taxonomy resolution: check code -> {code, label, axis}."""

    def test_axis_order_is_stable_freshness_coherence_coverage_lifecycle(self):
        assert LINT_AXIS_ORDER == ("Freshness", "Coherence", "Coverage", "Lifecycle")

    def test_taxonomy_covers_all_ten_wired_checks(self):
        assert set(LINT_CHECK_TAXONOMY.keys()) == set(_EXPECTED_AXIS_BY_CODE.keys())

    def test_each_check_resolves_to_its_context_md_axis(self):
        for code, expected_axis in _EXPECTED_AXIS_BY_CODE.items():
            meta = LINT_CHECK_TAXONOMY[code]
            assert meta.axis == expected_axis, (
                f"{code}: expected axis {expected_axis}, got {meta.axis}"
            )
            assert meta.code == code

    def test_each_check_resolves_to_its_context_md_label(self):
        for code, expected_label in _EXPECTED_LABEL_BY_CODE.items():
            assert LINT_CHECK_TAXONOMY[code].label == expected_label

    def test_every_axis_value_is_a_member_of_the_stable_order(self):
        for meta in LINT_CHECK_TAXONOMY.values():
            assert meta.axis in LINT_AXIS_ORDER


class TestGroupFindingsByAxis:
    """group_findings_by_axis: LintFindings -> ordered axis -> check -> findings."""

    def test_returns_one_group_per_axis_in_stable_order(self):
        groups = group_findings_by_axis(LintFindings())
        assert [g.axis for g in groups] == list(LINT_AXIS_ORDER)

    def test_empty_findings_still_enumerate_every_check_per_axis(self):
        """A check with zero findings is still present (empty list), never dropped —
        the empty-section convention is the renderer's job, not this helper's."""
        groups = group_findings_by_axis(LintFindings())
        codes_by_axis = {g.axis: [meta.code for meta, _findings in g.checks] for g in groups}
        assert codes_by_axis["Freshness"] == ["C6", "C3", "C11"]
        assert codes_by_axis["Coherence"] == ["C5", "C4", "C12"]
        assert codes_by_axis["Coverage"] == ["C1", "C2"]
        assert codes_by_axis["Lifecycle"] == ["C8", "C10", "C9"]
        for group in groups:
            for _meta, findings_list in group.checks:
                assert findings_list == []

    def test_findings_land_under_the_correct_check_and_axis(self):
        findings = LintFindings(
            orphans=[
                OrphanPageFinding(
                    page_slug="legacy-faq",
                    missing_sources=["deleted_source.md"],
                    suggested_action="delete or re-ingest",
                )
            ],
            failed_grounding=[
                FailedGroundingFinding(
                    page_slug="broken-page",
                    source="policy.md",
                    reason="claim_unsupported",
                    unsupported_claims=["claim x"],
                    suggested_action="review",
                )
            ],
            stale_pages=[
                StalePageFinding(
                    page_slug="aged",
                    source="aged_policy.md",
                    source_mtime=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
                    page_updated=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
                    drift_days=365.0,
                    suggested_action="re-ingest",
                )
            ],
        )
        groups = group_findings_by_axis(findings)
        freshness = next(g for g in groups if g.axis == "Freshness")
        by_code = dict(freshness.checks)
        c6_findings = by_code[LINT_CHECK_TAXONOMY["C6"]]
        c3_findings = by_code[LINT_CHECK_TAXONOMY["C3"]]
        c11_findings = by_code[LINT_CHECK_TAXONOMY["C11"]]
        assert [f.page_slug for f in c6_findings] == ["aged"]
        assert [f.page_slug for f in c3_findings] == ["broken-page"]
        assert [f.page_slug for f in c11_findings] == ["legacy-faq"]

        # Untouched axes stay all-empty.
        coherence = next(g for g in groups if g.axis == "Coherence")
        assert all(findings_list == [] for _meta, findings_list in coherence.checks)


class TestRenderReportAxisElision:
    """The report renderer elides an axis header when all its checks are empty."""

    def test_empty_lifecycle_axis_header_is_omitted(self):
        """A dormant qa lifecycle (C8/C9/C10 all empty) must NOT leave a dangling
        ``## Lifecycle`` header with nothing beneath it — while the always-rendered
        axes still appear with their "_No … found._" placeholders (issue #361).

        This is the common-case gap the e2e golden test misses: its fixture has
        Lifecycle findings, so it only ever exercises the header-present branch.
        """
        summary = LintSummary(
            total_findings=0,
            findings_by_check={},
            generated_at="2026-07-01T00:00:00Z",
        )
        content = _render_report_markdown(LintFindings(), summary, {})

        # Freshness/Coherence/Coverage always have content (their checks render a
        # "_No … found._" placeholder), so their axis headers are present.
        assert "## Freshness" in content
        assert "## Coherence" in content
        assert "## Coverage" in content
        # Lifecycle's three checks all self-omit when empty, so with a dormant qa
        # lifecycle the axis collapses to nothing — no bare header, no C8/C9/C10.
        assert "## Lifecycle" not in content
        assert "### C8" not in content
        assert "### C9" not in content
        assert "### C10" not in content
