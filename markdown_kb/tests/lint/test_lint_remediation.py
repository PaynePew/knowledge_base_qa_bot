"""Unit tests for the Remediation descriptor (issue #363, ADR-0023 tier-A S3).

``remediation_for`` maps a wired check code to its Remediation tier
(Direct / Authored / deferred) plus its executable actions — the shared,
unit-testable source of truth the Operator Console renders per-row buttons
from (issue #363). This module does not run any check or touch the
filesystem: it is a pure lookup over ``_REMEDIATION_TAXONOMY``, mirroring
S1's ``test_lint_axis_taxonomy.py`` style.
"""

from __future__ import annotations

import pytest

from app.lint import RemediationAction, RemediationDescriptor, remediation_for

# The ten wired check codes (issue #361 LINT_CHECK_TAXONOMY keys).
_ALL_CODES = {"C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10", "C11"}


class TestRemediationTierClassification:
    """Every wired check resolves to exactly one of the three governance tiers."""

    def test_covers_all_ten_wired_checks(self):
        for code in _ALL_CODES:
            assert remediation_for(code) is not None

    def test_unknown_code_raises_keyerror(self):
        with pytest.raises(KeyError):
            remediation_for("C99")

    @pytest.mark.parametrize("code", ["C6", "C3", "C8", "C10"])
    def test_direct_tier_checks(self, code):
        assert remediation_for(code).tier == "direct"

    @pytest.mark.parametrize("code", ["C5", "C4", "C1", "C2"])
    def test_authored_tier_checks_have_no_actions(self, code):
        """Authored-tier findings (Coherence/Coverage) render disabled/tier-B —
        no executable action, preserving the per-item human-approval gate."""
        descriptor = remediation_for(code)
        assert descriptor.tier == "authored"
        assert descriptor.actions == ()

    @pytest.mark.parametrize("code", ["C9", "C11"])
    def test_deferred_checks_have_no_actions(self, code):
        """C9 stale-qa and C11 orphan need a lifecycle endpoint that does not
        exist yet (ADR-0023 Consequences) — deferred, not Direct or Authored."""
        descriptor = remediation_for(code)
        assert descriptor.tier == "deferred"
        assert descriptor.actions == ()


class TestRemediationActions:
    """Direct-tier actions: verb, target field, and the C3 force invariant."""

    def test_c6_reingest_no_force(self):
        actions = remediation_for("C6").actions
        assert actions == (RemediationAction("reingest", "source"),)
        assert actions[0].force is False

    def test_c3_reingest_retry_requires_force(self):
        """ADR-0023 Invariant: a C3 re-ingest (single or batch) MUST send
        force:true, or hash-skip (#93) no-ops the retry into a false fix."""
        actions = remediation_for("C3").actions
        assert actions == (RemediationAction("reingest_retry", "source", force=True),)
        assert actions[0].force is True

    def test_c10_discard_targets_page_slug(self):
        actions = remediation_for("C10").actions
        assert actions == (RemediationAction("discard", "page_slug"),)

    def test_c8_offers_promote_and_discard(self):
        """C8's own dedicated Curation Queue block renders both controls
        (issue #363 AC) — the descriptor names both actions."""
        actions = remediation_for("C8").actions
        verbs = {a.verb for a in actions}
        assert verbs == {"promote", "discard"}
        for action in actions:
            assert action.target_field == "slug"
            assert action.force is False

    def test_only_c3_requires_force_across_all_direct_actions(self):
        """Reviewer guard (CODING_STANDARD §11 / §12.8): force:true is a C3
        invariant, not a general Direct-tier default."""
        forced_verbs = {
            action.verb
            for code in _ALL_CODES
            for action in remediation_for(code).actions
            if action.force
        }
        assert forced_verbs == {"reingest_retry"}
