"""Unit tests for the Remediation descriptor (issue #363, ADR-0023 tier-A S3).

``remediation_for`` maps a wired check code to its Remediation tier
(Direct / Authored / Confirmed / Routed / deferred) plus its executable
actions — the shared, unit-testable source of truth the Operator Console
renders per-row buttons from (issue #363). This module does not run any
check or touch the filesystem: it is a pure lookup over
``_REMEDIATION_TAXONOMY``, mirroring S1's ``test_lint_axis_taxonomy.py``
style.
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

    @pytest.mark.parametrize("code", ["C5", "C4"])
    def test_authored_tier_checks_have_no_actions(self, code):
        """Authored-tier findings (Coherence) render disabled/tier-B — no
        executable action, preserving the per-item human-approval gate."""
        descriptor = remediation_for(code)
        assert descriptor.tier == "authored"
        assert descriptor.actions == ()

    @pytest.mark.parametrize("code", ["C1", "C2"])
    def test_routed_tier_checks_have_no_actions_but_carry_a_route(self, code):
        """tier-B S7 (issue #383, ADR-0027): C1/C2 flip authored -> routed.
        Fill routes through the existing Upload -> Import -> Ingest pipeline,
        so no draft ever exists for a curator to approve — Routed has no
        gate and no executable action, unlike Authored it carries a
        ``route`` instead (the ONE field only Routed descriptors set) so
        every surface can navigate off the same shared taxonomy value."""
        descriptor = remediation_for(code)
        assert descriptor.tier == "routed"
        assert descriptor.actions == ()
        assert descriptor.route == "import"

    def test_non_routed_checks_carry_no_route(self):
        """``route`` is the one field only Routed descriptors set (ADR-0027)."""
        for code in _ALL_CODES - {"C1", "C2"}:
            assert remediation_for(code).route is None

    def test_c9_is_authored_with_a_refile_action(self):
        """tier-B S4 (issue #380, ADR-0026 decision 1): C9 stale-qa flips
        deferred -> authored now that ``POST /qa/{slug}/refile`` exists.
        Unlike C5/C4/C1/C2, C9's own human gate is the *downstream* Promote
        on the resulting draft (not a preview step here), so it carries a
        real action even while staying Authored-classified — see the
        ``_REMEDIATION_TAXONOMY`` comment for why this is the one exception
        to "authored-tier checks have no actions" above."""
        descriptor = remediation_for("C9")
        assert descriptor.tier == "authored"
        assert descriptor.actions == (RemediationAction("refile", "page_slug"),)

    def test_c11_is_confirmed_with_a_delete_action(self):
        """tier-B S5 (issue #381, ADR-0024/0025): C11 orphan flips
        deferred -> confirmed now that ``DELETE /pages/{slug}`` exists.
        Confirmed is a fourth tier value alongside Direct/Authored/deferred
        (ADR-0024: a human confirms an irreversible operation, not
        curator-drafted content) — the action wires unconditionally here;
        per-finding full/partial eligibility (``OrphanPageFinding.full``)
        is what a consumer reads to decide whether to render the delete
        button or advisory text only."""
        descriptor = remediation_for("C11")
        assert descriptor.tier == "confirmed"
        assert descriptor.actions == (RemediationAction("delete", "page_slug"),)


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
