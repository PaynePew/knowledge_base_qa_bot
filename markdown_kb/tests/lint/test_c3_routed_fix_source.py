"""Unit tests for C3's added Routed remediation class (issue #408, ADR-0029
decisions 2-4).

C3 is the first check carrying TWO remediation classes at once: its existing
Direct ``reingest_retry`` action stays wired (``verifier_unavailable`` is a
transient failure Re-ingest genuinely fixes), and it ALSO gains a Routed
navigation hint (``secondary_route="fix-source"``) for its dominant failure
mode, ``claim_unsupported`` — the real fix (amending what the Source says) is
knowledge only the human can supply.

This is a new, distinct field (``secondary_route``) rather than a reuse of
the existing ``route`` field, so the pre-existing
``test_lint_remediation.py::test_non_routed_checks_carry_no_route`` invariant
("route is the ONE field only C1/C2 set") stays true and unmodified — C3's
PRIMARY tier is still Direct, not Routed.

Mirrors ``test_lint_remediation.py``'s style: a pure lookup over
``_REMEDIATION_TAXONOMY``, no filesystem, no LLM.
"""

from __future__ import annotations

from app.lint import RemediationAction, remediation_for


class TestC3SecondaryRoutedRemediation:
    def test_c3_stays_direct_tier(self):
        """C3's PRIMARY tier is unaffected by this change — still Direct."""
        assert remediation_for("C3").tier == "direct"

    def test_c3_direct_action_is_unaffected(self):
        """The existing force:true reingest_retry action is unchanged (ADR-0023
        Invariant — must not regress while adding the Routed class)."""
        actions = remediation_for("C3").actions
        assert actions == (RemediationAction("reingest_retry", "source", force=True),)
        assert actions[0].force is True

    def test_c3_gains_a_secondary_route(self):
        """The new field, not a reuse of ``route`` (ADR-0029 decision 2)."""
        assert remediation_for("C3").secondary_route == "fix-source"

    def test_c3_route_field_stays_none(self):
        """``route`` is reserved for a check whose PRIMARY tier is Routed
        (C1/C2) — C3 is Direct-primary, so ``route`` itself stays ``None``
        even though ``secondary_route`` is set. This is what keeps the
        pre-existing ``test_non_routed_checks_carry_no_route`` assertion true
        without modification."""
        assert remediation_for("C3").route is None

    def test_only_c3_and_c5_carry_a_secondary_route(self):
        """C5 gains the SAME field (issue #534, ADR-0036) — see
        ``test_c5_routed_fix_source.py`` for C5's own dedicated coverage.
        No other wired check sets ``secondary_route``."""
        all_codes = {"C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10", "C11"}
        for code in all_codes - {"C3", "C5"}:
            assert remediation_for(code).secondary_route is None, (
                f"{code} unexpectedly carries a secondary_route"
            )
