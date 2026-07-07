"""Unit tests for C5's added Routed remediation class (issue #534, ADR-0036
decisions 1-2).

C5 is the SECOND check (after C3, issue #408/ADR-0029) carrying TWO
remediation classes at once: its existing Authored Reconcile tier stays
wired unchanged (a wiki-rooted contradiction still converges there), and it
ALSO gains a Routed navigation hint (``secondary_route="fix-source"``) for
the source-rooted case — both pages are faithfully grounded in their own
Sources, but the Sources themselves disagree, so Reconcile is structurally
the wrong tool.

Unlike C3, no per-finding field on ``PagePairFinding`` distinguishes
source-rooted from wiki-rooted — that signal only exists at
``reconcile.generate_reconcile`` time (see ``test_reconcile.py``). This file
tests only the static taxonomy value, mirroring
``test_c3_routed_fix_source.py``'s style: a pure lookup over
``_REMEDIATION_TAXONOMY``, no filesystem, no LLM.
"""

from __future__ import annotations

from app.lint import remediation_for


class TestC5SecondaryRoutedRemediation:
    def test_c5_stays_authored_tier(self):
        """C5's PRIMARY tier is unaffected by this change — still Authored."""
        assert remediation_for("C5").tier == "authored"

    def test_c5_has_no_executable_actions(self):
        """Unlike C3 (which keeps a real Direct action alongside its
        secondary_route), C5 carries no executable action at all — Reconcile
        stays a preview/approve flow, not a one-click action (ADR-0023)."""
        assert remediation_for("C5").actions == ()

    def test_c5_gains_a_secondary_route(self):
        """The same field C3 set first (ADR-0029 decision 2), not a reuse of
        ``route`` (ADR-0036 decision 1)."""
        assert remediation_for("C5").secondary_route == "fix-source"

    def test_c5_route_field_stays_none(self):
        """``route`` is reserved for a check whose PRIMARY tier is Routed
        (C1/C2) — C5 is Authored-primary, so ``route`` itself stays ``None``
        even though ``secondary_route`` is set (mirrors C3)."""
        assert remediation_for("C5").route is None
