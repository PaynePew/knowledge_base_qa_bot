"""Parametrized verifier judgment tests — RED in Slice #2, GREEN in Slice #3.

Each test calls grounding.verify() with one of the 7 anchor fixtures and
asserts the expected GroundingOutcome shape.  In this slice verify() raises
NotImplementedError("Slice #3"), so every case is marked xfail(strict=True):
the suite exits 0 on default invocation, but the failures remain visible and
will become errors if verify() accidentally starts returning something wrong.

When Slice #3 fills in verify(), remove the xfail markers and the tests
should pass straight to GREEN.
"""

from __future__ import annotations

import pytest

import app.grounding as grounding

from .fixtures import ALL_CASES, VerifierCase


@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=[c.name for c in ALL_CASES],
)
@pytest.mark.xfail(strict=True, reason="Slice #3 implements verify()")
def test_verifier_judgment(case: VerifierCase) -> None:
    """Verify that grounding.verify() produces the expected GroundingOutcome."""
    outcome = grounding.verify(draft=case.draft, sections=case.sections)

    assert outcome.passed == case.expected_passed, (
        f"[{case.name}] expected passed={case.expected_passed}, got {outcome.passed}"
    )

    if case.expected_unsupported_claims is not None:
        # Assert that all expected unsupported claims appear somewhere in the
        # outcome's unsupported_claims list (substring match tolerated because
        # the verifier decides exact claim boundaries).
        for expected_claim in case.expected_unsupported_claims:
            found = any(
                expected_claim.lower() in actual.lower()
                for actual in (outcome.result.unsupported_claims if outcome.result else [])
            )
            assert found, (
                f"[{case.name}] expected unsupported claim {expected_claim!r} "
                f"not found in {outcome.result}"
            )
