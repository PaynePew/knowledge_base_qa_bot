"""Parametrized verifier judgment tests — mock-mode (fast CI) and live-mode.

Mock-mode tests (default, no OPENAI_API_KEY needed):
    Each test uses a fake structured-output stub injected via monkeypatch
    so the suite runs without any real API calls.  The stub returns a
    GroundingResult that matches the fixture's expected outcome, verifying
    that verify() maps GroundingResult → GroundingOutcome correctly for all
    7 anchor cases.

Live-mode tests (opt-in with pytest -m live, requires OPENAI_API_KEY):
    The same 7 fixtures are run against the real gpt-4o-mini verifier to
    detect verifier quality regressions.  These are @pytest.mark.live so
    they are skipped in fast CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app.grounding as grounding
from app.grounding import (
    GroundingClaim,
    GroundingOutcome,
    GroundingResult,
)

from .fixtures import ALL_CASES, EMPTY_SECTIONS, VerifierCase

# ---------------------------------------------------------------------------
# Mock-mode helpers
# ---------------------------------------------------------------------------


def _make_fake_chain(case: VerifierCase) -> MagicMock:
    """Return a fake structured-output chain that returns the expected outcome.

    The chain's invoke() returns a GroundingResult consistent with the
    fixture's expected_passed value, without touching OpenAI.
    """
    fake_chain = MagicMock()

    if case.expected_passed:
        # All claims supported
        claims = [
            GroundingClaim(
                text="The draft claim is supported.",
                supported=True,
                citing_section_ids=[s.id for s in case.sections],
            )
        ]
        result = GroundingResult(
            reasoning="The draft accurately reflects the cited sections.",
            claims=claims,
            unsupported_claims=[],
            passed=True,
        )
    else:
        # At least one claim unsupported
        unsupported_text = "An unsupported claim."
        if case.expected_unsupported_claims:
            unsupported_text = case.expected_unsupported_claims[0]

        claims = [
            GroundingClaim(
                text=unsupported_text,
                supported=False,
                citing_section_ids=[],
            )
        ]
        result = GroundingResult(
            reasoning="The draft contains claims not supported by the cited sections.",
            claims=claims,
            unsupported_claims=[unsupported_text],
            passed=False,
        )

    fake_chain.invoke.return_value = result
    return fake_chain


# ---------------------------------------------------------------------------
# Mock-mode parametrized tests (7 fixture cases, no API key needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=[c.name for c in ALL_CASES],
)
def test_verifier_judgment_mock(case: VerifierCase, monkeypatch, tmp_path) -> None:
    """Verify grounding.verify() maps GroundingResult → GroundingOutcome correctly.

    Uses a fake chain stub to avoid real API calls.
    """
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_chain = _make_fake_chain(case)

    def fake_with_structured_output(schema):  # noqa: ARG001
        return fake_chain

    fake_llm = MagicMock()
    fake_llm.with_structured_output = fake_with_structured_output

    with patch("app.grounding.ChatOpenAI", return_value=fake_llm):
        outcome = grounding.verify(draft=case.draft, sections=case.sections)

    assert outcome.passed == case.expected_passed, (
        f"[{case.name}] expected passed={case.expected_passed}, got {outcome.passed}"
    )

    if case.expected_passed:
        assert outcome.reason == "claim_supported", (
            f"[{case.name}] expected reason=claim_supported, got {outcome.reason}"
        )
    else:
        assert outcome.reason in ("claim_unsupported", "verifier_unavailable"), (
            f"[{case.name}] unexpected reason={outcome.reason}"
        )

    if case.expected_unsupported_claims is not None and not case.expected_passed:
        # Assert that all expected unsupported claims appear somewhere in the
        # outcome's unsupported_claims list (substring match tolerated because
        # the verifier decides exact claim boundaries).
        result_unsupported = outcome.result.unsupported_claims if outcome.result else []
        for expected_claim in case.expected_unsupported_claims:
            found = any(expected_claim.lower() in actual.lower() for actual in result_unsupported)
            assert found, (
                f"[{case.name}] expected unsupported claim {expected_claim!r} "
                f"not found in {result_unsupported}"
            )


# ---------------------------------------------------------------------------
# Live-mode parametrized tests (require OPENAI_API_KEY, skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=[c.name for c in ALL_CASES],
)
def test_verifier_judgment_live(case: VerifierCase) -> None:
    """Run real gpt-4o-mini verifier against the 7 anchor fixtures.

    Requires OPENAI_API_KEY in the environment.
    Run with: pytest -m live
    AC #5 — human verification step.

    empty_sections is now deterministic (short-circuit before LLM): no model
    dependence for that fixture.  All other 6 fixtures still exercise the real
    gpt-4o-mini path.
    """
    outcome = grounding.verify(draft=case.draft, sections=case.sections)

    assert outcome.passed == case.expected_passed, (
        f"[{case.name}] expected passed={case.expected_passed}, got {outcome.passed}\n"
        f"reason={outcome.reason}\n"
        f"result={outcome.result}"
    )

    if case.expected_unsupported_claims is not None and not case.expected_passed:
        result_unsupported = outcome.result.unsupported_claims if outcome.result else []
        for expected_claim in case.expected_unsupported_claims:
            found = any(expected_claim.lower() in actual.lower() for actual in result_unsupported)
            assert found, (
                f"[{case.name}] expected unsupported claim {expected_claim!r} "
                f"not found in {result_unsupported}"
            )


# ---------------------------------------------------------------------------
# Empty-sections short-circuit: no LLM call (issue #191)
# ---------------------------------------------------------------------------


def test_verify_empty_sections_no_llm_call(tmp_path, monkeypatch) -> None:
    """verify(draft, sections=[]) returns passed=False without invoking ChatOpenAI.

    Per CODING_STANDARD §6.3: mock the LLM getter (ChatOpenAI) to RAISE if
    called — this proves the short-circuit fires before the client is built.
    The test is hermetic: no OPENAI_API_KEY required.
    """
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    def llm_must_not_be_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("ChatOpenAI was instantiated — empty-sections guard did not fire")

    with patch("app.grounding.ChatOpenAI", side_effect=llm_must_not_be_called):
        outcome = grounding.verify(draft=EMPTY_SECTIONS.draft, sections=[])

    assert isinstance(outcome, GroundingOutcome)
    assert outcome.passed is False
    assert outcome.reason == "claim_unsupported"
    assert outcome.result is not None
    assert outcome.result.passed is False
    # The draft itself should appear in unsupported_claims
    assert any(
        EMPTY_SECTIONS.draft.lower() in uc.lower()
        for uc in outcome.result.unsupported_claims
    ), f"Draft not found in unsupported_claims: {outcome.result.unsupported_claims}"
