"""Fixture-shape tests (#608).

Asserts the hand-written ``CASES`` mirror ``gateway.app.conversation_store``'s
documented turn-record shape (CODING_STANDARD §6.5 fixture fidelity) — every
history turn must carry all five required keys, not a simplified subset, so a
regression that only triggers on the real shape would be caught here too.
"""

from __future__ import annotations

from eval.contaminated_session.sessions import CASES

_REQUIRED_TURN_KEYS = {"question", "answer", "stack", "grounding_reason", "ts"}


def test_every_case_has_a_unique_name():
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))


def test_contaminated_history_turns_match_conversation_store_shape():
    for case in CASES:
        assert case.contaminated_history, (
            f"{case.name}: contaminated_history must be non-empty"
        )
        for turn in case.contaminated_history:
            assert set(turn) == _REQUIRED_TURN_KEYS, (
                f"{case.name}: turn {turn!r} missing/extra keys vs "
                f"conversation_store shape {_REQUIRED_TURN_KEYS}"
            )


def test_clean_history_is_the_no_contamination_control():
    """The control arm is empty history — no prior turn at all — so
    ``rewrite_query`` takes its turn-1-passthrough branch for every case."""
    for case in CASES:
        assert case.clean_history == []


def test_followup_question_is_literal_not_pre_resolved():
    """The fixture must not do the rewrite's job for it — the follow-up stays
    a raw, potentially elliptical ask (e.g. containing a bare reference like
    "that" / "it") for the driver to feed into rewrite_fn."""
    for case in CASES:
        assert case.followup_question
        assert case.followup_question == case.followup_question.strip()
