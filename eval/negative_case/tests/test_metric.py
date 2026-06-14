"""Pure-metric tests — no index, no LLM."""

from __future__ import annotations

from eval.negative_case.metric import correct_refusal_rate, is_refusal
from eval.negative_case.models import RefusalOutcome


def _outcome(refused: bool) -> RefusalOutcome:
    return RefusalOutcome(
        query="q",
        refused=refused,
        reason="below_threshold" if refused else "answered",
        top_score=0.0 if refused else 9.9,
    )


def test_is_refusal_reads_the_flag():
    assert is_refusal(_outcome(True)) is True
    assert is_refusal(_outcome(False)) is False


def test_correct_refusal_rate_fraction():
    """Rate is the fraction of cases correctly refused."""
    outcomes = [_outcome(True), _outcome(True), _outcome(False)]
    assert correct_refusal_rate(outcomes) == 2 / 3


def test_correct_refusal_rate_all_refused():
    assert correct_refusal_rate([_outcome(True), _outcome(True)]) == 1.0


def test_correct_refusal_rate_empty_is_zero():
    """Empty input → 0.0 (no evidence of correct refusal, never divides by zero)."""
    assert correct_refusal_rate([]) == 0.0
