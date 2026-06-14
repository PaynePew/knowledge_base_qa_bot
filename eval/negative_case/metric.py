"""Deep module per Ousterhout. Public surface: ``is_refusal``, ``correct_refusal_rate``.

Deterministic negative-case metric: did the bot correctly refuse?

The metric is binary per case (refused = 1.0, answered = 0.0); the headline is the
**correct-refusal rate** (the "fallback rate" the Week 6 deck asks for). Kept as
pure functions — no LLM, no DeepEval ceremony — because the refusal decision is
already made deterministically at the pre-LLM gate (see ``driver``).
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import RefusalOutcome


def is_refusal(outcome: RefusalOutcome) -> bool:
    """True iff the system correctly refused (Cannot Confirm) the out-of-scope query."""
    return outcome.refused


def correct_refusal_rate(outcomes: Iterable[RefusalOutcome]) -> float:
    """Fraction of negative cases correctly refused — the fallback rate.

    1.0 = every out-of-scope query was refused; 0.0 = the bot answered them all.
    Empty input returns 0.0 (no evidence of correct refusal; never divides by zero).
    """
    items = list(outcomes)
    if not items:
        return 0.0
    return sum(1 for o in items if is_refusal(o)) / len(items)
