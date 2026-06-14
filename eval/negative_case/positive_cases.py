"""Shallow module per Ousterhout. Committed in-scope query set for threshold calibration (#253).

Every query here HAS an answer in ``eval/negative_case/corpus`` (refund / shipping
/ account), so the correct behaviour is to answer, not refuse. Refusing one is an
*over-refusal* — the cost of raising ``KB_SCORE_THRESHOLD`` too high. Paired with
the negative set, these two sides bound the threshold trade-off.
"""

from __future__ import annotations

from .models import PositiveCase

POSITIVE_CASES: list[PositiveCase] = [
    PositiveCase("How long do refunds take?", "refund timeline"),
    PositiveCase("When will my refund be credited?", "refund timeline"),
    PositiveCase("What items are non-refundable?", "non-refundable items"),
    PositiveCase("How long does standard shipping take?", "delivery estimates"),
    PositiveCase("Is expedited shipping available?", "delivery estimates / expedited"),
    PositiveCase("Do you ship internationally?", "international shipping"),
    PositiveCase(
        "Who pays customs duties on international orders?", "international shipping"
    ),
    PositiveCase("How do I reset my password?", "password reset"),
    PositiveCase("How do I close my account?", "closing account"),
    PositiveCase("What happens to my data when I close my account?", "closing account"),
]
