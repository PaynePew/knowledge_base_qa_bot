"""Shallow module per Ousterhout. Committed out-of-scope query set for the negative-case eval.

The corpus (``eval/negative_case/corpus``) covers refunds / shipping / account
ONLY. Every query below has no answer there, so the correct behaviour is Cannot
Confirm. Split by difficulty so the report can show where the threshold gate
holds (clearly out-of-scope) vs where it may leak (adjacent-absent).
"""

from __future__ import annotations

from .models import NegativeCase

NEGATIVE_CASES: list[NegativeCase] = [
    # --- clearly out of scope: no vocabulary overlap, the gate should always fire
    NegativeCase(
        "Which restaurants are nearby?", "local/geo search", "clearly_out_of_scope"
    ),
    NegativeCase("What's the weather tomorrow?", "weather", "clearly_out_of_scope"),
    NegativeCase(
        "How do I invest in the stock market?", "finance", "clearly_out_of_scope"
    ),
    NegativeCase(
        "Write me a poem about cats.", "open-ended generation", "clearly_out_of_scope"
    ),
    NegativeCase(
        "What is the capital of France?", "general trivia", "clearly_out_of_scope"
    ),
    NegativeCase("How do I bake sourdough bread?", "cooking", "clearly_out_of_scope"),
    NegativeCase(
        "Recommend a good action movie.", "entertainment", "clearly_out_of_scope"
    ),
    NegativeCase(
        "How tall is Mount Everest?", "geography trivia", "clearly_out_of_scope"
    ),
    NegativeCase(
        "Translate hello into Japanese.", "translation", "clearly_out_of_scope"
    ),
    NegativeCase("What is the meaning of life?", "open-ended", "clearly_out_of_scope"),
    # --- adjacent-absent: shares commerce vocabulary, but the specific answer is
    #     absent from the corpus (refund timeline / non-refundable / shipping
    #     estimates / international / password reset / account closing only)
    NegativeCase(
        "Do you price match competitors?",
        "no price-match policy in KB",
        "adjacent_absent",
    ),
    NegativeCase(
        "Can I gift wrap my order?", "no gift-wrap info in KB", "adjacent_absent"
    ),
    NegativeCase(
        "How many loyalty points do I earn per purchase?",
        "no loyalty program in KB",
        "adjacent_absent",
    ),
    NegativeCase(
        "Is there a student discount?", "no discount policy in KB", "adjacent_absent"
    ),
    NegativeCase(
        "Can I change my delivery address after ordering?",
        "no address-change policy in KB",
        "adjacent_absent",
    ),
]
