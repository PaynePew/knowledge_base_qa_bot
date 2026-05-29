"""Shallow module per Ousterhout. Public surface: ``RULE``, ``ONE_SHOT``, ``build_prompt``.

Per-type prompt template for the ``specificity_narrowing`` Paraphrase Type (PRD #100, #102).

Rule: a Gold Section that answers several distinct questions can source this
type; the prompt asks about ONE high-distinctiveness sub-fact only, not the
section's headline topic. The former ``multi_sub_fact`` flag that restricted
sampling to such sections was dropped in issue #142 — sub-fact narrowing is now
steered by this prompt's RULE over any sampled Gold Section, and downstream by
the Synthesizer's context-bound evolutions (PRD #137).
Tests whether retrieval can land the right Section from a narrow, specific cue.
"""

from __future__ import annotations

RULE = (
    "This passage contains SEVERAL distinct facts. Pick ONE specific, "
    "high-distinctiveness sub-fact (not the section's broad topic) and write a "
    "narrow question that asks ONLY about that sub-fact. Name the targeted "
    "sub-fact in generation_notes. The question must still be answered by THIS "
    "passage and no other."
)

ONE_SHOT = (
    'Passage: "Approved refunds are issued within five to seven business days. '
    "Customers paying by bank transfer should allow an additional three business "
    'days because settlement times vary between banks."\n'
    'Rewritten (specificity_narrowing): "If I paid by bank transfer, how many '
    'extra days should I expect before my refund settles?"\n'
    "generation_notes: targeted sub-fact = bank-transfer +3 business days"
)


def build_prompt(*, heading: str, body: str) -> str:
    """Render the specificity_narrowing user prompt for one Gold Section."""
    return (
        f"{RULE}\n\n"
        f"Example:\n{ONE_SHOT}\n\n"
        f"Now write a narrow sub-fact question answered by this passage.\n"
        f"Section heading: {heading}\n"
        f"Passage:\n{body}\n"
    )
