"""Shallow module per Ousterhout. Public surface: ``RULE``, ``ONE_SHOT``, ``build_prompt``.

Per-type prompt template for the ``synonym_swap`` Paraphrase Type (PRD #100, #102).

Rule: rewrite the canonical question replacing its salient content words with
synonyms, deliberately AVOIDING the original Gold Section vocabulary — this
stresses lexical (BM25) retrieval, where surface-form overlap drives the score.
"""

from __future__ import annotations

RULE = (
    "Rewrite the question so it asks for the SAME fact but AVOIDS the salient "
    "content words used in the source passage. Swap each key noun/verb for a "
    "natural synonym a real customer might use. Do not reuse the passage's "
    "distinctive vocabulary. Keep it a single natural question."
)

ONE_SHOT = (
    'Passage: "Customers may return most items within 30 days of delivery for a '
    'full refund."\n'
    'Rewritten (synonym_swap): "How long do I have to send a product back for a '
    'complete reimbursement?"'
)


def build_prompt(*, heading: str, body: str) -> str:
    """Render the synonym_swap user prompt for one Gold Section."""
    return (
        f"{RULE}\n\n"
        f"Example:\n{ONE_SHOT}\n\n"
        f"Now rewrite a question answered by this passage.\n"
        f"Section heading: {heading}\n"
        f"Passage:\n{body}\n"
    )
