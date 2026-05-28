"""Shallow module per Ousterhout. Public surface: ``RULE``, ``ONE_SHOT``, ``build_prompt``.

Per-type prompt template for the ``verbosity_expansion`` Paraphrase Type (PRD #100, #102).

Rule: pad the canonical question with conversational framing, hedging, and
irrelevant preamble while keeping exactly one underlying information need. Tests
whether retrieval survives noise tokens diluting the query's signal.
"""

from __future__ import annotations

RULE = (
    "Rewrite the question as a LONGER, chattier message: add a sentence or two "
    "of polite framing, background context, or hedging around the same single "
    "information need. The extra words must be conversational filler, NOT new "
    "facts. The core question stays answerable by the passage."
)

ONE_SHOT = (
    'Passage: "A reset link valid for one hour is sent to that address."\n'
    'Rewritten (verbosity_expansion): "Hi, sorry to bother you — I think I '
    'locked myself out earlier today and I am a bit confused about the process. '
    'Once I ask for a password reset email, roughly how long does that link '
    'actually stay valid before it stops working?"'
)


def build_prompt(*, heading: str, body: str) -> str:
    """Render the verbosity_expansion user prompt for one Gold Section."""
    return (
        f"{RULE}\n\n"
        f"Example:\n{ONE_SHOT}\n\n"
        f"Now rewrite a question answered by this passage.\n"
        f"Section heading: {heading}\n"
        f"Passage:\n{body}\n"
    )
