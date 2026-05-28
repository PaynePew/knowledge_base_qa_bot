"""Shallow module per Ousterhout. Public surface: ``RULE``, ``ONE_SHOT``, ``build_prompt``.

Per-type prompt template for the ``implicit_reference`` Paraphrase Type (PRD #100, #102).

Rule: strip the EXPLICIT subject from the question so the topic is only implied
by context — e.g. drop "gift card" and ask "does it expire?". Tests whether
retrieval can recover the right Section when the anchoring noun is absent (the
hardest case for lexical retrieval, which leans on that noun's overlap).
"""

from __future__ import annotations

RULE = (
    "Rewrite the question so the EXPLICIT subject is removed — the topic should "
    "be only implied, the way a customer mid-conversation drops the noun and "
    "uses a pronoun or bare verb instead. Record the stripped subject in "
    "generation_notes. The question must still be answerable by this passage for "
    "a reader who knows the topic."
)

ONE_SHOT = (
    'Passage: "Gift cards never expire and incur no dormancy fees, so an unused '
    'balance stays available indefinitely."\n'
    'Rewritten (implicit_reference): "Does the balance ever run out if I leave it '
    'unused for a couple of years?"\n'
    "generation_notes: stripped subject = gift card"
)


def build_prompt(*, heading: str, body: str) -> str:
    """Render the implicit_reference user prompt for one Gold Section."""
    return (
        f"{RULE}\n\n"
        f"Example:\n{ONE_SHOT}\n\n"
        f"Now rewrite a subject-stripped question answered by this passage.\n"
        f"Section heading: {heading}\n"
        f"Passage:\n{body}\n"
    )
