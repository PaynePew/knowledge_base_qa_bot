"""Shallow module per Ousterhout. Public surface: ``RULE``, ``ONE_SHOT``, ``build_prompt``.

Per-type prompt template for the ``word_reorder`` Paraphrase Type (PRD #100, #102).

Rule: keep (most of) the original vocabulary but restructure the sentence —
fronting a clause, switching active/passive, or turning a statement into an
indirect question. Tests whether retrieval is robust to syntax when the lexical
content is largely preserved.
"""

from __future__ import annotations

RULE = (
    "Rewrite the question keeping the same key words but RESTRUCTURING the "
    "sentence: change the clause order, switch active/passive voice, or recast "
    "it as an indirect question. Do not introduce new synonyms — the lexical "
    "content stays; only the word order and structure change."
)

ONE_SHOT = (
    'Passage: "Expedited delivery guarantees arrival within two business days "'
    'for a twelve-dollar surcharge."\n'
    'Rewritten (word_reorder): "For a twelve-dollar surcharge, within how many '
    'business days does expedited delivery arrive?"'
)


def build_prompt(*, heading: str, body: str) -> str:
    """Render the word_reorder user prompt for one Gold Section."""
    return (
        f"{RULE}\n\n"
        f"Example:\n{ONE_SHOT}\n\n"
        f"Now rewrite a question answered by this passage.\n"
        f"Section heading: {heading}\n"
        f"Passage:\n{body}\n"
    )
