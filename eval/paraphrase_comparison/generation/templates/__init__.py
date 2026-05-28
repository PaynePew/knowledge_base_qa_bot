"""Per-type prompt templates for the five Core Paraphrase Types (PRD #100, #102).

One module per Core type, each exporting ``RULE``, ``ONE_SHOT`` and
``build_prompt(*, heading, body)``. ``CORE_TEMPLATES`` registers them by
Paraphrase Type name so the generator can look the builder up by type. The two
Structural probe types (``typo_fatfinger``, ``industry_jargon``) are NOT here —
they are hand-written, not LLM-generated (CONTEXT.md § Phase 8 > Paraphrase Type).

``TEMPLATE_VERSION`` is recorded in the ``queries.yaml`` metadata block so a
regenerated query set is traceable to the exact prompt wording that produced it.
Bump it whenever any template's ``RULE``/``ONE_SHOT`` changes.
"""

from __future__ import annotations

from collections.abc import Callable

from . import (
    implicit_reference,
    specificity_narrowing,
    synonym_swap,
    verbosity_expansion,
    word_reorder,
)

TEMPLATE_VERSION = "v1"

# Type name -> prompt builder. Keys are the Core Paraphrase Type literals from
# models.ParaphraseType; the two probe types are intentionally absent.
CORE_TEMPLATES: dict[str, Callable[..., str]] = {
    "synonym_swap": synonym_swap.build_prompt,
    "word_reorder": word_reorder.build_prompt,
    "verbosity_expansion": verbosity_expansion.build_prompt,
    "specificity_narrowing": specificity_narrowing.build_prompt,
    "implicit_reference": implicit_reference.build_prompt,
}
