"""Shallow module per Ousterhout. Public surface: ``PROMPT_TEMPLATE_VERSION``,
``build_prompt``.

Per-(scenario-stratum, language) prompt construction for the corpus v3 query
generator (issue #672, ``generation/SPEC.md``). Prompt WORDING is not a
tested contract (CODING_STANDARD §6.2 — LLM output content is never
asserted); what tests here assert is STRUCTURE: the target's reference text
is actually embedded, the requested language is named, and the stratum's
distinguishing instruction (e.g. "do not answer" for ``unanswerable``) is
present verbatim so a future edit cannot silently drop it (mirrors §6.4's
"structural, never specific model wording" rule already applied to language
assertions elsewhere in this project).
"""

from __future__ import annotations

from .targets import GenerationTarget

# Bumped whenever prompt wording changes meaningfully enough that a
# regenerated batch should not be compared to an older one (mirrors
# ``eval.paraphrase_comparison.generation.templates.TEMPLATE_VERSION``).
PROMPT_TEMPLATE_VERSION = 1

_LANGUAGE_NAME = {"en": "English", "zh": "Simplified Chinese (zh)"}

_STRATUM_INSTRUCTION = {
    "factoid": (
        "Write ONE single-hop factual question a customer could ask that is "
        "fully answered by the reference passage below. Do not require "
        "synthesizing any other passage."
    ),
    "cross_doc": (
        "Write ONE question that can only be fully answered by combining "
        "information from every reference passage below (a single passage "
        "alone must be insufficient)."
    ),
    "version_conflict": (
        "Write ONE question about the CURRENT (newest) policy value in the "
        "reference passage below, phrased so an out-of-date answer would be "
        "wrong."
    ),
    "unanswerable": (
        "Write ONE question that closely resembles the topic of the "
        "reference passage below but that the reference passage does NOT "
        "actually answer — the correct response is a refusal, not an "
        "answer drawn from this passage."
    ),
}


def build_prompt(target: GenerationTarget, *, language: str, variant_index: int) -> str:
    """Render the generation prompt for one ``(target, language, variant)``.

    ``variant_index`` is embedded so repeated calls against the SAME target
    (``targets.sample_targets`` cycles through a small pool many times to
    reach n=909) are nudged toward distinct phrasings rather than the model
    returning the same text on every call; it carries no other semantics.

    Raises ``ValueError`` for an unknown ``language`` (fail-fast per
    CODING_STANDARD §4.1) — a silently-ignored language would otherwise
    generate an English query mislabeled ``zh`` and fail the QC gate's
    language check for the wrong reason.
    """
    if language not in _LANGUAGE_NAME:
        raise ValueError(
            f"build_prompt: unknown language {language!r}, "
            f"expected one of {sorted(_LANGUAGE_NAME)}"
        )
    instruction = _STRATUM_INSTRUCTION[target.scenario_stratum]
    return (
        f"{instruction}\n\n"
        f"Respond in {_LANGUAGE_NAME[language]} only.\n"
        f"Variant: {variant_index}\n\n"
        f"Reference passage ({target.heading}):\n"
        f"{target.reference_text}\n\n"
        "Also list the distinctive key tokens (words) the correct answer "
        "must contain, or an empty list if this is meant to be unanswerable."
    )
