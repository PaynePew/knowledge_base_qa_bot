"""Hermetic string-presence tests for the Source-language directive in synthesis prompts.

AC (issue #166):
  - The "write in the Source's language" directive is present in both
    _CONCEPT_SYSTEM_PROMPT and _ENTITY_SYSTEM_PROMPT.
  - The red-link slug instruction acknowledges Unicode / CJK slugs (consistent
    with the Unicode-preserving slugify from Slice 16-1).

Prior art: test_ingest_grounding_failure.py::test_red_link_rule_block_appears_in_*
uses the same pattern ("wikilinks" in _CONCEPT_SYSTEM_PROMPT).

No LLM calls, no filesystem writes — pure string inspection.
"""

from __future__ import annotations

from app.templates import _CONCEPT_SYSTEM_PROMPT, _ENTITY_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# AC1: Source-language directive present in both prompts
# ---------------------------------------------------------------------------


def test_language_directive_in_concept_system_prompt():
    """Concept synthesis prompt instructs the model to write in the Source's language."""
    assert "same language as the Source" in _CONCEPT_SYSTEM_PROMPT, (
        "Expected 'same language as the Source' directive in _CONCEPT_SYSTEM_PROMPT"
    )


def test_language_directive_in_entity_system_prompt():
    """Entity synthesis prompt instructs the model to write in the Source's language."""
    assert "same language as the Source" in _ENTITY_SYSTEM_PROMPT, (
        "Expected 'same language as the Source' directive in _ENTITY_SYSTEM_PROMPT"
    )


# ---------------------------------------------------------------------------
# AC2: Red-link slug instruction covers CJK / non-ASCII slugs
# ---------------------------------------------------------------------------


def test_concept_prompt_red_link_slug_covers_cjk():
    """Concept prompt red-link rule acknowledges non-ASCII / CJK slug form."""
    assert "CJK" in _CONCEPT_SYSTEM_PROMPT or "non-ASCII" in _CONCEPT_SYSTEM_PROMPT, (
        "Expected CJK/non-ASCII slug mention in _CONCEPT_SYSTEM_PROMPT red-link rule"
    )


def test_entity_prompt_red_link_slug_covers_cjk():
    """Entity prompt red-link rule acknowledges non-ASCII / CJK slug form."""
    assert "CJK" in _ENTITY_SYSTEM_PROMPT or "non-ASCII" in _ENTITY_SYSTEM_PROMPT, (
        "Expected CJK/non-ASCII slug mention in _ENTITY_SYSTEM_PROMPT red-link rule"
    )


# ---------------------------------------------------------------------------
# Regression: existing wikilinks / do NOT verify markers still present
# (ensures we didn't accidentally remove the red-link block)
# ---------------------------------------------------------------------------


def test_concept_prompt_wikilinks_marker_still_present():
    """Updating the prompt did not remove the existing 'wikilinks' keyword."""
    assert "wikilinks" in _CONCEPT_SYSTEM_PROMPT


def test_entity_prompt_wikilinks_marker_still_present():
    """Updating the prompt did not remove the existing 'wikilinks' keyword."""
    assert "wikilinks" in _ENTITY_SYSTEM_PROMPT
