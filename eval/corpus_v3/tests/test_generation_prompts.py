"""Generation prompt-builder tests — STRUCTURE only (CODING_STANDARD §6.2:
LLM output content is never asserted; the same discipline applies to prompt
*input* text here -- assert the reference text and language directive are
present, never assert exact wording beyond the pinned instruction strings
this module owns itself).

Covers issue #672.
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.generation.prompts import build_prompt
from eval.corpus_v3.generation.targets import GenerationTarget


def _target(**overrides) -> GenerationTarget:
    fields = dict(
        scenario_stratum="factoid",
        group_id="g1",
        heading="Some Heading",
        gold_section_ids=["a.md#some-heading"],
        reference_ids=["a.md#some-heading"],
        reference_text="The distinctive reference passage body.",
    )
    fields.update(overrides)
    return GenerationTarget(**fields)


def test_prompt_embeds_the_target_reference_text():
    prompt = build_prompt(_target(), language="en", variant_index=0)
    assert "The distinctive reference passage body." in prompt


def test_prompt_names_the_requested_language():
    prompt = build_prompt(_target(), language="en", variant_index=0)
    assert "English" in prompt
    zh_prompt = build_prompt(_target(), language="zh", variant_index=0)
    assert "Chinese" in zh_prompt


def test_prompt_carries_the_variant_index():
    prompt = build_prompt(_target(), language="en", variant_index=3)
    assert "3" in prompt


@pytest.mark.parametrize(
    "stratum,keyword",
    [
        ("factoid", "single-hop"),
        ("cross_doc", "combining"),
        ("version_conflict", "CURRENT"),
        ("unanswerable", "NOT actually answer"),
    ],
)
def test_each_stratum_carries_its_distinguishing_instruction(stratum, keyword):
    prompt = build_prompt(
        _target(scenario_stratum=stratum), language="en", variant_index=0
    )
    assert keyword in prompt


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        build_prompt(_target(), language="fr", variant_index=0)
