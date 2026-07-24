"""Content-axis tests — external behaviour only (CODING_STANDARD §0.2).

Every ``AnswerRecord`` below is hand-authored (CODING_STANDARD §6.5), never
LLM-generated: this module scores already-produced answers, it does not
produce them.
"""

from __future__ import annotations

import pytest
from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

from eval.corpus_v3.content_axes import (
    AnswerRecord,
    build_cost_amortization_curve,
    contradiction_leak,
    correct_refusal,
    cost_per_grounded_correct_answer,
    grounding_pass,
    is_refusal,
)


def _answer(text: str, cited: frozenset[str] = frozenset()) -> AnswerRecord:
    return AnswerRecord(
        query_id="q1", arm="wiki", answer_text=text, cited_source_ids=cited
    )


# ---------------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------------
def test_is_refusal_true_for_the_exact_sentinel():
    assert is_refusal(_answer(CANNOT_CONFIRM_PHRASE)) is True


def test_is_refusal_false_for_a_grounded_answer():
    assert is_refusal(_answer("Store hours are 9-6.", frozenset({"a.md#h"}))) is False


def test_is_refusal_does_not_match_a_paraphrase():
    """Contract is the LITERAL sentinel (ADR-0001), not fuzzy matching."""
    assert (
        is_refusal(_answer("I cannot confirm this from the knowledge base.")) is False
    )


# ---------------------------------------------------------------------------
# grounding_pass
# ---------------------------------------------------------------------------
def test_grounding_pass_true_when_every_citation_was_retrieved():
    answer = _answer("Hours are 9-6. [Source: a.md#h]", frozenset({"a.md#h"}))
    assert grounding_pass(answer, retrieved_source_ids=["a.md#h", "b.md#h"]) is True


def test_grounding_pass_false_on_refusal():
    answer = _answer(CANNOT_CONFIRM_PHRASE)
    assert grounding_pass(answer, retrieved_source_ids=["a.md#h"]) is False


def test_grounding_pass_false_with_no_citations():
    answer = _answer("Hours are 9-6.", frozenset())
    assert grounding_pass(answer, retrieved_source_ids=["a.md#h"]) is False


def test_grounding_pass_false_on_a_fabricated_citation():
    """A citation naming an id outside the retrieved pool is unsupported."""
    answer = _answer(
        "Hours are 9-6. [Source: made-up.md#h]", frozenset({"made-up.md#h"})
    )
    assert grounding_pass(answer, retrieved_source_ids=["a.md#h"]) is False


# ---------------------------------------------------------------------------
# correct_refusal
# ---------------------------------------------------------------------------
def test_correct_refusal_true_when_an_unanswerable_query_is_refused():
    answer = _answer(CANNOT_CONFIRM_PHRASE)
    assert correct_refusal(answer, is_unanswerable=True) is True


def test_correct_refusal_false_when_an_unanswerable_query_is_answered_anyway():
    answer = _answer("Hours are 9-6.", frozenset({"a.md#h"}))
    assert correct_refusal(answer, is_unanswerable=True) is False


def test_correct_refusal_raises_for_an_answerable_query():
    answer = _answer("Hours are 9-6.", frozenset({"a.md#h"}))
    with pytest.raises(ValueError, match="is_unanswerable=True"):
        correct_refusal(answer, is_unanswerable=False)


# ---------------------------------------------------------------------------
# contradiction_leak
# ---------------------------------------------------------------------------
def test_contradiction_leak_true_when_a_superseded_version_is_cited():
    answer = _answer(
        "Labels cost $5. [Source: return_shipping_v1.md#label-cost]",
        frozenset({"return_shipping_v1.md#label-cost"}),
    )
    leaked = contradiction_leak(
        answer,
        leak_source_ids={
            "return_shipping_v1.md#label-cost",
            "return_shipping_v2.md#label-cost",
        },
    )
    assert leaked is True


def test_contradiction_leak_false_when_only_the_current_version_is_cited():
    answer = _answer(
        "Labels are free. [Source: return_shipping_v3.md#label-cost]",
        frozenset({"return_shipping_v3.md#label-cost"}),
    )
    leaked = contradiction_leak(
        answer,
        leak_source_ids={
            "return_shipping_v1.md#label-cost",
            "return_shipping_v2.md#label-cost",
        },
    )
    assert leaked is False


def test_contradiction_leak_false_on_refusal():
    """A refusal asserts nothing, so it cannot leak."""
    answer = _answer(CANNOT_CONFIRM_PHRASE)
    leaked = contradiction_leak(
        answer, leak_source_ids={"return_shipping_v1.md#label-cost"}
    )
    assert leaked is False


# ---------------------------------------------------------------------------
# cost_per_grounded_correct_answer
# ---------------------------------------------------------------------------
def test_cost_per_grounded_correct_answer_divides_cost_by_count():
    assert cost_per_grounded_correct_answer(10.0, 20) == pytest.approx(0.5)


def test_cost_per_grounded_correct_answer_none_when_zero_grounded_correct():
    assert cost_per_grounded_correct_answer(10.0, 0) is None


def test_cost_per_grounded_correct_answer_raises_on_negative_count():
    with pytest.raises(ValueError, match="grounded_correct_count"):
        cost_per_grounded_correct_answer(10.0, -1)


# ---------------------------------------------------------------------------
# build_cost_amortization_curve
# ---------------------------------------------------------------------------
def test_build_cost_amortization_curve_divides_build_cost_by_each_volume():
    curve = build_cost_amortization_curve(4.4, [1, 10, 100])
    assert curve == {
        1: pytest.approx(4.4),
        10: pytest.approx(0.44),
        100: pytest.approx(0.044),
    }


def test_build_cost_amortization_curve_skips_zero_and_negative_volumes():
    curve = build_cost_amortization_curve(4.4, [0, -5, 10])
    assert curve == {10: pytest.approx(0.44)}
