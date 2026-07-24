"""Deep module per Ousterhout. Public surface: ``AnswerRecord``, ``is_refusal``,
``grounding_pass``, ``correct_refusal``, ``contradiction_leak``,
``cost_per_grounded_correct_answer``, ``build_cost_amortization_curve``.

Content-axis + cost-quality measurement primitives for the corpus v3 verdict
report (issue #662, ADR-0045's three content axes + PRD #654 user story 21).
Every function here is pure and deterministic given its inputs -- nothing in
this module calls an LLM or an app's query surface; producing the
``AnswerRecord``s these functions score is ``run_verdict.py``'s job. Keeping
the axis math itself pure is what lets the report-assembly pipeline
(``verdict_report.py``) be unit-tested on canned axis results (issue #662
AC 1: "report-assembly and clause-walkthrough logic unit-tested on canned
axis results").

The refusal sentinel is imported, never paraphrased (implement.md trap #2 /
ADR-0001): all three apps share the exact SAME literal phrase --
``hybrid_kb/tests/test_query.py`` asserts ``markdown_kb``'s and
``vector_rag``'s ``CANNOT_CONFIRM_PHRASE`` constants are equal -- so importing
``markdown_kb``'s is correct for scoring every stack's answers here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE


@dataclass(frozen=True)
class AnswerRecord:
    """One retrieval arm's produced answer for one query -- the unit every
    content axis scores. ``cited_source_ids`` are the Section ids the
    answer's ``[Source: ...]`` citations name, already parsed out of the
    answer text (not this module's concern how)."""

    query_id: str
    arm: str
    answer_text: str
    cited_source_ids: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Public API — the three ADR-0045 content axes
# ---------------------------------------------------------------------------
def is_refusal(answer: AnswerRecord) -> bool:
    """True iff ``answer`` is the exact Cannot Confirm sentinel (no paraphrase
    match -- ADR-0001's contract is the literal string)."""
    return answer.answer_text.strip() == CANNOT_CONFIRM_PHRASE


def grounding_pass(answer: AnswerRecord, retrieved_source_ids: Iterable[str]) -> bool:
    """Grounding pass rate axis (ADR-0045): True iff ``answer`` is NOT a
    refusal, cites at least one Source, AND every cited Source id was
    actually retrieved for this query. A citation naming an id outside the
    retrieved pool is a fabricated / unsupported citation, not grounded.
    """
    if is_refusal(answer):
        return False
    if not answer.cited_source_ids:
        return False
    retrieved = set(retrieved_source_ids)
    return answer.cited_source_ids <= retrieved


def correct_refusal(answer: AnswerRecord, *, is_unanswerable: bool) -> bool:
    """Correct-refusal rate axis (ADR-0045): defined ONLY for unanswerable
    queries -- there is no "correct refusal" on an answerable query. True
    iff ``answer`` refused.

    Raises ``ValueError`` when ``is_unanswerable`` is False (fail-fast per
    CODING_STANDARD §4.1) rather than silently returning a meaningless
    result: callers must filter to the unanswerable stratum before scoring
    this axis, exactly as ``eval.corpus_v3.aggregation`` filters by stratum
    for every other metric.
    """
    if not is_unanswerable:
        raise ValueError(
            "correct_refusal is only defined for is_unanswerable=True queries "
            "(ADR-0045's correct-refusal axis has no meaning on an answerable "
            "query); callers must filter to the unanswerable stratum first"
        )
    return is_refusal(answer)


def contradiction_leak(answer: AnswerRecord, leak_source_ids: Iterable[str]) -> bool:
    """Contradiction-leak rate axis (ADR-0045, the curated layer's "home
    axis"): True iff ``answer`` cites ANY of ``leak_source_ids`` -- the ids a
    correct answer must NOT cite as authoritative (a superseded version, or
    one side of an unresolved contradiction cited without the honest
    both-sides / wiki framing). A refusal never leaks -- it asserts nothing.
    """
    if is_refusal(answer):
        return False
    return bool(answer.cited_source_ids & set(leak_source_ids))


# ---------------------------------------------------------------------------
# Public API — cost x quality (PRD #654 user story 21)
# ---------------------------------------------------------------------------
def cost_per_grounded_correct_answer(
    total_usd: float, grounded_correct_count: int
) -> float | None:
    """PRD #654 user story 21: total cost divided by the count of grounded
    AND correct answers -- the metric where cost and quality argue in the
    same chart.

    Returns ``None`` when ``grounded_correct_count`` is 0 (division is
    undefined, not infinite) rather than raising -- a stack that produced
    zero grounded-correct answers is a real, reportable outcome, not an
    error condition.
    """
    if grounded_correct_count < 0:
        raise ValueError(
            f"grounded_correct_count must be >= 0, got {grounded_correct_count!r}"
        )
    if grounded_correct_count == 0:
        return None
    return total_usd / grounded_correct_count


def build_cost_amortization_curve(
    build_cost_usd: float, query_volumes: Iterable[int]
) -> dict[int, float]:
    """PRD #654 user story 21's amortization curve: the one-time build cost
    divided by query volume, at each volume in ``query_volumes`` -- shows the
    curated layer's fixed cost diluting as the corpus is queried more.

    A volume of 0 (or negative) is skipped rather than raising: the curve is
    meant to be swept over a range of volumes, not asserted point-by-point,
    and 0 is division-undefined.
    """
    return {v: build_cost_usd / v for v in query_volumes if v > 0}
