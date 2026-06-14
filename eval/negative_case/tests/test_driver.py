"""Integration tests — drive the REAL pre-LLM Cannot Confirm gate (LLM-free).

Builds the committed in-scope corpus into markdown_kb's Section Index, then
asserts the gate's discrimination: a clearly out-of-scope query is refused, an
in-scope query is not. BM25 is deterministic, so these assertions are stable
without any LLM call.
"""

from __future__ import annotations

from eval.negative_case.driver import evaluate_case, index_corpus


def test_out_of_scope_query_is_refused():
    """A query with no corpus overlap → Cannot Confirm (gate fires)."""
    index_corpus()
    outcome = evaluate_case("Which restaurants are nearby?")
    assert outcome.refused is True
    assert outcome.reason in {"retrieval_empty", "below_threshold"}


def test_in_scope_query_is_not_refused():
    """An in-scope query → retrieval clears the threshold, no refusal."""
    index_corpus()
    outcome = evaluate_case("How long do refunds take?")
    assert outcome.refused is False
    assert outcome.reason == "answered"
    assert outcome.top_score > 0.0


def test_index_corpus_populates_sections():
    """index_corpus builds a non-empty index over the committed corpus."""
    pages, sections = index_corpus()
    assert sections > 0
