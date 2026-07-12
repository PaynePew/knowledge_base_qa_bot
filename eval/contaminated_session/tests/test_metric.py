"""Metric tests (#608) — pure functions over fixed fixture strings, LLM-free.

Exact-value assertions are safe here: ``compute_drift`` is deterministic
Python over the production tokenizer, never LLM output text or a
corpus-sensitive BM25 score (CODING_STANDARD §6.2 bars those two, not a
deterministic metric module's own output on fixed inputs).
"""

from __future__ import annotations

from eval.contaminated_session.metric import compute_drift


def test_identical_strings_have_full_overlap_and_unit_length_ratio():
    d = compute_drift("same text", "same text")
    assert d.token_overlap == 1.0
    assert d.length_ratio == 1.0
    assert d.added_tokens == ()


def test_disjoint_strings_have_zero_overlap():
    d = compute_drift("completely different", "unrelated words here")
    assert d.token_overlap == 0.0
    assert d.added_tokens == ("here", "unrelated", "words")


def test_rewrite_that_appends_context_has_partial_overlap_and_longer_length():
    literal = "And what if I don't have the receipt?"
    rewritten = f"{literal} [How long do refunds take?]"
    d = compute_drift(literal, rewritten)
    # The rewrite is a strict superset of the literal's tokens plus the
    # bracketed context, so overlap is between the two extremes and length
    # grew — the over-specification shape #579 reported.
    assert 0.0 < d.token_overlap < 1.0
    assert d.length_ratio > 1.0
    assert d.added_tokens == ("long", "refunds", "take")


def test_empty_inputs_do_not_divide_by_zero():
    """Both strings tokenize to nothing (pure stop words) — union is empty,
    defined as full overlap rather than raising."""
    d = compute_drift("a", "an")
    assert d.token_overlap == 1.0
    assert d.length_ratio == 0.0
