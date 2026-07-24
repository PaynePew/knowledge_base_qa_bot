"""Overlap-stratum computation tests — external behaviour only (CODING_STANDARD §0.2).

Covers issue #660 AC-2's "lexical-overlap stratum computed ... at generation
time": the ratio and its high/low classification, for both en (word tokens)
and zh (CJK bigrams) text, using the shared markdown_kb tokeniser (ADR-0002).
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.generation.overlap import (
    DEFAULT_OVERLAP_THRESHOLD,
    classify_overlap_stratum,
    lexical_overlap_ratio,
)


def test_identical_text_has_ratio_one():
    text = "How long is the return window for damaged items?"
    assert lexical_overlap_ratio(text, [text]) == pytest.approx(1.0)


def test_fully_paraphrased_query_has_low_ratio():
    query = "Within what timeframe can I send a defective product back?"
    reference = "The return window for damaged items is 30 days from delivery."
    ratio = lexical_overlap_ratio(query, [reference])
    assert ratio < DEFAULT_OVERLAP_THRESHOLD


def test_ratio_is_containment_not_jaccard():
    # A long reference containing every query token should score 1.0 even
    # though the reference has far more tokens than the query — containment,
    # not symmetric Jaccard (module docstring: DPR/SQuAD-style measure).
    query = "return window"
    reference = (
        "the return window policy covers damaged items subscription orders "
        "and warranty claims across every product category in the catalog"
    )
    assert lexical_overlap_ratio(query, [reference]) == pytest.approx(1.0)


def test_no_reference_texts_returns_zero():
    assert lexical_overlap_ratio("some query text", []) == 0.0


def test_reference_texts_with_no_tokens_returns_zero():
    assert lexical_overlap_ratio("some query text", ["   ", ""]) == 0.0


def test_empty_query_text_raises():
    with pytest.raises(ValueError, match="no tokens"):
        lexical_overlap_ratio("   ", ["some reference text"])


def test_zh_query_and_reference_use_cjk_tokenisation():
    query = "退貨期限是多久？"
    reference = "退貨期限是收到商品後三十天內。"
    ratio = lexical_overlap_ratio(query, [reference])
    assert ratio > 0.0


# ---------------------------------------------------------------------------
# classify_overlap_stratum
# ---------------------------------------------------------------------------
def test_classify_high_overlap_at_default_threshold():
    text = "return window for damaged items"
    assert classify_overlap_stratum(text, [text]) == "high_overlap"


def test_classify_low_overlap_below_default_threshold():
    query = "Within what timeframe can I send a defective product back?"
    reference = "The return window for damaged items is 30 days from delivery."
    assert classify_overlap_stratum(query, [reference]) == "low_overlap"


def test_classify_respects_custom_threshold():
    query = "return items damaged today"
    reference = "return window damaged"
    ratio = lexical_overlap_ratio(query, [reference])
    # Pick a threshold just above the observed ratio to flip the classification.
    assert classify_overlap_stratum(query, [reference], threshold=ratio + 0.01) == (
        "low_overlap"
    )
    assert classify_overlap_stratum(query, [reference], threshold=ratio) == (
        "high_overlap"
    )
