"""Generation QC gate tests — external behaviour only (CODING_STANDARD §0.2).

Covers issue #660 AC-2: every generated query must record its
generating_family, carry Key Tokens that survive tokenisation, and have its
language label agree with its text's detected script (the zh slice's "own
gate", ADR-0045 Prerequisite 3).
"""

from __future__ import annotations

from eval.corpus_v3.generation.qc import check_generated_query
from eval.corpus_v3.query_schema import Query


def _query(**overrides) -> Query:
    fields = dict(
        query_id="q-001",
        text="How long is the return window?",
        scenario_stratum="factoid",
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=["returns_policy.md#return-window"],
        key_tokens=["return", "window", "days"],
        generating_family="gpt-4o-mini",
    )
    fields.update(overrides)
    return Query(**fields)


def test_well_formed_query_passes():
    verdict = check_generated_query(_query())
    assert verdict.rejected is False
    assert verdict.reasons == []


def test_missing_generating_family_is_rejected():
    verdict = check_generated_query(_query(generating_family=""))
    assert verdict.rejected is True
    assert any("generating_family" in reason for reason in verdict.reasons)


def test_whitespace_only_generating_family_is_rejected():
    verdict = check_generated_query(_query(generating_family="   "))
    assert verdict.rejected is True


def test_all_stopword_key_tokens_is_rejected():
    verdict = check_generated_query(_query(key_tokens=["the", "a", "is"]))
    assert verdict.rejected is True
    assert any("stop-words" in reason for reason in verdict.reasons)


def test_unanswerable_query_with_no_key_tokens_is_not_flagged_for_stopwords():
    query = Query(
        query_id="q-002",
        text="What is the CEO's home address?",
        scenario_stratum="unanswerable",
        overlap_stratum="low_overlap",
        language="en",
        generating_family="human",
    )
    verdict = check_generated_query(query)
    assert verdict.rejected is False


def test_language_mismatch_en_labelled_zh_is_rejected():
    verdict = check_generated_query(
        _query(text="How long is the return window?", language="zh")
    )
    assert verdict.rejected is True
    assert any("detect_lang" in reason for reason in verdict.reasons)


def test_language_mismatch_zh_labelled_en_is_rejected():
    verdict = check_generated_query(
        _query(
            text="退貨期限是多久？",
            language="en",
            gold_section_ids=["returns_policy.md#return-window"],
            key_tokens=["退貨", "期限"],
        )
    )
    assert verdict.rejected is True


def test_correctly_labelled_zh_query_passes_the_language_gate():
    verdict = check_generated_query(
        _query(
            text="退貨期限是多久？",
            language="zh",
            gold_section_ids=["returns_policy.md#return-window"],
            key_tokens=["退貨", "期限"],
        )
    )
    assert verdict.rejected is False


def test_multiple_failures_are_all_reported():
    verdict = check_generated_query(
        _query(generating_family="", key_tokens=["the", "a"])
    )
    assert verdict.rejected is True
    assert len(verdict.reasons) == 2
