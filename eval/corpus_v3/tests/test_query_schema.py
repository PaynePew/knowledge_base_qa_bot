"""Stratified query schema tests — external behaviour only (CODING_STANDARD §0.2).

Covers the round-trip + validation acceptance criterion for issue #659: a
well-formed query file survives a dump/load round trip, and every way a query
can be unlabeled or under-specified is rejected with a ``ValueError`` rather
than silently accepted or silently dropped.
"""

from __future__ import annotations

import pytest
import yaml

from eval.corpus_v3.query_schema import (
    LANGUAGES,
    OVERLAP_STRATA,
    SCENARIO_STRATA,
    Query,
    dump_queries,
    load_queries,
    query_from_dict,
    query_to_dict,
)


def _factoid_query(**overrides) -> Query:
    fields = dict(
        query_id="q-001",
        text="How long is the return window?",
        scenario_stratum="factoid",
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=["returns_policy.md#return-window"],
        key_tokens=["return", "window", "days"],
    )
    fields.update(overrides)
    return Query(**fields)


def _unanswerable_query(**overrides) -> Query:
    fields = dict(
        query_id="q-002",
        text="What is the CEO's home address?",
        scenario_stratum="unanswerable",
        overlap_stratum="low_overlap",
        language="en",
    )
    fields.update(overrides)
    return Query(**fields)


# ---------------------------------------------------------------------------
# Stratum vocabularies
# ---------------------------------------------------------------------------
def test_scenario_strata_covers_the_four_scenarios():
    assert set(SCENARIO_STRATA) == {
        "factoid",
        "cross_doc",
        "version_conflict",
        "unanswerable",
    }


def test_overlap_strata_and_languages_are_non_empty():
    assert OVERLAP_STRATA
    assert LANGUAGES == ("en", "zh")


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
def test_round_trip_preserves_every_field(tmp_path):
    queries = [_factoid_query(), _unanswerable_query()]
    path = tmp_path / "queries.yaml"

    dump_queries(queries, path)
    loaded = load_queries(path)

    assert loaded == queries


def test_dict_round_trip_is_the_identity():
    query = _factoid_query()
    assert query_from_dict(query_to_dict(query)) == query


def test_load_queries_on_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_queries(path) == []


# ---------------------------------------------------------------------------
# Validation — unlabeled / under-specified queries are rejected
# ---------------------------------------------------------------------------
def test_missing_scenario_stratum_field_is_rejected():
    entry = query_to_dict(_factoid_query())
    del entry["scenario_stratum"]
    with pytest.raises(ValueError, match="scenario_stratum"):
        query_from_dict(entry)


def test_missing_overlap_stratum_field_is_rejected():
    entry = query_to_dict(_factoid_query())
    del entry["overlap_stratum"]
    with pytest.raises(ValueError, match="overlap_stratum"):
        query_from_dict(entry)


def test_missing_language_field_is_rejected():
    entry = query_to_dict(_factoid_query())
    del entry["language"]
    with pytest.raises(ValueError, match="language"):
        query_from_dict(entry)


def test_invalid_scenario_stratum_value_is_rejected():
    with pytest.raises(ValueError, match="scenario_stratum"):
        _factoid_query(scenario_stratum="made_up_scenario")


def test_invalid_overlap_stratum_value_is_rejected():
    with pytest.raises(ValueError, match="overlap_stratum"):
        _factoid_query(overlap_stratum="medium_overlap")


def test_invalid_language_value_is_rejected():
    with pytest.raises(ValueError, match="language"):
        _factoid_query(language="fr")


def test_answerable_query_without_gold_section_ids_is_rejected():
    with pytest.raises(ValueError, match="gold_section_id"):
        _factoid_query(gold_section_ids=[])


def test_gold_section_ids_without_key_tokens_is_rejected():
    with pytest.raises(ValueError, match="key_tokens"):
        _factoid_query(key_tokens=[])


def test_unanswerable_query_may_have_empty_gold_and_key_tokens():
    query = _unanswerable_query()
    assert query.gold_section_ids == []
    assert query.key_tokens == []


def test_load_queries_reports_the_offending_query_id(tmp_path):
    """A malformed entry buried in a batch file names its query_id in the error."""
    good = query_to_dict(_factoid_query(query_id="q-good"))
    bad = query_to_dict(_factoid_query(query_id="q-bad"))
    bad["scenario_stratum"] = "not_a_real_scenario"
    path = tmp_path / "queries.yaml"
    path.write_text(yaml.safe_dump({"queries": [good, bad]}), encoding="utf-8")

    with pytest.raises(ValueError, match="q-bad"):
        load_queries(path)
