"""Query-set artifact assembly tests — external behaviour only
(CODING_STANDARD §0.2).

Covers issue #672 AC 1: "deviations (if any) stated in the artifact header,
not silent."
"""

from __future__ import annotations

import pytest
import yaml

from eval.corpus_v3.generation.artifact import (
    StratumCount,
    build_deviations,
    build_metadata,
    render_query_artifact,
)
from eval.corpus_v3.query_schema import Query, load_queries


def test_no_deviations_when_every_cell_hits_its_target():
    counts = [StratumCount("factoid", "en", 909, 909, 12)]
    assert build_deviations(counts) == []


def test_deviation_reported_when_actual_falls_short_of_target():
    counts = [StratumCount("factoid", "en", 909, 900, 40)]
    deviations = build_deviations(counts)
    assert len(deviations) == 1
    assert "short by 9" in deviations[0]


def test_deviation_reported_when_actual_exceeds_target():
    counts = [StratumCount("unanswerable", "zh", 200, 205, 3)]
    deviations = build_deviations(counts)
    assert "over by 5" in deviations[0]


def test_build_metadata_records_both_generating_families():
    md = build_metadata(
        counts=[StratumCount("factoid", "en", 10, 10, 0)],
        family_a_model="gpt-4o-mini",
        family_b_source="human",
        embedding_family="text-embedding-3-small",
        generated_at="2026-07-24T00:00:00Z",
        cost_usd=0.42,
        prompt_template_version=1,
    )
    assert md["generator_family_a"] == "gpt-4o-mini"
    assert md["generator_family_b"] == "human"
    assert md["cost_usd"] == 0.42
    assert md["deviations"] == []


def test_build_metadata_rejects_family_b_matching_family_a():
    with pytest.raises(ValueError):
        build_metadata(
            counts=[],
            family_a_model="gpt-4o-mini",
            family_b_source="gpt-4o-mini",
            embedding_family="text-embedding-3-small",
            generated_at="2026-07-24T00:00:00Z",
            cost_usd=None,
            prompt_template_version=1,
        )


def test_build_metadata_rejects_blank_family_b():
    with pytest.raises(ValueError):
        build_metadata(
            counts=[],
            family_a_model="gpt-4o-mini",
            family_b_source="   ",
            embedding_family="text-embedding-3-small",
            generated_at="2026-07-24T00:00:00Z",
            cost_usd=None,
            prompt_template_version=1,
        )


def test_render_query_artifact_round_trips_through_load_queries(tmp_path):
    query = Query(
        query_id="q-001",
        text="How long is the return window?",
        scenario_stratum="factoid",
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=["returns_policy.md#return-window"],
        key_tokens=["return", "window"],
        generating_family="gpt-4o-mini",
    )
    md = build_metadata(
        counts=[StratumCount("factoid", "en", 1, 1, 0)],
        family_a_model="gpt-4o-mini",
        family_b_source="human",
        embedding_family="text-embedding-3-small",
        generated_at="2026-07-24T00:00:00Z",
        cost_usd=0.01,
        prompt_template_version=1,
    )
    text = render_query_artifact([query], metadata=md)
    path = tmp_path / "queries.yaml"
    path.write_text(text, encoding="utf-8")

    loaded = load_queries(path)
    assert loaded == [query]

    raw = yaml.safe_load(text)
    assert raw["metadata"]["generator_family_a"] == "gpt-4o-mini"


def test_render_query_artifact_raises_on_duplicate_query_ids():
    """Downstream joins key on query_id and load_queries never checks
    uniqueness, so a duplicate must fail loudly at write time."""
    query = Query(
        query_id="factoid-en-0000",
        text="How long is the return window?",
        scenario_stratum="factoid",
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=["returns_policy.md#return-window"],
        key_tokens=["return", "window"],
        generating_family="gpt-4o-mini",
    )
    md = build_metadata(
        counts=[StratumCount("factoid", "en", 2, 2, 0)],
        family_a_model="gpt-4o-mini",
        family_b_source="human",
        embedding_family="text-embedding-3-small",
        generated_at="2026-07-24T00:00:00Z",
        cost_usd=0.01,
        prompt_template_version=1,
    )
    with pytest.raises(ValueError, match="duplicate query_ids"):
        render_query_artifact([query, query], metadata=md)
