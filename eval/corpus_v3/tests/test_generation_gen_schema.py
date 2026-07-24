"""Query draft stitching + end-to-end generation-pipeline tests.

Covers issue #660 AC-2's "generated query files validate against the
stratified schema with all three labels populated": a ``QueryDraft`` (the
LLM- or human-supplied portion) stitched via ``to_query``, with its overlap
stratum computed by ``generation.overlap``, produces a ``Query`` that
round-trips through ``query_schema.dump_queries`` / ``load_queries`` exactly
like a hand-written one (CODING_STANDARD §0.2 — external behaviour only).
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.generation.gen_schema import QueryDraft, to_query
from eval.corpus_v3.generation.overlap import classify_overlap_stratum
from eval.corpus_v3.generation.qc import check_generated_query
from eval.corpus_v3.query_schema import dump_queries, load_queries, query_to_dict


def test_to_query_stitches_generator_owned_fields_onto_a_draft():
    draft = QueryDraft(
        text="How long is the return window?",
        key_tokens=["return", "window", "days"],
        generation_notes="targets the return-window sub-fact",
    )
    query = to_query(
        draft,
        query_id="gen-001",
        scenario_stratum="factoid",
        language="en",
        overlap_stratum="high_overlap",
        gold_section_ids=["returns_policy.md#return-window"],
        generating_family="gpt-4o-mini",
    )
    assert query.query_id == "gen-001"
    assert query.text == draft.text
    assert query.scenario_stratum == "factoid"
    assert query.overlap_stratum == "high_overlap"
    assert query.language == "en"
    assert query.gold_section_ids == ["returns_policy.md#return-window"]
    assert query.key_tokens == draft.key_tokens
    assert query.generating_family == "gpt-4o-mini"
    assert query.generation_notes == draft.generation_notes


def test_to_query_propagates_query_validation_failures():
    # An answerable scenario with no gold_section_ids is invalid — to_query
    # does not duplicate the check, it just doesn't swallow the raise.
    draft = QueryDraft(text="How long is the return window?", key_tokens=["window"])
    with pytest.raises(ValueError, match="gold_section_id"):
        to_query(
            draft,
            query_id="gen-002",
            scenario_stratum="factoid",
            language="en",
            overlap_stratum="low_overlap",
            gold_section_ids=[],
            generating_family="gpt-4o-mini",
        )


# ---------------------------------------------------------------------------
# End-to-end: draft -> computed overlap -> stitched Query -> QC gate ->
# dump/load round trip, i.e. a "generated query file" as the real pipeline
# would produce one.
# ---------------------------------------------------------------------------
def test_generated_queries_validate_against_the_stratified_schema(tmp_path):
    gold_section_text = (
        "Return window: items may be returned within 30 days of delivery "
        "for a full refund, provided the original packaging is intact."
    )

    drafts = [
        (
            QueryDraft(
                text="How long is the return window?",
                key_tokens=["return", "window", "days"],
            ),
            "factoid",
            "en",
            "gpt-4o-mini",
        ),
        (
            QueryDraft(
                text="退貨期限是多久？",
                key_tokens=["退貨", "期限"],
            ),
            "factoid",
            "zh",
            "human",
        ),
        (
            QueryDraft(text="What is the CEO's home phone number?"),
            "unanswerable",
            "en",
            "gpt-4o-mini",
        ),
    ]

    queries = []
    for idx, (draft, scenario, language, family) in enumerate(drafts):
        gold_ids = [] if scenario == "unanswerable" else ["returns.md#return-window"]
        overlap = classify_overlap_stratum(draft.text, [gold_section_text])
        query = to_query(
            draft,
            query_id=f"gen-{idx:03d}",
            scenario_stratum=scenario,
            language=language,
            overlap_stratum=overlap,
            gold_section_ids=gold_ids,
            generating_family=family,
        )
        verdict = check_generated_query(query)
        assert verdict.rejected is False, verdict.reasons
        queries.append(query)

    # Every query carries all three mandatory labels populated (non-empty).
    for query in queries:
        entry = query_to_dict(query)
        assert entry["scenario_stratum"]
        assert entry["overlap_stratum"]
        assert entry["language"]
        assert entry["generating_family"]

    # And the file round-trips through the stratified schema's own loader —
    # the acceptance bar issue #659's schema already enforces.
    path = tmp_path / "generated_queries.yaml"
    dump_queries(queries, path)
    loaded = load_queries(path)
    assert loaded == queries
