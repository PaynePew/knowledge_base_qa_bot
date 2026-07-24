"""Generation-target derivation tests — external behaviour only
(CODING_STANDARD §0.2).

Covers issue #672: deriving per-scenario-stratum generation targets from the
committed adversarial corpus, and deterministically sampling/cycling them.
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.build_corpus import AdversarialGroup, RawSection
from eval.corpus_v3.generation.targets import (
    derive_generation_targets,
    sample_targets,
    sha256_order,
)

_REDUNDANCY = AdversarialGroup(
    adversarial_class="redundancy",
    group_id="dup-group",
    wiki_title="Dup Title",
    wiki_body="wiki body",
    sections=[
        RawSection(basename="a.md", heading="H1", body="Body A text."),
        RawSection(basename="b.md", heading="H1", body="Body B text."),
    ],
    gold_section_ids=["a.md#h1", "b.md#h1"],
)

_CONTRADICTION = AdversarialGroup(
    adversarial_class="contradiction",
    group_id="conflict-group",
    wiki_title="Conflict Title",
    wiki_body="wiki body",
    sections=[
        RawSection(basename="c.md", heading="H2", body="Conflict C text."),
        RawSection(basename="d.md", heading="H2", body="Conflict D text."),
    ],
    gold_section_ids=["c.md#h2", "d.md#h2"],
)

_VERSION = AdversarialGroup(
    adversarial_class="version_evolution",
    group_id="version-group",
    wiki_title="Version Title",
    wiki_body="wiki body",
    sections=[
        RawSection(basename="v1.md", heading="H3", body="Old value."),
        RawSection(basename="v2.md", heading="H3", body="New value."),
    ],
    gold_section_ids=["v2.md#h3"],
)

_ALL_GROUPS = [_REDUNDANCY, _CONTRADICTION, _VERSION]


def test_redundancy_group_yields_one_factoid_target_per_gold_section():
    buckets = derive_generation_targets(_ALL_GROUPS)
    factoid_ids = {(t.group_id, t.gold_section_ids[0]) for t in buckets["factoid"]}
    assert ("dup-group", "a.md#h1") in factoid_ids
    assert ("dup-group", "b.md#h1") in factoid_ids


def test_contradiction_group_also_yields_factoid_targets():
    buckets = derive_generation_targets(_ALL_GROUPS)
    factoid_groups = {t.group_id for t in buckets["factoid"]}
    assert "conflict-group" in factoid_groups


def test_redundancy_and_contradiction_groups_yield_one_cross_doc_target_each():
    buckets = derive_generation_targets(_ALL_GROUPS)
    cross_doc_groups = {t.group_id for t in buckets["cross_doc"]}
    assert cross_doc_groups == {"dup-group", "conflict-group"}
    target = next(t for t in buckets["cross_doc"] if t.group_id == "dup-group")
    assert set(target.gold_section_ids) == {"a.md#h1", "b.md#h1"}
    assert "Body A text." in target.reference_text
    assert "Body B text." in target.reference_text


def test_version_evolution_group_yields_only_a_version_conflict_target():
    buckets = derive_generation_targets(_ALL_GROUPS)
    assert len(buckets["version_conflict"]) == 1
    target = buckets["version_conflict"][0]
    assert target.group_id == "version-group"
    assert target.gold_section_ids == ["v2.md#h3"]
    version_groups_elsewhere = [
        t
        for stratum in ("factoid", "cross_doc")
        for t in buckets[stratum]
        if t.group_id == "version-group"
    ]
    assert version_groups_elsewhere == []


def test_every_group_yields_exactly_one_unanswerable_target():
    buckets = derive_generation_targets(_ALL_GROUPS)
    assert len(buckets["unanswerable"]) == len(_ALL_GROUPS)
    for target in buckets["unanswerable"]:
        assert target.gold_section_ids == []
        assert target.reference_ids  # a distractor is always present


def test_unanswerable_target_prefers_a_non_gold_section_as_distractor():
    buckets = derive_generation_targets(_ALL_GROUPS)
    version_unanswerable = next(
        t for t in buckets["unanswerable"] if t.group_id == "version-group"
    )
    # v1.md#h3 is NOT gold (only v2 is) -- it is the natural near-miss distractor.
    assert version_unanswerable.reference_ids == ["v1.md#h3"]


def test_all_four_strata_are_always_present_even_when_empty():
    buckets = derive_generation_targets([])
    assert set(buckets.keys()) == {
        "factoid",
        "cross_doc",
        "version_conflict",
        "unanswerable",
    }
    assert all(v == [] for v in buckets.values())


def test_sha256_order_is_deterministic_across_calls():
    buckets = derive_generation_targets(_ALL_GROUPS)
    first = sha256_order(buckets["factoid"], seed="factoid:en")
    second = sha256_order(buckets["factoid"], seed="factoid:en")
    assert [t.group_id for t in first] == [t.group_id for t in second]


def test_sha256_order_differs_by_seed():
    buckets = derive_generation_targets(_ALL_GROUPS)
    a = sha256_order(buckets["factoid"], seed="seed-a")
    b = sha256_order(buckets["factoid"], seed="seed-b")
    # Not a hard guarantee for every possible seed pair, but true for this
    # fixture (regression check that seed actually participates in the key).
    assert [t.group_id for t in a] != [t.group_id for t in b] or len(a) <= 1


def test_sample_targets_cycles_when_count_exceeds_pool_size():
    buckets = derive_generation_targets(_ALL_GROUPS)
    pool = buckets["version_conflict"]  # size 1
    sampled = sample_targets(pool, seed="version_conflict:en", count=5)
    assert len(sampled) == 5
    assert all(t.group_id == "version-group" for t in sampled)


def test_sample_targets_is_reproducible():
    buckets = derive_generation_targets(_ALL_GROUPS)
    pool = buckets["factoid"]
    first = sample_targets(pool, seed="factoid:en", count=7)
    second = sample_targets(pool, seed="factoid:en", count=7)
    assert [t.group_id for t in first] == [t.group_id for t in second]


def test_sample_targets_raises_on_negative_count():
    buckets = derive_generation_targets(_ALL_GROUPS)
    with pytest.raises(ValueError):
        sample_targets(buckets["factoid"], seed="x", count=-1)


def test_sample_targets_raises_on_empty_pool_with_positive_count():
    with pytest.raises(ValueError):
        sample_targets([], seed="x", count=3)


def test_sample_targets_returns_empty_list_for_zero_count():
    buckets = derive_generation_targets(_ALL_GROUPS)
    assert sample_targets(buckets["factoid"], seed="x", count=0) == []
