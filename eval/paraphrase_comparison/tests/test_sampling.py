"""sha256 Gold Section sampling tests (external behaviour only, CODING_STANDARD §0.2).

The sampler's contract is *determinism* and the specificity_narrowing pool rule —
both fully deterministic and offline, so they are asserted directly. The actual
LLM Paraphrase content the sampled sections feed into is never asserted (§6.2).
"""

from __future__ import annotations

from eval.paraphrase_comparison.generation.sampling import (
    GoldSection,
    load_gold_sections,
    sample_sections,
    sha256_order,
)

_FIXTURE = [
    GoldSection("a.md#one", "one", multi_sub_fact=True),
    GoldSection("b.md#two", "two", multi_sub_fact=False),
    GoldSection("c.md#three", "three", multi_sub_fact=True),
    GoldSection("d.md#four", "four", multi_sub_fact=False),
    GoldSection("e.md#five", "five", multi_sub_fact=True),
]


def test_sha256_order_is_deterministic_across_calls():
    first = [s.section_id for s in sha256_order(_FIXTURE, "synonym_swap")]
    second = [s.section_id for s in sha256_order(list(reversed(_FIXTURE)), "synonym_swap")]
    # Same seed + same membership => identical order regardless of input order.
    assert first == second


def test_sha256_order_is_a_permutation():
    ordered = sha256_order(_FIXTURE, "word_reorder")
    assert {s.section_id for s in ordered} == {s.section_id for s in _FIXTURE}
    assert len(ordered) == len(_FIXTURE)


def test_different_seeds_can_produce_different_orders():
    a = [s.section_id for s in sha256_order(_FIXTURE, "synonym_swap")]
    b = [s.section_id for s in sha256_order(_FIXTURE, "verbosity_expansion")]
    # Different Paraphrase Types seed independent orderings (cross-type reuse is
    # allowed precisely because each type samples on its own seed).
    assert a != b


def test_order_does_not_depend_on_pythonhashseed():
    # The whole point of sha256 over Python hash(): a hard-coded expected order
    # that must hold no matter the per-process hash salt. If this ever flakes,
    # someone reintroduced hash()-based ordering.
    expected = sha256_order(_FIXTURE, "fixed-seed")
    again = sha256_order(_FIXTURE, "fixed-seed")
    assert [s.section_id for s in expected] == [s.section_id for s in again]


def test_specificity_narrowing_pool_is_multi_sub_fact_only():
    drawn = sample_sections(
        _FIXTURE, seed="specificity_narrowing", count=5, multi_sub_fact_only=True
    )
    assert drawn, "expected a non-empty multi-sub-fact pool"
    assert all(s.multi_sub_fact for s in drawn)
    # Only the three multi-sub-fact fixtures are eligible.
    assert len(drawn) == 3


def test_sample_sections_caps_at_count_and_never_overdraws():
    assert len(sample_sections(_FIXTURE, "synonym_swap", count=2)) == 2
    # count larger than the pool returns the whole pool, not an error.
    assert len(sample_sections(_FIXTURE, "synonym_swap", count=99)) == len(_FIXTURE)


def test_load_gold_sections_reads_committed_inventory():
    gold = load_gold_sections()
    # The committed inventory is the corpus's concept Gold Sections.
    assert len(gold) >= 39
    assert all(s.section_id.count("#") == 1 for s in gold)
    assert any(s.multi_sub_fact for s in gold)
    # Every entry's concept_slug is the heading-slug half of its section id
    # (1:1 concept page convention).
    for s in gold:
        assert s.section_id.split("#")[1] == s.concept_slug
