"""Adversarial corpus build tests — external behaviour only (CODING_STANDARD
§0.2). Issue #661, PRD #654 user stories 10-12 / 18.

Covers: rebuildability (regenerating the fixtures byte-for-byte reproduces
the committed files, no writes outside this package's own ``corpus/`` /
``wiki/concepts`` dirs), per-class instance-count floor, and that each
adversarial class resolves through the SAME gold-mapping code path
(``eval.corpus_v3.gold``, issue #658) the way the class is meant to:
redundancy dedups to one wiki id covering both raw ids, contradiction covers
both raw ids without picking a winner, version_evolution gives ONLY the
newest raw id gold coverage (an older version is a defined non-hit). Also
covers the build-cost ledger AC (a real, non-fabricated zero for this
offline/deterministic construction method).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.cost_ledger.ledger import CostLedger
from eval.corpus_v3 import build_corpus, gold
from eval.corpus_v3.build_corpus import ADVERSARIAL_GROUPS, MIN_INSTANCES_PER_CLASS

WIKI_DIR = Path(__file__).resolve().parents[1] / "wiki"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"


# ---------------------------------------------------------------------------
# Rebuildability (AC 1)
# ---------------------------------------------------------------------------
def test_regenerating_reproduces_the_committed_fixtures_byte_for_byte(
    tmp_path, monkeypatch
):
    """Re-running the build script over a fresh tmp dir produces output
    byte-identical to the committed fixtures — fixed timestamp, real content
    hashes, no wall-clock or random input, so "rebuildable" is a real claim,
    not aspirational."""
    monkeypatch.setattr(build_corpus, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(
        build_corpus, "WIKI_CONCEPTS_DIR", tmp_path / "wiki" / "concepts"
    )

    build_corpus.write_corpus_fixtures()

    for group in ADVERSARIAL_GROUPS:
        for section in group.sections:
            committed = (CORPUS_DIR / section.basename).read_text(encoding="utf-8")
            regenerated = (tmp_path / "corpus" / section.basename).read_text(
                encoding="utf-8"
            )
            assert regenerated == committed, (
                f"{section.basename} drifted from regeneration"
            )
        wiki_name = f"{group.group_id}.md"
        committed_wiki = (WIKI_DIR / "concepts" / wiki_name).read_text(encoding="utf-8")
        regenerated_wiki = (tmp_path / "wiki" / "concepts" / wiki_name).read_text(
            encoding="utf-8"
        )
        assert regenerated_wiki == committed_wiki, (
            f"{wiki_name} drifted from regeneration"
        )


def test_write_corpus_fixtures_never_writes_outside_its_own_fixture_dirs(
    tmp_path, monkeypatch
):
    """Production isolation: redirecting CORPUS_DIR/WIKI_CONCEPTS_DIR to tmp
    and regenerating touches nothing else under tmp_path."""
    corpus_dir = tmp_path / "corpus"
    wiki_dir = tmp_path / "wiki" / "concepts"
    monkeypatch.setattr(build_corpus, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(build_corpus, "WIKI_CONCEPTS_DIR", wiki_dir)

    build_corpus.write_corpus_fixtures()

    written = {p for p in tmp_path.rglob("*") if p.is_file()}
    outside = {
        p for p in written if corpus_dir not in p.parents and wiki_dir not in p.parents
    }
    assert not outside, f"wrote outside the fixture dirs: {outside}"


# ---------------------------------------------------------------------------
# Per-class instance-count floor (AC 2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "adversarial_class", ["redundancy", "contradiction", "version_evolution"]
)
def test_each_adversarial_class_meets_the_minimum_instance_floor(adversarial_class):
    count = sum(
        1 for g in ADVERSARIAL_GROUPS if g.adversarial_class == adversarial_class
    )
    assert count >= MIN_INSTANCES_PER_CLASS, (
        f"{adversarial_class} has only {count} instances, "
        f"below the {MIN_INSTANCES_PER_CLASS} floor (issue #661 AC 2)"
    )


def test_every_group_has_a_globally_unique_group_id():
    ids = [g.group_id for g in ADVERSARIAL_GROUPS]
    assert len(ids) == len(set(ids))


def test_every_raw_basename_is_globally_unique_across_the_corpus():
    """Basename uniqueness (docs/README.md's invariant) applies here too —
    ``markdown_kb`` ids Sections by ``{basename}#{slug}``; a collision would
    make ids ambiguous."""
    basenames = [s.basename for g in ADVERSARIAL_GROUPS for s in g.sections]
    assert len(basenames) == len(set(basenames))


# ---------------------------------------------------------------------------
# Gold-mapping behaviour per adversarial class (issue #658's gold.py, reused
# unmodified — these fixtures are the thing under test, not the code path)
# ---------------------------------------------------------------------------
def test_redundancy_groups_dedup_to_one_wiki_id_covering_both_raw_ids():
    gold_map = gold.build_gold_map(WIKI_DIR)
    groups = [g for g in ADVERSARIAL_GROUPS if g.adversarial_class == "redundancy"]
    assert groups

    for group in groups:
        assert gold_map[group.group_id] == frozenset(group.gold_section_ids)
        for raw_id in group.gold_section_ids:
            resolved = gold.resolve_gold_sections(gold_map, raw_id)
            assert group.group_id in resolved


def test_contradiction_groups_cover_both_conflicting_raw_ids_without_a_winner():
    gold_map = gold.build_gold_map(WIKI_DIR)
    groups = [g for g in ADVERSARIAL_GROUPS if g.adversarial_class == "contradiction"]
    assert groups

    for group in groups:
        assert gold_map[group.group_id] == frozenset(group.gold_section_ids)
        assert len(group.gold_section_ids) == 2, (
            "a contradiction pair has exactly two sides"
        )
        # Neither raw id is preferred: both resolve through the same wiki id.
        for raw_id in group.gold_section_ids:
            resolved = gold.resolve_gold_sections(gold_map, raw_id)
            assert group.group_id in resolved


def test_version_evolution_groups_give_gold_coverage_to_only_the_newest_id():
    gold_map = gold.build_gold_map(WIKI_DIR)
    groups = [
        g for g in ADVERSARIAL_GROUPS if g.adversarial_class == "version_evolution"
    ]
    assert groups

    for group in groups:
        assert len(group.gold_section_ids) == 1, (
            "only the newest version is the defined gold answer"
        )
        newest_id = group.gold_section_ids[0]
        older_ids = [s.source_id for s in group.sections if s.source_id != newest_id]
        assert older_ids, (
            "a version_evolution group must have at least one superseded version"
        )

        assert gold_map[group.group_id] == frozenset({newest_id})
        resolved_newest = gold.resolve_gold_sections(gold_map, newest_id)
        assert group.group_id in resolved_newest

        # An older version is a defined NON-hit: it resolves to only itself,
        # with no wiki coverage — the version-conflict scenario's whole point.
        for older_id in older_ids:
            resolved_older = gold.resolve_gold_sections(gold_map, older_id)
            assert resolved_older == frozenset({older_id})


# ---------------------------------------------------------------------------
# Build-cost ledger (AC 3)
# ---------------------------------------------------------------------------
def test_write_corpus_fixtures_records_one_real_zero_build_entry_per_group(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(build_corpus, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(
        build_corpus, "WIKI_CONCEPTS_DIR", tmp_path / "wiki" / "concepts"
    )

    ledger = build_corpus.write_corpus_fixtures(CostLedger())

    totals = ledger.totals(stack="wiki_curation", phase="build")
    assert totals.calls == len(ADVERSARIAL_GROUPS)
    assert totals.total_tokens == 0
    assert (
        totals.usd is None
    )  # unpriced offline-deterministic model, not a fabricated 0.0


def test_write_build_cost_report_is_committed_and_matches_regeneration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(build_corpus, "CORPUS_DIR", tmp_path / "corpus")
    monkeypatch.setattr(
        build_corpus, "WIKI_CONCEPTS_DIR", tmp_path / "wiki" / "concepts"
    )
    ledger = build_corpus.write_corpus_fixtures(CostLedger())
    report_path = tmp_path / "BUILD_COST.offline-tracer.md"

    build_corpus.write_build_cost_report(ledger, report_path)

    committed = build_corpus.BUILD_COST_REPORT_PATH.read_text(encoding="utf-8")
    assert report_path.read_text(encoding="utf-8") == committed
    # §6.6: an offline-tracer artifact carries its trust level in both the
    # filename and a loud top-of-file header.
    assert committed.startswith("⚠️ PLACEHOLDER")
    assert build_corpus.BUILD_COST_REPORT_PATH.name.endswith(".offline-tracer.md")
