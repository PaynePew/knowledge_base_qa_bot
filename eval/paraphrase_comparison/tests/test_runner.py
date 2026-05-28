"""In-process comparison runner tests (external behaviour only, CODING_STANDARD §0.2).

Asserts that both Retrieval Stacks are driven in one process (no HTTP), that the
report carries a per-type row for ALL seven Paraphrase Types (Stack A vs Stack B
hit_rate@3), and that running the comparison never touches production ``wiki/`` /
``docs/`` / ``.kb/``.
"""

from __future__ import annotations

from pathlib import Path

import markdown_kb.app.indexer as mk_indexer
import vector_rag.app.indexer as vr_indexer
from eval.paraphrase_comparison import stacks
from eval.paraphrase_comparison.loader import load_paraphrases
from eval.paraphrase_comparison.models import PARAPHRASE_TYPES
from eval.paraphrase_comparison.runner import (
    render_report,
    run_comparison,
    score_stack,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROD_KB = REPO_ROOT / ".kb"
PROD_WIKI_INDEX = REPO_ROOT / "wiki" / "index.md"
PROD_LOG = REPO_ROOT / "wiki" / "log.md"


def test_queries_yaml_holds_full_seven_type_set():
    paraphrases = load_paraphrases()
    # Full Slice 2 set: ~39-54 Paraphrases across all seven types.
    assert 39 <= len(paraphrases) <= 54
    assert {p.paraphrase_type for p in paraphrases} == set(PARAPHRASE_TYPES)
    ids = [p.paraphrase_id for p in paraphrases]
    assert len(ids) == len(set(ids)), "paraphrase_id must be unique"
    for p in paraphrases:
        assert p.gold_docs_section_id.count("#") == 1
        assert p.key_tokens_docs
        assert p.key_tokens_wiki


def test_specificity_narrowing_targets_multi_sub_fact_sections():
    # specificity_narrowing may only target multi-sub-fact Gold Sections; cross-
    # check each such Paraphrase's gold against the committed inventory's flag.
    from eval.paraphrase_comparison.generation.sampling import load_gold_sections

    multi = {s.section_id for s in load_gold_sections() if s.multi_sub_fact}
    spec = [
        p for p in load_paraphrases() if p.paraphrase_type == "specificity_narrowing"
    ]
    assert spec, "expected specificity_narrowing Paraphrases in the set"
    for p in spec:
        assert p.gold_docs_section_id in multi


def test_run_comparison_produces_a_row_per_type(tmp_path, fake_vector_index):
    report_path = tmp_path / "report.md"
    stack_a, stack_b = run_comparison(report_path=report_path, embedding_mode="fake")

    for ptype in PARAPHRASE_TYPES:
        assert ptype in stack_a.by_type
        assert ptype in stack_b.by_type
        # MRR is aggregated alongside hit_rate@k for every type.
        assert ptype in stack_a.mrr_by_type
        assert ptype in stack_b.mrr_by_type

    report = report_path.read_text(encoding="utf-8")
    # Full per-type table columns (PRD #100): hit_rate@3 + MRR for both Stacks.
    assert "hit_rate@3 (A)" in report
    assert "hit_rate@3 (B)" in report
    assert "MRR (A)" in report
    assert "MRR (B)" in report
    assert "Δ (B−A)" in report
    for ptype in PARAPHRASE_TYPES:
        assert f"| {ptype} |" in report


def test_run_comparison_separates_core_from_probes(tmp_path, fake_vector_index):
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")

    # Core and Structural-probe types live in distinct sections; the only
    # aggregate is a caveated Core macro-average (no naive cross-type aggregate).
    assert "## Core Comparison" in report
    assert "## Structural Probes" in report
    assert "Core macro-average" in report
    assert "Caveat" in report
    # All six honest disclosures + the offline-data disclosure are present.
    assert "## Limitations" in report
    assert "OFFLINE tracer data" in report
    assert "## Appendix — Interview Talking Points" in report


def test_run_comparison_writes_chart_pngs(tmp_path, fake_vector_index):
    report_path = tmp_path / "report.md"
    charts_dir = tmp_path / "charts"
    run_comparison(report_path=report_path, embedding_mode="fake")

    pngs = list(charts_dir.glob("*.png"))
    assert pngs, "expected chart PNGs written to the report's charts/ sibling"
    names = {p.name for p in pngs}
    assert any(n.startswith("core_") for n in names)
    assert any(n.startswith("probes_") for n in names)


def test_both_stacks_run_in_process_and_return_resolved_docs_ids(fake_vector_index):
    # Build both indexes against the eval fixtures and retrieve a query through
    # each Stack's callable directly — proves the in-process (no-HTTP) wiring.
    stacks.index_stack_a()
    stacks.index_stack_b()

    para = next(p for p in load_paraphrases() if p.paraphrase_id == "synonym_swap-005")
    a_items = stacks.stack_a_retrieval(para.text, k=3)
    b_items = stacks.stack_b_retrieval(para.text, k=3)

    assert a_items and b_items
    # Both Stacks normalise hits to docs Gold Section ids ('<file>.md#<slug>').
    for item in (*a_items, *b_items):
        assert item.source_section_id.split("#")[0].endswith(".md")


def test_stack_a_resolves_wiki_hit_to_docs_gold_section(fake_vector_index):
    stacks.index_stack_a()
    items = stacks.stack_a_retrieval("forgot my login passphrase reset", k=3)
    # The wiki page 'password-reset' synthesises the docs section
    # 'account_management.md#password-reset'; Stack A must report the docs id.
    assert any(
        it.source_section_id == "account_management.md#password-reset" for it in items
    )


def test_scored_hit_rate_is_a_fraction(fake_vector_index):
    stacks.index_stack_a()
    paraphrases = load_paraphrases()
    scores = score_stack("Stack A", paraphrases, stacks.stack_a_retrieval, k=3)
    rate = scores.by_type["synonym_swap"]
    assert 0.0 <= rate <= 1.0


def test_render_report_records_embedding_mode(fake_vector_index):
    paraphrases = load_paraphrases()
    stacks.index_stack_a()
    a = score_stack("Stack A", paraphrases, stacks.stack_a_retrieval, k=3)
    b = score_stack("Stack B", paraphrases, lambda q, k: [], k=3)
    report = render_report(a, b, embedding_mode="fake")
    assert "embedding mode**: **fake**" in report
    # The offline banner must warn a reader these are not the real experiment.
    assert "OFFLINE TRACER NUMBERS" in report


def test_running_comparison_does_not_touch_production_paths(
    tmp_path, fake_vector_index
):
    kb_before = sorted(p.name for p in PROD_KB.glob("*")) if PROD_KB.exists() else []
    wiki_index_before = (
        PROD_WIKI_INDEX.read_text() if PROD_WIKI_INDEX.exists() else None
    )
    log_before = PROD_LOG.read_text() if PROD_LOG.exists() else None

    run_comparison(report_path=tmp_path / "report.md", embedding_mode="fake")

    kb_after = sorted(p.name for p in PROD_KB.glob("*")) if PROD_KB.exists() else []
    wiki_index_after = PROD_WIKI_INDEX.read_text() if PROD_WIKI_INDEX.exists() else None
    log_after = PROD_LOG.read_text() if PROD_LOG.exists() else None

    assert kb_after == kb_before
    assert wiki_index_after == wiki_index_before
    assert log_after == log_before


def test_production_isolation_repoints_source_dirs_to_fixtures(fake_vector_index):
    stacks.index_stack_a()
    stacks.index_stack_b()
    # Both Stacks' corpus roots must point at the eval fixtures, never the
    # production wiki/ or docs/ directories.
    for d in mk_indexer.SOURCE_DIRS:
        assert "paraphrase_comparison" in str(d)
    assert "paraphrase_comparison" in str(vr_indexer.DOCS_DIR)
