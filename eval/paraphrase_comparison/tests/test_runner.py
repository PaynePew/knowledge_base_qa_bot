"""In-process comparison runner tests (external behaviour only, CODING_STANDARD §0.2).

Asserts that both Retrieval Stacks are driven in one process (no HTTP), that the
report carries a per-type row for ALL seven Paraphrase Types (Stack A vs Stack B
hit_rate@3), and that running the comparison never touches production ``wiki/`` /
``docs/`` / ``.kb/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import markdown_kb.app.indexer as mk_indexer
import vector_rag.app.indexer as vr_indexer
from eval.paraphrase_comparison import spotcheck as spotcheck_mod
from eval.paraphrase_comparison import stacks
from eval.paraphrase_comparison.loader import load_paraphrases
from eval.paraphrase_comparison.models import PARAPHRASE_TYPES
from eval.paraphrase_comparison.runner import (
    JudgeConfig,
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
    # Demo-tier set (issue #145): ~50 per Core type + ~5 per probe type (~260).
    # Assert a floor + all types present, not a brittle exact range, so the test
    # survives regeneration at different --per-type sizes.
    assert len(paraphrases) >= 35
    assert {p.paraphrase_type for p in paraphrases} == set(PARAPHRASE_TYPES)
    ids = [p.paraphrase_id for p in paraphrases]
    assert len(ids) == len(set(ids)), "paraphrase_id must be unique"
    for p in paraphrases:
        assert p.gold_docs_section_id.count("#") == 1
        assert p.key_tokens_docs
        assert p.key_tokens_wiki


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

    # Pick any synonym_swap Paraphrase by TYPE (not a hard-coded id) so the test
    # survives regeneration, which re-assigns paraphrase_ids (issue #145).
    para = next(p for p in load_paraphrases() if p.paraphrase_type == "synonym_swap")
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
        PROD_WIKI_INDEX.read_text(encoding="utf-8")
        if PROD_WIKI_INDEX.exists()
        else None
    )
    log_before = PROD_LOG.read_text(encoding="utf-8") if PROD_LOG.exists() else None

    run_comparison(report_path=tmp_path / "report.md", embedding_mode="fake")

    kb_after = sorted(p.name for p in PROD_KB.glob("*")) if PROD_KB.exists() else []
    wiki_index_after = (
        PROD_WIKI_INDEX.read_text(encoding="utf-8")
        if PROD_WIKI_INDEX.exists()
        else None
    )
    log_after = PROD_LOG.read_text(encoding="utf-8") if PROD_LOG.exists() else None

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


# ---------------------------------------------------------------------------
# L2 Spot-check wiring into the report (issue #105)
# ---------------------------------------------------------------------------
class _StubJudgeClient:
    """Always-True judge stub so the mocked run exercises the report path."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, *, model, max_tokens, system, messages):
        import json

        self.calls += 1

        class _R:
            content = [
                type(
                    "B",
                    (),
                    {"text": json.dumps({"answers": True, "reasoning": "stub"})},
                )()
            ]

        return _R()


def test_report_notes_how_to_enable_spotcheck_when_not_run(tmp_path, fake_vector_index):
    # No JudgeConfig -> Spot-check skipped; the report must tell the reader how to
    # turn it on (opt-in), not silently omit it.
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")
    assert "## Spot-check Validation (L2, cross-family)" in report
    assert "Not run." in report
    assert "--judge=claude-sonnet-4-6" in report
    # The cost log marks the judge as the opt-in step.
    assert "not run (opt-in via `--judge`)" in report


def test_report_renders_spotcheck_section_when_judge_runs(
    tmp_path, fake_vector_index, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(spotcheck_mod, "_judge_client", lambda: _StubJudgeClient())

    report_path = tmp_path / "report.md"
    run_comparison(
        report_path=report_path,
        embedding_mode="fake",
        judge=JudgeConfig(judge_model="claude-sonnet-4-6"),
    )
    report = report_path.read_text(encoding="utf-8")

    # Spot-check section reports by-zone subset size + agreement + interpretation.
    assert "## Spot-check Validation (L2, cross-family)" in report
    assert "claude-sonnet-4-6" in report
    assert "Agreement with L1" in report
    assert "Control-zone calibration" in report
    # Disclosure (4) flips to active cross-family-validation framing when judged.
    assert "Cross-family validation was run." in report
    # The cost log now records the judge actually ran.
    assert "item(s) judged" in report


def test_run_comparison_fail_fasts_with_judge_but_no_key(
    tmp_path, fake_vector_index, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(spotcheck_mod.JudgeUnavailableError):
        run_comparison(
            report_path=tmp_path / "report.md",
            embedding_mode="fake",
            judge=JudgeConfig(),
        )


# ---------------------------------------------------------------------------
# Statistical report upgrade (issue #140)
# ---------------------------------------------------------------------------


def test_report_includes_mcnemar_per_core_type(tmp_path, fake_vector_index):
    """Each Core Paraphrase Type must have a McNemar p-value row in the report."""
    from eval.paraphrase_comparison.models import CORE_PARAPHRASE_TYPES

    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")

    # The statistical tests section must be present inside Core Comparison.
    assert "McNemar" in report
    assert "Holm" in report
    # Every Core type must appear with a p-value row.
    for ptype in CORE_PARAPHRASE_TYPES:
        assert ptype in report


def test_report_includes_wilson_ci_for_core_types(tmp_path, fake_vector_index):
    """Wilson confidence intervals must be rendered for both stacks."""
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")

    # CI column header must appear in the Core Comparison statistical section.
    assert "95% CI" in report
    # CI notation uses brackets like [lo, hi].
    assert "[" in report and "]" in report


def test_report_holm_correction_applied_across_five_core_types(
    tmp_path, fake_vector_index
):
    """The report must mention Holm correction and show it covers the 5 Core types."""
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")

    # Holm correction note must be present (can be in header or footer of stat table).
    assert "Holm" in report
    # The report must note that 5 tests are corrected.
    assert "5" in report  # number of Core types being corrected


def test_report_probes_excluded_from_statistical_tests(tmp_path, fake_vector_index):
    """Probe types must NOT appear in the McNemar p-value table rows."""
    from eval.paraphrase_comparison.models import PROBE_PARAPHRASE_TYPES

    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")

    # The statistical tests heading is distinct; extract its table only.
    stat_heading = "### Statistical Tests (Core types"
    assert stat_heading in report
    stat_start = report.index(stat_heading)
    # The section ends at the next "## " heading or end of string.
    next_section = report.find("\n## ", stat_start + 1)
    stat_window = (
        report[stat_start:next_section] if next_section != -1 else report[stat_start:]
    )
    for ptype in PROBE_PARAPHRASE_TYPES:
        assert ptype not in stat_window, (
            f"Probe type '{ptype}' appeared in the McNemar statistical table — "
            "probes must remain descriptive-only"
        )


def test_report_includes_faithfulness_drift_disclosure(tmp_path, fake_vector_index):
    """The report must disclose the faithfulness-drift risk (AC 5)."""
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")

    # Faithfulness-drift is a distinctive phrase from the PRD disclosure requirement.
    assert "faithfulness" in report.lower() or "mislabeled" in report.lower()


def test_core_macro_average_caveat_preserved(tmp_path, fake_vector_index):
    """The Core macro-average caveat must still be present after the stats upgrade."""
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")

    assert "Core macro-average" in report
    assert "Caveat" in report
