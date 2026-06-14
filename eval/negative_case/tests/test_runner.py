"""Runner tests — build the corpus, run all cases, check the aggregate report.

Hermetic: the autouse conftest fixture isolates production paths, and the eval is
LLM-free, so ``run_negative_case`` runs directly without ``_isolate_production_paths``.
"""

from __future__ import annotations

from eval.negative_case.cases import NEGATIVE_CASES
from eval.negative_case.runner import render_report, run_negative_case


def test_run_covers_every_case():
    report = run_negative_case()
    assert len(report.outcomes) == len(NEGATIVE_CASES)
    assert 0.0 <= report.rate <= 1.0


def test_refusal_outcomes_are_internally_consistent():
    """Deterministic invariant that does NOT assume the threshold is well-tuned.

    The eval's whole purpose is to *measure* the refusal rate, which can legitimately
    be < 1.0 (the first run leaks several out-of-scope queries — junk tokens like a
    possessive ``'s`` or an unfiltered ``in`` create spurious BM25 hits over the 0.5
    threshold). So assert only the structural invariant: a refused case carries a
    gate reason; a leaked case scored at/above the threshold and is marked answered.
    """
    import markdown_kb.app.retrieval as retrieval

    report = run_negative_case()
    for _case, outcome in report.outcomes:
        if outcome.refused:
            assert outcome.reason in {"retrieval_empty", "below_threshold"}
        else:
            assert outcome.reason == "answered"
            assert outcome.top_score >= retrieval._SCORE_THRESHOLD


def test_render_report_has_headline_and_rows():
    report = run_negative_case()
    md = render_report(report)
    assert "Correct-refusal rate:" in md
    # Every case appears as a row in the per-case table.
    for case, _ in report.outcomes:
        assert case.query in md
