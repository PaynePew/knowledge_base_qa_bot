"""Traditional-Chinese (繁中) negative-case eval coverage (#256).

Mirrors test_driver / test_calibrate against the committed Chinese corpus + case
sets, and pins the ``KB_EVAL_LANG`` language switch. LLM-free; the autouse conftest
isolates production paths. No markdown_kb change is needed — ``indexer.tokenize`` is
already CJK-aware (sliding character bigrams, ADR-0014 / Phase 16).

The strict gate assertions use queries that overlap the corpus strongly (heading +
body bigram matches) so they are stable regressions; whether *every* in-scope query
clears the 0.5 default is the calibration question and is measured by ``calibrate``,
not asserted here.
"""

from __future__ import annotations

import pytest

from eval.negative_case.calibrate import collect_scores
from eval.negative_case.cases_zh import NEGATIVE_CASES_ZH
from eval.negative_case.driver import evaluate_case, index_corpus
from eval.negative_case.lang import CORPUS_DIR_ZH, resolve_lang
from eval.negative_case.positive_cases_zh import POSITIVE_CASES_ZH

# Reused verbatim from the committed CJK fixtures (proven retrievable).
_ZH_INSCOPE = "如何重設密碼？"
_ZH_CLEARLY_OOS = "附近有哪些餐廳？"


def test_zh_corpus_indexes_sections():
    """The Chinese corpus builds a non-empty Section Index."""
    _pages, sections = index_corpus(CORPUS_DIR_ZH)
    assert sections > 0


def test_zh_in_scope_query_is_not_refused():
    """A strongly-overlapping in-scope zh query clears the gate (not refused)."""
    index_corpus(CORPUS_DIR_ZH)
    outcome = evaluate_case(_ZH_INSCOPE)
    assert outcome.refused is False
    assert outcome.reason == "answered"
    assert outcome.top_score > 0.0


def test_zh_clearly_out_of_scope_is_refused():
    """A clearly out-of-scope zh query (no overlap) → Cannot Confirm."""
    index_corpus(CORPUS_DIR_ZH)
    outcome = evaluate_case(_ZH_CLEARLY_OOS)
    assert outcome.refused is True
    assert outcome.reason in {"retrieval_empty", "below_threshold"}


def test_zh_collect_scores_separates_positive_from_clearly_oos():
    """collect_scores over the zh sets: positives retrieve, clearly-oos negatives don't."""
    positive, negative = collect_scores(
        CORPUS_DIR_ZH, POSITIVE_CASES_ZH, NEGATIVE_CASES_ZH
    )
    assert len(positive) == len(POSITIVE_CASES_ZH)
    # Every in-scope query retrieves something (non-zero); a value < 0.5 is a real
    # over-refusal finding the calibration reports, not a test failure.
    assert all(s > 0.0 for s in positive)
    # The clearly-out-of-scope negatives have no overlap → some score exactly 0.
    assert any(s == 0.0 for s in negative)


def test_resolve_lang_defaults_to_english(monkeypatch):
    monkeypatch.delenv("KB_EVAL_LANG", raising=False)
    cfg = resolve_lang()
    assert cfg.lang == "en"
    assert cfg.report_suffix == ""


def test_resolve_lang_selects_zh(monkeypatch):
    monkeypatch.setenv("KB_EVAL_LANG", "zh")
    cfg = resolve_lang()
    assert cfg.lang == "zh"
    assert cfg.corpus_dir == CORPUS_DIR_ZH
    assert cfg.positive_cases is POSITIVE_CASES_ZH
    assert cfg.negative_cases is NEGATIVE_CASES_ZH
    assert cfg.report_suffix == "_zh"


def test_resolve_lang_rejects_unknown(monkeypatch):
    monkeypatch.setenv("KB_EVAL_LANG", "fr")
    with pytest.raises(ValueError):
        resolve_lang()
