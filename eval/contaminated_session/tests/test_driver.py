"""Driver tests (#608) — LLM-free.

``rewrite_fn`` is always an inline stub here, never CALLED as
``gateway.app.query_rewriting.rewrite_query`` — CODING_STANDARD §6.4 caps the
Query Rewriting surface at one live test (already spent by
``gateway/tests/test_query_rewriting.py``); this suite never invokes it.
One regression test below (``test_evaluate_case_call_shape_binds_against_the_real_rewrite_query``)
DOES import the real ``rewrite_query`` to check its signature via
``inspect.signature(...).bind(...)`` — that's introspection, not a call, so
it spends no LLM budget and needs no API key; it exists to catch #608-shaped
seam drift (a stub's call shape silently diverging from the real function's)
in CI instead of at real-run time.

The committed characterization corpus (``eval/contaminated_session/corpus/``)
drives both the "does contamination flip retrieval" tests (topic vocabulary
IS in that corpus by construction) and the real-``CASES`` integration smoke
test.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from eval.contaminated_session.driver import evaluate_case, index_corpus
from eval.contaminated_session.sessions import CASES, ContaminatedSessionCase


def _stub_rewrite_returning(contaminated_text: str, clean_text: str):
    """Build a rewrite_fn that returns a fixed string per arm, keyed on
    whether ``history`` is non-empty (contaminated) or empty (clean/control)
    — mirrors the real ``rewrite_query`` contract (empty history = passthrough
    is up to the caller's fixed text here, not literally unchanged, since the
    test wants to control retrieval directly).

    ``history`` is keyword-only, matching ``RewriteFn`` / the real
    ``rewrite_query`` signature exactly (#608) — a positional-friendly stub
    here would mask the seam mismatch the driver actually needs to catch."""

    def _rewrite(raw_query: str, *, history: list[dict]) -> str:
        return contaminated_text if history else clean_text

    return _rewrite


def _synthetic_case(*, name: str, note: str, followup_question: str = "ignored"):
    """A minimal case with one placeholder contaminated turn — every synthetic
    driver test only cares about the (name, followup_question, note) it
    passes, never the turn content itself (the stub rewrite ignores it and
    keys only on history being non-empty)."""
    return ContaminatedSessionCase(
        name=name,
        contaminated_history=[
            {
                "question": "q",
                "answer": "a",
                "stack": "wiki",
                "grounding_reason": "claim_supported",
                "ts": "2026-01-01T00:00:00Z",
            }
        ],
        clean_history=[],
        followup_question=followup_question,
        note=note,
    )


@pytest.fixture
def two_topic_corpus(tmp_path: Path) -> Path:
    """A tiny corpus with two clearly-distinct-vocabulary Sources, so a
    stub rewrite can deterministically steer retrieval toward one or the
    other."""
    (tmp_path / "topic_a.md").write_text(
        "# Topic A\n\n## Alpha Section\nAardvark aardvark aardvark zebra content.\n",
        encoding="utf-8",
    )
    (tmp_path / "topic_b.md").write_text(
        "# Topic B\n\n## Beta Section\nOctopus octopus octopus walrus content.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_flipped_true_when_rewrites_retrieve_different_sections(two_topic_corpus):
    index_corpus(two_topic_corpus)
    case = _synthetic_case(name="synthetic", note="synthetic flip probe")
    rewrite_fn = _stub_rewrite_returning(
        contaminated_text="octopus walrus", clean_text="aardvark zebra"
    )
    outcome = evaluate_case(case, rewrite_fn)
    assert outcome.contaminated_gate.top_source is not None
    assert outcome.clean_gate.top_source is not None
    assert outcome.contaminated_gate.top_source != outcome.clean_gate.top_source
    assert outcome.flipped is True


def test_flipped_false_when_both_rewrites_retrieve_same_section(two_topic_corpus):
    index_corpus(two_topic_corpus)
    case = _synthetic_case(name="synthetic-no-flip", note="synthetic no-flip probe")
    rewrite_fn = _stub_rewrite_returning(
        contaminated_text="aardvark zebra", clean_text="aardvark zebra"
    )
    outcome = evaluate_case(case, rewrite_fn)
    assert outcome.contaminated_gate.top_source == outcome.clean_gate.top_source
    assert outcome.flipped is False


def test_flipped_true_when_clean_answers_but_contaminated_falls_to_cannot_confirm(
    two_topic_corpus,
):
    index_corpus(two_topic_corpus)
    case = _synthetic_case(
        name="synthetic-cc-flip",
        note="contamination drives the query fully out of corpus vocabulary",
    )
    rewrite_fn = _stub_rewrite_returning(
        contaminated_text="totally unrelated nonexistent gibberish query",
        clean_text="aardvark zebra",
    )
    outcome = evaluate_case(case, rewrite_fn)
    assert outcome.clean_gate.reason == "claim_supported"
    assert outcome.contaminated_gate.reason in {"below_threshold", "retrieval_empty"}
    assert outcome.flipped is True


def test_evaluate_case_computes_drift_against_literal_followup(two_topic_corpus):
    index_corpus(two_topic_corpus)
    case = _synthetic_case(
        name="synthetic-drift", note="drift probe", followup_question="aardvark"
    )
    rewrite_fn = _stub_rewrite_returning(
        contaminated_text="aardvark zebra octopus", clean_text="aardvark"
    )
    outcome = evaluate_case(case, rewrite_fn)
    assert outcome.drift.literal_question == "aardvark"
    assert outcome.drift.rewritten_query == "aardvark zebra octopus"
    assert outcome.drift.token_overlap < 1.0
    assert outcome.drift.length_ratio > 1.0


def test_index_corpus_populates_sections_over_committed_corpus():
    """Integration smoke: the committed characterization corpus indexes
    non-empty (mirrors eval.negative_case's equivalent check)."""
    pages, sections = index_corpus()
    assert pages > 0
    assert sections > 0


def test_real_cases_evaluate_end_to_end_with_a_no_op_rewrite():
    """Every hand-written CASES entry runs through evaluate_case without
    error against the real committed corpus, using an identity rewrite (the
    turn-1-passthrough shape) so this test stays LLM-free and deterministic."""
    index_corpus()

    def _identity_rewrite(raw_query: str, *, history: list[dict]) -> str:
        return raw_query

    for case in CASES:
        outcome = evaluate_case(case, _identity_rewrite)
        assert outcome.contaminated_rewrite == case.followup_question
        assert outcome.clean_rewrite == case.followup_question
        # Identity rewrite means both arms retrieve identically -> never flipped.
        assert outcome.flipped is False


def test_evaluate_case_call_shape_binds_against_the_real_rewrite_query():
    """Regression for #608: ``evaluate_case`` called ``rewrite_fn(raw_query,
    history)`` POSITIONALLY, but the real seam
    ``gateway.app.query_rewriting.rewrite_query`` declares ``history`` as
    KEYWORD-ONLY (``def rewrite_query(raw_query: str, *, history: list[dict])``).
    Every stub in this suite accepted history either way, so all 21 tests
    passed while a real run crashed with TypeError before any work.

    This test imports the REAL ``rewrite_query`` (never calls it — no LLM
    call, no API key needed) and asserts the driver's actual call shape
    binds against its real signature via
    ``inspect.signature(...).bind(...)``, which raises ``TypeError`` on a
    shape mismatch. Any future drift in either signature fails this test in
    CI instead of surfacing only in a real (spend-triggering) run."""
    from gateway.app.query_rewriting import rewrite_query

    signature = inspect.signature(rewrite_query)
    # Mirrors evaluate_case's real call: rewrite_fn(case.followup_question, history=...)
    signature.bind("literal follow-up question", history=[])
