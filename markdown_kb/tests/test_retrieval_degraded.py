"""Tests for issue #598 Slice B — degraded key-free serving at full budget cap.

``stream_query(question, degraded=True)`` must never call the LLM. Two modes
(scope addendum point 3, adversarial-verify follow-up grill):

- ``mode: "cached-qa"`` — a live ``wiki/qa/`` page (a Filed Answer) ranked top
  and above the normal gate threshold is replayed verbatim.
- ``mode: "sections"`` — every other outcome (no ranked hit, below threshold,
  index missing, or a non-qa top hit) returns the top Sections' own raw text
  excerpts beneath an honest notice line. CANNOT_CONFIRM_PHRASE is NEVER
  streamed on the degraded path — that sentinel specifically means "the
  corpus cannot support an answer", which degraded mode has no basis to
  assert (it never attempted synthesis).

Neither path calls ``qa.dispatch_filing`` — filing needs the verifier's
grounding_outcome, which degraded mode never produces.

Hermetic — no OPENAI_API_KEY, no real network. Uses the autouse
``_redirect_paths_to_tmp`` fixture from ``conftest.py`` (WIKI_DIR/INDEX_PATH/
LOG_PATH all point at tmp_path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.indexer as _indexer
from app.grounding import GroundingOutcome
from app.qa import maybe_file_answer, promote
from app.retrieval import _DEGRADED_SECTIONS_NOTICE, CANNOT_CONFIRM_PHRASE, stream_query

_FIXTURE_DOCS = Path(__file__).resolve().parent / "fixtures" / "docs"


class _RaisingLLM:
    """Fails the test if the LLM is ever invoked -- degraded mode must not call it."""

    def invoke(self, *_args, **_kwargs):
        raise AssertionError("degraded serving must never call the LLM")


@pytest.fixture(autouse=True)
def _forbid_llm_calls(monkeypatch):
    """Every test in this file asserts zero LLM calls (degraded is LLM-free)."""
    raising = _RaisingLLM()
    monkeypatch.setattr("app.retrieval._llm", raising)
    monkeypatch.setattr("app.retrieval.get_llm", lambda: raising)
    monkeypatch.setattr("app.retrieval._retry_llm", raising)
    monkeypatch.setattr("app.retrieval.get_retry_llm", lambda: raising)


@pytest.fixture(autouse=True)
def _patch_source_dirs(tmp_path, monkeypatch):
    """``build_index()``'s no-arg (production) path scans the module-level
    ``SOURCE_DIRS`` list, which is snapshotted at import time from the
    original ``WIKI_DIR`` — the conftest autouse fixture repoints
    ``WIKI_DIR`` itself but not this derived snapshot (same gotcha
    ``test_indexer_qa_filter.py::_patch_indexer`` works around), so any test
    here calling ``build_index()`` with no args needs this too.
    """
    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr(
        _indexer, "SOURCE_DIRS", [wiki_dir / "entities", wiki_dir / "concepts", wiki_dir / "qa"]
    )


def _file_and_promote_qa(question: str, answer: str) -> str:
    """File then promote a wiki/qa/ page so it enters the live retrieval corpus."""
    from dataclasses import dataclass

    @dataclass
    class _StubSection:
        id: str
        heading_path: list[str]
        content: str

    filed = maybe_file_answer(
        question, answer, [_StubSection(id="seed#s", heading_path=["s"], content="")]
    )
    assert filed is not None
    promote(filed.slug)
    return filed.slug


# ---------------------------------------------------------------------------
# Cached-QA branch — no LLM, top hit is a live wiki/qa/ page above threshold
# ---------------------------------------------------------------------------


def test_degraded_serves_cached_qa_answer_verbatim():
    question = "Where do I find the return shipping label for a damaged item?"
    answer = (
        "The return shipping label for a damaged item is emailed within one "
        "business day of the return shipping label request."
    )
    _file_and_promote_qa(question, answer)
    _indexer.build_index()  # production default -> scans SOURCE_DIRS (wiki/qa included)

    events = list(stream_query(question, degraded=True))
    assert len(events) == 2, "stream_query must yield exactly two dicts"
    full = events[1]

    assert full["answer"] == answer
    assert full["grounding_outcome"].passed is True
    assert full["grounding_outcome"].reason == "degraded_cached_qa"
    assert full["mode"] == "cached-qa"


def test_degraded_cached_qa_citations_narrow_to_that_one_page():
    question = "How do I dispute a duplicate charge on my statement?"
    answer = "File a dispute from the billing page within 60 days of the charge date."
    slug = _file_and_promote_qa(question, answer)
    _indexer.build_index()

    events = list(stream_query(question, degraded=True))
    sources = events[1]["sources"]
    assert len(sources) == 1, f"citations must narrow to the one qa page, got {sources!r}"
    assert sources[0]["source"] == slug


def test_degraded_qa_hit_never_dispatches_filing(tmp_path):
    """No qa filing in degraded mode (AC) -- no new wiki/qa/ file is written."""
    question = "What is the process for requesting a price match adjustment?"
    answer = "Submit the competitor listing within 14 days of purchase for a price match."
    _file_and_promote_qa(question, answer)
    _indexer.build_index()

    qa_dir = tmp_path / "wiki" / "qa"
    before = sorted(p.name for p in qa_dir.glob("*.md"))

    list(stream_query(question, degraded=True))

    after = sorted(p.name for p in qa_dir.glob("*.md"))
    assert after == before, "degraded serving must not create/touch any wiki/qa/ file"


# ---------------------------------------------------------------------------
# Sections branch (scope addendum point 3) — no qualifying live QA hit.
# Retrieval-only excerpts + notice, NEVER CANNOT_CONFIRM_PHRASE.
# ---------------------------------------------------------------------------


def test_degraded_index_missing_serves_notice_only_no_excerpts():
    """No sections at all -> the answer is the notice line alone, no excerpts."""
    _indexer.sections.clear()
    events = list(stream_query("any question at all", degraded=True))
    full = events[1]
    assert full["answer"] == _DEGRADED_SECTIONS_NOTICE
    assert full["answer"] != CANNOT_CONFIRM_PHRASE
    assert full["sources"] == []
    assert full["mode"] == "sections"
    assert full["grounding_outcome"] == GroundingOutcome(
        passed=False, reason="degraded_budget_exhausted"
    )


def test_degraded_below_threshold_never_streams_cannot_confirm():
    _indexer.build_index(_FIXTURE_DOCS)
    events = list(stream_query("zzzzxxxxxqqqqqwwwwwvvvvv", degraded=True))
    full = events[1]
    assert full["answer"] != CANNOT_CONFIRM_PHRASE
    assert full["answer"].startswith(_DEGRADED_SECTIONS_NOTICE)
    assert full["mode"] == "sections"
    assert full["grounding_outcome"].reason == "degraded_budget_exhausted"


def test_degraded_non_qa_top_hit_serves_section_excerpts_not_raw_answer():
    """A docs-style (non-qa) Section clearing the gate is served as a labelled
    excerpt beneath the honest notice -- never masqueraded as a fresh,
    LLM-synthesized Grounded Answer, and never the Cannot Confirm sentinel
    (which would falsely claim the corpus has no answer)."""
    _indexer.build_index(_FIXTURE_DOCS)
    events = list(stream_query("What is the refund policy?", degraded=True))
    full = events[1]
    assert full["answer"] != CANNOT_CONFIRM_PHRASE
    assert full["answer"].startswith(_DEGRADED_SECTIONS_NOTICE)
    top_source = full["sources"][0]
    assert f"[Source: {top_source['source']}]" in full["answer"]
    assert top_source["content"] in full["answer"]
    assert full["grounding_outcome"].passed is False
    assert full["grounding_outcome"].reason == "degraded_budget_exhausted"
    assert full["mode"] == "sections"


def test_degraded_sources_ready_partial_unaffected():
    """The first yield (sources_ready) is identical regardless of ``degraded``."""
    _indexer.build_index(_FIXTURE_DOCS)
    events = list(stream_query("What is the refund policy?", degraded=True))
    partial = events[0]
    assert partial["_phase"] == "sources_ready"
    assert partial["sources"], "sources must still be populated on the degraded path"
