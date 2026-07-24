"""answer_fn.py tests — external behaviour only (CODING_STANDARD §0.2).

Every fake LLM here is hand-authored (CODING_STANDARD §6.5); only the
answer-synthesis + grounding-verifier LLM GETTERS are mocked (§6.3) — every
deep module (indexer search, dense search, ``grounding.verify`` itself) runs
for real over the committed corpus v3 fixtures, via ``stacks.py``'s own
index-building helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import hybrid_kb.app.query as hybrid_query
import hybrid_kb.app.retrieval as hybrid_retrieval
import markdown_kb.app.retrieval as mk_retrieval
import pytest
import vector_rag.app.retrieval as vr_retrieval
from markdown_kb.app import grounding as grounding_module
from markdown_kb.app.grounding import GroundingClaim, GroundingResult
from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

from eval.corpus_v3 import answer_fn as answer_fn_module
from eval.corpus_v3 import stacks
from eval.cost_ledger.ledger import CostLedger

STORE_HOURS_QUERY = "what are the store hours"


# ---------------------------------------------------------------------------
# Fakes — LLM getter seams only (CODING_STANDARD §6.3)
# ---------------------------------------------------------------------------
@dataclass
class _FakeLLMResponse:
    content: str
    usage_metadata: dict = field(
        default_factory=lambda: {
            "input_tokens": 111,
            "output_tokens": 22,
            "total_tokens": 133,
        }
    )


class _FakeSynthLLM:
    """Stand-in for an app's answer-synthesis ``ChatOpenAI`` singleton."""

    def __init__(self, content: str, usage: dict | None = None):
        self._response = _FakeLLMResponse(
            content=content, **({"usage_metadata": usage} if usage else {})
        )

    def invoke(self, messages):
        return self._response


class _FakeVerifierChain:
    def __init__(self, result: GroundingResult):
        self._result = result

    def invoke(self, message):
        return self._result


class _FakeVerifierLLM:
    """Stand-in for ``grounding.get_verifier_llm()``'s structured-output chain."""

    def __init__(self, result: GroundingResult):
        self._result = result

    def with_structured_output(self, schema):
        return _FakeVerifierChain(self._result)


def _passing_result(cited_id: str) -> GroundingResult:
    return GroundingResult(
        reasoning="Claim traces to the cited section.",
        claims=[
            GroundingClaim(text="ok", supported=True, citing_section_ids=[cited_id])
        ],
        unsupported_claims=[],
        passed=True,
    )


@pytest.fixture()
def fake_verifier(monkeypatch):
    """Install a passing grounding verifier (cited id is irrelevant to the fake)."""

    def _install(cited_id: str = "store-hours") -> None:
        monkeypatch.setattr(
            grounding_module,
            "get_verifier_llm",
            lambda: _FakeVerifierLLM(_passing_result(cited_id)),
        )

    return _install


# ---------------------------------------------------------------------------
# parse_cited_source_ids
# ---------------------------------------------------------------------------
def test_parse_cited_source_ids_extracts_multiple_ids():
    text = "A fact. [Source: a.md#h] Another fact. [Source: b.md#h2]"
    assert answer_fn_module.parse_cited_source_ids(text) == frozenset(
        {"a.md#h", "b.md#h2"}
    )


def test_parse_cited_source_ids_strips_whitespace():
    assert answer_fn_module.parse_cited_source_ids(
        "Hours. [Source:  a.md#h ]"
    ) == frozenset({"a.md#h"})


def test_parse_cited_source_ids_empty_for_a_refusal():
    assert answer_fn_module.parse_cited_source_ids(CANNOT_CONFIRM_PHRASE) == frozenset()


# ---------------------------------------------------------------------------
# ARM_QUERY_FNS / registry parity with stacks.ARM_REGISTRY
# ---------------------------------------------------------------------------
def test_arm_query_fns_cover_the_same_four_arms_as_stacks_registry():
    assert set(answer_fn_module.ARM_QUERY_FNS) == set(stacks.ARM_REGISTRY)


# ---------------------------------------------------------------------------
# build_answer_fn — per-arm happy path (real query() surface, fake LLM)
# ---------------------------------------------------------------------------
def test_answer_fn_wiki_arm_answers_through_the_real_query_surface(
    monkeypatch, fake_verifier
):
    stacks.index_wiki_corpus()
    fake_verifier("store-hours")
    monkeypatch.setattr(
        mk_retrieval,
        "get_llm",
        lambda: _FakeSynthLLM("Hours are 9-6 weekdays. [Source: store-hours]"),
    )

    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})
    record = fn("q1", "wiki", [])

    assert record.query_id == "q1"
    assert record.arm == "wiki"
    assert record.answer_text == "Hours are 9-6 weekdays. [Source: store-hours]"
    assert record.cited_source_ids == frozenset({"store-hours"})


def test_answer_fn_rag_arm_answers_through_the_real_query_surface(
    monkeypatch, fake_verifier, fake_vector_index
):
    stacks.index_docs_corpus()
    fake_verifier("store_hours_a.md#weekday-hours")
    monkeypatch.setattr(
        vr_retrieval,
        "get_llm",
        lambda: _FakeSynthLLM(
            "Hours are 9-6 weekdays. [Source: store_hours_a.md#weekday-hours]"
        ),
    )

    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})
    record = fn("q1", "rag", [])

    assert record.arm == "rag"
    assert record.cited_source_ids == frozenset({"store_hours_a.md#weekday-hours"})


def test_answer_fn_hybrid_arm_answers_through_the_real_query_surface(
    monkeypatch, fake_verifier
):
    stacks.index_wiki_corpus()
    stacks.index_stack_c()
    fake_verifier("store-hours")
    monkeypatch.setattr(
        hybrid_query,
        "get_llm",
        lambda: _FakeSynthLLM("Hours are 9-6 weekdays. [Source: store-hours]"),
    )

    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})
    record = fn("q1", "hybrid", [])

    assert record.arm == "hybrid"
    assert record.cited_source_ids == frozenset({"store-hours"})


def test_answer_fn_dense_over_wiki_arm_answers_through_its_own_synthesis(
    monkeypatch, fake_verifier
):
    stacks.index_wiki_corpus()
    stacks.index_dense_over_wiki()
    fake_verifier("store-hours")
    monkeypatch.setattr(
        hybrid_query,
        "get_llm",
        lambda: _FakeSynthLLM("Hours are 9-6 weekdays. [Source: store-hours]"),
    )

    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})
    record = fn("q1", "dense_over_wiki", [])

    assert record.arm == "dense_over_wiki"
    assert record.cited_source_ids == frozenset({"store-hours"})


def test_answer_fn_dense_over_wiki_never_invokes_reciprocal_rank_fusion(
    monkeypatch, fake_verifier
):
    """ADR-0045 Prerequisite 1: this arm must bypass RRF entirely."""
    stacks.index_wiki_corpus()
    stacks.index_dense_over_wiki()
    fake_verifier("store-hours")
    monkeypatch.setattr(
        hybrid_query,
        "get_llm",
        lambda: _FakeSynthLLM("Hours are 9-6 weekdays. [Source: store-hours]"),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("dense_over_wiki must not fuse via reciprocal_rank_fusion")

    monkeypatch.setattr(hybrid_retrieval, "reciprocal_rank_fusion", _boom)

    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})
    record = fn("q1", "dense_over_wiki", [])  # must not raise

    assert record.answer_text


# ---------------------------------------------------------------------------
# Refusal passthrough (issue #673 AC: "Cannot-Confirm gates untouched")
# ---------------------------------------------------------------------------
def test_wiki_arm_refusal_passes_through_with_no_llm_call(monkeypatch):
    stacks.index_wiki_corpus()

    def _boom():
        raise AssertionError("a below-threshold refusal must never reach the LLM")

    monkeypatch.setattr(mk_retrieval, "get_llm", _boom)

    fn = answer_fn_module.build_answer_fn(
        {"q1": "zzqx completely unrelated gibberish qwzt"}
    )
    record = fn("q1", "wiki", [])

    assert record.answer_text == CANNOT_CONFIRM_PHRASE
    assert record.cited_source_ids == frozenset()


# ---------------------------------------------------------------------------
# Cost ledger wiring (issue #657's hooks, wired here per its own docstring)
# ---------------------------------------------------------------------------
def test_answer_fn_records_query_phase_usage_into_the_ledger(
    monkeypatch, fake_verifier
):
    stacks.index_wiki_corpus()
    fake_verifier("store-hours")
    monkeypatch.setattr(
        mk_retrieval,
        "get_llm",
        lambda: _FakeSynthLLM(
            "Hours are 9-6 weekdays. [Source: store-hours]",
            usage={"input_tokens": 250, "output_tokens": 40, "total_tokens": 290},
        ),
    )

    ledger = CostLedger()
    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY}, ledger=ledger)
    fn("q1", "wiki", [])

    totals = ledger.totals(stack="wiki", phase="query")
    assert totals.calls == 1
    assert totals.input_tokens == 250
    assert totals.output_tokens == 40


def test_answer_fn_restores_the_original_getter_after_each_call(
    monkeypatch, fake_verifier
):
    """Wrap-then-restore: the app's real getter is back in place after the call."""
    stacks.index_wiki_corpus()
    fake_verifier("store-hours")

    def original_getter():
        return _FakeSynthLLM("Hours are 9-6 weekdays. [Source: store-hours]")

    monkeypatch.setattr(mk_retrieval, "get_llm", original_getter)

    ledger = CostLedger()
    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY}, ledger=ledger)
    fn("q1", "wiki", [])

    assert mk_retrieval.get_llm is original_getter


def test_answer_fn_without_a_ledger_records_nothing(monkeypatch, fake_verifier):
    stacks.index_wiki_corpus()
    fake_verifier("store-hours")
    monkeypatch.setattr(
        mk_retrieval,
        "get_llm",
        lambda: _FakeSynthLLM("Hours are 9-6 weekdays. [Source: store-hours]"),
    )

    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})  # no ledger
    record = fn("q1", "wiki", [])

    assert record.answer_text  # ran fine with no cost accounting at all


# ---------------------------------------------------------------------------
# Unknown arm / unknown query id
# ---------------------------------------------------------------------------
def test_build_answer_fn_raises_on_an_unknown_arm():
    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})
    with pytest.raises(ValueError, match="unknown arm"):
        fn("q1", "not_a_real_arm", [])


def test_build_answer_fn_raises_on_an_unknown_query_id():
    fn = answer_fn_module.build_answer_fn({"q1": STORE_HOURS_QUERY})
    with pytest.raises(KeyError):
        fn("does-not-exist", "wiki", [])
