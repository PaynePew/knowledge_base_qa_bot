"""live_runner.py tests — external behaviour only (CODING_STANDARD §0.2).

Covers issue #679: the real per-arm answering wired into ``run_verdict.py``
past the cost guard, checkpointing + resume, transient retry, the mid-run
spend abort, and content-axis scoring from real ``AnswerRecord``s. Every LLM
call is faked (the SAME synthesis / grounding-verifier getter seams
``test_answer_fn.py`` / ``test_pilot_batch.py`` use); no network call, no
real ``OPENAI_API_KEY`` anywhere in this file.
"""

from __future__ import annotations

import anthropic
import httpx
import hybrid_kb.app.query as hybrid_query
import markdown_kb.app.retrieval as mk_retrieval
import pytest
import vector_rag.app.retrieval as vr_retrieval
from markdown_kb.app import grounding as grounding_module
from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

from eval.corpus_v3 import live_runner, run_verdict
from eval.corpus_v3.build_corpus import ADVERSARIAL_GROUPS
from eval.corpus_v3.content_axes import AnswerRecord
from eval.corpus_v3.query_schema import Query, dump_queries
from eval.corpus_v3.run_verdict import load_pilot_ledger
from eval.corpus_v3.tests.test_answer_fn import (
    _FakeSynthLLM,
    _FakeVerifierLLM,
    _passing_result,
)
from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata


def _query(qid: str, stratum: str, gold: list[str] | None = None) -> Query:
    return Query(
        query_id=qid,
        text="what are the store hours",
        scenario_stratum=stratum,
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=(["store-hours"] if stratum != "unanswerable" else [])
        if gold is None
        else gold,
        key_tokens=["store", "hours"],
        generating_family="gpt-4o-mini",
    )


def _record(
    arm: str,
    query_id: str,
    *,
    answer_text: str = "answer",
    cited: frozenset[str] = frozenset(),
    retrieved: frozenset[str] = frozenset(),
) -> AnswerRecord:
    return AnswerRecord(
        query_id=query_id,
        arm=arm,
        answer_text=answer_text,
        cited_source_ids=cited,
        retrieved_source_ids=retrieved,
    )


# ---------------------------------------------------------------------------
# leak_source_ids_for_query (issue #679 point 5 -- derived, not invented)
# ---------------------------------------------------------------------------
@pytest.fixture()
def group_lookup():
    return live_runner._group_lookup()


def test_leak_ids_for_contradiction_factoid_is_the_other_conflicting_side(
    group_lookup,
):
    q = _query("factoid-en-0001", "factoid", gold=["gift_card_faq.md#expiration"])
    assert live_runner.leak_source_ids_for_query(q, group_lookup) == frozenset(
        {"gift_card_terms.md#expiration"}
    )


def test_leak_ids_for_contradiction_cross_doc_citing_both_sides_is_empty(
    group_lookup,
):
    """A cross_doc query's OWN gold already names both conflicting sides
    (targets.py: the dedup/conflict-comparison case) -- citing them is the
    honest framing, not a leak."""
    q = _query(
        "cross_doc-en-0001",
        "cross_doc",
        gold=["gift_card_terms.md#expiration", "gift_card_faq.md#expiration"],
    )
    assert live_runner.leak_source_ids_for_query(q, group_lookup) == frozenset()


def test_leak_ids_for_version_conflict_is_the_superseded_versions(group_lookup):
    q = _query(
        "version_conflict-en-0001",
        "version_conflict",
        gold=["return_shipping_v3.md#label-cost"],
    )
    assert live_runner.leak_source_ids_for_query(q, group_lookup) == frozenset(
        {"return_shipping_v1.md#label-cost", "return_shipping_v2.md#label-cost"}
    )


def test_leak_ids_for_redundancy_factoid_is_empty(group_lookup):
    """The sibling near-duplicate restates the SAME fact -- not a wrong one."""
    q = _query("factoid-en-0002", "factoid", gold=["store_hours_a.md#weekday-hours"])
    assert live_runner.leak_source_ids_for_query(q, group_lookup) == frozenset()


def test_leak_ids_for_unanswerable_query_is_empty(group_lookup):
    q = _query("unanswerable-en-0001", "unanswerable", gold=[])
    assert live_runner.leak_source_ids_for_query(q, group_lookup) == frozenset()


def test_leak_ids_when_gold_ids_span_multiple_groups_is_empty(group_lookup):
    """Fail soft on unexpected corpus drift rather than guessing."""
    q = _query(
        "factoid-en-0003",
        "factoid",
        gold=["gift_card_faq.md#expiration", "store_hours_a.md#weekday-hours"],
    )
    assert live_runner.leak_source_ids_for_query(q, group_lookup) == frozenset()


def test_group_lookup_covers_every_adversarial_group_section():
    lookup = live_runner._group_lookup()
    total_sections = sum(len(g.sections) for g in ADVERSARIAL_GROUPS)
    assert len(lookup) == total_sections


# ---------------------------------------------------------------------------
# build_axis_samples
# ---------------------------------------------------------------------------
def test_build_axis_samples_scores_grounding_refusal_and_leak_axes():
    queries = [
        _query("factoid-en-0000", "factoid", gold=["gift_card_faq.md#expiration"]),
        _query("unanswerable-en-0000", "unanswerable", gold=[]),
    ]
    records = {
        ("wiki", "factoid-en-0000"): _record(
            "wiki",
            "factoid-en-0000",
            cited=frozenset({"gift_card_faq.md#expiration"}),
            retrieved=frozenset({"gift_card_faq.md#expiration"}),
        ),
        ("rag", "factoid-en-0000"): _record(
            "rag",
            "factoid-en-0000",
            cited=frozenset({"gift_card_terms.md#expiration"}),  # the OTHER side
            retrieved=frozenset(
                {"gift_card_terms.md#expiration", "gift_card_faq.md#expiration"}
            ),
        ),
        ("wiki", "unanswerable-en-0000"): _record(
            "wiki", "unanswerable-en-0000", answer_text=CANNOT_CONFIRM_PHRASE
        ),
        ("rag", "unanswerable-en-0000"): _record(
            "rag", "unanswerable-en-0000", answer_text="answered anyway"
        ),
    }

    samples = live_runner.build_axis_samples(
        queries, records, wiki_backed_arms=["wiki"], baseline_arm="rag"
    )

    by_axis = {s.axis: s for s in samples}
    assert set(by_axis) == {
        "grounding_pass_rate",
        "correct_refusal_rate",
        "contradiction_leak_rate",
    }

    grounding = by_axis["grounding_pass_rate"]
    assert grounding.outcomes_a == [1]  # wiki: cited subset of retrieved
    assert grounding.outcomes_b == [
        1
    ]  # rag: citation is also within its own retrieved pool

    refusal = by_axis["correct_refusal_rate"]
    assert refusal.outcomes_a == [1]  # wiki correctly refused the unanswerable query
    assert refusal.outcomes_b == [0]  # rag answered it instead

    leak = by_axis["contradiction_leak_rate"]
    assert leak.outcomes_a == [0, 0]  # wiki never cites the OTHER side
    assert leak.outcomes_b == [1, 0]  # rag cites the leak id on the factoid query


# ---------------------------------------------------------------------------
# run_live_answering — happy path, skip, mid-run abort, resume
# ---------------------------------------------------------------------------
def _stub_answer_fn(ledger: CostLedger, *, usd_per_call: UsageMetadata):
    """A minimal fake ``AnswerFn``: records one ledger call per answer and
    returns a deterministic AnswerRecord — no real retrieval/LLM involved,
    purely to exercise ``run_live_answering``'s own control flow."""

    def _fn(query_id: str, arm: str, retrieved_items: list) -> AnswerRecord:
        ledger.record(stack=arm, phase="query", model="gpt-4o-mini", usage=usd_per_call)
        return AnswerRecord(query_id=query_id, arm=arm, answer_text=f"{arm}:{query_id}")

    return _fn


def test_run_live_answering_answers_every_pair_and_checkpoints_each_one(tmp_path):
    queries = [_query("q1", "factoid"), _query("q2", "factoid")]
    ledger = CostLedger()
    checkpoint = tmp_path / "checkpoint.jsonl"
    fn = _stub_answer_fn(ledger, usd_per_call=UsageMetadata())

    result = live_runner.run_live_answering(
        queries,
        ledger=ledger,
        checkpoint_path=checkpoint,
        answer_fn=fn,
        arms=("wiki", "rag"),
    )

    assert result.aborted is False
    assert len(result.rows) == 4  # 2 queries x 2 arms
    assert len(live_runner.load_checkpoint(checkpoint)) == 4


def test_run_live_answering_skips_pairs_already_in_already_done(tmp_path):
    queries = [_query("q1", "factoid"), _query("q2", "factoid")]
    ledger = CostLedger()
    checkpoint = tmp_path / "checkpoint.jsonl"
    calls: list[tuple[str, str]] = []

    def fn(query_id, arm, retrieved_items):
        calls.append((query_id, arm))
        ledger.record(
            stack=arm, phase="query", model="gpt-4o-mini", usage=UsageMetadata()
        )
        return AnswerRecord(query_id=query_id, arm=arm, answer_text="a")

    result = live_runner.run_live_answering(
        queries,
        ledger=ledger,
        checkpoint_path=checkpoint,
        answer_fn=fn,
        arms=("wiki", "rag"),
        already_done={("q1", "wiki"), ("q2", "rag")},
    )

    assert result.aborted is False
    assert len(result.rows) == 2  # only the 2 remaining pairs
    assert ("q1", "wiki") not in calls
    assert ("q2", "rag") not in calls


def test_run_live_answering_aborts_mid_run_when_actual_spend_crosses_the_cap(tmp_path):
    queries = [_query(f"q{i}", "factoid") for i in range(5)]
    ledger = CostLedger()
    checkpoint = tmp_path / "checkpoint.jsonl"
    # Each answer costs real money (priced model); a tiny cap trips after the
    # very first recorded call.
    fn = _stub_answer_fn(
        ledger,
        usd_per_call=UsageMetadata(
            input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000
        ),
    )

    result = live_runner.run_live_answering(
        queries,
        ledger=ledger,
        checkpoint_path=checkpoint,
        answer_fn=fn,
        arms=("wiki",),
        cap_usd=0.01,
    )

    assert result.aborted is True
    assert "cap" in result.abort_reason
    # Halted after the very first answer -- the rest of the 5 queries were
    # never attempted.
    assert len(result.rows) == 1
    assert len(live_runner.load_checkpoint(checkpoint)) == 1


def test_resume_skips_completed_pairs_and_spends_nothing_new_for_them(tmp_path):
    """The AC's own scenario: kill mid-run, resume, no duplicate spend for
    completed answers."""
    queries = [_query(f"q{i}", "factoid") for i in range(3)]
    checkpoint = tmp_path / "checkpoint.jsonl"
    usage = UsageMetadata(input_tokens=1_000, output_tokens=200, total_tokens=1_200)

    # First process: answers q0 and q1 only (simulates a kill after 2 answers).
    ledger1 = CostLedger()
    fn1 = _stub_answer_fn(ledger1, usd_per_call=usage)
    live_runner.run_live_answering(
        queries[:2],
        ledger=ledger1,
        checkpoint_path=checkpoint,
        answer_fn=fn1,
        arms=("wiki",),
    )
    assert ledger1.totals(phase="query").calls == 2

    # Resume: reload checkpoint, replay its ledger, and continue over the
    # FULL query set with a NEW ledger seeded from the replay.
    existing = live_runner.load_checkpoint(checkpoint)
    ledger2 = live_runner.replay_ledger(existing)
    done = {(r.query_id, r.arm) for r in existing}
    calls_made: list[str] = []

    def fn2(query_id, arm, retrieved_items):
        calls_made.append(query_id)
        ledger2.record(stack=arm, phase="query", model="gpt-4o-mini", usage=usage)
        return AnswerRecord(query_id=query_id, arm=arm, answer_text="a")

    result = live_runner.run_live_answering(
        queries,
        ledger=ledger2,
        checkpoint_path=checkpoint,
        answer_fn=fn2,
        arms=("wiki",),
        already_done=done,
    )

    assert result.aborted is False
    assert calls_made == ["q2"]  # only the un-answered pair re-spent anything
    assert ledger2.totals(phase="query").calls == 3  # 2 replayed + 1 new
    assert len(live_runner.load_checkpoint(checkpoint)) == 3


# ---------------------------------------------------------------------------
# Transient retry (mirrors generate_queries.py's own retry tests)
# ---------------------------------------------------------------------------
def _connect_timeout() -> Exception:
    return anthropic.APITimeoutError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


class _FlakyAnswerFn:
    """Raises ``errors`` (one per call, in order) before succeeding."""

    def __init__(self, errors: list[Exception]):
        self.errors = list(errors)
        self.attempts = 0

    def __call__(self, query_id, arm, retrieved_items):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return AnswerRecord(query_id=query_id, arm=arm, answer_text="ok")


def test_answer_with_retry_recovers_from_transient_errors(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(live_runner.time, "sleep", sleeps.append)
    fn = _FlakyAnswerFn([_connect_timeout(), _connect_timeout()])

    record = live_runner._answer_with_retry(fn, "q1", "wiki")

    assert record.answer_text == "ok"
    assert fn.attempts == 3
    assert len(sleeps) == 2


def test_answer_with_retry_reraises_when_errors_outlast_the_schedule(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(live_runner.time, "sleep", sleeps.append)
    budget = 1 + len(live_runner._TRANSIENT_RETRY_SLEEPS)
    fn = _FlakyAnswerFn([_connect_timeout() for _ in range(budget + 2)])

    with pytest.raises(anthropic.APIConnectionError):
        live_runner._answer_with_retry(fn, "q1", "wiki")

    assert fn.attempts == budget
    assert len(sleeps) == len(live_runner._TRANSIENT_RETRY_SLEEPS)


def test_answer_with_retry_does_not_retry_non_transient_errors(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(live_runner.time, "sleep", sleeps.append)
    fn = _FlakyAnswerFn([RuntimeError("programming bug")])

    with pytest.raises(RuntimeError):
        live_runner._answer_with_retry(fn, "q1", "wiki")

    assert fn.attempts == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# run_live_verdict — end-to-end wiring over real fixtures, faked LLMs
# ---------------------------------------------------------------------------
def _install_fakes(monkeypatch, fake_vector_index):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(
        grounding_module,
        "get_verifier_llm",
        lambda: _FakeVerifierLLM(_passing_result("store-hours")),
    )
    for module in (mk_retrieval, vr_retrieval, hybrid_query):
        monkeypatch.setattr(
            module,
            "get_llm",
            lambda: _FakeSynthLLM("Hours are 9-6 weekdays. [Source: store-hours]"),
        )


def _small_queries_path(tmp_path):
    queries_path = tmp_path / "queries.yaml"
    dump_queries(
        [
            _query("factoid-en-0000", "factoid"),
            _query("unanswerable-en-0000", "unanswerable"),
        ],
        queries_path,
    )
    return queries_path


def test_run_live_verdict_refuses_without_an_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = live_runner.run_live_verdict(
        queries_path=_small_queries_path(tmp_path),
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        ledger_out_path=tmp_path / "ledger.json",
        report_out_path=tmp_path / "VERDICT.md",
    )

    assert exit_code == 1
    assert not (tmp_path / "VERDICT.md").exists()


def test_run_live_verdict_writes_report_ledger_and_checkpoint(
    monkeypatch, tmp_path, fake_vector_index
):
    _install_fakes(monkeypatch, fake_vector_index)
    queries_path = _small_queries_path(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    ledger_out = tmp_path / "ledger.json"
    report_out = tmp_path / "VERDICT.md"

    exit_code = live_runner.run_live_verdict(
        queries_path=queries_path,
        checkpoint_path=checkpoint_path,
        ledger_out_path=ledger_out,
        report_out_path=report_out,
    )

    assert exit_code == 0
    report = report_out.read_text(encoding="utf-8")
    assert "## ADR-0045 clause walkthrough" in report
    assert "## Cost chapter" in report
    assert "## Method-comparison decision matrix" in report
    assert "## Honest limits" in report
    assert "dense_over_wiki" in report and "pre-LLM refusal gate" in report

    # 2 queries x 4 arms = 8 answers, each synthesis + grounding-verify.
    rows = live_runner.load_checkpoint(checkpoint_path)
    assert len(rows) == 8
    loaded_ledger = load_pilot_ledger(ledger_out)
    assert loaded_ledger.totals(phase="query").calls == 8 * 2


def test_run_live_verdict_resumes_and_spends_nothing_new_when_already_complete(
    monkeypatch, tmp_path, fake_vector_index
):
    _install_fakes(monkeypatch, fake_vector_index)
    queries_path = _small_queries_path(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    ledger_out = tmp_path / "ledger.json"
    report_out = tmp_path / "VERDICT.md"

    first = live_runner.run_live_verdict(
        queries_path=queries_path,
        checkpoint_path=checkpoint_path,
        ledger_out_path=ledger_out,
        report_out_path=report_out,
    )
    assert first == 0
    first_calls = load_pilot_ledger(ledger_out).totals(phase="query").calls

    # Make any NEW answer blow up loudly -- a second, fully-resumed run must
    # never reach the answering loop for a completed pair.
    def _boom():
        raise AssertionError("a completed pair must never be re-answered")

    for module in (mk_retrieval, vr_retrieval, hybrid_query):
        monkeypatch.setattr(module, "get_llm", _boom)
    monkeypatch.setattr(grounding_module, "get_verifier_llm", _boom)

    second = live_runner.run_live_verdict(
        queries_path=queries_path,
        checkpoint_path=checkpoint_path,
        ledger_out_path=ledger_out,
        report_out_path=report_out,
    )

    assert second == 0
    second_calls = load_pilot_ledger(ledger_out).totals(phase="query").calls
    assert second_calls == first_calls  # no duplicate spend on a full resume


# ---------------------------------------------------------------------------
# run_verdict.main wiring (issue #679: the exit-3 stub is gone)
# ---------------------------------------------------------------------------
def test_main_live_mode_without_api_key_refuses_past_the_guard(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        '[{"stack": "wiki", "phase": "query", "model": "gpt-4o-mini", '
        '"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}]',
        encoding="utf-8",
    )

    exit_code = run_verdict.main(
        [
            "--mode",
            "live",
            "--confirm-live",
            "--pilot-ledger",
            str(pilot_path),
            "--planned-calls",
            "10",
            "--queries",
            str(_small_queries_path(tmp_path)),
            "--checkpoint",
            str(tmp_path / "checkpoint.jsonl"),
            "--ledger-out",
            str(tmp_path / "ledger.json"),
            "--report-out",
            str(tmp_path / "VERDICT.md"),
        ]
    )

    assert exit_code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_live_mode_past_the_guard_runs_the_real_answer_fn(
    monkeypatch, tmp_path, fake_vector_index
):
    _install_fakes(monkeypatch, fake_vector_index)
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        '[{"stack": "wiki", "phase": "query", "model": "gpt-4o-mini", '
        '"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}]',
        encoding="utf-8",
    )
    report_out = tmp_path / "VERDICT.md"

    exit_code = run_verdict.main(
        [
            "--mode",
            "live",
            "--confirm-live",
            "--pilot-ledger",
            str(pilot_path),
            "--planned-calls",
            "10",
            "--queries",
            str(_small_queries_path(tmp_path)),
            "--checkpoint",
            str(tmp_path / "checkpoint.jsonl"),
            "--ledger-out",
            str(tmp_path / "ledger.json"),
            "--report-out",
            str(report_out),
        ]
    )

    assert exit_code == 0
    assert report_out.exists()
    assert "## ADR-0045 clause walkthrough" in report_out.read_text(encoding="utf-8")
