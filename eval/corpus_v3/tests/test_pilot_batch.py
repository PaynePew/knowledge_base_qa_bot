"""pilot_batch.py tests — external behaviour only (CODING_STANDARD §0.2).

The end-to-end test drives the REAL main() over the committed corpus v3
fixtures with the same fake-LLM getter seams ``test_answer_fn.py`` uses
(synthesis + verifier getters swapped, everything else real); the pure
helpers (subset selection, ledger serialisation, summary math) are tested
directly. No network call, no OPENAI_API_KEY required.

Covers issue #674.
"""

from __future__ import annotations

import hybrid_kb.app.query as hybrid_query
import markdown_kb.app.retrieval as mk_retrieval
import vector_rag.app.retrieval as vr_retrieval
from markdown_kb.app import grounding as grounding_module

from eval.corpus_v3 import pilot_batch
from eval.corpus_v3.query_schema import Query, dump_queries
from eval.corpus_v3.run_verdict import PLANNED_LIVE_CALLS, load_pilot_ledger
from eval.corpus_v3.tests.test_answer_fn import (
    _FakeSynthLLM,
    _FakeVerifierLLM,
    _passing_result,
)
from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata


def _query(qid: str, stratum: str, language: str = "en") -> Query:
    return Query(
        query_id=qid,
        text="what are the store hours",
        scenario_stratum=stratum,
        overlap_stratum="high_overlap",
        language=language,
        gold_section_ids=["store-hours"] if stratum != "unanswerable" else [],
        key_tokens=["store", "hours"],
        generating_family="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# select_pilot_queries
# ---------------------------------------------------------------------------
def test_select_pilot_queries_takes_en_head_per_stratum_in_id_order():
    queries = [
        _query("factoid-en-0001", "factoid"),
        _query("factoid-en-0000", "factoid"),
        _query("factoid-en-0002", "factoid"),
        _query("cross_doc-en-0000", "cross_doc"),
        _query("factoid-zh-0000", "factoid", language="zh"),
    ]

    subset = pilot_batch.select_pilot_queries(queries, per_stratum=2)

    ids = [q.query_id for q in subset]
    assert ids == ["cross_doc-en-0000", "factoid-en-0000", "factoid-en-0001"]


# ---------------------------------------------------------------------------
# ledger_to_json — must round-trip through run_verdict's real loader
# ---------------------------------------------------------------------------
def test_ledger_to_json_round_trips_through_load_pilot_ledger(tmp_path):
    ledger = CostLedger()
    ledger.record(
        stack="wiki",
        phase="query",
        model="gpt-4o-mini",
        usage=UsageMetadata(input_tokens=100, output_tokens=20, total_tokens=120),
    )
    ledger.record(
        stack="rag",
        phase="query",
        model="gpt-4o-mini",
        usage=UsageMetadata(input_tokens=50, output_tokens=10, total_tokens=60),
    )
    path = tmp_path / "pilot_ledger.json"
    path.write_text(pilot_batch.ledger_to_json(ledger), encoding="utf-8")

    loaded = load_pilot_ledger(path)

    totals = loaded.totals(phase="query")
    assert totals.calls == 2
    assert totals.total_tokens == 180
    assert loaded.totals(stack="wiki", phase="query").calls == 1


# ---------------------------------------------------------------------------
# summary math
# ---------------------------------------------------------------------------
def test_summarize_arms_classifies_pre_llm_and_post_grounding_refusals():
    rows = [
        pilot_batch.PilotRow("u1", "wiki", "unanswerable", refused=True, llm_calls=0),
        pilot_batch.PilotRow("u2", "wiki", "unanswerable", refused=True, llm_calls=2),
        pilot_batch.PilotRow("f1", "wiki", "factoid", refused=False, llm_calls=2),
    ]
    ledger = CostLedger()
    for _ in range(4):
        ledger.record(
            stack="wiki", phase="query", model="gpt-4o-mini", usage=UsageMetadata()
        )

    (summary,) = pilot_batch.summarize_arms(rows, ledger)

    assert summary.negatives == 2
    assert summary.negatives_refused == 2
    assert summary.negatives_refused_pre_llm == 1
    assert summary.answerable == 1
    assert summary.answerable_answered == 1
    assert summary.llm_calls == 4


def test_corrected_planned_calls_scales_by_measured_calls_per_answer():
    rows = [
        pilot_batch.PilotRow("q1", "wiki", "factoid", refused=False, llm_calls=2),
        pilot_batch.PilotRow("q2", "wiki", "factoid", refused=False, llm_calls=2),
        pilot_batch.PilotRow("q3", "wiki", "unanswerable", refused=True, llm_calls=0),
        pilot_batch.PilotRow("q4", "wiki", "unanswerable", refused=True, llm_calls=2),
    ]
    ledger = CostLedger()
    for _ in range(6):
        ledger.record(
            stack="wiki", phase="query", model="gpt-4o-mini", usage=UsageMetadata()
        )

    corrected = pilot_batch.corrected_planned_calls(
        rows, ledger, planned_answers=1000
    )

    assert corrected == 1500  # 6 calls / 4 answers = 1.5 per answer


# ---------------------------------------------------------------------------
# End-to-end main() — real fixtures, fake LLM getter seams
# ---------------------------------------------------------------------------
def test_main_records_two_calls_per_answer_and_writes_artifacts(
    monkeypatch, tmp_path, fake_vector_index
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(pilot_batch, "load_dotenv", lambda *a, **k: None)
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

    queries_path = tmp_path / "queries.yaml"
    dump_queries(
        [
            _query(f"{stratum}-en-{i:04d}", stratum)
            for stratum in ("factoid", "cross_doc", "version_conflict", "unanswerable")
            for i in range(2)
        ]
        + [_query("factoid-zh-0000", "factoid", language="zh")],
        queries_path,
    )
    ledger_out = tmp_path / "pilot_ledger.json"
    report_out = tmp_path / "PILOT_REPORT.md"

    code = pilot_batch.main(
        [
            "--queries",
            str(queries_path),
            "--per-stratum",
            "2",
            "--ledger-out",
            str(ledger_out),
            "--report-out",
            str(report_out),
        ]
    )

    assert code == 0
    # 8 en queries x 4 arms; every answer = synthesis + grounding-verify.
    loaded = load_pilot_ledger(ledger_out)
    assert loaded.totals(phase="query").calls == 8 * 4 * 2
    for arm in pilot_batch.ARMS:
        assert loaded.totals(stack=arm, phase="query").calls == 16

    report = report_out.read_text(encoding="utf-8")
    expected_corrected = PLANNED_LIVE_CALLS * 2
    assert f"--planned-calls {expected_corrected}" in report
    for arm in pilot_batch.ARMS:
        assert arm in report


def test_main_refuses_without_openai_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pilot_batch, "load_dotenv", lambda *a, **k: None)
    out = tmp_path / "pilot_ledger.json"

    code = pilot_batch.main(["--ledger-out", str(out)])

    assert code == 1
    assert not out.exists()
