"""run_verdict.py tests — external behaviour only (CODING_STANDARD §0.2).

No LLM calls anywhere in this file, matching the module under test: the
default (offline) path is pure/deterministic, and the cost-guard path is
exercised against a hand-scripted pilot ledger (CODING_STANDARD §6.5), never
real spend.
"""

from __future__ import annotations

import json

import pytest

from eval.corpus_v3 import run_verdict
from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata


# ---------------------------------------------------------------------------
# AxisSample.to_comparison
# ---------------------------------------------------------------------------
def test_axis_sample_to_comparison_computes_rates_and_a_real_p_value():
    sample = run_verdict.AxisSample(
        axis="grounding_pass_rate",
        arm_a="wiki",
        arm_b="rag",
        outcomes_a=[1, 1, 1, 1, 1, 0, 1, 1],
        outcomes_b=[1, 0, 1, 0, 1, 0, 0, 1],
    )
    comparison = sample.to_comparison()
    assert comparison.axis == "grounding_pass_rate"
    assert comparison.n == 8
    assert comparison.rate_a == pytest.approx(0.875)
    assert comparison.rate_b == pytest.approx(0.5)
    assert 0.0 <= comparison.p_value <= 1.0
    assert comparison.test_name == "mcnemar"


# ---------------------------------------------------------------------------
# load_pilot_ledger / run_cost_guard
# ---------------------------------------------------------------------------
def test_load_pilot_ledger_round_trips_recorded_calls(tmp_path):
    path = tmp_path / "pilot.json"
    path.write_text(
        json.dumps(
            [
                {
                    "stack": "wiki",
                    "phase": "query",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 50,
                        "total_tokens": 1050,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    ledger = run_verdict.load_pilot_ledger(path)
    assert ledger.totals(phase="query").calls == 1


def test_run_cost_guard_returns_false_and_prints_projection_when_over_cap(capsys):
    ledger = CostLedger()
    for _ in range(5):
        ledger.record(
            stack="wiki",
            phase="query",
            model="gpt-4o-mini",
            usage=UsageMetadata(
                input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000
            ),
        )
    proceed = run_verdict.run_cost_guard(ledger, planned_calls=1_000_000)
    assert proceed is False
    assert "cost guard" in capsys.readouterr().err


def test_run_cost_guard_returns_false_when_pilot_ledger_has_no_query_sample(capsys):
    ledger = CostLedger()  # empty -- no phase="query" sample at all
    proceed = run_verdict.run_cost_guard(ledger, planned_calls=100)
    assert proceed is False
    assert "no recorded calls" in capsys.readouterr().err


def test_run_cost_guard_returns_true_when_projection_is_within_the_cap(capsys):
    ledger = CostLedger()
    ledger.record(
        stack="wiki",
        phase="query",
        model="gpt-4o-mini",
        usage=UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    proceed = run_verdict.run_cost_guard(ledger, planned_calls=10)
    assert proceed is True
    assert "proceeding" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# build_offline_tracer_report
# ---------------------------------------------------------------------------
def test_build_offline_tracer_report_is_loudly_trust_marked_and_complete():
    report = run_verdict.build_offline_tracer_report()
    assert report.startswith(run_verdict.OFFLINE_TRACER_HEADER)
    assert "NOT A LIVE VERDICT RUN" in report
    assert "## ADR-0045 clause walkthrough" in report
    assert "## Cost chapter" in report
    assert "## Method-comparison decision matrix" in report
    assert "## Honest limits" in report


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------
def test_main_offline_mode_writes_the_tracer_report(tmp_path, monkeypatch):
    out_path = tmp_path / "VERDICT.offline-tracer.md"
    monkeypatch.setattr(run_verdict, "OFFLINE_TRACER_REPORT_PATH", out_path)

    exit_code = run_verdict.main(["--mode", "offline"])

    assert exit_code == 0
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith(
        run_verdict.OFFLINE_TRACER_HEADER
    )


def test_main_live_mode_without_confirm_live_refuses_to_run(capsys):
    exit_code = run_verdict.main(["--mode", "live"])
    assert exit_code == 2
    assert "--confirm-live" in capsys.readouterr().err


def test_main_live_mode_without_pilot_ledger_refuses_to_run(capsys):
    exit_code = run_verdict.main(["--mode", "live", "--confirm-live"])
    assert exit_code == 2
    assert "--pilot-ledger" in capsys.readouterr().err


def test_main_live_mode_halts_on_cost_guard_failure(tmp_path, capsys):
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        json.dumps(
            [
                {
                    "stack": "wiki",
                    "phase": "query",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 1_000_000,
                        "total_tokens": 2_000_000,
                    },
                }
            ]
        ),
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
            "1000000",
        ]
    )
    assert exit_code == 1
    assert "cost guard" in capsys.readouterr().err


def test_main_live_mode_past_the_guard_still_refuses_without_an_answer_fn(
    tmp_path, capsys
):
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        json.dumps(
            [
                {
                    "stack": "wiki",
                    "phase": "query",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    },
                }
            ]
        ),
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
        ]
    )
    assert exit_code == 3
    assert "answer_fn" in capsys.readouterr().err
