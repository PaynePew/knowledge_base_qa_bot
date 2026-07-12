"""Runner tests (#608) — trust-level path routing (CODING_STANDARD §6.6,
issue #328 precedent) + report rendering. LLM-free: every test drives the
``--fake`` / no-API-key path, never the real ``rewrite_query``.
"""

from __future__ import annotations

from eval.contaminated_session import runner
from eval.contaminated_session.driver import evaluate_case
from eval.contaminated_session.sessions import CASES


def _identity_rewrite(raw_query: str, *, history: list[dict]) -> str:
    return raw_query


def test_run_contaminated_session_returns_one_outcome_per_case():
    outcomes = runner.run_contaminated_session(_identity_rewrite)
    assert len(outcomes) == len(CASES)
    assert {o.case.name for o in outcomes} == {c.name for c in CASES}


def test_render_report_prepends_offline_header_only_when_not_real():
    outcomes = [evaluate_case(CASES[0], _identity_rewrite)]
    fake_report = runner.render_report(outcomes, real=False)
    real_report = runner.render_report(outcomes, real=True)
    assert fake_report.startswith(runner.OFFLINE_TRACER_HEADER)
    assert not real_report.startswith(runner.OFFLINE_TRACER_HEADER)
    assert runner.OFFLINE_TRACER_HEADER not in real_report


def test_render_report_includes_every_case_by_name():
    outcomes = [evaluate_case(case, _identity_rewrite) for case in CASES]
    report = runner.render_report(outcomes, real=True)
    for case in CASES:
        assert case.name in report
        assert case.followup_question in report


def test_offline_run_writes_only_tracer_not_canonical(tmp_path, monkeypatch):
    """A run with no OPENAI_API_KEY (the default --fake auto-detection) must
    write to OFFLINE_TRACER_REPORT_PATH only — the canonical report.md must
    stay untouched (the #328 footgun eval.paraphrase_comparison.runner
    already guards against, extended to this eval arm)."""
    canonical_report = tmp_path / "report.md"
    tracer_report = tmp_path / "report.offline-tracer.md"
    monkeypatch.setattr(runner, "REPORT_PATH", canonical_report)
    monkeypatch.setattr(runner, "OFFLINE_TRACER_REPORT_PATH", tracer_report)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = runner.main([])

    assert exit_code == 0
    assert tracer_report.exists(), "tracer report not written"
    assert not canonical_report.exists(), (
        "no-key run must NOT write canonical report.md"
    )
    content = tracer_report.read_text(encoding="utf-8")
    assert content.split("\n")[0] == runner.OFFLINE_TRACER_HEADER


def test_fake_flag_forces_tracer_path_even_with_a_key_present(tmp_path, monkeypatch):
    """--fake must win even when OPENAI_API_KEY is set (parity with
    run_comparison.py --fake-embeddings — a real key never auto-upgrades a
    forced fake run)."""
    canonical_report = tmp_path / "report.md"
    tracer_report = tmp_path / "report.offline-tracer.md"
    monkeypatch.setattr(runner, "REPORT_PATH", canonical_report)
    monkeypatch.setattr(runner, "OFFLINE_TRACER_REPORT_PATH", tracer_report)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-test-only")

    exit_code = runner.main(["--fake"])

    assert exit_code == 0
    assert tracer_report.exists()
    assert not canonical_report.exists()


def test_offline_rewrite_stub_preserves_turn1_passthrough():
    assert runner._offline_rewrite_stub("hello", history=[]) == "hello"


def test_offline_rewrite_stub_appends_prior_question_when_history_present():
    history = [
        {
            "question": "How long do refunds take?",
            "answer": "5-7 days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-01-01T00:00:00Z",
        }
    ]
    result = runner._offline_rewrite_stub("And store credit?", history=history)
    assert result == "And store credit? [How long do refunds take?]"
