"""One-off CLI: corpus v3 answer-phase pilot batch (issue #674, ADR-0045
Prerequisite 2; parent PRD #654, governing contract ADR-0045).

Usage (from repo root):

    uv run python -m eval.corpus_v3.pilot_batch
    # then produce the canonical full-run projection from the recorded ledger:
    uv run python -m eval.corpus_v3.run_verdict --mode live --confirm-live \\
        --pilot-ledger eval/corpus_v3/pilot_ledger.json \\
        --planned-calls <corrected figure printed by this CLI>

Runs a small deterministic pilot batch through the REAL wired AnswerFn seam
(``answer_fn.build_answer_fn`` -- every arm's production ``query()`` surface,
real retrieval over the committed corpus v3 fixtures, real LLM calls), records
a ``load_pilot_ledger``-compatible query-phase ledger, and writes a pilot
report covering issue #674's acceptance criteria:

    1. Pilot ledger JSON committed under ``eval/corpus_v3/`` (batch size and
       arm coverage stated in the report).
    2. The corrected ``--planned-calls`` figure for the full-run projection:
       ``run_verdict.PLANNED_LIVE_CALLS`` counts planned ANSWERS (3,636
       queries x 4 arms), but the ledger records LLM CALLS, and one answered
       query is TWO calls (synthesis + grounding-verify — recorded since
       #674 closed the verify undercount via
       ``hooks.instrument_structured_output``). Projecting 14,544 straight
       over a two-call-per-answer ledger would undercount spend by ~half, so
       this CLI derives ``corrected = ceil(planned_answers x measured
       avg_llm_calls_per_answer)`` and prints the exact run_verdict command.
    3. Refusal-gate fairness (ADR-0045 Prerequisite 2): per-arm behaviour on
       the pilot's unanswerable (negative) subset -- refusal rate, and
       whether the refusal happened at the pre-LLM gate (zero LLM calls: A's
       Cannot-Confirm score threshold / B's distance ceiling) or post-hoc at
       the Grounding Check. ``dense_over_wiki`` has NO calibrated pre-LLM
       gate (``answer_fn.dense_over_wiki_query``'s documented gap) -- its
       negative-set behaviour is reported as evidence, not assumed.
    4. Pilot spend guard: a stated worst-case pre-spend projection with a
       hard stop above ``cost_guard.BUDGET_USD_CAP``, and a mid-run abort if
       actual recorded spend ever crosses the cap.

Production isolation is ``stacks.isolate_production_paths`` (originated here
as a private helper; issue #679 extracted it to ``stacks.py`` so the live
runner can share the IDENTICAL recipe instead of a second copy): every
persisted-index/log path global is redirected to a temp dir before any
indexing, and ``WIKI_DIR`` points at a COPY of the wiki fixtures so
query-time ``derived_from`` lookups see real pages while the committed
fixtures stay untouched (the #316/#307 seed-overwrite lesson).

NOT run under pytest (``testpaths`` scopes collection to
``eval/corpus_v3/tests``); the pure helpers (subset selection, ledger
serialisation, summary math, report rendering) are unit-tested there with the
same fake-LLM seams ``test_answer_fn.py`` uses. This is a one-off script, so
stdout via ``print`` is acceptable (CODING_STANDARD §5.1's documented script
exemption, same as ``generate_queries.py``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata
from eval.cost_ledger.unit_prices import estimate_usd

from . import cost_guard
from . import stacks
from .answer_fn import build_answer_fn
from .query_schema import Query, load_queries
from .run_verdict import PLANNED_LIVE_CALLS

_PKG_ROOT = Path(__file__).resolve().parent
QUERIES_PATH = _PKG_ROOT / "generation" / "queries.yaml"
LEDGER_OUT_PATH = _PKG_ROOT / "pilot_ledger.json"
REPORT_OUT_PATH = _PKG_ROOT / "PILOT_REPORT.md"

ARMS = ("wiki", "rag", "hybrid", "dense_over_wiki")
NEGATIVE_STRATUM = "unanswerable"
PER_STRATUM_DEFAULT = 8

# Worst-case per-LLM-call token assumption for the PRE-SPEND pilot guard
# (deliberately fat: observed corpus v3 answer calls run ~1-2k in / a few
# hundred out). Only used before any money moves; the report states actuals.
_WORST_CASE_USAGE = UsageMetadata(
    input_tokens=4_000, output_tokens=800, total_tokens=4_800
)
_WORST_CASE_CALLS_PER_ANSWER = 3
_PILOT_GUARD_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------
def select_pilot_queries(queries: list[Query], *, per_stratum: int) -> list[Query]:
    """Deterministic pilot subset: the first ``per_stratum`` ENGLISH queries
    of each scenario stratum, in ``query_id`` order (the pre-registered
    enumeration -- no sampling randomness to argue about). zh is excluded
    because the live run's planned-call figure
    (``run_verdict.PLANNED_LIVE_CALLS``) is en-only by construction."""
    by_stratum: dict[str, list[Query]] = defaultdict(list)
    for q in sorted(queries, key=lambda q: q.query_id):
        if q.language == "en":
            by_stratum[q.scenario_stratum].append(q)
    subset: list[Query] = []
    for stratum in sorted(by_stratum):
        subset.extend(by_stratum[stratum][:per_stratum])
    return subset


def ledger_to_json(ledger: CostLedger) -> str:
    """Serialise a ledger to ``run_verdict.load_pilot_ledger``'s exact JSON
    shape (round-trip is unit-tested against the real loader)."""
    entries = [
        {
            "stack": c.stack,
            "phase": c.phase,
            "model": c.model,
            "usage": {
                "input_tokens": c.usage.input_tokens,
                "output_tokens": c.usage.output_tokens,
                "total_tokens": c.usage.total_tokens,
            },
        }
        for c in ledger.calls
    ]
    return json.dumps(entries, indent=2) + "\n"


@dataclass(frozen=True)
class PilotRow:
    """One pilot answer: which query/arm, how it resolved, what it cost."""

    query_id: str
    arm: str
    stratum: str
    refused: bool
    llm_calls: int


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    answers: int
    llm_calls: int
    usd: float | None
    negatives: int
    negatives_refused: int
    negatives_refused_pre_llm: int
    answerable: int
    answerable_answered: int


def summarize_arms(rows: list[PilotRow], ledger: CostLedger) -> list[ArmSummary]:
    """Per-arm rollup of the pilot rows, joined with the ledger's per-arm
    spend. Pre-LLM refusal = zero ledger calls for that answer (the gate
    fired before synthesis); a refusal WITH calls means the Grounding Check
    (or self-refusal) fired after a paid call."""
    summaries: list[ArmSummary] = []
    for arm in ARMS:
        arm_rows = [r for r in rows if r.arm == arm]
        if not arm_rows:
            continue
        negatives = [r for r in arm_rows if r.stratum == NEGATIVE_STRATUM]
        answerable = [r for r in arm_rows if r.stratum != NEGATIVE_STRATUM]
        totals = ledger.totals(stack=arm, phase="query")
        summaries.append(
            ArmSummary(
                arm=arm,
                answers=len(arm_rows),
                llm_calls=totals.calls,
                usd=totals.usd,
                negatives=len(negatives),
                negatives_refused=sum(r.refused for r in negatives),
                negatives_refused_pre_llm=sum(
                    r.refused and r.llm_calls == 0 for r in negatives
                ),
                answerable=len(answerable),
                answerable_answered=sum(not r.refused for r in answerable),
            )
        )
    return summaries


def corrected_planned_calls(
    rows: list[PilotRow], ledger: CostLedger, *, planned_answers: int
) -> int:
    """``planned_answers`` x the pilot's measured average LLM calls per
    answer, rounded up -- the honest ``--planned-calls`` input for
    ``run_verdict``'s per-LLM-call projection (see module docstring)."""
    total_calls = ledger.totals(phase="query").calls
    if not rows:
        raise ValueError("no pilot rows -- nothing to scale a projection from")
    return math.ceil(planned_answers * (total_calls / len(rows)))


def render_pilot_report(
    *,
    generated_at: str,
    per_stratum: int,
    rows: list[PilotRow],
    summaries: list[ArmSummary],
    ledger: CostLedger,
    planned_answers: int,
    ledger_path: Path,
) -> str:
    totals = ledger.totals(phase="query")
    corrected = corrected_planned_calls(rows, ledger, planned_answers=planned_answers)
    avg_calls = totals.calls / len(rows)
    models = sorted({c.model for c in ledger.calls})
    n_queries = len({r.query_id for r in rows})
    lines = [
        "# Corpus v3 pilot batch report (issue #674)",
        "",
        f"Generated: {generated_at}. Real pilot batch through the wired "
        "AnswerFn seam (`answer_fn.build_answer_fn`): every arm's production "
        "`query()` surface, real retrieval over the committed corpus v3 "
        "fixtures, real LLM calls.",
        "",
        "## Batch size and arm coverage",
        "",
        f"- {n_queries} English queries: the first {per_stratum} per scenario "
        "stratum in `query_id` order (deterministic pre-registered "
        "enumeration; zh excluded because `PLANNED_LIVE_CALLS` is en-only).",
        f"- All 4 arms answered every query: {len(rows)} answers, "
        f"{totals.calls} recorded LLM calls, models {', '.join(models)}.",
        f"- Pilot spend: ${totals.usd:.4f}"
        if totals.usd is not None
        else "- Pilot spend: unknown (unpriced model)",
        "",
        "## Per-arm results",
        "",
        "| arm | answers | LLM calls | USD | negatives refused | of which pre-LLM gate | answerable answered |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        usd = f"${s.usd:.4f}" if s.usd is not None else "unknown"
        lines.append(
            f"| {s.arm} | {s.answers} | {s.llm_calls} | {usd} "
            f"| {s.negatives_refused}/{s.negatives} "
            f"| {s.negatives_refused_pre_llm} "
            f"| {s.answerable_answered}/{s.answerable} |"
        )
    lines += [
        "",
        "## Refusal-gate fairness (ADR-0045 Prerequisite 2)",
        "",
        "Per-arm verdicts on the corpus v3 negative set (the pilot's "
        f"`{NEGATIVE_STRATUM}` slice, n={per_stratum} per arm):",
        "",
    ]
    for s in summaries:
        gate_name = {
            "wiki": "Cannot-Confirm score threshold (calibrated #253/#261)",
            "rag": "distance ceiling 1.1 (calibrated eval/rag_distance #257/#258)",
            "hybrid": "OR-gate over both calibrated gates",
            "dense_over_wiki": "NO calibrated pre-LLM gate (documented gap, "
            "`answer_fn.dense_over_wiki_query`)",
        }[s.arm]
        lines.append(
            f"- **{s.arm}** ({gate_name}): refused "
            f"{s.negatives_refused}/{s.negatives} negatives, "
            f"{s.negatives_refused_pre_llm} at the pre-LLM gate; answered "
            f"{s.answerable_answered}/{s.answerable} answerable queries."
        )
    lines += [
        "",
        "## Full-run projection input (the per-answer vs per-call correction)",
        "",
        f"- Planned ANSWERS: {planned_answers} "
        "(`run_verdict.PLANNED_LIVE_CALLS`: 3,636 en queries x 4 arms).",
        f"- Measured LLM calls per answer: {avg_calls:.3f} "
        f"({totals.calls} calls / {len(rows)} answers -- synthesis + "
        "grounding-verify per answered query; gate refusals cost 0).",
        f"- **Corrected `--planned-calls`: {corrected}** = "
        f"ceil({planned_answers} x {avg_calls:.3f}). Projecting the raw "
        f"{planned_answers} over this ledger would undercount spend by "
        f"~{(1 - planned_answers / corrected) * 100:.0f}%.",
        "",
        "Produce the canonical projection with:",
        "",
        "```",
        "uv run python -m eval.corpus_v3.run_verdict --mode live "
        f"--confirm-live --pilot-ledger {ledger_path.as_posix()} "
        f"--planned-calls {corrected}",
        "```",
        "",
        "## Projection output",
        "",
        "_(appended verbatim from the run_verdict invocation above)_",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _pilot_worst_case_usd(n_answers: int) -> float | None:
    per_call = estimate_usd(_PILOT_GUARD_MODEL, _WORST_CASE_USAGE)
    if per_call is None:
        return None
    return n_answers * _WORST_CASE_CALLS_PER_ANSWER * per_call


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corpus v3 pilot batch runner.")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--per-stratum", type=int, default=PER_STRATUM_DEFAULT)
    parser.add_argument("--ledger-out", type=Path, default=LEDGER_OUT_PATH)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT_PATH)
    args = parser.parse_args(argv)

    load_dotenv(find_dotenv(usecwd=True))
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY absent -- the pilot answers through real arm "
            "query() surfaces (real synthesis, verifier, and embedding "
            "calls). Nothing was run."
        )
        return 1
    if not 1 <= args.per_stratum <= 50:
        print(f"--per-stratum must be in [1, 50], got {args.per_stratum}.")
        return 1

    queries = load_queries(args.queries)
    subset = select_pilot_queries(queries, per_stratum=args.per_stratum)
    if not subset:
        print(f"no English queries found in {args.queries} -- nothing to pilot.")
        return 1
    n_answers = len(subset) * len(ARMS)

    worst_case = _pilot_worst_case_usd(n_answers)
    if worst_case is None:
        print(
            f"pilot guard: no pinned unit price for {_PILOT_GUARD_MODEL} -- "
            "cannot bound pilot spend; refusing to run (fail closed)."
        )
        return 1
    print(
        f"pilot guard: {n_answers} answers, worst-case "
        f"${worst_case:.2f} (<= ${cost_guard.BUDGET_USD_CAP:.2f} cap required)"
    )
    if worst_case > cost_guard.BUDGET_USD_CAP:
        print(
            "pilot guard: worst-case pilot spend exceeds the cap -- halting "
            "before any spend; shrink --per-stratum."
        )
        return 1

    iso = stacks.isolate_production_paths(prefix="corpus_v3_pilot_")
    print(f"production isolation: {iso}")
    wiki_sections, _ = stacks.index_wiki_corpus()
    dense_sections = stacks.index_dense_over_wiki()
    docs_files, docs_chunks = stacks.index_docs_corpus()
    print(
        f"indexed: wiki {wiki_sections} sections, dense {dense_sections} "
        f"sections, docs {docs_files} files / {docs_chunks} chunks "
        "(embedding spend is build-phase, one-off, excluded from the "
        "query-phase projection)"
    )

    ledger = CostLedger()
    answer_fn = build_answer_fn({q.query_id: q.text for q in subset}, ledger=ledger)
    rows: list[PilotRow] = []
    for arm in ARMS:
        for q in subset:
            calls_before = ledger.totals(phase="query").calls
            record = answer_fn(q.query_id, arm, [])
            calls_after = ledger.totals(phase="query").calls
            rows.append(
                PilotRow(
                    query_id=q.query_id,
                    arm=arm,
                    stratum=q.scenario_stratum,
                    refused=record.answer_text.strip().startswith(
                        CANNOT_CONFIRM_PHRASE
                    ),
                    llm_calls=calls_after - calls_before,
                )
            )
            spent = ledger.totals(phase="query").usd
            if spent is not None and spent > cost_guard.BUDGET_USD_CAP:
                print(
                    f"pilot guard: actual spend ${spent:.2f} crossed the "
                    f"${cost_guard.BUDGET_USD_CAP:.2f} cap mid-run -- "
                    "aborting; nothing written."
                )
                return 1
        print(f"arm {arm}: {len(subset)} answers done")

    summaries = summarize_arms(rows, ledger)
    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        # The committed report must not carry a machine-local absolute path.
        report_ledger_path = args.ledger_out.resolve().relative_to(Path.cwd())
    except ValueError:
        report_ledger_path = args.ledger_out
    args.ledger_out.write_text(ledger_to_json(ledger), encoding="utf-8")
    args.report_out.write_text(
        render_pilot_report(
            generated_at=generated_at,
            per_stratum=args.per_stratum,
            rows=rows,
            summaries=summaries,
            ledger=ledger,
            planned_answers=PLANNED_LIVE_CALLS,
            ledger_path=report_ledger_path,
        ),
        encoding="utf-8",
    )

    totals = ledger.totals(phase="query")
    corrected = corrected_planned_calls(
        rows, ledger, planned_answers=PLANNED_LIVE_CALLS
    )
    spend = f"${totals.usd:.4f}" if totals.usd is not None else "unknown"
    print(
        f"pilot done: {len(rows)} answers, {totals.calls} LLM calls, "
        f"spend {spend} -- ledger {args.ledger_out.name}, report "
        f"{args.report_out.name}"
    )
    print(
        "next: uv run python -m eval.corpus_v3.run_verdict --mode live "
        f"--confirm-live --pilot-ledger {args.ledger_out.as_posix()} "
        f"--planned-calls {corrected}"
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
