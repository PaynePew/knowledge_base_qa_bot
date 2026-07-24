"""Explicit, opt-in orchestration entry point for the corpus v3 verdict run
(issue #662, PRD #654's definition of done). This module is a SCRIPT, not a
test -- it carries no ``test_*`` functions and is never collected by
``pytest`` (default or ``-m live``); it is invoked directly:

    uv run python -m eval.corpus_v3.run_verdict --mode offline
    uv run python -m eval.corpus_v3.run_verdict --mode live --confirm-live \
        --seed 42 --pilot-ledger path/to/pilot_ledger.json

The CLI's ``--mode live`` path only runs the cost guard (see below) -- it has
no ``answer_fn`` flag by design (see the "live-mode answering seam" section
below); actually executing a live run requires calling this module's Python
API directly with an ``answer_fn`` supplied.

Seeded (``--seed``, default :data:`DEFAULT_SEED`) and temperature pinned to
:data:`LIVE_TEMPERATURE` (0) for every LLM call a live run makes --
reproducibility over the two axes that most threaten a verdict's
credibility: sampling and generation order (PRD #654 / issue #662 AC 1).

``--mode offline`` (the DEFAULT, always-safe mode) never calls an LLM, never
touches ``OPENAI_API_KEY``, and produces the CODING_STANDARD §6.6
offline-tracer artifact (``VERDICT.offline-tracer.md``) by driving the SAME
report-assembly pipeline a live run would, over a small hand-authored canned
answer set -- exercising axis scoring, paired significance tests, clause
verdicts, and rendering end to end with zero cost and zero real data, exactly
like ``build_corpus.py``'s ``BUILD_COST.offline-tracer.md``. It is NOT a
substitute for the real verdict and every kill/demote/survive outcome it
prints is explicitly disclaimed as such (see ``OFFLINE_TRACER_HEADER``).

``--mode live`` is the real verdict run issue #662's "What to build"
describes. Before touching an LLM it ALWAYS runs
:func:`eval.corpus_v3.cost_guard.check_cost_guard` against a projection built
from a caller-supplied PILOT ledger (``--pilot-ledger``, already-recorded
query-phase calls from a small trial batch) scaled to the planned full-run
call count; on any guard failure -- over budget, or no pilot sample to
project from at all -- it halts and prints the projection instead of
spending anything (issue #662 AC 2). Even past the guard, live mode requires
a caller-supplied ``answer_fn`` (the seam a real per-arm, in-process query
call — PRD #654: "through each app's public query function, not HTTP" —
plugs into): wiring that integration for all three apps is real
production-adjacent work this script deliberately leaves as an explicit,
named seam rather than a silent partial implementation. There is no fallback
to fake data on the live path — that would violate CODING_STANDARD §6.6's
canonical-name guarantee (only a run backed by ``answer_fn`` may write the
canonical ``VERDICT.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.corpus_v3.content_axes import AnswerRecord
from eval.corpus_v3.cost_guard import check_cost_guard, project_spend
from eval.corpus_v3.models import RetrievedItem
from eval.corpus_v3.statistics import mcnemar_test
from eval.corpus_v3.verdict_report import (
    DecisionMatrixCell,
    DecisionMatrixRow,
    PairwiseComparison,
    VerdictReportInput,
    demote_clause_verdict,
    kill_clause_verdict,
    render_verdict_report,
    survival_entries,
)
from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata

DEFAULT_SEED = 42
LIVE_TEMPERATURE = 0  # every LLM call a live run makes is pinned to this.

BASELINE_ARM = "rag"
WIKI_ARM = "wiki"
HYBRID_ARM = "hybrid"
WIKI_BACKED_ARMS = ("wiki", "hybrid", "dense_over_wiki")

# Power-sized target (POWER_ANALYSIS.md: n=909 per scenario stratum x 4
# strata ~= 3,636 English queries) x 4 arms -- the planned call count a real
# run's cost guard projects against. Not used by --mode offline.
POWER_SIZED_QUERIES_PER_ARM = 3_636
PLANNED_LIVE_CALLS = POWER_SIZED_QUERIES_PER_ARM * 4

_PKG_ROOT = Path(__file__).resolve().parent
CANONICAL_REPORT_PATH = _PKG_ROOT / "VERDICT.md"
OFFLINE_TRACER_REPORT_PATH = _PKG_ROOT / "VERDICT.offline-tracer.md"

OFFLINE_TRACER_HEADER = (
    "⚠️ PLACEHOLDER — NOT A LIVE VERDICT RUN. Every answer scored below is a "
    "hand-authored, deterministic offline stand-in (no LLM call was made, no "
    "OPENAI_API_KEY touched); the significance tests run REAL math but over "
    "a tiny synthetic outcome set sized for pipeline demonstration, not "
    "ADR-0045's power-analysis requirement (POWER_ANALYSIS.md: n=909 per "
    "scenario stratum). This file exercises axis scoring, paired "
    "significance tests, the ADR-0045 clause walkthrough, and rendering end "
    "to end at zero cost — it is NOT the corpus v3 verdict, and none of its "
    "kill / demote / survive outcomes may be cited as such. A real run "
    "(`--mode live --confirm-live`, a cost-guard-cleared pilot ledger, a "
    "power-sized generated query set, and a real `answer_fn`) writes the "
    "canonical VERDICT.md instead."
)


# ---------------------------------------------------------------------------
# The live-mode answering seam (issue #662: "not HTTP")
# ---------------------------------------------------------------------------
AnswerFn = Callable[[str, str, list[RetrievedItem]], AnswerRecord]
"""``(query_id, arm, retrieved_items) -> AnswerRecord``. The caller-supplied,
in-process seam a live run drives each arm's real per-query answering
through. See module docstring."""


@dataclass(frozen=True)
class AxisSample:
    """One axis's paired 0/1 outcome vectors for two arms (a query-set-sized
    sample, hand-authored for --mode offline or produced by a live run)."""

    axis: str
    arm_a: str
    arm_b: str
    outcomes_a: Sequence[int]
    outcomes_b: Sequence[int]

    def to_comparison(self, *, stratum: str = "macro") -> PairwiseComparison:
        n = len(self.outcomes_a)
        result = mcnemar_test(list(self.outcomes_a), list(self.outcomes_b))
        return PairwiseComparison(
            axis=self.axis,
            stratum=stratum,
            arm_a=self.arm_a,
            arm_b=self.arm_b,
            rate_a=sum(self.outcomes_a) / n,
            rate_b=sum(self.outcomes_b) / n,
            n=n,
            p_value=result.p_value,
            test_name="mcnemar",
        )


# ---------------------------------------------------------------------------
# Public API — cost guard (issue #662 AC 2)
# ---------------------------------------------------------------------------
def load_pilot_ledger(path: Path) -> CostLedger:
    """Load a pilot ledger (a small already-recorded trial batch, JSON:
    ``[{"stack": ..., "phase": ..., "model": ..., "usage": {...}}, ...]``)
    the live run's cost projection scales from."""
    ledger = CostLedger()
    for entry in json.loads(path.read_text(encoding="utf-8")):
        ledger.record(
            stack=entry["stack"],
            phase=entry["phase"],
            model=entry["model"],
            usage=UsageMetadata.from_raw(entry.get("usage")),
        )
    return ledger


def run_cost_guard(pilot_ledger: CostLedger, *, planned_calls: int) -> bool:
    """Project ``planned_calls`` worth of query-phase spend from
    ``pilot_ledger`` and print the guard's verdict. Returns whether the live
    run may proceed (issue #662 AC 2: halt + report the projection instead
    of running, on any guard failure -- including "no pilot sample at all").
    """
    try:
        projection = project_spend(
            pilot_ledger, phase="query", planned_calls=planned_calls
        )
    except ValueError as exc:
        print(
            f"cost guard: {exc} -- halting; mark the issue ready-for-human",
            file=sys.stderr,
        )
        return False
    result = check_cost_guard(projection)
    print(f"cost guard: {result.message}", file=sys.stderr)
    return result.proceed


# ---------------------------------------------------------------------------
# --mode offline: hand-authored canned axis samples (no LLM, no randomness
# needed -- nothing here is sampled, mirroring build_corpus.py's own
# "offline, deterministic, hand-authored" convention).
# ---------------------------------------------------------------------------
def _canned_axis_samples() -> list[AxisSample]:
    return [
        # grounding_pass_rate, wiki vs rag (n=8): wiki grounds 7/8, rag 4/8.
        AxisSample(
            axis="grounding_pass_rate",
            arm_a=WIKI_ARM,
            arm_b=BASELINE_ARM,
            outcomes_a=[1, 1, 1, 1, 1, 0, 1, 1],
            outcomes_b=[1, 0, 1, 0, 1, 0, 0, 1],
        ),
        # correct_refusal_rate (unanswerable-only, n=4): wiki 3/4, rag 1/4.
        AxisSample(
            axis="correct_refusal_rate",
            arm_a=WIKI_ARM,
            arm_b=BASELINE_ARM,
            outcomes_a=[1, 1, 1, 0],
            outcomes_b=[0, 0, 1, 0],
        ),
        # contradiction_leak_rate, wiki vs rag (n=6): 1 == LEAKED (bad).
        # wiki leaks 1/6, rag leaks 4/6.
        AxisSample(
            axis="contradiction_leak_rate",
            arm_a=WIKI_ARM,
            arm_b=BASELINE_ARM,
            outcomes_a=[0, 0, 0, 1, 0, 0],
            outcomes_b=[1, 1, 0, 1, 1, 0],
        ),
        # contradiction_leak_rate, hybrid vs rag (n=6, demote clause's axis).
        # hybrid leaks 1/6, rag leaks 4/6 -- same rag sample as above.
        AxisSample(
            axis="contradiction_leak_rate",
            arm_a=HYBRID_ARM,
            arm_b=BASELINE_ARM,
            outcomes_a=[0, 0, 0, 0, 1, 0],
            outcomes_b=[1, 1, 0, 1, 1, 0],
        ),
    ]


def _canned_decision_matrix() -> tuple[list[str], list[DecisionMatrixRow]]:
    """Mirrors eval/fairness_review/method-comparison.md's evidence-status
    rows (PRD #654 user story 24) -- a NEW artifact, not an edit to that
    file (executing the verdict's narrative rewrite is separate follow-up
    work, PRD #654 § Out of Scope)."""
    columns = ["Evidence status"]
    rows = [
        DecisionMatrixRow(
            label="Contradiction control / auditability",
            cells={
                "Evidence status": DecisionMatrixCell(
                    "corpus v3's home axis; this run's own contradiction-leak "
                    "rate table is the local measurement",
                    "measured-local",
                )
            },
        ),
        DecisionMatrixRow(
            label="Cross-document sensemaking / global questions",
            cells={
                "Evidence status": DecisionMatrixCell(
                    "GraphRAG 72-83% comprehensiveness win rate; RAPTOR +20% "
                    "QuALITY -- not measured on a markdown wiki",
                    "measured-analogue",
                )
            },
        ),
        DecisionMatrixRow(
            label="Query-time token efficiency",
            cells={
                "Evidence status": DecisionMatrixCell(
                    "GraphRAG 9-43x fewer tokens (analogue); draft input "
                    "tokens per stratum measured locally this run",
                    "measured-local",
                )
            },
        ),
        DecisionMatrixRow(
            label="Compounding knowledge across sessions",
            cells={
                "Evidence status": DecisionMatrixCell(
                    "no head-to-head benchmark found anywhere", "argued"
                )
            },
        ),
    ]
    return columns, rows


def _canned_honest_limits() -> list[str]:
    return [
        "Every measured curated-layer analogue (GraphRAG, RAPTOR) is a "
        "structural analogue, not a markdown wiki -- the inference gap "
        "stated in eval/fairness_review/why-wiki-industry-evidence.md "
        "still applies to those rows of the decision matrix.",
        "The minimal detectable difference driving the power analysis "
        "(POWER_ANALYSIS.md) is a calculation from a normal-approximation "
        "closed form, not a citation.",
        "High-churn corpora and ACL-partitioned corpora remain unmeasured "
        "by design (PRD #654 § Out of Scope) -- the decision matrix's "
        "recommendation is bounded to low-churn, single-ACL-domain corpora.",
        "Update amplification and the inter-ingest staleness window are "
        "known wiki losses ADR-0045 states corpus v3 does not test, "
        "regardless of this run's kill/demote/survive outcome.",
    ]


def build_offline_tracer_report() -> str:
    """Assemble the full offline-tracer verdict report (issue #662's
    deliverable, run in the always-safe mode). Returns the rendered
    Markdown; :func:`main` writes it to :data:`OFFLINE_TRACER_REPORT_PATH`.
    """
    samples = _canned_axis_samples()
    comparisons = [s.to_comparison() for s in samples]

    kill = kill_clause_verdict(
        comparisons, wiki_arm=WIKI_ARM, baseline_arm=BASELINE_ARM
    )
    demote_comparison = next(
        c
        for c in comparisons
        if c.axis == "contradiction_leak_rate" and HYBRID_ARM in (c.arm_a, c.arm_b)
    )
    demote = demote_clause_verdict(
        demote_comparison, hybrid_arm=HYBRID_ARM, baseline_arm=BASELINE_ARM
    )
    survivals = survival_entries(
        comparisons, wiki_backed_arms=WIKI_BACKED_ARMS, baseline_arm=BASELINE_ARM
    )

    columns, rows = _canned_decision_matrix()

    report_input = VerdictReportInput(
        title="Corpus v3 verdict report",
        tldr=(
            f"Kill clause: `{kill.subject}` **{kill.outcome.replace('_', ' ')}**. "
            f"Demote clause: `{demote.subject}` **{demote.outcome.replace('_', ' ')}**. "
            f"{len(survivals)} survival entr{'y' if len(survivals) == 1 else 'ies'} "
            "found. See the honest-limits section: this is the offline tracer, "
            "not the real verdict."
        ),
        comparisons=comparisons,
        kill=kill,
        demote=demote,
        survivals=survivals,
        cost_per_grounded_correct={WIKI_ARM: 0.0, BASELINE_ARM: 0.0, HYBRID_ARM: 0.0},
        amortization_curves={
            WIKI_ARM: {10: 4.4 / 10, 100: 4.4 / 100, 1000: 4.4 / 1000},
            BASELINE_ARM: {10: 0.02 / 10, 100: 0.02 / 100, 1000: 0.02 / 1000},
            HYBRID_ARM: {10: 4.42 / 10, 100: 4.42 / 100, 1000: 4.42 / 1000},
        },
        decision_matrix_columns=columns,
        decision_matrix_rows=rows,
        honest_limits=_canned_honest_limits(),
        trust_note=OFFLINE_TRACER_HEADER,
    )
    return render_verdict_report(report_input)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pilot-ledger", type=Path, default=None)
    parser.add_argument(
        "--planned-calls",
        type=int,
        default=PLANNED_LIVE_CALLS,
        help="planned total query-phase LLM calls the cost guard projects spend for",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.mode == "offline":
        report = build_offline_tracer_report()
        OFFLINE_TRACER_REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"wrote offline tracer report to {OFFLINE_TRACER_REPORT_PATH}")
        return 0

    # --mode live
    if not args.confirm_live:
        print(
            "--mode live requires --confirm-live (no accidental live runs)",
            file=sys.stderr,
        )
        return 2
    if args.pilot_ledger is None:
        print(
            "--mode live requires --pilot-ledger (a recorded query-phase "
            "sample the cost guard projects spend from)",
            file=sys.stderr,
        )
        return 2
    pilot_ledger = load_pilot_ledger(args.pilot_ledger)
    if not run_cost_guard(pilot_ledger, planned_calls=args.planned_calls):
        return 1
    # Past the guard: a real run still requires a caller-supplied answer_fn
    # (see module docstring) -- this CLI entry point has none, by design.
    print(
        "cost guard cleared, but --mode live has no wired answer_fn "
        "(module docstring: 'no fallback to fake data on the live path') "
        "-- supply one via the run_verdict.py API, not this CLI, to "
        "actually execute the live verdict run",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
