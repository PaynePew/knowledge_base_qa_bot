"""Deep module per Ousterhout. Public surface: ``LiveAnsweringResult``,
``leak_source_ids_for_query``, ``build_axis_samples``, ``build_live_verdict_report``,
``run_live_answering``, ``run_live_verdict``.

The real corpus v3 live-run answering loop (issue #679): wires
``eval.corpus_v3.answer_fn.build_answer_fn`` past ``run_verdict.py``'s cost
guard into the four arms' real per-query answering -- the follow-up that
module's docstring named as an explicit, unwired gap ("the real per-arm
answer_fn integration is not wired yet"). ``run_verdict.main`` imports this
module lazily (inside its own ``--mode live`` branch) to avoid an import
cycle: this module imports ``run_verdict`` (for :data:`run_verdict.AxisSample`
/ :data:`run_verdict.WIKI_BACKED_ARMS`) and ``pilot_batch`` (for its
``ledger_to_json`` serialiser), and ``pilot_batch`` already imports
``run_verdict`` at its own top level.

Checkpointing (issue #679 AC 2, the #672 lesson -- a generation run was lost
TWICE to write-at-end when a network blip outlasted an hour of paid work) is
``checkpoint.py``'s own responsibility (extracted as a clear sub-module,
CODING_STANDARD §2.2): every answered ``(query_id, arm)`` pair is appended to
a JSONL file (:data:`run_verdict.LIVE_CHECKPOINT_PATH` by default) the
instant it is produced, carrying its own ledger deltas. A restart
(``checkpoint.load_checkpoint`` + ``checkpoint.replay_ledger``) skips every
pair already on disk and reconstructs the FULL cost ledger (not just the
current process's) from the checkpoint alone, so the mid-run spend abort and
the final actual-spend report stay correct across any number of crashes and
resumes.

Transient-error retry (:func:`_answer_with_retry`) mirrors
``generation.generate_queries``'s ``_invoke_with_retry`` / ``_is_transient``
bounded-backoff pattern, applied "at the answer loop" (issue #679 point 3)
rather than inside ``answer_fn`` itself: ``answer_fn`` is an opaque call into
each app's real ``query()`` surface (retrieval + synthesis + grounding-verify
in one call), so the retry unit here is the WHOLE per-(query, arm) answer,
exactly like ``pilot_batch.py``'s own mid-run cap check operates at the
per-answer boundary.

Content-axis scoring (issue #679 point 5's design note): ``grounding_pass``
needs each answer's retrieved-Section pool, which ``answer_fn.build_answer_fn``
now captures into ``AnswerRecord.retrieved_source_ids`` (a small additive
field on ``content_axes.AnswerRecord``, this issue's own change).
``correct_refusal`` needs the unanswerable stratum, read straight off
``query_schema.Query.scenario_stratum`` -- no new data. ``contradiction_leak``
needs a "leak ids" set per query; :func:`leak_source_ids_for_query` derives
it PURELY from data already committed (``build_corpus.ADVERSARIAL_GROUPS`` +
the query's own ``gold_section_ids``) rather than inventing a new gold
dataset: for a query drawn from a ``contradiction`` or ``version_evolution``
group, every OTHER raw Section id in that same group (the group's other
conflicting side, or a superseded version) is a leak candidate; a
``redundancy`` group's "other side" is a harmless near-duplicate of the SAME
fact, not a wrong one, so it contributes no leak ids at all.

Only nine ``AxisSample`` comparisons are computed -- (arm, axis) vs the
``rag`` baseline, for each of the three ADR-0045 axes and each of the three
wiki-backed arms -- the exact set ``kill_clause_verdict`` /
``demote_clause_verdict`` / ``survival_entries`` need (mirrors
``run_verdict.build_offline_tracer_report``'s own selection). Per-stratum
significance breakdowns are an explicit follow-up, not this issue's job (the
macro pooled comparison is what the offline tracer itself produces too).
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from markdown_kb.app.errors import LLMError

from . import pilot_batch, run_verdict, stacks
from .answer_fn import build_answer_fn
from .build_corpus import ADVERSARIAL_GROUPS, AdversarialGroup
from .checkpoint import (
    CheckpointRow,
    append_checkpoint_row,
    ledger_call_dicts,
    load_checkpoint,
    replay_ledger,
)
from .content_axes import (
    AnswerRecord,
    contradiction_leak,
    correct_refusal,
    cost_per_grounded_correct_answer,
    grounding_pass,
)
from .cost_guard import BUDGET_USD_CAP
from .query_schema import Query, load_queries
from .verdict_report import (
    VerdictReportInput,
    demote_clause_verdict,
    kill_clause_verdict,
    render_verdict_report,
    survival_entries,
)
from eval.cost_ledger.ledger import CostLedger

BASELINE_ARM = "rag"
HYBRID_ARM = "hybrid"

# Bounded backoff for transient call failures (mirrors
# ``generation.generate_queries._TRANSIENT_RETRY_SLEEPS`` exactly -- same
# rationale: long enough to ride out a Wi-Fi flap or brief API incident,
# short enough that a real outage still fails within minutes).
_TRANSIENT_RETRY_SLEEPS = (5.0, 15.0, 45.0, 90.0)


# ---------------------------------------------------------------------------
# Transient-error classification (mirrors generate_queries._is_transient)
# ---------------------------------------------------------------------------
def _is_transient(exc: BaseException) -> bool:
    """True when ``exc`` is worth retrying: the arms' own ``LLMError`` with
    its ``retryable`` flag set, an SDK connection/timeout error, or a
    retryable HTTP status (408/429/5xx). Mirrors
    ``generation.generate_queries._is_transient`` -- duplicated, not
    imported, per this package's own precedent for this exact helper
    (``pilot_batch.py`` mirroring ``run_comparison``'s isolation recipe
    rather than importing a private name across packages) -- PLUS the
    ``LLMError`` branch: unlike ``generate_queries`` (which invokes the LLM
    directly, so SDK exception types surface), this retry sits at the
    answer-loop boundary, where every arm's ``_call_llm_with_error_handling``
    has already mapped SDK errors into ``LLMError`` (ADR-0015: rate-limit /
    timeout -> ``retryable=True``, auth / unexpected -> ``retryable=False``).
    Observed live 2026-07-25: an OpenAI 429 (daily request quota) surfaced
    here as ``LLMError`` and crashed the run as "non-transient"."""
    if isinstance(exc, LLMError):
        return exc.retryable
    connection_types: list[type] = []
    status_types: list[type] = []
    for sdk_name in ("openai", "anthropic"):  # function-scope: keep SDKs internal
        try:
            sdk = importlib.import_module(sdk_name)
        except ImportError:
            continue
        connection_types.append(sdk.APIConnectionError)
        status_types.append(sdk.APIStatusError)
    if isinstance(exc, tuple(connection_types)):
        return True
    if isinstance(exc, tuple(status_types)):
        status = exc.status_code
        return status in (408, 429) or status >= 500
    return False


def _answer_with_retry(
    answer_fn: Callable[[str, str, list], AnswerRecord], query_id: str, arm: str
) -> AnswerRecord:
    """``answer_fn(query_id, arm, [])`` with bounded backoff over transient
    failures -- the retry unit is the WHOLE answer (module docstring: retry
    "at the answer loop", not inside ``answer_fn`` itself). Exhausting the
    schedule re-raises the original error (fail-fast, CODING_STANDARD §4.1)."""
    for sleep_s in _TRANSIENT_RETRY_SLEEPS:
        try:
            return answer_fn(query_id, arm, [])
        except Exception as exc:
            if not _is_transient(exc):
                raise
            time.sleep(sleep_s)
    return answer_fn(query_id, arm, [])


# ---------------------------------------------------------------------------
# The answering loop (issue #679 AC 1 / AC 2 / AC 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiveAnsweringResult:
    """Outcome of one :func:`run_live_answering` call. ``aborted`` is the
    mid-run spend-cap stop (issue #679 point 3) -- a normal, expected halt,
    not an error: the checkpoint written so far stays intact and a re-run
    resumes (and immediately re-aborts, since nothing changed) until a human
    raises the cap or stands the run down."""

    rows: list[CheckpointRow]
    aborted: bool
    abort_reason: str | None = None


def run_live_answering(
    queries: Sequence[Query],
    *,
    ledger: CostLedger,
    checkpoint_path: Path,
    answer_fn: Callable[[str, str, list], AnswerRecord],
    arms: Sequence[str] = tuple(stacks.ARM_REGISTRY),
    cap_usd: float = BUDGET_USD_CAP,
    already_done: set[tuple[str, str]] | None = None,
) -> LiveAnsweringResult:
    """Answer every ``(query, arm)`` pair not already in ``already_done``,
    checkpointing each one immediately, aborting if actual recorded spend
    (``ledger``, which the caller seeds via :func:`replay_ledger` on a
    resume) crosses ``cap_usd`` mid-run (mirrors ``pilot_batch.py``'s own
    guard, generalised from a single pilot batch to the full live run)."""
    done = set(already_done or ())
    new_rows: list[CheckpointRow] = []
    for arm in arms:
        for q in queries:
            if (q.query_id, arm) in done:
                continue
            calls_before = len(ledger.calls)
            record = _answer_with_retry(answer_fn, q.query_id, arm)
            new_calls = ledger.calls[calls_before:]
            row = CheckpointRow(
                query_id=q.query_id,
                arm=arm,
                answer_text=record.answer_text,
                cited_source_ids=record.cited_source_ids,
                retrieved_source_ids=record.retrieved_source_ids,
                ledger_calls=ledger_call_dicts(new_calls),
            )
            append_checkpoint_row(checkpoint_path, row)
            new_rows.append(row)
            done.add((q.query_id, arm))

            spent = ledger.totals(phase="query").usd
            if spent is not None and spent > cap_usd:
                return LiveAnsweringResult(
                    rows=new_rows,
                    aborted=True,
                    abort_reason=(
                        f"actual spend ${spent:.2f} crossed the ${cap_usd:.2f} "
                        f"cap mid-run at (query_id={q.query_id!r}, arm={arm!r}) "
                        "-- halting; the checkpoint is intact, mark the issue "
                        "ready-for-human rather than resuming blindly"
                    ),
                )
    return LiveAnsweringResult(rows=new_rows, aborted=False)


# ---------------------------------------------------------------------------
# Content-axis scoring from real AnswerRecords (issue #679 point 5)
# ---------------------------------------------------------------------------
def _group_lookup() -> dict[str, AdversarialGroup]:
    """Every raw Section id -> the ``AdversarialGroup`` (corpus/build_corpus.py's
    single source of truth) that owns it."""
    lookup: dict[str, AdversarialGroup] = {}
    for group in ADVERSARIAL_GROUPS:
        for section in group.sections:
            lookup[section.source_id] = group
    return lookup


def leak_source_ids_for_query(
    query: Query, group_lookup: Mapping[str, AdversarialGroup]
) -> frozenset[str]:
    """The ids ``content_axes.contradiction_leak`` must treat as a leak for
    ``query`` -- derived from ``build_corpus.ADVERSARIAL_GROUPS`` and the
    query's own ``gold_section_ids`` (module docstring: no new gold data).

    Empty when: the query has no gold ids (``unanswerable``), its gold ids
    don't resolve to exactly one group (unexpected corpus drift -- fail soft,
    no leak concept applies), or the group is ``redundancy`` (its "other
    side" restates the SAME fact in different wording, not a wrong one).
    Otherwise every raw Section id in the group OTHER than this query's own
    gold ids: the group's other conflicting side (``contradiction``) or a
    superseded version (``version_evolution``) -- exactly what
    ``content_axes.contradiction_leak``'s docstring names.
    """
    if not query.gold_section_ids:
        return frozenset()
    matched = [
        group_lookup[gid] for gid in query.gold_section_ids if gid in group_lookup
    ]
    if not matched:
        return frozenset()
    # AdversarialGroup carries list fields, so it is not hashable -- compare
    # by group_id (a plain string) instead of collecting instances into a set.
    group_ids = {g.group_id for g in matched}
    if len(group_ids) != 1:
        return frozenset()
    group = matched[0]
    if group.adversarial_class == "redundancy":
        return frozenset()
    all_ids = frozenset(s.source_id for s in group.sections)
    return all_ids - frozenset(query.gold_section_ids)


def build_axis_samples(
    queries: Sequence[Query],
    records_by_arm_query: Mapping[tuple[str, str], AnswerRecord],
    *,
    wiki_backed_arms: Sequence[str] = run_verdict.WIKI_BACKED_ARMS,
    baseline_arm: str = BASELINE_ARM,
) -> list[run_verdict.AxisSample]:
    """Build the nine ``AxisSample``s (module docstring) every ADR-0045
    clause needs: (arm, axis) vs ``baseline_arm``, for ``arm`` in
    ``wiki_backed_arms`` and axis in the three content axes.

    ``grounding_pass_rate`` is scored over ANSWERABLE queries only (this
    run's reading of ADR-0045: an unanswerable query's correct refusal is not
    a grounding failure, mirroring how ``correct_refusal`` is itself only
    defined on the unanswerable stratum). ``correct_refusal_rate`` is scored
    over the unanswerable stratum only (``content_axes.correct_refusal``'s
    own hard requirement). ``contradiction_leak_rate`` is scored over every
    query (a refusal never leaks; a query with no leak ids never leaks).
    """
    lookup = _group_lookup()
    samples: list[run_verdict.AxisSample] = []
    for arm in wiki_backed_arms:
        g_a: list[int] = []
        g_b: list[int] = []
        r_a: list[int] = []
        r_b: list[int] = []
        c_a: list[int] = []
        c_b: list[int] = []
        for q in queries:
            record_a = records_by_arm_query[(arm, q.query_id)]
            record_b = records_by_arm_query[(baseline_arm, q.query_id)]

            if q.scenario_stratum == "unanswerable":
                r_a.append(int(correct_refusal(record_a, is_unanswerable=True)))
                r_b.append(int(correct_refusal(record_b, is_unanswerable=True)))
            else:
                g_a.append(int(grounding_pass(record_a, record_a.retrieved_source_ids)))
                g_b.append(int(grounding_pass(record_b, record_b.retrieved_source_ids)))

            leak_ids = leak_source_ids_for_query(q, lookup)
            c_a.append(int(contradiction_leak(record_a, leak_ids)))
            c_b.append(int(contradiction_leak(record_b, leak_ids)))

        samples.append(
            run_verdict.AxisSample(
                axis="grounding_pass_rate",
                arm_a=arm,
                arm_b=baseline_arm,
                outcomes_a=g_a,
                outcomes_b=g_b,
            )
        )
        samples.append(
            run_verdict.AxisSample(
                axis="correct_refusal_rate",
                arm_a=arm,
                arm_b=baseline_arm,
                outcomes_a=r_a,
                outcomes_b=r_b,
            )
        )
        samples.append(
            run_verdict.AxisSample(
                axis="contradiction_leak_rate",
                arm_a=arm,
                arm_b=baseline_arm,
                outcomes_a=c_a,
                outcomes_b=c_b,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Report assembly (issue #679: canonical VERDICT.md, real data)
# ---------------------------------------------------------------------------
DENSE_OVER_WIKI_CAVEAT = (
    "`dense_over_wiki` has NO calibrated pre-LLM refusal gate "
    "(`answer_fn.dense_over_wiki_query`'s documented gap, ADR-0045 "
    "Prerequisite 2) -- its correct-refusal-rate numbers below reflect "
    "refusal via synthesis/grounding alone, not a gate calibrated against the "
    "negative set the way `wiki` and `rag` are. Issue #674's pilot measured "
    "7/8 negatives refused this way; the live numbers may differ. Accepted "
    "as a stated caveat rather than a blocking gap (issue #679 authorization, "
    "recorded 2026-07-24) -- read this axis's `dense_over_wiki` row with that "
    "caveat, not as an apples-to-apples gate comparison."
)


def _grounded_correct_counts(
    queries: Sequence[Query],
    records_by_arm_query: Mapping[tuple[str, str], AnswerRecord],
    arms: Sequence[str],
) -> dict[str, int]:
    """Per arm: answerable-query answers that are BOTH grounded and free of a
    contradiction leak -- this run's "grounded AND correct" proxy for
    ``content_axes.cost_per_grounded_correct_answer`` (PRD #654 user story
    21). No axis named "correctness" exists on its own; this combines the two
    axes that most directly bear on it."""
    lookup = _group_lookup()
    counts = dict.fromkeys(arms, 0)
    for arm in arms:
        for q in queries:
            if q.scenario_stratum == "unanswerable":
                continue
            record = records_by_arm_query[(arm, q.query_id)]
            leak_ids = leak_source_ids_for_query(q, lookup)
            if grounding_pass(
                record, record.retrieved_source_ids
            ) and not contradiction_leak(record, leak_ids):
                counts[arm] += 1
    return counts


def build_live_verdict_report(
    queries: Sequence[Query],
    records_by_arm_query: Mapping[tuple[str, str], AnswerRecord],
    ledger: CostLedger,
) -> str:
    """Assemble the canonical verdict report from real per-arm answers
    (issue #679 AC: "writes VERDICT.md ... per-clause kill/demote/survive per
    ADR-0045's Consequences"). No ``trust_note`` (unlike the offline tracer)
    -- CODING_STANDARD §6.6's canonical-name guarantee: only an answer_fn
    -backed run may write this file without a placeholder header."""
    samples = build_axis_samples(queries, records_by_arm_query)
    comparisons = [s.to_comparison() for s in samples]

    kill = kill_clause_verdict(comparisons, wiki_arm="wiki", baseline_arm=BASELINE_ARM)
    demote_comparison = next(
        c
        for c in comparisons
        if c.axis == "contradiction_leak_rate" and HYBRID_ARM in (c.arm_a, c.arm_b)
    )
    demote = demote_clause_verdict(
        demote_comparison, hybrid_arm=HYBRID_ARM, baseline_arm=BASELINE_ARM
    )
    survivals = survival_entries(
        comparisons,
        wiki_backed_arms=run_verdict.WIKI_BACKED_ARMS,
        baseline_arm=BASELINE_ARM,
    )

    arms = list(run_verdict.WIKI_BACKED_ARMS) + [BASELINE_ARM]
    grounded_correct = _grounded_correct_counts(queries, records_by_arm_query, arms)
    totals = ledger.totals(phase="query")
    per_arm_usd = {arm: ledger.totals(stack=arm, phase="query").usd for arm in arms}
    cost_per_grounded_correct = {
        arm: cost_per_grounded_correct_answer(
            per_arm_usd[arm] or 0.0, grounded_correct[arm]
        )
        for arm in arms
    }
    # corpus v3's build phase is hand-authored (build_corpus.py's own
    # docstring: "no LLM synthesis call is made to produce it") -- $0.00 is
    # the real recorded build cost, not a placeholder.
    amortization_curves = {arm: {n: 0.0 for n in (10, 100, 1000)} for arm in arms}

    columns, rows = run_verdict.canned_decision_matrix()
    honest_limits = [DENSE_OVER_WIKI_CAVEAT, *run_verdict.canned_honest_limits()]

    spend = f"${totals.usd:.4f}" if totals.usd is not None else "unknown"
    report_input = VerdictReportInput(
        title="Corpus v3 verdict report",
        tldr=(
            f"Kill clause: `{kill.subject}` **{kill.outcome.replace('_', ' ')}**. "
            f"Demote clause: `{demote.subject}` **{demote.outcome.replace('_', ' ')}**. "
            f"{len(survivals)} survival entr{'y' if len(survivals) == 1 else 'ies'} "
            f"found. Real live run over {len(queries)} English queries x 4 arms; "
            f"actual query-phase spend {spend} ({totals.calls} LLM calls). See "
            "the honest-limits section for the dense_over_wiki correct-refusal "
            "caveat."
        ),
        comparisons=comparisons,
        kill=kill,
        demote=demote,
        survivals=survivals,
        cost_per_grounded_correct=cost_per_grounded_correct,
        amortization_curves=amortization_curves,
        decision_matrix_columns=columns,
        decision_matrix_rows=rows,
        honest_limits=honest_limits,
    )
    return render_verdict_report(report_input)


# ---------------------------------------------------------------------------
# Orchestration (issue #679: the top-level entry point run_verdict.main calls)
# ---------------------------------------------------------------------------
def run_live_verdict(
    *,
    queries_path: Path,
    checkpoint_path: Path,
    ledger_out_path: Path,
    report_out_path: Path,
    cap_usd: float = BUDGET_USD_CAP,
) -> int:
    """The real live-run orchestration: load the en query subset, resume from
    any existing checkpoint, production-isolate and index all four arms, and
    answer every remaining ``(query, arm)`` pair through the real
    ``answer_fn`` seam. Writes the canonical verdict report and the run's
    full cost ledger on success. Returns a process exit code (0 success, 1 on
    a missing API key / empty query set / mid-run spend-cap abort)."""
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY absent -- the live run answers through real arm "
            "query() surfaces (real synthesis, verifier, and embedding "
            "calls). Nothing was run.",
            file=sys.stderr,
        )
        return 1

    queries = [q for q in load_queries(queries_path) if q.language == "en"]
    if not queries:
        print(
            f"no English queries found in {queries_path} -- nothing to run.",
            file=sys.stderr,
        )
        return 1

    existing_rows = load_checkpoint(checkpoint_path)
    ledger = replay_ledger(existing_rows)
    done = {(r.query_id, r.arm) for r in existing_rows}
    prior_spend = ledger.totals(phase="query").usd
    print(
        f"resuming from checkpoint: {len(existing_rows)} answers already "
        f"recorded ({len(done)} unique pairs), "
        f"${prior_spend:.4f} spent so far"
        if prior_spend is not None
        else f"resuming from checkpoint: {len(existing_rows)} answers already recorded",
        file=sys.stderr,
    )

    iso = stacks.isolate_production_paths(prefix="corpus_v3_live_")
    print(f"production isolation: {iso}", file=sys.stderr)
    stacks.index_wiki_corpus()
    stacks.index_dense_over_wiki()
    stacks.index_docs_corpus()

    answer_fn = build_answer_fn({q.query_id: q.text for q in queries}, ledger=ledger)

    result = run_live_answering(
        queries,
        ledger=ledger,
        checkpoint_path=checkpoint_path,
        answer_fn=answer_fn,
        already_done=done,
        cap_usd=cap_usd,
    )
    if result.aborted:
        print(f"live run: {result.abort_reason}", file=sys.stderr)
        return 1

    all_rows = load_checkpoint(checkpoint_path)  # durable source of truth
    records_by_arm_query = {(r.arm, r.query_id): r.to_answer_record() for r in all_rows}

    report = build_live_verdict_report(queries, records_by_arm_query, ledger)
    report_out_path.write_text(report, encoding="utf-8")
    ledger_out_path.write_text(pilot_batch.ledger_to_json(ledger), encoding="utf-8")

    totals = ledger.totals(phase="query")
    spend = f"${totals.usd:.4f}" if totals.usd is not None else "unknown"
    print(
        f"live run complete: {len(all_rows)} answers, {totals.calls} LLM "
        f"calls, actual spend {spend} (vs the $4.05 pilot projection) -- "
        f"wrote {report_out_path.name} and {ledger_out_path.name}",
        file=sys.stderr,
    )
    return 0
