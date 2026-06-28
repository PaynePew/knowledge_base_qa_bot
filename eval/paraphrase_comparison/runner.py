"""Deep module per Ousterhout. Public surface: ``run_comparison``, ``build_test_cases``, ``StackScores``, ``REPORT_PATH``.

In-process DeepEval comparison runner for the Phase 8 retrieval comparison
(CONTEXT.md § Phase 8 > Retrieval Stack, PRD #100).

Drives both Retrieval Stacks through the same Paraphrase set in one process
(no HTTP), scores each retrieval with the deterministic C5c ``HitRateAtK``
metric (hit_rate@k AND MRR), and renders the full ``report.md`` deliverable plus
the matplotlib charts. The report separates **Core** Paraphrase Types from
**Structural probe** types (PRD #100 — no naive cross-type aggregate; a Core
macro-average WITH a caveat is the only aggregate, probes are framed as
expected-limit confirmation), records the (offline) generation cost honestly,
and carries the six+1 honest-limitation disclosures.

Production isolation (PRD #100 acceptance): index building points markdown_kb
``SOURCE_DIRS`` and vector_rag ``DOCS_DIR`` at the eval fixtures, and redirects
markdown_kb ``INDEX_PATH`` / ``WIKI_DIR`` (and disables its wiki-index side
write) to a tmp directory, so production ``wiki/`` / ``docs/`` / ``.kb/`` are
never read or written.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import hybrid_kb.app.dense_index as hk_dense
import hybrid_kb.app.logger as hk_logger
import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
from deepeval.test_case import LLMTestCase

from . import charts, stacks
from .loader import load_metadata, load_paraphrases, write_text_atomic
from .metric import DEFAULT_K, HitRateAtK, hit_at_k, reciprocal_rank_at_k
from .models import (
    CORE_PARAPHRASE_TYPES,
    PROBE_PARAPHRASE_TYPES,
    Paraphrase,
    RetrievedItem,
)
from .spotcheck import (
    DEFAULT_CONTROL_SAMPLE_SIZE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MARGINAL_THRESHOLD,
    ZONES,
    SpotcheckResult,
    run_spotcheck,
)
from .statistics import (
    CochranQResult,
    TypeStatResult,
    cochran_q,
    compute_type_stats,
    holm_correct,
    mcnemar_exact_p,
)

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "report.md"

# A Retrieval Stack's retrieval entry point.
StackRetrieval = Callable[[str, int], list[RetrievedItem]]

# ---------------------------------------------------------------------------
# Three-arm cutoff-sweep methodology (Phase 13, ADR-0018 / #316)
# ---------------------------------------------------------------------------
# The upgraded methodology that SUPERSEDES the Phase 8 single-cutoff (hit@3)
# report: each arm overfetches a deep candidate pool ONCE per Paraphrase (at
# ``DEEP_POOL``), and hit-rate is read off that one pool at every cutoff in
# ``SWEEP_CUTOFFS`` — never re-retrieving per cutoff. ``PRIMARY_CUTOFF`` is the
# cutoff the per-type table, charts, the legacy A-vs-B McNemar, and the three-arm
# omnibus are reported at (kept at 3 so the headline stays comparable to Phase 8).
SWEEP_CUTOFFS: tuple[int, ...] = (1, 3, 5, 10)
PRIMARY_CUTOFF = 3
DEEP_POOL = max(SWEEP_CUTOFFS)

# The three post-hoc pairwise comparisons run after a significant Cochran's Q,
# in a fixed order so the Holm correction and the report rows are deterministic.
_POSTHOC_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("Wiki ↔ RAG", "Stack A", "Stack B"),
    ("Hybrid ↔ Wiki", "Stack C", "Stack A"),
    ("Hybrid ↔ RAG", "Stack C", "Stack B"),
)

# Expected winner per Paraphrase Type — the architectural prediction the
# comparison tests (PRD #100, roadmap prep note #3). "B" = the rewrite stresses
# Stack B's structural advantage (semantic embedding); "A" = it plays to Stack
# A's keyword/synthesis strength; "either" = no strong directional prior. The
# report renders this verbatim in the ``expected`` column so a reader can read
# each measured Δ against the stated hypothesis.
_EXPECTED_WINNER: dict[str, str] = {
    "synonym_swap": "B (semantic)",
    "word_reorder": "either (bag-of-words robust)",
    "verbosity_expansion": "A (extra keywords aid BM25)",
    "specificity_narrowing": "B (sub-fact targeting)",
    "implicit_reference": "B (semantic)",
    "typo_fatfinger": "A (BM25 token tolerance) — probe",
    "industry_jargon": "B (semantic) — probe",
}


@dataclass(frozen=True)
class JudgeConfig:
    """Opt-in L2 Spot-check configuration threaded from the CLI ``--judge*`` flags.

    Present (non-None) only when ``--judge`` was passed; ``run_comparison`` then
    runs the cross-family Claude judge over the ambiguous subset. Carries the
    judge model and the three tunable zone knobs (PRD #100 user story 20).
    """

    judge_model: str = DEFAULT_JUDGE_MODEL
    zones: tuple[str, ...] = ZONES
    marginal_threshold: int = DEFAULT_MARGINAL_THRESHOLD
    control_sample_size: int = DEFAULT_CONTROL_SAMPLE_SIZE


@dataclass(frozen=True)
class StackScores:
    """Per-Paraphrase-Type hit_rate@k and MRR for one Retrieval Stack.

    ``by_type`` is the per-type hit_rate@k (mean of per-Paraphrase 1.0/0.0 hits);
    ``mrr_by_type`` is the per-type MRR (mean of per-Paraphrase reciprocal ranks
    of the first top-k hit). ``n_by_type`` is the Paraphrase count per type, so
    the report can render the ``n`` column and a Paraphrase-weighted Core
    macro-average (PRD #100).
    """

    stack: str
    k: int
    by_type: dict[str, float]
    mrr_by_type: dict[str, float] = field(default_factory=dict)
    n_by_type: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreStats:
    """Per-Core-Type McNemar / Wilson CI / Holm statistics for the report.

    ``by_type`` maps each Core Paraphrase Type to its ``TypeStatResult``
    (raw McNemar p, Wilson CIs, discordant-pair counts).  ``holm_p_by_type``
    holds the Holm-corrected p-values in the same type order as
    ``CORE_PARAPHRASE_TYPES`` — these are the reportable adjusted values.
    """

    by_type: dict[str, TypeStatResult]
    holm_p_by_type: dict[str, float]


@dataclass(frozen=True)
class SweepScores:
    """Per-arm hit_rate + MRR across the cutoff sweep, per Paraphrase Type.

    ``hit_by_cutoff[cutoff][ptype]`` is the type's hit_rate@cutoff and
    ``mrr_by_cutoff[cutoff][ptype]`` its MRR@cutoff, both read off ONE deep
    retrieval pool (no per-cutoff re-retrieval). ``n_by_type`` is the Paraphrase
    count per type so the report can weight a Core macro-average.
    """

    stack: str
    cutoffs: tuple[int, ...]
    hit_by_cutoff: dict[int, dict[str, float]]
    mrr_by_cutoff: dict[int, dict[str, float]]
    n_by_type: dict[str, int]


@dataclass(frozen=True)
class ThreeArmStats:
    """Cochran's Q omnibus + post-hoc pairwise McNemar over the pooled Core set.

    ``cutoff`` — the cutoff the binary hit outcomes were taken at (PRIMARY_CUTOFF).
    ``cochran`` — the three-way omnibus (Q, df=2, p). ``posthoc_significant`` is
    the gate: when ``cochran.p_value < 0.05`` the three pairwise McNemar tests
    (Wiki↔RAG, Hybrid↔Wiki, Hybrid↔RAG) are warranted and Holm-corrected; the
    parallel ``pair_*`` tuples carry each pair's label, discordant (b, c), raw
    McNemar p, and Holm-adjusted p in ``_POSTHOC_PAIRS`` order.
    """

    cutoff: int
    cochran: CochranQResult
    posthoc_significant: bool
    pair_labels: tuple[str, ...]
    pair_bc: tuple[tuple[int, int], ...]
    pair_raw_p: tuple[float, ...]
    pair_holm_p: tuple[float, ...]


@dataclass(frozen=True)
class ThreeArmScoring:
    """The full three-arm scoring bundle the report renders.

    ``primary`` — (Stack A, B, C) ``StackScores`` at PRIMARY_CUTOFF (per-type
    table, charts, legacy A-vs-B McNemar). ``sweep`` — the parallel
    ``SweepScores`` triple across SWEEP_CUTOFFS. ``core_stats`` — the legacy
    per-Core-type A-vs-B McNemar / Wilson / Holm. ``three_arm`` — the new
    Cochran's Q omnibus + post-hoc.
    """

    primary: tuple[StackScores, StackScores, StackScores]
    sweep: tuple[SweepScores, SweepScores, SweepScores]
    core_stats: CoreStats
    three_arm: ThreeArmStats


# ---------------------------------------------------------------------------
# Test-case assembly + scoring (the DeepEval seam)
# ---------------------------------------------------------------------------
def build_test_cases(
    paraphrases: list[Paraphrase],
    retrieve: StackRetrieval,
    k: int = DEFAULT_K,
) -> list[tuple[Paraphrase, LLMTestCase]]:
    """Run ``retrieve`` for each Paraphrase and pack a DeepEval LLMTestCase.

    The retrieved items and the Paraphrase's Key Tokens travel in ``metadata``
    so the deterministic C5c metric reads them without re-running retrieval.
    ``retrieval_context`` carries the source ids for DeepEval's own display.
    """
    cases: list[tuple[Paraphrase, LLMTestCase]] = []
    for para in paraphrases:
        items = retrieve(para.text, k)
        case = LLMTestCase(
            input=para.text,
            actual_output="",  # retrieval-only comparison; no generated answer
            expected_output=para.gold_docs_section_id,
            retrieval_context=[it.source_section_id for it in items] or ["<none>"],
            metadata={
                "retrieved_items": items,
                "key_tokens": sorted(para.key_tokens),
                "paraphrase_type": para.paraphrase_type,
            },
        )
        cases.append((para, case))
    return cases


def score_stack(
    stack_name: str,
    paraphrases: list[Paraphrase],
    retrieve: StackRetrieval,
    k: int = DEFAULT_K,
) -> StackScores:
    """Score one Stack over all Paraphrases, aggregating hit_rate@k AND MRR per type."""
    metric = HitRateAtK(k=k)
    per_type_hits: dict[str, list[float]] = defaultdict(list)
    per_type_rr: dict[str, list[float]] = defaultdict(list)
    for para, case in build_test_cases(paraphrases, retrieve, k):
        metric.measure(case)
        per_type_hits[para.paraphrase_type].append(metric.score)
        per_type_rr[para.paraphrase_type].append(metric.reciprocal_rank)
    by_type = {
        ptype: sum(scores) / len(scores) for ptype, scores in per_type_hits.items()
    }
    mrr_by_type = {ptype: sum(rrs) / len(rrs) for ptype, rrs in per_type_rr.items()}
    n_by_type = {ptype: len(scores) for ptype, scores in per_type_hits.items()}
    return StackScores(
        stack=stack_name,
        k=k,
        by_type=by_type,
        mrr_by_type=mrr_by_type,
        n_by_type=n_by_type,
    )


# ---------------------------------------------------------------------------
# Paired scoring — both stacks in one pass for McNemar
# ---------------------------------------------------------------------------


def score_paired(
    paraphrases: list[Paraphrase],
    retrieve_a: StackRetrieval,
    retrieve_b: StackRetrieval,
    k: int = DEFAULT_K,
) -> tuple[StackScores, StackScores, CoreStats]:
    """Score both Stacks in one pass and compute per-Core-Type McNemar statistics.

    Running both stacks against the same Paraphrases in one pass ensures the
    per-Paraphrase (hit_a, hit_b) pairs are aligned — essential for McNemar.
    Returns ``(stack_a_scores, stack_b_scores, core_stats)``.
    """
    metric = HitRateAtK(k=k)
    per_type_hits_a: dict[str, list[float]] = defaultdict(list)
    per_type_hits_b: dict[str, list[float]] = defaultdict(list)
    per_type_rr_a: dict[str, list[float]] = defaultdict(list)
    per_type_rr_b: dict[str, list[float]] = defaultdict(list)
    # Raw paired binary outcomes for McNemar (0/1 per paraphrase per type)
    paired_hits_a: dict[str, list[int]] = defaultdict(list)
    paired_hits_b: dict[str, list[int]] = defaultdict(list)

    for para in paraphrases:
        ptype = para.paraphrase_type
        # Score Stack A
        items_a = retrieve_a(para.text, k)
        case_a = LLMTestCase(
            input=para.text,
            actual_output="",
            expected_output=para.gold_docs_section_id,
            retrieval_context=[it.source_section_id for it in items_a] or ["<none>"],
            metadata={
                "retrieved_items": items_a,
                "key_tokens": sorted(para.key_tokens),
                "paraphrase_type": ptype,
            },
        )
        metric.measure(case_a)
        hit_a = int(metric.score)
        per_type_hits_a[ptype].append(metric.score)
        per_type_rr_a[ptype].append(metric.reciprocal_rank)
        # Score Stack B
        items_b = retrieve_b(para.text, k)
        case_b = LLMTestCase(
            input=para.text,
            actual_output="",
            expected_output=para.gold_docs_section_id,
            retrieval_context=[it.source_section_id for it in items_b] or ["<none>"],
            metadata={
                "retrieved_items": items_b,
                "key_tokens": sorted(para.key_tokens),
                "paraphrase_type": ptype,
            },
        )
        metric.measure(case_b)
        hit_b = int(metric.score)
        per_type_hits_b[ptype].append(metric.score)
        per_type_rr_b[ptype].append(metric.reciprocal_rank)
        # Accumulate paired binary outcomes for McNemar (Core types only)
        if ptype in CORE_PARAPHRASE_TYPES:
            paired_hits_a[ptype].append(hit_a)
            paired_hits_b[ptype].append(hit_b)

    def _to_scores(
        stack_name: str,
        hits: dict[str, list[float]],
        rrs: dict[str, list[float]],
    ) -> StackScores:
        return StackScores(
            stack=stack_name,
            k=k,
            by_type={t: sum(s) / len(s) for t, s in hits.items()},
            mrr_by_type={t: sum(r) / len(r) for t, r in rrs.items()},
            n_by_type={t: len(s) for t, s in hits.items()},
        )

    stack_a = _to_scores("Stack A", per_type_hits_a, per_type_rr_a)
    stack_b = _to_scores("Stack B", per_type_hits_b, per_type_rr_b)
    core_stats = _compute_core_stats(paired_hits_a, paired_hits_b)
    return stack_a, stack_b, core_stats


def _compute_core_stats(
    paired_hits_a: dict[str, list[int]],
    paired_hits_b: dict[str, list[int]],
) -> CoreStats:
    """Compute per-Core-Type TypeStatResults and apply Holm correction.

    Only Core Paraphrase Types contribute to the Holm correction (five tests);
    Probe types are excluded per AC 3/4 of issue #140.
    """
    core_types = [t for t in CORE_PARAPHRASE_TYPES if t in paired_hits_a]
    by_type: dict[str, TypeStatResult] = {}
    for ptype in core_types:
        by_type[ptype] = compute_type_stats(paired_hits_a[ptype], paired_hits_b[ptype])
    raw_ps = [by_type[t].mcnemar_p for t in core_types]
    corrected = holm_correct(raw_ps)
    holm_p_by_type = {t: corrected[i] for i, t in enumerate(core_types)}
    return CoreStats(by_type=by_type, holm_p_by_type=holm_p_by_type)


# ---------------------------------------------------------------------------
# Three-arm cutoff-sweep scoring (Phase 13 methodology — AC1 / AC2 / AC3)
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_three_arms(
    paraphrases: list[Paraphrase],
    retrieve_a: StackRetrieval,
    retrieve_b: StackRetrieval,
    retrieve_c: StackRetrieval,
    *,
    cutoffs: tuple[int, ...] = SWEEP_CUTOFFS,
    primary_cutoff: int = PRIMARY_CUTOFF,
    deep_pool: int = DEEP_POOL,
) -> ThreeArmScoring:
    """Score all three arms over one deep retrieval pass and the cutoff sweep.

    Each arm retrieves ONCE per Paraphrase at ``deep_pool`` (the max cutoff); the
    hit_rate + MRR at every cutoff in ``cutoffs`` are read off that single pool
    (the AC2 decoupling — candidate depth is never re-retrieved per cutoff). The
    three arms are scored in the SAME pass so the per-Paraphrase binary outcomes
    at ``primary_cutoff`` are aligned across arms — required for the paired
    Cochran's Q omnibus and the post-hoc pairwise McNemar.

    Returns a :class:`ThreeArmScoring`: the primary-cutoff ``StackScores`` triple
    (per-type table + charts), the ``SweepScores`` triple, the legacy A-vs-B
    per-Core-type ``CoreStats``, and the new three-arm ``ThreeArmStats``.
    """
    arms = (("Stack A", retrieve_a), ("Stack B", retrieve_b), ("Stack C", retrieve_c))
    # hit_acc[stack][cutoff][ptype] -> list of per-Paraphrase hit (1.0/0.0)
    hit_acc = {name: {c: defaultdict(list) for c in cutoffs} for name, _ in arms}
    rr_acc = {name: {c: defaultdict(list) for c in cutoffs} for name, _ in arms}
    n_by_type: dict[str, int] = defaultdict(int)
    # paired_at_primary[stack][ptype] -> aligned 0/1 hit at primary cutoff (Core only)
    paired_at_primary: dict[str, dict[str, list[int]]] = {
        name: defaultdict(list) for name, _ in arms
    }

    for para in paraphrases:
        ptype = para.paraphrase_type
        gold = para.gold_docs_section_id
        key_tokens = sorted(para.key_tokens)
        n_by_type[ptype] += 1
        for name, retrieve in arms:
            items = retrieve(para.text, deep_pool)  # one deep pass per arm
            for cutoff in cutoffs:
                hit_acc[name][cutoff][ptype].append(
                    hit_at_k(items, gold, key_tokens, k=cutoff)
                )
                rr_acc[name][cutoff][ptype].append(
                    reciprocal_rank_at_k(items, gold, key_tokens, k=cutoff)
                )
            if ptype in CORE_PARAPHRASE_TYPES:
                hit_primary = hit_at_k(items, gold, key_tokens, k=primary_cutoff)
                paired_at_primary[name][ptype].append(int(hit_primary))

    def _sweep(name: str) -> SweepScores:
        return SweepScores(
            stack=name,
            cutoffs=cutoffs,
            hit_by_cutoff={
                c: {t: _mean(v) for t, v in hit_acc[name][c].items()} for c in cutoffs
            },
            mrr_by_cutoff={
                c: {t: _mean(v) for t, v in rr_acc[name][c].items()} for c in cutoffs
            },
            n_by_type=dict(n_by_type),
        )

    def _primary(name: str) -> StackScores:
        return StackScores(
            stack=name,
            k=primary_cutoff,
            by_type={t: _mean(v) for t, v in hit_acc[name][primary_cutoff].items()},
            mrr_by_type={t: _mean(v) for t, v in rr_acc[name][primary_cutoff].items()},
            n_by_type=dict(n_by_type),
        )

    primary = (_primary("Stack A"), _primary("Stack B"), _primary("Stack C"))
    sweep = (_sweep("Stack A"), _sweep("Stack B"), _sweep("Stack C"))
    core_stats = _compute_core_stats(
        paired_at_primary["Stack A"], paired_at_primary["Stack B"]
    )
    three_arm = _compute_three_arm_stats(paired_at_primary, primary_cutoff)
    return ThreeArmScoring(
        primary=primary, sweep=sweep, core_stats=core_stats, three_arm=three_arm
    )


def _compute_three_arm_stats(
    paired_at_primary: dict[str, dict[str, list[int]]],
    primary_cutoff: int,
) -> ThreeArmStats:
    """Pooled Cochran's Q over the Core set, then Holm-corrected post-hoc McNemar.

    The Core Paraphrases are pooled across types (in CORE_PARAPHRASE_TYPES order,
    identically for every arm so the three hit vectors stay index-aligned) into a
    single paired binary sample per arm at ``primary_cutoff``. Cochran's Q is the
    omnibus; the three pairwise McNemar tests (``_POSTHOC_PAIRS``) are computed and
    Holm-corrected, and ``posthoc_significant`` records whether Q opened the gate.
    """
    core_types = [t for t in CORE_PARAPHRASE_TYPES if t in paired_at_primary["Stack A"]]

    def _pool(name: str) -> list[int]:
        return [hit for t in core_types for hit in paired_at_primary[name][t]]

    hits = {name: _pool(name) for name in ("Stack A", "Stack B", "Stack C")}
    cochran = cochran_q(hits["Stack A"], hits["Stack B"], hits["Stack C"])

    pair_labels: list[str] = []
    pair_bc: list[tuple[int, int]] = []
    pair_raw_p: list[float] = []
    for label, left, right in _POSTHOC_PAIRS:
        x, y = hits[left], hits[right]
        b = sum(1 for xi, yi in zip(x, y) if xi == 1 and yi == 0)
        c = sum(1 for xi, yi in zip(x, y) if xi == 0 and yi == 1)
        pair_labels.append(label)
        pair_bc.append((b, c))
        pair_raw_p.append(mcnemar_exact_p(b, c))
    pair_holm_p = holm_correct(pair_raw_p)

    return ThreeArmStats(
        cutoff=primary_cutoff,
        cochran=cochran,
        posthoc_significant=cochran.p_value < 0.05,
        pair_labels=tuple(pair_labels),
        pair_bc=tuple(pair_bc),
        pair_raw_p=tuple(pair_raw_p),
        pair_holm_p=tuple(pair_holm_p),
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_report(
    stack_a: StackScores,
    stack_b: StackScores,
    embedding_mode: str = "real",
    metadata: dict | None = None,
    chart_files: list[Path] | None = None,
    spotcheck: SpotcheckResult | None = None,
    core_stats: CoreStats | None = None,
    stack_c: StackScores | None = None,
    sweep: tuple[SweepScores, SweepScores, SweepScores] | None = None,
    three_arm: ThreeArmStats | None = None,
) -> str:
    """Render the full ``report.md`` deliverable for the retrieval comparison.

    Structure (PRD #100 + #140 + Phase 13 #316): TL;DR → Experiment Setup (incl.
    cost log + the methodology delta that supersedes Phase 8) → Cutoff Sweep
    (hit@{1,3,5,10} + MRR for all arms) → Core Comparison (per-type hit_rate@k +
    MRR + Δ + expected + n, then the three-arm Cochran's Q omnibus + post-hoc
    pairwise McNemar, then the legacy A-vs-B McNemar / Wilson CI / Holm, then a
    Core macro-average WITH a caveat) → Structural Probes (separate table) →
    Spot-check Validation (only when ``--judge`` was run) → Limitations → Interview
    Talking Points appendix.

    ``embedding_mode`` annotates how the dense vectors were produced ("real"
    OpenAI embeddings vs a "fake" deterministic offline stand-in) so a reader
    never mistakes an offline tracer number for a real-embedding result.
    ``metadata`` is the ``queries.yaml`` metadata block; ``chart_files`` are the
    rendered PNGs to embed. ``spotcheck`` is the primitive-only L2 result.
    ``core_stats`` carries the legacy A-vs-B McNemar / Wilson CI / Holm.

    ``stack_c`` / ``sweep`` / ``three_arm`` are the Phase 13 three-arm additions:
    when all three are present the report renders the Hybrid arm columns, the
    cutoff-sweep table, and the Cochran's Q omnibus + post-hoc section. When they
    are ``None`` the report falls back to the two-arm shape (a legacy caller).
    """
    metadata = metadata or {}
    chart_files = chart_files or []
    k = stack_a.k
    offline = embedding_mode == "fake"
    judged = spotcheck is not None
    three_arms = stack_c is not None

    parts = [
        _render_header(offline, three_arms),
        _render_tldr(stack_a, stack_b, k, offline, stack_c),
        _render_setup(embedding_mode, metadata, k, spotcheck, three_arms),
        _render_cutoff_sweep(sweep),
        _render_family_section(
            "Core Comparison",
            CORE_PARAPHRASE_TYPES,
            stack_a,
            stack_b,
            k,
            chart_files,
            with_macro_average=True,
            core_stats=core_stats,
            stack_c=stack_c,
            three_arm=three_arm,
        ),
        _render_family_section(
            "Structural Probes",
            PROBE_PARAPHRASE_TYPES,
            stack_a,
            stack_b,
            k,
            chart_files,
            with_macro_average=False,
            stack_c=stack_c,
        ),
        _render_spotcheck(spotcheck),
        _render_limitations(offline, judged),
        _render_talking_points(),
    ]
    return "\n\n".join(p for p in parts if p) + "\n"


def _render_header(offline: bool, three_arms: bool = False) -> str:
    banner = (
        "\n\n> ⚠️ **OFFLINE TRACER NUMBERS.** Every score below was produced WITHOUT "
        "`OPENAI_API_KEY`: the Core Paraphrases are hand-authored offline stand-ins "
        "(not gpt-4o output) and the dense arms (Stack B's Chunks and Stack C's wiki "
        "Sections) come from a deterministic hash/token-overlap stand-in, NOT real "
        "`text-embedding-3-small` embeddings. These numbers exercise the pipeline "
        "end-to-end but are **not the real experiment**. Re-run with `OPENAI_API_KEY` "
        "(and a regenerated `queries.yaml`) for headline figures.\n"
        if offline
        else ""
    )
    third_arm = (
        " A third arm — **Stack C** (Hybrid: BM25 + a dense-over-wiki Section index, "
        "fused by Reciprocal Rank Fusion over the SAME curated `wiki/` corpus) — is "
        "compared alongside, under an upgraded methodology (deep candidate overfetch, "
        "a cutoff sweep, and a three-way Cochran's Q omnibus) that **supersedes the "
        "Phase 8 single-cutoff hit@3 report**."
        if three_arms
        else ""
    )
    return (
        "# Paraphrase Comparison Report\n\n"
        "Phase 8 retrieval comparison (PRD #100): does Karpathy's curated-Wiki layer "
        "(**Stack A** — LLM-synthesised `wiki/` + BM25) out-retrieve a traditional "
        "Vector RAG pipeline (**Stack B** — chunk + embed + FAISS) fed the **same** raw "
        "corpus? Scored at the retrieval layer only by the deterministic C5c hit "
        "metric (source-match AND dual-side Key-Token overlap)." + third_arm + banner
    )


def _render_tldr(
    stack_a: StackScores,
    stack_b: StackScores,
    k: int,
    offline: bool,
    stack_c: StackScores | None = None,
) -> str:
    core_a = _macro_average(stack_a.by_type, CORE_PARAPHRASE_TYPES)
    core_b = _macro_average(stack_b.by_type, CORE_PARAPHRASE_TYPES)
    qualifier = "offline tracer" if offline else "L1 (deterministic)"
    n_sources = len(list(stacks.FIXTURES["corpus"].glob("*.md")))
    arm_c = ""
    if stack_c is not None:
        core_c = _macro_average(stack_c.by_type, CORE_PARAPHRASE_TYPES)
        arm_c = f" vs **Stack C (Hybrid) {core_c:.3f}**"
    return (
        "## TL;DR\n\n"
        f"On this {n_sources}-Source Acme Shop corpus, the Core macro-average hit_rate@{k} is "
        f"**Stack A {core_a:.3f}** vs **Stack B {core_b:.3f}**{arm_c} "
        f"({qualifier} numbers). "
        "The per-type breakdown and the cutoff sweep below are the real signal — the "
        "macro-average is a researcher-chosen type mix and is reported only with the "
        "caveat below. Structural probes are reported separately and framed as "
        "expected-limit confirmation, never folded into a headline number."
    )


def _render_setup(
    embedding_mode: str,
    metadata: dict,
    k: int,
    spotcheck: SpotcheckResult | None = None,
    three_arms: bool = False,
) -> str:
    cost = metadata.get("cost_usd", "n/a")
    generator = metadata.get("generator_model", "gpt-4o")
    critic = metadata.get("critic_model", "gpt-4o-mini")
    seed = metadata.get("seed", "n/a")
    snapshot = metadata.get("corpus_snapshot_git_sha", "n/a")
    # Counts are derived from the actual corpus + query set so the narrative never
    # drifts from the data after a regeneration (issue #145).
    paras = load_paraphrases()
    n_sources = len(list(stacks.FIXTURES["corpus"].glob("*.md")))
    core_per = {
        t: sum(p.paraphrase_type == t for p in paras) for t in CORE_PARAPHRASE_TYPES
    }
    probe_per = {
        t: sum(p.paraphrase_type == t for p in paras) for t in PROBE_PARAPHRASE_TYPES
    }
    n_core, n_probes = sum(core_per.values()), sum(probe_per.values())
    core_sz, probe_sz = sorted(set(core_per.values())), sorted(set(probe_per.values()))
    core_mult = (
        f"× {core_sz[0]}" if len(core_sz) == 1 else f"× {core_sz[0]}–{core_sz[-1]}"
    )
    probe_mult = (
        f"× {probe_sz[0]}" if len(probe_sz) == 1 else f"× {probe_sz[0]}–{probe_sz[-1]}"
    )
    judge_cost_line = (
        f"| L2 cross-family judge Spot-check ({spotcheck.judge_model}) | "
        f"{spotcheck.total_size} item(s) judged; per-call Anthropic cost |\n"
        if spotcheck is not None
        else "| L2 cross-family judge Spot-check | not run (opt-in via `--judge`) |\n"
    )
    stack_c_line = (
        "- **Stack C (Hybrid)**: BM25 AND a dense-over-wiki Section index over the "
        "SAME curated `wiki/` corpus Stack A indexes (dense ids align 1:1 with the "
        "BM25 Section ids), fused by Reciprocal Rank Fusion. Additive — it does not "
        "modify Stack A or Stack B.\n"
        if three_arms
        else ""
    )
    methodology_line = (
        "- **Methodology change (supersedes the Phase 8 single-cutoff report)**: each "
        "arm overfetches a deep candidate pool (target top-"
        f"{DEEP_POOL}) ONCE per Paraphrase, DECOUPLED from the final cutoff; hit-rate "
        f"is reported across a **cutoff sweep** (hit@{{{','.join(str(c) for c in SWEEP_CUTOFFS)}}}) "
        "plus MRR for all arms; and a three-way **Cochran's Q** omnibus gates "
        "**post-hoc pairwise McNemar** (Wiki↔RAG, Hybrid↔Wiki, Hybrid↔RAG) with Holm "
        "correction. The earlier Phase 8 report scored a SINGLE cutoff (hit@3) with "
        "shallow per-stack depth and could not fairly measure a fused arm; its numbers "
        "are superseded by this run, so a reader comparing the two should expect the "
        "headline figures to differ for this reason.\n"
        if three_arms
        else ""
    )
    return (
        "## Experiment Setup\n\n"
        f"- **Corpus**: {n_sources} raw Acme Shop Sources (`corpus/`), fed identically "
        "to both Stacks. Stack A runs `/ingest` over them into `wiki/{entities,concepts}/` "
        "then BM25; Stack B chunks + embeds the raw Sources into FAISS and never runs "
        "`/ingest`. This isolates curated-synthesis-then-keyword vs raw-chunk-then-vector "
        "as the single variable.\n"
        + stack_c_line
        + methodology_line
        + f"- **Paraphrases**: `queries.yaml` (DeepEval Synthesizer — generator "
        f"`{generator}` + `{critic}` same-family critic, seed `{seed}`, "
        f"corpus snapshot `{snapshot}`). {n_core} Core "
        f"({len(CORE_PARAPHRASE_TYPES)} LLM types {core_mult}) + {n_probes} hand-written "
        f"Structural probes ({len(PROBE_PARAPHRASE_TYPES)} types {probe_mult}).\n"
        f"- **Metric**: C5c L1 deterministic — hit_rate@{k} and MRR. A hit requires the "
        "retrieved unit's source to equal the Gold Section AND its content to share at "
        "least one dual-side Key Token, so a correct-id-wrong-content chunk is a miss.\n"
        f"- **Dense embedding mode**: **{embedding_mode}** (`fake` = deterministic "
        "offline stand-in when `OPENAI_API_KEY` is absent; `real` = OpenAI "
        "`text-embedding-3-small`).\n\n"
        "### Cost log\n\n"
        "| Item | Cost |\n"
        "|---|---|\n"
        f"| Paraphrase generation (Core, {generator} + {critic} critic) | `{cost}` |\n"
        + judge_cost_line
        + "| Stack A index-time LLM synthesis (`/ingest`) | one-shot at ingest; **zero** "
        "per-query cost |\n"
        "| Stack B index-time embedding | per-chunk at index; **per-query** embedding "
        "cost at retrieval |\n\n"
        + (
            "The committed query set was generated **offline** "
            f"(`cost_usd: {cost}` in `queries.yaml`), so no dollar figure is fabricated "
            "here. The cost-structure asymmetry above is the real takeaway: Stack A pays "
            "a one-shot LLM synthesis cost and then retrieves for free; Stack B pays a "
            "per-chunk embedding cost at index time AND a per-query embedding cost "
            "forever. At this corpus scale Stack A's zero-marginal-query-cost is a "
            "concrete operational advantage."
            if str(cost).startswith("n/a")
            else "The dollar figure above is the actual billed generation cost."
        )
    )


def _render_family_section(
    title: str,
    family_types: tuple[str, ...],
    stack_a: StackScores,
    stack_b: StackScores,
    k: int,
    chart_files: list[Path],
    with_macro_average: bool,
    core_stats: CoreStats | None = None,
    stack_c: StackScores | None = None,
    three_arm: ThreeArmStats | None = None,
) -> str:
    types = [t for t in family_types if t in stack_a.by_type or t in stack_b.by_type]
    if not types:
        return ""
    three_arms = stack_c is not None
    intro = (
        "The five LLM-generated natural-rewrite types. Read each Δ against the "
        "stated `expected` direction; the per-type rows are the real signal."
        if with_macro_average
        else "The two hand-written probe types, each rigged to exercise a known "
        "architectural limit. These are **expected-limit confirmation**, NOT a "
        "headline result — they are deliberately adversarial and must never be "
        "averaged into the Core story."
    )
    # The per-type table is rendered at the PRIMARY cutoff (k); the full sweep
    # lives in its own section above. Stack C adds three columns when present.
    if three_arms:
        header = (
            f"| Paraphrase Type | hit_rate@{k} (A) | hit_rate@{k} (B) | "
            f"hit_rate@{k} (C) | MRR (A) | MRR (B) | MRR (C) | Δ (B−A) | Δ (C−A) | "
            "expected | n |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    else:
        header = (
            f"| Paraphrase Type | hit_rate@{k} (A) | hit_rate@{k} (B) | MRR (A) | "
            f"MRR (B) | Δ (B−A) | expected | n |"
        )
        sep = "|---|---|---|---|---|---|---|---|"
    lines = [f"## {title}", "", intro, "", header, sep]
    for ptype in types:
        a = stack_a.by_type.get(ptype, 0.0)
        b = stack_b.by_type.get(ptype, 0.0)
        mrr_a = stack_a.mrr_by_type.get(ptype, 0.0)
        mrr_b = stack_b.mrr_by_type.get(ptype, 0.0)
        n = stack_a.n_by_type.get(ptype, stack_b.n_by_type.get(ptype, 0))
        delta = b - a
        expected = _EXPECTED_WINNER.get(ptype, "—")
        if three_arms:
            c = stack_c.by_type.get(ptype, 0.0)
            mrr_c = stack_c.mrr_by_type.get(ptype, 0.0)
            lines.append(
                f"| {ptype} | {a:.3f} | {b:.3f} | {c:.3f} | {mrr_a:.3f} | "
                f"{mrr_b:.3f} | {mrr_c:.3f} | {delta:+.3f} | {c - a:+.3f} | "
                f"{expected} | {n} |"
            )
        else:
            lines.append(
                f"| {ptype} | {a:.3f} | {b:.3f} | {mrr_a:.3f} | {mrr_b:.3f} | "
                f"{delta:+.3f} | {expected} | {n} |"
            )

    if with_macro_average:
        core_a = _macro_average(stack_a.by_type, types)
        core_b = _macro_average(stack_b.by_type, types)
        mrr_core_a = _macro_average(stack_a.mrr_by_type, types)
        mrr_core_b = _macro_average(stack_b.mrr_by_type, types)
        hit_c_clause = mrr_c_clause = ""
        if three_arms:
            core_c = _macro_average(stack_c.by_type, types)
            mrr_core_c = _macro_average(stack_c.mrr_by_type, types)
            hit_c_clause = f" vs Stack C **{core_c:.3f}**"
            mrr_c_clause = f" vs Stack C **{mrr_core_c:.3f}**"
        lines += [
            "",
            f"**Core macro-average** (unweighted mean across the {len(types)} Core "
            f"types): hit_rate@{k} Stack A **{core_a:.3f}** vs Stack B "
            f"**{core_b:.3f}**{hit_c_clause}; MRR Stack A **{mrr_core_a:.3f}** vs "
            f"Stack B **{mrr_core_b:.3f}**{mrr_c_clause}.",
            "",
            "> **Caveat (PRD #100).** This macro-average is reported ONLY as an "
            "unweighted mean over a researcher-chosen set of Core types. It is NOT a "
            "naive cross-type aggregate and must not be read as 'which stack wins' — "
            "the type mix is a design choice, not a representative query distribution. "
            "The per-type rows are authoritative.",
        ]
        if three_arm is not None:
            lines += ["", _render_three_arm_stats(three_arm)]
        if core_stats is not None:
            lines += ["", _render_statistical_tests(core_stats, k)]

    chart_md = _embed_family_charts(title, chart_files)
    if chart_md:
        lines += ["", chart_md]
    return "\n".join(lines)


def _render_cutoff_sweep(
    sweep: tuple[SweepScores, SweepScores, SweepScores] | None,
) -> str:
    """Render the Core macro-average hit_rate + MRR across the cutoff sweep.

    One row per cutoff in the sweep, columns = hit_rate (A/B/C) then MRR (A/B/C),
    each a Core macro-average. This is the AC2 deliverable: hit@{1,3,5,10} + MRR
    for all three arms, read off the single deep pool (no per-cutoff re-retrieval).
    ``None`` (a two-arm legacy call) renders nothing.
    """
    if sweep is None:
        return ""
    sweep_a, sweep_b, sweep_c = sweep
    cutoffs = sweep_a.cutoffs
    lines = [
        "## Cutoff Sweep (hit_rate + MRR across cutoffs, Core macro-average)",
        "",
        "Each arm overfetches a deep candidate pool ONCE per Paraphrase; the rows "
        "below read that one pool at each cutoff (no per-cutoff re-retrieval). The "
        "macro-average is the unweighted mean over the Core types — the per-type "
        "rows in the Core Comparison remain authoritative; this sweep shows whether "
        "a single-cutoff ceiling hid a real difference (the reason the Phase 8 "
        "hit@3 report is superseded).",
        "",
        "| Cutoff | hit_rate (A) | hit_rate (B) | hit_rate (C) | "
        "MRR (A) | MRR (B) | MRR (C) |",
        "|---|---|---|---|---|---|---|",
    ]
    for cutoff in cutoffs:
        ha = _macro_average(sweep_a.hit_by_cutoff[cutoff], CORE_PARAPHRASE_TYPES)
        hb = _macro_average(sweep_b.hit_by_cutoff[cutoff], CORE_PARAPHRASE_TYPES)
        hc = _macro_average(sweep_c.hit_by_cutoff[cutoff], CORE_PARAPHRASE_TYPES)
        ma = _macro_average(sweep_a.mrr_by_cutoff[cutoff], CORE_PARAPHRASE_TYPES)
        mb = _macro_average(sweep_b.mrr_by_cutoff[cutoff], CORE_PARAPHRASE_TYPES)
        mc = _macro_average(sweep_c.mrr_by_cutoff[cutoff], CORE_PARAPHRASE_TYPES)
        lines.append(
            f"| hit@{cutoff} | {ha:.3f} | {hb:.3f} | {hc:.3f} | "
            f"{ma:.3f} | {mb:.3f} | {mc:.3f} |"
        )
    return "\n".join(lines)


def _render_three_arm_stats(three_arm: ThreeArmStats) -> str:
    """Render the three-arm Cochran's Q omnibus + Holm-corrected post-hoc McNemar.

    The omnibus (Q, df, p) tests whether the three arms share one hit_rate at the
    primary cutoff over the pooled Core set; a significant Q gates the three
    pairwise McNemar comparisons (Holm-corrected). When Q is non-significant the
    post-hoc table is shown for completeness but flagged as not warranted.
    """
    q = three_arm.cochran
    sig = q.p_value < 0.05
    gate = (
        "significant (Q gate **open** → post-hoc pairwise McNemar warranted)"
        if sig
        else "not significant (Q gate **closed** → the arms are statistically "
        "indistinguishable overall; the post-hoc rows below are descriptive only)"
    )
    lines = [
        "### Three-Arm Statistical Tests (Cochran's Q omnibus + post-hoc McNemar)",
        "",
        f"Omnibus over the pooled Core set at hit@{three_arm.cutoff} (the three arms "
        "scored on the SAME Paraphrases — a paired design): **Cochran's Q = "
        f"{q.q:.4f}**, df = {q.df}, p = **{q.p_value:.4f}** — {gate}.",
        "",
        "Post-hoc: exact McNemar on each pair, Holm-corrected across the 3 "
        "comparisons. b = left-hit right-miss, c = left-miss right-hit.",
        "",
        "| Pair | b | c | McNemar p | Holm p | sig |",
        "|---|---|---|---|---|---|",
    ]
    for label, (b, c), raw_p, holm_p in zip(
        three_arm.pair_labels,
        three_arm.pair_bc,
        three_arm.pair_raw_p,
        three_arm.pair_holm_p,
    ):
        pair_sig = "✓" if (sig and holm_p < 0.05) else "—"
        lines.append(
            f"| {label} | {b} | {c} | {raw_p:.4f} | {holm_p:.4f} | {pair_sig} |"
        )
    lines += [
        "",
        "> **Interpretation.** Cochran's Q is the omnibus for 3+ related binary "
        "samples (χ² with k−1 df); it asks whether ANY arm differs before any "
        "pairwise claim is made, controlling the family-wise error a naive set of "
        "three McNemar tests would inflate. A pairwise row is read as significant "
        "(sig=✓) only when the omnibus gate is open AND its Holm-corrected p < 0.05.",
    ]
    return "\n".join(lines)


def _render_statistical_tests(core_stats: CoreStats, k: int) -> str:
    """Render the McNemar / Wilson CI / Holm sub-section for Core types.

    Reports per-type: paired McNemar exact p-value (raw), Holm-corrected p-value,
    95% Wilson CIs for each Stack's hit_rate@k, and discordant pair counts (b, c).
    Probes are never included here (they are excluded from the correction family).
    """
    lines = [
        "### Statistical Tests (Core types — paired McNemar + 95% Wilson CI)",
        "",
        f"Paired exact McNemar test (Stack A vs Stack B hit@{k} outcomes per "
        f"Paraphrase); Holm correction across the 5 Core-type tests.  "
        f"b = A-hit B-miss, c = A-miss B-hit.",
        "",
        "| Paraphrase Type | hit_rate (A) [95% CI] | hit_rate (B) [95% CI] | "
        "b | c | McNemar p | Holm p | sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ptype in CORE_PARAPHRASE_TYPES:
        if ptype not in core_stats.by_type:
            continue
        ts = core_stats.by_type[ptype]
        holm_p = core_stats.holm_p_by_type.get(ptype, ts.mcnemar_p)
        sig = "✓" if holm_p < 0.05 else "—"
        ci_a_lo, ci_a_hi = ts.ci_a
        ci_b_lo, ci_b_hi = ts.ci_b
        lines.append(
            f"| {ptype} "
            f"| {ts.hit_rate_a:.3f} [{ci_a_lo:.3f}, {ci_a_hi:.3f}] "
            f"| {ts.hit_rate_b:.3f} [{ci_b_lo:.3f}, {ci_b_hi:.3f}] "
            f"| {ts.b} | {ts.c} "
            f"| {ts.mcnemar_p:.4f} "
            f"| {holm_p:.4f} "
            f"| {sig} |"
        )
    # Power note derived from the actual per-Core-type sample size (issue #145),
    # so it never claims "underpowered, regenerate" once the Demo tier is reached.
    _paras = load_paraphrases()
    _core_ns = sorted(
        {sum(p.paraphrase_type == t for p in _paras) for t in CORE_PARAPHRASE_TYPES}
    )
    _n_desc = (
        f"{_core_ns[0]}"
        if _core_ns[0] == _core_ns[-1]
        else f"{_core_ns[0]}–{_core_ns[-1]}"
    )
    _power_note = (
        f"At n≈{_n_desc} per type the 95% Wilson CIs (see table) are tight enough to "
        "support a per-type claim; a non-significant result then means the two Stacks "
        "are statistically indistinguishable on that type, not merely underpowered."
        if _core_ns[0] >= 30
        else f"Wide CIs at n≈{_n_desc} per type mean the corpus is underpowered — "
        "regenerate with ~50 Paraphrases/type to reach ±0.13."
    )
    lines += [
        "",
        "> **Interpretation.** sig=✓ (Holm p < 0.05) means the two Stacks' "
        "hit_rate differ significantly on that type after family-wise correction.  "
        f"{_power_note}  "
        "Probes are descriptive-only and excluded from this correction family.",
    ]
    return "\n".join(lines)


def _embed_family_charts(section_title: str, chart_files: list[Path]) -> str:
    family = "core" if section_title.startswith("Core") else "probes"
    relevant = [p for p in chart_files if p.name.startswith(f"{family}_")]
    if not relevant:
        return ""
    md = ["### Charts", ""]
    for path in relevant:
        md.append(f"![{path.stem}](charts/{path.name})")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Spot-check (L2) section
# ---------------------------------------------------------------------------
def _render_spotcheck(spotcheck: SpotcheckResult | None) -> str:
    """Render the L2 Spot-check section, or a how-to-enable note when not run.

    When the opt-in judge ran, shows the by-zone subset size + agreement rate
    with L1 + an interpretation; the Control-zone agreement must approach 100% or
    the judge baseline is flagged mis-calibrated (PRD #100 user story 21). When it
    did not run, the section tells the reader exactly how to enable it (opt-in).
    """
    if spotcheck is None:
        return (
            "## Spot-check Validation (L2, cross-family)\n\n"
            "Not run. The deterministic L1 (C5c) metric above is the source of every "
            "headline number; the optional L2 **Spot-check** is a cross-family second "
            "opinion that re-judges L1's edge-case verdicts with a Claude judge (a "
            "different model family from the OpenAI embedding powering Stack B). Enable "
            "it with:\n\n"
            "```\n"
            "ANTHROPIC_API_KEY=... uv run python -m eval.paraphrase_comparison."
            "run_comparison --judge=claude-sonnet-4-6\n"
            "```\n\n"
            "Documented judge choices: `claude-haiku-4-5` / `claude-sonnet-4-6` "
            "(default) / `claude-opus-4-7`. Zone tuning: `--judge-zones`, "
            "`--judge-marginal-threshold` (default 1), `--judge-control-sample-size` "
            "(default 5)."
        )

    control = spotcheck.agreement_by_zone.get("control")
    control_flag = ""
    if control is not None:
        control_flag = (
            f"\n\n> **Control-zone calibration: agreement {control:.3f}.** "
            + (
                "This approaches 100% — the judge baseline is trustworthy, so its "
                "Marginal/Disagreement verdicts can be read as a genuine independent "
                "signal."
                if control >= 0.9
                else "⚠️ This is BELOW the ~100% the Control zone exists to confirm — "
                "the judge itself looks **mis-calibrated**, so treat its other-zone "
                "verdicts with suspicion (PRD #100 user story 21)."
            )
        )

    zone_labels = {
        "marginal": "Marginal (correct id, ≤ threshold Key-Token overlap)",
        "disagreement": "Disagreement (Stack A top-1 verdict ≠ Stack B top-1)",
        "control": "Control (seeded clear-hit + clear-miss baseline)",
    }
    rows = [
        f"| {zone_labels.get(z, z)} | {spotcheck.subset_size_by_zone.get(z, 0)} | "
        f"{spotcheck.agreement_by_zone.get(z, 0.0):.3f} |"
        for z in spotcheck.zones_requested
        if z in spotcheck.subset_size_by_zone
    ]
    return (
        "## Spot-check Validation (L2, cross-family)\n\n"
        f"An opt-in **Spot-check** re-judged L1's edge-case verdicts with the "
        f"cross-family judge **{spotcheck.judge_model}** (Claude — a different model "
        "family from the OpenAI embedding, so no shared blind spot with Stack B). The "
        f"ambiguous subset = {spotcheck.total_size} item(s) across the requested zones "
        f"(marginal threshold = {spotcheck.marginal_threshold}, control sample size = "
        f"{spotcheck.control_sample_size}). The Spot-check produces NO headline numbers "
        "— L1 owns those; it reports only how often the judge AGREES with L1 per "
        "zone.\n\n"
        "| Zone | Subset size | Agreement with L1 |\n"
        "|---|---|---|\n" + "\n".join(rows) + control_flag + "\n\n"
        "Interpretation: high Marginal/Disagreement agreement means L1's uncertain "
        "verdicts hold up under an independent cross-family judge; low agreement "
        "localises exactly where the deterministic metric and a semantic judge part "
        "ways (typically Stack B's correct-id-weak-content 'hits', PRD disclosure 5)."
    )


def _render_limitations(offline: bool, judged: bool = False) -> str:
    # Disclosure (4) flips framing once the cross-family judge has actually run:
    # before the run it is a caveat about an opt-in step; after, it is an active
    # cross-family-validation statement (PRD #100 disclosure 4, issue #105 AC).
    disclosure_4 = (
        "4. **Cross-family validation was run.** The L2 Spot-check used a Claude judge "
        "— a DIFFERENT model family from the OpenAI embedding powering Stack B — so the "
        "second opinion does not share Stack B's same-family blind spot (an OpenAI "
        "judge would only be a same-family opinion with a blindspot on Stack B's "
        "same-family-favoured false positives). The judge validates L1's edge cases "
        "ONLY; L1 remains the source of every headline number. Trust the Spot-check's "
        "Marginal/Disagreement verdicts only insofar as its Control-zone agreement "
        "approaches 100% (see the Spot-check section)."
        if judged
        else "4. **Spot-check family caveat.** The optional L2 judge (Claude) is chosen "
        "to be cross-family from the OpenAI embedding so it does not share a blind spot "
        "with Stack B. When the judge IS run, its control-zone agreement must approach "
        "100% or the judge itself is mis-calibrated and its other verdicts are suspect."
    )
    # Counts derive from the actual frozen corpus so the narrative never drifts
    # from the data after a regeneration (issue #145 — same principle the Setup
    # section already applies; this disclosure was previously a stale hardcoded
    # "16 Sources / ~42 Gold Sections" literal from before the Phase 8.5 corpus
    # growth to 20 Sources / 51 Gold Sections, PRD #143).
    from .generation.sampling import derive_gold_sections  # noqa: PLC0415

    n_sources = len(list(stacks.FIXTURES["corpus"].glob("*.md")))
    n_gold = len(derive_gold_sections(stacks.FIXTURES["corpus"]))
    disclosures = [
        f"1. **Corpus scale is Stack A's sweet spot.** {n_sources} Sources / "
        f"~{n_gold} Gold Sections is "
        "small enough that BM25 over a curated Wiki is hard to beat. The comparison does "
        "NOT claim BM25 wins at scale — it claims it wins *here*, which is exactly the "
        "regime this project operates in.",
        "2. **Synonym / semantic rewrites are Stack B's structural advantage.** Where a "
        "Paraphrase swaps in vocabulary absent from the Source, vector similarity can "
        "match where keyword overlap cannot. A Stack B win on `synonym_swap` / "
        "`implicit_reference` is the architecture working as designed, not noise.",
        "3. **Indexing-time cost scales differently.** Stack A pays a one-shot LLM "
        "synthesis cost at `/ingest` and then retrieves for free; Stack B pays a "
        "per-chunk embedding cost at index time AND a per-query embedding cost forever. "
        "The headline retrieval numbers do not capture this operational asymmetry — the "
        "cost log does.",
        disclosure_4,
        "5. **C5c over-estimates Stack B when `--judge` is skipped.** The deterministic "
        "metric counts a hit on source-match + any Key-Token overlap; without the L2 "
        "spot-check validating edge cases, marginal Stack B 'hits' (correct chunk, weak "
        "content match) are not independently confirmed and may flatter Stack B.",
        "6. **Paraphrase-generator family bias favours Stack B.** The Core Paraphrases "
        "are generated by gpt-4o, whose synonyms fall inside the embedding space "
        "the same model family encodes — systematically advantaging Vector RAG. This is "
        "preserved as a disclosed, measurable finding (the hand-written probes partially "
        "correct for it), not hidden.",
        "7. **Faithfulness-drift risk (residual).** An LLM-written query could "
        "occasionally be mislabeled: the generator might ask about a concept that is "
        "*mentioned* in a Section but whose primary answer lives elsewhere, so the "
        "Gold Section assignment is technically correct yet the query text drifts away "
        "from the canonical formulation.  Mitigations: (a) the answer key (Gold Section "
        "id + Key Tokens) is now derived deterministically from corpus content rather "
        "than asserted by the LLM, confining drift to the query-text layer only; "
        "(b) the McNemar test is a *paired* comparison — any consistent drift affects "
        "both Stacks equally and does not systematically bias the Δ verdict; "
        "(c) the optional L2 Spot-check (Claude judge) can flag query-quality outliers "
        "in the Marginal zone.",
    ]
    offline_disclosure = (
        "8. **The committed numbers are OFFLINE tracer data.** With no "
        "`OPENAI_API_KEY` in the generation environment, the Core Paraphrases are "
        "hand-authored stand-ins for gpt-4o-mini output (faithfully mirroring the "
        "deterministic sha256 section sampling and per-type rules) and BOTH dense "
        "arms use deterministic stand-ins — Stack B a token-overlap ranker and Stack "
        "C's dense-over-wiki arm a hash-based vector — NOT real "
        "`text-embedding-3-small` embeddings. **Stack C (Hybrid) is hit hardest by "
        "this:** with a random hash-vector dense arm, RRF fuses BM25's real ranking "
        "with noise, so the offline Hybrid numbers UNDERSTATE its true performance and "
        "are not a basis for any Hybrid-vs-Wiki/RAG verdict. Readers must NOT mistake "
        "these tracer numbers for the real experiment — the real run (which gates the "
        "`README.md` / `why-wiki.md` update) requires `OPENAI_API_KEY` and a "
        "regenerated `queries.yaml`."
    )
    body = "\n".join(disclosures + ([offline_disclosure] if offline else []))
    return (
        "## Limitations\n\n"
        "These biases are surfaced as findings, not buried — calling them out is the "
        "point of an honest comparison.\n\n" + body
    )


def _render_talking_points() -> str:
    return (
        "## Appendix — Interview Talking Points\n\n"
        '1. *"I chose Markdown KB over Vector RAG because at this corpus size, BM25 + '
        "an inspectable `.kb/index.json` is more debuggable and has zero per-query "
        "embedding cost. `vector_rag/` is preserved for the hybrid retrieval + rerank "
        "layer once the corpus warrants it.\"* — now backed by this comparison's "
        "per-type data and cost log, not assertion.\n"
        '2. *"The comparison isolates the architectural variable: both stacks read the '
        "**same** raw corpus, then each runs its own idiomatic indexing pipeline. Stack "
        "B never runs `/ingest` — it embeds un-curated text, which is the fair baseline "
        'for traditional RAG."*\n'
        '3. *"I separated Core from Structural-probe types and refused a naive '
        "cross-type aggregate, because a researcher-chosen type mix can covertly "
        'manipulate the verdict. The probes are framed as expected-limit confirmation."*\n'
        '4. *"I disclosed the paraphrase-generator family bias proactively: GPT-generated '
        "synonyms fall inside the embedding space the same family encodes, systematically "
        'favouring Vector RAG. Naming the bias is an interview plus, not a minus."*\n'
        '5. *"The metric is a custom DeepEval `BaseMetric` (C5c) — I borrowed the '
        "framework's runner/dataset/report at the leaf and hand-wrote the opinionated "
        "metric at the joint (ADR-0005), rather than adopting Ragas/DeepEval's stock "
        'metrics wholesale."*'
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _macro_average(
    by_type: dict[str, float], types: list[str] | tuple[str, ...]
) -> float:
    """Unweighted mean of a per-type metric over ``types`` present in ``by_type``."""
    present = [by_type[t] for t in types if t in by_type]
    return sum(present) / len(present) if present else 0.0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_comparison(
    k: int = DEFAULT_K,
    report_path: Path = REPORT_PATH,
    embedding_mode: str = "real",
    charts_dir: Path | None = None,
    judge: JudgeConfig | None = None,
) -> tuple[StackScores, StackScores]:
    """Index both Stacks over the eval fixtures, score them, render charts + report.md.

    Production isolation is enforced for the duration of the run (see module
    docstring). Requires OPENAI_API_KEY for Stack B's real embeddings; offline
    callers swap ``vector_rag.app.indexer._build_faiss`` first and pass
    ``embedding_mode="fake"`` so the report records it. ``charts_dir`` defaults to
    a ``charts/`` sibling of ``report_path`` so the report's relative
    ``charts/<file>.png`` links resolve.

    ``judge`` is the opt-in L2 Spot-check config (``None`` = skipped; the report
    then notes how to enable it). When present, the cross-family Claude judge runs
    over the ambiguous subset built from the SAME in-process retrieval callables,
    inside the same production-isolation context. ``run_spotcheck`` fail-fasts if
    ``ANTHROPIC_API_KEY`` is absent.
    """
    charts_dir = charts_dir or (report_path.parent / "charts")
    paraphrases = load_paraphrases()
    metadata = load_metadata()
    scoring, spotcheck = _run_scored(paraphrases, k, judge)
    stack_a, stack_b, stack_c = scoring.primary
    chart_files = charts.render_charts(
        stack_a, stack_b, charts_dir=charts_dir, stack_c=stack_c
    )
    write_text_atomic(
        report_path,
        render_report(
            stack_a,
            stack_b,
            embedding_mode,
            metadata=metadata,
            chart_files=chart_files,
            spotcheck=spotcheck,
            core_stats=scoring.core_stats,
            stack_c=stack_c,
            sweep=scoring.sweep,
            three_arm=scoring.three_arm,
        ),
    )
    return stack_a, stack_b


def _run_scored(
    paraphrases: list[Paraphrase], k: int, judge: JudgeConfig | None = None
) -> tuple[ThreeArmScoring, SpotcheckResult | None]:
    """Build all three indexes under production isolation, then score every arm.

    Uses ``score_three_arms`` so the cutoff-sweep, the per-type table, the
    legacy A-vs-B McNemar, and the three-arm Cochran's Q omnibus are all built
    from aligned per-Paraphrase outcomes (the three arms scored in one pass).
    Stack C's dense arm is built from BM25's Section list (``index_stack_c``
    after ``index_stack_a``) so dense ids align 1:1 with BM25 ids. When ``judge``
    is set, the opt-in L2 Spot-check (A vs B) runs here too — inside the same
    isolation context and against the same in-process callables.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _isolate_production_paths(tmp_path)
        stacks.index_stack_a()
        stacks.index_stack_b()
        stacks.index_stack_c()  # dense-over-wiki, from BM25's Section list
        scoring = score_three_arms(
            paraphrases,
            stacks.stack_a_retrieval,
            stacks.stack_b_retrieval,
            stacks.stack_c_retrieval,
        )
        spotcheck = None
        if judge is not None:
            spotcheck = run_spotcheck(
                paraphrases,
                stacks.stack_a_retrieval,
                stacks.stack_b_retrieval,
                judge_model=judge.judge_model,
                k=k,
                zones=judge.zones,
                marginal_threshold=judge.marginal_threshold,
                control_sample_size=judge.control_sample_size,
            )
    return scoring, spotcheck


def _isolate_production_paths(tmp_path: Path) -> None:
    """Redirect all three Stacks' persistent-state targets to ``tmp_path``.

    SOURCE_DIRS / DOCS_DIR are repointed inside ``stacks.index_stack_{a,b}``;
    here we redirect every persistence + log target so the builds' atomic-write
    side effects land in tmp, never in production ``.kb/`` / ``wiki/`` /
    ``vector_rag/log.md`` / ``hybrid_kb/log.md``. vector_rag's ``build_index``
    persists the FAISS index on success (issue #103) and hybrid_kb's dense
    ``build_index`` persists the dense seed under ``.kb/hybrid_dense/`` (#316), so
    both must be isolated here too (the #307 committed-seed lesson).
    """
    mk_indexer.INDEX_PATH = tmp_path / ".kb" / "index.json"
    mk_indexer.WIKI_DIR = tmp_path / "wiki"
    mk_logger.LOG_PATH = tmp_path / "wiki" / "log.md"
    vr_indexer.FAISS_INDEX_DIR = tmp_path / ".kb" / "faiss_index"
    vr_logger.LOG_PATH = tmp_path / "vector_rag" / "log.md"
    hk_dense.DENSE_INDEX_DIR = tmp_path / ".kb" / "hybrid_dense"
    hk_logger.LOG_PATH = tmp_path / "hybrid_kb" / "log.md"
