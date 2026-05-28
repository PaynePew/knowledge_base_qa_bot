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

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
from deepeval.test_case import LLMTestCase

from . import charts, stacks
from .loader import load_metadata, load_paraphrases, write_text_atomic
from .metric import DEFAULT_K, HitRateAtK
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

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "report.md"

# A Retrieval Stack's retrieval entry point.
StackRetrieval = Callable[[str, int], list[RetrievedItem]]

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
    mrr_by_type = {
        ptype: sum(rrs) / len(rrs) for ptype, rrs in per_type_rr.items()
    }
    n_by_type = {ptype: len(scores) for ptype, scores in per_type_hits.items()}
    return StackScores(
        stack=stack_name,
        k=k,
        by_type=by_type,
        mrr_by_type=mrr_by_type,
        n_by_type=n_by_type,
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
) -> str:
    """Render the full ``report.md`` deliverable for the retrieval comparison.

    Structure (PRD #100): TL;DR → Experiment Setup (incl. cost log) → Core
    Comparison (per-type hit_rate@k + MRR + Δ + expected + n, then a Core
    macro-average WITH a caveat) → Structural Probes (separate table, framed as
    expected-limit confirmation) → Spot-check Validation (only when ``--judge``
    was run) → Limitations (the six+1 honest disclosures) → Interview Talking
    Points appendix.

    ``embedding_mode`` annotates how Stack B's vectors were produced ("real"
    OpenAI embeddings vs a "fake" deterministic offline stand-in) so a reader
    never mistakes an offline tracer number for a real-embedding result.
    ``metadata`` is the ``queries.yaml`` metadata block (read for the honest cost
    log); ``chart_files`` are the rendered PNGs to embed. ``spotcheck`` is the
    primitive-only L2 result (``None`` when the opt-in judge was not run — the
    report then notes how to enable it).
    """
    metadata = metadata or {}
    chart_files = chart_files or []
    k = stack_a.k
    offline = embedding_mode == "fake"
    judged = spotcheck is not None

    parts = [
        _render_header(offline),
        _render_tldr(stack_a, stack_b, k, offline),
        _render_setup(embedding_mode, metadata, k, spotcheck),
        _render_family_section(
            "Core Comparison",
            CORE_PARAPHRASE_TYPES,
            stack_a,
            stack_b,
            k,
            chart_files,
            with_macro_average=True,
        ),
        _render_family_section(
            "Structural Probes",
            PROBE_PARAPHRASE_TYPES,
            stack_a,
            stack_b,
            k,
            chart_files,
            with_macro_average=False,
        ),
        _render_spotcheck(spotcheck),
        _render_limitations(offline, judged),
        _render_talking_points(),
    ]
    return "\n\n".join(p for p in parts if p) + "\n"


def _render_header(offline: bool) -> str:
    banner = (
        "\n\n> ⚠️ **OFFLINE TRACER NUMBERS.** Every score below was produced WITHOUT "
        "`OPENAI_API_KEY`: the Core Paraphrases are hand-authored offline stand-ins "
        "(not gpt-4o-mini output) and Stack B's vectors come from a deterministic "
        "token-overlap stand-in, NOT real `text-embedding-3-small` embeddings. These "
        "numbers exercise the pipeline end-to-end but are **not the real experiment**. "
        "Re-run with `OPENAI_API_KEY` (and a regenerated `queries.yaml`) for headline "
        "figures.\n"
        if offline
        else ""
    )
    return (
        "# Paraphrase Comparison Report\n\n"
        "Phase 8 retrieval comparison (PRD #100): does Karpathy's curated-Wiki layer "
        "(**Stack A** — LLM-synthesised `wiki/` + BM25) out-retrieve a traditional "
        "Vector RAG pipeline (**Stack B** — chunk + embed + FAISS) fed the **same** raw "
        "corpus? Scored at the retrieval layer only by the deterministic C5c hit "
        "metric (source-match AND dual-side Key-Token overlap). K=3."
        + banner
    )


def _render_tldr(
    stack_a: StackScores, stack_b: StackScores, k: int, offline: bool
) -> str:
    core_a = _macro_average(stack_a.by_type, CORE_PARAPHRASE_TYPES)
    core_b = _macro_average(stack_b.by_type, CORE_PARAPHRASE_TYPES)
    qualifier = "offline tracer" if offline else "L1 (deterministic)"
    return (
        "## TL;DR\n\n"
        f"On this 16-Source Acme Shop corpus, the Core macro-average hit_rate@{k} is "
        f"**Stack A {core_a:.3f}** vs **Stack B {core_b:.3f}** ({qualifier} numbers). "
        "The per-type breakdown is the real signal — the macro-average is a "
        "researcher-chosen type mix and is reported only with the caveat below. "
        "Structural probes are reported separately and framed as expected-limit "
        "confirmation, never folded into a headline number."
    )


def _render_setup(
    embedding_mode: str,
    metadata: dict,
    k: int,
    spotcheck: SpotcheckResult | None = None,
) -> str:
    cost = metadata.get("cost_usd", "n/a")
    generator = metadata.get("generator_model", "gpt-4o-mini")
    seed = metadata.get("seed", "n/a")
    snapshot = metadata.get("corpus_snapshot_git_sha", "n/a")
    judge_cost_line = (
        f"| L2 cross-family judge Spot-check ({spotcheck.judge_model}) | "
        f"{spotcheck.total_size} item(s) judged; per-call Anthropic cost |\n"
        if spotcheck is not None
        else "| L2 cross-family judge Spot-check | not run (opt-in via `--judge`) |\n"
    )
    return (
        "## Experiment Setup\n\n"
        "- **Corpus**: 16 raw Acme Shop Sources (`corpus/`), fed identically to both "
        "Stacks. Stack A runs `/ingest` over them into `wiki/{entities,concepts}/` "
        "then BM25; Stack B chunks + embeds the raw Sources into FAISS and never runs "
        "`/ingest`. This isolates curated-synthesis-then-keyword vs raw-chunk-then-vector "
        "as the single variable.\n"
        f"- **Paraphrases**: `queries.yaml` (generator `{generator}`, seed `{seed}`, "
        f"corpus snapshot `{snapshot}`). 40 Core (5 LLM types × 8) + 10 hand-written "
        "Structural probes (2 types × 5).\n"
        f"- **Metric**: C5c L1 deterministic — hit_rate@{k} and MRR. A hit requires the "
        "retrieved unit's source to equal the Gold Section AND its content to share at "
        "least one dual-side Key Token, so a correct-id-wrong-content chunk is a miss.\n"
        f"- **Stack B embedding mode**: **{embedding_mode}** (`fake` = deterministic "
        "offline stand-in when `OPENAI_API_KEY` is absent; `real` = OpenAI "
        "`text-embedding-3-small`).\n\n"
        "### Cost log\n\n"
        "| Item | Cost |\n"
        "|---|---|\n"
        f"| Paraphrase generation (Core, {generator}) | `{cost}` |\n"
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
) -> str:
    types = [
        t for t in family_types if t in stack_a.by_type or t in stack_b.by_type
    ]
    if not types:
        return ""
    intro = (
        "The five LLM-generated natural-rewrite types. Read each Δ against the "
        "stated `expected` direction; the per-type rows are the real signal."
        if with_macro_average
        else "The two hand-written probe types, each rigged to exercise a known "
        "architectural limit. These are **expected-limit confirmation**, NOT a "
        "headline result — they are deliberately adversarial and must never be "
        "averaged into the Core story."
    )
    lines = [
        f"## {title}",
        "",
        intro,
        "",
        f"| Paraphrase Type | hit_rate@{k} (A) | hit_rate@{k} (B) | MRR (A) | "
        f"MRR (B) | Δ (B−A) | expected | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ptype in types:
        a = stack_a.by_type.get(ptype, 0.0)
        b = stack_b.by_type.get(ptype, 0.0)
        mrr_a = stack_a.mrr_by_type.get(ptype, 0.0)
        mrr_b = stack_b.mrr_by_type.get(ptype, 0.0)
        n = stack_a.n_by_type.get(ptype, stack_b.n_by_type.get(ptype, 0))
        delta = b - a
        expected = _EXPECTED_WINNER.get(ptype, "—")
        lines.append(
            f"| {ptype} | {a:.3f} | {b:.3f} | {mrr_a:.3f} | {mrr_b:.3f} | "
            f"{delta:+.3f} | {expected} | {n} |"
        )

    if with_macro_average:
        core_a = _macro_average(stack_a.by_type, types)
        core_b = _macro_average(stack_b.by_type, types)
        mrr_core_a = _macro_average(stack_a.mrr_by_type, types)
        mrr_core_b = _macro_average(stack_b.mrr_by_type, types)
        lines += [
            "",
            f"**Core macro-average** (unweighted mean across the {len(types)} Core "
            f"types): hit_rate@{k} Stack A **{core_a:.3f}** vs Stack B "
            f"**{core_b:.3f}**; MRR Stack A **{mrr_core_a:.3f}** vs Stack B "
            f"**{mrr_core_b:.3f}**.",
            "",
            "> **Caveat (PRD #100).** This macro-average is reported ONLY as an "
            "unweighted mean over a researcher-chosen set of Core types. It is NOT a "
            "naive cross-type aggregate and must not be read as 'which stack wins' — "
            "the type mix is a design choice, not a representative query distribution. "
            "The per-type rows are authoritative.",
        ]

    chart_md = _embed_family_charts(title, chart_files)
    if chart_md:
        lines += ["", chart_md]
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
    disclosures = [
        "1. **Corpus scale is Stack A's sweet spot.** 16 Sources / ~42 Gold Sections is "
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
        "are generated by gpt-4o-mini, whose synonyms fall inside the embedding space "
        "the same model family encodes — systematically advantaging Vector RAG. This is "
        "preserved as a disclosed, measurable finding (the hand-written probes partially "
        "correct for it), not hidden.",
    ]
    offline_disclosure = (
        "7. **The committed numbers are OFFLINE tracer data.** With no "
        "`OPENAI_API_KEY` in the generation environment, the Core Paraphrases are "
        "hand-authored stand-ins for gpt-4o-mini output (faithfully mirroring the "
        "deterministic sha256 section sampling and per-type rules) and Stack B's "
        "retrieval uses a deterministic token-overlap stand-in, NOT real "
        "`text-embedding-3-small` embeddings. Readers must NOT mistake these tracer "
        "numbers for the real experiment — a real run requires `OPENAI_API_KEY` and a "
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
        "1. *\"I chose Markdown KB over Vector RAG because at this corpus size, BM25 + "
        "an inspectable `.kb/index.json` is more debuggable and has zero per-query "
        "embedding cost. `vector_rag/` is preserved for the hybrid retrieval + rerank "
        "layer once the corpus warrants it.\"* — now backed by this comparison's "
        "per-type data and cost log, not assertion.\n"
        "2. *\"The comparison isolates the architectural variable: both stacks read the "
        "**same** raw corpus, then each runs its own idiomatic indexing pipeline. Stack "
        "B never runs `/ingest` — it embeds un-curated text, which is the fair baseline "
        "for traditional RAG.\"*\n"
        "3. *\"I separated Core from Structural-probe types and refused a naive "
        "cross-type aggregate, because a researcher-chosen type mix can covertly "
        "manipulate the verdict. The probes are framed as expected-limit confirmation.\"*\n"
        "4. *\"I disclosed the paraphrase-generator family bias proactively: GPT-generated "
        "synonyms fall inside the embedding space the same family encodes, systematically "
        "favouring Vector RAG. Naming the bias is an interview plus, not a minus.\"*\n"
        "5. *\"The metric is a custom DeepEval `BaseMetric` (C5c) — I borrowed the "
        "framework's runner/dataset/report at the leaf and hand-wrote the opinionated "
        "metric at the joint (ADR-0005), rather than adopting Ragas/DeepEval's stock "
        "metrics wholesale.\"*"
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _macro_average(by_type: dict[str, float], types: list[str] | tuple[str, ...]) -> float:
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
    stack_a, stack_b, spotcheck = _run_scored(paraphrases, k, judge)
    chart_files = charts.render_charts(stack_a, stack_b, charts_dir=charts_dir)
    write_text_atomic(
        report_path,
        render_report(
            stack_a,
            stack_b,
            embedding_mode,
            metadata=metadata,
            chart_files=chart_files,
            spotcheck=spotcheck,
        ),
    )
    return stack_a, stack_b


def _run_scored(
    paraphrases: list[Paraphrase], k: int, judge: JudgeConfig | None = None
) -> tuple[StackScores, StackScores, SpotcheckResult | None]:
    """Build both indexes under production isolation, then score each Stack.

    When ``judge`` is set, the opt-in L2 Spot-check runs here too — inside the
    same isolation context and against the same in-process Stack retrieval
    callables — so its zones are built from the identical L1 verdicts.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _isolate_production_paths(tmp_path)
        stacks.index_stack_a()
        stacks.index_stack_b()
        a = score_stack("Stack A", paraphrases, stacks.stack_a_retrieval, k)
        b = score_stack("Stack B", paraphrases, stacks.stack_b_retrieval, k)
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
    return a, b, spotcheck


def _isolate_production_paths(tmp_path: Path) -> None:
    """Redirect both Stacks' persistent-state targets to ``tmp_path``.

    SOURCE_DIRS / DOCS_DIR are repointed inside ``stacks.index_stack_{a,b}``;
    here we redirect every persistence + log target so the builds' atomic-write
    side effects land in tmp, never in production ``.kb/`` / ``wiki/`` /
    ``vector_rag/log.md``. vector_rag's ``build_index`` now persists the FAISS
    index on success (issue #103), so its ``FAISS_INDEX_DIR`` must be isolated
    here too.
    """
    mk_indexer.INDEX_PATH = tmp_path / ".kb" / "index.json"
    mk_indexer.WIKI_DIR = tmp_path / "wiki"
    mk_logger.LOG_PATH = tmp_path / "wiki" / "log.md"
    vr_indexer.FAISS_INDEX_DIR = tmp_path / ".kb" / "faiss_index"
    vr_logger.LOG_PATH = tmp_path / "vector_rag" / "log.md"
