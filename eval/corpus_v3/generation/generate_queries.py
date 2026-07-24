"""One-off CLI to generate the corpus v3 power-sized query set and write it
as a committed artifact (issue #672, ADR-0045 Prerequisite 4,
``generation/SPEC.md``, ``POWER_ANALYSIS.md``).

Usage (from repo root):

    uv run python -m eval.corpus_v3.generation.generate_queries

NOT run under pytest (``pyproject.toml``'s ``testpaths`` scopes collection to
``eval/corpus_v3/tests``, so this module is never collected). The
deterministic seams it composes (``targets``, ``prompts``, ``overlap``,
``qc``, ``artifact``) are unit-tested there; LLM output content is never
asserted (CODING_STANDARD §6.2), mirroring
``eval.paraphrase_comparison.generate_paraphrases``. This is a one-off script,
so stdout via ``print`` is acceptable (CODING_STANDARD §5.1 -- the no-print
rule is scoped to committed library code, same exemption
``generate_paraphrases.py`` documents for itself).

Pipeline:

    1. Refuse up front if ``OPENAI_API_KEY`` is absent — Family A cannot run
       without it, and unlike the Phase 8 paraphrase generator there is no
       prior committed corpus v3 query file to fall back to QC-only mode
       over (this is the first-ever generation run). Nothing is written.
    2. Resolve Family B (``generation/SPEC.md``'s two sanctioned options).
       If ``ANTHROPIC_API_KEY`` is present, Family B is a second, non-OpenAI
       model family (``GENERATOR_MODEL_B``): each cell's deterministic
       enumeration is split between the families -- Family A generates
       ``[0, ceil(n/2))``, Family B ``[ceil(n/2), n)`` -- so both families'
       phrasing styles are represented in every stratum. Otherwise fall back
       to the human-written slice (``generation/human_slice.yaml``). Neither
       available is a HARD refusal, not a silent Family-A-only fallback --
       generating the whole set from one family would violate ADR-0045
       Prerequisite 3's multi-family requirement, and this project's honesty
       stance is to fail closed rather than write a known-biased artifact.
    3. Derive generation targets from the committed adversarial corpus
       (``targets.derive_generation_targets``) and deterministically sample
       (``targets.sample_targets``) each (stratum, language) cell up to its
       power-sized target count.
    4. PRE-SPEND cost guard (issue #672 AC 5 / #662 AC 2 pattern): run a
       small pilot batch of real Family A calls, record it into a
       ``CostLedger``, project the FULL planned run's spend, and hard-stop
       without spending further if the projection exceeds
       ``cost_guard.BUDGET_USD_CAP`` — the caller's job at that point is to
       label the issue ``ready-for-human``, not to raise the cap in code.
    5. Generate the remaining queries, stitch (``gen_schema.to_query``),
       classify overlap (``overlap.classify_overlap_stratum``), and QC-gate
       each (``qc.check_generated_query``); a rejected draft is dropped, not
       retried, and counted in the header's ``qc_rejected`` tally — a
       single-attempt-per-slot simplification, documented so a future slice
       can add bounded retry without this run's numbers being misread as
       "retry already happened."
    6. Write the artifact (``artifact.render_query_artifact``) with a
       metadata header naming both generating families, every stratum's
       target-vs-actual count, and total cost.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from eval.cost_ledger.hooks import record_usage_from_response
from eval.cost_ledger.ledger import CostLedger

from .. import cost_guard as ledger_cost_guard
from ..build_corpus import ADVERSARIAL_GROUPS
from ..query_schema import LANGUAGES, SCENARIO_STRATA, Language, Query, ScenarioStratum
from . import overlap, qc
from .artifact import StratumCount, build_metadata, render_query_artifact
from .gen_schema import QueryDraft, to_query
from .prompts import PROMPT_TEMPLATE_VERSION, build_prompt
from .targets import GenerationTarget, derive_generation_targets, sample_targets

_PKG_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = _PKG_ROOT / "queries.yaml"
HUMAN_SLICE_PATH = _PKG_ROOT / "human_slice.yaml"

GENERATOR_MODEL_A = "gpt-4o-mini"

# Family B (ADR-0045 Prerequisite 3: a second, non-OpenAI family). Haiku 4.5
# rather than an Opus-tier model for three build-time reasons: (1) the
# pre-registered $10 cost guard -- ~2,200 Family-B calls on an Opus-tier
# model project past the cap, while the whole two-family run on Haiku stays
# well under it; (2) ``generation/SPEC.md``'s temperature-0 convention --
# Claude 4.6+ models reject the ``temperature`` parameter outright, Haiku 4.5
# still honors an explicit 0; (3) generating short adversarial questions is
# not an intelligence-sensitive task. Free-text per SPEC's
# ``generating_family`` note; re-pin ``eval/cost_ledger/unit_prices.py`` if
# this changes.
GENERATOR_MODEL_B = "claude-haiku-4-5"

# arm B's dense embedding model (ADR-0005) -- recorded in the artifact header
# for audit, per PRD #654's "generating family recorded per query" spirit
# applied at the run level too. NOT compared against ``GENERATOR_MODEL_A``
# (see ``artifact.build_metadata``'s docstring: Family A is deliberately the
# SAME vendor family as this embedding model, per ``generation/SPEC.md`` --
# the multi-family requirement is that Family B differs from Family A, not
# that Family A avoids this model).
ARM_B_EMBEDDING_MODEL = "text-embedding-3-small"
STACK_NAME = "corpus_v3_generation"
LEDGER_PHASE = "query"

# POWER_ANALYSIS.md: n=909 per English scenario stratum. The zh slice's own
# relaxed gate (power=0.70, mdd=0.10) does not name a per-stratum n
# explicitly the way the English section does ("n=909 ... per scenario
# stratum" vs. zh's bare "n=200") — this run applies the SAME per-stratum
# shape to zh for structural consistency with English, at n=200 per stratum,
# and states that reading explicitly here rather than leaving it implicit.
EN_TARGET_PER_STRATUM = 909
ZH_TARGET_PER_STRATUM = 200

PILOT_CALLS = 20


# ---------------------------------------------------------------------------
# LLM singleton (lazy — ADR-0005; only constructed when actually generating)
# ---------------------------------------------------------------------------
def _get_family_a_llm():
    """Return a gpt-4o-mini client bound to ``QueryDraft`` with raw usage
    metadata exposed (``include_raw=True`` — required because a plain
    ``with_structured_output`` chain does not expose ``usage_metadata`` on
    its parsed-object-only return; see ``eval.cost_ledger.hooks``'s
    docstring, which names this exact call site)."""
    from langchain_openai import ChatOpenAI  # function-scope: keep LangChain internal

    llm = ChatOpenAI(
        model=GENERATOR_MODEL_A, temperature=0.0, timeout=60, max_retries=1
    )
    return llm.with_structured_output(QueryDraft, include_raw=True)


def _get_family_b_llm():
    """Return a ``GENERATOR_MODEL_B`` (Anthropic) client with the same
    structured-output + raw-usage shape as :func:`_get_family_a_llm`, so
    ``_generate_query`` drives both families through one seam."""
    from langchain_anthropic import (  # function-scope: keep LangChain internal
        ChatAnthropic,
    )

    llm = ChatAnthropic(
        model=GENERATOR_MODEL_B, temperature=0.0, timeout=60, max_retries=1
    )
    return llm.with_structured_output(QueryDraft, include_raw=True)


# ---------------------------------------------------------------------------
# Family B — human-written slice
# ---------------------------------------------------------------------------
def load_human_slice(path: Path = HUMAN_SLICE_PATH) -> list[Query]:
    """Load Family B's human-written queries, or ``[]`` if ``path`` is absent.

    Reuses ``query_schema.load_queries`` directly (the human slice file is
    the same ``{"queries": [...]}`` shape) rather than a bespoke parser, so a
    human-authored entry is validated by the exact same invariants
    (``Query.__post_init__``) a generated one is. A present-but-malformed
    file still raises (fail-fast per CODING_STANDARD §4.1) — only a MISSING
    file is treated as "not authored yet," not "empty is fine."
    """
    if not path.exists():
        return []
    from ..query_schema import load_queries

    return load_queries(path)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _generate_query(
    llm,
    ledger: CostLedger,
    target: GenerationTarget,
    *,
    model: str,
    language: Language,
    variant_index: int,
    query_id: str,
) -> Query | None:
    """Generate and QC-gate one query. Returns ``None`` (and records the
    rejection reasons via the caller's stats) when the QC gate rejects it —
    a rejected draft is never returned for the caller to write. ``model``
    names the generating family for both the ledger record and the query's
    ``generating_family`` field."""
    prompt = build_prompt(target, language=language, variant_index=variant_index)
    result = llm.invoke(prompt)
    record_usage_from_response(
        ledger,
        stack=STACK_NAME,
        phase=LEDGER_PHASE,
        model=model,
        response=result.get("raw"),
    )
    draft: QueryDraft | None = result.get("parsed")
    if draft is None:
        return None  # structured-output parse failure — treated as a reject
    overlap_stratum = overlap.classify_overlap_stratum(
        draft.text, [target.reference_text]
    )
    try:
        query = to_query(
            draft,
            query_id=query_id,
            scenario_stratum=target.scenario_stratum,
            language=language,
            overlap_stratum=overlap_stratum,
            gold_section_ids=target.gold_section_ids,
            generating_family=model,
        )
    except ValueError:
        # The draft violates a Query invariant (observed live: a model
        # returning empty key_tokens on an answerable slot). An invalid
        # draft is a rejection to tally, not a reason to abort a paid run
        # a thousand calls in.
        return None
    verdict = qc.check_generated_query(query)
    return None if verdict.rejected else query


def _plan_cells() -> list[tuple[ScenarioStratum, Language, int]]:
    """Every (stratum, language, target-count) cell this run must fill."""
    per_language = {"en": EN_TARGET_PER_STRATUM, "zh": ZH_TARGET_PER_STRATUM}
    return [
        (stratum, language, per_language[language])
        for stratum in SCENARIO_STRATA
        for language in LANGUAGES
    ]


def run_generation(
    llm,
    ledger: CostLedger,
    *,
    model: str = GENERATOR_MODEL_A,
    cells: list[tuple] | None = None,
) -> tuple[list[Query], list[StratumCount]]:
    """Generate every planned cell. Returns the accepted queries and a
    ``StratumCount`` per cell (target vs actual vs qc_rejected).

    A cell is ``(stratum, language, target_n)`` for the whole cell, or
    ``(stratum, language, target_n, start, stop)`` for a window over the
    cell's single deterministic enumeration. ``target_n`` is ALWAYS the
    cell's full plan count: the sample (and therefore ``query_id``
    numbering) is derived from it once, identically for every window, so a
    pilot window ``(.., 0, k)`` and its remainder ``(.., k, target_n)``
    partition the same id space instead of both restarting at 0. The
    returned ``StratumCount.target`` covers only the window (``stop -
    start``); summing a cell's windows recovers the full target."""
    all_targets = derive_generation_targets(ADVERSARIAL_GROUPS)
    queries: list[Query] = []
    counts: list[StratumCount] = []
    for cell in cells if cells is not None else _plan_cells():
        if len(cell) == 3:
            stratum, language, target_n = cell
            start, stop = 0, target_n
        else:
            stratum, language, target_n, start, stop = cell
        if not 0 <= start <= stop <= target_n:
            raise ValueError(
                f"run_generation: window [{start}, {stop}) outside cell "
                f"{stratum}/{language} target {target_n}"
            )
        pool = all_targets[stratum]
        sampled = sample_targets(pool, seed=f"{stratum}:{language}", count=target_n)
        accepted = 0
        rejected = 0
        for idx in range(start, stop):
            target = sampled[idx]
            query_id = f"{stratum}-{language}-{idx:04d}"
            query = _generate_query(
                llm,
                ledger,
                target,
                model=model,
                language=language,
                variant_index=idx,
                query_id=query_id,
            )
            if query is None:
                rejected += 1
            else:
                accepted += 1
                queries.append(query)
        counts.append(
            StratumCount(
                scenario_stratum=stratum,
                language=language,
                target=stop - start,
                actual=accepted,
                qc_rejected=rejected,
            )
        )
    return queries, counts


def _split_family_windows(
    plan: list[tuple[ScenarioStratum, Language, int]],
) -> tuple[list[tuple], list[tuple]]:
    """Split every cell's enumeration between the two model families:
    Family A generates ``[0, ceil(n/2))``, Family B ``[ceil(n/2), n)``.
    Both families draw from the SAME deterministic sample (``target_n`` is
    the full cell count in every window), so the split changes who phrases a
    slot, never which reference Sections are targeted."""
    a_windows = [(s, lang, n, 0, (n + 1) // 2) for s, lang, n in plan]
    b_windows = [
        (s, lang, n, (n + 1) // 2, n) for s, lang, n in plan if n > (n + 1) // 2
    ]
    return a_windows, b_windows


def _take_pilot(windows: list[tuple], n_calls: int) -> tuple[list[tuple], list[tuple]]:
    """Front-slice up to ``n_calls`` from ``windows``. Returns
    ``(pilot_windows, remainder_windows)`` — disjoint sub-windows of the
    input that together cover it exactly, so pilot and remainder partition
    the id space (the invariant ``run_generation``'s windowing exists for)."""
    pilot: list[tuple] = []
    rest: list[tuple] = []
    remaining = n_calls
    for stratum, language, target, start, stop in windows:
        take = min(stop - start, remaining)
        remaining -= take
        if take > 0:
            pilot.append((stratum, language, target, start, start + take))
        if start + take < stop:
            rest.append((stratum, language, target, start + take, stop))
    return pilot, rest


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corpus v3 query generator.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--human-slice", type=Path, default=HUMAN_SLICE_PATH)
    parser.add_argument("--pilot-calls", type=int, default=PILOT_CALLS)
    args = parser.parse_args(argv)

    load_dotenv(
        find_dotenv(usecwd=True)
    )  # pick up OPENAI_API_KEY from a repo-root .env

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY absent -- Family A (gpt-4o-mini) generation cannot run, "
            "and no prior committed corpus v3 query file exists to fall back to a "
            "QC-only pass over. Nothing was written. This run cannot proceed: "
            "supply OPENAI_API_KEY (and, for Family B, either a second non-OpenAI "
            "model or an authored generation/human_slice.yaml) or label the issue "
            "ready-for-human."
        )
        return 1

    use_model_b = bool(os.getenv("ANTHROPIC_API_KEY"))
    human_slice: list[Query] = []
    if use_model_b:
        family_b_source = GENERATOR_MODEL_B
        if args.human_slice.exists():
            print(
                f"Note: ANTHROPIC_API_KEY present -- Family B is "
                f"{GENERATOR_MODEL_B}; {args.human_slice.name} is ignored."
            )
    else:
        human_slice = load_human_slice(args.human_slice)
        if not human_slice:
            print(
                f"Family B unavailable -- ANTHROPIC_API_KEY is absent and "
                f"{args.human_slice.name} was not found. Generating Family A "
                "(gpt-4o-mini) alone would violate ADR-0045 Prerequisite 3's "
                "multi-family requirement, so this run refuses rather than "
                "writing a biased query set. Supply ANTHROPIC_API_KEY (Family B "
                f"= {GENERATOR_MODEL_B}) or author generation/human_slice.yaml "
                "before re-running, or label the issue ready-for-human."
            )
            return 1
        family_b_source = "human"

    ledger = CostLedger()
    plan = _plan_cells()
    if use_model_b:
        a_windows, b_windows = _split_family_windows(plan)
        families = [
            (GENERATOR_MODEL_A, _get_family_a_llm(), a_windows),
            (GENERATOR_MODEL_B, _get_family_b_llm(), b_windows),
        ]
    else:
        # Human-slice mode: Family A generates the whole plan; the human
        # slice is merged in as-is (its ids live in the "human-" namespace).
        full_windows = [(s, lang, n, 0, n) for s, lang, n in plan]
        families = [(GENERATOR_MODEL_A, _get_family_a_llm(), full_windows)]

    total_planned_calls = sum(
        stop - start
        for _model, _llm, windows in families
        for _s, _lang, _t, start, stop in windows
    )
    # Pilot split evenly across families so the mixed-price projection scales
    # a pilot whose family mix matches the full plan's.
    pilot_n = min(args.pilot_calls, total_planned_calls)
    shares = [pilot_n // len(families)] * len(families)
    for i in range(pilot_n % len(families)):
        shares[i] += 1

    queries: list[Query] = list(human_slice)
    all_counts: list[StratumCount] = []
    remainders: list[tuple] = []
    for (model, llm, windows), share in zip(families, shares):
        pilot_windows, rest_windows = _take_pilot(windows, share)
        pilot_queries, pilot_counts = run_generation(
            llm, ledger, model=model, cells=pilot_windows
        )
        queries.extend(pilot_queries)
        all_counts.extend(pilot_counts)
        remainders.append((model, llm, rest_windows))

    projection = ledger_cost_guard.project_spend(
        ledger, phase=LEDGER_PHASE, planned_calls=total_planned_calls
    )
    guard = ledger_cost_guard.check_cost_guard(projection)
    print(guard.message)
    if not guard.proceed:
        print(
            "Halting before further spend -- mark issue #672 ready-for-human "
            "with this projection instead of running."
        )
        return 1

    for model, llm, rest_windows in remainders:
        rest_queries, rest_counts = run_generation(
            llm, ledger, model=model, cells=rest_windows
        )
        queries.extend(rest_queries)
        all_counts.extend(rest_counts)

    counts_by_cell: dict[tuple[str, str], StratumCount] = {}
    for c in all_counts:
        key = (c.scenario_stratum, c.language)
        prior = counts_by_cell.get(key)
        if prior is None:
            counts_by_cell[key] = c
        else:
            counts_by_cell[key] = StratumCount(
                scenario_stratum=c.scenario_stratum,
                language=c.language,
                # Every entry in all_counts carries only its OWN window of
                # the cell's full target (pilot/rest x family A/B all slice
                # the same enumeration) -- sum the windows to recover the
                # true per-cell target, never just the first one seen.
                target=prior.target + c.target,
                actual=prior.actual + c.actual,
                qc_rejected=prior.qc_rejected + c.qc_rejected,
            )
    counts = list(counts_by_cell.values())

    totals = ledger.totals(phase=LEDGER_PHASE)
    metadata = build_metadata(
        counts=counts,
        family_a_model=GENERATOR_MODEL_A,
        family_b_source=family_b_source,
        embedding_family=ARM_B_EMBEDDING_MODEL,
        generated_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cost_usd=totals.usd,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )

    args.output.write_text(
        render_query_artifact(queries, metadata=metadata), encoding="utf-8"
    )
    print(
        f"Wrote {len(queries)} queries to {args.output.name} (cost ${totals.usd:.2f})"
        if totals.usd is not None
        else f"Wrote {len(queries)} queries to {args.output.name} (cost unknown)"
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
