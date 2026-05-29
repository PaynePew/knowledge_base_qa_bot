"""One-off CLI that replaces the bespoke gpt-4o-mini generator with DeepEval's
Synthesizer (issue #144, Phase 8.5 S6).

Usage (from repo root)
----------------------

    # Offline QC only (no API key, no generation):
    uv run python -m eval.paraphrase_comparison.generate_paraphrases_v2 --qc-only

    # Live generation (requires OPENAI_API_KEY — issue #145: HITL step):
    uv run python -m eval.paraphrase_comparison.generate_paraphrases_v2 \\
        --per-type 50   # Demo tier: ≈50 queries / Core Type (PRD #137)

    # Smaller run for a quick smoke-check:
    uv run python -m eval.paraphrase_comparison.generate_paraphrases_v2 \\
        --per-type 5

Pipeline (live path)
--------------------
1.  Freeze corpus snapshot from docs/fake-docs/ → eval/paraphrase_comparison/corpus/
    (inherited from generate_paraphrases.py — same freeze_corpus function, AC3 from
    issue #142).

2.  Derive the Gold Section inventory from the frozen corpus (derive_gold_sections,
    issue #142 — every Gold Section is offered, was 28/42 in the old generator).

3.  Load Section bodies from the frozen corpus (_docs_section_bodies from the
    original script, unchanged).

4.  Build (contexts, source_files) via build_section_contexts — one context per
    Gold Section; source_files carries the section id so DeepEval propagates it to
    Golden.source_file.

5.  For each Core Paraphrase Type, call the Synthesizer once per type with the
    full Gold Section pool split into per-type batches (sha256 sampling, same seed
    logic as before).  max_goldens_per_context=1 → one query per Section per Type.

    Quantity parameterisation: ``--per-type N`` controls the sample size.  With N
    Gold Sections offered (≤ pool size), the Synthesizer generates ≈N queries per
    type (before quality filtering).

6.  Post-filter Goldens by critic quality score (post_filter_by_score).

7.  Convert Goldens to Paraphrase objects (goldens_to_queries) — gold_docs_section_id
    is derived deterministically from golden.source_file.

8.  Re-key Key Tokens deterministically (rekey_paraphrase from rekey.py — inherited
    from issue #139, derives Key Tokens from Section body IDF, NOT from the LLM).

9.  Run the QC gate (generation.qc) and print the flagged-for-review report.

10. Merge the hand-written probe types from probes.yaml unchanged.

11. Write queries.yaml atomically with a metadata block.

Handoff to issue #145
---------------------
The live generation step (step 5) burns real OpenAI API budget ($15–30 for a
Demo-tier run, PRD #137).  Issue #145 (Demo-tier live regeneration + HITL trust
review) is the owner of that step.  This script is the deliverable for #145 —
running it with OPENAI_API_KEY set performs the full regeneration.

Stdout via ``print`` is acceptable here (one-off CLI, CODING_STANDARD §5.1).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


from .generate_paraphrases import (
    CORPUS_DIR,
    FAKE_DOCS_DIR,
    PROBES_PATH,
    _docs_section_bodies,
    freeze_corpus,
    load_probes,
    render_queries_yaml,
    run_qc,
    _print_qc,
)
from .generation import qc, sampling
from .generation.sampling import derive_gold_sections
from .generation.synthesizer_pipeline import (
    SynthesizerConfig,
    build_section_contexts,
    build_synthesizer,
    goldens_to_queries,
    post_filter_by_score,
)
from .loader import QUERIES_PATH, write_text_atomic
from .models import Paraphrase
from .rekey import rekey_paraphrase

_PKG_ROOT = Path(__file__).resolve().parent

# Core types that the Synthesizer generates (probe types are hand-written).
_CORE_TYPES: tuple[str, ...] = (
    "synonym_swap",
    "word_reorder",
    "verbosity_expansion",
    "specificity_narrowing",
    "implicit_reference",
)


def generate_core_via_synthesizer(
    config: SynthesizerConfig,
    *,
    corpus_dir: Path = CORPUS_DIR,
) -> list[Paraphrase]:
    """Generate the five Core Paraphrase Types via the DeepEval Synthesizer.

    For each Core Type:
      1. Sample ``config.per_type_count`` Gold Sections (sha256-keyed, cross-type
         reuse allowed — same sampling logic as the original generator).
      2. Build per-Section contexts and source_files (deterministic).
      3. Call the Synthesizer with ``max_goldens_per_context=1`` (one query /
         Section / Type).
      4. Post-filter by quality score.
      5. Convert Goldens to Paraphrase objects (gold_docs_section_id from
         golden.source_file — deterministic).
      6. Re-key Key Tokens from Section body IDF (issue #139 — not from LLM).

    Raises on missing OPENAI_API_KEY; call from the ``main`` live path only.

    Coverage: ``derive_gold_sections(corpus_dir)`` returns every Gold Section
    (≥42 in the Demo corpus).  ``sample_sections`` draws ``per_type_count`` of
    them per type.  With per_type_count ≥ 42 all sections are offered; with
    per_type_count < pool size a sha256-keyed subset is offered — coverage is
    per-type, not global, but the pool is unfiltered (was 28/42 before).
    """
    gold = derive_gold_sections(corpus_dir)
    docs_bodies = _docs_section_bodies()
    synth = build_synthesizer(config)

    # Build IDF table once over the frozen corpus for Key Token re-keying.
    idf = qc.build_idf([body for _, body in docs_bodies.values()])

    out: list[Paraphrase] = []
    for ptype in _CORE_TYPES:
        # Deterministic sha256-keyed sampling (same convention as original).
        sections = sampling.sample_sections(
            gold, seed=ptype, count=config.per_type_count
        )

        contexts, source_files = build_section_contexts(sections, docs_bodies)

        goldens = synth.generate_goldens_from_contexts(
            contexts=contexts,
            source_files=source_files,
            max_goldens_per_context=1,
            include_expected_output=False,
        )
        filtered = post_filter_by_score(goldens, threshold=config.quality_threshold)

        id_prefix = ptype.replace("_", "-")[:8]  # short prefix for readability
        paraphrases = goldens_to_queries(
            filtered, paraphrase_type=ptype, id_prefix=id_prefix
        )  # type: ignore[arg-type]

        # Re-key Key Tokens deterministically from Section body IDF (issue #139).
        rekeyed: list[Paraphrase] = []
        for p in paraphrases:
            heading, body = docs_bodies.get(p.gold_docs_section_id, ("", ""))
            rekeyed.append(rekey_paraphrase(p, section_body=body, idf=idf))
        out.extend(rekeyed)

        print(
            f"  [{ptype}] {len(sections)} sections offered → "
            f"{len(goldens)} generated → "
            f"{len(filtered)} post-filtered → "
            f"{len(rekeyed)} re-keyed"
        )

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.5 S6: DeepEval Synthesizer query generation (issue #144). "
            "Replaces the bespoke gpt-4o-mini generator with DeepEval's Synthesizer. "
            "Live generation is the issue #145 handoff — requires OPENAI_API_KEY."
        )
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=SynthesizerConfig.per_type_count
        if not callable(SynthesizerConfig.per_type_count)
        else SynthesizerConfig().per_type_count,
        help="Core Paraphrases per type (Demo tier default: 50).",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=SynthesizerConfig().quality_threshold,
        help="Post-filter quality threshold (default: 0.5).",
    )
    parser.add_argument(
        "--qc-only",
        action="store_true",
        help="Skip generation; just run the QC gate over the committed queries.yaml.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Snapshot docs/fake-docs/ into corpus/ before generation (AC3).",
    )
    args = parser.parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))

    if args.freeze and (args.qc_only or not os.getenv("OPENAI_API_KEY")):
        n = freeze_corpus()
        print(f"Corpus frozen: {n} file(s) copied from {FAKE_DOCS_DIR} → {CORPUS_DIR}.")
        return 0

    if args.qc_only or not os.getenv("OPENAI_API_KEY"):
        if not args.qc_only:
            print(
                "OPENAI_API_KEY absent — cannot run live Synthesizer generation.\n"
                "Running QC gate over the committed queries.yaml instead.\n"
                "For the full Demo-tier live regeneration, see issue #145."
            )
        from .loader import load_paraphrases

        verdicts = run_qc(load_paraphrases())
        _print_qc(verdicts)
        return 1 if any(v.rejected for v in verdicts) else 0

    # --- live generation path (issue #145 handoff) ---
    print("Phase 8.5 S6: DeepEval Synthesizer live generation (issue #144/145)")
    print("  Generator: gpt-4o | Critic: gpt-4o-mini (same-family, PRD #137)")
    print(
        f"  Per-type count: {args.per_type} | Quality threshold: {args.quality_threshold}"
    )

    # AC3: freeze corpus snapshot from docs/fake-docs/
    n_frozen = freeze_corpus()
    print(
        f"Corpus frozen: {n_frozen} file(s) copied from {FAKE_DOCS_DIR} → {CORPUS_DIR}."
    )

    config = SynthesizerConfig(
        per_type_count=args.per_type,
        quality_threshold=args.quality_threshold,
    )

    core = generate_core_via_synthesizer(config)
    verdicts = run_qc(core)
    _print_qc(verdicts)
    rejected_ids = {v.paraphrase_id for v in verdicts if v.rejected}
    core = [p for p in core if p.paraphrase_id not in rejected_ids]

    probes = load_probes(PROBES_PATH)
    full_set = core + probes

    write_text_atomic(
        QUERIES_PATH, render_queries_yaml(full_set, cost_usd="see run log")
    )
    print(
        f"\nWrote {len(full_set)} Paraphrases to {QUERIES_PATH.name} "
        f"({len(core)} core + {len(probes)} probes)."
    )
    print("\nNext step: issue #145 — HITL trust review of the generated queries.yaml.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
