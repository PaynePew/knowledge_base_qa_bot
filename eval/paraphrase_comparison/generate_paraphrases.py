"""One-off CLI to generate the Phase 8 Paraphrase set and write queries.yaml.

Usage (from repo root):

    uv run python -m eval.paraphrase_comparison.generate_paraphrases

NOT run under pytest — it makes real gpt-4o-mini calls (CONTEXT.md § Phase 8 >
Paraphrase). The deterministic seams it composes (sha256 sampling, the Key-Token
QC gate, the structured-output schema) are unit-tested offline; the LLM output
content is non-deterministic and never asserted (CODING_STANDARD §6.2).

Pipeline:

    1. Load the Gold Section inventory (gold_sections.yaml) and, per Core
       Paraphrase Type, sha256-sample which Gold Sections to target
       (generation.sampling — specificity_narrowing samples multi-sub-fact
       sections only; cross-type reuse allowed).
    2. For each (type, section), render the type's per-type prompt template
       over the docs Section body + concept Wiki Page body, and call gpt-4o-mini
       (temperature=0.7, seed=42, one-shot) with with_structured_output into the
       ParaphraseDraft schema (ADR-0005 structured-output adapter).
    3. Stitch generator-owned id/type/gold onto each draft (gen_schema.to_paraphrase).
    4. Run the Key-Token QC gate (generation.qc): reject all-stopword sets,
       flag low-distinctiveness tokens for human PR review.
    5. Merge the two hand-written probe types (typo_fatfinger, industry_jargon)
       from probes.yaml unchanged (NOT LLM-generated).
    6. Write queries.yaml atomically with a metadata block (model, timestamp,
       seed, prompt-template version, corpus snapshot git sha, total, cost).

OFFLINE (no OPENAI_API_KEY): the generator cannot make the gpt-4o-mini calls, so
it does NOT overwrite the committed, hand-authored queries.yaml. It instead runs
the QC gate over the committed set and prints the flagged-for-review report, so
the deterministic half of the pipeline is still exercisable. Re-running WITH a key
performs a full regeneration.

This is a one-off script, so stdout via ``print`` is acceptable (CODING_STANDARD
§5.1 — the no-print rule is scoped to committed library code).
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import yaml

from markdown_kb.app.indexer import parse_markdown, slugify

from .generation import qc, sampling
from .generation.gen_schema import ParaphraseDraft, to_paraphrase
from .generation.templates import CORE_TEMPLATES, TEMPLATE_VERSION
from .loader import QUERIES_PATH, load_paraphrases, write_text_atomic
from .models import PARAPHRASE_TYPES, Paraphrase

_PKG_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = _PKG_ROOT / "corpus"
WIKI_CONCEPTS_DIR = _PKG_ROOT / "wiki" / "concepts"
PROBES_PATH = _PKG_ROOT / "probes.yaml"

GENERATOR_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
SEED = 42
PER_TYPE_COUNT = 9  # Core types; total core ≈ 5 × 9 = 45 (+ probes → ~39-54 set)

CORE_TYPES = tuple(CORE_TEMPLATES.keys())
PROBE_TYPES = tuple(t for t in PARAPHRASE_TYPES if t not in CORE_TEMPLATES)


# ---------------------------------------------------------------------------
# LLM singleton (lazy — ADR-0005; only constructed when actually generating)
# ---------------------------------------------------------------------------
def _get_generator_llm():
    """Return a gpt-4o-mini client bound to the ParaphraseDraft schema.

    Lazy + function-scoped LangChain import so the module imports offline and the
    LangChain type never leaves this script (CODING_STANDARD §2.4 — this is an
    LLM-facing one-off; types stay internal). temperature/seed are pinned for
    run-to-run reproducibility of the generation (PRD #100).
    """
    from langchain_openai import ChatOpenAI  # function-scope: keep LangChain internal

    llm = ChatOpenAI(
        model=GENERATOR_MODEL,
        temperature=TEMPERATURE,
        model_kwargs={"seed": SEED},
        timeout=60,
        max_retries=1,
    )
    return llm.with_structured_output(ParaphraseDraft)


# ---------------------------------------------------------------------------
# Corpus access
# ---------------------------------------------------------------------------
def _docs_section_bodies() -> dict[str, tuple[str, str]]:
    """Map each docs Gold Section id to ``(heading, body)`` from the corpus."""
    bodies: dict[str, tuple[str, str]] = {}
    for md_file in sorted(CORPUS_DIR.glob("*.md")):
        for section in parse_markdown(md_file, source_id=None):
            if section.content.strip():
                bodies[f"{md_file.name}#{slugify(section.heading)}"] = (
                    section.heading,
                    section.content,
                )
    return bodies


def _concept_body(slug: str) -> str:
    """Return the prose body of a concept Wiki Page (heading + citation stripped)."""
    raw = (WIKI_CONCEPTS_DIR / f"{slug}.md").read_text(encoding="utf-8")
    # Drop the sentinel comment + frontmatter (everything up to the 2nd '---'),
    # the leading '# heading', and the trailing '[Source: ...]' citation line.
    after_fm = raw.split("\n---\n", 2)[-1]
    lines = [
        ln
        for ln in after_fm.splitlines()
        if ln.strip() and not ln.startswith("# ") and not ln.startswith("[Source:")
    ]
    return " ".join(lines).strip()


# ---------------------------------------------------------------------------
# Core generation (LLM)
# ---------------------------------------------------------------------------
def generate_core(per_type_count: int = PER_TYPE_COUNT) -> list[Paraphrase]:
    """Generate the five Core Paraphrase Types via their per-type templates.

    Deterministic sampling picks the target Gold Sections (seed = type name);
    each (type, section) yields one gpt-4o-mini structured-output call. Requires
    OPENAI_API_KEY — callers gate on it (see ``main``).
    """
    gold = sampling.load_gold_sections()
    docs_bodies = _docs_section_bodies()
    llm = _get_generator_llm()

    out: list[Paraphrase] = []
    for ptype in CORE_TYPES:
        build_prompt = CORE_TEMPLATES[ptype]
        sections = sampling.sample_sections(
            gold,
            seed=ptype,
            count=per_type_count,
            multi_sub_fact_only=(ptype == "specificity_narrowing"),
        )
        for idx, sec in enumerate(sections, start=1):
            heading, docs_body = docs_bodies[sec.section_id]
            wiki_body = _concept_body(sec.concept_slug)
            prompt = build_prompt(heading=heading, body=f"{docs_body}\n\n(Wiki phrasing: {wiki_body})")
            draft: ParaphraseDraft = llm.invoke(prompt)  # type: ignore[assignment]
            out.append(
                to_paraphrase(
                    draft,
                    paraphrase_id=f"{ptype}-{idx:03d}",
                    paraphrase_type=ptype,  # type: ignore[arg-type]
                    gold_docs_section_id=sec.section_id,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Probes (hand-written) + metadata
# ---------------------------------------------------------------------------
def load_probes(path: Path = PROBES_PATH) -> list[Paraphrase]:
    """Load the two hand-written probe types (typo_fatfinger, industry_jargon)."""
    return load_paraphrases(path)


def corpus_snapshot_sha() -> str:
    """Return the current git HEAD short sha (corpus snapshot id), or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_PKG_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def render_queries_yaml(
    paraphrases: list[Paraphrase],
    *,
    cost_usd: str,
) -> str:
    """Render the full queries.yaml: a metadata block + all Paraphrase entries."""
    metadata = {
        "generator_model": GENERATOR_MODEL,
        "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "temperature": TEMPERATURE,
        "prompt_template_version": TEMPLATE_VERSION,
        "corpus_snapshot_git_sha": corpus_snapshot_sha(),
        "total": len(paraphrases),
        "cost_usd": cost_usd,
    }
    entries = [
        {
            "paraphrase_id": p.paraphrase_id,
            "paraphrase_type": p.paraphrase_type,
            "text": p.text,
            "gold_docs_section_id": p.gold_docs_section_id,
            "key_tokens_docs": list(p.key_tokens_docs),
            "key_tokens_wiki": list(p.key_tokens_wiki),
            **({"generation_notes": p.generation_notes} if p.generation_notes else {}),
        }
        for p in paraphrases
    ]
    return yaml.dump(
        {"metadata": metadata, "paraphrases": entries},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


# ---------------------------------------------------------------------------
# QC reporting
# ---------------------------------------------------------------------------
def run_qc(paraphrases: list[Paraphrase]) -> list[qc.QcVerdict]:
    """Run the Key-Token QC gate over a Paraphrase set; return all verdicts.

    The IDF table spans BOTH Stack surfaces — docs Section bodies AND concept
    Wiki Page bodies — because the C5c metric overlaps Key Tokens against the
    retrieved *content* of either Stack (Stack B returns docs Chunks, Stack A
    returns wiki Sections). A token present only on one surface is still
    matchable, so the distinctiveness judgement must see both corpora.
    """
    docs_bodies = _docs_section_bodies()
    wiki_bodies = [
        _concept_body(s.concept_slug) for s in sampling.load_gold_sections()
    ]
    idf = qc.build_idf([body for _, body in docs_bodies.values()] + wiki_bodies)
    return [
        qc.check_key_tokens(p.paraphrase_id, sorted(p.key_tokens), idf) for p in paraphrases
    ]


def _print_qc(verdicts: list[qc.QcVerdict]) -> None:
    rejected = [v for v in verdicts if v.rejected]
    flagged = [v for v in verdicts if v.flagged_tokens and not v.rejected]
    print(f"  QC: {len(verdicts)} checked, {len(rejected)} rejected, {len(flagged)} flagged.")
    for v in rejected:
        print(f"    REJECT {v.paraphrase_id}: {'; '.join(v.reasons)}")
    for v in flagged:
        print(f"    FLAG   {v.paraphrase_id}: {'; '.join(v.reasons)}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 Paraphrase generator.")
    parser.add_argument(
        "--per-type", type=int, default=PER_TYPE_COUNT, help="Core Paraphrases per type."
    )
    parser.add_argument(
        "--qc-only",
        action="store_true",
        help="Skip generation; just run the QC gate over the committed queries.yaml.",
    )
    args = parser.parse_args(argv)

    if args.qc_only or not os.getenv("OPENAI_API_KEY"):
        if not args.qc_only:
            print(
                "OPENAI_API_KEY absent — cannot run gpt-4o-mini generation. Running the "
                "QC gate over the committed (hand-authored) queries.yaml instead; the "
                "committed set is NOT overwritten offline."
            )
        verdicts = run_qc(load_paraphrases())
        _print_qc(verdicts)
        return 1 if any(v.rejected for v in verdicts) else 0

    # --- live generation path ---
    core = generate_core(args.per_type)
    verdicts = run_qc(core)
    _print_qc(verdicts)
    rejected_ids = {v.paraphrase_id for v in verdicts if v.rejected}
    core = [p for p in core if p.paraphrase_id not in rejected_ids]

    probes = load_probes()
    full_set = core + probes
    # Cost is recorded by the live caller's billing; this offline-safe script does
    # not estimate it — a real run wires the token usage through here.
    write_text_atomic(QUERIES_PATH, render_queries_yaml(full_set, cost_usd="see run log"))
    print(f"Wrote {len(full_set)} Paraphrases to {QUERIES_PATH.name} ({len(core)} core + {len(probes)} probes).")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
