# Paraphrase Comparison — reproduce & extend

Maintainer-facing runbook for the Phase 8 / 8.5 Wiki-vs-Vector-RAG retrieval
comparison. The reader-facing *results* live in
[`report.md`](report.md); this file is **how to regenerate and re-run** them.

> All commands run **from the repo root** with [`uv`](https://docs.astral.sh/uv/).
> Live generation/embedding steps need API keys; every step also has an
> offline path so the pipeline stays exercisable without spend.

## What the corpus currently is

| Quantity | Value | Source of truth |
|---|---|---|
| Synthetic Sources (Acme Shop) | **20** | `corpus/*.md` (frozen snapshot) |
| Gold Sections | **51** | `derive_gold_sections(corpus/)` — `generation/sampling.py` |
| Queries | **260** = 250 Core (5 LLM types × 50) + 10 hand-written Structural probes (2 × 5) | `queries.yaml` `metadata.total` |

The Gold Section count is derived live from the frozen corpus — it is **not**
read from `gold_sections.yaml` (that hand-maintained file was superseded by
`derive_gold_sections` in issue #142; its header count may be stale and it is
no longer the source of truth).

## The three stages

```
corpus_generator  →  generate_paraphrases_v2  →  run_comparison
  docs/fake-docs/      freezes corpus/ +            report.md + charts/
                       writes queries.yaml
```

### 1. Regenerate / extend the corpus

Writes the Acme-Shop synthetic Source pool into `docs/fake-docs/`.

```bash
# Deterministic scaffold stubs — no API key:
uv run python -m eval.paraphrase_comparison.generation.corpus_generator

# Realistic LLM prose — requires OPENAI_API_KEY (generator gpt-4o):
uv run python -m eval.paraphrase_comparison.generation.corpus_generator --live

# Regenerate a single doc only (basename without .md, or a title keyword):
uv run python -m eval.paraphrase_comparison.generation.corpus_generator --live --doc warranty
```

To **extend** the corpus, add a `DocSpec` to `DOC_SPECS` in
`generation/corpus_generator.py`, then re-run with `--live`. The Gold Section
pool grows automatically — every body-bearing heading in a non-entity Source
becomes a Gold Section (entity Sources like `warranty.md` are excluded).

### 2. Regenerate the paraphrase query set

Freezes the current `docs/fake-docs/` into `corpus/` (the eval snapshot), then
generates `queries.yaml` via DeepEval's Synthesizer (generator `gpt-4o` +
same-family `gpt-4o-mini` critic; the answer key — Gold Section id + Key Tokens
— is derived deterministically from corpus content, never asserted by the LLM).

```bash
# Offline QC gate over the committed queries.yaml — no API key, no generation:
uv run python -m eval.paraphrase_comparison.generate_paraphrases_v2 --qc-only

# Live generation — requires OPENAI_API_KEY (~$15–30 Demo-tier run, PRD #137):
uv run python -m eval.paraphrase_comparison.generate_paraphrases_v2 --per-type 50

# Quick smoke-check (5 per type):
uv run python -m eval.paraphrase_comparison.generate_paraphrases_v2 --per-type 5
```

`--per-type N` sets Core Paraphrases per type. The 10 hand-written Structural
probes are merged unchanged from `probes.yaml`.

### 3. Re-run the comparison

Indexes both Stacks over `corpus/`, scores with the C5c hit metric
(`hit_rate@k` + MRR), and **writes `report.md` + `charts/*.png`** in this
directory.

```bash
# Real text-embedding-3-small vectors — requires OPENAI_API_KEY:
uv run python -m eval.paraphrase_comparison.run_comparison

# Offline deterministic stand-in (token-overlap ranker) — no API key.
# The report is banner-marked as OFFLINE TRACER NUMBERS, not the real experiment:
uv run python -m eval.paraphrase_comparison.run_comparison --fake-embeddings

# Add the opt-in L2 cross-family spot-check — requires ANTHROPIC_API_KEY:
uv run python -m eval.paraphrase_comparison.run_comparison --judge=claude-sonnet-4-6
```

Other flags: `--k` (hit_rate@k cutoff, default 3), `--judge-zones`,
`--judge-marginal-threshold` (default 1), `--judge-control-sample-size`
(default 5). See `run_comparison.py --help`.

## Where the report lands

- [`report.md`](report.md) — full methodology, per-type tables, McNemar +
  Wilson CI + Holm stats, cost log, limitations, talking points.
- `charts/*.png` — embedded in the report.

The report's narrative counts (Source / Gold Section / per-type query counts)
are **derived from the actual corpus + query set at render time** (issue #145),
so they track the data after a regeneration rather than drifting.
