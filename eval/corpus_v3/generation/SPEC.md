# Corpus v3 query generation spec

Issue #660, ADR-0045 Prerequisite 4, PRD #654 user stories 5, 7, 8. This is
the pinned spec a later corpus-build issue's generator script must follow.
The deterministic pieces (schema, overlap classification, QC gate) are
implemented in this package now, ahead of the corpus itself, so the spec
cannot be quietly rewritten once real generation runs and a convenient
shortcut becomes tempting.

## Multi-family requirement

At least two model families, OR one model family plus a human-written slice
(ADR-0045 Prerequisite 3: "a query generator not from the same model family
as B's [the dense stack's] embeddings" — the v2 eval's single-family
paraphrase generation is the bias this removes). Concretely for corpus v3:

- Family A: an OpenAI model (the family already in use for synthesis and
  grounding elsewhere in this project — ADR-0005).
- Family B: either a second, non-OpenAI model family, or a human-written
  query slice authored directly against the gold Sections without any LLM
  involvement.

`Query.generating_family` records which one produced each query
(`"gpt-4o-mini"`, `"human"`, etc. — a free-text field, not a closed
enumeration, since the second family's exact identity is a corpus-build-time
choice, not a schema-time one).

## Overlap stratum: computed, not authored

`generation.overlap.classify_overlap_stratum` runs against the query text
**as generated** (not as intended) immediately after generation, before the
query is committed. This is deliberate: an LLM's own claim about how
"paraphrased" its output is has no calibration, and the v2 eval's
paraphrase-only generation shows what happens when overlap is left
unmeasured — it silently favors whichever stack the generation style
happens to suit. Reference text for the ratio:

- Answerable queries (`factoid`, `cross_doc`, `version_conflict`): the gold
  Section body/bodies (`Query.gold_section_ids`, resolved via the corpus v3
  gold-mapping table — `eval.corpus_v3.gold`).
- `unanswerable` queries: the near-miss distractor Section(s) the query was
  written to resemble but not actually answer (there is no gold Section to
  measure against otherwise). The generator must record which distractor(s)
  it used in `generation_notes`, even though they are never gold Section ids.

## zh slice — its own gates

Per `POWER_ANALYSIS.md`, the zh slice is sized to a relaxed, explicitly
labelled power/MDD target (not silently smaller). It also runs its own QC
check: `generation.qc.check_generated_query`'s language-detection gate
rejects any query whose `language` label disagrees with
`markdown_kb.app.indexer.detect_lang(text)` — this is the mechanical half of
"own gates"; the zh-specific power target is the statistical half.

## LLM-call convention (for the actual generator script)

Follows the existing eval convention
(`eval/paraphrase_comparison/generate_paraphrases.py` /
`generation/synthesizer_pipeline.py`):

- **Seeded and temperature-0** — every generation call is deterministic
  given its prompt, so a re-run reproduces the same draft (modulo API-side
  nondeterminism the project already accepts for temperature-0 calls
  elsewhere).
- **Excluded from the default test suite** — the generator script itself is
  never collected by `pytest` (same convention as
  `generate_paraphrases.py`/`generate_paraphrases_v2.py`: a runnable script,
  not a test module). The deterministic seams it composes
  (`gen_schema.to_query`, `overlap.classify_overlap_stratum`,
  `qc.check_generated_query`) are unit-tested here; LLM output content is
  never asserted (CODING_STANDARD §6.2).
- Sampling which gold Sections each generated query targets should reuse the
  sha256-keyed deterministic-ordering pattern
  (`eval.paraphrase_comparison.generation.sampling.sha256_order`) rather than
  `random`/`hash()`, for the same cross-machine reproducibility reason.

## Out of scope for this issue (#660)

- The generator script itself and any committed query file — both require
  the adversarial corpus v3 fixtures, which are a separate, later issue in
  PRD #654's dependency order ("(f) corpus v3 fixtures", after "(e) power
  analysis + query generation spec").
- Choosing the concrete second model family / human-author process — a
  corpus-build-time decision, not a schema-time one.
