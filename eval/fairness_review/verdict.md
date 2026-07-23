# Fairness verdict on the v2 three-stack eval

> Synthesis, 2026-07-23. Combines the internal harness audit (this file's §1
> evidence lives in `eval/paraphrase_comparison/`), the IR-methodology survey
> (`literature.md`), and the curated-layer evidence survey
> (`why-wiki-industry-evidence.md`). Feeds ADR-0045's Prerequisites.

## Verdict in one paragraph

The v2 eval (n=260, hit@3) is a usable **stack** comparison but not a fair
**BM25-vs-dense** or **wiki-vs-RAG** verdict, and its measured tilts all point
the same way — against Stack A. Query provenance (gpt-4o paraphrases of docs
Sections), gold-label mapping (docs-native ids; wiki entity-page hits
unmappable → guaranteed miss), Key-Token matching (docs-body IDF), corpus
scale (~10² retrieval units — dense retrieval's best regime per Reimers &
Gurevych 2021), and generator/embedding family overlap (gpt-4o →
OpenAI-embedded Stack B; report.md Limitation 6) each independently favor B.
Meanwhile the observed A–B gap (5.6 points) sits below the ~6–7 point minimal
detectable difference at n=260, which is why it is not significant. The one
clean result survives: C > A (same corpus, same mapping, algorithm-only
contrast) — the BM25-only arm is the weak link, not the wiki corpus.

## Per-axis findings

| Axis | Suspicion | Verdict | Key evidence |
|---|---|---|---|
| Quantity (n=260) | fair? | n exceeds TREC's 50-topic norm, but MDD ≈ 6–7 pts > observed 5.6-pt gap; and the set is English-only, so zh retrieval is unmeasured | `literature.md` §2; `queries.yaml` (no `lang`, no CJK) |
| Length | fair? | Unfair as an algorithm test: retrieval units differ (B: 500-char chunks; A: ~1,181-char Sections), violating BEIR's same-corpus premise; tiny index also flatters dense | `literature.md` §1, scope note; `vector_rag/app/indexer.py` CHUNK_SIZE |
| k=3 | fair? | Defensible: matches what the production answerer consumes; sweep (1,3,5,10) + MRR exist; report hit@1/MRR alongside and note the 3/N ≈ 5% chance floor | `runner.py` PRIMARY_CUTOFF; `literature.md` §3 |
| Query provenance | (not originally suspected) | The largest bias, unidirectional toward B: paraphrase-depressed lexical overlap + docs-native gold + docs-IDF Key Tokens + same-family generator | `stacks.py` `_wiki_slug_to_gold_section`; `literature.md` §4; report.md Limitations 6 |

## What this does NOT change

- C > A stands: both arms share corpus, mapping, and Key-Token rules, so the
  significant pair is bias-symmetric. The BM25-only arm's weakness is real.
- Cost ordering B < A < C stands; it never depended on the query set.
- The wiki's governance-axis value remains untested either way (corpus v3's
  job, per ADR-0045). Industry-measured analogues (GraphRAG, RAPTOR) predict
  synthesis layers win global/multi-hop/sensemaking queries and lose
  single-hop factoid lookup — the v2 query set is 100% single-hop factoid
  paraphrases, i.e. exactly the class the literature predicts a curated layer
  cannot win (`why-wiki-industry-evidence.md`).

## Consequences (wired into ADR-0045 Prerequisites 3–4)

Corpus v3's harness must fix, before any kill clause may be invoked:
symmetric gold mapping (incl. entity pages), two-corpus Key-Token union,
overlap-stratified and family-diverse query generation, a zh slice, and a
power-sized n with hit@1/MRR reported alongside hit@3.
