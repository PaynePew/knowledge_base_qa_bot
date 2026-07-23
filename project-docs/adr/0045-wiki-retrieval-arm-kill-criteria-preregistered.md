# Pre-registered kill criteria for the wiki retrieval arm (corpus v3 verdict)

> **Status:** Accepted (2026-07-23). Pre-registered **before** the corpus v3 adversarial eval runs; the criteria below must not be edited after corpus v3 data exists, only superseded by a new ADR.

The three-stack eval (`eval/paraphrase_comparison/report.md`, n=260) left the wiki's value claim unproven: Core macro hit@3 is A(wiki/BM25) 0.880 / B(rag/dense) 0.936 / C(hybrid) 0.924, the **only statistically significant pair is C > A**, and pure query cost orders B < A < C. Two readings of this data are wrong and this ADR exists to block both. Reading one — "the wiki lost to RAG" — is unsupported: A and B differ in **both** algorithm (BM25 vs dense) and corpus (`wiki/` vs `docs/`), so A's loss cannot be attributed to the wiki layer; the one significant pair (C > A, same corpus, different algorithm) indicts the BM25-only arm, not the curated corpus, and B vs C (docs-dense vs wiki-hybrid) is not significant. Reading two — "the wiki is fine" — is equally unsupported: the wiki's actual value hypothesis (curation pays off on redundant, contradictory, versioned corpora via lower contradiction leakage and cleaner grounding) has never been on trial, because the eval corpus is small and clean, a track where curation cannot show value. The corpus v3 adversarial eval (redundancy + contradictions + version evolution; measuring contradiction-leak rate, grounding pass rate, correct-refusal rate, and draft input tokens) is that trial, and its verdict criteria are fixed here, in advance, so the outcome cannot be rationalized after the fact.

## Decision

Two separately killable things, deliberately distinguished:

- **Wiki retrieval arm** — the `stack=wiki` standalone BM25 query path.
- **Wiki layer** — the curated `wiki/` storage layer itself: Filed Answers, reconcile, the Console curation surface, and the hybrid stack's embedding substrate (ADR-0018: C embeds `wiki/` Sections).

**Kill the retrieval arm** if, on corpus v3, `stack=wiki` shows no statistically significant advantage over `stack=rag` on **all three** content axes — contradiction-leak rate, grounding pass rate, correct-refusal rate. Consequence: `stack=wiki` is retired as a standalone retrieval option; the wiki layer remains as the hybrid embedding substrate and governance surface. (Per ADR-0003's W2 hedge, this retirement is a config/routing change, not a rewrite.)

**Demote the wiki layer** if C (hybrid-over-wiki) fails to show a statistically significant advantage over B (dense-over-raw-docs) on **contradiction-leak rate** — the curated layer's home axis. Consequence: the layer is repositioned honestly as a demonstration artifact of a KB-governance workflow, not as a measured quality win; code is retained (the demo and Console depend on it) but every narrative claim of retrieval or grounding superiority is dropped.

**Survival:** any axis on which a wiki-backed stack significantly beats B becomes the lead narrative for that stack, with the corpus v3 numbers as backing.

Significance means a paired test on the shared query set (McNemar or bootstrap over per-query outcomes) at p < 0.05. Power is a first-class design input, not an afterthought: at the v2 eval's n=260, the minimal detectable difference on a paired binary metric is roughly 6–7 hit@3 points (`eval/fairness_review/literature.md` §2, marked inferred) — larger than the observed A–B gap, which is why that comparison resolved nothing. Corpus v3's query-set size must therefore be derived from a prospective power analysis (Sakai's topic-set-size design) such that the kill threshold exceeds the minimal detectable difference. A non-significant *advantage* still kills (the burden of proof is on the wiki, which carries the extra build cost — ~$4.4/corpus vs B's embedding-only build). The demotion clause carries the same burden: unless C shows a statistically significant advantage over B on contradiction-leak rate, the layer demotes — ties and non-significant differences both demote, because the contradiction axis is where curation claims its strongest ground; if it cannot win even there, the claim has no measurable content.

## Prerequisites (run before the verdict, else the comparison is confounded)

1. **2×2 missing cell**: run dense-over-wiki standalone (the hybrid's dense arm already embeds `wiki/`; evaluate it without RRF) so corpus effect and algorithm effect separate cleanly. Without this, neither kill clause can attribute cause.
2. **Refusal-gate fairness**: A's Cannot Confirm threshold was calibrated (#253/#261); B's distance gate (`vector_rag/app/retrieval.py` distance cutoff) was calibrated in `eval/rag_distance/` (#257/#258, ceiling 1.1); confirm that calibration still holds on the corpus v3 negative set. An uncalibrated gate on one side invalidates the correct-refusal axis.
3. **De-biased harness** (the v2 eval's measured tilts, all favoring B — audit in `eval/fairness_review/`): symmetric gold-label mapping (v2 gold ids are docs-native; wiki entity-page hits are unmappable and score as guaranteed misses, `eval/paraphrase_comparison/stacks.py` `_wiki_slug_to_gold_section`), Key-Token hit condition drawn from both corpora rather than docs-body IDF alone, query provenance stratified by query–document lexical overlap (LLM-paraphrased queries depress overlap and favor dense — Ren et al. 2022, DPR's SQuAD finding), a query generator not from the same model family as B's embeddings, and a zh query slice (v2 is English-only, so the bilingual product's zh retrieval is currently unmeasured).
4. **Power-sized query set**: n derived from a prospective power analysis so the kill threshold exceeds the minimal detectable difference (see Decision); hit@1 and MRR reported alongside hit@3.

## Consequences

- The corpus v3 issue must link this ADR; its report must state each clause's verdict explicitly (kill / demote / survive, per axis, with test statistics).
- Interview and README narrative follow the verdict, not the other way around. Until corpus v3 runs, the honest claim stays: "retrieval-axis winner is dense; the governance axis is unmeasured; the trial and its kill criteria are pre-registered."
- Known axes the wiki loses regardless of corpus v3, to be stated rather than hidden: update amplification (a Source edit re-triggers synthesis + grounding LLM calls, vs cheap re-embedding for B) and the staleness window between ingest runs. Corpus v3 does not test these; they bound the wiki's applicable domain (low-churn, contradiction-prone, audit-requiring corpora) even in the survival case.
