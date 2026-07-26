# Corpus v3 verdict report

## TL;DR

Kill clause: `wiki` **killed**. Demote clause: `hybrid` **demoted**. 0 survival entries found. Real live run over 3636 English queries x 4 arms; actual query-phase spend $3.9675 (24863 LLM calls). See the honest-limits section for the dense_over_wiki correct-refusal caveat.

## Per-axis, per-stratum results

### contradiction_leak_rate

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.032 vs rag 0.002 | n=3636 | mcnemar p=0.0000 | ✓ |
| macro | hybrid 0.032 vs rag 0.002 | n=3636 | mcnemar p=0.0000 | ✓ |
| macro | dense_over_wiki 0.047 vs rag 0.002 | n=3636 | mcnemar p=0.0000 | ✓ |

### correct_refusal_rate

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.845 vs rag 0.971 | n=909 | mcnemar p=0.0000 | ✓ |
| macro | hybrid 0.884 vs rag 0.971 | n=909 | mcnemar p=0.0000 | ✓ |
| macro | dense_over_wiki 0.871 vs rag 0.971 | n=909 | mcnemar p=0.0000 | ✓ |

### grounding_pass_rate

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.726 vs rag 0.784 | n=2727 | mcnemar p=0.0000 | ✓ |
| macro | hybrid 0.715 vs rag 0.784 | n=2727 | mcnemar p=0.0000 | ✓ |
| macro | dense_over_wiki 0.709 vs rag 0.784 | n=2727 | mcnemar p=0.0000 | ✓ |

## ADR-0045 clause walkthrough

### Kill clause

> **Kill the retrieval arm** if, on corpus v3, `stack=wiki` shows no statistically significant advantage over `stack=rag` on **all three** content axes — contradiction-leak rate, grounding pass rate, correct-refusal rate. Consequence: `stack=wiki` is retired as a standalone retrieval option; the wiki layer remains as the hybrid embedding substrate and governance surface. (Per ADR-0003's W2 hedge, this retirement is a config/routing change, not a rewrite.)

**Verdict: `wiki` killed.**

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.032 vs rag 0.002 | n=3636 | mcnemar p=0.0000 | ✓ |
| macro | wiki 0.845 vs rag 0.971 | n=909 | mcnemar p=0.0000 | ✓ |
| macro | wiki 0.726 vs rag 0.784 | n=2727 | mcnemar p=0.0000 | ✓ |

### Demote clause

> **Demote the wiki layer** if C (hybrid-over-wiki) fails to show a statistically significant advantage over B (dense-over-raw-docs) on **contradiction-leak rate** — the curated layer's home axis. Consequence: the layer is repositioned honestly as a demonstration artifact of a KB-governance workflow, not as a measured quality win; code is retained (the demo and Console depend on it) but every narrative claim of retrieval or grounding superiority is dropped.

**Verdict: `hybrid` demoted.**

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | hybrid 0.032 vs rag 0.002 | n=3636 | mcnemar p=0.0000 | ✓ |

### Survival clause

> **Survival:** any axis on which a wiki-backed stack significantly beats B becomes the lead narrative for that stack, with the corpus v3 numbers as backing.

No axis met the survival bar in this run.

## Cost chapter

### Cost per grounded-correct answer

| Arm | USD / grounded-correct answer |
|---|---|
| dense_over_wiki | $0.0005 |
| hybrid | $0.0005 |
| rag | $0.0005 |
| wiki | $0.0005 |

### Build-cost amortization curve

| Arm | n=10 | n=100 | n=1000 |
|---|---|---|---|
| dense_over_wiki | $0.0000 | $0.0000 | $0.0000 |
| hybrid | $0.0000 | $0.0000 | $0.0000 |
| rag | $0.0000 | $0.0000 | $0.0000 |
| wiki | $0.0000 | $0.0000 | $0.0000 |

## Method-comparison decision matrix (updated, evidence-graded)

| | Evidence status |
|---|---|
| Contradiction control / auditability | corpus v3's home axis; this run's own contradiction-leak rate table is the local measurement **[measured-local]** |
| Cross-document sensemaking / global questions | GraphRAG 72-83% comprehensiveness win rate; RAPTOR +20% QuALITY -- not measured on a markdown wiki **[measured-analogue]** |
| Query-time token efficiency | GraphRAG 9-43x fewer tokens (analogue); draft input tokens per stratum measured locally this run **[measured-local]** |
| Compounding knowledge across sessions | no head-to-head benchmark found anywhere **[argued]** |

## Honest limits

1. `dense_over_wiki` has NO calibrated pre-LLM refusal gate (`answer_fn.dense_over_wiki_query`'s documented gap, ADR-0045 Prerequisite 2) -- its correct-refusal-rate numbers below reflect refusal via synthesis/grounding alone, not a gate calibrated against the negative set the way `wiki` and `rag` are. Issue #674's pilot measured 7/8 negatives refused this way; the live numbers may differ. Accepted as a stated caveat rather than a blocking gap (issue #679 authorization, recorded 2026-07-24) -- read this axis's `dense_over_wiki` row with that caveat, not as an apples-to-apples gate comparison.
2. Every measured curated-layer analogue (GraphRAG, RAPTOR) is a structural analogue, not a markdown wiki -- the inference gap stated in eval/fairness_review/why-wiki-industry-evidence.md still applies to those rows of the decision matrix.
3. The minimal detectable difference driving the power analysis (POWER_ANALYSIS.md) is a calculation from a normal-approximation closed form, not a citation.
4. High-churn corpora and ACL-partitioned corpora remain unmeasured by design (PRD #654 § Out of Scope) -- the decision matrix's recommendation is bounded to low-churn, single-ACL-domain corpora.
5. Update amplification and the inter-ingest staleness window are known wiki losses ADR-0045 states corpus v3 does not test, regardless of this run's kill/demote/survive outcome.
