⚠️ PLACEHOLDER — NOT A LIVE VERDICT RUN. Every answer scored below is a hand-authored, deterministic offline stand-in (no LLM call was made, no OPENAI_API_KEY touched); the significance tests run REAL math but over a tiny synthetic outcome set sized for pipeline demonstration, not ADR-0045's power-analysis requirement (POWER_ANALYSIS.md: n=909 per scenario stratum). This file exercises axis scoring, paired significance tests, the ADR-0045 clause walkthrough, and rendering end to end at zero cost — it is NOT the corpus v3 verdict, and none of its kill / demote / survive outcomes may be cited as such. A real run (`--mode live --confirm-live`, a cost-guard-cleared pilot ledger, a power-sized generated query set, and a real `answer_fn`) writes the canonical VERDICT.md instead.

# Corpus v3 verdict report

## TL;DR

Kill clause: `wiki` **killed**. Demote clause: `hybrid` **demoted**. 0 survival entries found. See the honest-limits section: this is the offline tracer, not the real verdict.

## Per-axis, per-stratum results

### contradiction_leak_rate

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.167 vs rag 0.667 | n=6 | mcnemar p=0.2500 | — |
| macro | hybrid 0.167 vs rag 0.667 | n=6 | mcnemar p=0.2500 | — |

### correct_refusal_rate

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.750 vs rag 0.250 | n=4 | mcnemar p=0.5000 | — |

### grounding_pass_rate

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.875 vs rag 0.500 | n=8 | mcnemar p=0.2500 | — |

## ADR-0045 clause walkthrough

### Kill clause

> **Kill the retrieval arm** if, on corpus v3, `stack=wiki` shows no statistically significant advantage over `stack=rag` on **all three** content axes — contradiction-leak rate, grounding pass rate, correct-refusal rate. Consequence: `stack=wiki` is retired as a standalone retrieval option; the wiki layer remains as the hybrid embedding substrate and governance surface. (Per ADR-0003's W2 hedge, this retirement is a config/routing change, not a rewrite.)

**Verdict: `wiki` killed.**

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | wiki 0.167 vs rag 0.667 | n=6 | mcnemar p=0.2500 | — |
| macro | wiki 0.750 vs rag 0.250 | n=4 | mcnemar p=0.5000 | — |
| macro | wiki 0.875 vs rag 0.500 | n=8 | mcnemar p=0.2500 | — |

### Demote clause

> **Demote the wiki layer** if C (hybrid-over-wiki) fails to show a statistically significant advantage over B (dense-over-raw-docs) on **contradiction-leak rate** — the curated layer's home axis. Consequence: the layer is repositioned honestly as a demonstration artifact of a KB-governance workflow, not as a measured quality win; code is retained (the demo and Console depend on it) but every narrative claim of retrieval or grounding superiority is dropped.

**Verdict: `hybrid` demoted.**

| Stratum | Rates | n | Test | sig (p<0.05) |
|---|---|---|---|---|
| macro | hybrid 0.167 vs rag 0.667 | n=6 | mcnemar p=0.2500 | — |

### Survival clause

> **Survival:** any axis on which a wiki-backed stack significantly beats B becomes the lead narrative for that stack, with the corpus v3 numbers as backing.

No axis met the survival bar in this run.

## Cost chapter

### Cost per grounded-correct answer

| Arm | USD / grounded-correct answer |
|---|---|
| hybrid | $0.0000 |
| rag | $0.0000 |
| wiki | $0.0000 |

### Build-cost amortization curve

| Arm | n=10 | n=100 | n=1000 |
|---|---|---|---|
| hybrid | $0.4420 | $0.0442 | $0.0044 |
| rag | $0.0020 | $0.0002 | $0.0000 |
| wiki | $0.4400 | $0.0440 | $0.0044 |

## Method-comparison decision matrix (updated, evidence-graded)

| | Evidence status |
|---|---|
| Contradiction control / auditability | corpus v3's home axis; this run's own contradiction-leak rate table is the local measurement **[measured-local]** |
| Cross-document sensemaking / global questions | GraphRAG 72-83% comprehensiveness win rate; RAPTOR +20% QuALITY -- not measured on a markdown wiki **[measured-analogue]** |
| Query-time token efficiency | GraphRAG 9-43x fewer tokens (analogue); draft input tokens per stratum measured locally this run **[measured-local]** |
| Compounding knowledge across sessions | no head-to-head benchmark found anywhere **[argued]** |

## Honest limits

1. Every measured curated-layer analogue (GraphRAG, RAPTOR) is a structural analogue, not a markdown wiki -- the inference gap stated in eval/fairness_review/why-wiki-industry-evidence.md still applies to those rows of the decision matrix.
2. The minimal detectable difference driving the power analysis (POWER_ANALYSIS.md) is a calculation from a normal-approximation closed form, not a citation.
3. High-churn corpora and ACL-partitioned corpora remain unmeasured by design (PRD #654 § Out of Scope) -- the decision matrix's recommendation is bounded to low-churn, single-ACL-domain corpora.
4. Update amplification and the inter-ingest staleness window are known wiki losses ADR-0045 states corpus v3 does not test, regardless of this run's kill/demote/survive outcome.
