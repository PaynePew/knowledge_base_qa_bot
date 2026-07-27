# Method comparison — strengths, weaknesses, and fit

> Recorded 2026-07-23, rewritten 2026-07-27 against the corpus v3 verdict
> (ADR-0045 → [`eval/corpus_v3/VERDICT.md`](../corpus_v3/VERDICT.md)).
> Synthesizes the internal v2 eval audit (`verdict.md`), `literature.md` (IR
> methodology), and `why-wiki-industry-evidence.md` (curated-layer evidence)
> alongside the corpus v3 numbers. The v2 eval numbers below (n=260, a small
> clean corpus) are kept for provenance and marked *biased* where the v2
> harness audit found a tilt — read them as bounds, not point estimates, and
> as superseded on every content-quality axis corpus v3 measured.
> Retrieval-arm choice and curated-layer choice are TWO INDEPENDENT
> decisions (ADR-0045 distinguishes the killable retrieval arm from the wiki
> storage layer); read the verdict section first.

## Verdict (corpus v3, ADR-0045 pre-registered kill criteria)

Real live run: 3,636 English queries × 4 arms (`wiki`, `rag` = dense over raw
`docs/`, `hybrid` = BM25+dense RRF over `wiki/`, `dense_over_wiki` = the
missing-cell control), $3.97 query-phase spend, 24,863 LLM calls. Full
report: [`eval/corpus_v3/VERDICT.md`](../corpus_v3/VERDICT.md).

**Kill clause — `wiki` killed.** `rag` beat `wiki` on all three content
axes, every McNemar p < 0.0001:

| Axis | wiki | rag | n |
|---|---|---|---|
| contradiction-leak rate (lower is better) | 0.032 | 0.002 | 3,636 |
| correct-refusal rate | 0.845 | 0.971 | 909 |
| grounding-pass rate | 0.726 | 0.784 | 2,727 |

`stack=wiki` is retired as a standalone retrieval option.

**Demote clause — `hybrid` demoted.** `hybrid` needed a significant
advantage over `rag` on contradiction-leak rate — the curated layer's own
home axis — and instead measured significantly *worse* (0.032 vs 0.002,
McNemar p < 0.0001). Every claim below that framed Hybrid as a retrieval or
grounding win is retracted; the layer is a KB-governance demonstration
artifact, not a measured quality win.

**Survival clause — no axis survived.** No wiki-backed stack significantly
beat `rag` on any measured axis.

**Honest limits carried forward:**

1. `dense_over_wiki` ran with no calibrated pre-LLM refusal gate
   (`answer_fn.dense_over_wiki_query`'s documented gap) — read its
   correct-refusal numbers as reflecting synthesis/grounding refusal only,
   not an apples-to-apples gate comparison against `wiki` and `rag`
   (VERDICT.md Honest limits #1).
2. The governance axis — whether the curation workflow (Reconcile,
   fix-source, the Console) saves operator time or catches errors an
   unstructured corpus would miss — is unmeasured. Corpus v3 tested
   content-quality axes only (ADR-0045 Consequences).
3. This verdict covers the 3,636-query English slice only. The
   pre-registered zh slice
   ([`POWER_ANALYSIS.md`](../corpus_v3/POWER_ANALYSIS.md): n=200/stratum,
   power relaxed to 0.70) was never run, so the zh axis of this bilingual
   product remains unmeasured.

## Retrieval arms (v2 numbers; superseded on content-quality axes above)

| | A — BM25 over wiki (retired, kill clause) | B — dense (FAISS) over docs | C — hybrid RRF over wiki (demoted) |
|---|---|---|---|
| v2 hit@3 (Core macro, n=260) | 0.880 *(biased low)* | 0.936 *(biased high)* | 0.924; only significant pair in v2: C > A |
| corpus v3 content axes | lost to B on all three, see Verdict above | winner on all three measured content axes | lost to B on its own home axis, see Verdict above |
| Build cost | wiki synthesis ~$4.4/corpus (shared with C) | embeddings only (cheapest) | wiki synthesis + embeddings (highest) |
| Query cost | zero-cost retrieval; refusal is free (no embedding call) | query embedding per call, paid even on refusal | both arms' costs |
| Strengths (structural, v2) | exact identifiers / entity-centric queries (Sciavolino 2021); no external API dependency at query time; degrades gracefully at index scale (Reimers & Gurevych 2021) | paraphrase / low-lexical-overlap queries (DPR, Ren et al. 2022); best measured hit rate on clean factoid sets; only arm with a measured content-quality win on corpus v3 | covers both query families in v2; RRF needs no score calibration; its only significant v2 win (C > A) no longer implies a content-quality win now that corpus v3 has measured and demoted it |
| Weaknesses | loses paraphrase queries; zh needs its own threshold band (ADR-0014, #261); no significant advantage over B on any corpus v3 content axis (kill clause) | entity-centric misses; degrades with index growth; same-family generator bias inflated its v2 numbers (report.md Lim. 6) | most expensive; complexity of two indexes; no significant advantage over B on its home axis (demote clause); reranker upgrade doesn't fit the 512MB bulkhead (ADR-0019) |
| Best fit | fallback / air-gapped / zero-marginal-cost tiers *(narrow niche only — not a general recommendation after the kill clause)* | the evidence-backed default: numerically highest hit@3 in v2 (not significant over A or C at n=260) and the only arm with a statistically significant content-quality win on corpus v3 | governance/curation substrate for the demo and Console — not a retrieval-quality recommendation |

## Curated wiki layer (governance, not retrieval) — VERDICT.md decision matrix

| Claimed value axis | Evidence status |
|---|---|
| Contradiction control / auditability | corpus v3's home axis, now measured: `wiki` 0.032 vs `rag` 0.002, `hybrid` 0.032 vs `rag` 0.002, both McNemar p < 0.0001 — the curated layer lost its own home axis **[measured-local]** |
| Cross-document sensemaking / global questions | GraphRAG 72–83% comprehensiveness win rate; RAPTOR +20% QuALITY — not measured on a markdown wiki **[measured-analogue]** |
| Query-time token efficiency | GraphRAG 9–43× fewer tokens (analogue); draft input tokens per stratum measured locally this run **[measured-local]** |
| Compounding knowledge across sessions | no head-to-head benchmark found anywhere **[argued]** |
| Known losses regardless of eval | Single-hop factoid directness (GraphRAG concedes; arXiv 2502.11371: RAG wins detail queries); build cost (LazyGraphRAG: full synthesis ≈ 1000× vector-index cost); update amplification + staleness window (ADR-0045 Consequences); summary hallucination amplification (arXiv 2502.11371: 25% on unanswerable) |

## Enterprise chatbot decision framework

Default retrieval: **`rag` (dense over raw docs)** — it is the only arm with
a measured content-quality win on corpus v3's three axes and no significant
loss anywhere measured, v2 or v3. The historical v2 case for Hybrid as an
enterprise default (exact identifiers + paraphrase coverage in one stack) no
longer holds: corpus v3 measured Hybrid losing to `rag` on contradiction-leak
rate, its own strongest claimed axis, so recommending it as a general
default would contradict the evidence this project collected.

Build the curated wiki layer only for its governance value, and only if ALL
FOUR hold — going in expecting a demonstration workflow, not a
content-quality win, because corpus v3 tested exactly this profile
(condition 1) and the layer still lost the contradiction-control axis:

1. **Contradiction-prone corpus** — multi-version policies, overlapping
   owners (compliance, HR, medical SOP: yes; single-source FAQ: no).
2. **Low churn** — update amplification + staleness window make the layer
   perpetually stale on daily-edited corpora.
3. **A knowledge owner exists** — the pattern's own premise (Karpathy:
   humans "curate sources, direct the analysis"); an unowned curated layer
   rots into a distrusted second source of truth.
4. **Single ACL domain** — synthesis pages merge content across source
   documents; per-document access control cannot be enforced on a page that
   blends three differently-permissioned sources (GraphRAG community
   summaries share this problem). Per-ACL wiki partitions or
   strictest-permission inheritance are possible but expensive. A public or
   uniformly-permissioned KB (our demo) sidesteps this; an org-wide internal
   bot usually does not.

Our demo corpus (e-commerce customer-service FAQ: high-churn, single-source,
factoid-dominant, public) fails conditions 1–2, which is why the wiki layer
was never expected to show retrieval value there. Corpus v3 (redundant +
contradictory + versioned) was built to satisfy condition 1 as the fair
trial ADR-0045 pre-registered — and even there, on its own home axis, the
curated layer lost. The architecture keeps all three arms behind one gateway
(`stack=` dispatch), so choosing among them stays a configuration choice per
deployment, not a rewrite (ADR-0003 W2); on the evidence collected so far,
that configuration choice defaults to `rag`, with the wiki layer opted in
only for its governance value.
