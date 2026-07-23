# Method comparison — strengths, weaknesses, and fit

> Recorded 2026-07-23, synthesizing the internal v2 eval audit, `literature.md`
> (IR methodology), and `why-wiki-industry-evidence.md` (curated-layer
> evidence). Numbers marked *biased* carry the v2 harness tilts documented in
> `verdict.md` — read them as bounds, not point estimates. Retrieval-arm
> choice and curated-layer choice are TWO INDEPENDENT decisions (ADR-0045
> distinguishes the killable retrieval arm from the wiki storage layer).

## Retrieval arms

| | A — BM25 over wiki | B — dense (FAISS) over docs | C — hybrid RRF over wiki |
|---|---|---|---|
| v2 hit@3 (Core macro, n=260) | 0.880 *(biased low)* | 0.936 *(biased high)* | 0.924; only significant pair: C > A |
| Build cost | wiki synthesis ~$4.4/corpus (shared with C) | embeddings only (cheapest) | wiki synthesis + embeddings (highest) |
| Query cost | zero-cost retrieval; refusal is free (no embedding call) | query embedding per call, paid even on refusal | both arms' costs |
| Strengths | exact identifiers / entity-centric queries (Sciavolino 2021); no external API dependency at query time; degrades gracefully at index scale (Reimers & Gurevych 2021) | paraphrase / low-lexical-overlap queries (DPR, Ren et al. 2022); best measured hit rate on clean factoid sets | covers both query families; RRF needs no score calibration; only arm with a significant win in v2 |
| Weaknesses | loses paraphrase queries; zh needs its own threshold band (ADR-0014, #261); weakest measured arm | entity-centric misses; degrades with index growth; same-family generator bias inflated its v2 numbers (report.md Lim. 6) | most expensive; complexity of two indexes; reranker upgrade doesn't fit the 512MB bulkhead (ADR-0019) |
| Best fit | fallback / air-gapped / zero-marginal-cost tiers | clean, high-churn, factoid-dominant corpora | DEFAULT for enterprise query mixes (jargon + paraphrase) |

## Curated wiki layer (governance, not retrieval)

| Claimed value axis | Evidence status |
|---|---|
| Cross-document sensemaking / global questions | MEASURED on structural analogues: GraphRAG 72–83% comprehensiveness win rate vs vector RAG (arXiv 2404.16130); RAPTOR +20% QuALITY (arXiv 2401.18059). Not yet measured on a markdown wiki — inference gap stated in `why-wiki-industry-evidence.md`. |
| Query-time token efficiency (pre-built synthesis) | MEASURED on analogues: 9–43× fewer tokens (GraphRAG Table 2). Our corpus v3 measures draft input tokens to test locally. |
| Contradiction control / auditability | ARGUED (Karpathy gist; claude-obsidian). This is corpus v3's verdict axis (ADR-0045). |
| Compounding knowledge across sessions | ARGUED only; no head-to-head benchmark found anywhere. |
| Known losses regardless of eval | Single-hop factoid directness (GraphRAG concedes; arXiv 2502.11371: RAG wins detail queries); build cost (LazyGraphRAG: full synthesis ≈ 1000× vector-index cost); update amplification + staleness window (ADR-0045 Consequences); summary hallucination amplification (arXiv 2502.11371: 25% on unanswerable). |

## Enterprise chatbot decision framework

Default retrieval: **hybrid (C)** — enterprise query mixes contain both exact
identifiers (form codes, SKUs, internal jargon: BM25's home) and natural-
language paraphrase (dense's home). Industry enterprise-search defaults
(Azure AI Search, Elastic) are hybrid+RRF for the same reason.

Build the curated layer only if ALL FOUR hold; otherwise RAG-over-sources
with good provenance is the honest recommendation:

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
factoid-dominant, public) fails conditions 1–2 — which is why the wiki layer
cannot show retrieval value here and why corpus v3 (redundant + contradictory
+ versioned) is the fair trial. The architecture keeps all three arms behind
one gateway (`stack=` dispatch), so this framework is a configuration choice
per deployment, not a rewrite (ADR-0003 W2).
