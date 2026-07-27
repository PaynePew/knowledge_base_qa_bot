# Why Wiki (post-verdict): a KB-governance layer, not a measured retrieval win

This project built a curated, LLM-maintained Wiki layer over an immutable Source corpus, rather than a plain vector RAG pipeline. A third **Hybrid** stack was added later (ADR-0018) that runs a dense arm over the same curated layer, fused with BM25 by Reciprocal Rank Fusion. This page keeps the honest historical arc: why the layer was built, what an earlier small eval could and couldn't say about it, what the pre-registered corpus v3 trial then measured, and what survives now that the verdict is in. Full report and per-axis tables: [`../eval/corpus_v3/VERDICT.md`](../eval/corpus_v3/VERDICT.md). For the underlying design decisions, follow the ADR links below.

## The core tradeoff: when synthesis happens

All three approaches let you answer questions over a document collection. They differ in **when synthesis happens** — an architecture description, not a quality ranking:

- **RAG**: synthesis at query time. Embed the query, retrieve top-k chunks, ask the LLM to assemble an answer. Every query re-derives the answer from raw text.
- **Wiki**: synthesis at ingest time. The LLM reads each Source once, writes a structured page into `wiki/`, and queries read the pre-synthesised page. The Source stays immutable; the Wiki is the compounding artifact.
- **Hybrid**: synthesis at ingest time, like Wiki, with a dense-vector arm added over those same wiki pages at query time. It fuses the BM25 and dense rankings with Reciprocal Rank Fusion, so a relevant page that one method ranks low still surfaces ([ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md)).

Which of these produces the *better* answer is a measurement question the timing description doesn't settle by itself. This project ran that measurement: [ADR-0045](adr/0045-wiki-retrieval-arm-kill-criteria-preregistered.md) pre-registered the trial's kill/demote/survive criteria before the corpus v3 data existed, so the outcome below couldn't be rationalized after the fact.

## What corpus v3 measured

Corpus v3 is adversarial by design — redundant, contradictory, and version-evolving Sources, the track where curation is supposed to earn its keep — and it ran for real: 3,636 English queries × 4 arms (`wiki`, `rag` = dense retrieval over raw `docs/`, `hybrid` = BM25+dense RRF over `wiki/`, `dense_over_wiki` = the missing-cell control that isolates corpus effect from algorithm effect), $3.97 in query-phase spend, 24,863 LLM calls. Full report: [`../eval/corpus_v3/VERDICT.md`](../eval/corpus_v3/VERDICT.md).

**Kill clause — `wiki` killed.** `stack=wiki` needed a significant advantage over `stack=rag` on at least one of three content axes to survive. It lost all three, every McNemar p < 0.0001:

| Axis | wiki | rag | n |
| --- | --- | --- | --- |
| contradiction-leak rate (lower is better) | 0.032 | 0.002 | 3,636 |
| correct-refusal rate | 0.845 | 0.971 | 909 |
| grounding-pass rate | 0.726 | 0.784 | 2,727 |

`stack=wiki` is retired as a standalone retrieval option.

**Demote clause — `hybrid` demoted.** `stack=hybrid` needed a significant advantage over `rag` on contradiction-leak rate — the curated layer's own home axis — and instead measured significantly *worse* (0.032 vs 0.002, McNemar p < 0.0001). Every retrieval- or grounding-superiority claim this page used to make about Hybrid is retracted.

**Survival clause — no axis survived.** No wiki-backed stack significantly beat `rag` on any measured axis.

**Honest limits.** `dense_over_wiki` ran with no calibrated pre-LLM refusal gate, so its correct-refusal numbers reflect synthesis/grounding refusal only, not an apples-to-apples gate comparison against `wiki` and `rag`. The governance axis itself — whether the curation workflow saves operator time or catches errors an unstructured corpus would miss — remains unmeasured; corpus v3 tested content-quality axes only. The pre-registered zh slice ([`POWER_ANALYSIS.md`](../eval/corpus_v3/POWER_ANALYSIS.md): n=200/stratum, power relaxed to 0.70) never ran, so the zh axis of this bilingual product remains unmeasured too.

## Where the layer stands now

Per the demote clause, the wiki layer is repositioned honestly: **a demonstration artifact of a KB-governance workflow** — curation, provenance, and contradiction management via `/lint` and the Operator Console's Reconcile flow — not a measured quality win. The code is retained because the Hybrid stack and the Console still depend on it (Hybrid embeds `wiki/` Sections; ADR-0018), but no narrative claim of retrieval or grounding superiority survives this page.

The industry rationale that originally motivated building the layer — Karpathy's LLM Wiki pattern, GraphRAG-style derived summaries — is recorded separately in [`../eval/fairness_review/why-wiki-industry-evidence.md`](../eval/fairness_review/why-wiki-industry-evidence.md), re-scoped after this verdict as motivation for the governance value proposition and marked argued rather than measured.

## Architecture comparison (mechanism, not a quality ranking)

The table below describes how each stack works. Read it as an architecture reference, not a superiority claim — the measured quality comparison is the section above.

| Dimension | LLM Wiki | Vector RAG | Hybrid |
|---|---|---|---|
| **Storage shape** | Pre-compiled `wiki/entities/`, `wiki/concepts/` markdown with cross-references and a generated `wiki/index.md` | Raw chunks in a vector database, no inter-chunk structure | The same `wiki/` pages as the Wiki column, plus a dense index built over those Sections |
| **Finding answers** | Read `wiki/index.md`, follow links, answer from the Wiki page + cited Source | Embed query, similarity search, assemble from top-k chunks | BM25 and dense search over the wiki Sections, fused by RRF |
| **Provenance** | Section-level: every Wiki page declares its `sources:` in frontmatter; every answer cites `filename#heading` ([ADR-0001](adr/0001-strict-grounded-answers.md)) | Chunk-level + similarity score; section boundaries usually lost | Section-level, identical to Wiki (it reuses the wiki citation) |
| **Verifier integration** | Grounding Check runs on every produced Wiki page at ingest; failed pages get `status: failed_grounding` and are filtered from query-time reads ([ADR-0004](adr/0004-post-llm-grounding-check.md)) | Verifier runs only at query time (over the assembled answer), not over the retrieval substrate | Same as Wiki: it indexes the grounding-checked wiki pages |
| **Contradiction detection** | Cross-page scan during `/lint` (Phase 5) over structured pages with frontmatter and `[[wikilinks]]` | Manual; chunks have no relationship metadata | Same as Wiki (it operates on the wiki pages) |
| **Refresh semantics** | Full rewrite of a Source's derived pages, with orphan deletion on structure change ([Phase 3 Q7](roadmap.md)) | Chunk-level dedup is painful; partial re-embeds drift over time | Same as Wiki, plus a dense-index rebuild from the wiki Sections |
| **Query cost** | Low: synthesis is already done, so a query pays for retrieval + final answer only | Higher: every query pays for embedding + assembly | A per-query embedding cost on the dense arm, like RAG, on top of BM25 |
| **Setup floor** | Index file is enough for navigation under ~1000 pages | Embedding model + vector DB + chunking pipeline required from day 1 | Wiki layer + an embedding model + a dense index over the Sections |
| **Scale ceiling** | Hundreds to low thousands of pages (index-file navigation cost) | Millions of documents | The BM25 arm scales like Wiki; the dense arm like RAG |

## Earlier eval (v2), superseded on content-quality claims

Before corpus v3 existed, this page reported a smaller 260-query eval over one clean 20-Source FAQ corpus and used it to call Wiki "the load-bearing choice." That conclusion doesn't survive: the v2 corpus was small and clean, precisely the track where curation cannot show its value (ADR-0045), so the comparison was never a fair test of the wiki's actual value proposition. It stands only as a retrieval-arm comparison at small scale, kept here for provenance:

On 250 Core paraphrases, macro-average hit@3 was **Wiki 0.880 vs RAG 0.936 vs Hybrid 0.924** (real OpenAI embeddings; full method in [`../eval/paraphrase_comparison/report.md`](../eval/paraphrase_comparison/report.md)). A three-way Cochran's Q omnibus was significant (Q = 7.95, p = 0.019); after Holm correction the only pairwise gap that survived was Hybrid > Wiki (p = 0.010) — Wiki vs RAG (p = 0.077) and Hybrid vs RAG (p = 0.71) were statistically indistinguishable at this scale. Per Core type (n=50 each, paired McNemar with Holm correction across the 5 types), the largest raw Wiki-vs-RAG gap was `synonym_swap` (Wiki 0.840 vs RAG 0.940) — but that gap was not significant after Holm correction (p = 0.898): statistically indistinguishable, not a documented wiki loss. Separately, the two hand-written Structural probes (n=5 each, descriptive-only, never averaged into the Core story) are where the wiki actually lost, decisively: `typo_fatfinger` (Wiki 0.200 vs RAG 0.800 vs Hybrid 0.400) and `industry_jargon` (Wiki 0.400 vs RAG 1.000 vs Hybrid 0.600). None of this — Core or probes — speaks to the contradiction-control, correct-refusal, or grounding-pass axes corpus v3 measured, where the wiki lost. The corpus v3 verdict above supersedes this eval on every content-quality claim.

## What the table understates

The "just markdown files" framing for the Wiki column is misleading without context. The Wiki path is cheap to *store* but the ingest pipeline is not trivial — and this operational cost is now better understood as the price of the governance workflow, not as an investment that bought retrieval quality:

- 7-field frontmatter schema (`id`, `type`, `created`, `updated`, `sources`, `status`, `open_questions`); see [Phase 3 grill notes](roadmap.md#phase-3-q1-q3-resolved-2026-05-26).
- Grounding Check verifier on every produced page ([ADR-0004](adr/0004-post-llm-grounding-check.md)).
- Wiki Log with 5 ingest event kinds for audit and `/lint` consumption.
- Collision rule (`-2`, `-3` suffix), red-link convention, orphan deletion on re-ingest.

The Wiki pattern moves complexity from query time (RAG) to ingest time (Wiki). That complexity does not disappear; it concentrates where it can be audited, verified, and lint-checked — the governance surface the layer is now scoped to, per the demote clause above.

## See also

- [ADR-0001 — Strict grounded answers](adr/0001-strict-grounded-answers.md)
- [ADR-0002 — Two parallel retrieval apps](adr/0002-two-parallel-retrieval-apps.md)
- [ADR-0003 — W2 layered Wiki target (claude-obsidian)](adr/0003-w2-layered-wiki-target-claude-obsidian.md)
- [ADR-0004 — Post-LLM Grounding Check](adr/0004-post-llm-grounding-check.md)
- [ADR-0005 — Borrow components, keep opinions](adr/0005-framework-integration-borrow-components-keep-opinions.md)
- [ADR-0006 — W1 after Phase 3](adr/0006-w1-after-phase-3.md)
- [ADR-0018 — Hybrid retrieval, a third stack (RRF over Wiki)](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md)
- [ADR-0045 — Wiki retrieval-arm kill criteria, pre-registered](adr/0045-wiki-retrieval-arm-kill-criteria-preregistered.md)
- [`../eval/corpus_v3/VERDICT.md`](../eval/corpus_v3/VERDICT.md) — the measured verdict
- [Roadmap](roadmap.md)
