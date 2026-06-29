# Why Wiki (and where RAG and Hybrid still win)

This project builds a curated, LLM-maintained Wiki layer over an immutable Source corpus, rather than a plain vector RAG pipeline. A third **Hybrid** stack was added later (ADR-0018) that runs both methods over the same curated layer. This page is the short version of why, updated with the three-arm comparison. For the actual design decisions, follow the ADR links.

## The core tradeoff

All three approaches let you answer questions over a document collection. They differ in **when synthesis happens**.

- **RAG**: synthesis at query time. Embed the query, retrieve top-k chunks, ask the LLM to assemble an answer. Every query re-derives the answer from raw text.
- **Wiki**: synthesis at ingest time. The LLM reads each Source once, writes a structured page into `wiki/`, and queries read the pre-synthesised page. The Source stays immutable; the Wiki is the compounding artifact.
- **Hybrid**: synthesis at ingest time, like Wiki, with a dense-vector arm added over those same wiki pages at query time. It fuses the BM25 and dense rankings with Reciprocal Rank Fusion, so a relevant page that one method ranks low still surfaces ([ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md)).

For this project's corpus (under 1000 pages, curated FAQ / policy / KB content), Wiki is the load-bearing choice. Not because it out-retrieves: on the three-arm benchmark the stacks land close on natural questions, and Wiki trades a few points of recall for zero per-query cost, an inspectable index, and grounding-checked provenance. RAG and Hybrid retrieve a little better on synonym-heavy queries, and Hybrid keeps the wiki-Section citation while doing so.

## Comparison

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

## What the three-arm comparison showed

The Phase 8 comparison was re-run as a three-arm benchmark with real OpenAI embeddings (full method, statistics, and cost log in [`../eval/paraphrase_comparison/report.md`](../eval/paraphrase_comparison/report.md)).

On the 250 Core paraphrases, the macro-average hit@3 was **Wiki 0.880 vs RAG 0.936 vs Hybrid 0.924**. A three-way Cochran's Q omnibus was significant (Q = 7.95, p = 0.019), so post-hoc pairwise McNemar tests were warranted. After Holm correction the **only** pairwise gap that survived was **Hybrid > Wiki** (p = 0.010); Wiki vs RAG (p = 0.077) and Hybrid vs RAG (p = 0.71) were statistically indistinguishable.

The picture, then:

- On natural questions the three stacks are close, and the Wiki vs RAG difference does not survive correction at this corpus size.
- The two hand-written structural probes (n=5 each, descriptive only) are where they separate. On the unseen-jargon probe, Wiki 0.40 vs RAG 1.00 vs Hybrid 0.60; on the typo probe, Wiki 0.20 vs RAG 0.80 vs Hybrid 0.40. This is the keyword-miss failure mode RRF is meant to soften, and Hybrid does recover part of it.
- Cost is asymmetric and the hit-rate table hides it. Wiki pays a one-shot synthesis cost at ingest and then retrieves for free; RAG and Hybrid pay a per-query embedding cost forever.

## Verdict

- **Under ~1000 pages** → Wiki. Index navigation is cheap, the pre-compiled synthesis means every query benefits from everything previously read, provenance stays structured, and the recall gap to the dense stacks is small and not significant after correction.
- **When keyword recall misses** (synonyms, unseen jargon) → Hybrid (now built, [ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md)). It keeps the curated Wiki layer and its citation, adds a dense arm over those same wiki Sections, and fuses with RRF, buying back most of RAG's recall edge without giving up the structured provenance.
- **Over ~100K pages** → RAG. The index becomes too large to read, and embedding-based retrieval is more efficient than full-index scanning.

For this project, the corpus is curator-bounded (FAQ / policy / customer-support), so Wiki is the load-bearing choice and Hybrid is the pragmatic upgrade when a query's vocabulary drifts from the source. The earlier "preserve `vector_rag/` for a comparison once the Wiki layer is mature" plan has been carried out: that comparison is the three-arm benchmark above. See [ADR-0002](adr/0002-two-parallel-retrieval-apps.md) and Phase 7 (Paraphrase Comparison) in [roadmap.md](roadmap.md).

## What the table understates

The "just markdown files" framing for the Wiki column is misleading without context. The Wiki path is cheap to *store* but the ingest pipeline is not trivial:

- 7-field frontmatter schema (`id`, `type`, `created`, `updated`, `sources`, `status`, `open_questions`); see [Phase 3 grill notes](roadmap.md#phase-3-q1-q3-resolved-2026-05-26).
- Grounding Check verifier on every produced page ([ADR-0004](adr/0004-post-llm-grounding-check.md)).
- Wiki Log with 5 ingest event kinds for audit and `/lint` consumption.
- Collision rule (`-2`, `-3` suffix), red-link convention, orphan deletion on re-ingest.

The Wiki pattern moves complexity from query time (RAG) to ingest time (Wiki). That complexity does not disappear; it concentrates where it can be audited, verified, and lint-checked. Hybrid inherits all of it and layers a dense index on top.

## See also

- [ADR-0001 — Strict grounded answers](adr/0001-strict-grounded-answers.md)
- [ADR-0002 — Two parallel retrieval apps](adr/0002-two-parallel-retrieval-apps.md)
- [ADR-0003 — W2 layered Wiki target (claude-obsidian)](adr/0003-w2-layered-wiki-target-claude-obsidian.md)
- [ADR-0004 — Post-LLM Grounding Check](adr/0004-post-llm-grounding-check.md)
- [ADR-0005 — Borrow components, keep opinions](adr/0005-framework-integration-borrow-components-keep-opinions.md)
- [ADR-0006 — W1 after Phase 3](adr/0006-w1-after-phase-3.md)
- [ADR-0018 — Hybrid retrieval, a third stack (RRF over Wiki)](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md)
- [Roadmap](roadmap.md)
