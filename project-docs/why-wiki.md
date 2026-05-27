# Why Wiki (and where RAG still wins)

This project builds a curated LLM-maintained Wiki layer over an immutable Source corpus, rather than a vector RAG pipeline. This page is the short version of why. For the actual design decisions, follow the ADR links.

## The core tradeoff

Both approaches let you answer questions over a document collection. They differ in **when synthesis happens**.

- **RAG**: synthesis at query time. Embed the query, retrieve top-k chunks, ask the LLM to assemble an answer. Every query re-derives the answer from raw text.
- **Wiki**: synthesis at ingest time. The LLM reads each Source once, writes a structured page into `wiki/`, and queries read the pre-synthesised page. The Source stays immutable; the Wiki is the compounding artifact.

For this project's corpus (under 1000 pages, curated FAQ / policy / KB content), Wiki wins. RAG wins at the other end of the scale.

## Comparison

| Dimension | LLM Wiki | Vector RAG |
|---|---|---|
| **Storage shape** | Pre-compiled `wiki/entities/`, `wiki/concepts/` markdown with cross-references and a generated `wiki/index.md` | Raw chunks in a vector database, no inter-chunk structure |
| **Finding answers** | Read `wiki/index.md` → follow links → answer from Wiki page + cited Source | Embed query → similarity search → assemble from top-k chunks |
| **Provenance** | Section-level: every Wiki page declares its `sources:` in frontmatter; every answer cites `filename#heading` ([ADR-0001](adr/0001-strict-grounded-answers.md)) | Chunk-level + similarity score; section boundaries usually lost |
| **Verifier integration** | Grounding Check runs on every produced Wiki page at ingest; failed pages get `status: failed_grounding` and are filtered from query-time reads ([ADR-0004](adr/0004-post-llm-grounding-check.md)) | Verifier runs only at query time (over the assembled answer), not over the retrieval substrate |
| **Contradiction detection** | Cross-page scan during `/lint` (Phase 5) — operates on structured pages with frontmatter and `[[wikilinks]]` | Manual; chunks have no relationship metadata |
| **Refresh semantics** | Full rewrite of a Source's derived pages, with orphan deletion on structure change ([Phase 3 Q7](roadmap.md)) | Chunk-level dedup is painful; partial re-embeds drift over time |
| **Query cost** | Low — synthesis is already done; queries pay for retrieval + final answer only | Higher — every query pays for embedding + assembly |
| **Setup floor** | Index file is sufficient for navigation under ~1000 pages | Embedding model + vector DB + chunking pipeline required from day 1 |
| **Scale ceiling** | ~hundreds to low thousands of pages (index file navigation cost) | Millions of documents |

## Verdict

- **Under ~1000 pages** → Wiki. Index navigation is cheap, the pre-compiled synthesis means every query benefits from everything previously read, and provenance stays structured.
- **Over ~100K pages** → RAG. The index becomes too large to read, and embedding-based retrieval is more efficient than full-index scanning.
- **In between** → hybrid: run the Wiki pattern for active curation, then export to a vector store if the collection outgrows the index threshold.

For this project, the corpus is curator-bounded (FAQ / policy / customer-support), so Wiki is the load-bearing choice. The `vector_rag/` scaffold is preserved for an empirical comparison once the Wiki layer is mature — see [ADR-0002](adr/0002-two-parallel-retrieval-apps.md) and Phase 7 (Paraphrase Comparison) in [roadmap.md](roadmap.md).

## What the table understates

The "just markdown files" framing for the Wiki column is misleading without context. The Wiki path is cheap to *store* but the ingest pipeline is non-trivial:

- 7-field frontmatter schema (`id`, `type`, `created`, `updated`, `sources`, `status`, `open_questions`) — see [Phase 3 grill notes](roadmap.md#phase-3-q1-q3-resolved-2026-05-26).
- Grounding Check verifier on every produced page ([ADR-0004](adr/0004-post-llm-grounding-check.md)).
- Wiki Log with 5 ingest event kinds for audit and `/lint` consumption.
- Collision rule (`-2`, `-3` suffix), red-link convention, orphan deletion on re-ingest.

The Wiki pattern moves complexity from query time (RAG) to ingest time (Wiki). That complexity doesn't disappear — it concentrates where it can be audited, verified, and lint-checked.

## See also

- [ADR-0001 — Strict grounded answers](adr/0001-strict-grounded-answers.md)
- [ADR-0002 — Two parallel retrieval apps](adr/0002-two-parallel-retrieval-apps.md)
- [ADR-0003 — W2 layered Wiki target (claude-obsidian)](adr/0003-w2-layered-wiki-target-claude-obsidian.md)
- [ADR-0004 — Post-LLM Grounding Check](adr/0004-post-llm-grounding-check.md)
- [ADR-0005 — Borrow components, keep opinions](adr/0005-framework-integration-borrow-components-keep-opinions.md)
- [ADR-0006 — W1 after Phase 3](adr/0006-w1-after-phase-3.md)
- [Roadmap](roadmap.md)
