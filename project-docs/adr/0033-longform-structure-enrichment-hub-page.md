# Longform Sources: semantic Structure Enrichment materialized into docs/, hub-page ingest route, and an end to silent preamble loss

A 63-page scanned book (《蘇格拉底的申辯》, PRD grill 2026-07-06) exposed a degenerate path through the pipeline: Transcribe (ADR-0032) faithfully produced 28.6K chars of Markdown containing exactly **one** heading — a page running-header the vision model rendered as `# 柏拉图对话集 43` mid-file. `parse_markdown` then silently dropped every content line before that heading (~60% of the text — no Section, no `parse_warning`), and the concept route synthesized the single surviving Section into one short paragraph page. The wiki answered nothing; RAG answered partially (its chunks come from the same Sections, so it lost the same 60%). Everything reported success.

Three decisions close this class of failure:

**1. The preamble is a Section.** Body content before the first heading of a heading-bearing Source becomes a Section (`{source-filename}#intro`, heading = file stem) plus a `parse_warning`, instead of being dropped. This is a parser contract fix independent of anything longform — any Source with intro text above its first heading is losing that text today. CONTEXT.md § Section is amended accordingly.

**2. Segmentation intelligence moves to where the structure is missing.** When a Source's mechanical structure is degenerate (*longform detection*: zero or one heading, dominant preamble, or any single Section over `KB_INGEST_MAX_SECTION_TOKENS`), a **Structure Enrichment** pass runs at the Import/Transcribe stage: an LLM reads the full text, proposes chapter boundaries and titles (each bounded by the per-section cap; oversized proposals re-split mechanically at paragraph boundaries), and the headings are **written into the derived Markdown in `docs/`**. The same pass strips *page furniture* — running headers/footers/page numbers that repeat per transcribed page (in the trigger case: 34 identical timestamp/URL/page-counter lines). The transcription prompt gains a matching "omit repeated page furniture" rule for future runs; ADR-0032's faithful-transcription contract is amended to mean faithful to **content**, not to layout residue.

   Materializing into `docs/` (rather than keeping segmentation as side metadata) is the load-bearing choice: `parse_markdown`, Citations, RAG chunk anchors, C3 re-ingest resolution, and the Console source viewer all consume literal headings and work unchanged. Immutability is not violated — it lives in `raw/` (the untouched original); `docs/` is a derived artifact, exactly the reasoning that lets Transcribe write it with a provenance envelope. Enrichment runs once at import time, so ingest hash-skip idempotency keys on the enriched body and never re-runs it.

**3. Longform ingests as hub + chapters, not N flat pages and not one paragraph.** A third ingest route (beside `entity` and `concept`): one entity-style **Hub Page** ("about this document": themes, structure, wikilinks to every chapter page) plus the normal per-Section concept synthesis + Grounding Check + linkify over the enriched chapters. This is claude-obsidian parity — the project's declared pattern source segments semantically and links concepts; our pipeline previously segmented only on literal `#` characters, so the LLM never saw documents whose structure lived in meaning rather than markup. Industry alignment: LlamaIndex DocumentSummaryIndex (summary as router, chunks as evidence), GraphRAG's global/local split (hub answers "what is this book about", chapters + RAG answer detail), RAPTOR's machine-discovered hierarchy. No mainstream system emits one flat page per PDF page.

Grill context: an interlocking budget finding (estimates 5–50x over real cost; lint charged $0.15/run while executing zero LLM calls) is a calibration task in the same PRD, not an architectural decision — no ADR.

## Considered Options

- **Do nothing / declare books out of scope.** Rejected: the silent preamble drop damages ordinary Sources too, and "upload a real PDF and ask about it" is the demo's core promise.
- **One overview page only (pure DocumentSummaryIndex).** Rejected as the end state (kept as the degraded fallback if enrichment fails): wiki/Hybrid could answer only global questions; the knowledge owner's expectation — set by claude-obsidian on 10–20-page handouts — is chapter-level concepts with links.
- **Per-PDF-page headings from the transcriber (`## p.N`).** Rejected: page boundaries are physical, not semantic; 63 near-meaningless concept pages would flood the curated layer, at 63 synthesis calls.
- **Virtual segmentation metadata (offsets, docs/ untouched).** Rejected: every downstream consumer of literal headings (citations, anchors, viewer, re-ingest) would need a parallel resolution path; permanent complexity to avoid a one-time derived-file write.
- **Full RAPTOR tree / GraphRAG communities.** Rejected for scale: million-token indexing runs against a ≤$15/mo posture. The hub + chapter layer is the two-level version; deeper hierarchy is a recorded upgrade path.
- **Enrichment at ingest time instead of import time.** Rejected: ingest is read-only on `docs/` today and its hash-skip would loop (enriching changes the hash it keys on); import is where derived `docs/` content is already produced under provenance.

## Consequences

- New vocabulary: **Longform Source**, **Structure Enrichment**, **Hub Page**; **preamble** folded into § Section (CONTEXT.md).
- Enriched transcripts carry LLM-proposed headings; provenance must say so (e.g. `structure: enriched` beside `origin: transcribed`) so lint/governance can treat machine-titled chapters as a class.
- Chapter concept pages enter the wiki Section corpus, so Hybrid's dense layer can finally answer book-content questions — closing the gap the grill's screenshots showed.
- Ingest responses stop being silent about structure: per-source section counts and dropped/enriched character counts are surfaced (observability decision from the same grill).
- A Source too large for one enrichment context window is a follow-up issue (windowed outlining); the 80-page transcribe cap bounds the common case.
- The one-page `柏拉图对话集-43` wiki page and the furniture-polluted transcript on prod get regenerated by re-running import+ingest after this ships — enrichment works on the existing transcript text, so **no re-transcription** (no new per-page spend) is needed.
