# Knowledge Base Q&A Bot

The shared language of this project. The bot answers questions against a small Markdown knowledge base, with grounded citations back to the original Markdown.

This file is a glossary, not a spec. Implementation details belong in code or ADRs.

## Language

**Source**:
A markdown file in `docs/` that the bot indexes. Each Source contains one or more Sections (split by markdown heading).
_Avoid_: Document (overloaded — LangChain's `Document` type stays internal to the library, not used as a domain term), Article (too narrow — won't cover future inputs like podcast notes or book chapters), Doc (collides with the `docs/` folder name).

**Section**:
The retrieval unit. A leaf-heading slice of a Source: heading text plus body content up to the next heading at the same or shallower depth. Carries a `heading_path` breadcrumb listing its parent headings. Identified by `{source-filename}#{heading-slug}` (e.g. `refund_policy.md#refund-timeline`). A Source with no headings degrades to a single Section identified by the bare filename.
_Avoid_: Chunk (implies fixed-size splitting, which we deliberately don't do), Paragraph (too granular — would fragment a single topic).

**Section Index**:
The persisted inverted index over all Sections, saved to `.kb/index.json`. Stores both the BM25 metadata (doc frequencies, average length) and a full snapshot of each Section's content, so the server can answer queries after restart without re-reading `docs/`. Goes stale when `docs/` changes and `/index` has not been re-run.
_Avoid_: Index (the bare word is reserved for the future wiki-layer Wiki Index — see below), BM25 Index (couples the name to one algorithm; we may later move to a hybrid retriever).

**Citation**:
A reference of the form `{source-filename}#{heading-slug}` (e.g. `refund_policy.md#refund-timeline`) returned with every grounded answer and printed in the `[Source: ...]` header of each context Section in the LLM prompt. The slug is the lowercased, hyphen-collapsed heading text and matches the GitHub/Obsidian anchor convention so the reference is clickable in both. Within a single Source, if two leaf headings produce the same slug, the second collides and gets a `-2` suffix (`-3`, `-4`, …); a Section is never silently overwritten. The breadcrumb context (`heading_path`) is conveyed separately in the prompt's `Heading:` line, not embedded in the Citation.
_Avoid_: Source (Citation is a reference to a Section, not the Source as a whole), Reference (ambiguous in the LLM/RAG literature).

**Grounded Answer**:
An answer composed strictly from the cited Sections present in the LLM prompt's CONTEXT. Synthesis across multiple cited Sections is permitted; inference beyond what is written and any use of outside world knowledge is not. Every factual claim must carry at least one Citation. See ADR-0001.
_Avoid_: Sourced answer (vague), Cited answer (citing without grounding is still possible and we explicitly reject it).

**Cannot Confirm**:
The literal phrase `"I cannot confirm from the knowledge base."` returned as the answer whenever the Section Index yields no Sections, only Sections below the score threshold, or Sections that partially relate but do not answer the question. Treated as a successful, expected response — not a failure mode. The score threshold is configured via the `KB_SCORE_THRESHOLD` env var (default `0.5` for the current sample corpus; recalibrate as the corpus grows). The fallback is gated *before* the LLM call whenever retrieval is empty or below threshold, so the LLM is never tempted to confabulate around weak context.
_Avoid_: "I don't know" (sounds like a model limitation rather than a KB boundary), "Out of scope" (overloaded with product-feature language).

## Reserved (not yet implemented)

**Wiki Index** _(future)_:
The human- and LLM-readable catalog of wiki pages, planned to live at `wiki/index.md` once the wiki layer exists. Distinct from the Section Index: a navigation surface, not a search data structure. Reserved here so the term is available without renaming the Section Index later.

## Flagged ambiguities

**Slug generation for non-ASCII headings is undefined.**
The current `slugify()` strips everything outside `[a-z0-9]`, so a heading like `## 退款政策` collapses to the literal fallback string `section`. Multiple non-ASCII headings inside one Source will therefore all collide on `section` and pile up `section-2`, `section-3`, … — usable but unreadable. Acceptable for the English-only sample `docs/`; must be revisited before ingesting personal notes that contain CJK or other non-Latin headings (likely at the same time the wiki layer is added).
