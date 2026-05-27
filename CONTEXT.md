# Knowledge Base Q&A Bot

The shared language of this project. The bot answers questions against a small Markdown knowledge base, with grounded citations back to the original Markdown.

This file is a glossary, not a spec. Implementation details belong in code or ADRs.

## Language

**Source**:
A markdown file in `docs/` that the bot indexes. Each Source contains one or more Sections (split by markdown heading).
_Avoid_: Document (overloaded — LangChain's `Document` type stays internal to the library, not used as a domain term), Article (too narrow — won't cover future inputs like podcast notes or book chapters), Doc (collides with the `docs/` folder name).

**Section**:
The retrieval unit. A heading-anchored slice of a Source. A heading becomes a Section when either (i) it has no child headings (a *leaf*), or (ii) it carries non-whitespace body content directly between itself and its first child heading (a *body-bearing intermediate*). The Section's content is the body it owns directly — never the recursive body of its children, which are their own Sections. Carries a `heading_path` breadcrumb listing its parent headings. Identified by `{source-filename}#{heading-slug}` (e.g. `refund_policy.md#refund-timeline`). A Source with no headings degrades to a single Section identified by the bare filename. Empty-body leaves are still Sections (heading-only); their `tokens` come from the heading text alone.
_Avoid_: Chunk (implies fixed-size splitting, which we deliberately don't do), Paragraph (too granular — would fragment a single topic), Leaf section (the rule is broader than leaves alone).

**Section Index**:
The persisted inverted index over all Sections, saved to `.kb/index.json`. Stores both the BM25 metadata (doc frequencies, average length) and a full snapshot of each Section's content, so the server can answer queries after restart without re-reading the source. Under W1 ([[adr-0006]]) the corpus is `wiki/entities/*.md` + `wiki/concepts/*.md` (whitelisted subdirectories — meta files like `wiki/index.md`, `wiki/log.md`, `wiki/hot.md`, `wiki/README.md` are excluded as a correctness invariant). Goes stale when those subdirectories change and `/index` has not been re-run.
_Avoid_: Index (the bare word is reserved for Wiki Index — see below), BM25 Index (couples the name to one algorithm; we may later move to a hybrid retriever).

**Citation**:
A reference of the form `{slug}#{heading-slug}` (e.g. `refund-policy#cancellation-window`) returned with every grounded answer and printed in the `[Source: ...]` header of each context Section in the LLM prompt. Under W1 ([[adr-0006]]) the leading `{slug}` is the wiki page's stable slug (no type subdirectory, no `.md` extension) and is globally unique across `wiki/entities/` and `wiki/concepts/`; under the prior prototype it was `{source-filename}`. The heading slug is the lowercased, hyphen-collapsed heading text and matches the GitHub/Obsidian anchor convention so the reference is clickable in both. Within a single wiki page, if two leaf headings produce the same slug, the second collides and gets a `-2` suffix (`-3`, `-4`, …); a Section is never silently overwritten. The breadcrumb context (`heading_path`) is conveyed separately in the prompt's `Heading:` line, not embedded in the Citation.
_Avoid_: Source (Citation is a reference to a Section, not the Source as a whole), Reference (ambiguous in the LLM/RAG literature).

**Grounded Answer**:
An answer composed strictly from the cited Sections present in the LLM prompt's CONTEXT. Synthesis across multiple cited Sections is permitted; inference beyond what is written and any use of outside world knowledge is not. Every factual claim must carry at least one Citation. See ADR-0001.
_Avoid_: Sourced answer (vague), Cited answer (citing without grounding is still possible and we explicitly reject it).

**Grounding Check**:
A post-LLM validation step performed after the main answer is drafted. A second structured LLM call (the verifier) extracts atomic factual claims from the draft and judges each against the cited Sections. If any single claim is unsupported, the entire draft is discarded and `Cannot Confirm` is returned (Block & Replace contract). The check operates at claim-level granularity: each claim is judged independently, and the verifier records `citing_section_ids` per supported claim. The verifier uses `gpt-4o-mini` by default, independently configurable via `OPENAI_VERIFIER_MODEL`. On verifier failure after bounded retry, the system fails closed. Implemented in `grounding.py`, which consumes a `CitableContent` Protocol (not the markdown_kb-specific `Section` type) so `vector_rag/` can adopt it without changes. See ADR-0004.
_Avoid_: Verification (too generic), Fact-check (overloaded with journalism/social-media usage).

**Import**:
The mechanical operation of converting a raw source file (`.html` or `.txt`) from `raw/` into a normalized Markdown file in `docs/` with provenance frontmatter (`imported_from`, `original_format`, `imported_at`). No LLM calls; format conversion only. Implemented as `POST /import` in `importer.py`. Phase 7 ships `.html` (via `markdownify`) and `.txt` (passthrough). The full pipeline is: `raw/foo.html → POST /import → docs/foo.md → POST /ingest → wiki/<type>/<slug>.md → POST /index → POST /chat`. Import is completely disjoint from Ingest.
_Avoid_: Ingest (Ingest is LLM-driven synthesis from docs/ to wiki/; Import is mechanical format conversion from raw/ to docs/).

**Ingest**:
The operation of taking a new Source and integrating it into the Wiki: classify the Source type (`entity` or `concept`), generate synthesis page(s) via the LLM, run the Grounding Check verifier, and write the resulting Wiki Page(s) to `wiki/entities/` or `wiki/concepts/`. Implemented as `POST /ingest` in `ingest.py`. Uses a two-template MVP scope (`entity` + `concept`); the 5–7 sweet-spot target (`comparison`, `question`, `source` and others) remains deferred as the corpus grows.
_Avoid_: Import (more about format conversion than synthesis), Process (too generic).

**Source Template**:
A per-source-type Markdown skeleton used by `/ingest` to produce a Wiki Page from a Source. Templates are defined as structured-output schemas in `templates.py` and drive LLM generation via `with_structured_output`. Phase 3 ships two active templates: `entity` (one page per Source, collapsing all Sections) and `concept` (one page per Section, 1:N expansion). The long-term target is 5–7 types (empirical sweet spot from `claude-obsidian`; see `project-docs/inspiration.md`); `comparison`, `question`, and `source` remain deferred.
_Avoid_: Schema (overloaded — we already use "schema" for frontmatter field definitions), Note template (too generic).

**Wiki Page**:
A curated synthesis page in `wiki/entities/` or `wiki/concepts/` produced by `POST /ingest`. Carries the 8-field frontmatter schema: `id`, `type`, `created`, `updated`, `sources`, `status`, `open_questions`, `source_hashes`. Distinct from Source (the immutable input in `docs/`) and Section (the retrieval unit within a Source). Every Wiki Page carries a sentinel HTML comment (`<!-- Auto-generated by POST /ingest — manual edits will be overwritten. Re-ingest the source to regenerate. -->`) so editors know not to hand-edit.

The 8th field `source_hashes` maps each source filename to a pair of hash digests:
```yaml
source_hashes:
  <source_filename>:
    raw: <content_sha256 from docs frontmatter — written by /import; null if hand-authored>
    docs_body: <sha256(source_path.read_text('utf-8').encode()) — written by /ingest>
```
`raw` enables full raw→wiki chain detection by Phase 5 lint. `docs_body` enables hash-skip idempotency in `/ingest` (no LLM call when source unchanged). Empty `source_hashes` (default on Phase 6 legacy pages) means "drift state unknown" — `/ingest` does NOT skip on empty `source_hashes`. See ADR-0008.
_Avoid_: Wiki entry (vague), Synth page (informal).

**Red Link**:
A speculative `[[concept-slug]]` wikilink inside a Wiki Page that does not yet resolve to an existing page. Used by the LLM during ingest to mark concepts warranting their own page that the wiki does not yet cover. Phase 5 `/lint` will scan and rank these into a backlog. Red Links are intentional placeholders — they represent gaps the knowledge owner should fill.
_Avoid_: Broken link (negative connotation; Red Links are intentional), Wikilink (overloaded — wikilinks can be resolved or unresolved).

**Wiki Index**:
The human- and LLM-readable catalog of all Sections in the knowledge base, generated at `wiki/index.md` as a deterministic mechanical projection of `.kb/index.json` after every `POST /index` call. One H2 per Source, a bullet list of its Sections beneath with clickable `../docs/<filename>#<slug>` links. A navigation surface, not a search data structure — distinct from the Section Index, which is the BM25 inverted index. Gitignored (regenerable runtime artifact). Implemented in `wiki_index.py`; see ADR-0003.
_Avoid_: Catalog (vague), TOC (too narrow — implies linear ordering, but Wiki Index groups by Source).

**Wiki Log**:
A chronological, append-only event log at `wiki/log.md`. Each entry follows the format `## [<ISO-8601 UTC>] <kind> | <summary>` so it remains parseable with `grep`. Records every `POST /index`, `POST /chat`, `POST /ingest`, and `POST /import` event, fallback path, OpenAI error, parse warning, and verifier outcome. Per-kind summary fields are documented in each phase's PRD (current kinds: `index_built`, `chat`, `chat_fallback`, `chat_error`, `parse_warning`, `ingest_batch_started`, `ingest_source`, `ingest_grounding_failed`, `ingest_batch_completed`, `ingest_error`, `ingest_skipped`, `import_batch_started`, `import_source`, `import_skipped`, `import_error`, `import_batch_completed`). Slice 7-1 emits `import_batch_started`, `import_source`, and `import_batch_completed`; `import_skipped` lands in slice 7-3 and `import_error` in slice 7-2. `ingest_skipped` is emitted by Phase 3 amendment (#93) when `/ingest` hash-skips a source. Implemented via `log_event` in `logger.py`. Gitignored (runtime trace; not part of curated state).
_Avoid_: Changelog (project root may have its own CHANGELOG.md for releases), Audit log (overloaded with compliance language).

**Cannot Confirm**:
The literal phrase `"I cannot confirm from the knowledge base."` returned as the answer whenever the bot cannot safely back the answer from cited Sections. Treated as a successful, expected response — not a failure mode. Three situations produce this response:

- **(a) Pre-LLM threshold gate**: the Section Index yields no Sections, or only Sections below the score threshold. The LLM is never called; the fallback fires before any model invocation, so the model is never tempted to confabulate around weak context. Corresponds to `grounding.reason` values `retrieval_empty`, `below_threshold`, or `index_missing`.
- **(b) Post-LLM grounding failure**: the Grounding Check verifier finds at least one claim in the draft that is not supported by the cited Sections. The draft is discarded and `Cannot Confirm` is returned. Corresponds to `grounding.reason = "claim_unsupported"`.
- **(c) Post-LLM verifier unavailable**: the Grounding Check verifier fails after bounded retry (transient errors, hard errors). The system fails closed: `Cannot Confirm` is returned rather than releasing an unverified draft. Corresponds to `grounding.reason = "verifier_unavailable"`.

All three situations produce the identical surface response (`"I cannot confirm from the knowledge base."`); the `grounding` field on `ChatResponse` carries the structured reason. The score threshold is configured via the `KB_SCORE_THRESHOLD` env var (default `0.5` for the current sample corpus; recalibrate as the corpus grows).
_Avoid_: "I don't know" (sounds like a model limitation rather than a KB boundary), "Out of scope" (overloaded with product-feature language).

## Reserved (not yet implemented)

> The future wiki layer (curated synthesis above immutable Sources) is modelled on the patterns in `AgriciDaniel/claude-obsidian` (5.4K⭐ — see ADR-0003), which is the most-starred reference implementation of Karpathy's LLM Wiki gist. We borrow the patterns; the project's positioning is enterprise KB management (FAQ / policy / customer-support), where the curated layer enables governance (ownership, review cadence, audit) on top of the raw Source layer. Vocabulary reserved below mirrors the reference repo's structure so vocabulary aligns from day one. Operational patterns we want to remember but that are not vocabulary live in [`project-docs/inspiration.md#deferred-patterns`](project-docs/inspiration.md) — re-read that section before starting any phase beyond the current prototype.

**Wiki** _(future)_:
The LLM-maintained markdown layer that sits between Sources and queries. Planned as a `wiki/` directory of synthesis pages, organized by entity type (entities, concepts, comparisons, …). Distinct from Sources: Sources are immutable raw input; the Wiki is the LLM's compiled view, where cross-references and synthesis accumulate. **The Wiki is the sole query-time retrieval surface** (Phase 4 W1 model, ADR-0006); `docs/` is read only by `/ingest`, never by `/chat`. When `/chat` finds no adequately scoring Wiki Section, it returns Cannot Confirm with reason `wiki_gap`, which Phase 5 `/lint` aggregates into a coverage backlog (see [[lint-pass]]).
_Avoid_: Notes (too generic), Knowledge base (the project as a whole is a knowledge base; this term refers to the LLM-maintained layer specifically).

**Hot Cache** _(future)_:
A small (~500 words) working-memory file at `wiki/hot.md` containing the most recent context — the answer to "where were we?" between sessions. First file an agent reads on session start, before the Wiki Index. Multiple commenters on Karpathy's gist proposed this and the `claude-obsidian` repo implements it; the access pattern there is hot → index → domain-index → individual pages (a 4-level depth budget).
_Avoid_: Working memory (overloaded with LLM-architecture terms), Session cache (overloaded with HTTP).

**Lint Pass** _(future)_:
A periodic health check over the Wiki: contradiction detection, stale claims, orphan pages, missing cross-references, gaps suggested by repeated cannot-confirm queries. Karpathy's third core operation, alongside Ingest and Query. Operates on the **horizontal axis** — page-vs-page consistency — which is structurally orthogonal to [[grounding-check]]'s vertical axis (page-vs-Source). Neither subsumes the other: a Wiki Page can be individually grounded yet still contradict another grounded page (e.g., two summaries that each cherry-pick different parts of the same nuanced Source). The horizontal axis is therefore Lint Pass's exclusive responsibility — Grounding Check is structurally page-isolated and cannot see it.
_Avoid_: Sanity check (vague), Audit (compliance overtone).

**Query Rewriting** _(future)_:
Reformulating a follow-up user question into a self-contained query that carries the necessary conversational context, so retrieval can be performed without the multi-turn history. Required for the PROMPT.md "Conversation Memory" stretch goal.
_Avoid_: Query expansion (a different technique that adds synonyms to a single-turn query), Question reformulation (verbose).

**Conversation Store** _(future)_:
A session-scoped store of recent turns, keyed by session id, used to feed Query Rewriting in multi-turn flows. Planned with a sliding window (~10 turns) and TTL eviction. Does not exist in the prototype.
_Avoid_: Session store (overloaded with auth/cookie usage), Chat history (vague).

**Filed Answer** _(future)_:
A Wiki Page in `wiki/qa/*.md` produced by `POST /chat` when a Grounded Answer passes the filing gate. Closes the Two-output rule on the query side (Sources close on the ingest side via Phase 3). Carries the standard 7-field frontmatter plus a literal `question` field and inherits the `status: draft|live` lifecycle from the schema (entity/concept pages default to `live` at ingest; Filed Answers default to `draft` and only pages with `status: live` enter the BM25 corpus). The draft→live transition is the **promotion gate** — an explicit decision (human curator or `/lint`-driven rule) that separates *filing* (capture every Grounded Answer) from *consumption* (only review-approved content is retrievable). Phase 6 ships the filing path; promotion candidates are surfaced by Phase 5 `/lint` via the read-only C8 check, while the actual draft→live mutation is owned by Phase 6's `POST /qa/{slug}/promote` endpoint (Slice 6-4), invoked by the curator after reviewing `lint-report.md`.
_Avoid_: Q&A page (collides casually with "Q&A bot" — the product), Saved answer (implies user-initiated save, but filing is automatic), FAQ page (FAQ is an input Source type, not the output).

## Flagged ambiguities

**Slug generation for non-ASCII headings is undefined.**
The current `slugify()` strips everything outside `[a-z0-9]`, so a heading like `## 退款政策` collapses to the literal fallback string `section`. Multiple non-ASCII headings inside one Source will therefore all collide on `section` and pile up `section-2`, `section-3`, … — usable but unreadable. Acceptable for the English-only sample `docs/`; must be revisited before ingesting personal notes that contain CJK or other non-Latin headings (likely at the same time the wiki layer is added).

**Section ID is not stable across renames.**
A Section's identifier is derived purely from `{source-filename}#{heading-slug}`. Renaming a Source file or editing a heading text changes the identifier, which silently invalidates every Citation a caller already received. Accepted unflagged for the prototype because the PROMPT.md verification freezes both filenames and headings; a content-derived stable identifier (UUID sidecar or content hash) becomes worth implementing at the same time the wiki layer is added, since that is when accumulated Citations start being expensive to invalidate.
