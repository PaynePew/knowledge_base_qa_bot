# Knowledge Base Q&A Bot

The shared language of this project. The bot answers questions against a small Markdown knowledge base, with grounded citations back to the original Markdown.

This file is a glossary, not a spec. Implementation details belong in code or ADRs.

## Language

**Source**:
A markdown file in `docs/` that the bot indexes. Each Source contains one or more Sections (split by markdown heading).
_Avoid_: Document (overloaded — LangChain's `Document` type stays internal to the library, not used as a domain term), Article (too narrow — won't cover future inputs like podcast notes or book chapters), Doc (collides with the `docs/` folder name).

**Section**:
The retrieval unit. A heading-anchored slice of a Source. A heading becomes a Section when either (i) it has no child headings (a *leaf*), or (ii) it carries non-whitespace body content directly between itself and its first child heading (a *body-bearing intermediate*). The Section's content is the body it owns directly — never the recursive body of its children, which are their own Sections. Carries a `heading_path` breadcrumb listing its parent headings. Identified by `{source-filename}#{heading-slug}` (e.g. `refund_policy.md#refund-timeline`). A Source with no headings degrades to a single Section identified by the bare filename. Non-whitespace body content that appears **before the first heading** of a heading-bearing Source (the *preamble*) forms its own Section (id `{source-filename}#intro`, heading = the file stem) — it is never silently dropped (see [[adr-0033]]; before that decision it was, which cost a transcribed book 60% of its text). Empty-body leaves are still Sections (heading-only); their `tokens` come from the heading text alone.
_Avoid_: Chunk (implies fixed-size splitting, which markdown_kb deliberately does not do — but **vector_rag's Chunk is a separate, blessed term**; see § Phase 8 vocabulary below), Paragraph (too granular — would fragment a single topic), Leaf section (the rule is broader than leaves alone).

**Section Index**:
The persisted inverted index over all Sections, saved to `.kb/index.json`. Stores both the BM25 metadata (doc frequencies, average length) and a full snapshot of each Section's content, so the server can answer queries after restart without re-reading the source. Under W1 ([[adr-0006]]) the corpus is `wiki/entities/*.md` + `wiki/concepts/*.md` (whitelisted subdirectories — meta files like `wiki/index.md`, `wiki/log.md`, `wiki/hot.md`, `wiki/README.md` are excluded as a correctness invariant). Goes stale when those subdirectories change and `/index` has not been re-run. Within the whitelisted subdirectories, a page with `status: failed_grounding` is **quarantined** — excluded from the corpus until a passing re-ingest or a reconcile clears the failure ([[adr-0029]]); a page with no `status` field is treated as live.
_Avoid_: Index (the bare word is reserved for Wiki Index — see below), BM25 Index (couples the name to one algorithm; we may later move to a hybrid retriever).

**Citation**:
A reference of the form `{slug}#{heading-slug}` (e.g. `refund-policy#cancellation-window`) returned with every grounded answer and printed in the `[Source: ...]` header of each context Section in the LLM prompt. Under W1 ([[adr-0006]]) the leading `{slug}` is the wiki page's stable slug (no type subdirectory, no `.md` extension) and is globally unique across `wiki/entities/` and `wiki/concepts/`; under the prior prototype it was `{source-filename}`. The heading slug is the hyphen-collapsed heading text with ASCII lowercased and non-ASCII letters (including CJK) preserved verbatim, matching the GitHub/Obsidian Unicode anchor convention so the reference is clickable in both (e.g. `## 退款政策` → `退款政策`). Within a single wiki page, if two leaf headings produce the same slug, the second collides and gets a `-2` suffix (`-3`, `-4`, …); a Section is never silently overwritten. The breadcrumb context (`heading_path`) is conveyed separately in the prompt's `Heading:` line, not embedded in the Citation.
_Avoid_: Source (Citation is a reference to a Section, not the Source as a whole), Reference (ambiguous in the LLM/RAG literature).

**Grounded Answer**:
An answer composed strictly from the cited Sections present in the LLM prompt's CONTEXT. Synthesis across multiple cited Sections is permitted; inference beyond what is written and any use of outside world knowledge is not. Every factual claim must carry at least one Citation. See ADR-0001.
_Avoid_: Sourced answer (vague), Cited answer (citing without grounding is still possible and we explicitly reject it).

**Grounding Check**:
A post-LLM validation step performed after the main answer is drafted. A second structured LLM call (the verifier) extracts atomic factual claims from the draft and judges each against the cited Sections. If any single claim is unsupported, the entire draft is discarded and `Cannot Confirm` is returned (Block & Replace contract). The check operates at claim-level granularity: each claim is judged independently, and the verifier records `citing_section_ids` per supported claim. The verifier uses `gpt-4o-mini` by default, independently configurable via `OPENAI_VERIFIER_MODEL`. On verifier failure after bounded retry, the system fails closed. Implemented in `grounding.py`, which consumes a `CitableContent` Protocol (not the markdown_kb-specific `Section` type) so `vector_rag/` can adopt it without changes. See ADR-0004.
_Avoid_: Verification (too generic), Fact-check (overloaded with journalism/social-media usage).

**Import**:
The mechanical operation of converting a raw source file (`.html`, `.txt`, or `.pdf`) from `raw/` into a normalized Markdown file in `docs/` with provenance frontmatter (`imported_from`, `original_format`, `imported_at`). No LLM calls; format conversion only. Implemented as `POST /import` in `importer.py`. Phase 7 ships `.html` (via `markdownify`) and `.txt` (passthrough); the Phase 7 amendment (PRD #414) adds `.pdf` — text-layer extraction via MarkItDown, digital-native PDFs only: a scanned PDF has no text layer and is rejected with a typed failure, never OCR'd (see [[adr-0031]]). The full pipeline is: `raw/foo.html → POST /import → docs/foo.md → POST /ingest → wiki/<type>/<slug>.md → POST /index → POST /chat`. Import is completely disjoint from Ingest.
_Avoid_: Ingest (Ingest is LLM-driven synthesis from docs/ to wiki/; Import is mechanical format conversion from raw/ to docs/).

**Ingest**:
The operation of taking a new Source and integrating it into the Wiki: classify the Source type (`entity` or `concept`), generate synthesis page(s) via the LLM, run the Grounding Check verifier, and write the resulting Wiki Page(s) to `wiki/entities/` or `wiki/concepts/`. A [[longform-source]] takes a third route: one [[hub-page]] plus per-Section concept pages (see [[adr-0033]]). Implemented as `POST /ingest` in `ingest.py`. Uses a two-template MVP scope (`entity` + `concept`); the 5–7 sweet-spot target (`comparison`, `question`, `source` and others) remains deferred as the corpus grows.
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
A speculative `[[concept-slug]]` wikilink inside a Wiki Page that does not yet resolve to an existing page. Used by the LLM during ingest to mark concepts warranting their own page that the wiki does not yet cover. Phase 5 `/lint` will scan and rank these into a backlog. Red Links are intentional placeholders — they represent gaps the knowledge owner should fill. A wikilink resolves iff its slug matches an existing page's slug **or one of its [[alias]]es**; a Red Link is one that resolves by neither (see [[adr-0030]]).
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

**Filed Answer**:
A Wiki Page in `wiki/qa/*.md` produced by `POST /chat` when a Grounded Answer passes the filing gate (Phase 6 — implemented). Closes the Two-output rule on the query side (Sources close on the ingest side via Phase 3). Carries the standard frontmatter plus a literal `question` field and the `status: draft|live` lifecycle (entity/concept pages default to `live` at ingest; Filed Answers default to `draft`; only `status: live` pages enter the retrieval corpus). The draft→live transition is the **promotion gate** — an explicit curator decision (`POST /qa/{slug}/promote`, candidates surfaced by `/lint`'s read-only C8 check) separating *filing* (capture every Grounded Answer) from *consumption* (only review-approved content is retrievable); promoting **auto-reindexes** so the page is immediately retrievable. Filing carries a **novelty gate**: an answer derived entirely from existing Filed Answers is not re-filed. The loop as a whole is human-governed *validated write-back* — grounding + citation + novelty acceptance, then human promotion — a curated layer, deliberately **not** a semantic cache. See [[adr-0020]].
_Avoid_: Q&A page (collides casually with "Q&A bot" — the product), Saved answer (implies user-initiated save, but filing is automatic), FAQ page (FAQ is an input Source type, not the output).

### Phase 11 — Conversation Memory vocabulary

> These terms name code in the Phase 11 Conversation Memory feature (PRD #158, slices #159–#163). Promoted from `## Reserved (not yet implemented)` to active Language (2026-05-29, post-grill) per CODING_STANDARD §3.2 — they are now usable as class/variable names. See ADR-0013 for the cross-stack design decisions.

**Query Rewriting**:
Reformulating a follow-up user question into a self-contained query that carries the necessary conversational context, so retrieval can be performed without the multi-turn history. Implemented as the **Gateway's first LLM-facing module** (`gateway/app/query_rewriting.py` — ADR-0005, ADR-0013). Triggers on turn 2+ only (turn 1 is a passthrough — no LLM call, no latency). Uses `OPENAI_REWRITE_MODEL` with a two-layer fallback (`OPENAI_REWRITE_MODEL` → `OPENAI_MODEL` → `gpt-4o-mini`). The four-rule rewriter prompt governs: (1) resolve references / fill ellipsis only; (2) an already-self-contained query is returned unchanged; (3) output only the rewritten string; (4) on ambiguous reference, conservatively keep the original. The raw follow-up is discarded after rewriting; `gateway/log.md` records `chat_rewrite` kind for debugging.
_Avoid_: Query expansion (a different technique that adds synonyms to a single-turn query), Question reformulation (verbose).

**Conversation Store**:
A session-scoped in-memory store of recent turns, keyed by `session_id` (a UUID minted by the Gateway), used to feed Query Rewriting in multi-turn flows. Implemented in `gateway/app/conversation_store.py` (Phase 11 — ADR-0013). **Keyed by `session_id` only** — `stack` is per-turn metadata, not a partition key, so a Wiki↔RAG toggle within one session shares history. Holds a sliding window of 10 turns (oldest evicted on the 11th append); idle sessions are TTL-evicted after 30 min idle. A turn is written only on a normal `done` event (grounding-pass, grounding-fail, or Cannot Confirm); `error`-before-`done` writes nothing. Exposes `dump(session_id)` as a full session-window export (originally framed as a "Phase 10 Hot Cache seam"; that framing is retired — Hot Cache is deferred into the Phase 12 MCP grill and its input is the MCP host's session, not this store). `evict_expired()` is called at the top of `chat_stream` (before the history lookup) so sweeps are lazy and request-driven.
_Avoid_: Session store (overloaded with auth/cookie usage), Chat history (vague).

### Phase 8 — Paraphrase Comparison vocabulary

> These terms name code in the Phase 8 retrieval comparison (PRD #100, slices #101-#105). Resolved during the Phase 8 grill (2026-05-28) and placed in **active Language** — not Reserved — so they are usable as class/variable names (per CODING_STANDARD §3.2, Reserved terms are off-limits as names). Most are scoped to the `eval/paraphrase_comparison/` package and `vector_rag/`.

**Retrieval Stack**:
A whole retrieval architecture under comparison in Phase 8. **Stack A** = Karpathy Wiki (`/ingest` synthesis → BM25 over `wiki/`). **Stack B** = Vector RAG (Chunk + embed → FAISS over the raw corpus). Both consume the same raw corpus; each runs its own indexing. The comparison isolates the architectural difference, not a single algorithm.
_Avoid_: Strategy (PROMPT.md uses "retrieval strategy" loosely for the same idea, but "Stack" names the whole pipeline), Backend, Pipeline.

**Chunk**:
The retrieval unit of `vector_rag/` (Stack B): a character-bounded slice within a Section, produced by recursive character splitting of a Section's body after heading-aware sectioning. **Distinct from Section** (markdown_kb's heading-anchored unit). A Chunk carries its parent Section's id as its `source` so Citations stay Section-granular. The CODING_STANDARD §10 "no Chunk class" rejection is scoped to markdown_kb (where Section is the unit); vector_rag's Chunk is the deliberate, blessed exception.
_Avoid_: Document (LangChain's internal type — stays inside vector_rag's LLM-facing modules, never leaks per §2.4), Fragment, Passage.

**Paraphrase**:
A query variant of a canonical question, generated by LLM or hand-written, used to probe retrieval robustness in the Phase 8 comparison. Each Paraphrase targets one Gold Section, carries Key Tokens, and belongs to one Paraphrase Type. Committed to `queries.yaml`.
_Avoid_: Rephrasing (verb-shaped), Query variant (vague).

**Paraphrase Type**:
The transformation category of a Paraphrase. Seven types in two groups — **Core** (LLM-generated): `synonym_swap`, `word_reorder`, `verbosity_expansion`, `specificity_narrowing`, `implicit_reference`; **Structural probes** (hand-written): `typo_fatfinger`, `industry_jargon`. Core and probes are reported separately (no naive cross-type aggregate).
_Avoid_: Category (overloaded), Variant kind.

**Gold Section**:
The docs Section a Paraphrase is expected to retrieve — the correct answer used to score a Retrieval Stack. Identified by the same `{source-filename}#{heading-slug}` form as any Section. Paraphrases are sourced only from concept-type Wiki Pages (1:1 with a docs Section) so the Gold Section is unambiguous.
_Avoid_: Ground truth (verbose), Target (vague), Answer (collides with Grounded Answer).

**Key Tokens**:
The distinctive tokens the Phase 8 hit metric uses to verify that a retrieved item's *content* (not merely its `source` id) actually answers the Paraphrase. **Dual-side**: extracted from both the docs Section (`key_tokens_docs`) and the corresponding Wiki Page (`key_tokens_wiki`), so either Stack's surface vocabulary can match. A hit requires `source` match AND non-empty overlap with the union of the two token sets.
_Avoid_: Keywords (overloaded with BM25 query terms), Tokens (the bare word is the BM25 tokeniser output).

**Spot-check**:
The optional Phase 8 L2 validation step: a cross-family LLM judge (Claude) re-judges the deterministic hit metric's verdicts over an ambiguous subset, validating the metric's edge cases. Opt-in via `--judge`. Produces no headline numbers — the deterministic L1 metric does; the Spot-check only reports an agreement rate.
_Avoid_: Audit (compliance overtone), Review (overloaded), Validation (too generic).

### Phase 15 — Operator Console vocabulary

> These terms name the Phase 15 curator-facing management surface and its staging operation (PRD #168; [[adr-0011]] / [[adr-0012]]). Promoted from Reserved to **active Language** in Slice S1 (#169) so they are usable as module / function / variable names — per CODING_STANDARD §3.2, Reserved terms are off-limits as names, so `upload`, `upload_files`, … are now legal.

**Operator Console**:
The curator-facing management surface served by the [[gateway]] at `GET /console` — the write/maintain counterpart to the reader chat UI. Drives the existing Wiki lifecycle operations (Import → Ingest → Index → Lint → Filed Answer promotion) and exposes RAG index rebuild as a single parallel action; it orchestrates each Retrieval Stack's operations but does not merge them (ADR-0002). Wiki-centric by design — the RAG stack has no Ingest/Lint/Filing, so only its index build is operable. Also hosts a resource browser over `docs/` and `wiki/`. Scope is a single local operator (no auth, no multi-user); demo posture shapes scope, not engineering rigor.
_Avoid_: Admin panel (implies user/role administration — out of scope), Dashboard (implies metrics/monitoring, not lifecycle actions), Backend page (vague).

**Upload**:
The transport/staging operation behind the Operator Console drop zone: writes dropped browser file bytes onto the server via `POST /upload`. `.html`/`.txt` land in `raw/` (becoming Import candidates); `.md` lands directly in `docs/` (already canonical Markdown — skips Import). Distinct from Import — Upload only moves bytes onto the server, it never converts. As a system boundary, all untrusted-input validation (type allow-list, size limit, traversal-safe filename) lives in the Upload module. See [[adr-0011]].
_Avoid_: Import (Import converts raw→docs; Upload only stages bytes), Ingest (LLM synthesis docs→wiki).

### Interface parity vocabulary

> Under the single-operator posture ([[adr-0017]], which supersedes [[adr-0016]]'s read-only MCP surface), the Browser ([[operator-console]]), the CLI (`kb`), and the MCP server are three independent, feature-equivalent interfaces over the same on-disk corpus — each can drive the full Import → Ingest → Index → Lint lifecycle, not only the Browser. They share state only through the filesystem ("they meet at the filesystem"). What stays interface-specific is *byte-ingress* — Upload (Browser), a filesystem path (CLI/MCP), and Capture (MCP).

**Capture**:
The operation by which an MCP agent authors a Markdown Source directly from its session content and persists it to `docs/`, carrying conversation provenance (`origin: mcp-conversation`). Distinct from [[import]] (mechanical conversion of an existing `raw/` file) and [[upload]] (byte staging only): Capture has no source file and no format conversion — the agent is the author. MCP-only, because only the agent holds content that exists nowhere on disk; the Browser and CLI reach `docs/` through a file. The provenance frontmatter is mandatory — it is what keeps conversation-derived knowledge distinguishable from authoritative external Sources. See [[adr-0017]].
_Avoid_: Import (Import converts an existing raw file; Capture authors new content), Save (Hot Cache already uses "save" for working memory), Ingest (Ingest synthesises docs→wiki; Capture only creates the Source).

### Phase 13 — Hybrid Retrieval vocabulary

> Scoped in the 2026-05-28 and 2026-06-28 grills. Promoted from Reserved to **active Language** in Slice S2 (#312) — per CODING_STANDARD §3.2, Reserved terms are off-limits as names, so `hybrid_kb`, `Hybrid Retrieval`, and the retrieval-core identifiers (RRF fusion, the OR-gate) are now legal in code. Hybrid is the third Retrieval Stack alongside Wiki (Stack A) and Vector RAG (Stack B) — see [[retrieval-stack]].

**Hybrid Retrieval**:
A third [[retrieval-stack]] that retrieves over the **same `wiki/` Section corpus as the Wiki stack** (not `docs/`), combining two search methods — BM25 (keyword / sparse) and dense-vector (semantic) — and fusing their two ranked lists with **Reciprocal Rank Fusion (RRF)**. The "RAG" in the informal name "wiki + RAG hybrid" denotes the *dense-vector technique applied to wiki Sections*, **not** the Vector RAG stack's `docs/` [[chunk]] index. Because both arms return wiki [[section]]s, their ids align 1:1, so RRF is true same-corpus fusion and everything downstream ([[grounding-check]], [[citation]], expand-to-pages, [[cannot-confirm]]) is reused unchanged. RRF is a **recall-union** step (rescues relevant items one method ranked low), **not** a precision filter; a cross-encoder [[reranker]] for precision (the FM2 semantic-false-positive case) is a separate, default-off stage layered on top, never part of v1 fusion.
_Avoid_: "fusing the Wiki and RAG **stacks**" (it fuses two **methods over one corpus**, not two stacks over two corpora — the cross-corpus reading was explicitly rejected), Hybrid Search (acceptable informally, but "Retrieval" matches [[retrieval-stack]]), Reranking (the [[reranker]] is a distinct precision stage layered on top — Hybrid's own fusion is RRF-only, so never call the fusion itself "reranking").

**Reranker**:
A precision stage that re-scores the RRF-fused candidate pool with a **cross-encoder** — query and [[section]] scored *jointly* in one model pass, unlike the dense arm's separate-encoding similarity — and reorders the pool before the final top-k cut. It is the FM2 semantic-false-positive fix that RRF, a recall-union step, structurally cannot do. Distinct from and downstream of Hybrid Retrieval's fusion: it never feeds the [[cannot-confirm]] gate (which stays on each arm's native pre-fusion score) and exposes no score to a [[citation]] (the fused/reranked magnitude is not a calibrated relevance score). Default-off and gated on the paraphrase-comparison eval — see [[adr-0019]].
_Avoid_: Reranking (prefer the noun "Reranker" for the stage and the verb "rerank" for the act), Fusion (RRF fuses two ranked lists; the reranker reorders one), Re-scoring (too generic).

**Dense rebuild**:
The operator action that re-embeds the entire `wiki/` [[section]] corpus into the [[hybrid-retrieval]] stack's **dense arm**, refreshing it to match the current wiki. Distinct from the console's **RAG rebuild**, which re-embeds `docs/` [[chunk]]s into the separate Vector RAG stack — different corpus, different stack. The dense arm is refreshed **only** by this explicit action: unlike the BM25 arm (which [[promote]] rebuilds automatically), a dense rebuild re-embeds the whole corpus, so it is deliberately a **manual** operator action — RRF tolerates a lagging dense arm, which degrades recall/freshness, not correctness (see [[adr-0022]]). Scoped in the 2026-06-30 grill (#348).
_Avoid_: "RAG rebuild" for this action (that is the `docs/` Vector RAG re-embed — conflating them repeats the rejected wiki-vs-RAG cross-corpus reading); "reindex" unqualified (the BM25 reindex on promote and the dense rebuild are separate operations).

### Lint remediation vocabulary

> Scoped in the 2026-07-01 grill. Names the executable "fix" layer over the [[lint-pass]] report — the answer to "the lint told me what is wrong, now what?". The two terms are the shared vocabulary the Browser, CLI, and MCP surfaces all use. See [[adr-0023]].

**Lint Axis**:
The four-way classification of [[lint-pass]] findings by *what diverged*, and the shared grouping the Browser, CLI, and MCP all render findings under ([[adr-0017]] interface parity): **Freshness** (a Wiki Page vs its [[source]] — C6 stale, C3 failed-grounding, C11 orphan), **Coherence** (page vs page — C5 contradiction *(same-question incompatibility only — not topical similarity or consistent redundancy; the `duplicate`/loose-`tension` buckets were retired, see [[adr-0034]]; a [[source-rooted-contradiction]] exits via Routed fix-source, not Reconcile, see [[adr-0036]])*, C4 collision, C12 alias-collision; the horizontal axis that is lint's exclusive job, structurally invisible to [[grounding-check]]), **Coverage** (corpus vs demand — C1 coverage-gap, C2 red-link), and **Lifecycle** ([[filed-answer]] governance — C8 promotion, C10 invalid-schema, C9 stale-qa). The axis a finding sits on determines its [[remediation]] family.
_Avoid_: Category (overloaded), Check group (UI-flavoured — the axis is a domain classification, not a screen region), Dimension (collides with the Phase 8 Paraphrase-comparison vocabulary).

**Remediation**:
The executable operation that resolves a [[lint-pass]] finding — distinct from the advisory `suggested_action` string the finding already carries (that is prose; a Remediation *runs*). Governance splits on one question — *does the operation commit without a human gate?* A **Direct Remediation** has no gate: it re-runs deterministic machinery ([[ingest]] re-sync for Freshness), flips a reversible [[filed-answer]] lifecycle bit (promote/discard for Lifecycle), or records a reversible curator-supplied mapping (assigning an [[alias]] to close a C2 red link, [[adr-0030]]); it yields no new synthesis and is therefore one-click and review-free, and it is the only class that may batch (assign-alias deliberately gets no batch — each mapping is an independent judgment). A **Gated Remediation** commits nothing until a human approves, in two flavours by *what the human reviews*: an **Authored Remediation** has the LLM draft new curated synthesis (reconcile/merge for Coherence, [[re-file]] for stale-qa) that a curator approves before validated write-back ([[adr-0020]]); a **Confirmed Remediation** involves no LLM draft — the human confirms an irreversible lifecycle operation (orphan-delete for Freshness C11). A third class sits outside the seam because nothing executes: a **Routed Remediation** is one the system cannot perform at all — the missing ingredient is knowledge only the human can supply — so the affordance *navigates* into an existing workflow with the finding's context (Coverage fill → Import → Ingest, [[adr-0027]]; Freshness C3 fix-the-Source → Upload → force re-ingest, [[adr-0029]]); no gate (nothing to approve), no batch (nothing to run). "Tier B" of the remediation build-out = Gated ∪ Routed; tier A shipped Direct. Re-cut in the 2026-07-02 grill from the earlier synthesis-based binary ([[adr-0024]]). Gates resolve on **human surfaces only** (Console, CLI): the MCP agent surface sees findings and drafts but exposes no gate-resolving tool ([[adr-0026]]). [[lint-pass]] itself stays read-only: a Remediation is always a separate operation triggered from the report, never a side-effect of the lint run.
_Avoid_: Fix (generic; collides casually with bug-fix), Auto-fix (implies no human gate — Gated Remediations always have one), Action (vague, and `suggested_action` already names the advisory text), Destructive Remediation (names C11's risk, not its governance class — say Confirmed).

**Orphan Page**:
A wiki page (`entities/` or `concepts/`) whose `sources` frontmatter cites at least one file that no longer exists under `docs/` — the C11 [[lint-pass]] finding on the Freshness [[lint-axis]]. The remediation-deciding distinction is **full vs partial**: a **full orphan** has *every* cited Source missing (nothing can ground it → eligible for the Confirmed delete, [[adr-0025]]); a **partial orphan** has surviving Sources and may still be validly grounded (→ advisory only: repair the citation, never delete). Distinct from the *ingest-time* orphan that `delete_orphans` cleans (a page a re-ingested Source no longer produces — its Source still exists); a C11 orphan's Source is gone, so ingest never runs for it and no automatic path can clean it.
_Avoid_: Orphan unqualified when the full/partial distinction matters (they have opposite remediations), Dead page (undefined here), Dangling page (reserve "dangling" for red-links — C2 is a *reference to* a missing page, C11 is a *page missing its* Source).

**Re-file**:
The C9 stale-qa [[remediation]] (Authored class): re-derive a live [[filed-answer]]'s answer by running its question through the chat synthesis pipeline with `wiki/qa/` excluded from retrieval, grounding-check the result, and only on pass stage it onto the same slug as a `status: draft` for the Curation Queue gate — the stale answer leaves the corpus until a curator promotes the replacement; a failed re-ground changes nothing ([[adr-0026]]). The verb is the codebase's own (qa.py: "delete the file to re-file fresh") — the Filed Answer goes through filing again.
_Avoid_: Refresh (already means "re-run lint" on the Curation Queue header — same screen, different act), Regenerate/Update (vague). (Historical note: Re-file's subtractive half — [[demote]] — was promoted to a standalone primitive in [[adr-0037]]; "Re-file" still names the full re-derive-and-stage compound, not the bare demote.)

**Demote**:
The standalone Lifecycle primitive that flips a [[filed-answer]] `status: live → draft` in place — content preserved, the page leaving the retrieval corpus for the Curation Queue where a curator re-promotes or discards it. Introduced inside [[re-file]]'s fail-closed retire path ([[adr-0035]]) and promoted to a first-class one-click Direct [[remediation]] in [[adr-0037]], where it becomes C10's exit for a schema-invalid **live** Filed Answer (which `delete` refuses, ADR-0012). The reversible inverse of `promote`; no LLM, no synthesis. The content-lifecycle framework's *soft-demote, not hard-delete* rule (KB-industry survey, 2026-07-07) applied to Filed Answers.
_Avoid_: Retire (names the C9-specific motivation — an un-groundable answer — not the general bit-flip; a demote is also how C10 exits), Unpublish (not the codebase's word), Delete (delete removes an inert page; demote moves a live one back to reviewable draft).

**Source-Rooted Contradiction**:
A C5 [[lint-pass]] Coherence finding whose incompatibility traces NOT to a wiki-layer synthesis divergence but to the **Sources themselves disagreeing** — both pages are each faithfully [[grounding-check|grounded]], yet their cited Sources state incompatible facts (the planted 14 天 vs 30 天 refund window). Structurally **unreconcilable at the wiki layer**: any reconcile draft that stays faithful to both Sources still contradicts itself, so no page edit converges the pair. Detected at reconcile-generate time by the **Convergence Re-judge** (below) — NOT by a grounding failure, which an *existence*-only [[grounding-check]] structurally cannot provide (the self-contradictory Source union individually supports each faithful draft, so grounding passes; the original [[adr-0036]] signal is superseded by [[adr-0038]]). Its exit is the Routed **fix-the-Source** remediation (edit `docs/**` so the Sources agree → re-ingest → re-lint), mirroring C3 ([[adr-0029]]), surfaced through the Reconcile modal's Source-comparison view ([[adr-0036]]). Contrast a **wiki-rooted** contradiction (Sources agree or are conditionally consistent; Reconcile converges it — the re-judge returns `none` and Apply is enabled).
_Avoid_: Deep contradiction (vague), Unfixable finding (it IS fixable — at the Source layer, not the wiki layer), Source conflict unqualified (say Source-Rooted Contradiction to keep the C5-finding framing).

**Convergence Re-judge**:
The check that decides whether a Reconcile draft actually resolved a C5 contradiction — and therefore whether the finding is wiki-rooted (Apply) or a [[source-rooted-contradiction]] (fix-source). After `generate_reconcile` drafts `content_a`/`content_b`, the existing C5 contradiction oracle (`_judge_page_pair`) is re-run on the two **drafts**: `none` → **converged** (the pages now agree — genuine reconciliation, Apply enabled, Wiki-comparison view); `direct`/`tension` → **not converged** (still disagree → source-rooted, Apply disabled, Source-comparison view). Orthogonal to the [[grounding-check]], which verifies each draft's *faithfulness to its Sources* (existence) but structurally cannot see cross-page *consistency* — the same Coherence-vs-Grounding axis split this glossary already draws. Surfaced as the `converged` field on the reconcile response; the modal's default view and the Apply gate both key on it, never on `grounding.passed`. On re-judge error the pair fails safe to not-converged (Apply disabled). Realizes [[adr-0034]]'s deferred claim-level "ceiling" via the oracle already trusted, not a new extractor. See [[adr-0038]].
_Avoid_: Grounding (existence/faithfulness — a different axis; do not conflate), Reconcile check (Reconcile is the remediation; this is its convergence test), Contradiction judge unqualified (that names the audit-time detector; this is its reuse at reconcile-generate time).

### Alias & wikilink navigation vocabulary

> Scoped in the 2026-07-03 grill (curated-layer bundle: C3 exit, alias layer, linkify). See [[adr-0029]] and [[adr-0030]].

**Alias**:
An alternate slug recorded in a Wiki Page's frontmatter `aliases:` list that resolves a `[[wikilink]]` to that page. One shared resolver builds the map *existing slugs ∪ aliases → canonical slug*, and every wikilink-resolution consumer uses it: the C2 [[red-link]] judgment, linkify rendering, and inbound-reference computation. An Alias affects **link resolution only** — never retrieval (BM25 or dense) — so adding one changes what is clickable and what lints red, not what `/chat` finds. It is a **curator-owned field**: `/ingest`'s page overwrite preserves it (like `created`), and assigning one is a Direct [[remediation]] on human surfaces only (Console/CLI; MCP sees aliases, writes none). Distinct from the `[[slug|display]]` pipe, which changes one link occurrence's display text and has no resolution effect.
_Avoid_: Redirect (implies an intermediate hop or a standalone redirect page — the mapping lives on the target page and no extra resource exists), Synonym (suggests retrieval/query expansion, which Aliases deliberately do not do), Nickname (informal).

### PDF conversion vocabulary

> Scoped in the 2026-07-04 grill of issue #419 (scan-heavy corpus confirmed; real-artifact degradations on designed PDFs). See [[adr-0031]] (mechanical path) and [[adr-0032]] (model-assisted path).

**Transcribe**:
The model-assisted operation that converts a text-less PDF (scanned/image-only — or any PDF the curator explicitly forces) from `raw/` into a normalized Markdown Source in `docs/`, by reading page images with a vision model under a faithful-transcription contract: convert form only — no summarization, no synthesis, no completion. Sits beside [[import]] as the second `raw/` → `docs/` converter; a deterministic text-layer probe at the shared entry routes text-less PDFs here automatically, and digital-native PDFs stay on the free mechanical Import path unless forced. Output carries the standard provenance envelope plus `origin: transcribed` and `transcribe_model`, keeping model-derived Sources distinguishable from mechanical Imports and from Captures (the [[capture]] precedent). Idempotent via the same raw-bytes hash-skip as Import, so nondeterministic model output never causes re-billing. See [[adr-0032]].
_Avoid_: Import (mechanical — "No LLM calls" is the load-bearing distinction), OCR (names one implementation technique, not the operation; any vision model may serve it), Ingest (Ingest synthesises docs→wiki; Transcribe converts form and never creates content), Extract (collides with the Grounding Check's claim-extraction language).

**Longform Source**:
A Source whose mechanical heading structure is degenerate relative to its size — no headings, a single stray heading, a dominant preamble, or any single Section over the per-section token cap. Typically a transcribed book or long report whose visual layout carried no heading typography for [[transcribe]] to preserve. Detected structurally (never by filename or page count) after Import/Transcribe; a well-headed employee handbook of the same length is NOT longform — it splits into concept pages through the normal route. A Longform Source receives [[structure-enrichment]] and then ingests via the longform route: one [[hub-page]] plus per-Section concept pages. See [[adr-0033]].
_Avoid_: Book (names one genre, not the structural condition), Large file (size alone does not trigger this — structure does; the async-routing soft cap is a separate, purely capacity concern).

**Structure Enrichment**:
The model-assisted pass that gives a Longform Source the heading structure its layout never carried: an LLM reads the full text, proposes chapter-level boundaries and titles (each bounded by the per-section token cap), and the headings are **materialized into the derived Markdown in `docs/`** — so Sections, Citations, RAG chunk anchors, and the source viewer all work unchanged downstream. Also strips *page furniture*: running headers, footers, and page numbers that repeat on every transcribed page (layout residue, not content). Runs at the Import/Transcribe stage, only when longform detection fires; the raw original in `raw/` stays untouched — immutability lives there, `docs/` is a derived artifact (same reasoning as [[transcribe]]'s provenance envelope). See [[adr-0033]].
_Avoid_: Segmentation (names the analysis, not the whole operation including materialization and furniture stripping), Outlining (collides with `build_outline`, the mechanical classifier input), Chunking (vector_rag's fixed-size term).

**Hub Page**:
The entity-style Wiki Page generated for a [[longform-source]] as its single "about this document" entry point: what the document is, its themes and structure, with wikilinks to each of its chapter concept pages. Answers global questions ("what is this book about?") that no individual chapter page can; chapter-detail questions belong to the chapter pages and, at full fidelity, to the RAG stack. Mirrors the summary-layer role in document-summary indexing (LlamaIndex DocumentSummaryIndex) and the Wikipedia/Wikisource split: the wiki holds the page *about* the document, the Source layer holds the text *of* it. See [[adr-0033]].
_Avoid_: Overview page (vague), Index page (collides with [[wiki-index]]), MOC (Obsidian jargon).

### Reader Feedback vocabulary

> Scoped in the 2026-07-11 mini-grill (user-feedback tracking). Active Language — usable as module / function / variable names per CODING_STANDARD §3.2.

**Reader Feedback**:
An opinion record a reader submits on a single answer card — one **Reaction** (`up` / `down`) plus an optional Comment (≤500 chars) — bound to that specific answer render by a client-minted `answer_id` (UUID minted in the browser at answer completion; no server-side turn id exists). The record is **self-contained**: the client back-fills what it already holds (raw query, stack, session id, citations, grounding reason, answer preview), so the `/chat/stream` SSE contract is untouched. Reader Feedback is opinion data **about** the corpus, never part of it: it enters no retrieval index, is invisible to the [[grounding-check]], and is not a precursor to a [[filed-answer]] — which is why `POST /feedback` sits on the public surface (neither `READ_PATHS` nor `ADMIN_PATHS`; the ADMIN precedent keys on *live-corpus* mutation, which this is not) and is guarded by its own payload and store-size caps instead. Persisted append-only to `.kb/feedback.jsonl` (gitignored, container-ephemeral — resets to empty on deploy); duplicate submissions are resolved at read time by folding on `answer_id`, last write wins.
_Avoid_: Rating (implies a scale, not a binary reaction), Verdict (triple collision — the lint contradiction judge's verdict, the `verify/verdict` commit status, the orchestrator's VERDICT_SCHEMA), Review (collides with code review and the Curation Queue's human review), Vote (implies aggregation toward a decision; each record is an independent opinion).

**Reaction**:
The binary `up` / `down` field of a [[reader-feedback]] record — the GitHub-reactions borrowing chosen precisely to avoid the `verdict` collision above. A Reaction is **always a subjective, user-initiated act** — never pre-selected, never auto-submitted; a single click posts immediately, and the optional Comment is a follow-up append on the same `answer_id` (read-time fold makes it supersede). A Reaction on a [[cannot-confirm]] answer is deliberately allowed — a `down` there says "this should have been answerable" — and **complements, never replaces**, the objective C1 coverage-gap channel (lint aggregates repeated `wiki_gap` Cannot Confirms with no user action involved). Subjective dissatisfaction and objective corpus gaps are separate signals, kept in separate channels.
_Avoid_: Thumb (names the widget, not the datum), Score (numeric connotation), Default reaction (a contradiction in terms — an un-clicked card produces no record).

## Reserved (not yet implemented)

> The future wiki layer (curated synthesis above immutable Sources) is modelled on the patterns in `AgriciDaniel/claude-obsidian` (5.4K⭐ — see ADR-0003), which is the most-starred reference implementation of Karpathy's LLM Wiki gist. We borrow the patterns; the project's positioning is enterprise KB management (FAQ / policy / customer-support), where the curated layer enables governance (ownership, review cadence, audit) on top of the raw Source layer. Vocabulary reserved below mirrors the reference repo's structure so vocabulary aligns from day one. Operational patterns we want to remember but that are not vocabulary live in [`project-docs/inspiration.md#deferred-patterns`](project-docs/inspiration.md) — re-read that section before starting any phase beyond the current prototype.

**Wiki** _(future)_:
The LLM-maintained markdown layer that sits between Sources and queries. Planned as a `wiki/` directory of synthesis pages, organized by entity type (entities, concepts, comparisons, …). Distinct from Sources: Sources are immutable raw input; the Wiki is the LLM's compiled view, where cross-references and synthesis accumulate. **The Wiki is the sole query-time retrieval surface** (Phase 4 W1 model, ADR-0006); `docs/` is read only by `/ingest`, never by `/chat`. When `/chat` finds no adequately scoring Wiki Section, it returns Cannot Confirm with reason `wiki_gap`, which Phase 5 `/lint` aggregates into a coverage backlog (see [[lint-pass]]).
_Avoid_: Notes (too generic), Knowledge base (the project as a whole is a knowledge base; this term refers to the LLM-maintained layer specifically).

**Hot Cache** _(future, folded into Phase 12 / MCP)_:
A small (~500 words) working-memory file at `wiki/hot.md` containing the most recent context — the answer to "where were we?" between sessions. First file an agent reads on session start, before the Wiki Index. Multiple commenters on Karpathy's gist proposed this and the `claude-obsidian` repo implements it; the access pattern there is hot → index → domain-index → individual pages (a 4-level depth budget). The "agent" here is a **persistent agent that returns to work on the KB across sessions** — a role this project gains only once the KB is exposed via MCP and an MCP host (Claude) becomes that agent. The stateless `/chat` is not such an agent and does not read `hot.md` (it is excluded from the [[section-index]] corpus). Hence Hot Cache is deferred and folded into the Phase 12 (MCP) grill; see `project-docs/roadmap.md`.
_Avoid_: Working memory (overloaded with LLM-architecture terms), Session cache (overloaded with HTTP).

**Lint Pass** _(future)_:
A periodic health check over the Wiki: contradiction detection, stale claims, orphan pages, missing cross-references, gaps suggested by repeated cannot-confirm queries. Karpathy's third core operation, alongside Ingest and Query. Operates on the **horizontal axis** — page-vs-page consistency — which is structurally orthogonal to [[grounding-check]]'s vertical axis (page-vs-Source). Neither subsumes the other: a Wiki Page can be individually grounded yet still contradict another grounded page (e.g., two summaries that each cherry-pick different parts of the same nuanced Source). The horizontal axis is therefore Lint Pass's exclusive responsibility — Grounding Check is structurally page-isolated and cannot see it.
_Avoid_: Sanity check (vague), Audit (compliance overtone).

**Gateway** _(future, Phase 9)_:
The single-origin parent app that fronts both Retrieval Stacks. Mounts `markdown_kb` and `vector_rag` as sub-apps, serves the browser UI, and exposes one `POST /chat/stream?stack=wiki|rag` that dispatches to the selected Stack. The designated home for cross-stack and session-scoped concerns: the Retrieval Stack toggle (Phase 9) and Conversation Memory (Phase 11 — [[query-rewriting]] + [[conversation-store]]) live here, so the sub-apps stay stateless and single-turn. Distinct from a Retrieval Stack — a Stack owns retrieval + grounding; the Gateway owns composition, not retrieval. See [[adr-0010]] (and [[adr-0002]] for why the Stacks stay independent).
_Avoid_: Proxy (it dispatches in-process, not via HTTP forwarding), Router (collides with FastAPI's APIRouter), Server (every app is a server).

### Source lifecycle vocabulary

> Scoped in the 2026-07-12 grill (#530). The governed, reversible, audited Source-lifecycle surface — the reframe of [[adr-0003]]'s immutability posture: the app never **silently or irreversibly** mutates a Source; whole-file moves under governance are lawful, byte edits are not. See [[adr-0041]].

**Retire (Source)**:
The Confirmed lifecycle act that removes a [[source]] from the working corpus tree: one atomic whole-file move from `docs/` into the [[source-trash]] — bytes untouched, reversible by [[restore]]. Retire acts on the Source layer only: derived wiki pages are deliberately not cascaded — they surface as C11 [[orphan-page]] findings on the next lint and exit through the existing Confirmed delete (the retire response routes there); the window in between is accepted and curator-held. The confirmation dialog is a server-computed impact preview (which pages go full/partial orphan). Contrast [[demote]]: Demote flips a wiki-layer status bit on a Filed Answer; Retire moves a canonical file.
_Avoid_: Delete/Remove (implies byte destruction — retire never destroys bytes; purge does, and purge is out-of-app per [[adr-0041]]), Archive (suggests a served read-only state; a retired Source serves nothing).

**Restore**:
The Direct inverse of Retire: an atomic move of a [[source-trash]] entry back to its original `docs/` relpath. Refuses when the relpath is occupied or the basename collides elsewhere under `docs/` — never overwrites, never mints a new name (the *refuse, never fall back* posture of [[adr-0036]] §6). Zero bookkeeping: surviving orphan pages are cleared by the next lint recompute (a stale delete click 409s per [[adr-0025]]'s at-delete recompute); already-deleted pages re-synthesize on the next ingest because their `source_hashes` skip-state died with them.
_Avoid_: Undelete (nothing was deleted), Recover (vague).

**Rename (Source)**:
The Direct lifecycle act that changes a Source's basename in place: an atomic same-directory move plus a mechanical re-point of every derived page's `sources` entries and `source_hashes` keys, then one BM25 reindex. No LLM (re-synthesizing unchanged bytes is cost plus drift for zero information). No alias is created — page slugs are outline-minted ([[adr-0027]]) and unaffected by a file rename; an [[alias]] maps page slugs, not filenames.
_Avoid_: Move (v1 renames the basename only; subdirectory moves are out of scope), Alias-redirect (a category error at the file layer — see [[adr-0041]]).

**Source Trash**:
The timestamped tree `<kb-root>/.trash/<UTC-timestamp>/docs/<original-relpath>` holding retired Sources. Deliberately **outside `docs/`**, so every Source scanner (upload origin resolution, ingest pairing, lint citation resolution) is structurally blind to it — no exclusion lists, no re-ingest resurrection. The tree itself is the physical audit trail (what, when, from where); each lifecycle act also writes one bounded append-only log line. Readable through the read whitelist for pre-restore inspection; listed by the trash endpoint. No in-app purge and no retention expiry: emptying it is an out-of-app act.
_Avoid_: Tombstone (names the industry *pattern*; the artifact here is a trash tree), Recycle bin (OS flavour), Archive (see Retire).

## Flagged ambiguities

**Slug generation for non-ASCII headings.** *Resolved (2026-05-29, Phase 16 Chinese-corpus grill).* `slugify()` preserves Unicode letters (including CJK) instead of stripping to `[a-z0-9]`, following the GitHub/Obsidian Unicode anchor convention — so `## 退款政策` slugs to `退款政策`, distinct CJK headings no longer collide on the `section` fallback, and the `-2`/`-3` suffix rule still resolves any genuine collision. The `section` fallback now fires only for headings with no slug-able character at all. See the [[citation]] term. Companion decision: retrieval tokenisation is language-agnostic (Latin scripts by word; CJK by character n-gram) so a Chinese corpus is indexable and queryable end-to-end — full design captured in the Phase 16 PRD.

**Section ID is not stable across renames.** *Partially resolved (2026-07-12, [[adr-0041]]).*
A Section's identifier is derived purely from `{source-filename}#{heading-slug}`. Renaming a Source file or editing a heading text changes the identifier, which silently invalidates every Citation a caller already received. Accepted unflagged for the prototype because the PROMPT.md verification freezes both filenames and headings; a content-derived stable identifier (UUID sidecar or content hash) becomes worth implementing at the same time the wiki layer is added, since that is when accumulated Citations start being expensive to invalidate. *The filename half is now governed: an in-app [[rename-source]] mechanically re-points every derived `sources`/`source_hashes` reference in the same atomic act, so a governed rename no longer silently invalidates them. Heading-edit instability and out-of-app renames remain exactly as flagged.*
