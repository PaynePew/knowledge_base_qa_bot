# Inspiration & Reading Notes

Curated outside references that informed this project's design, with explicit notes on which ideas we adopted, which we deferred (and **when** to re-evaluate them), and which we rejected. Useful for interview prep and — more importantly — for surfacing the right patterns when the project moves into its next phase.

## How to use this file

- **Each phase boundary**, before writing code, search this file for the phase tag. The relevant patterns surface together. Phase tags: `phase: wiki`, `phase: ingest`, `phase: query`, `phase: lint`, `phase: conversation`, `phase: streaming`, `phase: production`.
- Patterns that became formal vocabulary live in `CONTEXT.md` (reserved terms section).
- Patterns that became formal decisions live in `project-docs/adr/`.
- Operational patterns that are not yet either — but are too valuable to forget — live in [§ Deferred Patterns](#deferred-patterns) below, each with a phase tag.

---

## Primary reference implementation

### `AgriciDaniel/claude-obsidian` (5.4K⭐, actively maintained)

**Source:** <https://github.com/AgriciDaniel/claude-obsidian>

The highest-signal community validation of Karpathy's LLM Wiki pattern: a Claude Code plugin + Obsidian vault that ships the three-layer structure, Hot Cache, Wiki Log, Source Templates, lint workflow, and `/autoresearch` autonomous loop. We adopted it as the **concrete target shape** for the future wiki layer ([ADR-0003](adr/0003-w2-layered-wiki-target-claude-obsidian.md)).

Repository structure observed (May 2026):

```
.raw/             ← Sources (immutable; their .raw is our docs/)
wiki/
├── hot.md        ← Hot Cache (~500 words)
├── index.md      ← Wiki Index
├── log.md        ← Wiki Log
├── entities/
├── concepts/
├── comparisons/
├── folds/
└── canvases/
_templates/       ← 5 Source Templates: comparison, concept, entity, question, source
_attachments/
WIKI.md           ← Karpathy "schema" document at root
```

Skills observed: `/wiki`, `ingest [source]`, `query: [question]`, `lint the wiki`, `/save`, `/autoresearch`, `/canvas`.

Cross-project access pattern documented in their README (verifies the L0/L1/L2/L3 token budget pattern is real, not single-commenter speculation):

```
1. Read wiki/hot.md first (recent context, ~500 words)
2. If not enough, read wiki/index.md
3. If domain specifics, read wiki/<domain>/_index.md
4. Only then individual wiki pages
```

---

## Foundational reference

### Andrej Karpathy — LLM Wiki (April 2026)

**Source:** <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>

The seed of this project's long-term direction. Three-layer pattern: raw Sources (immutable), an LLM-maintained Wiki (markdown), and a Schema file (`CLAUDE.md` / `AGENTS.md` / `WIKI.md`) codifying workflows. Operations are Ingest, Query, and Lint. The Wiki is "a persistent, compounding artifact" rather than a query-time RAG retrieval.

| Karpathy concept | Our status |
|---|---|
| Three-layer architecture (Sources / Wiki / Schema) | Adopted as target via ADR-0003; prototype implements only Sources + Section Index |
| `index.md` as LLM/human navigation | Reserved as **Wiki Index** in `CONTEXT.md` |
| `log.md` chronological append-only log | Reserved as **Wiki Log** in `CONTEXT.md` |
| Ingest / Query / Lint operations | Reserved as **Ingest** / **Lint Pass** in `CONTEXT.md`; only Query is implemented |
| `qmd`-style local hybrid search (BM25 + vector rerank) | Deferred; current retrieval is BM25-only via Markdown KB |

### Discussion-thread signals

The ~835-comment thread under Karpathy's gist surfaced many patterns. We weight them by *multiple-source consensus* + *implementation evidence*, not single-commenter opinions:

- **High-signal** (consensus across multiple commenters *and* shipped in `claude-obsidian`): Hot Cache; structured Wiki Log; per-type Source Templates; the L0–L3 read-depth budget; Lint as a periodic operation; the Two-output rule (every query produces both an answer and a Wiki update).
- **Medium-signal** (one or two commenters, no shipped reference yet): SHA-256 provenance binding; compile-at-query-time deltas; speculative `[[wikilink]]` red links; reflect step between operations.
- **Down-weighted** (single anonymous commenter, no verified track record): tournament-result-style claims from accounts whose repos cannot be independently verified. An earlier draft of this file leaned too heavily on one such commenter; that has been corrected.

---

## System-design reference

### "Design Q&A Support Agent" (PDF, buildmoat.org)

**Local copy:** `C:\Users\MaxL\work\ebook\系統設計實戰營\真實大型應用設計\Design Q&A Support Agent.pdf` (25 pages, 2.4 MB)

A system-design-interview-grade walkthrough for a Booking.com / Agoda style Q&A support agent. Production scale; the architectural ladder is what this prototype is the first rung of.

**Adopted into the project:**

- **Four-layer anti-hallucination framing** (Prompt engineering → Retrieval quality → Output validation → Fallback). Prototype implements layers 1 and 4 (ADR-0001 + score threshold); layer 3 (**Grounding Check**) is reserved in `CONTEXT.md` and added to the README stretch goals; layer 2 (hybrid + rerank) is the natural next step once `vector_rag/` is implemented.
- **Validates structure-based chunking** for FAQ-shaped content (the PDF explicitly says: use the Q&A pair as the chunk). This matches our **Section per leaf heading** choice in `CONTEXT.md`.
- **Validates the `sources: [...]` response shape**.

**Reserved for future work:** multi-turn Query Rewriting + Conversation Store, SSE streaming, knowledge-freshness via versioning and webhooks.

**Out of scope for this project:** Booking DB lookup / personalized queries, query routing with intent classification, managed vector stores (Pinecone/Weaviate/pgvector), WebSocket sticky-session + L4 LB, gRPC between services.

### Interview talking points distilled from the PDF

1. *"I chose Markdown KB over Vector RAG because at this corpus size, BM25 + inspectable `.kb/index.json` is more debuggable and has zero per-query embedding cost. The vector_rag app is preserved for the hybrid retrieval and rerank layer once the corpus warrants it."*
2. *"Production-grade anti-hallucination is four layers: prompt rules, retrieval quality, output validation, and fallback. The prototype implements layers 1 and 4 — see ADR-0001."*
3. *"Knowledge freshness at production scale needs webhooks, versioning, and write-then-delete. The prototype uses manual `POST /index` because the corpus is hand-curated. The upgrade path is explicit, not accidental."*
4. *"Streaming uses SSE because the data flow is one-way and the protocol is dramatically simpler than WebSocket. WebSocket is only worth it when the server initiates messages."*

---

## Deferred Patterns

Operational patterns we want to inherit when the corresponding phase begins. Each entry is grep-able by `phase: ...` tag.

### Pattern: Two-output rule
**phase: ingest, phase: query** · multiple gist commenters + implemented in `claude-obsidian`

Every `/ingest` and every interesting `/query` produces **two** outputs: (1) the immediate answer to the user, and (2) a structured update to the Wiki (a new page, a cross-reference addition, a contradiction flag). Without an explicit write-back path, knowledge evaporates into chat history and the compounding artifact never compounds. **Trigger condition for adoption:** the first `/ingest` operation, or the first `/save` of a `/chat` answer.

### Pattern: L0/L1/L2/L3 read-depth budget
**phase: wiki, phase: query** · `claude-obsidian` README + several gist commenters

Token budget for context loading: ~200 tokens (Hot Cache), ~1–2K (Wiki Index), ~2–5K (domain-level index or search results), ~5–20K (full pages). Read from shallowest to deepest, stop when the question can be answered. **Trigger condition for adoption:** when the Wiki layer ships and the LLM starts reading from `wiki/`.

### Pattern: Frontmatter schema for Wiki pages
**phase: wiki, phase: ingest** · `claude-obsidian` Source Templates + several gist commenters

Every Wiki page carries YAML frontmatter with at minimum: `created`, `updated`, `confidence` (`low|medium|high`), `status` (`draft|live|stale|superseded`), `sources` (list of Citations the page derives from), `open_questions` (free-text list). Enables Lint Pass to run as SQL-like queries. **Trigger condition for adoption:** when the first **curated** Wiki page is generated (i.e., `POST /ingest` writes the first `wiki/entities/*.md` or `wiki/concepts/*.md`). Generated artifacts like `wiki/index.md` do NOT trigger this pattern — they are filesystem projections, not governance subjects. See PRD #17 for the artifact taxonomy that draws this line.

### Pattern: Hot Cache as session bridge
**phase: wiki, phase: conversation** · multiple commenters + `claude-obsidian` ships `wiki/hot.md`

A small (~500-word) `wiki/hot.md` rewritten at the end of every meaningful session: "what we just worked on, current open threads, last decisions." First file the LLM reads on new session. Solves the "where were we?" problem without forcing a Wiki Index scan every time. **Trigger condition for adoption:** when multi-session work begins (i.e., once the user starts using the project beyond single-shot Q&A).

### Pattern: `@import` Wiki Index into agent context
**phase: wiki, phase: production** · @vitalii-ivanov-rakuten and others

Put `@import wiki/index.md` (or equivalent) into `CLAUDE.md` so the Wiki Index is **always** in the LLM's context, no skill activation required. Combined with a `git submodule` setup for cross-project reuse. **Trigger condition for adoption:** when the Wiki is used across multiple projects (or before — review when wiki layer ships).

### Pattern: Speculative `[[wikilinks]]` (red links)
**phase: wiki, phase: ingest** · @xoai + `claude-obsidian` likely uses this in `wiki/folds/`

When the LLM mentions a concept that does not yet have its own page, create a `[[concept-name]]` link anyway. The link is unresolved (red in Obsidian); the next `/lint` pass surfaces these as candidates for new pages. Lets the wiki grow organically without forcing a page creation decision mid-ingest. **Trigger condition for adoption:** during `/ingest` implementation.

### Pattern: Auto-pruning expired Wiki entries
**phase: wiki, phase: lint** · @glaucobrito

Tactical lessons / temporary notes carry a TTL (e.g. 30 days). `/lint` archives expired entries to `wiki/.archive/` rather than growing `lessons.md` indefinitely. Without this, the Wiki accumulates stale advice and Lint Passes get slower over time. **Trigger condition for adoption:** when the Lint Pass is implemented.

### Pattern: Compile-at-query-time deltas
**phase: query, phase: production** · @Jwcjwc12

Each `/query` recomputes only the *delta* of the Wiki that changed since last query, rather than reading the full index. Important when the Wiki grows past ~200 pages (per @plundrpunk's 12-month production report — `index.md` blows the context window at that point). **Trigger condition for adoption:** when the Wiki passes ~100 pages, or when query latency starts becoming a complaint.

### Pattern: SHA-256 provenance binding
**phase: ingest, phase: lint** · @Jwcjwc12, @tomjwxf, `palinode` repo

Each Wiki proposition records the SHA-256 of the Source section it derives from. On `/lint`, recompute Source hashes and flag mismatches as stale (the Source changed, the Wiki page didn't update). Stronger than the current `last_updated` heuristic. **Trigger condition for adoption:** when the Wiki has > 50 pages and manual stale-tracking becomes a problem.

### Pattern: Reciprocal Rank Fusion / cross-encoder rerank
**phase: query** · @bitsofchris, @marktran0710

Overfetch top-K × 3, dedupe with MMR, rerank with a cross-encoder before passing to the LLM. Specifically valuable when `vector_rag/` is implemented and we want a hybrid retriever. **Trigger condition for adoption:** when implementing the hybrid retrieval layer in `vector_rag/`.

### Pattern: PageIndex / tree traversal retrieval
**phase: query** · @earaizapowerera

For deeply nested Wiki structures, replace flat vector chunks with a tree-traversal retriever that walks the Wiki's hierarchical structure. Alternative to vector chunks when the Wiki is well-structured. **Trigger condition for adoption:** when the Wiki passes ~500 pages and a flat index becomes lossy.

### Pattern: Object-Oriented RAG (OORAG)
**phase: production, phase: ingest** · @minchieh-fay

Treat each entity (Source, Section, Wiki page) as a queryable object with its own methods, rather than as opaque text chunks. Reported precision lift 60–70% → 95%+. Conceptually clean fit with our existing Section/Source/Citation glossary. **Trigger condition for adoption:** post-prototype, when precision becomes the bottleneck (rather than speed or coverage).

### Pattern: Source-type-specific extraction pipeline
**phase: ingest** · gist commenter consensus (multiple)

Classify a Source before extracting from it, and route to a pipeline tuned for its kind. A 50-page report needs deep summarization + multi-page synthesis; a 2-page email needs light entity extraction + a single Wiki update. Forcing both through the same pipeline produces uniformly shallow summaries. Closely paired with the **Source Template** vocabulary in `CONTEXT.md`: classification picks the template, pipeline runs the extraction. **Trigger condition for adoption:** at the start of `/ingest` implementation, before designing the ingest workflow.

### Pattern: Reflect step between operations
**phase: ingest, phase: lint** · @bendetron

Insert a `reflect` step between `ingest`, `compile`, `query`, and `lint`. Each Wiki mutation carries a small decision record: what changed, what was considered, why this version was chosen. Surfaces drift early — if `reflect` keeps recording "compacted away X to make room for Y," that is a signal the Wiki is hitting context-budget pressure and needs structural change. **Trigger condition for adoption:** when a second Wiki-mutating operation is added beyond `/ingest` (e.g., `/save` or `/lint`-driven edits).

### Pattern: Skills / MCP distribution of the Wiki
**phase: production** · `luna-prompts/skillnote`, `YesIamGodt/knowledge-pipline`, gist commenter consensus

Package the Wiki (or a slice of it) as a Claude Skill or MCP server so it can be loaded into other projects without copy-pasting. Adds a registry layer: install via `npx skills add ...`, version it, depend on it, share across teams. The natural home for the cross-project access pattern documented in `claude-obsidian`'s README. **Trigger condition for adoption:** when this Wiki is wanted in a second project, or when sharing it with a team or collaborator becomes interesting.

### Pattern: Two-dimensional Wiki content (machine + human)
**phase: wiki, phase: ingest** · @gitdexgit

Each Wiki page carries two layers in one file: a **machine version** (dense, token-efficient, frontmatter-rich, the form the LLM consumes) and a **human TL;DR** (paragraph or two for a human skimmer). Sections clearly delimited (`## For the agent` vs `## TL;DR`). Solves the tension between "compact for LLM context budget" and "readable for the human curator." Likely composes well with the **Frontmatter schema** deferred pattern above and with the **Source Template** vocabulary. **Trigger condition for adoption:** at first Wiki-page template design — defer the decision to commit to two layers until you have at least one real page to look at.

---

## Considered and Rejected

Patterns that surfaced in the discussion and that we deliberately do **not** plan to adopt. Recorded here so future-us does not re-litigate them (or wonder, six months from now, "why did we skip this?"). A rejection here is reversible — if circumstances change the trade-off, reopen.

### Rejected: Ed25519 signed receipt chain
**Source:** @tomjwxf, IETF draft `draft-farley-acta-signed-receipts`

The idea is to cryptographically sign every LLM response so a third party can verify offline that a particular answer came from a particular Wiki state. This is an enterprise audit / regulatory-compliance pattern — it solves a zero-trust threat model where the answer's recipient does not trust the sender.

**Why we skip it:** a personal Wiki has no such threat model. The user *is* the only writer, the only reader, and the only auditor. The SHA-256 provenance binding (a deferred pattern above) gives us source-state staleness detection, which is the *useful* property of receipt chains in this context. Signing adds key management, signature storage, verification tooling, and zero recipient-side value. Reopen if this project ever ships into a context with external auditors or compliance requirements.

### Rejected: Caveman / aggressive prose compression for Wiki content
**Source:** @gitdexgit, @jurajskuska, `JuliusBrussee/caveman`

The idea is to compress all Wiki page content into a token-minimal form (drop articles, prepositions, filler) for ~32% token savings.

**Why we skip it:** this is a *reader preference* — useful in LLM-to-LLM dialogue (we use the caveman skill ourselves for compact replies) — not a *Wiki architecture* decision. Compressed Wiki content fights several patterns we *do* want: the Frontmatter schema expects normal prose for `confidence` and `open_questions` fields; the Two-dimensional Wiki content pattern explicitly wants the human-TL;DR layer to read naturally; the Lint Pass becomes harder when prose is mangled. Token savings at the Wiki layer (~32%) are also tiny compared to savings from the L0/L1/L2/L3 read-depth budget (which can cut context tokens by 10×+). Use caveman for chat compression, not Wiki content.
