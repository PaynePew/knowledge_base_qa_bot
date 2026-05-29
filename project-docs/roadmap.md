# Roadmap

The implementation sequence for the project, from the current prototype through the full set of stretch goals. This file is the source of truth for "what comes next and why" — every new grill or implementation session should read this before scoping work.

## Current state (2026-05-26)

| Phase | Scope | Status |
|---|---|---|
| Prototype | Markdown KB BM25 + Section Index + Cannot Confirm fallback | ✅ Done |
| Phase 1 | Grounding Check (post-LLM claim-level verification) — [ADR-0004](adr/0004-post-llm-grounding-check.md) | ✅ Done |
| Phase 2 | Wiki Index Generation (mechanical projection of `.kb/index.json` → `wiki/index.md`) — PRD #17 | ✅ Done |

## Sequencing principle

Phases are ordered by four criteria, in this priority:

1. **Narrative coherence** — phases on the same axis (Karpathy layered KB / product surface / retrieval comparison) ship consecutively so the interview story stays consistent.
2. **Hard dependency** — a phase that writes to `wiki/` must precede a phase that reads from `wiki/`, or the written content is dead weight.
3. **Diminishing returns** — later phases add less interview signal per hour. The sequence has natural stopping points (see § Stopping points).
4. **Unlock effect** — early phases activate deferred patterns (see [`inspiration.md`](inspiration.md)) that later phases reuse, avoiding repeat engineering.

## Roadmap (Phase 3-16)

| Phase | Scope | Est | Axis | Status |
|---|---|---|---|---|
| **3** | `/ingest` — Source → `wiki/entities/`, `wiki/concepts/`. LLM-assisted synthesis. Per-type Source Templates. Frontmatter schema becomes real (no longer deferred). | 12-19h | Karpathy write | ✅ Done |
| **4** | W2 layered retrieval — `/chat` queries `wiki/` + `docs/`. Flips `SOURCE_DIRS = [DOCS_DIR]` to `[DOCS_DIR, WIKI_DIR]` (the upgrade hook ADR-0003 already pre-installed). | 3-6h | Karpathy read | ✅ Done |
| **5** ⭐ | `/lint` — Lint Pass. Contradiction detection, stale claims, orphan pages, gaps from repeated cannot-confirm queries. Completes Karpathy's Ingest + Query + Lint trio. **Hard deps: (a) slice 7-3 (#92, merged) writes `content_sha256` to `docs/` frontmatter — required for "raw drifted but docs not re-imported" staleness detection without recomputing hashes online; (b) Phase 3 amendment (issue #93, merged) writes `source_hashes` to wiki frontmatter — required for full-chain raw→wiki drift detection. NOTE: Phase 5 lint must treat empty `source_hashes` as "unknown drift" (not "no drift") to avoid false-negative reports on Phase 6 legacy pages.** | 6-10h | Karpathy maintain | ✅ Done |
| **6** | Answer Filing — `/chat` answers (when high-confidence) get written back to `wiki/qa/*.md`. Closes the Two-output rule on the query side (the Source side closes in Phase 3). PROMPT.md stretch #7. | 4-8h | Karpathy Two-output | ✅ Done |
| **7** | Multi-Format Import — `raw/*.txt` and `raw/*.html` normalized into `docs/*.md` via a pre-processing step (`markdownify` or equivalent), so non-Markdown Sources can enter the pipeline without losing the canonical Markdown format guarantee. PROMPT.md stretch #4. Includes hash chain follow-up: Phase 3 amendment (issue #93) wrote `source_hashes` to wiki frontmatter using `content_sha256` written by Phase 7-3 (PR #92). | 3-5h | input flexibility | ✅ Done |
| **8** | Paraphrase Comparison — revive `vector_rag/` (langchain 0.3 → 1.x), run paraphrased queries against both retrieval strategies, produce a comparison report. Empirically validates ADR-0002. PROMPT.md stretch #9. | 6-12h | retrieval comparison | ✅ Done |
| **9** | Streaming + Browser UI — `POST /chat/stream` (SSE) + a minimal HTML page rendering sources first, then streamed answer. PROMPT.md stretches #2 + #3. All 6 slices merged (#117/#119/#120/#121/#118/#122): Gateway introduced (`gateway/` package, ADR-0010), Wiki + RAG SSE streams, filing parity, chosen UI wired to real SSE stream. | 4-8h | product surface | ✅ Done |
| **10** | Hot Cache — `wiki/hot.md` (~500 words), rewritten at end of meaningful sessions. First file the agent reads. | 2-4h | convenience | **Deferred → folded into Phase 12 (MCP)** (grill 2026-05-29, consumer analysis; see prep note) — the original spec has no consumer until an MCP host agent exists |
| **11** | Conversation Memory — Query Rewriting + Conversation Store. Multi-turn `/chat`. PROMPT.md stretch #8. | 6-10h | multi-turn | ✅ Done (slices #159–#163 merged; ADR-0013) |
| **12** | Alternative Interfaces — CLI (`kb index`, `kb ask`) + MCP server. PROMPT.md stretch #5. | 4-8h | packaging | Planned |
| **13** | Hybrid Retrieval over Wiki — BM25 + dense vector **both over the curated `wiki/` Section corpus**, fused via Reciprocal Rank Fusion (RRF). Served as a **third `Retriever`** alongside Stack A (Wiki) and Stack B (Vector RAG). Additive — does **not** touch the two existing apps. Optional cross-encoder reranker deferred to a follow-on. Relates to #107 (becomes a 3rd implementation under the pluggable `Retriever` protocol if that lands). | 14-22h (RRF-only; +4-6h with reranker) | retrieval quality (new axis) | Future / want-to-do |
| **14** | **Getting-Started documentation** — rewrite `README.md` (+ `.env.example`) so a newcomer can go `git clone` → install → configure → run without reading source. Covers: clone/install steps, `.env.example` AI API keys (which to set, required vs optional), starting the back-end + front-end (gateway/browser UI) servers, importing Sources + generating fake data, an API reference for `/index` / `/ingest` / `/lint` / `/chat` (what each does, **how & when** to call it), and how to run the Wiki-vs-Vector-RAG comparison. **Wrap-up phase** — documents shipped capability, adds no features; full coverage implies Phases 3/4/5/8/9 have landed. Full content TBD via `grill-with-docs`. | TBD (pre-grill) | documentation / onboarding | Planned — grill pending |
| **15** | **Operator Console + Upload** — curator-facing management surface on the gateway (the write/maintain counterpart to the reader chat UI). Drives the Wiki lifecycle (Import → Ingest → Index → Lint → Filed-Answer promotion) and exposes RAG index rebuild as a parallel action (orchestrates each stack, never merges — ADR-0002); adds a `docs/`+`wiki/` resource browser and a drag-drop **Upload** drop zone (`.html`/`.txt` → `raw/`; `.md` → `docs/`). Single local operator, no auth. | TBD (pre-PRD) | product surface | Planned — grill done 2026-05-29 (CONTEXT terms + [ADR-0011](adr/0011-upload-separate-from-import.md) Upload boundary + [ADR-0012](adr/0012-delete-inert-filed-answers-only.md) DELETE inert qa added), PRD pending |
| **16** | **Chinese / language-agnostic corpus support** — make retrieval language-agnostic so a Chinese Source + Chinese query works end-to-end: Unicode-aware tokeniser (CJK **bigram** + unigram fallback; ASCII byte-identical → zero English regression), Unicode-preserving `slugify`, and output-language-follows-input prompt directives. **W1 linchpin:** `/ingest` must synthesise wiki pages in the Source's language since `/chat` reads only `wiki/`. `Cannot Confirm` stays the English sentinel; product i18n out of scope. | 4-7h (est) | language coverage | PRD #164 (`ready-for-agent`) |

**Total: 48-87h** for Phases 3-12 (excluding the ⭐ recommended stopping point at Phase 5, which clocks in at 17-31h after Phase 3). **Phase 13 sits on a separate "retrieval quality" axis — it is off the linear Karpathy narrative and is an optional add-on, not on the critical path.** **Phase 14 is a documentation / onboarding wrap-up — also off the feature axis; it tracks and documents whatever has shipped rather than adding capability.** **Phases 15 (product surface) and 16 (language coverage) are likewise additive and off the linear Karpathy narrative: Phase 15 has its CONTEXT vocabulary reserved with the PRD still pending; Phase 16's PRD is published as #164.**

## Hard dependencies

These are not preferences — they are constraints. Violating them produces broken demos.

| Constraint | Why |
|---|---|
| Phase 4 **must** follow Phase 3 | `/ingest` writes Wiki pages. Without W2 retrieval, `/chat` never reads them. Demo collapses: "the LLM wrote a page but the bot still queries `docs/`." |
| Phase 6 **must** follow Phase 4 | Answer Filing writes to `wiki/qa/`. Same logic — dead content unless `/chat` reads `wiki/`. |
| Phase 10 **depends on** Phase 12 (MCP) | Hot Cache's consumer is a *persistent agent that returns to work on the KB across sessions* — the precondition for the whole pattern. The current `/chat` is stateless and does **not** read `wiki/hot.md` (it is excluded from the Section Index corpus by a correctness invariant), so "must follow Phase 4 / same dead-content logic" was a mis-analysis: there is no `/chat` reader. That persistent agent only materialises once the KB is exposed as MCP tools and an MCP **host** (Claude) becomes the agent. Building Hot Cache before then produces dead weight. (grill 2026-05-29, consumer analysis.) |
| Phase 8 **must** follow Phase 4 | A fair Paraphrase Comparison compares the Karpathy-Wiki stack (Stack A — BM25 over the LLM-synthesised `wiki/` layer, ADR-0006 W1: wiki is the sole query surface) against Vector RAG (Stack B), both fed the same raw corpus. Running it before the wiki layer exists produces misleading data. |
| Phase 11 **best** after Phase 9 | Multi-turn UX needs a streaming UI to demo well. Without it, conversation memory is two endpoints, not a conversation. |
| Phase 5 `/lint` drift checks **must** follow slice 7-3 (#92) + Phase 3 amendment (issue #93, merged) | `content_sha256` in `docs/` frontmatter (slice 7-3, merged) and `source_hashes` in wiki frontmatter (Phase 3 amendment #93, merged) are the preconditions for full-chain raw→wiki drift detection. Without them, Phase 5 lint can only check structural invariants, not staleness. Both deps have now landed. |

## Stopping points (interview-ready snapshots)

Each row corresponds to a complete, self-contained interview narrative. Stopping at any of these is honest and defensible.

| Stop | Story | Cumulative |
|---|---|---|
| After Phase 4 | "Minimum layered KB — `/ingest` writes Wiki pages, `/chat` queries them. Same query before/after ingest shows different answers." | 15-25h beyond prototype |
| After Phase 5 ⭐ | "Complete Karpathy three-operation pattern (Ingest + Query + Lint). The KB maintains itself." | 21-35h |
| After Phase 6 | "Two-output rule fully implemented on both sides — Sources are ingested, good Queries are filed." | 25-43h |
| After Phase 7 | "+ accepts `.txt` / `.html` Sources via the `raw/` normalization step." | 28-48h |
| After Phase 8 | "+ quantitative retrieval comparison data. Concrete numbers on when BM25 wins vs Vector RAG." | 34-60h |
| After Phase 9 | "+ product surface (SSE streaming + minimal browser UI)." | 38-68h |
| After Phase 12 | Full stretch-goal walkthrough. | 48-87h |

⭐ **Recommended stop: Phase 5.** Completes the Karpathy narrative with the strongest interview leverage per hour. Beyond Phase 5, marginal narrative value drops sharply — additional phases are polish, not core capability.

## Per-phase prep notes

Detail not worth duplicating into the phase tables, but important enough that the relevant phase's PRD must address it.

### Phase 3 (`/ingest`)

- Triggers four deferred patterns from `inspiration.md`: **Source Template** vocabulary, **Frontmatter schema for Wiki pages**, **Source-type-specific extraction pipeline**, **Two-output rule** (ingest side). Re-read each before designing.
- Multi-Format Import (`.txt`/`.html` → `.md` normalization) is **NOT** merged into Phase 3 — kept separate as Phase 7. Earlier "merge saves 1-3h" speculation was wrong: per PROMPT.md spec, Multi-Format Import is a `raw/` → `docs/` pre-processing pipeline, completely disjoint from `/ingest` code path. No shared code = no shared work.

### Phase 5 (`/lint`) — demo corpus problem

The current `docs/` (3 FAQ Sources, 9 Sections) has no stale content, no contradictions, no orphan pages, no gaps. `/lint` will run on empty input and produce a boring "all clean" report. Plan to **manually plant test cases as part of Phase 5 design**: 1-2 contradiction pairs (e.g., two Sources giving different refund timelines), 1 stale-flagged Section, 1 cannot-confirm query repeated enough times to surface a "gap" flag. These plants belong in a separate `eval/lint_fixtures/` corpus, not in `docs/` (which stays canonical).

### Phase 8 (Paraphrase Comparison) — three gotchas

1. **Adopt DeepEval as the runner/dataset/report framework, but hand-write the metric.** *(Amended at Phase 8 design — PRD #100.)* The earlier "don't adopt Ragas/DeepEval/LlamaIndex" guidance was disproven by docs research: DeepEval natively supports retriever comparison, custom metrics, and cross-family judge config, with retrievers as plain callables. The C5c hit metric (hit_rate@k + MRR) stays hand-written as a DeepEval `BaseMetric` subclass — borrowing the framework's runner/dataset/report at the leaf while keeping the project's opinionated metric at the joint (ADR-0005 "borrow at leaf nodes, hand-write at joints"). LlamaIndex (needs retriever wrapping) and RAGAs (less pytest-native) were the runners-up.
2. **Grow the corpus first.** 3 FAQ Sources is too small for statistically meaningful comparison. Target 15-20 fake Sources generated via LLM prompts to a fictional company (e.g., "Acme Shop"). Commit to `fake-docs/` as eval-only corpus.
3. **LLM-generated paraphrases have systematic bias favoring Vector RAG.** Both are LLM-trained, so synonyms generated by GPT-4o-mini fall inside the embedding space that the same model family encodes. Mix in ~10% hand-written adversarial samples (typos, cross-lingual queries, dialect terms, industry jargon) to correct. Call out this limitation proactively in the report — it's an interview plus, not a minus.

### Phase 10 (Hot Cache) — DEFERRED, folded into Phase 12 (grill 2026-05-29, consumer analysis)

**Status: Phase 10 is no longer a standalone phase. Its design is handed off to the Phase 12 (MCP) grill — see the Phase 12 prep note below.**

The earlier "reorder after Phase 11" decision was superseded by a deeper consumer analysis. Hot Cache (Karpathy / `claude-obsidian`) is **the persisted short-term working memory of a persistent agent** that returns to work on the KB across sessions: the agent rewrites `wiki/hot.md` at session end and reads it first (L0 of the hot→index→domain→page read-depth budget) at the next session start, to recover "where were we?" without re-scanning the whole Wiki Index.

That pattern has a hard precondition: **a persistent agent.** This project does not have one — `/chat` is a stateless request/response bot, and `wiki/hot.md` is deliberately excluded from the Section Index corpus (CONTEXT.md), so `/chat` structurally never reads it. The two candidate consumers both fail:

- **Reader cross-session memory** (inject hot.md into Query Rewriting / chat): thin value for a stateless Q&A bot, and it fights the ADR-0001/0006 grounding contract (uncited conversation summary leaking into the answer path).
- **Curator situational awareness** (a "where were we?" panel in the Operator Console): redundant — the Phase 15 Console already shows real KB state directly (resource browser over `docs/`/`wiki/`, lint findings, Curation Queue, `log.md`). A lossy summary of state the Console shows authentically adds nothing.

**The consumer only materialises with Phase 12 (MCP).** Once the KB is exposed as MCP tools and an MCP **host** (Claude) becomes the persistent agent, hot.md becomes the genuine Karpathy Hot Cache — and specifically *KB-local, agent-agnostic working memory* (it travels with the KB, so a different agent / install / teammate that attaches to the same KB inherits "where were we" without the original host's private memory). That fits the project's enterprise-KB-management positioning (ownership / handoff / audit), which is why we lean toward keeping it rather than dropping it outright.

**Seam correction (important for the Phase 12 grill):** Phase 11 left `ConversationStore.dump(session_id)` open as "the Phase 10 Hot Cache seam." That was predicated on the now-rejected browser-reader framing. The MCP host does **not** go through `/chat/stream`; it calls MCP tools directly, so the MCP Hot Cache's input is the **host's own MCP session**, not the Conversation Store. Do not build the MCP Hot Cache on top of `dump()`. The `dump()` docstring and the CONTEXT.md Conversation Store term have been neutralised accordingly (no longer claim to be "the Phase 10 seam").

**Open decision deferred to the Phase 12 grill:** is KB-local agent-agnostic working memory worth it vs. just letting the MCP host keep its own memory (CLAUDE.md / host memory)? We leaned yes, but it is the real make-or-break call and should be re-litigated when MCP is actually scoped.

### Phase 11 (Conversation Memory) — filing × toggle interaction (locked at Phase 9 grill, 2026-05-28)

Conversation Memory lives in the **gateway**, not the sub-apps (Phase 9 decision: sub-apps stay stateless single-turn; Query Rewriting + Conversation Store are gateway concerns). Two constraints the Phase 11 PRD **must** honour:

1. **File the rewritten self-contained query, not the raw elliptical follow-up.** When a multi-turn answer is filed (Phase 6 Answer Filing, Wiki stack only), `frontmatter.question` must be the Query-Rewriting output (self-contained), so `wiki/qa/*.md` stays independently readable. Filing the raw follow-up ("and how long does *that* take?") would write an incoherent standalone page. Query Rewriting's output is self-contained by construction, so this is free — just file the rewritten form.
2. **Cross-stack fact leakage is already firewalled by the Grounding Check — do NOT add a redundant per-stack conversation partition.** Even if turn N-1 was answered by Vector RAG and turn N toggles to Wiki, the Wiki answer's claims must pass grounding against `wiki/` Sections, so RAG-sourced facts cannot enter a filed Wiki page (an ungrounded claim → Cannot Confirm → no filing). Conversation context shapes the *question*; it can never inject facts into the *answer*. The grounding firewall + the Phase 6 draft/promote gate already cover the "polluted/interrupted filed data" concern raised at the Phase 9 grill.

### Phase 12 (Alternative Interfaces — CLI + MCP) — scope handoff (pre-grill; from the Phase 10 consumer grill 2026-05-29)

Not yet grilled — a future session will grill this from scratch. This note hands off the context that surfaced while deferring Phase 10, so the Phase 12 grill does not re-derive it.

**Both interfaces are thin adapters over the existing deep modules — they re-implement no logic.** The domain core (`markdown_kb/app/indexer`, `retrieval`, `ingest`, `lint`, `qa`) is already interface-agnostic; today's FastAPI `routes.py` is one shallow adapter (CODING_STANDARD §2.3). CLI and MCP are simply the second and third faces over the same functions. This is exactly what the §2.3 shallow-route discipline buys — no triple-implementation.

- **CLI** (`kb index`, `kb ask`) — for a **human** at a terminal: one-shot process, human-readable output, `--help` is the contract.
  - `kb index` → `indexer.build_index()` (returns `(files, sections)`).
  - `kb ask "Q"` → `retrieval.query(Q)` (returns `{answer, sources, grounding_outcome}`); print answer + citations.
  - Register via `[project.scripts]` in `pyproject.toml`.
- **MCP** (expose index / search / chat as agent tools) — for an **LLM agent** (Claude host): long-lived server; the model reads machine-readable tool schemas and decides *when/how* to call. Tool **descriptions are prompt engineering for the model.**
  - **Tool granularity differs from the CLI**: give the agent `kb_search` (→ `indexer.search(q, k)` — raw Sections, **no LLM**, agent reasons over evidence itself) AND `kb_chat` (→ `retrieval.query(q)` — full grounded answer + citations). Both functions already exist, so the search/chat split is near-free. `kb_index` → `build_index()`.
  - Transport: stdio (Claude Desktop) or HTTP.
- **Precondition to confirm first:** the deep modules must be import-safe (no FastAPI coupling at module import). Looks already true (routes.py imports them from the outside), but verify before building.

**Hot Cache rides on this phase (Phase 10 folded in here).** Once the MCP host exists it is the persistent agent the Hot Cache pattern requires. Shape:
- Expose `wiki/hot.md` as an MCP **resource** (e.g. `kb://hot`) the host reads first each session, plus a write tool (e.g. `kb_save_hot(summary)`) — mirrors `claude-obsidian`'s `/save`.
- **The server only persists bytes; the host agent (which holds the session context) writes the summary.** No server-side LLM summariser, no "when does the session end" trigger on the server — those problems dissolve in the MCP model.
- Input is the **host's MCP session**, NOT `ConversationStore.dump()` (that seam was aimed at the rejected browser-reader framing — see the Phase 10 prep note).
- Make-or-break question to settle in the grill: is KB-local, agent-agnostic working memory worth it vs. the host keeping its own memory? Leaned yes (fits the enterprise-KB handoff/audit positioning), but re-litigate.

### Phase 13 (Hybrid Retrieval over Wiki) — design decisions already locked

Scoped during a 2026-05-28 grill (after Phase 8's comparison shipped). Capture so a future session does not re-litigate:

- **Corpus = wiki Sections, both arms.** BM25-over-wiki already exists (`markdown_kb.indexer.search`). The new piece is **dense-over-wiki**. Build the dense (FAISS) index directly from `markdown_kb.indexer.sections` — that list is already filtered (entities/concepts/qa + qa `status==live`) and slug-id'd. Do **not** point `vector_rag` at `wiki/`: its `_load_documents` uses filename-based ids (no slug), which would break id-alignment.
- **Embed at Section granularity, not char-chunk.** Dense-returned ids must align **1:1** with BM25's slug ids so RRF is true same-corpus fusion. Char-chunking (vector_rag's model) would force a chunk→Section aggregation step (+2-3h, extra design). Section-level keeps fusion trivial.
- **RRF is recall-union (補漏), NOT precision-filter.** RRF rescues docs BM25 missed; it does **not** remove BM25's false positives. "Eliminate BM25's errors" (precision) needs a **cross-encoder reranker AFTER RRF** — deferred. Ship RRF-only first, measure, then decide on the reranker. Cross-encoder rerankers are small BERT-class models (not LLMs, not token-billed) applied to top-20/50 only.
- **Token cost is NOT a disadvantage** (the worry that prompted this scoping). Retrieval is LLM-token-free; at fixed top-k the generation cost is identical to single RAG (RRF changes *which* k docs, not *how many*). Real added cost: ~1.4x index storage (inverted + vector on same corpus) + ~6ms p50 latency (noise vs 500ms-2s LLM) + optional reranker.
- **Everything downstream is reused unchanged.** Because the unit is `Section`, `expand_to_pages` / `build_prompt` / `grounding.verify` (Section satisfies `CitableContent`) / Cannot Confirm gate / `derived_from` all work as-is. The only genuinely new code is the dense index + the ~20-line RRF merge.
- **Slice plan:** H-1 dense-over-wiki index (4-6h) → H-2 RRF + hybrid `query()` (4-6h) → H-3 thin serving surface `/health`/`/index`/`/chat` (3-5h) → ADR + CONTEXT + reviewer (2-3h) → regression + real-embedding smoke (1-2h).
- **Relation to #107:** if the pluggable `Retriever` protocol lands, this hybrid is its 3rd implementation (Wiki / Vector-RAG / Hybrid), togglable alongside the other two — so it strengthens, not conflicts with, #107.

### Phase 14 (README / Getting-Started docs) — grill outline (content TBD)

Goal: a newcomer can go from `git clone` to a running, queryable system using only `README.md` + `.env.example`, no source-reading required. This is the **收尾 (wrap-up)** phase — it documents shipped capability, it does not add features. Run **`grill-with-docs`** to lock the full content; the topics to cover (placeholder outline, not yet grilled):

1. **Install & bootstrap** — clone, Python version, virtualenv, dependency install. Which requirements files / packages (`markdown_kb`, `vector_rag`, `gateway`)? Any system deps?
2. **Environment config (`.env.example`)** — which AI API keys/vars to set and what each is for: `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_INGEST_MODEL`, `OPENAI_VERIFIER_MODEL`, the `vector_rag` embedding model, etc. Required vs optional, and the documented fallback chain (e.g. `OPENAI_INGEST_MODEL` → `OPENAI_MODEL` → hardcoded default).
3. **Running the servers** — how to start the back-end and the front-end (gateway / browser UI). The single-origin gateway (ADR-0010) mounts both stacks; document the `stack=wiki|rag` toggle and the streaming endpoint (`POST /chat/stream`).
4. **Importing data** — how Sources enter the system: `docs/*.md`, `raw/*.txt|*.html` via Multi-Format Import (Phase 7), and how a user points the pipeline at their own corpus.
5. **Generating fake data** — how the `fake-docs/` corpus is generated (LLM prompts to a fictional company), and how a user regenerates or extends it.
6. **API reference** — each endpoint: what it does, how to call it (curl examples), and **when** to use it. `/index` (build/rebuild the Section index), `/ingest` (Source → `wiki/` projection), `/lint` (maintenance pass), `/chat` + `/chat/stream`. Include the typical end-to-end ordering (e.g. index → ingest → re-index → chat; lint periodically).
7. **Wiki-vs-Vector-RAG comparison** — how to run the Paraphrase Comparison (Phase 8 / DeepEval), where the report lands, and how to read it.
8. **Typical usage flows** — narrative walkthroughs that tie the APIs together: "I have a new FAQ doc — what do I do?", "I want to ask a question", "the KB feels stale — how do I maintain it?"

Open questions to settle in the grill: target audience (an interviewer reviewing the repo vs an operator running it)? How much lives in `README.md` vs a split-out `project-docs/USAGE.md` / API reference? Does README coverage gate on phases still in progress (6, 9), or do we document "current state" and explicitly mark unfinished surfaces?

### Phase 15 (Operator Console + Upload) — grill done, PRD pending

Vocabulary reserved in `CONTEXT.md` (the **Operator Console** and **Upload** terms, added during a grill on 2026-05-29). No PRD published yet. Scope from those terms: a curator-facing management surface on the gateway that drives the Wiki lifecycle (Import → Ingest → Index → Lint → Filed-Answer promotion) and exposes RAG index rebuild as a single parallel action (orchestrates each Retrieval Stack's operations but does not merge them — ADR-0002; Wiki-centric because the RAG stack has no Ingest/Lint/Filing, only an index build). Also hosts a resource browser over `docs/` and `wiki/`, and a drag-drop **Upload** drop zone whose staging is distinct from Import (`.html`/`.txt` land in `raw/` as Import candidates; `.md` lands directly in `docs/`, skipping Import — Upload only moves bytes, it never converts). Single local operator, no auth (demo posture shapes scope, not rigor). Run `/to-prd` to lock the full content before implementation.

### Phase 16 (Chinese / language-agnostic corpus) — design locked (grill 2026-05-29, PRD #164)

Scoped via `grill-with-docs`. Capture so implementation does not re-litigate:

- **Language-agnostic, not Chinese-specific.** Unicode tokeniser + Unicode slug; the same change incidentally covers other non-Latin scripts. Product i18n (UI translation, per-locale response strings, locale negotiation, message catalogs, RTL) is explicitly OUT.
- **Tokeniser = CJK bigram + unigram fallback** for length-1 runs. **Hard invariant: pure-ASCII input tokenises byte-identically** (new logic only fires on codepoints > 127) → zero regression on English BM25 / `KB_SCORE_THRESHOLD` / Phase 8 baselines. One mixed Section Index, **no per-language partition** (Chinese-bigram and Latin-word token vocabularies are disjoint). jieba/CKIP dictionary segmentation rejected (Simplified-leaning dict / heavy model / language-specific — conflicts with the language-agnostic decision).
- **W1 linchpin (most easily missed):** `/chat` queries only `wiki/` (ADR-0006), so fixing the tokeniser alone is insufficient. The `/ingest` entity+concept synthesis templates MUST get a "write in the Source's language" directive, or a Chinese Source synthesises to an English Wiki Page that a Chinese query can never lexically match. `/chat` `SYSTEM_PROMPT` gets "answer in the QUESTION's language" with a carve-out: the `Cannot Confirm` phrase stays verbatim English (sentinel contract — tests, `grounding.reason`, filing gate depend on it).
- **Cross-lingual is not a requirement.** English-source + Chinese-query on Wiki/BM25 → `Cannot Confirm` (accepted). On Vector RAG it works for free via multilingual `text-embedding-3-small` (reported as a finding, not built). Retrieval-semantics contrast worth documenting: BM25 segregates languages, Vector RAG blends them — extends the Phase 8 comparison narrative to the multilingual axis.
- **Tests A+B:** hermetic Chinese fixtures in `tests/fixtures/` (bigram unit, ASCII zero-regression, slug, ingest-language, end-to-end Chinese) + a separate Chinese demo Source **not** in `fake-docs/` (protect the Phase 8 baseline); the English `docs/` sample stays untouched. Implementation produces an ADR ("language-agnostic retrieval — CJK bigram + Unicode slug").
- **Already done / out of this PRD:** companion `CONTEXT.md` glossary edit (commit `0227420`) — the Citation term documents Unicode slugs and the non-ASCII slug Flagged-ambiguity is marked resolved. Encoding is already UTF-8-safe across the codebase; Big5/cp950 raw decode is **not** handled (Import already fails loud with `UnicodeDecodeError`, no silent mojibake) — deferred follow-up. Chinese stop-word handling is a deferred follow-up (trigger: function-word noise observed after real Chinese use; with bigram it is pre-bigram char stripping, not an English-style stop set).

## Out of scope (still)

The original PRD § Out of Scope list remains in force. Items not appearing in this roadmap stay deferred behind their original triggers — see [`README.md`](../README.md#stretch-goals) for the trigger table.

## Updating this file

Any phase reorder, new dependency discovery, or stopping-point revision is a doc-level change worth its own commit. Re-read [`inspiration.md`](inspiration.md) for the `phase:` tag of the relevant phase before each edit.
