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

## Roadmap (Phase 3-14)

| Phase | Scope | Est | Axis | Status |
|---|---|---|---|---|
| **3** | `/ingest` — Source → `wiki/entities/`, `wiki/concepts/`. LLM-assisted synthesis. Per-type Source Templates. Frontmatter schema becomes real (no longer deferred). | 12-19h | Karpathy write | ✅ Done |
| **4** | W2 layered retrieval — `/chat` queries `wiki/` + `docs/`. Flips `SOURCE_DIRS = [DOCS_DIR]` to `[DOCS_DIR, WIKI_DIR]` (the upgrade hook ADR-0003 already pre-installed). | 3-6h | Karpathy read | ✅ Done |
| **5** ⭐ | `/lint` — Lint Pass. Contradiction detection, stale claims, orphan pages, gaps from repeated cannot-confirm queries. Completes Karpathy's Ingest + Query + Lint trio. **Hard deps: (a) slice 7-3 (#92, merged) writes `content_sha256` to `docs/` frontmatter — required for "raw drifted but docs not re-imported" staleness detection without recomputing hashes online; (b) Phase 3 amendment (issue #93, merged) writes `source_hashes` to wiki frontmatter — required for full-chain raw→wiki drift detection. NOTE: Phase 5 lint must treat empty `source_hashes` as "unknown drift" (not "no drift") to avoid false-negative reports on Phase 6 legacy pages.** | 6-10h | Karpathy maintain | ✅ Done |
| **6** | Answer Filing — `/chat` answers (when high-confidence) get written back to `wiki/qa/*.md`. Closes the Two-output rule on the query side (the Source side closes in Phase 3). PROMPT.md stretch #7. | 4-8h | Karpathy Two-output | ✅ Done |
| **7** | Multi-Format Import — `raw/*.txt` and `raw/*.html` normalized into `docs/*.md` via a pre-processing step (`markdownify` or equivalent), so non-Markdown Sources can enter the pipeline without losing the canonical Markdown format guarantee. PROMPT.md stretch #4. Includes hash chain follow-up: Phase 3 amendment (issue #93) wrote `source_hashes` to wiki frontmatter using `content_sha256` written by Phase 7-3 (PR #92). | 3-5h | input flexibility | ✅ Done |
| **8** | Paraphrase Comparison — revive `vector_rag/` (langchain 0.3 → 1.x), run paraphrased queries against both retrieval strategies, produce a comparison report. Empirically validates ADR-0002. PROMPT.md stretch #9. | 6-12h | retrieval comparison | ✅ Done |
| **9** | Streaming + Browser UI — `POST /chat/stream` (SSE) + a minimal HTML page rendering sources first, then streamed answer. PROMPT.md stretches #2 + #3. All 6 slices merged (#117/#119/#120/#121/#118/#122): Gateway introduced (`gateway/` package, ADR-0010), Wiki + RAG SSE streams, filing parity, chosen UI wired to real SSE stream. | 4-8h | product surface | ✅ Done |
| **10** | Hot Cache — `wiki/hot.md` (~500 words), rewritten at end of meaningful sessions. First file the agent reads. | 2-4h | convenience | Planned |
| **11** | Conversation Memory — Query Rewriting + Conversation Store. Multi-turn `/chat`. PROMPT.md stretch #8. | 6-10h | multi-turn | Planned |
| **12** | Alternative Interfaces — CLI (`kb index`, `kb ask`) + MCP server. PROMPT.md stretch #5. | 4-8h | packaging | Planned |
| **13** | Hybrid Retrieval over Wiki — BM25 + dense vector **both over the curated `wiki/` Section corpus**, fused via Reciprocal Rank Fusion (RRF). Served as a **third `Retriever`** alongside Stack A (Wiki) and Stack B (Vector RAG). Additive — does **not** touch the two existing apps. Optional cross-encoder reranker deferred to a follow-on. Relates to #107 (becomes a 3rd implementation under the pluggable `Retriever` protocol if that lands). | 14-22h (RRF-only; +4-6h with reranker) | retrieval quality (new axis) | Future / want-to-do |
| **14** | **Getting-Started documentation** — rewrite `README.md` (+ `.env.example`) so a newcomer can go `git clone` → install → configure → run without reading source. Covers: clone/install steps, `.env.example` AI API keys (which to set, required vs optional), starting the back-end + front-end (gateway/browser UI) servers, importing Sources + generating fake data, an API reference for `/index` / `/ingest` / `/lint` / `/chat` (what each does, **how & when** to call it), and how to run the Wiki-vs-Vector-RAG comparison. **Wrap-up phase** — documents shipped capability, adds no features; full coverage implies Phases 3/4/5/8/9 have landed. Full content TBD via `grill-with-docs`. | TBD (pre-grill) | documentation / onboarding | Planned — grill pending |

**Total: 48-87h** for Phases 3-12 (excluding the ⭐ recommended stopping point at Phase 5, which clocks in at 17-31h after Phase 3). **Phase 13 sits on a separate "retrieval quality" axis — it is off the linear Karpathy narrative and is an optional add-on, not on the critical path.** **Phase 14 is a documentation / onboarding wrap-up — also off the feature axis; it tracks and documents whatever has shipped rather than adding capability.**

## Hard dependencies

These are not preferences — they are constraints. Violating them produces broken demos.

| Constraint | Why |
|---|---|
| Phase 4 **must** follow Phase 3 | `/ingest` writes Wiki pages. Without W2 retrieval, `/chat` never reads them. Demo collapses: "the LLM wrote a page but the bot still queries `docs/`." |
| Phase 6 **must** follow Phase 4 | Answer Filing writes to `wiki/qa/`. Same logic — dead content unless `/chat` reads `wiki/`. |
| Phase 10 **must** follow Phase 4 | Hot Cache lives at `wiki/hot.md`. Same logic. |
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

### Phase 10 (Hot Cache) — defer if single-shot

Hot Cache solves the "where were we?" problem between sessions. If the project stays in single-shot demo usage, Hot Cache has no observable value. Implement only if multi-session use becomes real, or skip and reuse the slot for a different stretch goal.

### Phase 11 (Conversation Memory) — filing × toggle interaction (locked at Phase 9 grill, 2026-05-28)

Conversation Memory lives in the **gateway**, not the sub-apps (Phase 9 decision: sub-apps stay stateless single-turn; Query Rewriting + Conversation Store are gateway concerns). Two constraints the Phase 11 PRD **must** honour:

1. **File the rewritten self-contained query, not the raw elliptical follow-up.** When a multi-turn answer is filed (Phase 6 Answer Filing, Wiki stack only), `frontmatter.question` must be the Query-Rewriting output (self-contained), so `wiki/qa/*.md` stays independently readable. Filing the raw follow-up ("and how long does *that* take?") would write an incoherent standalone page. Query Rewriting's output is self-contained by construction, so this is free — just file the rewritten form.
2. **Cross-stack fact leakage is already firewalled by the Grounding Check — do NOT add a redundant per-stack conversation partition.** Even if turn N-1 was answered by Vector RAG and turn N toggles to Wiki, the Wiki answer's claims must pass grounding against `wiki/` Sections, so RAG-sourced facts cannot enter a filed Wiki page (an ungrounded claim → Cannot Confirm → no filing). Conversation context shapes the *question*; it can never inject facts into the *answer*. The grounding firewall + the Phase 6 draft/promote gate already cover the "polluted/interrupted filed data" concern raised at the Phase 9 grill.

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

## Out of scope (still)

The original PRD § Out of Scope list remains in force. Items not appearing in this roadmap stay deferred behind their original triggers — see [`README.md`](../README.md#stretch-goals) for the trigger table.

## Updating this file

Any phase reorder, new dependency discovery, or stopping-point revision is a doc-level change worth its own commit. Re-read [`inspiration.md`](inspiration.md) for the `phase:` tag of the relevant phase before each edit.
