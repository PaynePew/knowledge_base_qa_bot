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

## Roadmap (Phase 3-11)

| Phase | Scope | Est | Axis | Status |
|---|---|---|---|---|
| **3** | `/ingest` — Source → `wiki/entities/`, `wiki/concepts/`. LLM-assisted synthesis. Per-type Source Templates. Frontmatter schema becomes real (no longer deferred). | 12-19h | Karpathy write | Planned |
| **4** | W2 layered retrieval — `/chat` queries `wiki/` + `docs/`. Flips `SOURCE_DIRS = [DOCS_DIR]` to `[DOCS_DIR, WIKI_DIR]` (the upgrade hook ADR-0003 already pre-installed). | 3-6h | Karpathy read | Planned |
| **5** ⭐ | `/lint` — Lint Pass. Contradiction detection, stale claims, orphan pages, gaps from repeated cannot-confirm queries. Completes Karpathy's Ingest + Query + Lint trio. | 6-10h | Karpathy maintain | Planned |
| **6** | Answer Filing — `/chat` answers (when high-confidence) get written back to `wiki/qa/*.md`. Closes the Two-output rule on the query side (the Source side closes in Phase 3). PROMPT.md stretch #7. | 4-8h | Karpathy Two-output | Planned |
| **7** | Multi-Format Import — `raw/*.txt` and `raw/*.html` normalized into `docs/*.md` via a pre-processing step (`markdownify` or equivalent), so non-Markdown Sources can enter the pipeline without losing the canonical Markdown format guarantee. PROMPT.md stretch #4. | 3-5h | input flexibility | Planned |
| **8** | Paraphrase Comparison — revive `vector_rag/` (langchain 0.3 → 1.x), run paraphrased queries against both retrieval strategies, produce a comparison report. Empirically validates ADR-0002. PROMPT.md stretch #9. | 6-12h | retrieval comparison | Planned |
| **9** | Streaming + Browser UI — `POST /chat/stream` (SSE) + a minimal HTML page rendering sources first, then streamed answer. PROMPT.md stretches #2 + #3. | 4-8h | product surface | Planned |
| **10** | Hot Cache — `wiki/hot.md` (~500 words), rewritten at end of meaningful sessions. First file the agent reads. | 2-4h | convenience | Planned |
| **11** | Conversation Memory — Query Rewriting + Conversation Store. Multi-turn `/chat`. PROMPT.md stretch #8. | 6-10h | multi-turn | Planned |
| **12** | Alternative Interfaces — CLI (`kb index`, `kb ask`) + MCP server. PROMPT.md stretch #5. | 4-8h | packaging | Planned |

**Total: 48-87h** (excluding the ⭐ recommended stopping point at Phase 5, which clocks in at 17-31h after Phase 3).

## Hard dependencies

These are not preferences — they are constraints. Violating them produces broken demos.

| Constraint | Why |
|---|---|
| Phase 4 **must** follow Phase 3 | `/ingest` writes Wiki pages. Without W2 retrieval, `/chat` never reads them. Demo collapses: "the LLM wrote a page but the bot still queries `docs/`." |
| Phase 6 **must** follow Phase 4 | Answer Filing writes to `wiki/qa/`. Same logic — dead content unless `/chat` reads `wiki/`. |
| Phase 10 **must** follow Phase 4 | Hot Cache lives at `wiki/hot.md`. Same logic. |
| Phase 8 **must** follow Phase 4 | A fair Paraphrase Comparison compares "BM25 over docs + wiki" against Vector RAG, not partial. Running it before Phase 4 produces misleading data. |
| Phase 11 **best** after Phase 9 | Multi-turn UX needs a streaming UI to demo well. Without it, conversation memory is two endpoints, not a conversation. |

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

1. **Don't adopt Ragas / DeepEval / LlamaIndex DatasetGenerator as frameworks.** ADR-0005's "borrow components, keep opinions" rule applies — a ~50-line OpenAI script for paraphrase generation + a `pytest` runner is enough at this scale. Re-evaluate framework adoption when the project has > 3 different metrics to compute.
2. **Grow the corpus first.** 3 FAQ Sources is too small for statistically meaningful comparison. Target 15-20 fake Sources generated via LLM prompts to a fictional company (e.g., "Acme Shop"). Commit to `fake-docs/` as eval-only corpus.
3. **LLM-generated paraphrases have systematic bias favoring Vector RAG.** Both are LLM-trained, so synonyms generated by GPT-4o-mini fall inside the embedding space that the same model family encodes. Mix in ~10% hand-written adversarial samples (typos, cross-lingual queries, dialect terms, industry jargon) to correct. Call out this limitation proactively in the report — it's an interview plus, not a minus.

### Phase 10 (Hot Cache) — defer if single-shot

Hot Cache solves the "where were we?" problem between sessions. If the project stays in single-shot demo usage, Hot Cache has no observable value. Implement only if multi-session use becomes real, or skip and reuse the slot for a different stretch goal.

## Out of scope (still)

The original PRD § Out of Scope list remains in force. Items not appearing in this roadmap stay deferred behind their original triggers — see [`README.md`](../README.md#stretch-goals) for the trigger table.

## Updating this file

Any phase reorder, new dependency discovery, or stopping-point revision is a doc-level change worth its own commit. Re-read [`inspiration.md`](inspiration.md) for the `phase:` tag of the relevant phase before each edit.
