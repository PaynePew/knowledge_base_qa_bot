# Framework integration: borrow components, keep opinions

We adopt LangChain (`langchain-core`, `langchain-openai`) at the LLM-call wrapper layer only. The rest of the stack — Markdown parsing, BM25 scoring, prompt building, `.kb/index.json` persistence, FastAPI routing — stays hand-written. Future framework swap-ins are pre-decided per component and gated on explicit triggers (corpus size, hybrid retrieval activation, support for non-Markdown sources), not opportunistic refactors. LlamaIndex is not currently a dependency; specific components (`MarkdownNodeParser`, `CitationQueryEngine`, LlamaHub loaders) are reviewable additions when their triggers fire.

We chose partial borrowing because the project's value is in its *opinions* (the strict grounded contract per ADR-0001, the Section retrieval unit with the body-bearing-intermediate rule, the inspectable `.kb/index.json`, the `filename#heading-slug` citation format), and these opinions resist clean expression inside framework abstractions. `MarkdownHeaderTextSplitter` does not encode our Section rules; `LangServe` mismatches the `/chat` / `/index` / `/health` endpoint contract `PROMPT.md` requires; `ChatPromptTemplate` adds a layer of indirection where prompt iteration needs to be one file away. Conversely, `ChatOpenAI` solves real production-grade concerns (retry, streaming, structured output, token counting) without imposing on our design choices — borrowing it is pure subtraction of effort. The middle ground — borrow at the leaf nodes where framework adds value, hand-write at the joints where opinion lives — produces the leanest codebase that still demonstrates judgment, and the leanest is the one easiest to grill on in an interview.

## Considered Options

- **Full framework adoption (LangChain `RetrievalQA` or LlamaIndex `CitationQueryEngine`).** Rejected: hides the strict grounded contract behind chain abstractions, makes prompt iteration cross-file, and would require `LangServe` to expose the API — losing control over the `/chat`, `/index`, `/health` shape `PROMPT.md` requires. Also couples our upgrade cadence to the framework's (LangChain 1.x packaging restructure in 2025 was a meaningful migration).
- **Zero framework, full hand-roll.** Rejected: re-implementing `ChatOpenAI`'s retry / streaming / structured-output handling is real work with no design payoff. The wrapper layer is a solved problem; reinventing it dilutes attention from the parts that matter. The Q5 fail-mode design (bounded retry on verifier `timeout` / 5xx) is also vastly easier to express through `ChatOpenAI`'s built-in retry config than from scratch.
- **Opportunistic mixing — borrow whatever feels easy at the moment.** Rejected: produces an incoherent dependency surface, with framework lock-in creeping in component by component. Pre-committing per-component swap triggers makes the integration strategy reviewable as one decision, not as drift.

## Consequences

- `langchain-core` and `langchain-openai` are first-class dependencies (currently pinned at 1.4.0 / 1.2.2 in `markdown_kb/pyproject.toml`). Upgrades are treated as real changes — assume yearly churn of LangChain's packaging surface.
- Specific future triggers (these are the only framework additions we pre-bless; anything else needs a new ADR or amendment):
  - **Adopt `rank_bm25`** when `docs/` corpus exceeds ~1,000 Sections and `bm25_score()` shows up in profiling. `BM25Retriever` (LangChain) wraps `rank_bm25` and is the easy swap path. Until then, hand-written BM25 in `indexer.py` stays.
  - **Adopt `EnsembleRetriever` (LangChain) or `QueryFusionRetriever` (LlamaIndex)** when `vector_rag/` is activated and the hybrid retrieval layer is wanted. Matches the `inspiration.md` "Reciprocal Rank Fusion / cross-encoder rerank — phase: query" deferred pattern.
  - **Adopt format-appropriate converter at the leaf-node layer when supporting non-Markdown sources.** The principle is unchanged from line 5 of this ADR (`borrow at leaf nodes where framework adds value, hand-write at joints where opinion lives`). Trigger refinement: `non-Markdown source` alone is too coarse — the right framework depends on whether the framework actually produces Markdown, not just on whether the input is non-Markdown. Per-format mapping:
    - **HTML → Markdown**: `markdownify` (single-purpose conversion lib; LlamaHub loaders output stripped text, not markdown — not applicable here). Applied in Phase 7 (PRD #89).
    - **PDF → Markdown**: LlamaHub `PDFReader` or `pypdf` (when triggered).
    - **Notion export → Markdown**: LlamaHub `NotionPageReader` (when triggered).
    - **Generic binary / structured formats**: LlamaHub readers or Unstructured (when triggered).
    Markdown itself stays on the hand-written parser.
  - **Adopt `ChatOpenAI.with_structured_output(GroundingResult)`** when implementing Grounding Check (see ADR-0004 once written). Wraps the verifier call's JSON parsing + retry-on-malformed, which directly serves the Q5 fail-mode design.
- Permanently hand-written (these are contractual, not implementation laziness):
  - **Markdown parsing** (`indexer.py`) — encodes Section rules that no framework matches (body-bearing intermediate, slug collision suffix, heading-only leaves, `heading_path` breadcrumb).
  - **Persistence format** (`.kb/index.json`) — `PROMPT.md` contract requires plain-text inspectability; any framework persistence (pickle, SQLite, parquet) loses this.
  - **Prompt builder** (`prompt_builder.py`) — opinion lives here; abstraction would slow iteration. ADR-0001's strict contract is expressed in the `SYSTEM_PROMPT` literal, which must stay one `grep` away.
  - **FastAPI routes** (`routes.py`) — avoids `LangServe` lock-in, preserves API shape control, and keeps middleware (auth, rate limiting, CORS) on standard FastAPI ground.
- The integration story is interview-ready: the answer to "why not use LangChain?" is no longer "I haven't" but "I do, for `ChatOpenAI`; here are the four components I will swap in next, here are the four I will never swap, and here is why each."

### LLM-facing surface enumeration (CODING_STANDARD §2.4 / §6.4)

**Invariant** — The modules below own an LLM call site and are therefore the only modules permitted to import `langchain` / `langchain_openai` / `langchain_community` and to see LangChain message/client/structured-output types (CODING_STANDARD §2.4). Each LLM-facing *surface* (a user-reachable operation backed by such a call) gets exactly one `@pytest.mark.live` smoke test (§6.4); a second live test on the same surface, or a live test on a new surface absent here, is scope creep.

| Module | LLM call site | Surface(s) | Live smoke test |
|---|---|---|---|
| `markdown_kb/app/retrieval.py` | `ChatOpenAI.invoke` (answer synthesis) | `markdown_kb` `POST /chat` | `markdown_kb/tests/test_chat_live.py` |
| `markdown_kb/app/grounding.py` | `ChatOpenAI.with_structured_output` (verifier) | grounding check (shared; consumed by both apps) | covered via the `/chat` live tests |
| `markdown_kb/app/templates.py` | `ChatOpenAI` (classify + synthesise) | `markdown_kb` `POST /ingest` | `markdown_kb/tests/ingest/` live test |
| `markdown_kb/app/lint.py` | `ChatOpenAI` (contradiction / staleness checks) | `markdown_kb` `POST /lint` | `markdown_kb/tests/lint/` live test |
| `vector_rag/app/indexer.py` | `OpenAIEmbeddings` via `FAISS.from_documents` (embedding) | `vector_rag` `POST /index` (embedding) | shared with the `vector_rag` `/chat` live test (one live call exercises index→chat) |
| `vector_rag/app/retrieval.py` | `ChatOpenAI.invoke` (answer synthesis) | `vector_rag` `POST /chat` | `vector_rag/tests/test_chat_live.py` |

vector_rag's `/chat` + embeddings is the surface added by issue #103 (Phase 8 Slice 3). It adopts `markdown_kb`'s `grounding.py` unchanged through the `CitableContent` Protocol (ADR-0004 Q9) rather than owning a second verifier, so it adds no new structured-output call site. The single PRD-authorised `@pytest.mark.live` test for this surface lives in `vector_rag/tests/test_chat_live.py`.
