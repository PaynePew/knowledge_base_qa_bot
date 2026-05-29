# Markdown KB Q&A Bot — Prototype

This PRD is the synthesis of a structured `/grill-with-docs` alignment session covering scope, vocabulary, architecture, test philosophy, error handling, and parsing rules. It draws on `CONTEXT.md` (project glossary), three ADRs in `project-docs/adr/`, and the deferred-patterns register in `project-docs/inspiration.md`. Use those documents for the canonical definitions of every term used below.

## Problem Statement

I want a grounded Q&A bot over a Markdown knowledge base that I can ship as a Friday prototype, **and** that has a real upgrade path to enterprise KB management (FAQ / policy / customer-support knowledge with a curated synthesis layer above the immutable Sources). The teaching exercise in `PROMPT.md` is the immediate contract; the long-term destination is a layered KB modelled on the patterns in [`AgriciDaniel/claude-obsidian`](https://github.com/AgriciDaniel/claude-obsidian) (5.4K⭐, the most-starred reference implementation of Karpathy's LLM Wiki gist). I do not want to choose between "ship the prototype" and "build a real maintainable project" — I want a foundation that does both.

A naïve scaffold completion would lock me out of the Karpathy direction (silent body-loss in `parse_markdown`, no log, no future-proof vocabulary, no architectural records). I have a hard deadline of next Friday and limited focused-work hours.

## Solution

Implement the **Markdown KB** retrieval strategy (Strategy A in `PROMPT.md`) as a working prototype that passes every verification case, **while** investing the small additional discipline ("b-disciplined" per ADR notes) that keeps the future Wiki layer addable without renaming or restructuring:

- BM25 over a heading-anchored **Section Index** persisted at `.kb/index.json`, with a body-bearing parsing rule so personal-notes content is not silently dropped.
- **Strict-grounded answers** (ADR-0001): retrieval gates a Cannot Confirm fallback *before* the LLM is called when scores are below `KB_SCORE_THRESHOLD`; inside the prompt, the LLM is told to cite every claim inline as `[Source: filename#heading]`.
- A **Wiki Log** (`wiki/log.md`) appended on every `/index`, `/chat`, fallback, and error — Karpathy's log discipline starting from day one rather than after the wiki layer ships.
- File layout already root-level (`markdown_kb/`, `vector_rag/`), so the `scaffold/` framing no longer implies throwaway code. The `vector_rag/` scaffold is preserved unimplemented for post-prototype comparison work (ADR-0002).
- Reserved vocabulary in `CONTEXT.md` aligned with the `claude-obsidian` reference implementation, so when the wiki layer is added, nothing has to be renamed.

The architecture target after prototype is **W2 layered** (ADR-0003): `docs/` remains immutable Sources, a future `wiki/` directory becomes the LLM-maintained synthesis layer, both surfaces queryable.

## User Stories

1. As a curator, I want to drop a Markdown file into `docs/` and trigger `POST /index`, so the bot indexes the file without any manual preprocessing.
2. As a user, I want to ask a question via `POST /chat`, so I receive a grounded answer assembled from my own Sources.
3. As a user, I want every factual claim in an answer cited inline as `[Source: filename#heading]`, so I can verify the answer against the original Markdown.
4. As a user, I want the bot to reply with the exact phrase `"I cannot confirm from the knowledge base."` when my Sources do not cover a question, so it never invents a confident wrong answer.
5. As a curator, I want out-of-scope questions like `"Which restaurants are nearby?"` to trigger Cannot Confirm, so I trust the system's stated boundaries.
6. As a developer, I want `.kb/index.json` written as inspectable, pretty-printed JSON, so I can `cat` it and debug retrieval issues.
7. As a developer, I want the server to load `.kb/index.json` on startup and continue serving queries without a rebuild, so I am not re-indexing on every restart.
8. As a developer, I want OpenAI timeout or rate-limit errors surfaced as `HTTP 503`, so monitoring and clients can retry intelligently.
9. As a developer, I want OpenAI auth errors surfaced as `HTTP 500`, so I am alerted to a configuration problem rather than a transient hiccup.
10. As a curator, I want a chronological `wiki/log.md` appended on every `/index`, `/chat`, fallback, and error, so I can `grep` "what happened, when" without running the server.
11. As a developer, I want `parse_markdown` to apply the body-bearing rule (heading becomes a Section if it is a leaf *or* if it carries non-whitespace body before its first child heading), so an h1 intro paragraph in personal notes is not silently dropped.
12. As a developer, I want fenced code blocks in Markdown to be inert — a `# comment` inside a fenced block is content, not a heading — so technical notes are sectioned correctly.
13. As a developer, I want YAML frontmatter stripped from Sources and stored as a `metadata: dict` on each Section, so the future Wiki layer with frontmatter schemas does not need a parser change.
14. As a user, I want the top 3 ranked Sections forwarded to the LLM, so cross-Section synthesis is possible without overflowing the context window.
15. As a user, I want the LLM's prompt to carry `heading_path` as a breadcrumb (`Refund Policy > Refund Timeline`), so the LLM has document-structural context beyond the raw text.
16. As a developer, I want concurrent `POST /index` calls serialized through an in-process lock, so the persisted index never corrupts under contention.
17. As a developer, I want `.kb/index.json` written atomically (tmp file → rename), so a crash mid-write does not leave a half-written index for the next startup.
18. As a developer, I want a `KB_SCORE_THRESHOLD` env var with a `0.5` default, so I can tune the Cannot Confirm gate without redeploying code.
19. As a developer, I want one `@pytest.mark.live` smoke test exercising a real OpenAI call, so before pushing I can confirm the model actually follows the `SYSTEM_PROMPT`.
20. As a developer, I want all other tests to use a fake LLM that returns canned grounded answers, so CI stays free, deterministic, and offline.
21. As a developer, I want integration tests built on FastAPI's `TestClient` that translate the four `PROMPT.md` verification cases one-to-one, so "all green" means "the deliverable is met."
22. As a future maintainer, I want the `parse_markdown` 10-rule spec embedded in the function's docstring, so when I extend the parser for nested h3/h4 headings I know exactly which rules to keep invariant.
23. As a future maintainer, I want `wiki/log.md` committed to git rather than ignored, so the operation history is preserved across machine moves and acts as a secondary backup. **[Superseded: the `wiki/` artifact taxonomy (`wiki/README.md`, commit `d00d9e3`) later reclassified `wiki/log.md` as a runtime-trace artifact and gitignored it; per-app log channels (`wiki/log.md`, `vector_rag/log.md`, `gateway/log.md`) are gitignored runtime traces per CODING_STANDARD §5.1.]**
24. As a future maintainer, I want a `metadata: dict` field on the `Section` dataclass even though it is unused in the prototype, so frontmatter ingestion in the Wiki phase does not require a dataclass migration.
25. As a future maintainer, I want the vocabulary used in code variable names (`Section`, `Section Index`, `Citation`) to match `CONTEXT.md`, so a new contributor reading the glossary recognises the symbols in the codebase immediately.
26. As an interviewer reviewing the project, I want ADRs in `project-docs/adr/` that explain the strict-grounded discipline, the dual-app layout, and the W2 layered Wiki target, so I can assess engineering judgment without re-deriving every trade-off from code.
27. As the agent running this work, I want a triage label `ready-for-agent` to signal that the PRD is fully specified and can be picked up without further clarification from the human.

## Implementation Decisions

### Architecture
- **Two retrieval apps** live at the repository root: `markdown_kb/` (the active prototype) and `vector_rag/` (preserved scaffold for post-prototype comparison). See **ADR-0002**.
- **W2 layered Wiki** is the post-prototype target (ADR-0003); the prototype's `build_index()` is designed today with `SOURCE_DIRS = [DOCS_DIR]` as a *list*, so the future `WIKI_DIR` can be appended without changing the signature.

### Modules in `markdown_kb/app/`
- `indexer.py` — **deep module**. Public surface: `build_index()`, `search(query, k)`, `load_index_json()`, `Section` dataclass, in-memory `sections` list. Encapsulates `parse_markdown` (10-rule body-bearing spec), `slugify`, `tokenize`, BM25 math, in-process lock, atomic `.kb/index.json` write, frontmatter stripping, fenced-code-block tracking.
- `retrieval.py` — **deep module**. Public surface: `query(question: str) -> dict`. Encapsulates threshold gating against `KB_SCORE_THRESHOLD`, the LLM call, the three OpenAI error classes mapped to HTTP 503 / 500 / passthrough, a light grounding check (response must contain `[Source:` or the exact Cannot Confirm phrase, retry once on failure, fall back to Cannot Confirm on second failure), and integration with `log_event`.
- `prompt_builder.py` — **new, extracted**. Public surface: `build_prompt(question, ranked_sections) -> str`. Owns `SYSTEM_PROMPT` (the rule-only strict-grounded prompt) and `build_prompt` (`CONTEXT:` before `QUESTION:`, `[Source: filename#heading]` header per Section, `Heading: parent > leaf` line per Section, top-3 Sections inline). Extracted from `retrieval.py` so its output can be asserted in isolation without mocking the LLM.
- `logger.py` — **new, small**. Public surface: `log_event(kind: str, summary: str)`. Writes one line `## [<ISO-8601 UTC>] <kind> | <summary>\n` to `wiki/log.md`, creating the directory if needed. Atomic single-line append; concurrency relies on POSIX `open("a")` semantics, acceptable for prototype QPS.
- `routes.py`, `schemas.py`, `main.py` — shallow HTTP / data / app-lifecycle layers, kept as-is from the scaffold.

### Section model (body-bearing rule)
A heading becomes a Section when **either** (i) it has no child headings (a *leaf*), **or** (ii) it carries non-whitespace body content directly between itself and its first child heading (a *body-bearing intermediate*). The Section's content is the body it owns directly — never the recursive body of its children, which are their own Sections. Empty-body leaves are still Sections (heading-only); their `tokens` come from the heading text alone.

The full `parse_markdown` contract is the 10-rule docstring already committed to `markdown_kb/app/indexer.py`.

### `Section` dataclass shape
```
@dataclass
class Section:
    id: str            # "{filename}#{heading-slug}", or bare filename when no headings
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]
    metadata: dict     # YAML frontmatter; empty by default in prototype, populated for future Wiki use
```
(Encodes the b-disciplined commitment from ADR-0003: `metadata` reserved on day one so the Wiki layer with frontmatter schemas does not require a dataclass migration.)

### Citation format
`{source-filename}#{heading-slug}`, e.g. `refund_policy.md#refund-timeline`. The slug is the lowercased, hyphen-collapsed heading text (matches GitHub / Obsidian anchor conventions so the reference is clickable). Within a single Source, slug collisions append `-2`, `-3`, … suffixes; a Section is never silently overwritten. `heading_path` is communicated separately in the prompt's `Heading:` line, not embedded in the Citation.

### Cannot Confirm contract (ADR-0001)
The literal phrase `"I cannot confirm from the knowledge base."` is returned whenever:
- the Section Index yields no Sections at all,
- the top-scored Section is below `KB_SCORE_THRESHOLD` (env-configurable, default `0.5`),
- or the LLM fails the light grounding check twice in a row.

In the first two cases the LLM is **not** called — the fallback is gated *before* the LLM, so the model is never tempted to confabulate around weak context.

### `SYSTEM_PROMPT` rules
Plain markdown (not XML tags — the prototype targets `gpt-4o-mini`):
1. Only use the CONTEXT; no outside knowledge.
2. Inline `[Source: filename#heading]` after every factual claim; multi-source per claim is allowed.
3. Synthesis across cited Sections is allowed; inference beyond what is written is not.
4. The exact-phrase Cannot Confirm fallback, with explicit "do not paraphrase, do not apologise, do not explain what is missing."
5. For partial-match situations, quote what the Section *does* say, cite it, and add "The knowledge base does not specify {missing part}."

### Index lifecycle
- **Rebuild policy**: every `POST /index` is a full rebuild (current sample-scale rebuild < 0.1 s; incremental indexing is post-prototype). Section IDs are *not* stable across heading or filename renames (flagged in `CONTEXT.md`; addressed when the Wiki layer ships).
- **Concurrency**: in-process `threading.Lock` around the index swap.
- **Persistence**: atomic write — `index.json.tmp` → `os.replace(..., index.json)`. Reads do not need the lock; mid-rebuild readers see the previous in-memory snapshot until the swap completes.
- **Startup**: `load_index_json()` on FastAPI startup; if the file is missing, the server still starts and `/chat` returns the "knowledge base has not been indexed yet" response.
- **Failure mode**: corrupt `.kb/index.json` at startup → `raise` so the server fails to start (fail-fast — better than silently serving stale or wrong data).

### Error handling (HTTP status mapping)
- `APITimeoutError`, `RateLimitError` → `HTTPException(503)` with `detail="LLM service temporarily unavailable, please retry."` and a `chat_error | kind=openai_transient` log event.
- `AuthenticationError` → `HTTPException(500)` with `detail="LLM service auth failed (check OPENAI_API_KEY)."` and a `chat_error | kind=openai_auth` log event.
- Other `APIError` subclasses → `HTTPException(500)` and a `chat_error | kind=openai_api` log event.
- LangChain `max_retries=1` retained from the scaffold (a single automatic retry; not extending to aggressive backoff for the prototype).

### Log entry conventions
Every entry begins `## [<ISO-8601-UTC>] <kind> | <summary>` for `grep`-ability. Kinds emitted in this prototype:
- `index_built | files=N sections=M`
- `chat | "<first 60 chars of query>" top=<section-id>:<score>`
- `chat_fallback | "<query>" reason=below_threshold top_score=<score>`
- `chat_fallback | "<query>" reason=not_indexed`
- `chat_error | "<query>" kind=<openai_transient|openai_auth|openai_api> [detail=...]`
- `parse_warning | <description>` (e.g. non-leaf heading with whitespace-only body, slug collision suffix added)

## Testing Decisions

### Test philosophy (already documented in `markdown_kb/tests/README.md`)
A good test asserts external behaviour. It exercises a module through its public interface, not its implementation. For this prototype the **PROMPT.md verification cases are the contract**; integration tests at the FastAPI `TestClient` level translate that contract one-to-one into executable assertions, so "all green" means "the deliverable is met."

The pyramid is **intentionally inverted** compared to a strict-TDD codebase: thick at integration, thin at unit. Unit tests on trivia like `slugify` add no signal beyond what component tests already cover transitively. BM25 score absolute values are *not* asserted — only ranking order is, because the absolute numerics are sensitive to corpus parameters and would create brittle tests.

### Modules under test
- **`indexer.py`** — component tests:
  - `parse_markdown(fixture.md)` returns the exact expected Sections (id, heading, heading_path, content).
  - `build_index()` over the sample `docs/` produces `(files_indexed=3, sections_indexed=9)` and a `.kb/index.json` whose round-trip via `load_index_json()` reproduces the same `sections` list.
  - `search("how long do refunds take")` ranks `refund_policy.md#refund-timeline` first; `search("which restaurants are nearby")` returns either an empty list or all scores below `KB_SCORE_THRESHOLD`.
  - `parse_markdown` body-bearing rule: a fixture with `# H1\nIntro.\n## Child\nDetail.\n` produces *two* Sections (the h1 because of its intro paragraph, and the leaf h2).
  - Fenced-code-block fixture: `# heading\ntext\n\`\`\`\n# bash comment\n\`\`\`\n` produces one Section, not two.
  - Slug collision: two `## Overview` in the same Source produce `…#overview` and `…#overview-2`.
- **`retrieval.py`** — integration tests via `TestClient` (LLM mocked):
  - The four `PROMPT.md` verification cases (`/health`, `/chat` before `/index`, `/index`, `/chat` after `/index`).
  - The grounded query case asserts the response includes the expected Citation (`refund_policy.md#refund-timeline`) and the `sources` array contains the matching Section.
  - The out-of-scope query case asserts the exact `"I cannot confirm from the knowledge base."` response and an empty `sources` array, *and* that the mock LLM was never called.
  - The transient-error case (mock LLM raises `APITimeoutError`) asserts `HTTP 503` and a `chat_error | kind=openai_transient` log entry.
  - The auth-error case (mock LLM raises `AuthenticationError`) asserts `HTTP 500` and a `chat_error | kind=openai_auth` log entry.
  - Restart-persistence: after `/index`, recreate the FastAPI app and assert `/chat` works without re-indexing.
- **`prompt_builder.py`** — unit tests:
  - `build_prompt` output contains `CONTEXT:` before `QUESTION:`.
  - Each ranked Section appears under its `[Source: filename#heading]` header.
  - `heading_path` is rendered as `Heading: parent > leaf`.
  - Cited identifiers in the rendered prompt match the IDs in `ranked_sections`.
- **`logger.py`** — unit tests:
  - `log_event("chat", "hello")` produces a line matching `^## \[\d{4}-\d{2}-\d{2}T.*Z\] chat \| hello\n$` and appends to (rather than overwrites) the file.
  - Multiple `log_event` calls preserve insertion order.
  - Auto-creates `wiki/` directory if absent.
- **Live smoke** — one `@pytest.mark.live` test:
  - Real OpenAI call against a grounded query.
  - Asserts only response shape: `200`, `[Source:` present in `answer`, non-empty `sources` array. No specific words asserted so the test stays robust across model versions.
  - Skipped by default in CI; run via `pytest -m live` before pushing.

### Modules **not** independently tested
- `schemas.py`, `main.py`, `routes.py` — exercised transitively by the `retrieval.py` integration tests.
- Trivial helpers (`slugify`, `tokenize`) — covered transitively by `parse_markdown` component tests.

### Prior art
This is a fresh project with no prior tests. The chosen integration-first style is itself a deliberate departure from the global `development-workflow.md` rule that says "Strict TDD"; that departure is recorded in `markdown_kb/tests/README.md` ("Why integration-first") and is justified by the Friday deadline and by the fact that `PROMPT.md`'s verification cases are themselves an integration-test contract.

## Out of Scope

The following are explicitly excluded from this PRD; several are reserved as deferred patterns or future ADRs.

- **`vector_rag/` implementation** — scaffold preserved, marked as post-prototype work in `vector_rag/README.md` and ADR-0002.
- **Wiki layer (`wiki/` synthesis pages, `/ingest`, `/lint`, Source Templates)** — target shape recorded in ADR-0003 and reserved-terms section of `CONTEXT.md`; implementation deferred.
- **Hybrid retrieval (BM25 + vector rerank), cross-encoder reranking, MMR dedup** — `phase: query` deferred patterns.
- **Multi-turn conversation memory, Query Rewriting, Conversation Store** — `phase: conversation` deferred patterns; PROMPT.md "Conversation Memory" stretch goal.
- **Streaming `POST /chat/stream`** (SSE) — `phase: streaming` deferred pattern; PROMPT.md stretch goal.
- **Browser UI, CLI interface, MCP server interface** — PROMPT.md stretch goals.
- **Multi-format ingest (`.txt` / `.html` → `.md`)** — PROMPT.md stretch goal.
- **Wiki Index Generation (`wiki/index.md`)** — PROMPT.md stretch goal; tied to Wiki layer.
- **Answer Filing** — `phase: ingest` deferred pattern.
- **Output validation (Grounding Check second LLM call)** — reserved in `CONTEXT.md`; documented in `inspiration.md` as the missing fourth anti-hallucination layer.
- **CJK / non-ASCII heading slug support** — flagged ambiguity in `CONTEXT.md`; revisit when ingesting personal CJK notes.
- **Stable Section IDs across heading or filename renames** — flagged ambiguity; addressed when the Wiki layer ships (UUID sidecar or content hash).
- **Incremental `/index` rebuild, file watcher auto-reindex** — full rebuild only in prototype.
- **Concurrent multi-user write contention beyond a single in-process lock** — single-server prototype scope.
- **Ed25519 signed receipt chain, Caveman compression** — explicit rejections in `inspiration.md#considered-and-rejected`.
- **Aggressive retry backoff for OpenAI errors** — LangChain default `max_retries=1` retained; tuning is post-prototype.

## Further Notes

- The PRD assumes the reader is familiar with `CONTEXT.md` (vocabulary), `project-docs/adr/0001..0003` (architectural decisions), `project-docs/inspiration.md` (external reference + deferred patterns), and `markdown_kb/tests/README.md` (test philosophy). Those documents are the source of truth for any term used here.
- A Workflow trigger in the repo's `CLAUDE.md` ensures that before any phase beyond this prototype (Wiki layer, `/ingest`, multi-turn, streaming, …) the agent re-reads `inspiration.md` and `grep`s the relevant `phase:` tag. This is the project's chosen mechanism against forgetting operational patterns we deliberately deferred.
- `wiki/log.md` was originally specified as committed to git (long-term audit trail + secondary backup). **Superseded** by the `wiki/` artifact taxonomy (`wiki/README.md`, commit `d00d9e3`): `wiki/log.md` is now a **runtime-trace** artifact and is **gitignored**, consistent with the per-app log-channel rule (CODING_STANDARD §5.1) and the later `vector_rag/log.md` / `gateway/log.md` channels.
- The intended interview narrative is: *"I delivered a Friday prototype that passes its verification contract, and the codebase plus its ADRs plus its deferred-patterns register show the architecture for enterprise KB management — a grounded Q&A bot with a clear upgrade path to a layered KB with curated synthesis. The patterns are sourced from the most-starred reference implementation of Karpathy's LLM Wiki gist; the destination is FAQ / policy / customer-support knowledge that compounds over time under knowledge-owner curation."*
