# Coding Standard — `knowledge_base_qa_bot`

How code is shaped in this repo, what conventions hold across modules, what design patterns are in play, and what tooling enforces them. This file is **not the spec** — that lives in `project-docs/prd.md` and `project-docs/adr/*.md` / `CONTEXT.md`. This file is the **consistency layer** that keeps the spec implementable across slices without drifting.

> **Freshness**: last reconciled through **ADR-0040** (2026-07-11). If `project-docs/adr/` holds a newer Accepted ADR than this stamp, this file is stale — reconcile per §0.3.

## 0. Reading order

In every fresh session, read in this order before writing code:

1. `CLAUDE.md` — agent skills configuration, workflow triggers
2. `CONTEXT.md` — vocabulary (Source, Section, Citation, Cannot Confirm, …)
3. `project-docs/adr/*.md` — architectural decisions
4. `project-docs/prd.md` — what we're building and why
5. **this file** — when reviewing code
6. `markdown_kb/tests/README.md` — when about to write or review a test
7. `project-docs/orchestration-plan.md` — when running the implementer/reviewer loop

## 0.1 Authority

Where this file conflicts with `CONTEXT.md`, the ADRs, or the PRD, **those documents win**. Where this file is silent:

- PEP 8 (style), PEP 257 (docstrings), PEP 484 + 604 (types).
- FastAPI / Pydantic / pytest official recommendations.
- John Ousterhout, *A Philosophy of Software Design* — the source for "deep modules" used throughout this codebase.

## 0.2 Reviewer injection scope

This document is the **reviewer agent's** standards reference (see [`project-docs/agents/review.md`](agents/review.md)). To keep review-agent context bounded, the reviewer reads sections **lazily, only when relevant to the diff under review**. The injection contract:

**Mandatory for every review** (read first, before looking at the diff):
- **§3 Domain rules** — vocabulary discipline (Source / Section / Citation / Cannot Confirm), reserved terms, sentinel string constants
- **§4 Error handling** — OpenAI exception → HTTP status mapping, fail-fast on corruption, Cannot Confirm as a success
- **§5 Logging and observability** — single `log_event` channel, bounded summaries, no `print()`
- **§11 Drift signals** — the actionable reviewer checklist (see § 11 itself for severity guide)

**Conditional — read only when the diff touches them:**
- **§1 Style** — if reformatting / naming / docstring issues come up (most style is handled by `ruff`; this section is for the cases ruff cannot catch, e.g. docstring intent quality)
- **§2 Architecture** — if a new module is added or an existing one significantly restructured
- **§6 Testing** — if test files are in the diff
- **§7 Dependencies** — if `pyproject.toml` or `uv.lock` is in the diff
- **§10 Design patterns in use** — when pattern-recognition is needed (e.g. reviewer suspects an anti-pattern is being introduced)
- **§12 Frontend (Gateway UI)** — if the diff touches any Gateway frontend (the reader chat UI or the Operator Console), the SSE client, or static assets

**Out of reviewer scope** (these are author-time / orchestrator-time):
- **§0** (Reading order, Authority, this section)
- **§8 Tooling recommendations** — adopting these is a separate `chore:` commit, not a per-slice review concern
- **§9 Commits and review** — the commit message rules are checked by the implementer; the reviewer-checklist subset (§9.2) is duplicated in agents/review.md's review process for self-containment

**Citation discipline when flagging**: when the reviewer flags an issue, it must cite the section by number and locate the offending code by function or symbol name (e.g. "§3.1 vocabulary drift — `Document` used in `build_index`, should be `Source`"). Avoid line-number citations (file:42 style) — they rot the moment the file is edited. Do NOT dump the section's full prose into the report — the section number is enough for the human to look up.

**Budget**: an active review typically loads §3 + §4 + §5 + §11 eagerly (~80 lines combined) plus 0-2 conditional sections on demand. Total injection is well under 200 lines — fits cleanly in any sub-agent's context window.

## 0.3 Freshness contract

This file lags the ADRs by design (§0.1 — ADRs win on conflict), but the lag must stay visible and bounded:

- A session that lands an Accepted ADR reconciles this file against it **in the same session**: fold repo-wide conventions into the affected sections (most often §11), or note "nothing standard-worthy" in the commit message, then bump the Freshness stamp at the top of this file.
- Staleness is checkable: newest ADR number > stamp = drift. Sub-agents flag it to the human (this file is human territory — orchestration-plan stop condition 5); they do not edit it.
- When folding a rule in, keep it abstract: code-site anchors (filenames of modules, scripts, tests) belong in the driving ADR's § Consequences, never here — the purity guard test fails this file on any Python-filename reference (ADR-0007).

---

## 1. Style

### 1.1 Line endings

LF everywhere, enforced by `.gitattributes`. Windows-only scripts (`.bat`, `.cmd`, `.ps1`) keep CRLF. Mixed-EOL diffs are a process bug — see the `.gitattributes` comment block.

### 1.2 Indentation, line length

- 4 spaces, never tabs.
- Soft target 88 chars (black default), hard ceiling 120. Long docstrings, log format strings, and URLs in comments may exceed.

### 1.3 Imports

- **`from __future__ import annotations` is the first import** in every module (after the module docstring). Lets `list[X]`, `dict[K, V]`, `X | None` work even when later type-resolution targets <3.10 environments. Already consistent across the codebase.
- Grouping (one blank line between groups):
  1. stdlib
  2. third-party
  3. local (`from . import …` or `from .foo import …`)
- Within each group, sort alphabetically.
- Prefer **relative imports within a single package** (`from .indexer import Section`); absolute imports for cross-package.
- Function-scope imports are allowed **only** to break circular dependencies. Always paired with a comment explaining why the cycle exists (see §1.8 on WHY-comments and §1.6 on rule-based functions).

### 1.4 Naming

| Kind | Convention | Example |
|---|---|---|
| Functions / variables / modules | `snake_case` | `build_index`, `ranked_sections` |
| Classes (incl. dataclasses) | `PascalCase` | `Section` (must match CONTEXT.md vocabulary) |
| Module-level constants & singletons | `SCREAMING_SNAKE_CASE` | `SYSTEM_PROMPT`, `INDEX_PATH`, `STOP_WORDS` |
| Module-private state / helpers | leading `_` | `_llm`, `_index_lock`, `_apply_grounding_check` |
| Pytest fixtures | `snake_case`, descriptive | `tmp_docs`, `tmp_kb` |

**Domain identifiers MUST match `CONTEXT.md` vocabulary** — see § 4.1. This is a hard rule, not a preference.

### 1.5 File layout (within a module)

```python
"""Module docstring (mandatory).

What this module does, in 1–3 sentences. Reference the ADR / PRD section
that drives the design — not what the code already shows.
"""
from __future__ import annotations

import stdlib_foo
import stdlib_bar

import third_party

from .local_module import Thing


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
CONSTANT_A = ...


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Foo: ...


# ---------------------------------------------------------------------------
# In-memory state (if any)
# ---------------------------------------------------------------------------
foo_state: list[Foo] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def public_function(...) -> ...: ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _private_helper(...) -> ...: ...
```

`# ---` divider lines are **mandatory** once a module has ≥4 logical sections. They make `grep -n "# ---"` a free table of contents. The deep-module indexer is the reference example.

### 1.6 Docstrings (PEP 257)

- **Every module:** triple-quoted docstring at the top. Intent + ADR/PRD reference.
- **Every public function / method:** triple-quoted docstring describing intent, non-obvious args, return shape, raised exceptions.
- **Private helpers:** one-line docstring only if the name is not self-explanatory.
- **Rule-based functions** (parsing, grounding, scoring): embed the rule spec inline in the docstring, numbered `1.` through `N.`, then reference rules in code with `# Rule N:` comments.
- Never write a "what" docstring on a trivial helper (`# return the sum` over `return a + b` is noise).

### 1.7 Type hints

- All **public** function signatures: type-hint every parameter and the return type.
- **Modern syntax only:** `X | None` not `Optional[X]`; `list[Foo]` not `List[Foo]`; `dict[str, int]` not `Dict[str, int]`. Enabled by `from __future__ import annotations`.
- Internal helpers can be untyped when types are obvious from the body.
- `Any` requires a comment explaining why a more specific type is impossible.

### 1.8 Comments

**Default to none.** Write a comment only when *why* is non-obvious:

- A hidden invariant (`# callers hold _index_lock when swapping the sections list`).
- A workaround for a specific bug or library quirk.
- A reference to an ADR / PRD section / CONTEXT term that drives the design.
- A guard against future drift (`# ADR-0003: SOURCE_DIRS is a list so future WIKI_DIR can be appended without signature change`).

**Never** write comments that restate the code or anchor to the current task ("added for Slice 5", "used by the routes module"). Git log and call graph already convey those.

---

## 2. Architecture

### 2.1 Deep modules (Ousterhout)

Modules are organized as **deep modules**: small public interface, large private implementation. The PRD declares the indexer and retrieval modules as deep modules. Every new module passes the deep-module test:

> Could a reasonable user of this module use it correctly knowing **only** the names + signatures of its public functions, with no need to read the implementation?

If no — refactor the boundary before merging.

### 2.2 Module size

- Up to ~500 lines per module is acceptable.
- Beyond ~500, split **only when a clear sub-responsibility falls out**. Do not split prophylactically (the PRD lists the prompt-builder module as a deliberate extraction so its output can be asserted in isolation without mocking the LLM — that's the bar).
- Per **ADR-0002**: do NOT extract a pluggable `Retriever` protocol until BOTH `markdown_kb` and `vector_rag` are end-to-end working. **Both now work, yet the protocol stays deferred to #107** ([ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md)): Phase 13's Hybrid stack is added via the existing string→callable dispatch, NOT a protocol, because extracting one would be a cross-cutting refactor touching the two existing apps. #107 lands later as its own refactor with Hybrid as the third implementation.

### 2.3 Module depth

Every module in the app package declares its Ousterhout depth on the first non-blank line of its docstring:

```python
"""Deep module per Ousterhout. Public surface: X, Y, Z.
...
"""
```

The reviewer opens the source file to discover a module's depth — there is no central inventory here to fall behind reality. Adding a new module **requires** this declaration; reviewers fail any PR that omits it.

**Deep modules own all conditional logic. Shallow modules wire them together.** Adding business logic into a module declared `Shallow module per Ousterhout` is a violation.

### 2.4 Forbidden cross-module patterns

- **No reaching into private state of another module.** Module-level public lists and dicts are public; private attributes (leading `_`) — including module-level `_private` *functions* — are not. **Blessed exception:** a cross-package `_private` import is acceptable ONLY when a named ADR records it as blessed coupling (ADR-0018 blesses `hybrid_kb`'s reuse of `markdown_kb._passes_index_filter` / `_section_lang` and `vector_rag._max_rag_distance()`, because the dense arm MUST share the BM25 arm's exact filter + threshold). Absent that ADR line, it is a §11 drift signal. **Escalation:** the moment a *second* package needs the same `_private` symbol, promote it to the owner's public API instead of importing it privately again (#326 promotes `hybrid_kb._ensure_indexes_loaded` to a public warmup seam for exactly this reason).
- **No circular imports.** When a module needs another, the import is at the top. When the cycle is unavoidable, use a function-scope import + a comment explaining the cycle (see §1.8 on WHY-comments).
- **No LangChain types leak to non-LLM modules.** LangChain message types, client types, and `with_structured_output` schemas stay inside **LLM-facing modules** — defined as modules that own an LLM call site. LLM-facing modules are enumerated in ADR-0005 § Consequences. Routes / schemas / indexer / logger / prompt-builder / wiki-index modules see only Python primitives and Pydantic models. The prompt-builder module was extracted precisely so its output can be asserted *as a string* without touching LangChain types.
- **No bare interpolation of untrusted text into LLM prompts.** Untrusted content spliced into a prompt (Source content, wiki page bodies, chat queries, uploaded/transcribed text) passes through `prompt_safety.wrap_untrusted()`'s fixed-sentinel fence, paired with the `UNTRUSTED_GUARD` system-prompt clause (ADR-0040). Interpolating it as a bare f-string is a §11 drift signal.

### 2.5 ADR- and PRD-encoded invariants

Architectural decisions that the codebase must preserve across phases are encoded directly in the documents that drive them:

- ADRs under [`project-docs/adr/`](adr/) tag each invariant with a `**Invariant**` prefix inside their `## Consequences` section.
- PRD-encoded invariants live in [`project-docs/prd.md`](prd.md) and the phase-specific PRD issues.

The reviewer must check that any PR which touches the code site of an invariant either preserves it or ships **a new ADR superseding the existing one**. Discovery flow: read the diff, identify the code sites it touches, then `grep -nE "Invariant" project-docs/adr/*.md` and scan the PRD for matching anchors. The reviewer fails any PR that breaks an invariant without a paired ADR.

### 2.6 Concurrency

- The index swap is the only contended operation **on the Section Index**. Hold the index lock **only** when assigning to the module-level `sections` list (see ADR-0003 § Consequences for the invariant).
- Readers do not lock. Mid-rebuild readers see the previous snapshot until the swap completes.
- **Persistent state writes are atomic**: write to `<file>.tmp`, then `os.replace(...)`. Never write to the target file directly. The indexer and wiki-index modules carry the canonical implementation of this pattern.
- **The Conversation Store is the second per-turn-mutated, TTL-swept structure** (Phase 11 — ADR-0013). Single-process CPython `dict` / `deque` operations are GIL-atomic for single-statement ops — no lock is needed for append or read under the current single-worker model. The TTL sweep (`evict_expired()`) iterates over a **snapshot** of keys (`list(self._sessions)`) so that deleting expired entries inside the loop never triggers `RuntimeError: dictionary changed size during iteration`.
- Beyond a single FastAPI worker, both the Section Index lock model and the in-memory Conversation Store break. Multi-worker is **post-prototype**; will need an external lock / Redis-backed store. Do not refactor proactively.

### 2.7 State management

- Module-level mutable globals (sections list, doc-frequency map, LLM client) are **acceptable for the prototype's single-process model**. They are explicitly designed so tests can swap them via `monkeypatch`.
- Singleton LLM clients use lazy init via getter functions. This lets tests stub them without instantiating a real OpenAI client.
- When this codebase outgrows the single-process model, lift state onto `app.state` or a DI container. **Do not refactor today.**

---

## 3. Domain rules

### 3.1 Vocabulary discipline (mandatory)

Code identifiers MUST use the [`CONTEXT.md`](../CONTEXT.md) vocabulary verbatim. The active glossary is the single source of truth — any concept the codebase names (in class, function, variable, log kind, or comment) must already exist there.

When you need a new domain concept, propose the term via `/grill-with-docs` **before** naming a class/function for it. Inventing vocabulary directly in code creates drift that's expensive to undo.

### 3.2 Reserved terms are off-limits as variable names

The `## Reserved (not yet implemented)` section in [`CONTEXT.md`](../CONTEXT.md) is the single source of truth for terms that name future phases. Using one of those terms as a local helper or variable name silently consumes the namespace before the matching feature ships; the reviewer downgrades to a non-reserved synonym. Read that section before naming anything that *sounds* domain-y — promotion of a term from Reserved to active is the only path to using it in code.

### 3.3 Constants for sentinel strings

Any literal string with semantic meaning that appears more than once gets a module-level constant. The Cannot Confirm phrase and the system-prompt string are the canonical examples — they are defined once in the module that owns them and imported everywhere else. Inline repetition of a sentinel string in tests or routes is a drift signal (see §11).

---

## 4. Error handling

### 4.1 Fail fast on data corruption

A corrupt `.kb/index.json` at startup **raises** and prevents the server from starting. Silently serving stale or wrong data is worse than not serving. Apply the same rule to every persistent-state load you add.

### 4.2 OpenAI exception mapping (amended by ADR-0015)

The LLM-call wrapper raises a **transport-agnostic `LLMError`** (defined in
`markdown_kb/app/errors` — see ADR-0015) instead of `fastapi.HTTPException`.
The HTTP status table moves from the wrapper to the HTTP route adapter:

| Exception | `LLMError.retryable` | Log kind | HTTP route renders as |
|---|---|---|---|
| `APITimeoutError`, `RateLimitError` | `True` | `openai_transient` | 503 |
| `AuthenticationError` | `False` | `openai_auth` | 500 |
| Any other `APIError` subclass | `False` | `openai_api` | 500 |

Every branch emits a `chat_error` log entry with the right `kind=` tag
**before** raising.  Use `raise LLMError(...) from exc` to preserve the
exception chain.  Never bare-raise.

Each interface adapter renders `LLMError` per its transport:
- **HTTP routes** (both stacks): `raise HTTPException(503 if e.retryable else 500, e.message) from e`
- **Gateway SSE generator**: terminal `error{detail: e.message, retryable: e.retryable}` event
- **MCP / CLI** (Phase 12+): per-transport rendering, no HTTP leak

The retrieval module no longer imports `fastapi`; the wrapper is
decoupled from HTTP transport (ADR-0015 § Consequences).

### 4.3 `Cannot Confirm` is a success, not an error

Per **ADR-0001** and **ADR-0004**:

- Empty / sub-threshold retrieval returns the exact Cannot Confirm sentinel phrase with HTTP **200**.
- The LLM is **not called** in this path (pre-LLM gate).
- All Block & Replace grounding logic — what happens when the LLM response is ungrounded, how many retries, what happens to `sources` — is defined in ADR-0001 and ADR-0004 § Consequences. This file does not duplicate impl detail.

Adding a shortcut that bypasses the pre-LLM gate ("if score is just barely below threshold, send to LLM anyway") is a deliberate ADR-0001 violation. Reviewer must fail it.

**Invariant — gate parity across stacks and interfaces.** Every retrieval stack (wiki/BM25, RAG/FAISS, future hybrid) MUST enforce its pre-LLM relevance gate inside its **own deep module** (`retrieval._retrieve_and_gate`), never in an interface adapter — so Browser (gateway), MCP, and CLI all inherit the identical gate and no interface can bypass it by switching transport. The corollary is a real failure mode: a stack that *retrieves but never gates relevance* is a parity gap. FAISS k-NN always returns `k` neighbours, so a RAG path with **no distance/relevance gate** forwards arbitrarily-far chunks to the LLM and leans on the (expensive) post-LLM grounding net alone — asymmetric with the BM25 path's pre-LLM score gate (issue #257). Reviewer must flag a retrieval path whose relevance gate is missing, or implemented in an adapter rather than the stack's deep module. When tuning such a gate, the threshold must be **calibrated against an eval** (the `KB_SCORE_THRESHOLD` / #253 precedent), not hand-picked.

### 4.4 Validation at boundaries only

- Request validation: Pydantic does it at the route boundary via request/response schemas. **No** defensive re-validation inside route handlers.
- Deep-module ↔ deep-module: trust types. Don't `isinstance`-check inputs from your own codebase.

### 4.5 LLM-judge failures take the restrictive branch

When an LLM-judge call (grounding verify, lint judges, the C5 convergence re-judge, …) errors, times out, or returns indeterminate output, treat the result as the **restrictive** branch: never auto-enable a mutating action (e.g. Reconcile Apply) under judge uncertainty (ADR-0038). Fail-closed here mirrors §4.1 — a judge you could not run is a judge that said no.

---

## 5. Logging and observability

### 5.1 Per-app log channel

Each package owns **one** log channel: `<package>/log.md`, written via that
package's own `log_event(kind, summary)`. The three current channels are
`wiki/log.md` (`markdown_kb`), `vector_rag/log.md` (`vector_rag`), and
`gateway/log.md` (`gateway`).

**The violation is** `print()`, `logging.getLogger(...)`, or `sys.stderr.write(...)`
in production code — **or** writing into *another* package's log channel. A package
having its own `<package>/log.md` is correct; cross-package writes are not.

If you want a debug-only signal inside a package, either:

- Add a new `kind` to that package's log (e.g. `parse_warning`, `chat_grounding_retry`).
- Or use `pytest.fail` / `assert` for test-only diagnostics.

### 5.2 Log line format

```
## [<ISO-8601 UTC>] <kind> | <summary>
```

- `kind` is `snake_case`. [`log-kinds.md`](log-kinds.md) is the single source of truth for every `kind` value in use across all phases; adding a new one means adding a row there in the same commit.
- `summary` is `grep`-friendly: KEY=value pairs separated by spaces; query strings double-quoted; never embed newlines.

### 5.3 Summaries are bounded

- Truncate user queries to 60 chars: `question[:60].replace('"', "'")`. This idiom is established in the LLM-call wrapper module — when adding a new log site, copy it verbatim.
- Never log API keys, full request bodies, or full document content.
- Score values rounded to 3 decimal places: `round(score, 3)`.

---

## 6. Testing

### 6.1 Inverted pyramid (per `markdown_kb/tests/README.md`)

- **Many** integration tests (`TestClient` + fake LLM) covering PROMPT.md verification cases.
- **Some** component tests for parsing, indexing, and BM25 ranking order.
- **Few-to-zero** unit tests on trivial helpers (tokenisation, slugification) — **but a helper graduates the moment it carries a hard invariant or a non-trivial algorithm, and then unit-testing it is not over-testing.** Phase 16 makes `tokenize` (language-agnostic: CJK character bigram + unigram fallback) and `slugify` (Unicode-preserving) non-trivial: the **ASCII-byte-identical invariant** — pure-ASCII input must tokenise/slug exactly as before, the guard against English BM25 / `KB_SCORE_THRESHOLD` / Phase 8 baseline regression — plus the bigram and Unicode behaviour each warrant focused unit tests. The "trivial helper" guidance above applies only while the helper stays trivial.
- **One live test per LLM-facing surface** (see §6.4). Opt-in only; auto-skipped via the conftest collection hook. A new language/script (e.g. Phase 16 Chinese) does **not** earn a second live test on a surface that already has one (`/chat`, `/ingest`); validate it with hermetic mocked-LLM integration tests instead, and assert language **structurally** (CJK code-point presence + directive-presence in the prompt constant), never specific model wording (§6.2).

### 6.2 What to assert vs not

| Assert | Don't assert |
|---|---|
| HTTP status code | BM25 score absolute values (corpus-sensitive, brittle) |
| Response shape (keys, types, list lengths) | LLM output text content beyond shape + `[Source:` markers |
| Exact literal sentinel strings (Cannot Confirm phrase, citation format) | Wall-clock timing |
| Section IDs, ranking order | Specific words in the model's reply |
| Log line presence + structure | Anything that breaks across model updates |

### 6.3 Mock the LLM, not the indexer

- The LLM is the **only** thing that should be replaced with a stub. Use `monkeypatch` on the LLM getter functions (see §2.7 on lazy singleton getters).
- The indexer always runs against real fixture files under `tmp_docs`. Mocking a deep-module entry point masks integration drift — fail any PR that does this.
- **The LLM getter is not the only live network seam.** The RAG and Hybrid stacks embed the query *at retrieval time*, reached through a lazily-loaded committed index seed (`vector_rag`'s `.kb/faiss_index` via `load_vector_index`, `hybrid_kb`'s `.kb/hybrid_dense` via `load_dense_index`) — **not** through the mocked LLM getter. A test that touches `stack=rag` / `stack=hybrid` (or otherwise reaches those load paths) MUST also redirect that index directory to an empty `tmp_path` so retrieval takes the `index_missing` early-exit (or mock the embeddings getter). Mocking only the LLM getter leaves this seam live: the test 401s on a dummy key and **bills real tokens on a live one**, while still looking hermetic. (Incident #332: the gateway `stack=rag` test mocked the wiki LLM yet lazy-loaded the committed FAISS seed and embedded the query.)

### 6.4 Live smoke discipline

- **One live test per LLM-facing surface** is the policy. LLM-facing surfaces are enumerated in ADR-0005 § Consequences (updated when a new surface ships). Adding a second live test to an existing surface, or a live test to a new surface without explicit PRD authorisation, is scope creep; push the assertion into a mocked integration test instead.
- A live test asserts **shape** (200, citation pattern present, non-empty sources, all expected frontmatter fields parseable), **never** specific words. Models update; tests outlive them.
- A post-deploy security/attack probe (e.g. the ADR-0040 injection-probe runner under `project-docs/security/injection-probe/`) verifies a hardening decision against the live deployment; it is an ops runbook artifact, **not** a `@pytest.mark.live` test, and does not consume the one-live-test-per-surface budget.

### 6.5 Fixtures

- Per-test isolation via `tmp_path`-derived fixtures (see the test-suite conftest for the established pattern).
- Tests that mutate the module-level sections list MUST restore it via `monkeypatch` (auto-restores) or an explicit teardown.
- **Tests MUST NOT mutate committed on-disk invariants** — chiefly `.kb/index.json`. Any test that exercises a real write path (`build_index`, `ingest_sources`, `import_sources`) MUST redirect `INDEX_PATH` / `WIKI_DIR` / `LOG_PATH` (and `SOURCE_DIRS` when calling `build_index()` with no `docs_dir`) to `tmp_path`. The repo-root `conftest` session guard snapshots and restores `.kb/index.json` as a backstop and **warns** if a test mutated it — that warning means a test's isolation leaked (typically a live test), not that the guard is the fix. The guard is a safety net, not a license to skip per-test redirection.
- Test fixtures are hand-written, deterministic, and mirror the shape of real Sources / Wiki Pages; never LLM-generated at test time.
- **Fixture fidelity (a real, recurring trap).** "Mirror the shape" means *include the structural elements the real producer writes* — not a simplified ideal. Canonical example: every Wiki Page written by `POST /ingest` (`wiki_writer._render_page`) begins with an auto-generated sentinel HTML comment *before* the `---` frontmatter; a wiki-page fixture that starts directly with `---` does not mirror reality, so a parser/consumer regression that only triggers on the real shape passes under it. (Incident 2026-06-28: `_derived_from_for_section` and the #266 citation path were silently broken in production while the suite stayed green, because `_write_concept_page` omitted the sentinel comment — see `project-docs/findings-indexer-frontmatter-comment.md`.) When a producer gains a new structural element, every consumer's fixtures must gain it too.

### 6.6 Eval & report artifacts encode trust level

Committed eval/report artifacts (e.g. `eval/paraphrase_comparison/report.md`) carry a **trust level in the filename**, never only in their contents. A run that produces anything less than real data — `--fake-embeddings`, offline stand-ins, placeholder/synthetic numbers — MUST write to a trust-marked path (`<name>.offline-tracer.<ext>`) led by a loud top-of-file header (`⚠️ PLACEHOLDER — NOT REAL DATA …`, defined once as a module constant per §3.3). Only a real-data run writes the **canonical** name. Trust is **file-level and worst-case**: if *any* arm/column is non-real (e.g. the BM25 numbers are real but the dense arms are faked), the whole file takes the trust-marked name. The writer picks the path from the run mode — a `--fake` flag must never overwrite a canonical artifact (#328; the direct lesson from the fake-`report.md` footgun).

---

## 7. Dependencies

### 7.1 Source of truth

- `pyproject.toml` (root + each workspace member) is the source of truth for declared deps.
- `uv.lock` is committed and is the source of truth for resolved transitive versions.
- **No `requirements.txt`.** Anyone reading the codebase should know exactly one place to look.

### 7.2 Adding a dependency

- Use `uv add <pkg>` for runtime, `uv add --dev <pkg>` for test/lint tooling. Editing `pyproject.toml` by hand bypasses lockfile updates.
- Pin to `==` for libs you depend on at the public-API surface (e.g. LangChain class names); use `>=` for transitive-only or test-tooling deps.
- A new dependency requires a one-sentence rationale in the commit message: why this lib over stdlib / over an existing dep / over a hand-rolled solution.

### 7.3 Python version

- Pinned to `>=3.11` via `pyproject.toml` (every member) and the root version pin file.
- Upgrading: change the root version pin file, bump `requires-python` in every member's `pyproject.toml`, run `uv sync --all-packages`, re-run pytest.

---

## 8. Tooling (recommended additions — not in repo today)

Adopting each is a self-contained `chore:` commit. Listed in priority order.

### 8.1 `ruff` — formatter + linter (replaces black / isort / flake8)

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = []
```

Commands: `uv run ruff format .`, `uv run ruff check .`.

### 8.2 `mypy` — static type checking

Start in `--check-untyped-defs` mode, tighten over time. A type error fails CI. Type-only changes don't need an ADR.

### 8.3 Pre-commit hooks

Run `ruff format` + `ruff check` (+ optionally `mypy`) on staged files. README documents `pre-commit install` as a one-time setup step.

### 8.4 Coverage (deliberately deferred)

The PRD's "integration-first, thick at top, thin at bottom" philosophy makes line-coverage a misleading metric. Re-evaluate if a class of bugs starts slipping through.

---

## 9. Commits and review

### 9.1 Commit message format (per `git-workflow.md`)

```
<type>: <description>

<optional body — focus on WHY, not WHAT>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.

For slice commits per `orchestration-plan.md`, the body MUST include:
- 2–3 sentence summary
- Acceptance-criteria checklist (`- [x] …`)
- Files touched section

### 9.2 Reviewer checklist (apply in order)

1. `pytest` (default markers) is green after the commit.
2. Acceptance criteria genuinely met — read code + tests; don't trust the commit message.
3. **Vocabulary discipline** (§ 3.1) — no smuggled-in synonyms.
4. **ADR invariants** (§ 2.5) — none silently broken.
5. **Drift signals** (§ 11) — none present.
6. Style / type / docstring rules — fix-on-merge for trivia, request-changes for systematic drift.
7. Scope creep — anything outside the issue's "What to build" gets called out.

### 9.3 No broken commits on `main`

- Orchestration plan is single-branch (no PR flow).
- The implementer is responsible for `pytest` being green **before** committing. A red `main` is a process bug.
- Hotfix ships in a **new** commit, never via `--amend` (per `git-workflow.md`).

---

## 10. Design patterns in use

For quick recognition during code review. Code-site anchors for each pattern live in the relevant ADR § Consequences — grep `**Invariant**` in `project-docs/adr/*.md` to find them.

| Pattern | Why this one |
|---|---|
| **Deep module** (Ousterhout) | Small public surface; large private implementation (BM25 math, parsing rules, error mapping, grounding heuristics). The indexer, retrieval, and grounding modules are the canonical examples. |
| **Lazy singleton** | Avoids constructing a real OpenAI client at import time; lets tests stub via `monkeypatch` before first use. LLM getter functions are the canonical sites. |
| **Guard clause / early return** | ADR-0001: never hand weak context to the LLM. Two early returns before the prompt is even built (pre-LLM Cannot Confirm gate). |
| **Atomic write (tmp + rename)** | Crash mid-write must not leave a half-written file for the next read. POSIX `os.replace` is atomic on a single filesystem. The indexer and wiki-index modules are the canonical sites. |
| **Append-only log** | Karpathy log discipline; survives across crashes; `grep`-able audit trail. Each package routes events to its own `<package>/log.md` via `log_event` (§5.1). |
| **DI via monkeypatch** | No DI framework; tests use pytest's `monkeypatch` to swap module-level state. LLM getter functions and the log path are the canonical swap points. |
| **Strategy (deferred / implicit)** | ADR-0002 explicitly defers a `Retriever` protocol until both retrieval implementations work. The two workspace packages are two strategies; the abstraction is *not* extracted yet. |
| **Repository / in-memory store** | Single-process, single-writer; module-level list is the "repository." When this model breaks, the upgrade path is `app.state` or external store. |
| **Adapter** | LangChain client wraps the OpenAI SDK. Provides timeout/retry plumbing; isolated to the LLM-call wrapper module so the rest of the codebase never sees LangChain types. |
| **Structured-output adapter via `with_structured_output`** | ADR-0005 pre-blessed component pattern. LLM bound to a Pydantic schema; schema is never exposed outside the owning module. Both classification and synthesis calls use this pattern so LLM output is always validated at the boundary. |
| **Soft-demote / tombstone** | A defective or stale `status: live` record demotes in place to `draft` (content preserved, reversible, logged) rather than hard-delete; the curator edits / re-promotes or discards it. `qa.demote` is the canonical primitive (ADR-0035 / ADR-0037). |

Notable patterns **rejected** (do not introduce):

- A `Retriever` protocol / plugin architecture (ADR-0002 / [ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md) — deferred to #107). Both stacks now work, so ADR-0002's "premature" bar is lifted, but the protocol is still not extracted inside a feature slice; a third stack (Hybrid) is added via the existing dispatch.
- Writing into *another* package's log channel, or using `print` / `logging.getLogger` / `stderr` instead of `log_event` (§ 5.1 — each package owns one `<package>/log.md` channel; cross-package writes are the violation).
- A `Document` or `Chunk` class **in `markdown_kb`** (§ 3.1). `vector_rag`'s `Chunk` is the blessed exception — its distinct retrieval unit (a char-bounded slice within a Section), defined in [`CONTEXT.md`](../CONTEXT.md) § Phase 8 vocabulary. LangChain's `Document` stays inside vector_rag's LLM-facing modules and never leaks (§ 2.4).
- A DI container / app-state object (§ 2.7 — only when single-process breaks).

---

## 11. Drift signals (the reviewer's actionable checklist)

When the **review agent** ([`project-docs/agents/review.md`](agents/review.md)) inspects a diff, it walks this checklist top-to-bottom and ticks anything that appears in the actual diff (verified via `git diff`, NOT inferred from the issue body or PRD spec — see Factual discipline in `review.md`).

Each signal has a **severity** that determines the reviewer's action:

- **FAIL** = AC is not actually satisfied (or a hard ADR invariant is broken without a paired new ADR). Reviewer returns `FAIL`; the implementer must revisit. Reviewer does NOT silently fix these.
- **FIX** = small, safe, mechanical refactor the reviewer can apply directly with a `refactor:` commit. Reviewer fixes, commits immediately (per Turn-budget discipline), and lists the commit in "Changes made".
- **FLAG** = correctness or scope concern that needs human judgment. Reviewer notes in "Concerns flagged for human" and does NOT make the change.

### Vocabulary drift (§3.1)

- [ ] **FAIL** — A new domain term appears in code without a [`CONTEXT.md`](../CONTEXT.md) entry, OR a synonym smuggles in for an existing CONTEXT term. Check `CONTEXT.md`'s active glossary for the canonical name.
- [ ] **FIX** — A local variable name consumes a term from `CONTEXT.md`'s `## Reserved (not yet implemented)` section. The reserved list there is the live source of truth — read it before flagging, not from memory.

### Sentinel string drift (§3.3)

- [ ] **FIX** — A branch returns a paraphrase of "Cannot Confirm" instead of the sentinel-string constant defined in the LLM-call wrapper module.
- [ ] **FIX** — An inline Cannot Confirm literal appears in tests or routes (use the constant, imported from its owning module).

### Architecture & dependency drift (§2)

- [ ] **FAIL** — A new module imports `langchain` or `langchain_openai` outside the LLM-call wrapper module, OR LangChain types leak through that module's return values into other modules. (Violates §2.4.)
- [ ] **FAIL** — Business / conditional logic appears in a module whose docstring declares it `Shallow module per Ousterhout` (§2.3). Should be in a deep module.
- [ ] **FAIL** — A code change breaks any `**Invariant**`-tagged line in `project-docs/adr/*.md` (or a PRD-encoded invariant) without a paired new ADR superseding it. See §2.5.
- [ ] **FAIL** — Pre-LLM Cannot Confirm gate is bypassed when retrieval score is "just barely below threshold" (violates ADR-0001 + §4.3).
- [ ] **FAIL** — A `Retriever` protocol / plugin layer is extracted **as part of a feature slice** instead of the dedicated #107 refactor. Both stacks now work (ADR-0002's "premature" bar is lifted), but [ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md) adds the third stack (Hybrid) via the existing string→callable dispatch; extracting the protocol would touch the two existing apps, which is out of Phase 13 scope.
- [ ] **FLAG** — A module imports another *package's* `_private` symbol (function or state) that no ADR blesses (§2.4). Note it + require a WHY-comment and a tracked promotion issue (the #315→#326 pattern). **Escalates to FAIL** if it ships undocumented, or if a *second* package imports the same `_private` symbol — then it MUST be promoted to the owner's public API. Blessed exceptions (grep the ADR for the symbol): ADR-0018 → `_passes_index_filter` / `_section_lang` / `_max_rag_distance`.
- [ ] **FAIL** — A new LLM-facing prompt splices untrusted content (Source text, wiki bodies, user queries, uploaded/transcribed text) as a bare string instead of through `prompt_safety.wrap_untrusted()` + the `UNTRUSTED_GUARD` clause (ADR-0040; §2.4).

### Error handling drift (§4)

- [ ] **FAIL** — HTTP error mapping for OpenAI exceptions drifts away from §4.2 (e.g. `RateLimitError` returns 500 instead of 503).
- [ ] **FAIL** — A persistent-state load (e.g. `.kb/index.json`) silently fallbacks to empty on corruption instead of raising (violates §4.1 fail-fast).
- [ ] **FIX** — A handler uses bare `raise` instead of `raise LLMError(...) from exc` or `raise HTTPException(...) from exc` (loses the exception chain per §4.2).
- [ ] **FAIL** — The retrieval module (either stack) imports `fastapi`; the LLM-call wrapper must be transport-agnostic (ADR-0015).
- [ ] **FAIL** — Pydantic boundary validation is re-implemented inside a route handler (violates §4.4).

### Logging drift (§5)

- [ ] **FAIL** — `print()`, `logging.getLogger(...)`, or `sys.stderr.write(...)` lands in production code (violates §5.1 per-app log channel).
- [ ] **FAIL** — A module writes into *another* package's log channel (e.g. gateway code calling `markdown_kb.app.logger.log_event`, or monkey-patching `LOG_PATH` to point at a sibling package's log file in production code — violates §5.1 per-app log channel).
- [ ] **FIX** — A new log site logs full user queries or full document content (violates §5.3 bounded summaries; truncate to 60 chars).
- [ ] **FIX** — A new log site logs unrounded float scores (violates §5.3; use `round(score, 3)`).
- [ ] **FAIL** — A new log `kind=` is used without a corresponding row in [`log-kinds.md`](log-kinds.md).

### Testing drift (§6)

- [ ] **FAIL** — A test mocks any deep-module entry point (indexer search, grounding verify, etc.) instead of the LLM getter; per §6.3.
- [ ] **FAIL** — A new `@pytest.mark.live` test appears on a surface already covered by an existing live test, OR a live test is added to a new LLM-facing surface without explicit PRD authorisation (one-per-surface is the policy; per §6.4).
- [ ] **FAIL** — A test asserts an absolute BM25 score value (corpus-sensitive, brittle; per §6.2 — assert ranking order or shape instead).
- [ ] **FAIL** — A change to `tokenize` / `slugify` alters **pure-ASCII** token/slug output **without eval evidence that the retrieval baseline is preserved or improved** — a *silent* change is the FAIL. Run `negative_case` + `paraphrase_comparison` and show the before/after. The Phase-16 "byte-identical pure-ASCII" guarantee is **superseded** (it was a proxy for "don't silently regress English BM25 / `KB_SCORE_THRESHOLD` / the Phase 8 baseline"): an eval-backed change **is** allowed. Precedent — #252 intentionally dropped junk tokens (possessive `'s`, unfiltered `in`) and moved the negative-case correct-refusal **73%→87% with no paraphrase regression**. What stays a FAIL is changing tokenisation on a hunch with no measurement. See §6.1.
- [ ] **FLAG** — A test asserts specific LLM output text content beyond shape + `[Source:` marker (will break across model updates; per §6.2). For Phase 16, asserting an answer is "in Chinese" means CJK code-point presence, not specific words.
- [ ] **FIX** — A test mutates the module-level sections list without restoring via `monkeypatch` or explicit teardown (per §6.5).
- [ ] **FAIL** — A test exercises a real write path (`build_index` / `ingest_sources` / `import_sources`) without redirecting `INDEX_PATH` / `WIKI_DIR` / `LOG_PATH` (and `SOURCE_DIRS` for a default `build_index()`) to `tmp_path`, so it writes the committed `.kb/index.json` or real `wiki/` (per §6.5). The repo-root session guard restoring the file + warning is the signal, not the remedy.
- [ ] **FAIL** — A fixture for a Wiki Page / Source omits a structural element its real producer writes (e.g. the `/ingest` sentinel HTML comment before frontmatter, or `importer`-written provenance frontmatter), so it cannot catch a regression the real artifact would trigger (per §6.5 fixture fidelity).

### Alias & quarantine drift (ADR-0029 / ADR-0030)

- [ ] **FAIL** — A surface computes wikilink resolution (red-link judgment, linkify, inbound-reference scanning) from a locally-built slug set instead of the shared resolver (ADR-0030: one resolver, all consumers).
- [ ] **FAIL** — An alias value reaches the Section Index (BM25 tokens) or the dense arm (ADR-0030: link-layer only in v1), or `/ingest`'s overwrite drops the `aliases` frontmatter field (preserve list is `{created, aliases}`).
- [ ] **FAIL** — The Section Index admits a `status: failed_grounding` page (ADR-0029 quarantine), or an MCP tool writes aliases (ADR-0030 / ADR-0026 posture).

### Dependencies drift (§7)

- [ ] **FAIL** — `requirements.txt` reappears anywhere in the tree (uv is the single source of truth per §7.1).
- [ ] **FAIL** — A new dependency is added by hand-editing `pyproject.toml` instead of `uv add` (lockfile drift risk per §7.2).
- [ ] **FLAG** — A new dependency lacks a one-sentence rationale in the commit message (per §7.2).

### Documentation discipline (§1.6, §1.8)

- [ ] **FIX** — A new module is missing the top-of-file docstring with intent + ADR/PRD reference.
- [ ] **FIX** — A function-scope import lacks a comment explaining the circular-dep workaround.
- [ ] **FIX** — A comment paraphrases obvious code (delete it — only WHY-comments per §1.8).

### Git hygiene

- [ ] **FAIL** — A PRD-encoded invariant is broken. (Note: prd.md's original "`wiki/log.md` committed, not gitignored" intent was **superseded** by the `wiki/` artifact taxonomy — `wiki/README.md`, commit `d00d9e3`. `wiki/index.md` / `wiki/log.md` / `wiki/hot.md` and every `<package>/log.md` are generated / runtime-trace artifacts and stay **gitignored** per §5.1; *committing* one is now the drift.)
- [ ] **FAIL** — An ADR-tagged `**Invariant**` is broken **without** a paired new ADR superseding it. Locate them via `grep -nE "Invariant" project-docs/adr/*.md`.
- [ ] **FAIL** — A code path that can emit fake/offline/placeholder eval data writes to a *canonical* artifact name (e.g. `report.md`) instead of a trust-marked `*.offline-tracer.*` path with the placeholder header (§6.6). Committing a non-real artifact under a real name is the footgun #328 closes.

### Frontend drift (§12)

- [ ] **FAIL** — A frontend framework, bundler, or build step is introduced; the Gateway UI must stay vanilla single-file (§12.1).
- [ ] **FAIL** — `EventSource` is used for the POST `/chat/stream` (it is GET-only; must use `fetch()` + ReadableStream per §12.2).
- [ ] **FAIL** — Server- or LLM-derived content (answer text, source snippets, headings, citations) is inserted via `innerHTML` instead of `textContent` (XSS; §12.4).
- [ ] **FAIL** — The client implements optimistic-render-then-retract for answer tokens (nothing to retract under verify-then-stream; §12.3).
- [ ] **FAIL** — An answer area renders before the `sources` event arrives (violates sources-first; §12.3).
- [ ] **FIX** — The client branches on a special `cannot_confirm` event instead of the uniform `token` + `done{passed:false}` representation (§12.3).
- [ ] **FLAG** — Grounding / citation / filing logic is re-implemented client-side instead of consumed verbatim from the server events (§12.5).
- [ ] **FAIL** — The Operator Console issues per-file `/ingest` calls in a loop instead of one batch call over the (≤5) named sources; breaks the per-call `used_slugs` cross-source slug-collision guarantee (#54; §12.8).
- [ ] **FIX** — A frontend shows a fake / time-eased progress percentage where the client has no real progress signal; use an indeterminate indicator or a client-owned counter ("Batch k/N") instead (§12.8).
- [ ] **FAIL** — A lint-remediation "Re-ingest (retry)" for a **C3 failed-grounding** finding omits `force:true`, so hash-skip idempotency (#93) no-ops the retry and the finding silently persists while the UI implies a fix (ADR-0023 Invariant; §12.8).
- [ ] **FAIL** — A one-click **batch** remediation button is offered on an **Authored-tier** finding (Coherence C5/C4, Coverage C1/C2), bypassing the per-item human-approval gate (ADR-0023 Invariant — batch is Direct-tier-only; §12.8).

The reviewer's job is to spot these and act per the severity. The implementer's job is to not write them in the first place — reading this section before writing code is cheaper than re-doing it after a `FAIL`.

---

## 12. Frontend (Gateway UI)

The repo has **two** Gateway-served browser frontends: the **reader chat UI** (Phase 9 — [ADR-0009](adr/0009-streaming-verify-then-stream.md), [ADR-0010](adr/0010-gateway-mounts-both-apps.md)), a presentation shell over the SSE stream; and the **Operator Console** (Phase 15 — [ADR-0011](adr/0011-upload-separate-from-import.md), [ADR-0012](adr/0012-delete-inert-filed-answers-only.md)), served at `/console`, a curator-facing management surface (Upload drop zone, pipeline stepper, Curation Queue, resource browser) that is **not** SSE-based. Both are presentation shells — all retrieval, grounding, filing, and lifecycle logic stays server-side.

**Scope of this section:** §12.1 (vanilla single-file / no build), §12.4 (textContent XSS), §12.5 (no business logic in the client), and §12.7 (testing) apply to **both** frontends. §12.2 (SSE client) and §12.3 (grounding-inspector invariants) are **reader-UI-specific**. §12.8 covers the Operator Console.

### 12.1 Stack — vanilla, single-file, no build

Vanilla HTML/CSS/JS, a **single file**, served by the Gateway at `/` (FastAPI `StaticFiles`). **No framework, no bundler, no build step** — introducing React/Vue/a build tool is a violation (a rejected pattern, like a second log channel in §5.1). The "no framework friction in a Python repo" rule is deliberate, not an oversight.

### 12.2 SSE client — `fetch` + ReadableStream, never `EventSource`

- `/chat/stream` is **POST**, and native `EventSource` is GET-only. The client MUST use `fetch()` + `ReadableStream` + a small hand-written SSE parser (buffer, split on the `\n\n` event delimiter, read `event:` / `data:` lines, dispatch on event type).
- Keep the **SSE parser a pure function** (text chunks → parsed events), separable from DOM rendering, so it is unit-testable without a browser. It is the deep/testable unit of the frontend (mirrors §2.1).
- Handle the full event contract (ADR-0009): `sources`, `status`, `token`, `done`, `error`. Unknown event types are ignored (forward-compat).

### 12.3 Grounding-inspector UX invariants

These encode ADR-0009 in the UI — they are correctness, not decoration:

- **Sources render first.** Never render an answer area before the `sources` event arrives (the whole point — PROMPT.md "Return selected sources first").
- **The answer area only ever shows verified text.** The client renders `token` events as-is; the server already gated them. The client MUST NOT implement any optimistic-render-then-retract behaviour — under verify-then-stream there is nothing to retract.
- **Cannot Confirm is uniform.** It arrives as `token`(s) of the fixed phrase + `done{passed:false, reason}`. The client does NOT branch on a special `cannot_confirm` event.
- **`done` drives the grounding badge** (✓/✗ + reason) and the filed indicator (Wiki only; `done.filed`).
- **The stack toggle** maps to the `stack` query param; switching stacks is a fresh request.

### 12.4 Security — render server/LLM content as text

Answer text, source `content` snippets, headings, and citations are LLM/corpus-derived → **untrusted for rendering**. Insert them with `textContent` / safe DOM construction, **never** `innerHTML`. No `eval`, no dynamic `<script>` injection. This is the one hard frontend security rule (XSS).

### 12.5 No business logic in the client

The client renders events and issues requests. It does NOT re-implement grounding, gating, citation formatting, or filing decisions — those are server concerns (mirrors §2.3: the UI is a shallow presentation shell). Source/citation shape comes from the `sources` event verbatim (the trio + `derived_from`); the client does not reconstruct citation ids.

### 12.6 Design exploration is out-of-loop

Generating multiple visual variations (`frontend-design` / `prototype` skills, or claude.ai design) is a **human-in-the-loop creative step**, not an implement-agent TDD slice. The orchestration loop consumes the **chosen** mockup and wires it to the real events. A "pick a design" step is not test-reviewed for coverage; the wiring slice is.

### 12.7 Testing

- **SSE parser:** unit-tested as a pure function (hermetic).
- **Event-sequence behaviour:** asserted at the gateway/endpoint level with a mocked LLM, no `OPENAI_API_KEY` (per §6.3 / §6.4) — same discipline as the backend.
- **DOM rendering / visuals:** verified manually / via Preview tooling (screenshots), NOT unit-tested. State this honestly — do not claim coverage of visual rendering.

### 12.8 Operator Console (Phase 15)

The Operator Console (`/console` — [ADR-0011](adr/0011-upload-separate-from-import.md), [ADR-0012](adr/0012-delete-inert-filed-answers-only.md)) is a second vanilla single-file page (§12.1) sharing the reader's design tokens via a `shared.css`. It drives the existing lifecycle endpoints; it is a presentation shell with no business logic (§12.5) and inserts all server-derived content via `textContent` (§12.4) — including raw Markdown in the read-only resource browser, which is shown verbatim, never rendered to HTML.

- **Batch Ingest is ONE call, never a per-file loop.** `ingest_sources` resolves cross-source slug collisions via an in-memory `used_slugs` set scoped to a **single call** (`resolve_slug_collision`, "within a single batch call"). The console MUST send a drop batch (capped at 5 files) as one `POST /ingest` over the named sources; more than 5 files are chunked into **sequential** single calls ("Batch k/N"). A client loop of single-source `/ingest` calls resets `used_slugs` and silently overwrites a colliding slug — breaking the "a Section is never silently overwritten" guarantee (#54). This is a correctness invariant, not a performance choice.
- **No fake progress.** Without SSE the client has no intra-call visibility. Show an indeterminate indicator + status label for single blocking calls (Import / Index / Lint / Ingest-within-a-batch). A determinate percentage or counter appears **only** where the client owns the count ("Batch k/N", "k/N files"). A time-eased fake percentage is a drift signal.
- **Destructive actions are server-gated.** Promote (`/qa/{slug}/promote`) and Discard (`DELETE /qa/{slug}`, inert-only, refuses `status: live`) are server decisions; the client does not replicate the inert/live eligibility policy (§12.5).
- **Lint-remediation buttons drive existing lifecycle endpoints, grouped by Lint Axis (ADR-0023).** Direct Remediations wire to `POST /ingest` (C6/C3 Re-ingest — **C3 MUST send `force:true`**, else hash-skip no-ops the retry into a false fix) and `DELETE /qa` (C10 Discard); a Freshness "Re-ingest all" is the single batch call over the named sources (the #54 one-call invariant above), never a per-finding loop. **Authored-tier findings (Coherence/Coverage) get no batch and no one-click write** — they are validated write-back (ADR-0020/ADR-0023), out of tier-A scope. While a remediation is in flight, the triggering and sibling buttons are disabled (an in-flight guard prevents double-submit) and a `beforeunload` confirm guards accidental navigation; the fix is durable server-side across disconnect (sync `/ingest` under `_index_lock`), so the notice is a **soft** "runs in the background; re-run Lint to see the result", never a false "don't refresh or you'll corrupt it". Progress stays indeterminate + a source-count estimate (no fake %, per this section), and the report re-lints (`include_c5=false`) on completion.

### 12.9 Multi-turn UI (Phase 11 — Conversation Memory)

These encode ADR-0013 in the reader chat UI — they are session-correctness invariants:

- **Session id hold.** The client reads `done.session` from every `done` event and stores it in a local variable. The id is never shown to the user. On subsequent requests, the client echoes it via `?session=<id>` in the query string. No manual id entry by the user — the server mints the UUID on the first request.
- **Session id persistence across toggle.** A stack toggle (Wiki ↔ RAG) is a fresh `POST /chat/stream?stack=<new>` request, but it MUST echo the same `?session=<id>`. Toggling stacks does NOT reset the session; the Conversation Store is keyed by `session_id` only (ADR-0013). The session variable is only reset when the user explicitly starts a new conversation.
- **`status:{phase:"rewriting"}` indicator.** When the gateway emits `status:{phase:"rewriting"}`, the client renders a status indicator (e.g. "understanding your question…"). This event is emitted only on turn 2+ (passthrough on turn 1). The indicator clears when the next event (`sources`) arrives. Render via `textContent` only (§12.4).
- **Turn 1 has no `status:rewriting`.** Turn 1 (empty history / no `?session=`) skips Query Rewriting. The client must not render the rewriting indicator on turn 1.
- **All multi-turn state is server-side.** The client holds only the session id; it never re-implements history tracking, reference resolution, or query rewriting. Those are gateway concerns (§12.5).
