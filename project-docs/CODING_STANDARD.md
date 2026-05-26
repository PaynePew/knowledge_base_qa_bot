# Coding Standard — `knowledge_base_qa_bot`

How code is shaped in this repo, what conventions hold across modules, what design patterns are in play, and what tooling enforces them. This file is **not the spec** — that lives in `PRD.md` / `ADR.md` / `CONTEXT.md`. This file is the **consistency layer** that keeps the spec implementable across slices without drifting.

## 0. Reading order

In every fresh session, read in this order before writing code:

1. `CLAUDE.md` — agent skills configuration, workflow triggers
2. `CONTEXT.md` — vocabulary (Source, Section, Citation, Cannot Confirm, …)
3. `project-docs/adr/*.md` — architectural decisions
4. `project-docs/prd.md` — what we're building and why
5. **this file** — when about to write or review code
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

**Out of reviewer scope** (these are author-time / orchestrator-time):
- **§0** (Reading order, Authority, this section)
- **§8 Tooling recommendations** — adopting these is a separate `chore:` commit, not a per-slice review concern
- **§9 Commits and review** — the commit message rules are checked by the implementer; the reviewer-checklist subset (§9.2) is duplicated in agents/review.md's review process for self-containment

**Citation discipline when flagging**: when the reviewer flags an issue, it must cite the section by number (e.g. "§3.1 vocabulary drift — `Document` used at `app/indexer.py:42`, should be `Source`"). Do NOT dump the section's full prose into the report — the section number is enough for the human to look up.

**Budget**: an active review typically loads §3 + §4 + §5 + §11 eagerly (~80 lines combined) plus 0-2 conditional sections on demand. Total injection is well under 200 lines — fits cleanly in any sub-agent's context window.

---

## 1. Style

### 1.1 Line endings

LF everywhere, enforced by `.gitattributes`. Windows-only scripts (`.bat`, `.cmd`, `.ps1`) keep CRLF. Mixed-EOL diffs are a process bug — see the `.gitattributes` comment block.

### 1.2 Indentation, line length

- 4 spaces, never tabs.
- Soft target 88 chars (black default), hard ceiling 120. Long docstrings, log format strings, and URLs in comments may exceed.

### 1.3 Imports

- **`from __future__ import annotations` is the first import** in every `.py` file (after the module docstring). Lets `list[X]`, `dict[K, V]`, `X | None` work even when later type-resolution targets <3.10 environments. Already consistent across the codebase.
- Grouping (one blank line between groups):
  1. stdlib
  2. third-party
  3. local (`from . import …` or `from .foo import …`)
- Within each group, sort alphabetically.
- Prefer **relative imports within a single package** (`from .indexer import Section`); absolute imports for cross-package.
- Function-scope imports are allowed **only** to break circular dependencies. Always paired with a comment explaining why (see `indexer.py:166` — `from .logger import log_event` inside `parse_markdown` to avoid the import cycle).

### 1.4 Naming

| Kind | Convention | Example |
|---|---|---|
| Functions / variables / modules | `snake_case` | `build_index`, `ranked_sections` |
| Classes (incl. dataclasses) | `PascalCase` | `Section`, `ChatRequest` |
| Module-level constants & singletons | `SCREAMING_SNAKE_CASE` | `SYSTEM_PROMPT`, `INDEX_PATH`, `STOP_WORDS` |
| Module-private state / helpers | leading `_` | `_llm`, `_index_lock`, `_apply_grounding_check` |
| Pytest fixtures | `snake_case`, descriptive | `tmp_docs`, `tmp_kb` |

**Domain identifiers MUST match `CONTEXT.md` vocabulary** — see § 4.1. This is a hard rule, not a preference.

### 1.5 File layout (within a `.py`)

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

`# ---` divider lines are **mandatory** once a module has ≥4 logical sections. They make `grep -n "# ---"` a free table of contents. `indexer.py` is the reference example.

### 1.6 Docstrings (PEP 257)

- **Every module:** triple-quoted docstring at the top. Intent + ADR/PRD reference.
- **Every public function / method:** triple-quoted docstring describing intent, non-obvious args, return shape, raised exceptions.
- **Private helpers:** one-line docstring only if the name is not self-explanatory.
- **Rule-based functions** (e.g. `parse_markdown`): embed the rule spec inline in the docstring, numbered `1.` through `N.`, then reference rules in code with `# Rule N:` comments. See `indexer.py:122`.
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
- A guard against future drift (`# ADR-0003: build_index iterates this list so adding WIKI_DIR needs no signature change`).

**Never** write comments that restate the code or anchor to the current task ("added for Slice 5", "used by routes.py"). Git log and call graph already convey those.

---

## 2. Architecture

### 2.1 Deep modules (Ousterhout)

Modules are organized as **deep modules**: small public interface, large private implementation. The PRD declares `indexer.py` and `retrieval.py` as deep modules. Every new module passes the deep-module test:

> Could a reasonable user of this module use it correctly knowing **only** the names + signatures of its public functions, with no need to read the implementation?

If no — refactor the boundary before merging.

### 2.2 Module size

- Up to ~500 lines per `.py` is acceptable.
- Beyond ~500, split **only when a clear sub-responsibility falls out**. Do not split prophylactically (PRD lists `prompt_builder.py` as a deliberate extraction so its output can be asserted in isolation without mocking the LLM — that's the bar).
- Per **ADR-0002**: do NOT extract a pluggable `Retriever` protocol until BOTH `markdown_kb` and `vector_rag` are end-to-end working.

### 2.3 Layers within `markdown_kb/app/`

| Module | Depth | Owns |
|---|---|---|
| `main.py` | shallow | FastAPI lifecycle, `.env` loader, startup hooks |
| `routes.py` | shallow | HTTP wiring only; no domain logic |
| `schemas.py` | shallow | Pydantic request/response shapes; no behavior |
| `prompt_builder.py` | shallow | SYSTEM_PROMPT + `build_prompt(question, ranked_sections)` |
| `logger.py` | shallow | Wiki Log writer (`log_event`) |
| `indexer.py` | **deep** | Section parsing, BM25 index, persistence, concurrency, atomic write |
| `retrieval.py` | **deep** | Query orchestration, threshold gate, error mapping, grounding check |

**Deep modules own all conditional logic. Shallow modules wire them together.** Adding business logic into `routes.py`, `schemas.py`, or `main.py` is a violation.

### 2.4 Forbidden cross-module patterns

- **No reaching into private state of another module.** `indexer.sections` is treated as public (and is read by `retrieval.py`); `indexer._index_lock` is not.
- **No circular imports.** When `indexer.py` needs `logger.py`, the import is at the top. When the cycle is unavoidable (e.g. `parse_markdown` → `log_event`), use a function-scope import + a comment.
- **No LangChain types leak past `retrieval.py`.** `HumanMessage`, `SystemMessage`, `ChatOpenAI` stay inside `retrieval.py`. Routes / schemas / indexer see only Python primitives and Pydantic models. The PRD lists this exactly — `prompt_builder.py` was extracted precisely so its output can be asserted *as a string* without touching LangChain types.

### 2.5 Future-proofing patterns (mandatory — ADR-encoded)

| Pattern | Code site | ADR / PRD |
|---|---|---|
| `SOURCE_DIRS: list[Path]` (not a single `Path`) | `indexer.py:32` | ADR-0003 |
| `Section.metadata: dict` reserved even when unused | `indexer.py:76` | PRD § Section dataclass shape |
| `wiki/log.md` committed (NOT gitignored) | `.gitignore` | PRD US #23 |
| Pre-LLM Cannot Confirm gate before any LLM call | `retrieval.py:88` | ADR-0001 |

Changing any of these requires a **new ADR superseding the current one**. Reviewers must fail any PR that breaks an invariant without a paired ADR.

### 2.6 Concurrency

- The index swap is the only contended operation. Hold `_index_lock` **only** when assigning to the module-level `sections` list (see `build_index` and `load_index_json` in `indexer.py`).
- Readers do not lock. Mid-rebuild readers see the previous snapshot until the swap completes.
- **Persistent state writes are atomic**: write to `<file>.tmp`, then `os.replace(...)`. Never write to the target file directly. See `write_index_json` (`indexer.py:326`).
- Beyond a single FastAPI worker, this model breaks. Multi-worker is **post-prototype**; will need an external lock (filesystem flock, redis, or DB). Do not refactor proactively.

### 2.7 State management

- Module-level mutable globals (`sections`, `doc_freq`, `_llm`) are **acceptable for the prototype's single-process model**. They are explicitly designed so tests can swap them via `monkeypatch`.
- Singleton LLM clients use lazy init via `get_llm()` / `get_retry_llm()`. This lets tests stub them without instantiating a real OpenAI client.
- When this codebase outgrows the single-process model, lift state onto `app.state` or a DI container. **Do not refactor today.**

---

## 3. Domain rules

### 3.1 Vocabulary discipline (mandatory)

Code identifiers MUST use the `CONTEXT.md` vocabulary verbatim. Concrete rules:

| Concept | Use | Don't use |
|---|---|---|
| A markdown file the bot indexes | `Source` | `Document`, `Article`, `Doc` |
| The retrieval unit | `Section` | `Chunk`, `Paragraph`, `Leaf section` |
| The persisted inverted index | `Section Index` | `Index` (reserved for `Wiki Index`), `BM25 Index` |
| A `filename#heading-slug` reference | `Citation` | `Source` (that's the file), `Reference` |
| A strictly-grounded reply | `Grounded Answer` | "sourced answer", "cited answer" |
| The literal sentinel string | `Cannot Confirm` (constant: `CANNOT_CONFIRM_PHRASE`) | Any paraphrase |

When you need a new domain concept, propose the term via `/grill-with-docs` **before** naming a class/function for it. Inventing vocabulary directly in code creates drift that's expensive to undo.

### 3.2 Reserved terms are off-limits as variable names

The reserved-but-not-yet-implemented terms in `CONTEXT.md` are **off-limits as variable names today**: `Wiki`, `Wiki Index`, `Hot Cache`, `Wiki Log`, `Source Template`, `Lint Pass`, `Ingest`, `Grounding Check`, `Query Rewriting`, `Conversation Store`. Using `wiki` or `ingest` as a local helper name silently consumes the namespace.

### 3.3 Constants for sentinel strings

Any literal string with semantic meaning that appears more than once gets a module-level constant:

- ✅ `CANNOT_CONFIRM_PHRASE = "I cannot confirm from the knowledge base."` (`retrieval.py:138`)
- ✅ `SYSTEM_PROMPT` (`prompt_builder.py`)
- ❌ Inline `"I cannot confirm from the knowledge base."` in tests or routes — use the constant.

---

## 4. Error handling

### 4.1 Fail fast on data corruption

A corrupt `.kb/index.json` at startup **raises** and prevents the server from starting (`load_index_json`). Silently serving stale or wrong data is worse than not serving. Apply the same rule to every persistent-state load you add.

### 4.2 OpenAI exception mapping (HTTP status)

Mandatory mapping in `_call_llm_with_error_handling`:

| Exception | HTTP | Log kind |
|---|---|---|
| `APITimeoutError`, `RateLimitError` | 503 | `openai_transient` |
| `AuthenticationError` | 500 | `openai_auth` |
| Any other `APIError` subclass | 500 | `openai_api` |

Every branch emits a `chat_error` log entry with the right `kind=` tag. Use `raise HTTPException(...) from exc` to preserve the exception chain. Never bare-raise.

### 4.3 `Cannot Confirm` is a success, not an error

Per **ADR-0001**:

- Empty / sub-threshold retrieval returns the exact literal phrase with HTTP **200**.
- The LLM is **not called** in this path (pre-LLM gate at `retrieval.py:88`).
- An ungrounded LLM response gets **one** retry at `temperature=0`. If still ungrounded, replace with `CANNOT_CONFIRM_PHRASE` and clear `sources`.

Adding a shortcut that bypasses the gate ("if score is just barely below threshold, send to LLM anyway") is a deliberate ADR-0001 violation. Reviewer must fail it.

### 4.4 Validation at boundaries only

- Request validation: Pydantic does it via `ChatRequest` / `IndexResponse`. **No** defensive re-validation in route handlers.
- Deep-module ↔ deep-module: trust types. Don't `isinstance`-check inputs from your own codebase.

---

## 5. Logging and observability

### 5.1 Single log channel

Every operationally-interesting event goes to `wiki/log.md` via `log_event(kind, summary)`. **No** `print()`, **no** `logging.getLogger(...)`, **no** `sys.stderr.write(...)` in production code.

If you want a debug-only channel, instead either:

- Add a new `kind` to the unified log (e.g. `parse_warning`, `chat_grounding_retry`).
- Or use `pytest.fail` / `assert` for test-only diagnostics.

### 5.2 Log line format

```
## [<ISO-8601 UTC>] <kind> | <summary>
```

- `kind` is `snake_case`. PRD § Log entry conventions enumerates the kinds in use; adding a new one means adding a row there.
- `summary` is `grep`-friendly: KEY=value pairs separated by spaces; query strings double-quoted; never embed newlines.

### 5.3 Summaries are bounded

- Truncate user queries to 60 chars: `question[:60].replace('"', "'")`. This idiom is repeated in `retrieval.py` — when adding a new log site, copy it verbatim.
- Never log API keys, full request bodies, or full document content.
- Score values rounded to 3 decimal places: `round(score, 3)`.

---

## 6. Testing

### 6.1 Inverted pyramid (per `markdown_kb/tests/README.md`)

- **Many** integration tests (`TestClient` + fake LLM) covering PROMPT.md verification cases.
- **Some** component tests for `parse_markdown`, `build_index`, BM25 ranking order.
- **Few-to-zero** unit tests on trivial helpers (`slugify`, `tokenize`).
- **Exactly one** `@pytest.mark.live` smoke test. Opt-in only; auto-skipped via `conftest.py:pytest_collection_modifyitems`.

### 6.2 What to assert vs not

| Assert | Don't assert |
|---|---|
| HTTP status code | BM25 score absolute values (corpus-sensitive, brittle) |
| Response shape (keys, types, list lengths) | LLM output text content beyond shape + `[Source:` markers |
| Exact literal sentinel strings (`CANNOT_CONFIRM_PHRASE`, citation format) | Wall-clock timing |
| Section IDs, ranking order | Specific words in the model's reply |
| Log line presence + structure | Anything that breaks across model updates |

### 6.3 Mock the LLM, not the indexer

- The LLM is the **only** thing that should be replaced with a stub. Use `monkeypatch` on `get_llm` / `get_retry_llm`.
- The indexer always runs against real fixture files under `tmp_docs`. Mocking `indexer.search` masks integration drift — fail any PR that does this.

### 6.4 Live smoke discipline

- Exactly one live test exists today (`test_chat_live.py`). Adding a second is scope creep — push the assertion into a mocked integration test instead.
- A live test asserts **shape** (200, citation pattern present, non-empty sources), **never** specific words. Models update; tests outlive them.

### 6.5 Fixtures

- Per-test isolation via `tmp_path`-derived fixtures (`tmp_docs`, `tmp_kb`, `tmp_wiki` in `conftest.py`).
- Tests that mutate `indexer.sections` MUST restore it via `monkeypatch` (auto-restores) or an explicit teardown.
- Fixture filenames under `tests/fixtures/docs/` deliberately mirror real `docs/` filenames so PROMPT.md cases translate one-to-one.

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

- Pinned to `>=3.11` via `pyproject.toml` (every member) and `.python-version` at the root.
- Upgrading: change `.python-version`, bump `requires-python` in every member's `pyproject.toml`, run `uv sync --all-packages`, re-run pytest.

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

For quick recognition during code review. These are documented here so a new contributor sees the pattern names attached to concrete code sites.

| Pattern | Where | Why this one |
|---|---|---|
| **Deep module** (Ousterhout) | `indexer.py`, `retrieval.py` | Small public surface (`build_index`, `search`, `query`); large private implementation (BM25 math, parsing rules, error mapping, grounding heuristics). |
| **Lazy singleton** | `get_llm()`, `get_retry_llm()` (`retrieval.py:37,48`) | Avoids constructing a real OpenAI client at import time; lets tests stub via `monkeypatch` before first use. |
| **Guard clause / early return** | Pre-LLM Cannot Confirm gate (`retrieval.py:73,88`) | ADR-0001: never hand weak context to the LLM. Two early returns before the prompt is even built. |
| **Atomic write (tmp + rename)** | `write_index_json` (`indexer.py:326`) | Crash mid-write must not leave a half-written index for the next startup. POSIX `os.replace` is atomic on a single filesystem. |
| **Append-only log** | `log_event` → `wiki/log.md` | Karpathy log discipline; survives across crashes; `grep`-able audit trail. |
| **DI via monkeypatch** | Tests stub `get_llm` / `_llm` / `LOG_PATH` | No DI framework; tests use pytest's `monkeypatch` to swap module-level state. The functions return module globals on purpose so this is cheap. |
| **Strategy (deferred / implicit)** | `markdown_kb/` vs `vector_rag/` behind same HTTP contract | ADR-0002 explicitly defers a `Retriever` protocol until both implementations work. The two directories are two strategies; the abstraction is *not* extracted yet. |
| **Repository / in-memory store** | `indexer.sections` + `_index_lock` | Single-process, single-writer; module-level list is the "repository." When this model breaks, the upgrade path is `app.state` or external store. |
| **Adapter** | LangChain `ChatOpenAI` wraps the OpenAI SDK | Provides timeout/retry plumbing; isolated to `retrieval.py` so the rest of the codebase never sees LangChain types. |

Notable patterns **rejected** (do not introduce):

- A `Retriever` protocol / plugin architecture (ADR-0002 — premature today).
- A second log channel beyond `wiki/log.md` (§ 5.1).
- A `Document` or `Chunk` class (§ 3.1).
- A DI container / app-state object (§ 2.7 — only when single-process breaks).

---

## 11. Drift signals (the reviewer's actionable checklist)

When the **review agent** ([`project-docs/agents/review.md`](agents/review.md)) inspects a diff, it walks this checklist top-to-bottom and ticks anything that appears in the actual diff (verified via `git diff`, NOT inferred from the issue body or PRD spec — see Factual discipline in `review.md`).

Each signal has a **severity** that determines the reviewer's action:

- **FAIL** = AC is not actually satisfied (or a hard ADR invariant is broken without a paired new ADR). Reviewer returns `FAIL`; the implementer must revisit. Reviewer does NOT silently fix these.
- **FIX** = small, safe, mechanical refactor the reviewer can apply directly with a `refactor:` commit. Reviewer fixes, commits immediately (per Turn-budget discipline), and lists the commit in "Changes made".
- **FLAG** = correctness or scope concern that needs human judgment. Reviewer notes in "Concerns flagged for human" and does NOT make the change.

### Vocabulary drift (§3.1)

- [ ] **FAIL** — A new domain term appears in code without a `CONTEXT.md` entry, OR a synonym smuggles in for an existing CONTEXT term (e.g. `Document` / `Article` / `Doc` for what should be `Source`).
- [ ] **FIX** — A local variable name consumes a reserved CONTEXT term (`wiki`, `ingest`, `hot_cache`, `wiki_index`, `lint_pass`, `query_rewriting`, `conversation_store`, `grounding_check`, `source_template`).

### Sentinel string drift (§3.3)

- [ ] **FIX** — A branch returns a paraphrase of "Cannot Confirm" instead of the constant `CANNOT_CONFIRM_PHRASE`.
- [ ] **FIX** — An inline `"I cannot confirm from the knowledge base."` literal appears in tests or routes (use the constant).

### Architecture & dependency drift (§2)

- [ ] **FAIL** — A new module imports `langchain` or `langchain_openai` outside `retrieval.py`, OR LangChain types (`HumanMessage`, `SystemMessage`, `ChatOpenAI`) leak through `retrieval.py`'s return values into other modules. (Violates §2.4.)
- [ ] **FAIL** — Business / conditional logic appears in `routes.py`, `schemas.py`, or `main.py` (shallow modules per §2.3). Should be in a deep module.
- [ ] **FAIL** — `SOURCE_DIRS` is reduced to a single `Path` "for simplicity" (violates ADR-0003 + §2.5).
- [ ] **FAIL** — `Section.metadata` is removed because it's "unused" (violates PRD § Section dataclass shape + §2.5).
- [ ] **FAIL** — Pre-LLM Cannot Confirm gate (`retrieval.py:88`) is bypassed when score is "just barely below threshold" (violates ADR-0001 + §4.3).
- [ ] **FAIL** — A `Retriever` protocol / plugin layer is extracted before both `markdown_kb` and `vector_rag` are end-to-end working (premature per ADR-0002 + §2.2).

### Error handling drift (§4)

- [ ] **FAIL** — HTTP error mapping for OpenAI exceptions drifts away from §4.2 (e.g. `RateLimitError` returns 500 instead of 503).
- [ ] **FAIL** — A persistent-state load (e.g. `.kb/index.json`) silently fallbacks to empty on corruption instead of raising (violates §4.1 fail-fast).
- [ ] **FIX** — A handler uses bare `raise` instead of `raise HTTPException(...) from exc` (loses the exception chain per §4.2).
- [ ] **FAIL** — Pydantic boundary validation is re-implemented inside a route handler (violates §4.4).

### Logging drift (§5)

- [ ] **FAIL** — `print()`, `logging.getLogger(...)`, or `sys.stderr.write(...)` lands in production code (violates §5.1 single log channel).
- [ ] **FIX** — A new log site logs full user queries or full document content (violates §5.3 bounded summaries; truncate to 60 chars).
- [ ] **FIX** — A new log site logs unrounded float scores (violates §5.3; use `round(score, 3)`).
- [ ] **FAIL** — A new log `kind=` is used without a corresponding row in PRD § Log entry conventions.

### Testing drift (§6)

- [ ] **FAIL** — A test mocks `indexer.search` or any other deep-module entry point (mock the LLM, not the index; per §6.3).
- [ ] **FAIL** — A second `@pytest.mark.live` test appears (one is the policy; per §6.4).
- [ ] **FAIL** — A test asserts an absolute BM25 score value (corpus-sensitive, brittle; per §6.2 — assert ranking order or shape instead).
- [ ] **FLAG** — A test asserts specific LLM output text content beyond shape + `[Source:` marker (will break across model updates; per §6.2).
- [ ] **FIX** — A test mutates `indexer.sections` without restoring via `monkeypatch` or explicit teardown (per §6.5).

### Dependencies drift (§7)

- [ ] **FAIL** — `requirements.txt` reappears anywhere in the tree (uv is the single source of truth per §7.1).
- [ ] **FAIL** — A new dependency is added by hand-editing `pyproject.toml` instead of `uv add` (lockfile drift risk per §7.2).
- [ ] **FLAG** — A new dependency lacks a one-sentence rationale in the commit message (per §7.2).

### Documentation discipline (§1.6, §1.8)

- [ ] **FIX** — A new module is missing the top-of-file docstring with intent + ADR/PRD reference.
- [ ] **FIX** — A function-scope import lacks a comment explaining the circular-dep workaround.
- [ ] **FIX** — A comment paraphrases obvious code (delete it — only WHY-comments per §1.8).

### Git hygiene

- [ ] **FAIL** — `wiki/log.md` is added to `.gitignore` (violates PRD US #23 + §2.5).
- [ ] **FAIL** — An ADR-0001 / 0002 / 0003 / 0004 / 0005 invariant is broken **without** a paired new ADR superseding it.

The reviewer's job is to spot these and act per the severity. The implementer's job is to not write them in the first place — reading this section before writing code is cheaper than re-doing it after a `FAIL`.
