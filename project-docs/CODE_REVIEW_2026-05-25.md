# Code Review — `markdown_kb/` against `CODING_STANDARD.md`

- **Date:** 2026-05-25
- **Reviewer:** Claude Opus 4.7 (assistant)
- **Scope:** `markdown_kb/` (app + tests + `pyproject.toml`). `vector_rag/` excluded per user request.
- **Standard:** `project-docs/CODING_STANDARD.md`
- **Disposition:** Report only. No code changes; the user decides which findings to act on.

## Executive summary

`markdown_kb/` is a well-disciplined codebase. The deep-module split (`indexer.py`, `retrieval.py`), pre-LLM Cannot Confirm gate (ADR-0001), atomic-write index persistence, append-only Wiki Log, and inverted test pyramid are all present and correctly applied. **No CRITICAL ADR-invariant violations were found** — none of the `SOURCE_DIRS` / `Section.metadata` / `wiki/log.md`-tracking / pre-LLM gate guarantees are broken.

The findings below are predominantly **HIGH–MEDIUM tier** — they cluster around three themes:

1. **Sentinel-phrase duplication** between production code and tests (`CANNOT_CONFIRM_PHRASE`, the not-indexed message) — the production constant exists but tests bypass it. This is the one finding with a real failure mode (silent drift between prod and test).
2. **Test-fixture DRY** — `REAL_DOCS`, `FakeLLM`, `indexed_corpus` are re-defined across 5–6 test files with subtle variations. A shared `conftest.py` factory would eliminate ~150 lines and one latent bug (the class-attribute trick in `test_persistence.py:46`).
3. **Standard adoption gaps** — no linter / formatter / type-checker yet, so style drift accumulates silently. A 30-minute ruff setup would catch ~3 of the LOW findings automatically and prevent future ones.

The single test that the standard's `§ 6.3` would explicitly fail — *"a test mocks `indexer.search` or any other deep-module entry point"* — **does not exist in the suite**. Mocking is correctly restricted to the LLM. That's the most important assertion in the review.

---

## Methodology

For each file under review, I checked:

1. **§ 4.3 ADR-encoded invariants** — direct grep + read of the cited code sites.
2. **§ 11 Drift signals** — 18-item checklist, applied to every diff candidate.
3. **§ 1 Style** — imports order, naming, layout, docstrings, type hints, comments.
4. **§ 2 Architecture** — layer boundaries, LangChain type containment, deep-module test.
5. **§ 3 Domain rules** — vocabulary matches `CONTEXT.md`; reserved terms not consumed.
6. **§ 4 Error handling** — fail-fast, OpenAI exception mapping, validation boundaries.
7. **§ 5 Logging** — single channel, line format, bounded summaries.
8. **§ 6 Testing** — Mock LLM not indexer, brittle assertions, live-test discipline.
9. **§ 7 Dependencies** — `pyproject.toml` consistency; no `requirements.txt` leakage.

Findings are sorted by severity. Each cites the exact file + line and the standard section it relates to.

---

## CRITICAL — none

No findings at this tier. All ADR-encoded invariants hold.

---

## HIGH — sentinel & dispatch drift

### H1. `CANNOT_CONFIRM_PHRASE` is duplicated as a literal across 3 test files

**Sites:**
- `markdown_kb/app/retrieval.py:138` — `CANNOT_CONFIRM_PHRASE = "I cannot confirm from the knowledge base."` (the constant)
- `markdown_kb/tests/test_chat_errors.py:32` — duplicated literal
- `markdown_kb/tests/test_chat_fallback.py:24` — duplicated literal
- `markdown_kb/tests/test_chat_grounded.py:314` — duplicated literal (inline `in` check)

**Standard:** § 3.3 ("Constants for sentinel strings") explicitly forbids this — any literal string with semantic meaning appearing more than once gets a module-level constant.

**Failure mode:** if a future implementer changes the phrase in `retrieval.py` (which ADR-0001 forbids, but a typo or rename could happen), the tests would silently keep passing because they assert against their own hardcoded copy, not against production. The PR that broke ADR-0001 would land green.

**Fix:** in each test file, replace the local literal with `from app.retrieval import CANNOT_CONFIRM_PHRASE`.

---

### H2. The not-indexed message is a sentinel without a constant

**Sites:**
- `markdown_kb/app/retrieval.py:80` — hardcoded inline: `"The knowledge base has not been indexed yet. Call POST /index first."`
- `markdown_kb/tests/test_chat_fallback.py:248` and `test_persistence.py:213` — both assert via lowercase substring match: `"not been indexed" in body["answer"].lower()`

**Standard:** § 3.3 same as H1. This is the second sentinel phrase the system returns; it deserves the same treatment as `CANNOT_CONFIRM_PHRASE`.

**Bonus failure mode:** the tests only check `"not been indexed"` substring (case-insensitive). A future implementer could rewrite the message to `"Index missing. Call POST /index."` and the tests would fail loudly — but they'd fail loudly on the substring, not on a constant-equality mismatch, which is a weaker signal. With a constant, the test is `assert body["answer"] == NOT_INDEXED_MESSAGE`, which fails on any drift at all.

**Fix:** promote `"The knowledge base has not been indexed yet. Call POST /index first."` to `NOT_INDEXED_MESSAGE` in `retrieval.py`; tests import + assert equality.

---

### H3. `parse_markdown` swallows `Exception` in frontmatter parsing

**Site:** `markdown_kb/app/indexer.py:181`

```python
except (ValueError, ImportError, Exception):
    # If YAML parse fails or PyYAML not installed, treat as no frontmatter
    body = raw
```

**Problems:**
1. `Exception` in the tuple already covers `ValueError` and `ImportError` — the explicit names are dead.
2. More seriously: catching bare `Exception` masks **any** bug in the frontmatter branch (e.g. a typo on `raw.index(...)`, a `KeyError` on `metadata.get(...)`). Silent failures that drop frontmatter without anyone noticing.
3. Per § 5.1 the silent-failure case (PyYAML not installed but frontmatter present) should emit a `parse_warning` log entry. Today it doesn't.

**Standard:** § 4.1 (fail fast on data corruption) — corrupt frontmatter is a data-state condition the user wants to know about, not silently absorb.

**Fix (minimal):** `except (yaml.YAMLError, ImportError, ValueError):` — explicit list, drop bare `Exception`. Add a `log_event("parse_warning", f"frontmatter parse failed in {filename}")` in the except block.

**Fix (better):** since `Section.metadata` is reserved for the future Wiki layer (PRD § Section dataclass shape), make `pyyaml` a real dependency now — `uv add pyyaml` — and drop the try-import dance. The cost is a small wheel; the gain is the ImportError branch disappears.

---

### H4. `FakeLLM` in `test_persistence.py` mutates a class attribute instead of an instance attribute

**Site:** `markdown_kb/tests/test_persistence.py:43-51`

```python
def invoke(self, messages: list):
    class _Resp:
        pass
    _Resp.content = (
        f"Approved refunds are processed within 5-7 business days. "
        f"[Source: {self.source_id}]"
    )
    return _Resp()
```

This assigns to `_Resp.content` (a class attribute), then constructs `_Resp()`. Today every call creates a fresh `_Resp` class inside the method scope, so the bug is latent — but if this is ever refactored to define `_Resp` outside the method (a common cleanup), every test will see the *last* assignment's content, not their own. That's a several-hours-to-debug class of bug.

**Compare:** `test_chat_grounded.py:51-57` and `test_chat_errors.py:104-109` correctly assign `_Resp.content` only as a class attribute (no instance assignment) and rely on `_Resp` being scoped to the method. Both patterns are ugly, but neither has H4's specific landmine.

**Fix:** use a single `FakeLLMResponse(NamedTuple)` or `@dataclass(frozen=True)` in `conftest.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class FakeLLMResponse:
    content: str
```

then return `FakeLLMResponse(content=...)` from `invoke`. One canonical shape, no surprises.

---

### H5. Test fixtures duplicated across files

**Sites:** `REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"` appears verbatim in:

- `test_chat_grounded.py:21`
- `test_chat_errors.py:30`
- `test_chat_fallback.py:21`
- `test_chat_live.py:23`
- `test_indexing.py:33`
- `test_persistence.py:26`

Same for `indexed_corpus` (the build-index-then-clean-up fixture), defined separately in `test_chat_grounded.py`, `test_chat_errors.py`, `test_chat_fallback.py` — three near-identical copies (`tmp_path` + `monkeypatch INDEX_PATH` + `monkeypatch LOG_PATH` + `build_index(REAL_DOCS)` + cleanup).

**Standard:** No explicit rule on fixture DRY, but § 2.4 (no reaching across modules for the same thing) and § 6.5 (fixtures in `conftest.py`) both point at this.

**Fix:** lift `REAL_DOCS` and `indexed_corpus` into `tests/conftest.py`. Net deletion ~80 lines + one less place for the FakeLLM-divergence bug (H4) to live.

---

## MEDIUM — style / consistency

### M1. Missing module docstrings on shallow modules

**Sites:**
- `markdown_kb/app/routes.py` — starts directly with `from fastapi import APIRouter`
- `markdown_kb/app/schemas.py` — starts directly with `from pydantic import BaseModel`
- `markdown_kb/app/main.py` — starts with `from dotenv import ...` (no module docstring)

**Standard:** § 1.6 "Every module: triple-quoted docstring at the top."

**Fix:** one-line docstring each. Example for `routes.py`: `"""HTTP wiring for /health, /index, /chat. No domain logic."""`

---

### M2. Missing return-type annotations on route handlers

**Site:** `markdown_kb/app/routes.py:10-23`

```python
@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/index", response_model=IndexResponse)
def index_docs():
    ...
```

**Standard:** § 1.7 — "All public function signatures: type-hint every parameter and the return type." FastAPI infers from `response_model=`, but the function signature itself documents intent without making the reader cross-reference.

**Fix:**

```python
def health() -> dict[str, str]: ...
def index_docs() -> IndexResponse: ...
def chat(req: ChatRequest) -> dict: ...    # query() returns dict, not ChatResponse
```

(Note: `chat()` currently returns `query()`'s raw dict, not a `ChatResponse` instance — FastAPI serializes via `response_model`. This works but is worth a comment.)

---

### M3. `@app.on_event("startup")` is deprecated in FastAPI ≥ 0.93

**Site:** `markdown_kb/app/main.py:10-12`

```python
@app.on_event("startup")
def load_persisted_index():
    load_index_json()
```

You're on FastAPI 0.136.3, which emits a `DeprecationWarning` for this pattern. The modern equivalent is the lifespan context manager:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index_json()
    yield

app = FastAPI(title=..., lifespan=lifespan)
```

**Standard:** not directly covered, but § 8.2 (mypy) and the spirit of "use current best practice" applies.

**Fix:** small refactor; preserves test semantics (`TestClient` as context manager still fires the hook).

---

### M4. `Section.to_dict()` reinvents `dataclasses.asdict`

**Site:** `markdown_kb/app/indexer.py:78-87`

```python
def to_dict(self) -> dict:
    return {
        "id": self.id,
        "file": self.file,
        ...
    }
```

**Risk:** adding a new field to `Section` requires editing `to_dict()` separately. Forgetting silently truncates the JSON snapshot.

**Fix:** `from dataclasses import asdict` and return `asdict(self)`. One line. Field additions become free.

---

### M5. `_apply_grounding_check` and `_write_chat_log` have untyped `list` parameters

**Site:** `markdown_kb/app/retrieval.py:201, 247`

```python
def _apply_grounding_check(
    question: str,
    prompt_text: str,
    answer: str,
    sources: list,
) -> tuple[str, list]: ...

def _write_chat_log(question: str, ranked: list) -> None: ...
```

**Standard:** § 1.7 — modern generics for elements. `sources: list[dict]` and `ranked: list[tuple[Section, float]]` would document intent and let mypy catch wrong-element types.

**Fix:** add element types. A `TypedDict` for `sources` items would be even better since they have a fixed shape.

---

### M6. `_SCORE_THRESHOLD` is read at import time, not request time

**Site:** `markdown_kb/app/retrieval.py:29`

```python
_SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", "0.5"))
```

**Behavior:** runtime changes to `KB_SCORE_THRESHOLD` after server start have no effect. Tests work around this by monkeypatching the resolved value (`test_chat_fallback.py:205`), not the env var.

**Standard:** not directly covered, but worth flagging as a documented constraint. PRD § Index lifecycle says "I can tune the Cannot Confirm gate without redeploying code" — this is satisfied for *restart* tuning, not *runtime* tuning.

**Fix (if it matters):** wrap in a function: `def _score_threshold() -> float: return float(os.getenv("KB_SCORE_THRESHOLD", "0.5"))`. Trivial perf cost; semantically clean. Or: leave as-is and add a comment that the value is fixed at boot.

---

### M7. Dead import in `test_chat_grounded.py`

**Site:** `markdown_kb/tests/test_chat_grounded.py:11` — `import re` is never used in the file.

**Standard:** § 8.1 (ruff would catch this).

**Fix:** delete the line. Also: dead `LOG_LINE_RE` constant at the top of `test_logger.py:11-13` (defined but never referenced — the test inlines the regex at line 39).

---

### M8. `parse_markdown`'s broad-exception pattern is mirrored in test setup

Less a single site than a theme: the codebase trusts `except (..., Exception)` in a few places. M3-style narrowing would help mypy + readers.

---

## LOW — nice-to-have

### L1. Issue-number references in production comments

**Sites:** `markdown_kb/app/retrieval.py:106, 144` — both contain `(issue #5)`.

**Standard:** § 1.8 — "Don't write comments that ... reference the current task ('added for Slice 5')." The git log carries this; production code shouldn't.

**Fix:** replace with `# Per ADR-0001 § Consequences` or drop entirely (the `_call_llm_with_error_handling` docstring already describes the mapping).

---

### L2. Repeated truncation idiom across 4 sites

**Site:** `question[:60].replace('"', "'")` appears at `retrieval.py:76, 92, 152, 220, 254`.

The standard (§ 5.3) explicitly says "copy it verbatim" — this is in tension with general DRY. The argument **for** keeping the duplication: each call site is a different log kind with slightly different framing; a generic `_truncate_for_log(text)` helper hides where the constant `60` comes from.

The argument **against**: 5 copies of an identical expression is the point at which "copy verbatim" tips into noise.

**Recommendation:** I'd revise the standard rather than the code. Add a `_summarize_query(text: str, max_chars: int = 60) -> str` helper at the top of `retrieval.py` and update § 5.3 to point at it. (Self-aware finding: my own standard is wrong here.)

---

### L3. `STOP_WORDS` as `set` not `frozenset`

**Site:** `markdown_kb/app/indexer.py:36-57`

`frozenset` signals immutability and is constant-foldable. Cosmetic.

---

### L4. `main.py` startup function lacks a docstring

**Site:** `markdown_kb/app/main.py:11-13`

```python
@app.on_event("startup")
def load_persisted_index():
    load_index_json()
```

One-line docstring expected per § 1.6. Will become moot once M3 lifespan migration lands.

---

### L5. `ChatRequest.query: str` allows empty strings

**Site:** `markdown_kb/app/schemas.py:9-11`

```python
class ChatRequest(BaseModel):
    query: str
```

An empty `query` produces empty BM25 tokens → no results → Cannot Confirm fallback. **The current behavior is correct** (empty query is treated as "ask nothing, get cannot-confirm"). But explicit `Field(min_length=1)` would (a) reject the case at validation boundary per § 4.4 and (b) save the round-trip through retrieval + log entry for a degenerate input.

Stay-as-is is also defensible; small judgment call.

---

### L6. `_make_*_error` helpers in `test_chat_errors.py` could be parametrized

**Site:** `test_chat_errors.py:40-73` — 4 helper functions, near-identical body.

Cosmetic. The current form is more readable than a parametrized version; leave it.

---

## Drift-signal cross-check (§ 11)

For each of the 18 drift signals in the standard, the verdict on `markdown_kb/`:

| # | Signal | Status |
|---|---|---|
| 1 | Test mocks `indexer.search` or deep-module entry point | ✅ Not violated. Only `_llm` / `get_llm` are mocked. |
| 2 | New domain term appears in code without `CONTEXT.md` entry | ✅ All identifiers (`Section`, `Source`, `Citation`, `query`, `search`) match. |
| 3 | New module imports `langchain*` outside `retrieval.py` | ✅ Confined to `retrieval.py:19-20`. |
| 4 | Branch returns paraphrase of "Cannot Confirm" instead of constant | ⚠️ See H1 — production uses the constant, tests duplicate the literal. |
| 5 | `SOURCE_DIRS` reduced to single `Path` | ✅ Still `list[Path]` (`indexer.py:32`). |
| 6 | `Section.metadata` removed because "unused" | ✅ Field present (`indexer.py:76`). |
| 7 | `requirements.txt` reappears | ✅ Deleted in commit `7e132a8`; only `pyproject.toml` remains. |
| 8 | `print()` in production code | ✅ Grep clean. |
| 9 | Second `@pytest.mark.live` test | ✅ Exactly one (`test_chat_live.py:32`). |
| 10 | Test asserts absolute BM25 score | ✅ Only ranking order asserted (`test_chat_grounded.py:259-268`). |
| 11 | HTTP error mapping drifts from § 4.2 | ✅ Mapping exactly matches the table (`retrieval.py:159-185`). |
| 12 | `wiki/log.md` added to `.gitignore` | ✅ Not in `.gitignore`. |
| 13 | ADR invariant broken without paired ADR | ✅ All four invariants in § 2.5 hold. |
| 14 | New dependency added by hand-editing `pyproject.toml` | ✅ `uv.lock` is consistent (verified via `uv sync` no-op). |
| 15 | LangChain message types leak out of `retrieval.py` | ✅ `HumanMessage`/`SystemMessage` only inside `retrieval.py`. |
| 16 | `wiki`, `ingest`, `hot_cache` consumed as local variable | ✅ `wiki_dir` (test fixtures) is borderline but used as a filesystem path, not as the conceptual Wiki. Acceptable. |
| 17 | Module-level docstring missing or "what" instead of "intent" | ⚠️ See M1 — `routes.py`, `schemas.py`, `main.py` lack module docstrings. |
| 18 | Function-scope import lacks circular-dep comment | ✅ `indexer.py:166` (`from .logger import log_event`) is correctly justified with `# avoid circular dependency at module level`. |

**Net:** 16/18 clean; 2 partial (H1 + M1). Both are addressable with small mechanical changes.

---

## Positive patterns worth preserving

Calling these out so future contributors know which patterns are *deliberate excellence*, not coincidence.

### P1. `retrieval.py` is the reference example of a deep module
- Public `query()` is 70 lines including comments, with clear flow.
- All conditional logic + error mapping + grounding heuristic encapsulated in `_call_llm_with_error_handling`, `_apply_grounding_check`, `_is_grounded`, `_write_chat_log`.
- Zero LangChain types leak past the module boundary.

### P2. `parse_markdown`'s embedded 10-rule docstring (`indexer.py:122-163`)
Future me can extend the parser for h3/h4 nesting and know exactly which 10 rules must remain invariant. This is the platinum standard the coding-standard § 1.6 cites by reference.

### P3. Atomic write in `write_index_json` (`indexer.py:326-361`)
`tempfile.mkstemp` in the same directory + `os.fdopen` + `os.replace` + tmp cleanup on exception. Textbook correct. Single FS guarantee for POSIX rename atomicity is satisfied.

### P4. Sentinel-LLM in `test_chat_fallback.py:31-41`
```python
class SentinelLLM:
    def invoke(self, messages):
        self.call_count += 1
        raise AssertionError("LLM must NOT be invoked when the pre-LLM Cannot Confirm gate fires.")
```
Beautiful enforcement of the pre-LLM gate: the test physically cannot pass unless the gate fires. ADR-0001 made executable.

### P5. The live test's shape-only assertions (`test_chat_live.py:76-99`)
Status code, `"[Source:"` substring, non-empty list, source filename prefix. No prose words asserted. Survives model updates.

### P6. `conftest.py`'s opt-in live-test handling
```python
def pytest_collection_modifyitems(config, items):
    if "live" in marker_expr:
        return
    skip_live = pytest.mark.skip(reason="live test — run with: pytest -m live")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)
```
Cleaner than the more common `pytest.skip(...)` inside the test body. Discovery-time skip; test runs show as skipped (not collected then bailed).

### P7. `build_index(docs_dir=DOCS_DIR)` identity comparison (`indexer.py:428`)
```python
if docs_dir is not DOCS_DIR:
    scan_dirs = [docs_dir]
else:
    scan_dirs = SOURCE_DIRS
```
Uses `is not` for sentinel detection — correct. Subtle, but well-commented.

### P8. Function-scope import for cycle-break (`indexer.py:166`)
```python
def parse_markdown(path: Path) -> list[Section]:
    """..."""
    # Import here to avoid circular dependency at module level; logger imports
    # nothing from indexer.
    from .logger import log_event
```
Standard § 1.3 cites this exact pattern as the only acceptable use of function-scope imports.

---

## Recommended next steps (in priority order)

Each is a self-contained `chore:` or `refactor:` commit, < 30 minutes each.

1. **Fix H1 + H2** (sentinel constants) — promote `NOT_INDEXED_MESSAGE`; import `CANNOT_CONFIRM_PHRASE` in tests. ~15 minutes; eliminates the silent-drift failure mode.
2. **Fix H4** (`FakeLLMResponse` dataclass in conftest.py) + **H5** (lift `REAL_DOCS` and `indexed_corpus` to conftest.py). ~30 minutes; net deletion ~100 lines.
3. **Fix H3** (narrow exception + log warning + decide on PyYAML dep). ~15 minutes.
4. **Adopt ruff** (Coding Standard § 8.1) — would auto-catch M7 + L3 + future drift. ~20 minutes.
5. **Fix M1 + M2 + M3** (module docstrings, route return types, lifespan migration). Bundle as one `refactor(markdown_kb): modernize FastAPI app skeleton` commit. ~20 minutes.
6. **Fix M4 + M5 + M6** (asdict, typed list params, threshold helper). Bundle as one `refactor: tighten type hints + reduce hand-rolled patterns`. ~20 minutes.
7. **L1** — strip issue-number comments. 2 minutes.
8. **L2** — *revise the standard* to allow the truncation helper; then refactor (or accept the standard as-is). Discuss before acting.

Everything else (L3–L6) can wait until ruff is in place and surfaces them automatically.

---

## What this review didn't cover

- **`vector_rag/`** — explicitly excluded per scope choice.
- **Root config files** (`.gitattributes`, `.gitignore`, `.python-version`, root `pyproject.toml`) — excluded per scope choice; reviewed during the uv migration session and assumed clean.
- **`docs/*.md`** (the bot's runtime KB content) — out of scope (this is data, not code).
- **`project-docs/`** — out of scope (governance, not code).
- **Test fixture markdown files** under `tests/fixtures/docs/` — not opened; assumed conformant with `tests/README.md` rules.
- **CI / GitHub Actions** — none present today.
- **Performance** — not assessed. PRD says BM25 rebuild < 0.1s for the sample corpus; no profiling performed.
- **Security** — not assessed beyond the standard's scope. Notably: `.env` is gitignored ✅; no SQL / shell injection surface; OPENAI_API_KEY isn't logged. Full security review is a separate skill.
