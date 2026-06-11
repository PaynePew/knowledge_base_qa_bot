# Phase 12 — Live-Verify Runbook (CLI + MCP)

Manual end-to-end verification for Phase 12 (Alternative Interfaces: CLI + MCP).
This is the **one thing the agent cannot do for you** — it needs a real
`OPENAI_API_KEY` and a local Claude Desktop. Everything else in Phase 12 is
shipped, tested (full suite green), and documented.

- **Repo:** `C:\Users\MaxL\work\projects\live_sessions\knowledge_base_qa_bot`
- **Last verified main:** `07bcaa5` (after PRs #219 / #220 — `.env` parity — and
  #221 / #222 — the `kb` / `kb_mcp` `sys.path` bootstraps that make both entry points
  runnable from any cwd)
- **Automated coverage already green:** full `pytest` suite (1069 passed) + `ruff`.
  This runbook covers only the parts that need a real LLM / a real MCP host.

> **Two ground rules (carry them everywhere below):**
> 1. **Run every command from the repo root.** From a subdirectory pytest collects
>    only a *partial* suite and silently under-reports.
> 2. **`.kb/index.json` and `wiki/index.md` are byte-stability invariants (#204).**
>    Don't let a verify step mutate them. Sign-off check at the bottom.

---

## 0. Prerequisites (already in place — just confirm)

`.env` at the repo root already holds a real `OPENAI_API_KEY`. Confirm:

```powershell
cd C:\Users\MaxL\work\projects\live_sessions\knowledge_base_qa_bot
Select-String -Path .env -Pattern '^OPENAI_API_KEY=sk-' -Quiet   # -> True
```

> Since PRs #219 (`kb_mcp`) and #220 (`kb_cli`), **all four entry points**
> (`markdown_kb.app.main`, `gateway.app.main`, `kb_mcp.__main__`, `kb_cli.main`)
> call `load_dotenv(find_dotenv(usecwd=True))` at import, so the runtime CLI / MCP
> paths pick up `.env` from the cwd automatically. No `--env-file` needed anymore.

---

## A. CLI surface (`kb ask` / REPL / `kb index` / `kb import` / `kb ingest` / `kb lint`)

The KB is e-commerce customer support (refunds, orders, warranty, returns, …).
The wiki index (`.kb/index.json`, built from `wiki/{entities,concepts,qa}`)
contains the refund material, so the question below is genuinely grounded.

> **Read surface unchanged.** `kb ask` and the REPL are untouched by the
> interface-parity work (ADR-0017). The checks below add coverage for the new
> write/maintenance subcommands only.

- [ ] **A1 — Grounded answer (wiki stack).** Expect `Grounding: passed`.
  ```powershell
  uv run kb ask "What is the refund timeline?"
  ```
  Expected shape:
  ```
  Stack: wiki
  Answer: <refund-timeline text>

  Citations:
    [1] <section-id> (score: <float>)
        <excerpt>
  Grounding: passed
  ```

- [ ] **A2 — Cannot-Confirm boundary.** Ask something outside the KB. Expect a
  refusal, NOT a hallucinated answer. This is the most important CLI check — a
  grounded bot must decline what it can't support.
  ```powershell
  uv run kb ask "What is the company's parental leave policy?"
  ```
  Expected: `Grounding: cannot confirm (reason: ...)`.

- [ ] **A3 — RAG arm (comparison stack).** Same question, vector_rag arm.
  ```powershell
  uv run kb ask "What is the refund timeline?" --stack rag
  ```
  Expected: an answer, but RAG citations expose no score (`score: null`).

- [ ] **A4 — Interactive REPL (warm index).**
  ```powershell
  uv run kb
  ```
  In the REPL: type a question → see the answer; `:stack rag` toggles the engine;
  `quit` exits. An `LLMError` is printed to stderr and does **not** crash the REPL.

- [ ] **A5 — (optional) Rebuild index.** Needs no key.
  ```powershell
  uv run kb index        # -> "Indexed N file(s), M section(s)."
  ```
  > ⚠️ This **rewrites `.kb/index.json`** (a #204 invariant file). #204 made the
  > write byte-deterministic, so a rebuild should be byte-identical — but verify
  > afterward with `git status --porcelain`; if `.kb/index.json` changed, run
  > `git checkout -- .kb/index.json`. **Skip this step if you only want to verify
  > `ask`.**

- [ ] **A6 — Import a local file.** Needs no key.
  ```powershell
  uv run kb import docs\refund_policy.md
  ```
  Expected: `Imported: refund_policy.md → docs/refund_policy.md [md] status=skipped`
  (or `status=updated` if the hash drifted). Exit code 0.

- [ ] **A7 — Ingest a single Source.** Calls the LLM — needs `OPENAI_API_KEY`.
  ```powershell
  uv run kb ingest refund_policy.md
  ```
  Expected: progress line then `Ingested refund_policy.md: N page(s) created/updated.`
  (or `Skipped ...` on hash-match). Exit code 0. `LLMError` exits with code 1.

- [ ] **A8 — Lint the wiki.** Calls the LLM for the C5 contradiction check — needs
  `OPENAI_API_KEY`.
  ```powershell
  uv run kb lint
  ```
  Expected: `No findings — KB is clean.` (or a count + categorised findings) and
  `Report written to: <path>`. Exit code 0. `LLMError` exits with code 1.

---

## B. Live test suite (`pytest -m live`)

These make real OpenAI calls (small cost + a little time). They load `.env`
through each package's `conftest.py`, so they work out of the box.

- [ ] **B1 — All live tests.**
  ```powershell
  uv run pytest -m live
  ```
  Covers `markdown_kb` (chat / grounding / lint / ingest), `gateway`
  (query rewriting), `vector_rag` (chat live), and `eval` (synthesizer).
  Expect all to pass.

- [ ] **B2 — (optional) Narrow run** to save time / tokens, e.g.:
  ```powershell
  uv run pytest markdown_kb/tests/test_chat_live.py -m live -v
  ```

---

## C. MCP surface via Claude Desktop (four tools)

Uses the `kb_mcp` `.env` parity fix (#219): `uv run --directory <repo>` sets the
server's cwd to the repo root, so `find_dotenv(usecwd=True)` loads `<repo>\.env`
and `OPENAI_API_KEY` is present — **no `env` block needed** in the config. (Getting
a grounded `kb_ask_v1` answer is itself the live proof that #219 works.)

### C1 — Edit the Claude Desktop config

File (Windows): `%APPDATA%\Claude\claude_desktop_config.json`
(i.e. `C:\Users\MaxL\AppData\Roaming\Claude\claude_desktop_config.json`; create it
if absent).

```json
{
  "mcpServers": {
    "kb": {
      "command": "C:\\Users\\MaxL\\.local\\bin\\uv.exe",
      "args": [
        "run",
        "--directory",
        "C:\\Users\\MaxL\\work\\projects\\live_sessions\\knowledge_base_qa_bot",
        "python",
        "-m",
        "kb_mcp"
      ]
    }
  }
}
```

> - The full path to `uv.exe` is used because Claude Desktop may not inherit your
>   shell PATH. Confirm the path with `(Get-Command uv).Source` in PowerShell.
> - **Fallback** if `.env` auto-load doesn't take: add
>   `"env": { "OPENAI_API_KEY": "sk-...your-key..." }` to the `"kb"` object. But try
>   without it first — that's the point of verifying #219.

### C2 — Restart Claude Desktop

Fully quit (also exit from the system tray), then relaunch so it re-reads the
config and spawns the server.

### C3 — Exercise the MCP tools (prompt Claude Desktop in natural language)

> **Read surface unchanged.** `kb_ask_v1`, `kb_search_v1`, `kb_read_hot_v1`, and
> `kb_save_hot_v1` are untouched by the interface-parity work (ADR-0017). The checks
> below add coverage for the five new tools.

**Read / ask (unchanged tools):**

- [ ] **`kb_read_hot_v1`** — "Read the hot cache." First time → `content: ""`.
  An empty string is **normal**, not an error.
- [ ] **`kb_search_v1`** — "Search the KB for refund timeline (raw evidence)." →
  `{stack:"wiki", results:[{id, content, score}]}`; `score` is a BM25 float.
- [ ] **`kb_ask_v1` (grounded)** — "Ask the KB: what is the refund timeline?" →
  `{stack, answer, citations, grounding}` with `grounding.passed = true`.
- [ ] **`kb_ask_v1` (boundary)** — "Ask the KB: what's the parental leave policy?"
  → `grounding.passed = false` + a reason, and **not** `isError` (a valid KB
  boundary, per ADR-0016).
- [ ] **`kb_save_hot_v1`** — "Save this to the hot cache: <a short summary>." →
  `{ok: true}`.
- [ ] **`kb_read_hot_v1` (again)** — "Read the hot cache again." → returns the
  summary you just saved (**round-trip works**).

> `kb_save_hot_v1` writes `wiki/hot.md`, which is **git-ignored** — it won't dirty
> the repo and needs no cleanup.

**Write / maintenance (new parity tools — all need no key except `kb_ingest_v1` and `kb_lint_v1`):**

- [ ] **`kb_import_v1`** — "Import the file at `<absolute-path-to>/docs/refund_policy.md`
  into the KB." → `{ok: true, source: "refund_policy.md", status: "skipped"|"updated"}`.
  A bad path or unsafe basename returns `isError` with `code: "IMPORT_REJECTED"`.

- [ ] **`kb_capture_v1`** — "Capture this as a KB source named `test_note.md`:
  `# Test\nThis is a test note.`" → `{ok: true, path: "<abs-path>/docs/test_note.md"}`.
  An unsafe filename (e.g. `../evil.md`) returns `isError` with
  `code: "CAPTURE_REJECTED"`.

- [ ] **`kb_ingest_v1`** — "Ingest the source `refund_policy.md`." Needs
  `OPENAI_API_KEY`. Progress notifications appear during the run. →
  `{source, pages_created, pages_overwritten, grounding_failed_pages, failed, status}`.
  `LLMError` returns `isError` with `code: "LLM_UNAVAILABLE"` or `"LLM_ERROR"`.

- [ ] **`kb_index_v1`** — "Rebuild the KB index." Needs no key. →
  `{files_indexed: N, sections_indexed: M}`.
  > ⚠️ This **rewrites `.kb/index.json`** — same caveat as A5 above. Run
  > `git checkout -- .kb/index.json` afterward if you need a clean repo state.

- [ ] **`kb_lint_v1`** — "Run the KB lint check." Needs `OPENAI_API_KEY` for the C5
  contradiction pass (pass `include_c5: false` to skip it). → structured `LintResponse`
  (report_path, findings, summary, check_errors). `LLMError` on total C5 failure
  returns `isError`; per-pair LLM errors appear in `check_errors["c5"]` only.

### C4 — Key fix confirmation

A successful grounded `kb_ask_v1` answer proves #219 works (the server loaded
`OPENAI_API_KEY` from the repo `.env` on its own). If it returns `isError` with a
message about auth / missing key, the auto-load didn't take — fall back to the
`env` block in C1.

---

## D. Sign-off — invariant check

After verifying (and **assuming you did NOT run `kb index` in A5**), confirm
nothing mutated the byte-stability invariants:

```powershell
git status --porcelain
python -c "import hashlib; [print(p, hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]) for p in ['.kb/index.json','wiki/index.md']]"
```

Expected:
- `git status --porcelain` → empty (clean).
- `.kb/index.json` → `9d901d6ba9a90724`
- `wiki/index.md` → `2909a454de2ffd03`

If `.kb/index.json` changed (e.g. you ran `kb index`): `git checkout -- .kb/index.json`.

---

## Appendix — reference facts

| Thing | Value |
|---|---|
| `kb` CLI entry point | `kb = "kb_cli.main:app"` (Typer) |
| CLI subcommands | `kb ask` · `kb index` · `kb import` · `kb ingest` · `kb lint` · bare `kb` (REPL) |
| MCP launch | `python -m kb_mcp` (FastMCP, stdio transport) |
| MCP tools (read/ask) | `kb_ask_v1` · `kb_search_v1` · `kb_read_hot_v1` · `kb_save_hot_v1` |
| MCP tools (write/maintenance) | `kb_capture_v1` · `kb_import_v1` · `kb_ingest_v1` · `kb_index_v1` · `kb_lint_v1` |
| Wiki index source dirs | `wiki/{entities,concepts,qa}` → `.kb/index.json` |
| Hot cache file | `wiki/hot.md` (git-ignored) |
| `uv` path | `C:\Users\MaxL\.local\bin\uv.exe` (uv 0.11.13) |
| Live test marker | `@pytest.mark.live` — skipped unless `-m live` |
| `.env` required key | `OPENAI_API_KEY` (both stacks call OpenAI to write the answer) |
| Relevant ADRs | 0015 (LLMError), 0016 (MCP/CLI deep-module adapter, strict schema), 0017 (symmetric interface parity) |

### Concurrency recovery

Concurrent writes from two interfaces (e.g. `kb_ingest_v1` via MCP while `kb ingest`
runs in a terminal) risk leaving `.kb/index.json` in a stale state at worst. The index
is fully regenerable: re-run `kb index` (CLI) or call `kb_index_v1` (MCP) to rebuild it
from the wiki corpus.

### Gotchas (from the Phase 12 handoff)

1. ~~`kb_mcp` / `kb_cli` don't load `.env`~~ — **fixed** in #219 / #220. All four
   entry points now have dotenv parity.
2. ~~`uv run kb` / `python -m kb_mcp` crash with `ModuleNotFoundError: No module
   named 'markdown_kb'` from a non-repo cwd~~ — **fixed** in #221 (`kb_cli`) and #222
   (`kb_mcp`). `markdown_kb` / `vector_rag` are `package = false` PEP 420 namespace
   members, importable only with the repo root on `sys.path`; an installed launcher
   has neither cwd nor pytest's `pythonpath`, so both entry points now insert the repo
   root (`Path(__file__).resolve().parents[2]`) themselves. After #222, `kb_mcp`
   resolves `markdown_kb` regardless of cwd — `--directory <repo>` in §C is now needed
   only so `find_dotenv(usecwd=True)` still loads `.env` (the `OPENAI_API_KEY`), not for
   imports.
3. **Always run from the repo root** (partial collection otherwise).
4. **`.kb/index.json` / `wiki/index.md` byte-stability (#204)** — see §D.
5. The ~12% gateway CC-test flake was **pre-existing and non-hermetic**, fixed in
   #204. Don't re-attribute it to new work if it ever resurfaces.
