# Slice 8 (#234) — HITL runbook: Claude Desktop tool timeout vs `kb_ingest_v1`

> **Status:** ready-for-human (manual verification — needs the operator's machine + Claude Desktop).
> **Parent:** PRD #226 · **Blocked by:** #232 (the `kb_ingest_v1` tool, now merged).
> **Goal:** confirm that single-Source **synchronous** ingest with **progress notifications**
> (the locked decision from #232) survives the real Claude Desktop MCP tool-call timeout —
> or, if it doesn't, file a follow-up async-fallback slice with the observed limits.

This is the manual companion to the automated runbook in
[`phase12-live-verify.md`](./phase12-live-verify.md) §C (Claude Desktop wiring). Do §C1/§C2
first if the server isn't wired yet; this file focuses on the **timeout question** only.

---

## 0. Why this slice exists (context)

`kb_ingest_v1` runs the synthesis pipeline **synchronously** and emits MCP **progress
notifications** during the run so the Desktop host doesn't kill the call as "hung". The open
PRD question: _does a realistic — or large — Source still blow past the host's tool-call
timeout despite progress?_ MCP hosts generally **reset** their inactivity timer on each
progress notification, so a steady stream of progress should keep an arbitrarily long call
alive. This slice verifies that empirically. If it fails, the remedy is a **deferred async
start/poll ingest** slice (per ADR-0017) — **not** built here.

---

## 1. Prerequisites

- [x] Windows machine with **Claude Desktop** installed (this is a desktop-app-only check —
      the in-process pytest harness cannot exercise the real host timeout).
- [ ] Repo synced to `main` at or after the #232 merge (PR #241). Confirm the tool exists:
      ```powershell
      uv run python -c "import kb_mcp.server as s; print('kb_ingest_v1' in [t for t in dir(s)] or 'see registry')"
      uv run python -c "import kb_mcp.__main__; print('OK: imports cleanly — repo-root bootstrap (#222) works')"
      ```
      Note: bare `uv run python -m kb_mcp` (no flags) **starts the stdio server and then waits
      silently with no output** — that is success, press Ctrl+C to stop. The module takes no CLI
      flags, so `--help` does **not** print usage; it just starts the server and hangs the same way.
      The two import smokes above exit immediately, so use those for a quick check.
      Or just grep: the tool is registered in `kb_mcp/kb_mcp/server.py` (`@mcp.tool(name="kb_ingest_v1" ...)` and in the `_add_strict_schema()` tuple).
- [x] `OPENAI_API_KEY` available — `kb_ingest_v1` calls the LLM (synthesis + grounding).
      `kb_ingest_v1` is the **only** tool in this check that needs a key.
- [ ] `uv` on PATH (this repo: `C:\Users\MaxL\.local\bin\uv.exe`).
- [ ] At least one **representative** `docs/` Source (e.g. `docs/refund_policy.md`) and,
      if you have one, a **large** Source (long / many sections) — that's the stress case.

---

## 2. Wire `kb_mcp` into Claude Desktop

Config file (Windows): `%APPDATA%\Claude\claude_desktop_config.json`.

```json
{
  "mcpServers": {
    "knowledge_base_qa_bot": {
      "command": "uv",
      "args": [
        "run", "--directory",
        "C:\\Users\\MaxL\\work\\projects\\live_sessions\\knowledge_base_qa_bot",
        "python", "-m", "kb_mcp"
      ]
    }
  }
}
```

- [ ] **No API key in this config.** `__main__.py` runs `load_dotenv(find_dotenv(usecwd=True))`
      before importing the server, and `uv run --directory <repo>` sets the process cwd to the
      repo root — so the repo-root `.env` (the same `OPENAI_API_KEY` you already set for the CLI /
      gateway, per `.env.example`) is loaded automatically. Do **not** duplicate the secret here.
- [ ] ⚠️ **Strict JSON — no trailing commas, no comments.** `claude_desktop_config.json` is parsed
      as strict JSON; a stray trailing comma makes Claude Desktop **fail to load the server
      silently** (it never appears, with no error). This is the #1 cause of "configured but
      nothing happens." Keep the block exactly as above.
- [ ] Save the file.
- [ ] **Fully quit** Claude Desktop (also exit from the system tray), then relaunch so it
      re-reads the config and spawns the server (per `phase12-live-verify.md` §C2).
- [ ] In a new chat, confirm the tools are listed (the MCP/plug icon shows
      `knowledge_base_qa_bot` with `kb_ingest_v1` among the nine `kb_*_v1` tools).

---

## 3. Baseline timing (optional but recommended)

Before testing in Desktop, time the ingest from the CLI so you know roughly how long the
host call will take (the CLI path runs the same pipeline):

```powershell
Measure-Command { uv run kb ingest refund_policy.md }   # representative Source
# and, if available:
Measure-Command { uv run kb ingest <your-large-source>.md }
```

Record the wall-clock seconds. This tells you whether you're testing **above** or **below**
a typical ~60 s host timeout, which frames the result.

> ⚠️ `kb ingest` / `kb_ingest_v1` **write** `wiki/` pages and may touch `.kb/index.json`.
> See §6 cleanup. On a hash-match re-run it reports `Skipped ...` and does little work — to
> force a real run, edit the Source slightly first (or `git checkout -- docs/<source>` to reset).

---

## 4. Drive `kb_ingest_v1` through Claude Desktop

In the Claude Desktop chat:

1. [ ] **Representative Source.** Prompt in natural language, e.g.:
   > "Use the knowledge base tools: ingest the source `refund_policy.md`."
   > Approve the tool call when prompted.
2. [ ] **Observe during the run:**
   - Do **progress notifications** appear / does the call show an in-progress state rather
     than spinning silently?
   - Does the call **complete** and return the result dict
     (`{source, pages_created, pages_overwritten, grounding_failed_pages, failed, status}`)?
   - Any host error like _"the tool call timed out"_ / _"MCP server did not respond"_?
3. [ ] **Large Source (the real stress test).** Repeat with the largest Source you have.
       This is where a single sync call is most likely to exceed the host timeout. Watch the
       same three things.
4. [ ] If you can, **force a slow run** to probe the ceiling: a very large Source, or a
       Source that produces many concept/entity pages (each page = an LLM call). The question is
       whether progress keeps the call alive **regardless** of total duration.

---

## 5. Record the result on issue #234

Paste a comment on #234 using this template (fill the brackets):

```markdown
## HITL result — Claude Desktop × kb_ingest_v1

**Environment:** Claude Desktop <version> on Windows <ver>; repo @ <commit>; OPENAI_API_KEY present.

**Baseline CLI timing:** representative `refund_policy.md` = <N> s; large `<source>` = <M> s.

**Representative Source:** completed = <yes/no>; progress visible = <yes/no>; host timeout = <none / at ~Xs>.
**Large Source:** completed = <yes/no>; progress visible = <yes/no>; host timeout = <none / at ~Xs>.

**Observed host timeout behaviour:** <e.g. "no timeout up to <M>s; progress notifications
reset the host inactivity timer" OR "host killed the call at ~<X>s despite progress">.

**Verdict:** <one of the two below>
```

---

## 6. Verdict & next step (decision tree)

- [ ] **PASS — single-Source sync + progress is sufficient.**
      Both Sources completed without a host timeout (progress kept the call alive). Record
      the verdict on #234; a human can then close #234. No new code. This confirms the PRD's
      locked decision and the open question is resolved.

- [ ] **FAIL — a large Source still exceeds the host timeout despite progress.**
      Do **NOT** build the fix here. File a **new deferred slice** for an async start/poll
      ingest fallback (ADR-0017), capturing the observed ceiling. Suggested new issue:

      ```markdown
      Title: Slice N (deferred): async start/poll fallback for kb_ingest_v1 (host-timeout)
      Parent: #226

      ## What to build
      kb_ingest_v1 (single-Source sync + progress) exceeds the Claude Desktop tool-call
      timeout on large Sources: observed timeout at ~<X>s on `<source>` (<M>s CLI baseline),
      progress notifications did NOT extend it past <X>s. Add an async surface: a
      kb_ingest_start_v1 that returns a job id immediately + kb_ingest_status_v1 to poll,
      keeping the sync kb_ingest_v1 for small Sources. (ADR-0017 anticipated this fallback.)

      ## Acceptance criteria
      - [ ] kb_ingest_start_v1 returns a job handle within the host timeout for any Source size
      - [ ] kb_ingest_status_v1 reports progress/completion/result without re-running the pipeline
      - [ ] sync kb_ingest_v1 retained for small Sources; no batch param (ADR-0016)
      - [ ] hermetic FastMCP tests for start + poll states
      ## Blocked by
      - none (kb_ingest_v1 already merged)
      ```
      Label the new issue `ready-for-agent` once triaged, then record on #234 that the
      verdict was "async fallback filed as #<new>" and close #234.

---

## 7. Cleanup

- [ ] `kb_ingest_v1` may have rewritten `.kb/index.json` and added `wiki/` pages. To reset the
      repo to a clean state:
      ```powershell
      git checkout -- .kb/index.json
      git status   # review any new wiki/ pages; discard if this was only a smoke test
      ```
- [ ] `wiki/hot.md` is git-ignored — no cleanup needed.
- [ ] No secret-cleanup needed in `claude_desktop_config.json`: §2 keeps the API key in the
      repo-root `.env`, not in the Desktop config, so nothing sensitive was written there.

---

## 8. Troubleshooting

- **Server doesn't appear in Desktop / "failed to start":** check the `--directory` path is
  the repo root and `uv` is on PATH; try the `args` command manually in a terminal:
  `uv run --directory <repo> python -m kb_mcp` should start and wait on stdio.
- **`ModuleNotFoundError: markdown_kb` (or `vector_rag`):** the entry point must bootstrap the
  repo root onto `sys.path` — this was fixed in #222; confirm you're on a recent `main`.
- **`isError` with `LLM_UNAVAILABLE` / `LLM_ERROR`:** the `OPENAI_API_KEY` is missing/invalid
  in the repo-root `.env` (§2 loads it from there, not the Desktop config), or OpenAI is
  unreachable. This is the LLMError path, not a timeout.
- **Call returns instantly with `status: "skipped"`:** the Source hash matched an earlier
  ingest (no-op). Edit the Source to force a real run, then `git checkout --` it after.
