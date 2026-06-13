# Findings — large-file `/ingest` & the missing Source size limit

> **Date:** 2026-06-13 · **Trigger:** "we never set a file-size limit — what's reasonable?"
> plus an attempt to time `kb_ingest_v1` / index on a 10 MB Source in Claude Desktop.
> **Verdict:** the 10 MB test is **structurally impossible to ingest** — it dies on the very
> first LLM call (classify) before any timing signal exists. There is **no size guard** in
> `ingest_sources`. Separately, the test artifacts **polluted the canonical KB** (see §6).

---

## 1. TL;DR

- A 10 MB Source can **never** be ingested with the current design. `classify_source` sends the
  **entire Source, untruncated, in one LLM call**. 10 MB ≈ **2.6 M tokens** vs the ingest model's
  **128 K context** (gpt-4o-mini) → the call is rejected at **classify**, ~34 s in, **0 pages**.
- There is **no size pre-check** anywhere in the ingest path (`ingest_sources` → classify →
  synth). `grep` for any byte/size guard in `markdown_kb/app/ingest.py` returns nothing.
- The "time the ingest" goal is therefore **unmeasurable on this file** — it measures *time to
  classify failure*, not ingest throughput.
- **Upload caps at 10 MB** (`upload.py:51` `MAX_UPLOAD_BYTES`), but ingest can't handle anything
  near that. A user can **upload a file they can never ingest**. The two limits are ~25× apart.

## 2. What was tested + evidence

| Item | Value |
|---|---|
| Test file | `docs/large_test_10mb.md` — 10,485,812 bytes (~10 MB), 7,677 `##` sections |
| `kb_ingest_v1` result | `status="failed"`, `failed=true`, 0 pages, ~33.97 s |
| Actual error (from `wiki/log.md`) | `RateLimitError:classify_failed` (06:57) then `OpenAIError:classify_failed` (06:58) |

The two log lines are **two different runs**, not one cause:
- **`RateLimitError`** — the MCP run **with** the key loaded (server `load_dotenv`). The LLM *was*
  reached; OpenAI returned **429** (the 2.6 M-token request exceeds the per-request / TPM
  allowance). This is the real `kb_ingest_v1` outcome.
- **`OpenAIError`** — a later Python repro of `ingest_sources` whose harness did **not** load
  `.env`, so it failed on a missing key. This is a harness artifact, **not** the ingest limit.

> Note: the ingest LLM is **OpenAI** (`get_ingest_llm()` → ChatOpenAI; `OPENAI_INGEST_MODEL` →
> `OPENAI_MODEL` → `gpt-4o-mini`, 128 K context, `timeout=60`, `max_retries=1`). Any earlier note
> about a "Claude 200 K context limit" is wrong on both vendor and number.

## 3. Root cause — classify is an un-chunked single call

`markdown_kb/app/templates.py`:
- `_build_classifier_user_message(content)` → `f"Classify this Source document:\n\n{content}"`
  — **no truncation**. The whole Source (minus frontmatter) goes in one call.
- This is the **hard ceiling**: the entire Source must fit the ingest model's context window.

Downstream cost model (only reached if classify passes):
- **concept** Source → **1:N**, one `generate_page` LLM call **per Section** + one grounding
  verify per draft ≈ **~2 calls/Section**. 7,677 Sections ≈ **~15 K LLM calls** — pathological.
- **entity** Source → all Sections concatenated into **one** `generate_entity_page` call → same
  context-overflow failure as classify, just one step later.

Either way the 10 MB file is doomed: entity → context overflow; concept → ~15 K calls (and it
never even gets there, because classify overflows first).

## 4. Why the timing test is void

`kb_ingest_v1` failed at **classify**, the first LLM call. It never reached synthesis, so the
~34 s is "time for OpenAI to reject an oversized request", not ingest timing. To get real timing
you need a Source that *passes* classify (see §7).

Also: this was run **in-process** (Python + MCP-in-process). #234's question is the **Claude
Desktop host** tool-call timeout with progress notifications — there is no host in an in-process
run, so this says nothing about #234 either.

## 5. The size-limit question — recommendation

**Where the ceiling comes from:** classify (and entity-synthesis) do single whole-Source calls,
so the limit is *the ingest model's context window*, minus room for system prompt + structured
output. For gpt-4o-mini (128 K) a safe **input budget is ~100 K tokens**.

**Bytes vs tokens (CJK matters):** English ≈ 4 chars/token → ~400 KB. But this project supports
Chinese (Phase 16), and CJK is far denser (~1–2 chars/token), so the *same* byte size is several
× more tokens. A naive byte cap that's safe for English is unsafe for Chinese.

**Recommendation:**
1. **Add a fail-fast size pre-check at the top of `ingest_sources`** (before classify), mirroring
   `upload.py`'s `MAX_UPLOAD_BYTES` pattern, raising a domain error the adapters render
   (consistent with the ADR-0015 `LLMError` contract — HTTP 413/422, MCP `INVALID_INPUT`, CLI msg).
2. **Use a conservative token estimate**, not raw bytes, so CJK is covered:
   `est_tokens ≈ len(content) // 3` (deliberately pessimistic across scripts); reject if
   `est_tokens > KB_INGEST_MAX_TOKENS` (default **~64 K**, ≈ 50 % of the 128 K window — leaves
   headroom for the entity path and the prompt). Env-configurable.
   - MVP shortcut if token-estimation is too much: a byte cap of **256 KB** default
     (`KB_INGEST_MAX_BYTES`), documented as English-leaning and to be lowered for CJK corpora.
3. **Reconcile the upload cap.** `upload.py` MAX_UPLOAD_BYTES = 10 MB is ~25× what ingest can
   accept. Either lower it toward the ingest ceiling, or document that upload-staging ≠
   ingestable and surface the ingest limit at upload time.
4. **(Secondary, cost not correctness)** a soft **section-count** warning for concept Sources,
   since cost/time scale ~2 calls/Section. Not a hard gate.

## 6. KB pollution found (needs cleanup)

The test runs left **test/meta content inside the canonical KB**:
- `docs/large_test_10mb.md` — a 10 MB test file sitting in `docs/` (the canonical Source corpus).
  A bare `kb ingest` / index build would pick it up.
- `docs/session_force_reingest_findings.md` — a *findings doc* placed in `docs/`, then **ingested**,
  producing real wiki pages: `wiki/concepts/force-re-ingest-of-a-kb-source.md`,
  `.../how-to-force-a-re-ingest.md`, `.../the-force-flag-and-where-it-is-exposed.md`,
  `.../why-kb-ingest-v1-returns-skipped.md`. These are meta-notes about the tool, not KB content —
  they would surface in real `/chat` answers.
- `.kb/index.json` + `wiki/concepts/{cancellation-window,non-refundable-items,refund-timeline}.md`
  were also modified during the test runs.

**Lesson:** findings/test files must live in `project-docs/` (like this file), never in `docs/`.
Proposed cleanup (destructive — confirm before running):
```powershell
git checkout -- .kb/index.json wiki/concepts/cancellation-window.md wiki/concepts/non-refundable-items.md wiki/concepts/refund-timeline.md
Remove-Item docs/large_test_10mb.md, docs/session_force_reingest_findings.md
Remove-Item wiki/concepts/force-re-ingest-of-a-kb-source.md, wiki/concepts/how-to-force-a-re-ingest.md, wiki/concepts/the-force-flag-and-where-it-is-exposed.md, wiki/concepts/why-kb-ingest-v1-returns-skipped.md
```
(If the force-re-ingest write-up is worth keeping, move it to `project-docs/` instead of deleting.)

## 7. How to actually test what was wanted

To measure real ingest timing **and** probe the #234 Desktop host timeout, use a Source that
**passes classify** but generates **many** synthesis calls:
- Total Source **< ~64 K tokens** (≈ < 256 KB English / less for CJK) so classify succeeds.
- **~300–500 small Sections** so the concept path makes ~600–1,000 sequential LLM calls →
  several minutes of wall-clock → that is the real progress-notification-keeps-alive stress.
- **Drive it from Claude Desktop** (runbook `slice8-234-…md` §4), with the key in `.env`, so a
  real host timeout is in the loop. Baseline the same Source via `kb ingest <file>` first (§3 of
  that runbook) to know the wall-clock before testing the host.

## 8. Recommended follow-ups (issues to file)

- **`ingest_sources` size guard** (§5) — fail-fast pre-check + env-configurable token/byte limit;
  reconcile with `upload.py` MAX_UPLOAD_BYTES. Hermetic test: oversized Source → domain error,
  no LLM call.
- **KB cleanup** (§6) — revert test pollution; add a contributor note "findings → `project-docs/`,
  never `docs/`".
- (Optional) section-count soft warning for concept Sources (§5.4).

## 9. Why claude-obsidian / Karpathy-wiki don't hit this wall (and what to borrow)

The project's inspiration sources (CLAUDE.md: Karpathy LLM Wiki + AgriciDaniel/claude-obsidian)
**never hit the large-file context wall — because they do NO server-side LLM synthesis.** Verified:

- **claude-obsidian is not even an MCP server** — it's a Claude Code *plugin* (skills + agents +
  commands). Its `/wiki-ingest` agent's declared tools are `Read, Write, Edit, Glob, Grep, Bash`;
  `/save`'s are `Read Write Edit Glob Grep`. **There is no LLM-call tool — because the host agent
  *is* the LLM.** "Ingest" is prompt instructions the host Claude executes step by step (read
  source → read index → write entity/concept pages).
- The Obsidian MCP servers it can use (**MarkusPfundstein/mcp-obsidian**, **StevenStavrakis/
  obsidian-mcp**) are **pure file CRUD + search** — verified zero LLM SDK in their dependencies.
  The MCP layer is a file-access *transport*, not a synthesis engine.
- **`/save`**: the **host** composes the summary; the server (if any) only persists the finished
  bytes. (This is exactly the pattern our **Hot Cache** `kb_save_hot_v1` already follows.)
- **Karpathy LLM Wiki**: the agent reads/writes files; there is no server-side "ingest tool" in the
  concept at all.

**Why this avoids the wall:** the host reads a large doc **incrementally** — section by section,
summarize-as-you-go, multi-turn agentic loop, can `Grep`/skim first — so it never needs the whole
file in one call. We hit the wall because we deliberately put `classify + generate` **inside a
server-side tool** (ADR-0016 deep-module) and fed it the whole file in one shot, inheriting a
single-call context ceiling the file size blows straight through.

**The asymmetry in our own design:** `kb_save_hot_v1` follows the claude-obsidian "server persists
bytes, host composes" pattern — but `kb_ingest_v1` does the opposite (server-side synthesis). That
asymmetry is the root of this problem.

**What to borrow (do NOT just rip out server-side ingest — it's deterministic, testable,
interface-agnostic across CLI/HTTP/MCP, and enforces the grounding contract):**
- **Short term (shipped):** the §5 size guard — reject what one server-side call can't handle, with
  a clear reason.
- **Medium term:** keep server-side synthesis but **chunk the Source before classify** so size
  scales (classify on a sample/outline, synthesize per-Section which the concept path already does).
- **Long term (if large docs become a real need):** add a **host-driven ingest** path for big
  Sources — the agent reads the doc in passes and calls `kb_save`-style write tools — matching the
  inspiration sources. This trades the grounding-contract guarantee for scale, so it's a real ADR.

Sources (verified by sub-agent): `github.com/AgriciDaniel/claude-obsidian`
(`agents/wiki-ingest.md`, `skills/save/SKILL.md`, README), `github.com/MarkusPfundstein/mcp-obsidian`
(`src/mcp_obsidian/tools.py`, `pyproject.toml` — no LLM dep), `github.com/StevenStavrakis/obsidian-mcp`,
Karpathy LLM Wiki gist `442a6bf555914893e9891c11519de94f`.
