# Orchestration Plan — Implementing Slices 2–6

This file is the operational playbook for a Claude Code session whose only job is to drive Slices 2–6 of the prototype implementation to completion. It exists because the orchestration design and TDD policy were settled in a long `/grill-with-docs` + `/to-prd` + `/to-issues` session, and we want a fresh session (with a clean context window) to execute them without re-doing the alignment work.

## Handoff one-liner

> "Execute `project-docs/orchestration-plan.md`, starting from the first issue that is still open and `ready-for-agent` (dependency order: #3 → #4 → #5 → #6 → #7, all gated on #2 being closed first)."

That is all the new session needs. Everything else lives in the repo.

## What the orchestrator agent should read first

In this order:

1. `CLAUDE.md` — workflow triggers, agent skills configuration
2. This file (`project-docs/orchestration-plan.md`)
3. `project-docs/prd.md` — the PRD (also GitHub issue #1)
4. `CONTEXT.md` — vocabulary (Source, Section, Section Index, Citation, Grounded Answer, Cannot Confirm; reserved Wiki / Hot Cache / Wiki Log / Source Template / Lint Pass / Ingest)
5. `project-docs/adr/0001-strict-grounded-answers.md` — pre-LLM Cannot Confirm gate
6. `project-docs/adr/0002-two-parallel-retrieval-apps.md` — dual-app layout
7. `project-docs/adr/0003-w2-layered-wiki-target-claude-obsidian.md` — W2 layered Wiki, target = claude-obsidian, `SOURCE_DIRS` as a list
8. `markdown_kb/tests/README.md` — test philosophy (integration-first, fake LLM by default, one `@pytest.mark.live` smoke)
9. `markdown_kb/app/indexer.py` — Section dataclass and the 10-rule `parse_markdown` docstring

The deferred patterns in `project-docs/inspiration.md` are NOT in scope for these slices. If an implementer suggests pulling one in, the orchestrator must reject it — those are post-prototype.

## Mode

**B — Semi-auto, non-blocking advance.** After each slice closes successfully, the orchestrator immediately starts the next slice. The human can interrupt at any time via the Claude Code interrupt keys (`Esc Esc` or `Ctrl+C`). There is no timed veto window — the design is "proceed unless interrupted," not "wait for confirmation."

If the human wants explicit confirmation between slices, switch to **Mode C — Manual gate**: orchestrator stops after each slice and waits for the human to type "continue" (or equivalent). The human can request the switch at any time.

## Roles and models

| Role | Model | How spawned |
|---|---|---|
| Orchestrator | Opus 4.7 | The top-level Claude Code session |
| Implementer | Sonnet 4.6 | `Agent(subagent_type="general-purpose", model="sonnet", prompt=…)` |
| Reviewer | Opus 4.7 | `Agent(subagent_type="general-purpose", model="opus", prompt=…)` |

Issue tracker: `PaynePew/knowledge_base_qa_bot` (verified — see `project-docs/agents/issue-tracker.md`).
Triage label: `ready-for-agent` (already applied to all six slice issues).

## Issue order (strict dependency)

1. `#3` Slice 2 — Answer a grounded query through `/chat`
2. `#4` Slice 3 — Cannot Confirm fallback for out-of-scope queries
3. `#5` Slice 4 — Error handling for OpenAI failures + light grounding check
4. `#6` Slice 5 — Server restart preserves the Section Index
5. `#7` Slice 6 — Live smoke test against real OpenAI

All five depend on `#2` (Slice 1) being closed first. If `#2` is still open, start with that one using the same Implementer/Reviewer protocol below.

After each successful slice, the next slice in the list above becomes the active one. Slices `#5`, `#6`, `#7` (Error handling, Restart, Live smoke) only depend on `#4` (Slice 3) via `#3` (Slice 2). They could parallelize, but Mode B runs them sequentially for simplicity.

## Per-slice loop

For each open issue with label `ready-for-agent` in the order above, the orchestrator runs three steps.

### Step 1 — Implementer (Sonnet 4.6)

Spawn with this prompt template (substitute `<N>`, `<TITLE>`):

> You are implementing GitHub issue #<N> for the `knowledge_base_qa_bot` repo at `PaynePew/knowledge_base_qa_bot`.
>
> 1. Read the issue body: `gh issue view <N> -R PaynePew/knowledge_base_qa_bot`.
> 2. Read `CONTEXT.md`, the three ADRs in `project-docs/adr/`, `markdown_kb/tests/README.md` (test philosophy), and the `parse_markdown` 10-rule docstring in `markdown_kb/app/indexer.py`.
> 3. **TDD step (RED):** Translate every unchecked acceptance criterion in the issue body into a failing pytest test. Use the test style indicated in the per-slice TDD table in `project-docs/orchestration-plan.md`. Place tests under `markdown_kb/tests/`. Run pytest and confirm they fail for the right reason (not a bug in the test).
> 4. **Implementation step (GREEN):** Implement the production code until every test from step 3 passes.
> 5. Do NOT add strict unit tests for trivial helpers (`slugify`, `tokenize`, etc.). The component and integration tests cover them transitively.
> 6. Run the full test suite (`pytest` with default markers — exclude `live`). If anything outside this slice goes red, fix it in the same commit. Do not commit a broken state.
> 7. Commit with this message format:
>    ```
>    feat: implement Slice <N> — <TITLE> (closes #<N>)
>
>    <2-3 sentence summary of what was built end-to-end>
>
>    Acceptance criteria results:
>    - [x] Criterion 1 — short verification note
>    - [x] Criterion 2 — short verification note
>    ...
>
>    Files touched:
>    - path/to/file.py — what changed there
>    - path/to/file.py — what changed there
>    ```
> 8. Post a comment on issue #<N> with the same body as the commit message:
>    `gh issue comment <N> -R PaynePew/knowledge_base_qa_bot --body-file <tmp-file>`
> 9. Do NOT close the issue. The orchestrator closes it after the reviewer signs off.
> 10. Return a one-paragraph summary of what you did to the orchestrator.

### Step 2 — Reviewer (Opus 4.7)

Spawn with this prompt template:

> You are reviewing the HEAD commit on the `knowledge_base_qa_bot` repo. The implementer just claimed to have closed issue #<N>.
>
> 1. Read the issue body: `gh issue view <N> -R PaynePew/knowledge_base_qa_bot`.
> 2. Read the implementer's commit: `git log -1 --stat HEAD` and `git diff HEAD~1..HEAD`.
> 3. **Re-run the full test suite** (`pytest`, default markers). If anything is red, the review is FAIL.
> 4. For each acceptance criterion in the issue body, verify it is *genuinely* met by inspecting code + tests — not just claimed in the commit message.
> 5. Verify the implementation respects:
>    - `CONTEXT.md` vocabulary — code identifiers use the glossary terms (Source, Section, Citation, etc.).
>    - ADR-0001 — strict grounded; pre-LLM Cannot Confirm gate runs before any LLM call; exact-phrase fallback.
>    - ADR-0002 — single retrieval app per directory; no premature plugin/protocol refactor.
>    - ADR-0003 — `SOURCE_DIRS` is a list (not a single path) so the future Wiki layer can be appended.
>    - The 10-rule `parse_markdown` spec (only if Slice 2 / `#3` touches the parser).
>    - The test philosophy in `markdown_kb/tests/README.md`.
> 6. Flag scope creep: if the implementer added anything not authorized by the issue's "What to build" section, call it out.
> 7. Post a review comment on the issue starting with the literal first word **PASS** or **FAIL**:
>    `gh issue comment <N> -R PaynePew/knowledge_base_qa_bot --body "<your review>"`
> 8. If PASS: just post the comment, do nothing to the code.
> 9. If FAIL: list the specific changes the implementer must make. Do NOT make the changes yourself.
> 10. Return your final verdict (PASS / FAIL) and the comment body to the orchestrator.

### Step 3 — Orchestrator advances

- If Reviewer returned `PASS`:
  - Close the issue: `gh issue close <N> -R PaynePew/knowledge_base_qa_bot --reason completed`
  - Print a one-line update to the user (e.g., `Slice <N> merged. Advancing to #<next>.`)
  - Continue to the next issue in the order above.
- If Reviewer returned `FAIL`:
  - Spawn the Implementer again with the Reviewer's feedback as additional context (paste the FAIL comment).
  - Re-run Step 1 → Step 2 → Step 3.
  - Max **2 reviewer-driven retries per slice**. If the third review still fails, STOP and post an issue comment summarizing the deadlock; do not advance.

## TDD style per slice

The grilling session decided that strict unit-TDD on every function (slugify, tokenize, BM25 numerics) is too heavy. The acceptance criteria in each issue body are themselves the RED specification.

| Slice | Issue | TDD style | Where the RED test lives |
|---|---|---|---|
| Slice 2 | `#3` | **Integration TDD** | `markdown_kb/tests/test_chat_grounded.py` — two PROMPT.md curl cases via `TestClient` + fake LLM |
| Slice 3 | `#4` | **Integration TDD** | `markdown_kb/tests/test_chat_fallback.py` — out-of-scope query returns exact phrase + mock LLM never invoked |
| Slice 4 | `#5` | **Component TDD** | `markdown_kb/tests/test_chat_errors.py` — mock LLM raises each error class → assert HTTP status + log entry |
| Slice 5 | `#6` | **Integration TDD** | `markdown_kb/tests/test_persistence.py` — TestClient app recreation reads `.kb/index.json` |
| Slice 6 | `#7` | **No TDD layer** — the test IS the deliverable | `markdown_kb/tests/test_chat_live.py` — single `@pytest.mark.live` smoke against real OpenAI |

## Merge-gate hard rules

- The Implementer's commit is ONLY allowed if `pytest` (default markers, excluding `live`) is fully green in the repo after the commit.
- If pytest fails, the Implementer must fix before committing. No broken commits land on `main`.
- The "merge" in this single-branch flow is the commit reaching `main` directly. No PR workflow.
- The "merge summary" is the commit message body — conventional commits format + acceptance-criteria checklist + files touched. The Implementer writes it; the orchestrator does not edit it.
- The Reviewer re-runs `pytest` as part of review. A red test suite means automatic FAIL.

## Context economy for sub-agents

Do NOT paste the full orchestrator conversation or this `orchestration-plan.md` into sub-agent prompts. Each sub-agent rebuilds its own context from the repo:

- The issue body (acceptance criteria + what to build)
- `CONTEXT.md`
- The three ADRs
- `markdown_kb/tests/README.md`
- The `parse_markdown` 10-rule docstring (only slices touching the parser)

This keeps each Sonnet / Opus run's context bounded and cheap.

## When to STOP autonomous mode and ask the human

Stop and surface a blocker to the user if any of the following happens:

- A slice fails review twice (third attempt would be the deadlock case above).
- A slice's implementation breaks more than 2 unrelated tests in the suite (regression too wide to fix in-slice).
- The Implementer wants to add a NEW deferred pattern from `project-docs/inspiration.md` that the issue did not authorize — that is scope creep and must be human-approved.
- ANY proposed change to `PROMPT.md`, `CONTEXT.md`, the ADRs in `project-docs/adr/`, or `project-docs/inspiration.md` — those are human territory. A sub-agent can suggest changes via an issue comment but must not commit them.
- An unexpected failure in `gh` CLI, git, or the Python environment that the sub-agent cannot self-diagnose within one attempt.

## After the last slice

When `#7` (Slice 6 — Live smoke) is closed:

- The orchestrator runs `gh issue list -R PaynePew/knowledge_base_qa_bot --label ready-for-agent --state open` to confirm zero open ready-for-agent issues.
- It then asks the human to run `pytest -m live` once manually (live smoke needs the human's `OPENAI_API_KEY` and confirms the real model follows the SYSTEM_PROMPT).
- Final user-facing message: a summary of all five slices closed + git log oneline of the five commits + reminder to run live smoke before pushing.

## Out of scope for the orchestrator

The orchestrator does NOT:

- Change the PRD, ADRs, vocabulary, or test philosophy. Those are products of the human-led grilling session.
- Create new issues, even for surfaced patterns. If something genuinely deserves a new issue, post a comment on the current slice's issue describing it and surface to the human.
- Run `git push`. The human controls pushing to remote.
- Run `pytest -m live`. That requires the human's API key and is a manual verification step.
