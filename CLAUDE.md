# knowledge_base_qa_bot

A grounded Q&A bot over a small Markdown knowledge base, designed long-term as a personal LLM Wiki along the lines of Karpathy's pattern (see [ADR-0003](project-docs/adr/0003-w2-layered-wiki-target-claude-obsidian.md)). Two retrieval strategies coexist at the root: `markdown_kb/` (active prototype, BM25 over Sections) and `vector_rag/` (post-prototype scaffold for the hybrid layer). Sample Sources live in `docs/`.

See `PROMPT.md` for the exercise spec + the answered design questions, `README.md` for running the app, and `CONTEXT.md` for the project's shared vocabulary.

## Workflow triggers

**To process any `ready-for-agent` issue, follow** [`project-docs/orchestration-plan.md`](project-docs/orchestration-plan.md). It defines the four-role loop (plan → implement → review → merge → human-merges-PR), the hand-off contract (labels, branch naming `slice/<N>-<desc>`, commit format, PR auto-close via `Closes #N`), the resume protocol, and the stop conditions. The per-role agent prompts live under [`project-docs/agents/`](project-docs/agents/) — one file per role, deliberately decoupled from the meta-doc so each can be tuned independently:

- [`agents/plan.md`](project-docs/agents/plan.md) — ranks open `ready-for-agent` issues, outputs `<plan>` JSON
- [`agents/implement.md`](project-docs/agents/implement.md) — TDD RGR, incremental commits, COMPLETE / BLOCKED report
- [`agents/review.md`](project-docs/agents/review.md) — diff review, `CODING_STANDARD.md` injection, PASS / FAIL report
- [`agents/merge.md`](project-docs/agents/merge.md) — push branch, open PR with `Closes #N`, link PR on issue

A fresh session can pick up the work by running:
> "Execute the orchestration loop in `project-docs/orchestration-plan.md`, starting from `agents/plan.md` against the currently open `ready-for-agent` issues."

**Before starting any phase beyond the current prototype** (wiki layer, `/ingest`, multi-turn conversation, streaming, etc.), re-read [`project-docs/inspiration.md`](project-docs/inspiration.md) and `grep` the section for the relevant phase tag — for example `grep "phase: wiki" project-docs/inspiration.md`. Operational patterns that are too detailed for `CONTEXT.md` or an ADR but too valuable to forget live there, each tagged with the phase that should trigger its review.

## Agent skills

### Issue tracker

Issues live in GitHub Issues at `PaynePew/knowledge_base_qa_bot`, accessed via the `gh` CLI. See `project-docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles using default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `project-docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` at the root, ADRs under `project-docs/adr/`. See `project-docs/agents/domain.md`.

### Coding standard

The reviewer agent injects [`project-docs/CODING_STANDARD.md`](project-docs/CODING_STANDARD.md) into its prompt — see that file's §0.2 "Reviewer injection scope" for which sections are mandatory vs on-demand. Implementers should also skim §11 (drift signals) before starting a slice to avoid common FAIL conditions.
