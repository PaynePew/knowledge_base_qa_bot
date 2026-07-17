# knowledge_base_qa_bot

Grounded Q&A bot over a Markdown knowledge base with layered KB management — a curated synthesis layer above immutable Sources ([ADR-0003](project-docs/adr/0003-w2-layered-wiki-target-claude-obsidian.md); pattern sources: Karpathy's LLM Wiki gist + AgriciDaniel/claude-obsidian). Two retrieval strategies coexist at the root: `markdown_kb/` (active prototype, BM25 over Sections) and `vector_rag/` (post-prototype hybrid scaffold). Sample Sources in `docs/`. See `PROMPT.md` (exercise spec + answered design questions), `README.md` (running the app), `CONTEXT.md` (shared vocabulary).

## Workflow triggers

**To process any `ready-for-agent` issue, follow [`project-docs/orchestration-plan.md`](project-docs/orchestration-plan.md)** — the four-role loop (plan → implement → review → merge → human-merges-PR), hand-off contract (labels, `slice/<N>-<desc>` branches, commit format, `Closes #N`), resume protocol, and stop conditions. Per-role prompts live under [`project-docs/agents/`](project-docs/agents/): `plan.md` (ranks issues, outputs `<plan>` JSON), `implement.md` (TDD RGR, COMPLETE/BLOCKED report), `review.md` (diff review + `CODING_STANDARD.md` injection, PASS/FAIL), `merge.md` (push, PR with `Closes #N`). Fresh-session bootstrap: "Execute the orchestration loop in `project-docs/orchestration-plan.md`, starting from `agents/plan.md`."

For parallel fan-out, use this repo's **own** orchestrator via `Workflow({ scriptPath: "project-docs/agents/orchestrator.js", args: {…} })` — gh-native and version-controlled. **Do NOT use the global `slice-orchestrator` skill/workflow here** — it was beads-based and its note step once committed `bd init` to `main`. This repo tracks issues in GitHub Issues + `gh` only.

**Before starting any phase beyond the current prototype**: read [`project-docs/roadmap.md`](project-docs/roadmap.md) for the phase's scope, dependencies, and prep gotchas; then re-read [`project-docs/inspiration.md`](project-docs/inspiration.md) and grep its tag for the phase (e.g. `grep "phase: wiki" project-docs/inspiration.md`) — operational patterns too detailed for `CONTEXT.md` or an ADR live there.

## Agent skills

- **Issue tracker**: GitHub Issues at `PaynePew/knowledge_base_qa_bot` via `gh`. See `project-docs/agents/issue-tracker.md`.
- **Triage labels**: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `project-docs/agents/triage-labels.md`.
- **Domain docs**: one root `CONTEXT.md`, ADRs under `project-docs/adr/`. See `project-docs/agents/domain.md`.
- **Coding standard**: the reviewer injects [`project-docs/CODING_STANDARD.md`](project-docs/CODING_STANDARD.md) (§0.2 defines mandatory vs on-demand sections); implementers skim §11 drift signals before starting a slice.
