# knowledge_base_qa_bot

A grounded Q&A bot over a small Markdown knowledge base, designed long-term as a personal LLM Wiki along the lines of Karpathy's pattern (see [ADR-0003](project-docs/adr/0003-w2-layered-wiki-target-claude-obsidian.md)). Two retrieval strategies coexist at the root: `markdown_kb/` (active prototype, BM25 over Sections) and `vector_rag/` (post-prototype scaffold for the hybrid layer). Sample Sources live in `docs/`.

See `PROMPT.md` for the exercise spec + the answered design questions, `README.md` for running the app, and `CONTEXT.md` for the project's shared vocabulary.

## Workflow triggers

**Before starting any phase beyond the current prototype** (wiki layer, `/ingest`, multi-turn conversation, streaming, etc.), re-read [`project-docs/inspiration.md`](project-docs/inspiration.md) and `grep` the section for the relevant phase tag — for example `grep "phase: wiki" project-docs/inspiration.md`. Operational patterns that are too detailed for `CONTEXT.md` or an ADR but too valuable to forget live there, each tagged with the phase that should trigger its review.

## Agent skills

### Issue tracker

Issues live in GitHub Issues at `PaynePew/knowledge_base_qa_bot`, accessed via the `gh` CLI. See `project-docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles using default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `project-docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` at the root, ADRs under `project-docs/adr/`. See `project-docs/agents/domain.md`.
