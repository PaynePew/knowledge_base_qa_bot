# knowledge_base_qa_bot

A live-session teaching repo for building a grounded Q&A bot over a small Markdown knowledge base. Two retrieval strategies are scaffolded under `scaffold/` (Markdown KB with BM25, and Vector RAG with FAISS); sample knowledge-base content lives under `docs/`.

See `PROMPT.md` for the exercise spec and `README.md` for the guided tracks.

## Agent skills

### Issue tracker

Issues live in GitHub Issues at `PaynePew/knowledge_base_qa_bot`, accessed via the `gh` CLI. See `project-docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles using default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `project-docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` at the root, ADRs under `project-docs/adr/`. See `project-docs/agents/domain.md`.
