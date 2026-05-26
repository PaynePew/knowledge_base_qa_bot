# Knowledge Base Q&A Bot

A grounded Q&A bot over a small Markdown knowledge base, with citations back to the original `filename#heading`. The repo holds two parallel retrieval strategies; the prototype targets the Markdown KB strategy first, with the Vector RAG app preserved for post-prototype comparison work (see [ADR-0002](project-docs/adr/0002-two-parallel-retrieval-apps.md)).

For the exercise spec and verification, see [`PROMPT.md`](PROMPT.md). For the project's shared vocabulary, see [`CONTEXT.md`](CONTEXT.md). For decisions, see [`project-docs/adr/`](project-docs/adr/).

## Retrieval strategies

| Strategy | Folder | Core idea | Status |
|----------|--------|-----------|--------|
| Markdown KB | [`markdown_kb/`](markdown_kb/) | Parse Markdown headings into Sections, BM25 over a persisted Section Index | Active — prototype target |
| Vector RAG | [`vector_rag/`](vector_rag/) | Split Markdown into chunks, embed with OpenAI, retrieve via FAISS | Scaffold only — post-prototype work |

Both apps expose the same external API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/index` | Read `docs/*.md` and build the retrieval index |
| `POST` | `/chat` | Answer a question with grounded Sections and Citations |

After calling `POST /index`, each strategy persists its retrieval artifact under `.kb/`:

| Strategy | Persisted artifact | Startup behavior |
|----------|--------------------|------------------|
| Markdown KB | `.kb/index.json` (Section Index) | Loads the index into memory on startup |
| Vector RAG | `.kb/faiss_index/` | Loads the FAISS index into memory on startup |

Restarting the server does not require rebuilding immediately. Re-run `POST /index` after editing `docs/*.md`.

## Running the Markdown KB app

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (single `.venv/` at the repo root, single `uv.lock`). See [`pyproject.toml`](pyproject.toml) for the workspace layout.

```bash
# One-time: install deps for all workspace members
uv sync --all-packages

# Run the server (relative imports require running from markdown_kb/)
cd markdown_kb
export OPENAI_API_KEY="sk-..."
export KB_SCORE_THRESHOLD="0.5"    # optional; default 0.5
uv run uvicorn app.main:app --reload
```

Run the tests with:

```bash
cd markdown_kb
uv run pytest                       # default: skips live OpenAI tests
uv run pytest -m live               # opt-in: real OpenAI API calls
```

Then run the curl verification cases listed in [`PROMPT.md`](PROMPT.md).

> The `vector_rag/` scaffold is intentionally **not** in the uv workspace — it pins the legacy langchain 0.3.x ecosystem, which conflicts with `markdown_kb`'s 1.x deps. When you reactivate it, either upgrade its imports to langchain 1.x and add it to `[tool.uv.workspace].members`, or run it standalone (`cd vector_rag && uv sync`).

## Prerequisites

Both apps use OpenAI for final answer generation:

```bash
export OPENAI_API_KEY="sk-..."
```

The Markdown KB app does not need embeddings. The Vector RAG app uses OpenAI embeddings.

## Layout

```
/
├── CLAUDE.md                  ← agent-skill configuration
├── CONTEXT.md                 ← shared vocabulary (glossary)
├── PROMPT.md                  ← exercise spec + design answers + verification
├── README.md                  ← this file
├── docs/                      ← Sources (the bot's runtime knowledge base)
├── project-docs/
│   ├── adr/                   ← architectural decisions
│   └── agents/                ← issue-tracker, triage-labels, domain docs
├── markdown_kb/               ← active retrieval app (BM25 + Section Index)
└── vector_rag/                ← scaffold app for future comparison work
```

## Stretch goals

The following stretch goals from `PROMPT.md` are described here for orientation.

- **Score threshold and Cannot Confirm fallback** — already part of the core design (see [ADR-0001](project-docs/adr/0001-strict-grounded-answers.md)).
- **Output validation (Grounding Check)** — **in progress** (Phase 1, Slices #1-#4). A second structured LLM call after the draft answer verifies every claim traces back to a cited Section. Complements the pre-LLM threshold gate; closes the anti-hallucination loop. Design locked in [ADR-0004](project-docs/adr/0004-post-llm-grounding-check.md).
- **Wiki Index generation** — planned (Phase 2). Emit `wiki/index.md` from the Section Index so humans and agents can browse topics without calling the API. First concrete step toward the Karpathy-style LLM Wiki layer (ADR-0003). Design to be locked in a follow-up session before implementation.
- **Answer Filing** — deferred until `wiki/` directory exists (requires Phase 2).
- **Conversation memory** — deferred until real multi-turn usage demand emerges.
- **Paraphrase comparison** — deferred until `vector_rag/` is reactivated (langchain 0.3 → 1.x migration).
- **Streaming interface** (`POST /chat/stream` via SSE) — deferred; adoption trigger: sustained user demand after interview feedback.
- **Browser UI** showing retrieved Sections before the streamed answer — deferred; adoption trigger: same as streaming.
- **Multi-format import** (`.txt` / `.html` → canonical Markdown in `docs/`) — deferred; adoption trigger: first non-Markdown Source in practice.
- **Alternative interfaces** (CLI, MCP server, web UI) — deferred; adoption trigger: concrete downstream consumer with interface requirements.

## Roadmap

**Path B** is the chosen implementation sequence: Grounding Check first (post-LLM validation, layer 3 of the anti-hallucination stack), then Wiki Index Generation (navigation surface for the Karpathy-style wiki layer). Axis A stretch goals (streaming, UI, multi-format, alternative interfaces) are deferred until after both phases ship and first interview feedback is gathered.

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1 — Grounding Check** | Post-LLM claim-level verification, Block & Replace contract, unified `grounding` field on `ChatResponse`. Four slices: design docs → schemas → verifier → route wiring. See [ADR-0004](project-docs/adr/0004-post-llm-grounding-check.md). | In progress |
| **Phase 2 — Wiki Index Generation** | Project `.kb/index.json` into `wiki/index.md`. Design to be locked in a follow-up grill session (projection format, trigger route, LLM-assisted summarisation). See [ADR-0003](project-docs/adr/0003-w2-layered-wiki-target-claude-obsidian.md). | Planned |

**Deferred (axis A) — adoption triggers:**

| Item | Trigger |
|------|---------|
| Streaming Interface (`POST /chat/stream` via SSE) | Sustained user demand after interview feedback |
| Browser UI (retrieved Sections + streamed answer) | Same as streaming |
| Multi-format import (`.txt` / `.html` → Markdown) | First non-Markdown Source in practice |
| Alternative interfaces (CLI / MCP server / web UI) | Concrete downstream consumer with interface requirements |
