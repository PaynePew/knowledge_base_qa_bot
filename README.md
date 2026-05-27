# Knowledge Base Q&A Bot

A grounded Q&A bot over a small Markdown knowledge base, with citations back to the original `filename#heading`. The repo holds two parallel retrieval strategies; the prototype targets the Markdown KB strategy first, with the Vector RAG app preserved for post-prototype comparison work (see [ADR-0002](project-docs/adr/0002-two-parallel-retrieval-apps.md)).

## Positioning

This is a grounded Q&A service designed for **enterprise knowledge management** — FAQ automation, policy lookup, customer-support routing — where the answers must trace back to source documents (no hallucination) and the knowledge base itself benefits from a curator-maintained synthesis layer above the immutable Sources. The prototype implements the retrieval + grounded-answer path (`/chat`); the layered architecture (ADR-0003) preserves the upgrade path to LLM-maintained synthesis pages (`/ingest`, future) without architectural rewrite.

Karpathy's LLM Wiki gist and [`AgriciDaniel/claude-obsidian`](https://github.com/AgriciDaniel/claude-obsidian) are the pattern source for the curated layer's design — not the project's final form. The patterns translate to enterprise contexts: Hot Cache → session-scoped agent memory; Wiki Log → audit trail; Lint Pass → KB health audit; frontmatter `confidence`/`status` → document governance.

For the exercise spec and verification, see [`PROMPT.md`](PROMPT.md). For the project's shared vocabulary, see [`CONTEXT.md`](CONTEXT.md). For decisions, see [`project-docs/adr/`](project-docs/adr/). For the short version of why Wiki over RAG, see [`project-docs/why-wiki.md`](project-docs/why-wiki.md).

## Retrieval strategies

| Strategy | Folder | Core idea | Status |
|----------|--------|-----------|--------|
| Markdown KB | [`markdown_kb/`](markdown_kb/) | Parse Markdown headings into Sections, BM25 over a persisted Section Index | Active — prototype target |
| Vector RAG | [`vector_rag/`](vector_rag/) | Split Markdown into chunks, embed with OpenAI, retrieve via FAISS | Scaffold only — post-prototype work |

Both apps expose the same external API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/import` | Convert `raw/**/*.{html,txt}` → `docs/*.md` with provenance frontmatter |
| `POST` | `/ingest` | Synthesise `docs/*.md` Sources → curated `wiki/` pages (LLM) |
| `POST` | `/index` | Build the retrieval Section Index from `wiki/entities/` + `wiki/concepts/` |
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

### Multi-format import demo (Phase 7)

Drop HTML or plain-text files into `raw/` (gitignored local inbox) and run
`POST /import` to convert them to Markdown docs before ingesting:

```bash
# Copy the example files into the local raw/ inbox
cp examples/raw/*.html raw/
cp examples/raw/*.txt raw/

# Convert raw sources to docs/
curl -s -X POST http://localhost:8000/import | jq .

# Single-file mode: import only one source
curl -s -X POST http://localhost:8000/import \
  -H "Content-Type: application/json" \
  -d '{"source": "clean_article.html"}' | jq .

# Then run the full pipeline to make the content queryable
curl -s -X POST http://localhost:8000/ingest | jq .
curl -s -X POST http://localhost:8000/index | jq .
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}' | jq .
```

The `raw/` directory is gitignored — place your own HTML exports or text files
there. The `examples/raw/` directory contains sample files you can copy in to
try the pipeline immediately.

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
├── wiki/                      ← generated/curated wiki layer (see wiki/README.md)
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
- **Multi-format import** (`.txt` / `.html` → canonical Markdown in `docs/`) — **in progress** (Phase 7, Slice 7-1 tracer bullet shipped). `POST /import` converts `raw/**/*.{html,txt}` to `docs/*.md` with provenance frontmatter. Slices 7-2 (full error handling) and 7-3 (idempotency hash chain) follow.
- **Alternative interfaces** (CLI, MCP server, web UI) — deferred; adoption trigger: concrete downstream consumer with interface requirements.

## Roadmap

For the full multi-phase implementation sequence, dependencies, effort estimates, and interview-ready stopping points, see [`project-docs/roadmap.md`](project-docs/roadmap.md).

**Done:** Prototype, Phase 1 (Grounding Check, [ADR-0004](project-docs/adr/0004-post-llm-grounding-check.md)), Phase 2 (Wiki Index Generation).

**Next up:** Phase 3 (`/ingest` — Source → curated `wiki/` synthesis pages). Design grill in progress.

**⭐ Recommended stopping point:** Phase 5 (`/lint`), which closes the Karpathy Ingest + Query + Lint trio.
