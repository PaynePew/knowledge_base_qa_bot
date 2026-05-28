# Knowledge Base Q&A Bot

A grounded Q&A bot over a small Markdown knowledge base, with citations back to the original `filename#heading`. The repo holds two parallel retrieval strategies; the prototype targets the Markdown KB strategy first, with the Vector RAG app preserved for post-prototype comparison work (see [ADR-0002](project-docs/adr/0002-two-parallel-retrieval-apps.md)).

## Positioning

This is a grounded Q&A service designed for **enterprise knowledge management** — FAQ automation, policy lookup, customer-support routing — where the answers must trace back to source documents (no hallucination) and the knowledge base itself benefits from a curator-maintained synthesis layer above the immutable Sources. The prototype implements the retrieval + grounded-answer path (`/chat`); the layered architecture (ADR-0003) supports LLM-maintained synthesis pages (`/ingest`, Phase 3) and Answer Filing (`/chat` → `wiki/qa/`, Phase 6) without architectural rewrite.

Karpathy's LLM Wiki gist and [`AgriciDaniel/claude-obsidian`](https://github.com/AgriciDaniel/claude-obsidian) are the pattern source for the curated layer's design — not the project's final form. The patterns translate to enterprise contexts: Hot Cache → session-scoped agent memory; Wiki Log → audit trail; Lint Pass → KB health audit; frontmatter `confidence`/`status` → document governance.

For the exercise spec and verification, see [`PROMPT.md`](PROMPT.md). For the project's shared vocabulary, see [`CONTEXT.md`](CONTEXT.md). For decisions, see [`project-docs/adr/`](project-docs/adr/). For the short version of why Wiki over RAG, see [`project-docs/why-wiki.md`](project-docs/why-wiki.md).

## Retrieval strategies

| Strategy | Folder | Core idea | Status |
|----------|--------|-----------|--------|
| Markdown KB | [`markdown_kb/`](markdown_kb/) | Parse Markdown headings into Sections, BM25 over a persisted Section Index | Active — prototype target |
| Vector RAG | [`vector_rag/`](vector_rag/) | Split Markdown into chunks, embed with OpenAI, retrieve via FAISS | Active — uv workspace member, langchain 1.x; served via Gateway (Phase 9) |

A head-to-head retrieval comparison of these two strategies on the same raw corpus — per-Paraphrase-Type `hit_rate@3` and MRR, charts, a cost log, and six honest-limitation disclosures — lives in [`eval/paraphrase_comparison/report.md`](eval/paraphrase_comparison/report.md) (Phase 8).

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

> `vector_rag/` is a uv workspace member running langchain 1.x (migrated in Phase 8). Run `uv sync --all-packages` from the repo root to install all workspace members together.

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
└── vector_rag/                ← Stack B retrieval app (langchain 1.x, served via Gateway)
```

## Stretch goals

The following stretch goals from `PROMPT.md` are described here for orientation.

- **Score threshold and Cannot Confirm fallback** — already part of the core design (see [ADR-0001](project-docs/adr/0001-strict-grounded-answers.md)).
- **Output validation (Grounding Check)** — **done** (Phase 1). A second structured LLM call after the draft answer verifies every claim traces back to a cited Section. Design locked in [ADR-0004](project-docs/adr/0004-post-llm-grounding-check.md).
- **Wiki Index generation** — **done** (Phase 2). Emits `wiki/index.md` from the Section Index so humans and agents can browse topics without calling the API.
- **Answer Filing** — **in progress** (Phase 6). High-confidence `/chat` answers are written back to `wiki/qa/*.md`, closing the Two-output rule on the query side.
- **Paraphrase comparison** — **done** (Phase 8). Head-to-head retrieval comparison (`hit_rate@3`, MRR) of Markdown KB vs Vector RAG across seven paraphrase types. Report in [`eval/paraphrase_comparison/report.md`](eval/paraphrase_comparison/report.md).
- **Streaming interface** (`POST /chat/stream` via SSE) — **in progress** (Phase 9, [#116](https://github.com/PaynePew/knowledge_base_qa_bot/issues/116)).
- **Browser UI** showing retrieved Sections before the streamed answer — **in progress** (Phase 9, [#116](https://github.com/PaynePew/knowledge_base_qa_bot/issues/116)).
- **Multi-format import** (`.txt` / `.html` → canonical Markdown in `docs/`) — **done** (Phase 7). `POST /import` converts `raw/**/*.{html,txt}` to `docs/*.md` with provenance frontmatter.
- **Conversation memory** — deferred until real multi-turn usage demand emerges (Phase 11).
- **Alternative interfaces** (CLI, MCP server, web UI) — deferred; adoption trigger: concrete downstream consumer with interface requirements (Phase 12).

## Roadmap

For the full multi-phase implementation sequence, dependencies, effort estimates, and interview-ready stopping points, see [`project-docs/roadmap.md`](project-docs/roadmap.md).

**Done:** Prototype · Phase 1 (Grounding Check) · Phase 2 (Wiki Index Generation) · Phase 3 (`/ingest`) · Phase 4 (W2 layered retrieval) · Phase 5 (`/lint`) · Phase 6 (Answer Filing) · Phase 7 (Multi-format import) · Phase 8 (Paraphrase Comparison).

**In progress:** Phase 9 (Streaming + Browser UI — [#116](https://github.com/PaynePew/knowledge_base_qa_bot/issues/116)).

**⭐ Recommended stopping point:** Phase 5 (`/lint`), which closes the Karpathy Ingest + Query + Lint trio. See [`project-docs/roadmap.md`](project-docs/roadmap.md) for the full sequence.
