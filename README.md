# Knowledge Base Q&A Bot

A grounded Q&A bot over a small Markdown knowledge base, with citations back to the original `filename#heading`. The repo holds two parallel retrieval strategies; the prototype targets the Markdown KB strategy first, with the Vector RAG app preserved for post-prototype comparison work (see [ADR-0002](project-docs/adr/0002-two-parallel-retrieval-apps.md)).

## Quick start (demo)

The retrieval indexes for **both** stacks ship pre-built and committed (see
[First-run guarantee](#first-run-guarantee)), so a fresh clone answers questions
immediately — no `ingest`/`index` step required on first run.

```bash
# 1. Install all workspace members (single .venv at the repo root)
uv sync --all-packages

# 2. Provide an OpenAI key — both stacks call OpenAI for the final grounded
#    answer. A repo-root .env is loaded automatically:
echo 'OPENAI_API_KEY=sk-...' > .env          # or: export OPENAI_API_KEY="sk-..."

# 3. Launch the Gateway — serves the browser UI and BOTH stacks on one origin.
#    Run from the repo root (not from inside a stack folder).
uv run uvicorn gateway.app.main:app --port 8000
```

Then open <http://localhost:8000/> and use the **Wiki / RAG toggle** to ask the
same question against either retrieval stack. The Gateway mounts the Wiki stack
at `/wiki`, the RAG stack at `/rag`, and exposes `POST /chat/stream` (Server-Sent
Events, sources-first) which the UI consumes.

> The Gateway is the recommended demo surface. To run a single stack on its own,
> see [Running a single stack](#running-a-single-stack). To rebuild the indexes
> after editing the corpus, see [Rebuilding the indexes](#rebuilding-the-indexes).

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

Both apps share a core API; the Wiki stack additionally implements the curated-layer endpoints (`/import`, `/ingest`):

| Method | Endpoint | Description | Wiki | RAG |
|--------|----------|-------------|:----:|:---:|
| `GET` | `/health` | Liveness check | ✓ | ✓ |
| `POST` | `/import` | Convert `raw/**/*.{html,txt}` → `docs/*.md` with provenance frontmatter | ✓ | — |
| `POST` | `/ingest` | Synthesise `docs/*.md` Sources → curated `wiki/` pages (LLM) | ✓ | — |
| `POST` | `/index` | Build the retrieval index (Wiki: `wiki/concepts` + `wiki/entities`; RAG: chunk + embed `docs/`) | ✓ | ✓ |
| `POST` | `/chat` | Answer a question with grounded Sections and Citations | ✓ | ✓ |

The Gateway (Phase 9) adds `POST /chat/stream` (SSE) for the browser UI and routes to either stack via a `?stack=wiki|rag` query param.

After calling `POST /index`, each strategy persists its retrieval artifact under `.kb/`:

| Strategy | Persisted artifact | Startup behavior |
|----------|--------------------|------------------|
| Markdown KB | `.kb/index.json` (Section Index) | Loads the index into memory on startup |
| Vector RAG | `.kb/faiss_index/` | Loads the FAISS index into memory on startup |

Each app loads its persisted index on startup; if the artifact is missing it
serves an **empty** index (every query returns *"I cannot confirm"*) until you
rebuild. That is why the indexes are committed — see below.

### First-run guarantee

Both retrieval artifacts under `.kb/` are committed as a demo seed, so the bot
answers `/chat` on the very first run without `/ingest` or `/index`:

| Stack | Committed artifact | Coverage |
|-------|--------------------|----------|
| Wiki (markdown_kb) | `.kb/index.json` | 60 sections — original FAQ + `fake-docs/` + Chinese `demo-zh/`, synthesised into `wiki/` |
| Vector RAG (vector_rag) | `.kb/faiss_index/` | all 25 Sources in `docs/` (282 chunks), incl. `fake-docs/` + `demo-zh/` |

`.kb/` stays gitignored; these two artifacts are intentionally force-added (`git
add -f`). Rebuild and re-commit them whenever `docs/` or `wiki/` change — see
[Rebuilding the indexes](#rebuilding-the-indexes).

> **Coverage note (Wiki vs RAG).** Both stacks now cover the full demo corpus —
> the English `fake-docs/` topics (loyalty, gift cards, payments, warranty, bulk
> orders, …) and the Chinese `demo-zh/退款政策.md`. RAG indexes the *raw* `docs/`
> chunks; Wiki answers from the curated `wiki/` concept/entity pages synthesised
> by `/ingest`.
>
> One nuance: a few minimal original docs (`account_help`, `refund_policy`,
> `shipping_faq`) overlap with the richer `fake-docs/` on topics like
> cancellation and shipping, and state *different* facts (e.g. cancel window
> 24h vs 1–2h). To keep the Wiki stack from surfacing contradictory sources (the
> grounding check would refuse, returning *"I cannot confirm"*), the original
> concepts are kept canonical for those overlapping topics and the duplicate
> `fake-docs/` concepts were dropped at ingest time. Fake-docs-unique topics and
> the Chinese concepts are fully indexed.

## Running a single stack

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (single `.venv/` at the repo root, single `uv.lock`). See [`pyproject.toml`](pyproject.toml) for the workspace layout. The [Gateway](#quick-start-demo) is the usual entry point; to exercise one stack directly:

```bash
# Wiki stack (markdown_kb) — relative imports require running from markdown_kb/
cd markdown_kb
export OPENAI_API_KEY="sk-..."
export KB_SCORE_THRESHOLD="0.5"    # optional; default 0.5
uv run uvicorn app.main:app --reload --port 8000

# RAG stack (vector_rag) — run from the repo root so markdown_kb resolves
export OPENAI_API_KEY="sk-..."
uv run uvicorn vector_rag.app.main:app --reload --port 8001
```

Each stack exposes the shared API (`/health`, `/index`, `/chat`; the Wiki stack adds `/import` and `/ingest`) directly at its own root, e.g. `curl -X POST http://localhost:8000/chat -d '{"query":"..."}'`. Run the curl verification cases listed in [`PROMPT.md`](PROMPT.md).

Run the tests from the repo root (collects every workspace member's suite):

```bash
uv run pytest                       # default: skips live OpenAI tests
uv run pytest -m live               # opt-in: real OpenAI API calls
```

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

## Rebuilding the indexes

`.kb/` is regenerable. Rebuild and re-commit after editing `docs/` (RAG + Wiki
Sources) or `wiki/` (the Wiki curated layer). With the Gateway running on `:8000`:

```bash
# (optional) regenerate the curated Wiki layer from docs/ first — LLM synthesis
# that rewrites wiki/concepts/. Skip this to keep the existing curated pages.
curl -s -X POST http://localhost:8000/wiki/ingest | jq .

# Wiki BM25 index  -> .kb/index.json   (local; no embeddings, no key needed)
curl -s -X POST http://localhost:8000/wiki/index | jq .

# RAG FAISS index  -> .kb/faiss_index/ (re-embeds docs/ via OpenAI; key required)
curl -s -X POST http://localhost:8000/rag/index | jq .

# Persist the demo seed (these paths are gitignored, so force-add)
git add -f .kb/index.json .kb/faiss_index/
git commit -m "chore: rebuild .kb demo seed"
```

> **Rebuild the Wiki index via `POST /wiki/index`, never from an eval run.** The
> [`eval/paraphrase_comparison/`](eval/paraphrase_comparison/) harness builds a
> *`docs/`-based* index for its own comparison and overwrites `.kb/index.json`
> with ~76 raw-doc sections. If that snapshot is committed by mistake, the Wiki
> stack retrieves over the wrong layer (raw docs instead of the 9 curated
> concepts). After any eval run, re-run `POST /wiki/index` before committing.

## Prerequisites

Both apps call OpenAI for the final grounded answer (`/chat`), so an
`OPENAI_API_KEY` is required to answer questions on either stack:

```bash
export OPENAI_API_KEY="sk-..."     # or put it in a repo-root .env (auto-loaded)
```

| Stack | Needs OpenAI for answering (`/chat`) | Needs OpenAI for indexing (`/index`) |
|-------|--------------------------------------|--------------------------------------|
| Wiki (markdown_kb) | yes | no — BM25 is local |
| Vector RAG (vector_rag) | yes | yes — uses `text-embedding-3-small` |

Because both indexes ship pre-built, you only need the key for indexing when you
[rebuild](#rebuilding-the-indexes) (and the RAG rebuild is the only step that
spends embedding tokens).

## Layout

```
/
├── CLAUDE.md                  ← agent-skill configuration
├── CONTEXT.md                 ← shared vocabulary (glossary)
├── PROMPT.md                  ← exercise spec + design answers + verification
├── README.md                  ← this file
├── docs/                      ← Sources (the bot's runtime knowledge base)
├── wiki/                      ← generated/curated wiki layer (see wiki/README.md)
├── .kb/                       ← committed demo seed: index.json (Wiki) + faiss_index/ (RAG)
├── project-docs/
│   ├── adr/                   ← architectural decisions
│   └── agents/                ← issue-tracker, triage-labels, domain docs
├── gateway/                   ← Gateway app + browser UI (mounts both stacks; demo entry point)
├── markdown_kb/               ← Wiki retrieval app (BM25 + Section Index)
└── vector_rag/                ← RAG retrieval app (langchain 1.x, FAISS over docs/)
```

## Stretch goals

The following stretch goals from `PROMPT.md` are described here for orientation.

- **Score threshold and Cannot Confirm fallback** — already part of the core design (see [ADR-0001](project-docs/adr/0001-strict-grounded-answers.md)).
- **Output validation (Grounding Check)** — **done** (Phase 1). A second structured LLM call after the draft answer verifies every claim traces back to a cited Section. Design locked in [ADR-0004](project-docs/adr/0004-post-llm-grounding-check.md).
- **Wiki Index generation** — **done** (Phase 2). Emits `wiki/index.md` from the Section Index so humans and agents can browse topics without calling the API.
- **Answer Filing** — **done** (Phase 6). High-confidence `/chat` answers are written back to `wiki/qa/*.md`, closing the Two-output rule on the query side.
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
