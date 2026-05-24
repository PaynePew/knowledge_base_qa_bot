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

```bash
cd markdown_kb
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export KB_SCORE_THRESHOLD="0.5"    # optional; default 0.5

uvicorn app.main:app --reload
```

Then run the curl verification cases listed in [`PROMPT.md`](PROMPT.md).

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

The following stretch goals from `PROMPT.md` are deferred until after the Markdown KB prototype is verified end-to-end. They are described here for orientation only.

- **Score threshold and Cannot Confirm fallback** — already part of the core design (see [ADR-0001](project-docs/adr/0001-strict-grounded-answers.md)).
- **Streaming interface** (`POST /chat/stream` via SSE).
- **Browser UI** showing retrieved Sections before the streamed answer.
- **Multi-format import** (`.txt` / `.html` → canonical Markdown in `docs/`).
- **Alternative interfaces** (CLI, MCP server, web UI).
- **Wiki Index generation** — emit `wiki/index.md` from the Section Index so humans and agents can browse topics without calling the API. This is the first step toward the Karpathy-style LLM Wiki layer.
- **Answer Filing** — write reviewed Q&A results back into `wiki/`, with Citations preserved.
- **Conversation memory** — short context for follow-up questions; retrieved Sources still control the final answer.
- **Paraphrase comparison** — paraphrased queries to expose BM25 synonym misses vs vector semantic false positives.
