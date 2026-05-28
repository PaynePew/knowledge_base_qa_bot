# Vector RAG (Stack B)

This app is the **Vector RAG** arm (Stack B) of the Phase 8 retrieval comparison — FAISS over recursively-chunked raw Sources. Its companion is `markdown_kb/` (Stack A: Wiki + BM25). See [CONTEXT.md § Phase 8 vocabulary](../CONTEXT.md) and PRD #100.

As of Phase 8 Slice 1 the **retrieval core is implemented**:

- `app/indexer.py` — heading-aware sectioning (reuses `markdown_kb.parse_markdown` + `slugify`, ADR-0002) then recursive character splitting (`chunk_size=500`, `chunk_overlap=50`) into `Chunk`s, embedded into an in-memory FAISS index. `search()` returns domain `Chunk` objects whose `source` is a single docs Section id under the canonical slug convention.
- LangChain imports were migrated 0.x → 1.x: `langchain_core.documents.Document`, `langchain_text_splitters.RecursiveCharacterTextSplitter`, `langchain_community.vectorstores.FAISS`, `langchain_openai.OpenAIEmbeddings`. LangChain types stay inside this module — `search()` never returns a raw `Document` (CODING_STANDARD §2.4).

The structure (`app/main.py`, `routes.py`, `schemas.py`, `indexer.py`, `retrieval.py`) mirrors `markdown_kb/` so the comparison is symmetric:

- Both expose the same external API (`GET /health`, `POST /index`, `POST /chat`).
- Both produce Citations of the form `filename#heading-slug`.
- Stack B reads the **raw corpus** (Stack A reads the curated `wiki/` per ADR-0006 W1).

The grounded `/chat` answer synthesis remains scaffold and is thickened by later Phase 8 slices. Per [ADR-0002](../project-docs/adr/0002-two-parallel-retrieval-apps.md), no pluggable `Retriever` protocol is extracted — the two apps stay independent; the Phase 8 comparison runner in `eval/paraphrase_comparison/` adapts each Stack's retrieval callable in-process.

## Running

```bash
# From repo root (vector_rag is a uv workspace member):
uv sync --all-packages
uv run uvicorn vector_rag.app.main:app  # POST /index requires OPENAI_API_KEY for embeddings
```

## Not yet decided (later slices)

- FAISS index persistence (currently in-memory only).
- Whether to keep FAISS or move to a Postgres / SQLite-based vector store.
- Hybrid retrieval (BM25 + vector rerank) — possibly the actual long-term target instead of pure vector.
