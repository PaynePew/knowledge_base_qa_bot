# Vector RAG (post-prototype work)

This app is a parallel retrieval strategy to `markdown_kb/`. It is intentionally **left as a scaffold** for now; the prototype delivered by the original deadline targets `markdown_kb/` only.

The structure (`app/main.py`, `routes.py`, `schemas.py`, `indexer.py`, `retrieval.py`) deliberately mirrors `markdown_kb/` so the eventual comparison work is symmetric:

- Both expose the same external API (`GET /health`, `POST /index`, `POST /chat`).
- Both consume the same `docs/` Sources.
- Both must produce Citations of the form `filename#heading-slug`.

When the comparison work happens, the two apps will likely be unified behind a pluggable `Retriever` protocol — see [ADR-0002](../project-docs/adr/0002-two-parallel-retrieval-apps.md). Until then, keep this app's TODOs untouched; we want the eventual implementation to be informed by what we learn from running the Markdown KB strategy in real use.

## Not yet decided

- Chunk size and overlap calibration for this corpus.
- Whether to keep FAISS or move to a Postgres / SQLite-based vector store.
- Hybrid retrieval (BM25 + vector rerank, per `qmd`-style designs) — possibly the actual target instead of pure vector.
