"""Re-bake the committed ``.kb/`` seed after a Source doc changes.

Regenerates every retrieval artifact so all three stacks reflect the current
``docs/`` + ``wiki/`` state, in dependency order:

    1. re-ingest the changed Source(s)  -> wiki/ pages + orphan deletion  (LLM)
    2. BM25            -> .kb/index.json        (reads wiki/, no key)
    3. Hybrid dense    -> .kb/hybrid_dense/      (reads wiki/, needs key)
    4. RAG FAISS       -> .kb/faiss_index/       (reads docs/, needs key)

Run from the repo root with the workspace venv::

    uv run python scripts/rebake.py

Then commit the regenerated (gitignored) seed::

    git add -f .kb wiki

Needs ``OPENAI_API_KEY`` in ``.env`` (ingest + dense + RAG embeddings; BM25 does
not). Deterministic: ingest LLM calls are pinned ``temperature=0`` and the
``text-embedding-3-small`` embeddings are stable, so re-running is safe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The namespace members (markdown_kb / hybrid_kb / vector_rag) are package=false
# workspace dirs, not pip-installed packages, so a bare ``python scripts/…`` (whose
# sys.path[0] is scripts/, not the repo root) can't import them. Put the repo root
# first — same bootstrap the installed console scripts need (see memory
# "entry-points-need-repo-root-on-syspath").
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402  (after sys.path bootstrap)

# Source doc(s) changed in THIS re-bake. Ingest only these (force=True defeats the
# docs_body hash-skip); wiki/ pages the changed Source no longer produces are
# deleted by ingest's delete_orphans, keeping the dense arm 1:1 with BM25.
# NOTE: these live under docs/demo-zh/, and single-source ingest resolves a bare
# name as ``docs_dir / name`` with no recursive search — so DOCS_SUBDIR must point
# at the subdirectory, not the docs/ root.
CHANGED_SOURCES = ["退款與退貨.md"]
DOCS_SUBDIR = REPO / "docs" / "demo-zh"


def main() -> int:
    load_dotenv(REPO / ".env")
    # Corpus v2 was baked with gpt-5 (quarantine→0). The code default is gpt-4o-mini,
    # which regenerates terser pages and wipes ``open_questions`` — a partial re-bake
    # would then diverge from the committed corpus. Pin gpt-5; override via the env.
    os.environ.setdefault("OPENAI_INGEST_MODEL", "gpt-5")

    # 1. Re-ingest changed Sources -> regenerate wiki pages + delete orphans (LLM).
    from markdown_kb.app.ingest import ingest_sources

    print(f"[1/4] ingest {CHANGED_SOURCES} from {DOCS_SUBDIR.name}/ (force) …", flush=True)
    ingest_sources(CHANGED_SOURCES, docs_dir=DOCS_SUBDIR, force=True)

    # 2. BM25 -> .kb/index.json (reads wiki/, no OpenAI key).
    from markdown_kb.app.indexer import build_index as bm25_build

    print("[2/4] BM25 index (wiki/) …", flush=True)
    files, sections = bm25_build()
    print(f"      BM25: {files} files, {sections} sections", flush=True)

    # 3. Hybrid dense arm -> .kb/hybrid_dense/ (reads wiki/, needs key).
    from hybrid_kb.app.dense_index import build_index as dense_build

    print("[3/4] hybrid dense index (wiki/) …", flush=True)
    embedded = dense_build()
    print(f"      dense: {embedded} sections embedded", flush=True)

    # 4. RAG FAISS -> .kb/faiss_index/ (reads docs/, needs key).
    from vector_rag.app.indexer import build_index as rag_build

    print("[4/4] RAG FAISS index (docs/) …", flush=True)
    rfiles, chunks = rag_build()
    print(f"      RAG: {rfiles} files, {chunks} chunks", flush=True)

    print("\nRe-bake complete. Commit the regenerated seed (it is gitignored):", flush=True)
    print("    git add -f .kb wiki", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
