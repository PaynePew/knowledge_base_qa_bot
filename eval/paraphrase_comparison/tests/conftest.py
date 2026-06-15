"""Shared fixtures for the paraphrase_comparison test suite.

The default suite must run offline (no OPENAI_API_KEY). Stack B's only
network dependency is the embedding call inside ``FAISS.from_documents`` and
``similarity_search_with_score``. ``fake_vector_index`` swaps
``vector_rag.app.indexer._build_faiss`` for a deterministic token-overlap
vectorstore so the whole comparison runs without embeddings, while the real
chunking + metadata path (``_load_documents``) is exercised unchanged — the
fake replaces only the embedding/similarity layer (CODING_STANDARD §6.3: mock
the network leaf, not the deep retrieval module).

Path redirection mirrors markdown_kb's conftest: every test that builds an
index gets markdown_kb's INDEX_PATH / WIKI_DIR / LOG_PATH pointed at tmp, so no
test can pollute production ``.kb/`` / ``wiki/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
from markdown_kb.app.indexer import tokenize


@dataclass
class _FakeDoc:
    """Minimal stand-in for a LangChain Document (page_content + metadata)."""

    page_content: str
    metadata: dict


class _FakeVectorStore:
    """Deterministic token-overlap ranker standing in for a FAISS index.

    Ranks the indexed chunk Documents by Jaccard-like token overlap with the
    query (more shared tokens = closer = lower distance). This makes Stack B
    retrieval reproducible offline while preserving the real chunk metadata
    produced by ``_load_documents``.
    """

    def __init__(self, documents):
        self._docs = [
            _FakeDoc(page_content=d.page_content, metadata=dict(d.metadata))
            for d in documents
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 3, filter=None, fetch_k: int = 20
    ):
        # Mirror FAISS's dict-metadata filter (#290 RAG language filter): drop
        # docs whose metadata does not match every key in ``filter`` before
        # ranking, so the offline fake reproduces same-language retrieval rather
        # than cross-language leaking. ``fetch_k`` is accepted for signature
        # parity with the real FAISS method; the in-memory fake ranks all docs.
        docs = self._docs
        if filter:
            docs = [
                doc
                for doc in docs
                if all(doc.metadata.get(fk) == fv for fk, fv in filter.items())
            ]
        q = set(tokenize(query))
        scored = []
        for doc in docs:
            overlap = len(q & set(tokenize(doc.page_content)))
            distance = 1.0 / (1.0 + overlap)  # more overlap -> smaller distance
            scored.append((doc, distance, overlap))
        # Sort by descending overlap (ascending distance); stable for ties.
        scored.sort(key=lambda t: (-t[2],))
        return [(doc, dist) for doc, dist, _ in scored[:k]]

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """No-op persistence stub.

        vector_rag's ``build_index`` now persists the FAISS index on success
        (issue #103); the fake is in-memory only and the comparison never
        reloads from disk, so the save is a harmless no-op here. The autouse
        ``_redirect_markdown_kb_paths`` fixture still repoints FAISS_INDEX_DIR
        to tmp as a belt-and-braces guard.
        """
        return None


@pytest.fixture()
def fake_vector_index(monkeypatch):
    """Swap vector_rag's FAISS factory for the deterministic fake (offline)."""
    monkeypatch.setattr(
        vr_indexer, "_build_faiss", lambda documents: _FakeVectorStore(documents)
    )
    yield
    vr_indexer.vectorstore = None


@pytest.fixture(autouse=True)
def _redirect_markdown_kb_paths(tmp_path, monkeypatch):
    """Autouse safety net: keep every index build off production paths.

    ``index_stack_a`` / ``index_stack_b`` reassign ``SOURCE_DIRS`` / ``DOCS_DIR``
    directly (not via monkeypatch) so the production-isolation behaviour is the
    one under test; snapshot + restore them here so the mutation never leaks
    into another test or test suite.
    """
    monkeypatch.setattr(mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(mk_indexer, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    # vector_rag's build_index now persists to FAISS_INDEX_DIR (issue #103);
    # redirect it (and the log) to tmp so a direct index_stack_b() call in a
    # test never writes to production .kb/ / vector_rag/log.md.
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    source_dirs_snapshot = mk_indexer.SOURCE_DIRS
    docs_dir_snapshot = vr_indexer.DOCS_DIR
    yield
    mk_indexer.SOURCE_DIRS = source_dirs_snapshot
    vr_indexer.DOCS_DIR = docs_dir_snapshot
    mk_indexer.sections.clear()
    vr_indexer.vectorstore = None
