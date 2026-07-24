"""Shared fixtures for the corpus v3 eval test suite.

Mirrors ``eval.paraphrase_comparison``'s conftest (CODING_STANDARD §6.3 / §6.5):
every test that builds an index gets ``markdown_kb`` / ``vector_rag`` /
``hybrid_kb`` paths redirected to ``tmp_path`` so no test can pollute
production ``.kb/`` / ``wiki/``, and both the dense arm's embeddings leaf AND
vector_rag's FAISS factory are swapped for deterministic offline fakes so the
suite runs with no ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
from langchain_core.embeddings import Embeddings

import hybrid_kb.app.dense_index as hk_dense
import hybrid_kb.app.logger as hk_logger
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

    Mirrors ``eval.paraphrase_comparison.tests.conftest``'s fake exactly: ranks
    the indexed chunk Documents by shared-token overlap with the query, so
    Stack B retrieval is reproducible offline while preserving the real chunk
    metadata produced by ``vr_indexer._load_documents``.
    """

    def __init__(self, documents):
        self._docs = [
            _FakeDoc(page_content=d.page_content, metadata=dict(d.metadata))
            for d in documents
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 3, filter=None, fetch_k: int = 20
    ):
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
            distance = 1.0 / (1.0 + overlap)
            scored.append((doc, distance, overlap))
        scored.sort(key=lambda t: (-t[2],))
        return [(doc, dist) for doc, dist, _ in scored[:k]]

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """No-op persistence stub — the fake is in-memory only."""
        return None


@pytest.fixture()
def fake_vector_index(monkeypatch):
    """Swap vector_rag's FAISS factory for the deterministic fake (offline)."""
    monkeypatch.setattr(
        vr_indexer, "_build_faiss", lambda documents: _FakeVectorStore(documents)
    )
    yield
    vr_indexer.vectorstore = None


class _FakeDenseEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings.

    Maps text to a fixed-length SHA-256-derived vector so the REAL FAISS build
    / persist / similarity-search path of ``hybrid_kb.dense_index`` runs
    without any network call (mirrors ``eval.paraphrase_comparison``'s
    ``_FakeDenseEmbeddings``).
    """

    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(autouse=True)
def fake_dense_embeddings(monkeypatch):
    """Swap hybrid_kb's embeddings leaf for the deterministic offline fake."""
    fake = _FakeDenseEmbeddings()
    monkeypatch.setattr(hk_dense, "get_embeddings", lambda: fake)
    yield fake
    hk_dense.vectorstore = None
    hk_dense.sections_indexed = 0


@pytest.fixture(autouse=True)
def _redirect_production_paths(tmp_path, monkeypatch):
    """Autouse safety net: keep every index build off production paths.

    ``index_wiki_corpus`` / ``index_dense_over_wiki`` reassign ``SOURCE_DIRS``
    directly (the production-isolation behaviour under test); snapshot +
    restore it here so the mutation never leaks into another test.
    """
    monkeypatch.setattr(mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(mk_indexer, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(hk_dense, "DENSE_INDEX_DIR", tmp_path / ".kb" / "hybrid_dense")
    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    source_dirs_snapshot = mk_indexer.SOURCE_DIRS
    docs_dir_snapshot = vr_indexer.DOCS_DIR
    yield
    mk_indexer.SOURCE_DIRS = source_dirs_snapshot
    vr_indexer.DOCS_DIR = docs_dir_snapshot
    mk_indexer.sections.clear()
    vr_indexer.vectorstore = None
