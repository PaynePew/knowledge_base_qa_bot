"""Hermetic fixtures for the RAG distance-gate calibration tests.

Mirrors ``vector_rag/tests/conftest.py``: the offline ``_FakeEmbeddings`` leaf +
path redirection keep the suite network-free (CODING_STANDARD §6.3 — mock the
embedding leaf, not the deep retrieval module). The REAL FAISS build / search path
runs unchanged; only the embedding call is faked, so ``collect_distances`` can be
exercised offline (its *shape*, not the semantic separation — that needs the real
embeddings the calibration run uses).
"""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.embeddings import Embeddings

import markdown_kb.app.logger as mk_logger
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger


class _FakeEmbeddings(Embeddings):
    """Deterministic offline stand-in for OpenAIEmbeddings (hash-derived vectors)."""

    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(autouse=True)
def _isolate_and_fake(tmp_path, monkeypatch):
    """Redirect vector_rag's index/log writes to tmp and fake the embedding leaf."""
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: _FakeEmbeddings())
    yield
    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0
