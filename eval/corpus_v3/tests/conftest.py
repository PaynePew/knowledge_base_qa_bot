"""Shared fixtures for the corpus v3 eval test suite.

Mirrors ``eval.paraphrase_comparison``'s conftest (CODING_STANDARD §6.3 / §6.5):
every test that builds an index gets ``markdown_kb`` / ``hybrid_kb`` paths
redirected to ``tmp_path`` so no test can pollute production ``.kb/`` /
``wiki/``, and the dense arm's embeddings leaf is swapped for a deterministic
offline fake so the suite runs with no ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.embeddings import Embeddings

import hybrid_kb.app.dense_index as hk_dense
import hybrid_kb.app.logger as hk_logger
import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger


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
    source_dirs_snapshot = mk_indexer.SOURCE_DIRS
    yield
    mk_indexer.SOURCE_DIRS = source_dirs_snapshot
    mk_indexer.sections.clear()
