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


class _FakeDenseEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings (Stack C dense arm).

    Maps text to a fixed-length SHA-256-derived vector so the REAL FAISS build /
    persist / similarity-search path of ``hybrid_kb.dense_index`` runs without any
    network call (mirrors the hybrid_kb test conftest's ``_FakeEmbeddings``).
    Stable across processes so a save→load roundtrip returns the same neighbours.
    """

    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture()
def fake_vector_index(monkeypatch):
    """Swap vector_rag's FAISS factory for the deterministic fake (offline)."""
    monkeypatch.setattr(
        vr_indexer, "_build_faiss", lambda documents: _FakeVectorStore(documents)
    )
    yield
    vr_indexer.vectorstore = None


@pytest.fixture(autouse=True)
def fake_dense_embeddings(monkeypatch):
    """Swap hybrid_kb's embeddings leaf for the deterministic offline fake.

    Patches ``get_embeddings`` (not ``_build_faiss``) so Stack C's whole real
    FAISS path — build, save_local, similarity search — runs offline, exactly as
    the hybrid_kb suite does (CODING_STANDARD §6.3 — mock the network leaf).

    Autouse: Stack C's dense arm is now built in every ``run_comparison`` (the
    three-arm methodology), and the whole eval suite must run hermetically with
    no ``OPENAI_API_KEY``. A test that does not build the dense index is simply
    unaffected by the patch; a test wanting the handle requests it by name.
    """
    fake = _FakeDenseEmbeddings()
    monkeypatch.setattr(hk_dense, "get_embeddings", lambda: fake)
    yield fake
    hk_dense.vectorstore = None
    hk_dense.sections_indexed = 0


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
    # Stack C's dense-over-wiki index persists to its own committed seed dir
    # (.kb/hybrid_dense/); redirect it (and hybrid_kb's log) to tmp so an
    # index_stack_c() call in a test never writes the production seed (#316 / #307).
    monkeypatch.setattr(hk_dense, "DENSE_INDEX_DIR", tmp_path / ".kb" / "hybrid_dense")
    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")
    source_dirs_snapshot = mk_indexer.SOURCE_DIRS
    docs_dir_snapshot = vr_indexer.DOCS_DIR
    yield
    mk_indexer.SOURCE_DIRS = source_dirs_snapshot
    vr_indexer.DOCS_DIR = docs_dir_snapshot
    mk_indexer.sections.clear()
    vr_indexer.vectorstore = None
    hk_dense.vectorstore = None
    hk_dense.sections_indexed = 0
