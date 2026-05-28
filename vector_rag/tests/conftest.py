"""Shared fixtures for the vector_rag (Stack B) test suite.

The default suite runs OFFLINE (no OPENAI_API_KEY). Stack B's only network
dependencies are the OpenAI embedding call (inside ``FAISS.from_documents`` /
``FAISS.load_local``) and the answer-synthesis ``ChatOpenAI`` call. Two seams
keep the suite hermetic (CODING_STANDARD §6.3 — mock the network leaf, not the
deep retrieval module):

  * ``fake_embeddings`` monkeypatches ``indexer.get_embeddings`` to return a
    deterministic, hash-based ``Embeddings`` instance. The REAL FAISS build /
    save / load / search path runs unchanged — only the embedding leaf is faked,
    so the persistence roundtrip exercises actual ``FAISS.save_local`` /
    ``load_local``.
  * Tests that hit ``/chat`` inject a fake ``ChatOpenAI`` via ``get_llm`` and a
    fake ``grounding.verify`` (the verifier constructs its own ChatOpenAI, so it
    must be patched separately).

Path redirection (autouse) points ``indexer.FAISS_INDEX_DIR``, vector_rag's
``LOG_PATH``, and markdown_kb's ``LOG_PATH`` (the grounding module writes there)
at tmp so no test pollutes production ``.kb/`` / ``vector_rag/log.md`` /
``wiki/log.md``.

Also loads .env at the top so the one live test picks up OPENAI_API_KEY the
same way uvicorn does via vector_rag.app.main.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(find_dotenv(usecwd=True))

import markdown_kb.app.logger as mk_logger  # noqa: E402
import vector_rag.app.indexer as vr_indexer  # noqa: E402
import vector_rag.app.logger as vr_logger  # noqa: E402

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


@dataclass(frozen=True)
class FakeLLMResponse:
    """Canonical LLM response shape for fakes (mirrors a langchain message's .content)."""

    content: str


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings.

    Maps text to a fixed-length vector derived from a SHA-256 digest so the
    real FAISS index build / persistence / similarity search runs without any
    network call. Stable across processes, so a save-then-load roundtrip
    returns the same neighbours.
    """

    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.live tests unless explicitly selected with -m live."""
    marker_expr = config.option.markexpr if hasattr(config.option, "markexpr") else ""
    if "live" in marker_expr:
        return
    skip_live = pytest.mark.skip(reason="live test — run with: pytest -m live")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Autouse safety net: keep every index build / log write off production paths.

    Redirects vector_rag's FAISS_INDEX_DIR + LOG_PATH and markdown_kb's LOG_PATH
    (grounding.verify writes there) to tmp. Resets the in-memory index globals
    on teardown so tests never bleed into each other.
    """
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    yield
    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0


@pytest.fixture()
def fake_embeddings(monkeypatch):
    """Swap vector_rag's embeddings leaf for the deterministic offline fake.

    Patches ``get_embeddings`` (not ``_build_faiss``) so the whole real FAISS
    path — build, save_local, load_local, similarity search — runs offline.
    """
    fake = _FakeEmbeddings()
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: fake)
    return fake


@pytest.fixture()
def indexed_corpus(fake_embeddings):
    """Build the FAISS index from the real docs/ corpus using fake embeddings.

    Relies on the autouse path redirect so the persisted index lands in tmp.
    """
    vr_indexer.build_index(REAL_DOCS)
    yield
    vr_indexer.vectorstore = None
