"""Shared fixtures for the hybrid_kb (Stack C) dense-index test suite (S1).

The default suite runs OFFLINE (no OPENAI_API_KEY). Stack C's only network
dependency in S1 is the OpenAI embedding call (inside ``FAISS.from_documents`` /
``FAISS.load_local``). One seam keeps the suite hermetic (CODING_STANDARD §6.3 —
mock the network leaf, not the deep retrieval module):

  * ``fake_embeddings`` monkeypatches ``dense_index.get_embeddings`` to return a
    deterministic, hash-based ``Embeddings`` instance. The REAL FAISS build /
    save / load / search path runs unchanged — only the embedding leaf is faked,
    so the persistence roundtrip exercises actual ``FAISS.save_local`` /
    ``load_local`` and the language filter runs over real FAISS metadata.

Path redirection (autouse) points ``dense_index.DENSE_INDEX_DIR``, hybrid_kb's
``LOG_PATH``, and markdown_kb's ``LOG_PATH`` (``parse_markdown`` /
``_passes_index_filter`` write parse warnings there) at tmp so no test pollutes
the committed ``.kb/hybrid_dense/`` seed, ``hybrid_kb/log.md``, or
``wiki/log.md``.

Also loads .env at the top so the one live test picks up OPENAI_API_KEY the
same way the bake driver does.
"""

from __future__ import annotations

import hashlib

import pytest
from dotenv import find_dotenv, load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(find_dotenv(usecwd=True))

import hybrid_kb.app.dense_index as dense_index  # noqa: E402
import hybrid_kb.app.logger as hk_logger  # noqa: E402
import markdown_kb.app.logger as mk_logger  # noqa: E402
from markdown_kb.app.indexer import Section  # noqa: E402


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings.

    Maps text to a fixed-length vector derived from a SHA-256 digest so the real
    FAISS index build / persistence / similarity search runs without any network
    call. Stable across processes, so a save-then-load roundtrip returns the same
    neighbours.
    """

    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_section(
    section_id: str,
    content: str,
    heading: str | None = None,
    heading_path: list[str] | None = None,
    file: str | None = None,
    lang: str | None = None,
) -> Section:
    """Build a synthetic ``Section`` for hermetic dense-index tests.

    ``lang`` defaults to ``None`` so the index-time tag is derived from content
    via ``_section_lang`` (the same path production uses); pass it explicitly to
    pin a language regardless of content.
    """
    heading = heading if heading is not None else section_id.split("#")[-1]
    metadata: dict = {} if lang is None else {"lang": lang}
    return Section(
        id=section_id,
        file=file if file is not None else section_id.split("#")[0],
        heading=heading,
        heading_path=heading_path if heading_path is not None else [heading],
        content=content,
        tokens=[],
        metadata=metadata,
    )


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

    Redirects hybrid_kb's DENSE_INDEX_DIR + LOG_PATH and markdown_kb's LOG_PATH
    (parse warnings land there) to tmp. Resets the in-memory index globals on
    teardown so tests never bleed into each other.
    """
    monkeypatch.setattr(
        dense_index, "DENSE_INDEX_DIR", tmp_path / ".kb" / "hybrid_dense"
    )
    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    # Keep the reranker (ADR-0019) OFF by default so the suite is deterministic
    # even if a dev's .env sets KB_HYBRID_RERANK — tests that exercise it opt in
    # by monkeypatching rerank.is_enabled / get_cross_encoder, never the real model.
    monkeypatch.delenv("KB_HYBRID_RERANK", raising=False)
    yield
    dense_index.vectorstore = None
    dense_index.sections_indexed = 0


@pytest.fixture()
def fake_embeddings(monkeypatch):
    """Swap hybrid_kb's embeddings leaf for the deterministic offline fake.

    Patches ``get_embeddings`` (not ``_build_faiss``) so the whole real FAISS
    path — build, save_local, load_local, similarity search — runs offline.
    """
    fake = _FakeEmbeddings()
    monkeypatch.setattr(dense_index, "get_embeddings", lambda: fake)
    return fake
