"""Shared fixtures for the gateway test suite.

Provides:
  - pytest_collection_modifyitems: skip @pytest.mark.live unless -m live

Also loads .env so live tests pick up OPENAI_API_KEY, mirroring
how gateway.app.main does it via load_dotenv.
"""

from __future__ import annotations

import pytest
from dotenv import find_dotenv, load_dotenv

# Load .env BEFORE imports — mirrors gateway.app.main.
load_dotenv(find_dotenv(usecwd=True))

# Import the markdown_kb wiki modules via their workspace namespace so the
# autouse redirect below patches the SAME module objects gateway production
# code calls (build_index / log_event read these attributes at call time).
import markdown_kb.app.indexer as _mk_indexer  # noqa: E402
import markdown_kb.app.logger as _mk_logger  # noqa: E402


@pytest.fixture(autouse=True)
def _redirect_wiki_paths_to_tmp(tmp_path, monkeypatch):
    """Suite-wide safety net: redirect the markdown_kb wiki write paths to tmp.

    The gateway app exposes admin/chat routes that drive the wiki ``build_index()``
    (e.g. ``POST /wiki/index``) and ``log_event()`` against the production
    ``WIKI_DIR`` / ``INDEX_PATH`` / ``LOG_PATH``. Several gateway test FILES define
    their own per-file redirect fixture, but any file that hits a real wiki write
    path *without* one leaks the committed ``wiki/index.md`` + ``wiki/log.md`` (and
    ``.kb/index.json``) — #303 traced this to ``test_prod_middleware.py`` POSTing
    ``/wiki/index`` in open mode. Redirecting here at the conftest level makes EVERY
    gateway test hermetic, not just the files that remembered (CODING_STANDARD §6.5).

    Composes cleanly with the per-file ``_redirect_paths_to_tmp`` fixtures: both are
    autouse monkeypatch stacks pointing at the same ``tmp_path`` subdirs, so the
    duplication is harmless. ``vector_rag`` redirection stays in the per-file rag
    fixtures — the #303 leak is wiki-specific.
    """
    monkeypatch.setattr(_mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_mk_indexer, "WIKI_DIR", tmp_path / "wiki")


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
def _neutralize_rag_distance_gate(monkeypatch):
    """Disable the calibrated RAG distance gate for the gateway suite.

    The RAG distance gate ships ON by default (calibrated ceiling 1.1 against REAL
    text-embedding-3-small distances — eval/rag_distance). Gateway RAG-stream tests
    build the index with offline fake embeddings whose hash-derived distances are not
    comparable to that ceiling, so an enabled gate would refuse every fake query and
    mask the grounded-answer / SSE-ordering assertions. A large permissive ceiling
    neutralises it (the wiki BM25 path is unaffected). The gate's own behaviour is
    covered in vector_rag/tests/test_rag_distance_gate.py.
    """
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "1000.0")
