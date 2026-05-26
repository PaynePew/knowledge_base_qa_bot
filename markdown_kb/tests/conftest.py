"""Shared fixtures for the markdown_kb test suite.

Provides:
  - REAL_DOCS:            absolute path to docs/ for tests that index real content
  - FakeLLMResponse:      single canonical shape for LLM stubs (kills the
                          class-attribute-mutation landmine across the suite)
  - _redirect_paths_to_tmp (autouse): redirects INDEX_PATH and LOG_PATH to
                          tmp so no test can pollute the real .kb/ or
                          wiki/log.md, even if the test itself forgets
  - indexed_corpus:       builds the real Section Index against REAL_DOCS into
                          the tmp paths set up by the autouse redirect
  - pytest_collection_modifyitems: skip @pytest.mark.live unless -m live

Also loads .env at the very top so live tests pick up OPENAI_API_KEY
the same way uvicorn does via app.main.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

# Load .env BEFORE importing app modules — mirrors what app.main does for the
# running server. find_dotenv walks up from cwd (markdown_kb/ when pytest runs)
# and locates the repo-root .env. Without this, live tests that read
# OPENAI_API_KEY at test-function start fail before app.main has a chance to
# call load_dotenv itself.
load_dotenv(find_dotenv(usecwd=True))

import app.indexer as _indexer  # noqa: E402
import app.logger as _logger  # noqa: E402

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


@dataclass(frozen=True)
class FakeLLMResponse:
    """Canonical LLM response shape for fakes.

    langchain_core message objects expose a `.content` attribute; this dataclass
    mirrors that shape. Using a single frozen dataclass instead of the
    per-file `class _Resp: pass` + `_Resp.content = ...` pattern avoids a
    latent landmine — if the inner class were ever lifted out of the method
    scope, the class-attribute mutation would leak across calls.
    """

    content: str


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.live tests unless they were explicitly selected with -m live."""
    marker_expr = config.option.markexpr if hasattr(config.option, "markexpr") else ""
    if "live" in marker_expr:
        return
    skip_live = pytest.mark.skip(reason="live test — run with: pytest -m live")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Autouse safety net: redirect INDEX_PATH and LOG_PATH to tmp.

    Without this, any test that calls build_index() or log_event() without
    its own monkeypatch (notably test_indexing.py and test_logger.py callers
    of parse_markdown's parse_warning path) pollutes the real .kb/index.json
    and wiki/log.md. Tests that do their own patching compose fine —
    monkeypatch applies in order and the test's setattr wins. Tests that
    reload modules (test_persistence) bypass this entirely, which is also fine.
    """
    monkeypatch.setattr(_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")


@pytest.fixture()
def indexed_corpus(tmp_path):
    """Build the section index from REAL_DOCS into the tmp paths.

    Relies on the autouse `_redirect_paths_to_tmp` fixture for path setup.
    Yields a dict with the log_path so tests can read it back. Clears the
    in-memory sections list on teardown so tests don't bleed into each other.
    """
    _indexer.build_index(REAL_DOCS)
    yield {"log_path": _logger.LOG_PATH}
    _indexer.sections.clear()


@pytest.fixture()
def tmp_docs(tmp_path):
    """Create a minimal docs/ directory for testing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    return docs


@pytest.fixture()
def tmp_kb(tmp_path):
    """Provide a tmp .kb directory path."""
    return tmp_path / ".kb"


@pytest.fixture()
def tmp_wiki(tmp_path):
    """Provide a tmp wiki directory path."""
    return tmp_path / "wiki"
