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

# Hermetic 3-doc sample corpus (account_help / refund_policy / shipping_faq),
# NOT the live repo ``docs/`` — since #142 that tree also holds the 20-doc
# ``docs/fake-docs/`` demo corpus, which pollutes BM25 rankings these tests
# assert (issue #145: a richer regenerated fake-docs doc out-ranked the intended
# sample Section). The fixture files are byte-identical copies of the originals.
REAL_DOCS = Path(__file__).resolve().parent / "fixtures" / "docs"


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
def _mock_ingest_verifier_supported(request, monkeypatch):
    """Default ``app.ingest.verify`` to ``claim_supported`` for hermetic tests.

    Why this exists (issue #42): ``grounding.verify()`` constructs a fresh
    ``ChatOpenAI`` instance per call (see ``markdown_kb/app/grounding.py``),
    so it is **not** covered by fixtures that mock ``templates._ingest_llm``.
    Without this autouse default, every test that exercises ``/ingest`` (or
    calls ``ingest_sources`` directly) silently hits the real OpenAI API for
    the grounding check — non-hermetic, costs money, and is non-deterministic
    (the verifier's "claim supported / unsupported" judgment flakes around
    ~10 % even for the fake ``FIXED_BODY``). The most visible symptom was
    ``test_ingest_creates_wiki_page_with_correct_structure`` intermittently
    failing on ``parsed["status"] == "live"``.

    Tests that want a different verifier outcome override this with their
    own ``with patch("app.ingest.verify", return_value=other_outcome):``
    context manager — ``unittest.mock.patch`` composes correctly on top of
    the autouse monkeypatch (the inner patch wins inside the ``with`` block,
    and the autouse default is restored on exit).

    Live tests (``@pytest.mark.live``) opt out so they continue to exercise
    the real verifier end-to-end.
    """
    if request.node.get_closest_marker("live"):
        return

    from app.grounding import GroundingOutcome

    supported = GroundingOutcome(passed=True, reason="claim_supported", result=None)

    # ``app.ingest`` may not be imported yet for non-ingest tests; importing
    # it lazily here keeps the autouse cost negligible.
    import app.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "verify", lambda *_a, **_kw: supported)


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Autouse safety net: redirect INDEX_PATH, LOG_PATH, and WIKI_DIR to tmp.

    Without this, any test that calls build_index() or log_event() without
    its own monkeypatch (notably test_indexing.py and test_logger.py callers
    of parse_markdown's parse_warning path) pollutes the real .kb/index.json,
    wiki/log.md, and wiki/index.md (since Slice #2 of Phase 2 wires
    write_wiki_index into build_index). Tests that do their own patching
    compose fine — monkeypatch applies in order and the test's setattr wins.

    Modules are re-imported inside the fixture (rather than relying on the
    module-level ``_indexer`` / ``_logger`` bindings) so the patch targets the
    *current* sys.modules entry. ``test_persistence`` removes ``app.*`` from
    sys.modules and re-imports them; subsequent tests in the same session
    would otherwise patch the stale (pre-reload) module objects, leaking the
    real WIKI_DIR / INDEX_PATH into production paths.
    """
    import app.indexer as current_indexer
    import app.logger as current_logger

    monkeypatch.setattr(current_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(current_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(current_indexer, "WIKI_DIR", tmp_path / "wiki")


@pytest.fixture(autouse=True)
def _reset_verifier_llm(monkeypatch):
    """Reset the grounding verifier's lazy LLM singleton between tests.

    ADR-0042 / issue #572 hoisted ``grounding.verify()``'s per-call
    ``ChatOpenAI(...)`` construction into a lazy-singleton getter
    (``get_verifier_llm``, CODING_STANDARD §2.7) — before this, a fresh client
    was built on every call. Several existing tests (``grounding/test_retry.py``,
    ``grounding/test_verifier.py``, ``test_reconcile.py``, ``test_collision.py``,
    ``test_llm_determinism.py``) patch ``app.grounding.ChatOpenAI`` directly and
    rely on that per-call construction so their patch takes effect. Resetting
    the cached singleton before every test preserves that isolation without
    rewriting each patch call site to target the new getter instead (mirrors
    ``gateway/tests/test_query_rewriting.py``'s ``_reset_rewrite_llm`` for the
    analogous rewriter singleton).

    Imports ``app.grounding`` fresh inside the fixture (not at module scope) so
    the patch targets the *current* sys.modules entry — same reload-safety
    reasoning as ``_redirect_paths_to_tmp`` above.
    """
    import app.grounding as current_grounding

    monkeypatch.setattr(current_grounding, "_verifier_llm", None)


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
