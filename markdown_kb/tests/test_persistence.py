"""Integration tests for Slice 5 — Server restart preserves the Section Index.

Tests translate every acceptance criterion from issue #6 directly into
executable assertions. All tests simulate a server restart by building
an index in one app instance and then creating a fresh app instance
(forcing module reload) to verify the persisted state is rehydrated.

The startup event in FastAPI fires when TestClient is used as a context
manager (``with TestClient(app) as client:``). Tests use that pattern so
the lifespan startup hook runs as it would in a real server restart.

Run with:
    pytest -m "not live"   (from markdown_kb/)
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.retrieval import NOT_INDEXED_MESSAGE

from .conftest import REAL_DOCS, FakeLLMResponse

# The section that the happy-path refund query should return
REFUND_SECTION_ID = "refund_policy.md#refund-timeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLLM:
    """Minimal LLM stub for grounded responses."""

    def __init__(self, source_id: str = REFUND_SECTION_ID):
        self.source_id = source_id

    def invoke(self, messages: list):
        return FakeLLMResponse(
            content=(
                f"Approved refunds are processed within 5-7 business days. "
                f"[Source: {self.source_id}]"
            )
        )


def _make_approved_outcome(source_id: str):
    """Build a passed GroundingOutcome for the canned FakeLLM draft.

    Mirrors the helper in ``test_wiki_index_route.py`` so the /chat path can
    run without constructing a real ``ChatOpenAI`` verifier (which would need
    OPENAI_API_KEY). Imported lazily so test_persistence's sys.modules reload
    does not leave stale references.
    """
    from app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims trace to the cited section.",
            claims=[
                GroundingClaim(
                    text="Approved refunds are processed within 5-7 business days.",
                    supported=True,
                    citing_section_ids=[source_id],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


def _reload_app_modules(monkeypatch, index_path: Path, log_path: Path):
    """Remove cached app modules from sys.modules so that module-level globals
    (sections, doc_freq, etc.) are re-initialised, simulating a real server
    restart within a single process.

    Patches INDEX_PATH, LOG_PATH, and WIKI_DIR on the freshly-imported modules
    so any IO performed during the simulated startup / build is redirected to
    ``tmp_path`` and does not pollute the production working tree. ``wiki_dir``
    is derived from ``log_path.parent`` to keep the helper's signature stable.

    Returns (app, indexer_module, logger_module, retrieval_module).
    """
    for key in list(sys.modules.keys()):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]

    import app.indexer as new_indexer
    import app.logger as new_logger
    import app.retrieval as new_retrieval

    # Patch paths *before* importing app.main so the startup hook sees the
    # right INDEX_PATH. WIKI_DIR is patched on the indexer module; the
    # wiki_index module reads it at call time so this propagates correctly.
    monkeypatch.setattr(new_indexer, "INDEX_PATH", index_path)
    monkeypatch.setattr(new_logger, "LOG_PATH", log_path)
    monkeypatch.setattr(new_indexer, "WIKI_DIR", log_path.parent)

    import app.main as new_main

    return new_main.app, new_indexer, new_logger, new_retrieval


# ---------------------------------------------------------------------------
# AC 1: After /index + simulated restart, /chat answers grounded without
#        a second /index call
# ---------------------------------------------------------------------------


def test_restart_preserves_index_and_chat_answers(tmp_path, monkeypatch):
    """Build the index in one app instance; create a fresh instance and verify
    /chat returns a grounded answer without calling /index again."""
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"
    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"

    # ---- Phase 1: first app instance — build the index ----
    import app.indexer as indexer
    import app.logger as logger_module
    import app.retrieval as retrieval_module

    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(indexer, "WIKI_DIR", wiki_dir)

    from app.main import app as first_app

    fake = FakeLLM()
    monkeypatch.setattr(retrieval_module, "_llm", fake)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake)

    with TestClient(first_app) as client1:
        resp_index = client1.post("/index")
        assert resp_index.status_code == 200, f"/index failed: {resp_index.text}"

    assert index_path.exists(), ".kb/index.json must exist after /index"

    # Clean up in-memory state so phase 2 starts fresh
    indexer.sections.clear()

    # ---- Phase 2: fresh app instance — simulate restart ----
    fresh_app, fresh_indexer, fresh_logger, fresh_retrieval = _reload_app_modules(
        monkeypatch, index_path, log_path
    )

    fresh_fake = FakeLLM()
    monkeypatch.setattr(fresh_retrieval, "_llm", fresh_fake)
    monkeypatch.setattr(fresh_retrieval, "get_llm", lambda: fresh_fake)
    # Bypass the real grounding verifier — it would otherwise construct
    # ChatOpenAI(model=...) inside grounding.verify() and fail with
    # "Missing credentials" when OPENAI_API_KEY is unset (e.g. CI without
    # the secret, or local runs without the .env loaded). Same pattern as
    # test_wiki_index_route.py::test_chat_works_after_wiki_failure_index.
    monkeypatch.setattr(
        fresh_retrieval.grounding_module,
        "verify",
        lambda draft, sections: _make_approved_outcome(REFUND_SECTION_ID),
    )

    with TestClient(fresh_app) as client2:
        # Startup hook should have loaded sections before any request
        assert fresh_indexer.sections, (
            "sections must be non-empty after startup load from index.json"
        )

        # /chat must succeed without calling /index again
        resp_chat = client2.post("/chat", json={"query": "How long do refunds take?"})
        assert resp_chat.status_code == 200, (
            f"Expected 200, got {resp_chat.status_code}: {resp_chat.text}"
        )
        body = resp_chat.json()
        assert "answer" in body
        assert "sources" in body

        # Sources must include the refund section
        source_ids = [s["source"] for s in body["sources"]]
        assert REFUND_SECTION_ID in source_ids, (
            f"sources must contain '{REFUND_SECTION_ID}' after restart, got: {source_ids}"
        )

    # Clean up
    fresh_indexer.sections.clear()


# ---------------------------------------------------------------------------
# AC 2: Corrupt .kb/index.json → startup raises / TestClient fails on first request
# ---------------------------------------------------------------------------


def test_corrupt_index_json_raises_at_startup(tmp_path, monkeypatch):
    """If .kb/index.json exists but is invalid JSON, startup must fail visibly.

    The spec says load_index_json re-raises; the FastAPI app should raise at
    startup (inside the TestClient context manager __enter__).
    """
    kb_dir = tmp_path / ".kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    index_path = kb_dir / "index.json"
    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"

    # Write corrupt JSON (truncated)
    index_path.write_text('{"sections": [{"id": "broken"', encoding="utf-8")

    fresh_app, fresh_indexer, fresh_logger, fresh_retrieval = _reload_app_modules(
        monkeypatch, index_path, log_path
    )

    # The startup hook should raise JSONDecodeError, causing TestClient's
    # context manager __enter__ to propagate it.
    with (
        pytest.raises(json.JSONDecodeError),
        TestClient(fresh_app, raise_server_exceptions=True),
    ):
        pass  # startup fires in __enter__; corrupt JSON must raise here

    # Clean up
    fresh_indexer.sections.clear()


# ---------------------------------------------------------------------------
# AC 3: No .kb/index.json → server starts normally, /chat returns not-indexed
# ---------------------------------------------------------------------------


def test_missing_index_json_server_starts_and_chat_returns_not_indexed(tmp_path, monkeypatch):
    """With no .kb/index.json present the server starts fine and /chat returns
    the 'knowledge base has not been indexed yet' response."""
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"
    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"

    # Do NOT create index_path — it should not exist
    assert not index_path.exists(), "Pre-condition: index.json must not exist"

    fresh_app, fresh_indexer, fresh_logger, fresh_retrieval = _reload_app_modules(
        monkeypatch, index_path, log_path
    )

    with TestClient(fresh_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200, f"/health must return 200, got {resp.status_code}"

        resp_chat = client.post("/chat", json={"query": "How long do refunds take?"})
        assert resp_chat.status_code == 200
        body = resp_chat.json()
        assert body["answer"] == NOT_INDEXED_MESSAGE, (
            f"Expected NOT_INDEXED_MESSAGE, got: {body['answer']!r}"
        )
        assert body["sources"] == [], f"Expected sources == [], got: {body['sources']}"

    # Clean up
    fresh_indexer.sections.clear()


# ---------------------------------------------------------------------------
# AC 4: wiki/log.md includes index_loaded | files=N sections=M on startup-load
# ---------------------------------------------------------------------------


def test_startup_load_writes_index_loaded_log_entry(tmp_path, monkeypatch):
    """On successful startup-load from .kb/index.json, wiki/log.md must contain
    an 'index_loaded | files=N sections=M' entry."""
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"
    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"

    # ---- Phase 1: build index with a first app instance ----
    import app.indexer as indexer
    import app.logger as logger_module

    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(indexer, "WIKI_DIR", wiki_dir)

    from app.main import app as first_app

    with TestClient(first_app) as client1:
        client1.post("/index")
    assert index_path.exists()

    indexer.sections.clear()

    # ---- Phase 2: fresh restart — the startup hook must log index_loaded ----
    fresh_app, fresh_indexer, fresh_logger, fresh_retrieval = _reload_app_modules(
        monkeypatch, index_path, log_path
    )

    with TestClient(fresh_app):
        # Startup fires in __enter__; just verify the log after context entry
        pass

    assert log_path.exists(), "wiki/log.md must exist after startup-load"
    content = log_path.read_text(encoding="utf-8")

    assert "index_loaded |" in content, f"Expected 'index_loaded |' entry in log, got:\n{content}"
    # Must contain files=N sections=M pattern
    assert re.search(r"files=\d+", content), f"Expected 'files=N' in log entry, got:\n{content}"
    assert re.search(r"sections=\d+", content), (
        f"Expected 'sections=M' in log entry, got:\n{content}"
    )

    # Clean up
    fresh_indexer.sections.clear()
