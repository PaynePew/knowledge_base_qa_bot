"""Gateway startup warmup (issue #439) — fixes the post-deploy /chat cold start.

Two independent fixes, both exercised here:
  - AC1: the Hybrid (Stack C) dense index is loaded at Gateway startup,
    unconditional and token-free — closing the parity gap issue #398 left for
    ``hybrid_kb`` (library-only, no sub-app lifespan of its own).
  - AC2/AC3: ``KB_WARMUP_PING`` gates one tiny ping per distinct OpenAI client
    (3 LLM + 2 embeddings). Unset/false (the default) fires zero OpenAI calls
    at boot, preserving the reset workflow's pre-issue-#439 semantics; a
    truthy value pings every client.

Two layers of test:
  1. Unit tests against ``gateway.app.warmup`` directly (no ASGI lifespan) —
    fast, precise coverage of the flag parsing and the best-effort catch.
  2. Integration tests driving the real Gateway ``lifespan`` via
    ``with TestClient(gateway_app):`` (mirrors ``test_lifespan_cold_start.py``'s
    established cold-process-simulation pattern, but in this new file so that
    pre-existing file is never touched).
"""

from __future__ import annotations

import hashlib

import hybrid_kb.app.dense_index as dense_index
import hybrid_kb.app.logger as hk_logger
import hybrid_kb.app.query as hybrid_query_module
import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger
import markdown_kb.app.retrieval as mk_retrieval
import pytest
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
import vector_rag.app.retrieval as vr_retrieval
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from markdown_kb.app.indexer import Section

import gateway.app.logger as gateway_logger
import gateway.app.warmup as warmup_module

# ---------------------------------------------------------------------------
# Part 1 — unit tests against gateway.app.warmup (no ASGI lifespan)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "YES"])
def test_warmup_ping_enabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv("KB_WARMUP_PING", value)
    assert warmup_module.warmup_ping_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "", "banana"])
def test_warmup_ping_enabled_falsy_values(monkeypatch, value):
    monkeypatch.setenv("KB_WARMUP_PING", value)
    assert warmup_module.warmup_ping_enabled() is False


def test_warmup_ping_enabled_defaults_false_when_unset(monkeypatch):
    monkeypatch.delenv("KB_WARMUP_PING", raising=False)
    assert warmup_module.warmup_ping_enabled() is False


def test_warm_client_fns_covers_three_llm_and_two_embeddings_clients():
    """Five distinct clients per the issue: wiki/rag/hybrid LLM + rag/hybrid embeddings."""
    assert len(warmup_module._WARM_CLIENT_FNS) == 5


def test_warm_openai_clients_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("KB_WARMUP_PING", raising=False)
    calls: list[int] = []
    monkeypatch.setattr(warmup_module, "_WARM_CLIENT_FNS", (lambda: calls.append(1),))

    warmup_module.warm_openai_clients()

    assert calls == []


def test_warm_openai_clients_calls_every_client_when_enabled(monkeypatch):
    monkeypatch.setenv("KB_WARMUP_PING", "true")
    calls: list[int] = []
    fakes = tuple((lambda n=i: calls.append(n)) for i in range(5))
    monkeypatch.setattr(warmup_module, "_WARM_CLIENT_FNS", fakes)

    warmup_module.warm_openai_clients()

    assert calls == [0, 1, 2, 3, 4]


def test_warm_hybrid_indexes_calls_ensure_indexes_loaded(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-457")
    calls: list[int] = []
    monkeypatch.setattr(warmup_module, "_ensure_hybrid_indexes_loaded", lambda: calls.append(1))

    warmup_module.warm_hybrid_indexes()

    assert calls == [1]


def test_warm_hybrid_indexes_skips_and_logs_when_key_absent(monkeypatch, tmp_path):
    """Issue #457 AC1/AC3: a keyless boot SKIPS the hybrid warmup (does not even
    attempt the index load) and logs ``status=skipped`` — the boot stays green.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(gateway_logger, "LOG_PATH", log_path)

    calls: list[int] = []
    monkeypatch.setattr(warmup_module, "_ensure_hybrid_indexes_loaded", lambda: calls.append(1))

    warmup_module.warm_hybrid_indexes()  # must not raise

    assert calls == [], "keyless boot must not attempt the index load at all"
    text = log_path.read_text(encoding="utf-8")
    assert "startup_warmup" in text
    assert "target=hybrid_dense_index status=skipped reason=no_openai_api_key" in text


def test_warm_hybrid_indexes_propagates_failure_when_key_present(monkeypatch):
    """Issue #457 AC2: with a key present, a seed-load failure PROPAGATES —
    it is no longer caught and logged, so the Gateway boot fails fast instead
    of reporting healthy on a broken hybrid stack.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-457")

    def _raise() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(warmup_module, "_ensure_hybrid_indexes_loaded", _raise)

    with pytest.raises(RuntimeError, match="boom"):
        warmup_module.warm_hybrid_indexes()


# ---------------------------------------------------------------------------
# Part 2 — integration: the real Gateway lifespan, cold-process simulation
# ---------------------------------------------------------------------------


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings (build/load only)."""

    _DIM = 8

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class _SpyLLM:
    """Records .invoke() calls without making a network call."""

    def __init__(self) -> None:
        self.invoke_calls = 0

    def invoke(self, *args, **kwargs):
        self.invoke_calls += 1
        return None


class _SpyEmbeddings:
    """Records .embed_query() calls without making a network call."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_query(self, *args, **kwargs):
        self.embed_calls += 1
        return [0.0]


@pytest.fixture()
def cold_gateway_env(tmp_path, monkeypatch):
    """Cold-process simulation across all three stacks (issue #439).

    Wiki/BM25: an empty tmp wiki dir (no persisted index.json) — the sub-app's
    own lifespan's ``load_index_json()`` no-ops, leaving ``sections`` empty;
    irrelevant to this file's assertions.
    RAG: ``FAISS_INDEX_DIR`` points at a non-existent tmp dir — the sub-app's
    own lifespan's ``load_vector_index()`` returns ``(0, 0)`` without ever
    reaching ``get_embeddings()`` (belt-and-suspenders fake installed anyway,
    mirroring the established ``test_lifespan_cold_start.py`` pattern).
    Hybrid: build + persist a REAL dense index (offline fake embeddings) over
    one synthetic Section, then clear the in-memory ``vectorstore`` to
    simulate a fresh process — the ONLY way it can come back is the Gateway
    lifespan's ``warm_hybrid_indexes()`` under test. ``OPENAI_API_KEY`` is set
    to a dummy value so the warmup actually runs deterministically (issue
    #457 gates it on key presence) regardless of whatever the host machine's
    own ``.env`` does or does not set.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-457-cold-gateway")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(mk_indexer, "WIKI_DIR", tmp_path / "wiki")

    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: _FakeEmbeddings())

    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")
    monkeypatch.setattr(dense_index, "DENSE_INDEX_DIR", tmp_path / ".kb" / "hybrid_dense")
    monkeypatch.setattr(dense_index, "get_embeddings", lambda: _FakeEmbeddings())

    section = Section(
        id="alpha#window",
        file="alpha.md",
        heading="window",
        heading_path=["Alpha", "Window"],
        content="Refunds are processed within five business days.",
        tokens=[],
        metadata={},
    )
    dense_index.build_index(sections=[section])
    assert dense_index.DENSE_INDEX_DIR.exists(), "dense seed must be persisted after build"

    # Simulate a fresh process: no /hybrid/index or /chat has run yet.
    dense_index.vectorstore = None
    assert dense_index.vectorstore is None

    yield

    dense_index.vectorstore = None
    dense_index.sections_indexed = 0


def test_cold_gateway_warms_hybrid_dense_index_on_startup(cold_gateway_env):
    """AC1: a cold gateway TestClient populates the hybrid dense index with no
    ``/chat`` or ``/hybrid/index`` call.
    """
    from gateway.app.main import app as gateway_app

    with TestClient(gateway_app):
        assert dense_index.vectorstore is not None, (
            "hybrid_kb's dense index must be populated by the Gateway lifespan "
            "alone — no /chat or /hybrid/index call was made"
        )


def test_cold_gateway_fails_boot_on_corrupt_hybrid_seed_when_key_present(tmp_path, monkeypatch):
    """Issue #457 AC2: a key is present AND the committed hybrid dense seed is
    corrupt (present ``.kb/hybrid_dense/`` dir, missing ``metadata.json`` —
    ``load_dense_index``'s own documented fail-fast trigger) -> the Gateway
    lifespan raises and ``TestClient`` never finishes entering, so the process
    never reaches ``yield`` and never reports healthy — closing the gap where
    this used to boot "healthy" and 500 on the first real ``stack=hybrid``
    query instead.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-457-corrupt-seed")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(mk_indexer, "WIKI_DIR", tmp_path / "wiki")

    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: _FakeEmbeddings())

    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")
    corrupt_dense_dir = tmp_path / ".kb" / "hybrid_dense"
    corrupt_dense_dir.mkdir(parents=True)  # present dir, no metadata.json -> fail-fast
    monkeypatch.setattr(dense_index, "DENSE_INDEX_DIR", corrupt_dense_dir)
    monkeypatch.setattr(dense_index, "get_embeddings", lambda: _FakeEmbeddings())
    dense_index.vectorstore = None

    from gateway.app.main import app as gateway_app

    with pytest.raises(RuntimeError, match="metadata.json"), TestClient(gateway_app):
        pass

    dense_index.vectorstore = None


@pytest.fixture()
def spy_openai_clients(cold_gateway_env, monkeypatch):
    """Install call-counting spies behind all 5 lazy-singleton getters."""
    spies = {
        "wiki_llm": _SpyLLM(),
        "rag_llm": _SpyLLM(),
        "rag_embeddings": _SpyEmbeddings(),
        "hybrid_llm": _SpyLLM(),
        "hybrid_embeddings": _SpyEmbeddings(),
    }
    monkeypatch.setattr(mk_retrieval, "get_llm", lambda: spies["wiki_llm"])
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: spies["rag_llm"])
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: spies["rag_embeddings"])
    monkeypatch.setattr(hybrid_query_module, "get_llm", lambda: spies["hybrid_llm"])
    monkeypatch.setattr(dense_index, "get_embeddings", lambda: spies["hybrid_embeddings"])
    return spies


def test_cold_gateway_zero_openai_calls_when_flag_unset(spy_openai_clients, monkeypatch):
    """AC3: flag unset/false -> boot makes zero OpenAI calls (unchanged semantics)."""
    monkeypatch.delenv("KB_WARMUP_PING", raising=False)
    from gateway.app.main import app as gateway_app

    with TestClient(gateway_app):
        pass

    assert spy_openai_clients["wiki_llm"].invoke_calls == 0
    assert spy_openai_clients["rag_llm"].invoke_calls == 0
    assert spy_openai_clients["rag_embeddings"].embed_calls == 0
    assert spy_openai_clients["hybrid_llm"].invoke_calls == 0
    assert spy_openai_clients["hybrid_embeddings"].embed_calls == 0


def test_cold_gateway_pings_every_client_when_flag_enabled(spy_openai_clients, monkeypatch):
    """AC2: flag on -> exactly one ping per distinct client, no more."""
    monkeypatch.setenv("KB_WARMUP_PING", "true")
    from gateway.app.main import app as gateway_app

    with TestClient(gateway_app):
        pass

    assert spy_openai_clients["wiki_llm"].invoke_calls == 1
    assert spy_openai_clients["rag_llm"].invoke_calls == 1
    assert spy_openai_clients["rag_embeddings"].embed_calls == 1
    assert spy_openai_clients["hybrid_llm"].invoke_calls == 1
    assert spy_openai_clients["hybrid_embeddings"].embed_calls == 1


def test_cold_gateway_survives_a_failing_client_ping(cold_gateway_env, monkeypatch):
    """A single client raising during warmup must not crash Gateway startup
    or block the remaining pings (best-effort, issue #439).
    """
    monkeypatch.setenv("KB_WARMUP_PING", "true")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated OpenAI outage")

    monkeypatch.setattr(mk_retrieval, "get_llm", lambda: type("X", (), {"invoke": _raise})())
    rag_spy = _SpyLLM()
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: rag_spy)

    from gateway.app.main import app as gateway_app

    with TestClient(gateway_app):  # must not raise despite the wiki client failing
        pass

    assert rag_spy.invoke_calls == 1, "a failing client must not block the remaining pings"
