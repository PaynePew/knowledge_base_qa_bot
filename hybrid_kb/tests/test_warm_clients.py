"""Hybrid (Stack C) opt-in startup pings — ``query.warm_llm_client`` and
``dense_index.warm_embeddings_client`` (issue #439).

Hermetic: the LLM / embeddings singletons are monkeypatched to fakes so no
real OpenAI call is made. Mirrors ``markdown_kb/tests/test_warm_llm_client.py``
and ``vector_rag/tests/test_warm_clients.py`` for Stack C's own two distinct
clients (LLM + embeddings). The autouse ``_redirect_paths_to_tmp`` conftest
fixture keeps ``hybrid_kb/log.md`` writes off the real file.
"""

from __future__ import annotations

import hybrid_kb.app.dense_index as dense_index_module
import hybrid_kb.app.logger as logger_module
import hybrid_kb.app.query as query_module


class _FakeLLM:
    def __init__(self, *, raises: Exception | None = None):
        self.calls: list[tuple[tuple, dict]] = []
        self._raises = raises

    def invoke(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._raises is not None:
            raise self._raises
        return None


class _FakeEmbeddings:
    def __init__(self, *, raises: Exception | None = None):
        self.calls: list[str] = []
        self._raises = raises

    def embed_query(self, text: str):
        self.calls.append(text)
        if self._raises is not None:
            raise self._raises
        return [0.0]


# ---------------------------------------------------------------------------
# warm_llm_client
# ---------------------------------------------------------------------------


def test_warm_llm_client_pings_the_singleton_with_bounded_tokens(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(query_module, "get_llm", lambda: fake)

    query_module.warm_llm_client()

    assert len(fake.calls) == 1
    args, kwargs = fake.calls[0]
    assert args == ("Hi",)
    assert kwargs.get("max_tokens") == 1


def test_warm_llm_client_swallows_failure_and_logs_it(monkeypatch):
    monkeypatch.setattr(
        query_module, "get_llm", lambda: _FakeLLM(raises=RuntimeError("boom"))
    )

    query_module.warm_llm_client()  # must not raise

    text = logger_module.LOG_PATH.read_text(encoding="utf-8")
    assert "client=hybrid_llm status=failed exc=RuntimeError" in text


# ---------------------------------------------------------------------------
# warm_embeddings_client
# ---------------------------------------------------------------------------


def test_warm_embeddings_client_pings_the_singleton(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(dense_index_module, "get_embeddings", lambda: fake)

    dense_index_module.warm_embeddings_client()

    assert fake.calls == ["hi"]


def test_warm_embeddings_client_logs_success(monkeypatch):
    monkeypatch.setattr(dense_index_module, "get_embeddings", lambda: _FakeEmbeddings())

    dense_index_module.warm_embeddings_client()

    text = logger_module.LOG_PATH.read_text(encoding="utf-8")
    assert "client=hybrid_embeddings status=ok" in text


def test_warm_embeddings_client_swallows_failure_and_logs_it(monkeypatch):
    monkeypatch.setattr(
        dense_index_module,
        "get_embeddings",
        lambda: _FakeEmbeddings(raises=RuntimeError("boom")),
    )

    dense_index_module.warm_embeddings_client()  # must not raise

    text = logger_module.LOG_PATH.read_text(encoding="utf-8")
    assert "client=hybrid_embeddings status=failed exc=RuntimeError" in text
