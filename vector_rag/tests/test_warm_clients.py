"""RAG (Stack B) opt-in startup pings — ``retrieval.warm_llm_client`` and
``indexer.warm_embeddings_client`` (issue #439).

Hermetic: the LLM / embeddings singletons are monkeypatched to fakes so no
real OpenAI call is made. Mirrors ``markdown_kb/tests/test_warm_llm_client.py``
for Stack B's own two distinct clients (LLM + embeddings).
"""

from __future__ import annotations

import vector_rag.app.indexer as indexer_module
import vector_rag.app.logger as logger_module
import vector_rag.app.retrieval as retrieval_module


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
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake)

    retrieval_module.warm_llm_client()

    assert len(fake.calls) == 1
    args, kwargs = fake.calls[0]
    assert args == ("Hi",)
    assert kwargs.get("max_tokens") == 1


def test_warm_llm_client_swallows_failure_and_logs_it(monkeypatch, tmp_path):
    log_path = tmp_path / "vector_rag" / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(
        retrieval_module, "get_llm", lambda: _FakeLLM(raises=RuntimeError("boom"))
    )

    retrieval_module.warm_llm_client()  # must not raise

    text = log_path.read_text(encoding="utf-8")
    assert "client=rag_llm status=failed exc=RuntimeError" in text


# ---------------------------------------------------------------------------
# warm_embeddings_client
# ---------------------------------------------------------------------------


def test_warm_embeddings_client_pings_the_singleton(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(indexer_module, "get_embeddings", lambda: fake)

    indexer_module.warm_embeddings_client()

    assert fake.calls == ["hi"]


def test_warm_embeddings_client_logs_success(monkeypatch, tmp_path):
    log_path = tmp_path / "vector_rag" / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(indexer_module, "get_embeddings", lambda: _FakeEmbeddings())

    indexer_module.warm_embeddings_client()

    text = log_path.read_text(encoding="utf-8")
    assert "client=rag_embeddings status=ok" in text


def test_warm_embeddings_client_swallows_failure_and_logs_it(monkeypatch, tmp_path):
    log_path = tmp_path / "vector_rag" / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(
        indexer_module,
        "get_embeddings",
        lambda: _FakeEmbeddings(
            raises=RuntimeError("OPENAI_API_KEY is not set in the server environment")
        ),
    )

    indexer_module.warm_embeddings_client()  # must not raise

    text = log_path.read_text(encoding="utf-8")
    assert "client=rag_embeddings status=failed exc=RuntimeError" in text
