"""``retrieval.warm_llm_client`` — the Wiki stack's opt-in startup ping (issue #439).

Hermetic: ``get_llm`` is monkeypatched to a fake object so no real OpenAI call
is made. Covers:
  - the happy path fires ``.invoke("Hi", max_tokens=1)`` on the singleton and
    logs a ``startup_warmup`` success line;
  - a raising client is caught and logged, never propagated (best-effort —
    Gateway startup must never fail because a warmup ping failed).
"""

from __future__ import annotations

import app.logger as logger_module
import app.retrieval as retrieval_module


class _FakeLLM:
    def __init__(self, *, raises: Exception | None = None):
        self.calls: list[tuple[tuple, dict]] = []
        self._raises = raises

    def invoke(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._raises is not None:
            raise self._raises
        return None


def test_warm_llm_client_pings_the_singleton_with_bounded_tokens(monkeypatch):
    """A successful ping calls invoke("Hi", max_tokens=1) on get_llm()'s client."""
    fake = _FakeLLM()
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake)

    retrieval_module.warm_llm_client()

    assert len(fake.calls) == 1
    args, kwargs = fake.calls[0]
    assert args == ("Hi",)
    assert kwargs.get("max_tokens") == 1


def test_warm_llm_client_logs_success(monkeypatch, tmp_path):
    """A successful ping writes a startup_warmup status=ok line to wiki/log.md."""
    log_path = tmp_path / "wiki" / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: _FakeLLM())

    retrieval_module.warm_llm_client()

    text = log_path.read_text(encoding="utf-8")
    assert "startup_warmup" in text
    assert "client=wiki_llm status=ok" in text


def test_warm_llm_client_swallows_failure_and_logs_it(monkeypatch, tmp_path):
    """A raising client must not propagate — Gateway startup must never fail here."""
    log_path = tmp_path / "wiki" / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    fake = _FakeLLM(raises=RuntimeError("boom"))
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake)

    retrieval_module.warm_llm_client()  # must not raise

    text = log_path.read_text(encoding="utf-8")
    assert "startup_warmup" in text
    assert "client=wiki_llm status=failed exc=RuntimeError" in text
