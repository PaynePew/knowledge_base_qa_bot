"""Tests for KB_INGEST_MAX_RETRIES env-var controlling LLM retry budget.

Hermetic: no OPENAI_API_KEY required.  get_ingest_llm() is a lazy singleton;
tests must reset ``templates._ingest_llm = None`` before calling it so the
singleton is rebuilt with the current env.
"""

from __future__ import annotations

import app.templates as templates_module


def test_get_ingest_llm_default_max_retries(monkeypatch):
    """Unset KB_INGEST_MAX_RETRIES → default max_retries is 5."""
    monkeypatch.delenv("KB_INGEST_MAX_RETRIES", raising=False)
    # Force singleton rebuild
    monkeypatch.setattr(templates_module, "_ingest_llm", None)

    llm = templates_module.get_ingest_llm()

    assert llm.max_retries == 5, f"Expected max_retries=5 by default, got {llm.max_retries}"


def test_get_ingest_llm_max_retries_env(monkeypatch):
    """KB_INGEST_MAX_RETRIES override is respected at construction time."""
    monkeypatch.setenv("KB_INGEST_MAX_RETRIES", "3")
    # Force singleton rebuild so the new env takes effect
    monkeypatch.setattr(templates_module, "_ingest_llm", None)

    llm = templates_module.get_ingest_llm()

    assert llm.max_retries == 3, (
        f"Expected max_retries=3 from KB_INGEST_MAX_RETRIES=3, got {llm.max_retries}"
    )
