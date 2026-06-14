"""Determinism guard (multi-turn): the query-rewriter LLM must run at
``temperature=0``.

The turn-2+ follow-up rewriter reformulates the user's question into the
retrieval query. With the langchain default temperature the same follow-up
rewrites differently across calls -> different Sections retrieved -> the same
grounded/Cannot-Confirm flip-flop, one stage earlier in the pipeline. Pin it
deterministic so multi-turn retrieval is stable.

No live OpenAI call -- get_rewrite_llm() only constructs the client; we read the
temperature field.
"""

from __future__ import annotations

import gateway.app.query_rewriting as rewrite_module


def test_rewrite_llm_pinned_to_temperature_zero(monkeypatch):
    """query_rewriting.get_rewrite_llm() must build the LLM with temperature=0."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(rewrite_module, "_rewrite_llm", None)  # bypass cached singleton

    llm = rewrite_module.get_rewrite_llm()

    assert llm.temperature == 0, "query rewriter must be deterministic (temperature=0)"
