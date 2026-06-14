"""Determinism guard (RAG stack): the draft LLM must run at ``temperature=0``.

Mirror of the markdown_kb determinism test. With the langchain default
temperature the RAG draft samples non-deterministically, so the same question
flips between a correct grounded answer and a false "Cannot Confirm". The RAG
stack shares markdown_kb's grounding verifier (covered by the markdown_kb test),
so this file only pins the RAG draft LLM.

No live OpenAI call — get_llm() only constructs the client; we read temperature.
"""

from __future__ import annotations

import vector_rag.app.retrieval as retrieval_module


def test_rag_draft_llm_pinned_to_temperature_zero(monkeypatch):
    """vector_rag retrieval.get_llm() must build the draft LLM with temperature=0."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(retrieval_module, "_llm", None)

    llm = retrieval_module.get_llm()

    assert llm.temperature == 0, "RAG draft LLM must be deterministic (temperature=0)"
