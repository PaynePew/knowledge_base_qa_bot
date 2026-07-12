"""Determinism guard (RAG stack): the draft LLM must run at ``temperature=0``
**and** a fixed ``seed`` (ADR-0042 / issue #572, completed for Stack B by
issue #619 — the ADR's four-site enumeration had missed this call site).

Mirror of the markdown_kb determinism test. With the langchain default
temperature the RAG draft samples non-deterministically, so the same question
flips between a correct grounded answer and a false "Cannot Confirm"; the seed
narrows the residual sampling flap further (best-effort only — OpenAI's seed
is not a hard guarantee). The RAG stack shares markdown_kb's grounding
verifier (covered by the markdown_kb test), so this file only pins the RAG
draft LLM.

No live OpenAI call — get_llm() only constructs the client; we read
temperature/seed directly off the constructed instance.
"""

from __future__ import annotations

import vector_rag.app.retrieval as retrieval_module


def test_rag_draft_llm_pinned_to_temperature_zero_and_seed(monkeypatch):
    """vector_rag retrieval.get_llm() must build the draft LLM with temperature=0 and the fixed seed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(retrieval_module, "_llm", None)

    llm = retrieval_module.get_llm()

    assert llm.temperature == 0, "RAG draft LLM must be deterministic (temperature=0)"
    assert llm.seed == retrieval_module._RAG_DRAFT_LLM_SEED, (
        "RAG draft LLM must pin the fixed seed"
    )
