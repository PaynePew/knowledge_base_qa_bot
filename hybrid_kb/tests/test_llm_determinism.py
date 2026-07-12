"""Determinism guard (Hybrid stack): the answer-synthesis LLM must run at
``temperature=0`` **and** a fixed ``seed`` (ADR-0042 / issue #572, completed
for Stack C by issue #619 — the ADR's four-site enumeration had missed this
call site).

Mirror of the vector_rag / markdown_kb determinism tests. With the langchain
default temperature the draft samples non-deterministically, so the same
question flips between a correct grounded answer and a false "Cannot
Confirm"; the seed narrows the residual sampling flap further (best-effort
only — OpenAI's seed is not a hard guarantee). Hybrid shares markdown_kb's
grounding verifier (covered by the markdown_kb test), so this file only pins
Hybrid's own answer-synthesis LLM.

No live OpenAI call — get_llm() only constructs the client; we read
temperature/seed directly off the constructed instance.
"""

from __future__ import annotations

import hybrid_kb.app.query as query_module


def test_hybrid_draft_llm_pinned_to_temperature_zero_and_seed(monkeypatch):
    """hybrid_kb query.get_llm() must build the answer LLM with temperature=0 and the fixed seed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(query_module, "_llm", None)

    llm = query_module.get_llm()

    assert llm.temperature == 0, (
        "Hybrid answer LLM must be deterministic (temperature=0)"
    )
    assert llm.seed == query_module._HYBRID_DRAFT_LLM_SEED, (
        "Hybrid answer LLM must pin the fixed seed"
    )
