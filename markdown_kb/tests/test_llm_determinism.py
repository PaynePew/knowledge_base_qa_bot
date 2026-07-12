"""Determinism guard (Wiki stack): every answer-chain LLM call runs at
``temperature=0`` **and** a fixed ``seed`` (ADR-0038's C5-judge pattern,
extended to the serving chain by ADR-0042 / issue #572).

Why: with the langchain default temperature the draft and the verifier sample
non-deterministically, so the *same* question intermittently flip-flops between
a correct grounded answer and a false "Cannot Confirm" (the draft sometimes
self-refuses; the strict fail-closed verifier sometimes marks a supported claim
unsupported). ``temperature=0`` makes the answer/grounding stage close to
stable; the seed narrows the residual sampling flap further (best-effort only
— OpenAI's seed is not a hard guarantee).

No live OpenAI call: every getter here only constructs the ``ChatOpenAI``
client — we read the ``temperature`` / ``seed`` fields directly off the
constructed instance. The verifier getter (``get_verifier_llm``) is asserted
the same way as ``get_llm`` / ``get_retry_llm`` — through the getter's own
monkeypatch seam (§6.3) — never by patching ``ChatOpenAI`` internals.
"""

from __future__ import annotations

import app.grounding as grounding_module
import app.retrieval as retrieval_module


def test_draft_llm_pinned_to_temperature_zero_and_seed(monkeypatch):
    """retrieval.get_llm() must build the draft LLM with temperature=0 and the fixed seed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(retrieval_module, "_llm", None)  # bypass any cached singleton

    llm = retrieval_module.get_llm()

    assert llm.temperature == 0, "draft LLM must be deterministic (temperature=0)"
    assert llm.seed == retrieval_module._ANSWER_CHAIN_LLM_SEED, "draft LLM must pin the fixed seed"


def test_retry_llm_stays_temperature_zero_and_seed(monkeypatch):
    """The grounding-retry LLM was already temperature=0; guard against drift, plus the seed pin."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(retrieval_module, "_retry_llm", None)

    llm = retrieval_module.get_retry_llm()

    assert llm.temperature == 0
    assert llm.seed == retrieval_module._ANSWER_CHAIN_LLM_SEED


def test_verifier_llm_pinned_to_temperature_zero_and_seed(monkeypatch):
    """grounding.get_verifier_llm() must build the verifier LLM with temperature=0 and the fixed seed.

    Asserts through the getter's own monkeypatch seam (§6.3) — reset the cached
    singleton, then read the constructed instance's fields directly — rather
    than patching ``ChatOpenAI`` internals (ADR-0042 / issue #572, the #483
    singleton discipline: the verifier construction site used to live inline
    in ``verify()`` and rebuild on every call; it is now a lazy singleton
    getter like every other LLM call site).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(grounding_module, "_verifier_llm", None)  # bypass any cached singleton

    llm = grounding_module.get_verifier_llm()

    assert llm.temperature == 0, "verifier LLM must be deterministic (temperature=0)"
    assert llm.seed == grounding_module._VERIFIER_LLM_SEED, "verifier LLM must pin the fixed seed"
