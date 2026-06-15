"""Determinism guard (Wiki stack): the draft LLM and the grounding verifier
must run at ``temperature=0``.

Why: with the langchain default temperature the draft and the verifier sample
non-deterministically, so the *same* question intermittently flip-flops between
a correct grounded answer and a false "Cannot Confirm" (the draft sometimes
self-refuses; the strict fail-closed verifier sometimes marks a supported claim
unsupported). Pinning temperature=0 makes the answer/grounding stage stable.

No live OpenAI call: ``get_llm`` / ``get_retry_llm`` only construct the client
(we read the ``temperature`` field); the verifier construction site is exercised
via a spy that records the ``ChatOpenAI`` kwargs and short-circuits the
structured-output call.
"""

from __future__ import annotations

import app.grounding as grounding_module
import app.retrieval as retrieval_module
from app.grounding import GroundingClaim, GroundingResult


def test_draft_llm_pinned_to_temperature_zero(monkeypatch):
    """retrieval.get_llm() must build the draft LLM with temperature=0."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(retrieval_module, "_llm", None)  # bypass any cached singleton

    llm = retrieval_module.get_llm()

    assert llm.temperature == 0, "draft LLM must be deterministic (temperature=0)"


def test_retry_llm_stays_temperature_zero(monkeypatch):
    """The grounding-retry LLM was already temperature=0; guard against drift."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-determinism")
    monkeypatch.setattr(retrieval_module, "_retry_llm", None)

    llm = retrieval_module.get_retry_llm()

    assert llm.temperature == 0


class _FakeSection:
    """Minimal CitableContent (id / heading_path / content) for verify()."""

    id = "s1"
    heading_path = ["Heading"]
    content = "Standard shipping usually takes 3-5 business days."


def test_verifier_llm_pinned_to_temperature_zero(monkeypatch):
    """grounding.verify() must construct the verifier LLM with temperature=0."""
    captured: dict = {}

    class _FakeChain:
        def invoke(self, _user_message):
            return GroundingResult(
                reasoning="claim supported by the cited section",
                claims=[GroundingClaim(text="c", supported=True, citing_section_ids=["s1"])],
                unsupported_claims=[],
                passed=True,
            )

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, _schema):
            return _FakeChain()

    monkeypatch.setattr(grounding_module, "ChatOpenAI", _FakeChatOpenAI)

    outcome = grounding_module.verify("Shipping takes 3-5 business days.", [_FakeSection()])

    assert outcome.passed is True  # the spy returns a passing verification
    assert captured.get("temperature") == 0, "verifier LLM must be temperature=0"
