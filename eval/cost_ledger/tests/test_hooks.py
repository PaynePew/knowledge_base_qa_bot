"""Tests for hooks.py — instrumenting a lazy-singleton LLM getter with a
CostLedger. Every LLM stand-in here is a hand-rolled fake carrying scripted
``usage_metadata`` (PRD #654 Testing Decisions); no LangChain type, no real
LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eval.cost_ledger.hooks import (
    instrument_invoke,
    record_usage_from_response,
)
from eval.cost_ledger.ledger import CostLedger


@dataclass
class _FakeResponse:
    usage_metadata: dict | None = None


@dataclass
class _FakeLLM:
    """Stand-in for a ``ChatOpenAI`` singleton: ``.invoke()`` returns a
    scripted response and every call is recorded so tests can assert the
    getter's client was actually used (not bypassed)."""

    model_name: str = "gpt-4o-mini"
    scripted_response: _FakeResponse = field(default_factory=_FakeResponse)
    invoke_calls: list = field(default_factory=list)

    def invoke(self, prompt):
        self.invoke_calls.append(prompt)
        return self.scripted_response


# ---------------------------------------------------------------------------
# record_usage_from_response
# ---------------------------------------------------------------------------


def test_record_usage_from_response_reads_usage_metadata_attribute():
    ledger = CostLedger()
    response = _FakeResponse(
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    )

    record_usage_from_response(
        ledger, stack="A", phase="query", model="gpt-4o-mini", response=response
    )

    totals = ledger.totals(stack="A", phase="query")
    assert totals.calls == 1
    assert totals.input_tokens == 7
    assert totals.output_tokens == 3
    assert totals.total_tokens == 10


def test_record_usage_from_response_reads_raw_dict_shape():
    ledger = CostLedger()
    raw = {"usage_metadata": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}}

    record_usage_from_response(
        ledger, stack="B", phase="build", model="gpt-4o-mini", response=raw
    )

    assert ledger.totals(stack="B", phase="build").total_tokens == 6


def test_record_usage_from_response_none_records_zero_token_call():
    """A call still happened even when no usage is observable — the calls
    axis must not silently drop it."""
    ledger = CostLedger()

    record_usage_from_response(
        ledger, stack="A", phase="build", model="gpt-4o-mini", response=None
    )

    totals = ledger.totals(stack="A", phase="build")
    assert totals.calls == 1
    assert totals.total_tokens == 0


# ---------------------------------------------------------------------------
# instrument_invoke
# ---------------------------------------------------------------------------


def test_instrumented_getter_records_each_invoke_call():
    ledger = CostLedger()
    fake_llm = _FakeLLM(
        scripted_response=_FakeResponse(
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
        )
    )
    get_llm = lambda: fake_llm  # noqa: E731 — mirrors a production lazy-singleton getter

    instrumented_get_llm = instrument_invoke(get_llm, ledger, stack="A", phase="query")
    llm = instrumented_get_llm()
    llm.invoke("what is the refund window?")
    llm.invoke("and for gift cards?")

    assert fake_llm.invoke_calls == [
        "what is the refund window?",
        "and for gift cards?",
    ]
    totals = ledger.totals(stack="A", phase="query")
    assert totals.calls == 2
    assert totals.total_tokens == 140


def test_instrumented_getter_resolves_model_from_client_when_not_given():
    ledger = CostLedger()
    fake_llm = _FakeLLM(model_name="gpt-4o")
    instrumented_get_llm = instrument_invoke(
        lambda: fake_llm, ledger, stack="A", phase="build"
    )

    instrumented_get_llm().invoke("x")

    calls = ledger.calls
    assert calls[0].model == "gpt-4o"


def test_instrumented_getter_model_override_wins_over_client_attribute():
    ledger = CostLedger()
    fake_llm = _FakeLLM(model_name="gpt-4o")
    instrumented_get_llm = instrument_invoke(
        lambda: fake_llm, ledger, stack="A", phase="build", model="gpt-4o-mini"
    )

    instrumented_get_llm().invoke("x")

    assert ledger.calls[0].model == "gpt-4o-mini"


def test_instrumented_client_delegates_other_attributes_unchanged():
    """Only .invoke() is observed — any other attribute/method reaches the
    real underlying client untouched (e.g. .with_structured_output chains)."""
    ledger = CostLedger()

    class _FakeLLMWithExtra(_FakeLLM):
        def with_structured_output(self, schema):
            return f"chain-for-{schema}"

    fake_llm = _FakeLLMWithExtra()
    instrumented_get_llm = instrument_invoke(
        lambda: fake_llm, ledger, stack="A", phase="build"
    )

    proxy = instrumented_get_llm()
    assert (
        proxy.with_structured_output("GroundingResult") == "chain-for-GroundingResult"
    )
    assert ledger.calls == [], "non-.invoke() attribute access must not record a call"
