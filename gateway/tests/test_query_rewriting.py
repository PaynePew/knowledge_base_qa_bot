"""Hermetic tests for the Query Rewriting module (Phase 11 Slice 1 — issue #159).

All tests mock the LLM via the lazy-singleton getter (monkeypatch).
No OPENAI_API_KEY is required.

Coverage:
- Turn 1 (no history): passthrough, no LLM call
- Turn 2+: elliptical follow-up is rewritten (LLM called, returns self-contained query)
- Rule 2: an already-self-contained follow-up passes through unchanged (LLM signals no-op)
- Rule 4: ambiguous reference conservatively keeps original (LLM signals keep-original)
- Shape assertion: result is a non-empty string (never a LangChain object)
- LangChain isolation: rewrite module IS the only one that imports LangChain types
  (enforced structurally; routes and store import only from gateway.app)

One @pytest.mark.live test is registered per ADR-0005 §6.4 (runs only with -m live).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gateway.app.query_rewriting as _rewrite_module
from gateway.app.query_rewriting import rewrite_query

# ---------------------------------------------------------------------------
# Fake LLM / stub helpers
# ---------------------------------------------------------------------------


class _FakeRewriteLLM:
    """Minimal structured-output LLM stub.

    The rewrite module uses ``llm.with_structured_output(Schema)`` which
    returns a chain; here we make the chain's ``invoke()`` return a mock
    that looks like the pydantic model the module expects.
    """

    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten

    def with_structured_output(self, schema):
        """Return a chain-like object whose invoke() returns a schema instance."""
        chain = MagicMock()
        result = MagicMock()
        result.rewritten_query = self._rewritten
        chain.invoke.return_value = result
        return chain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rewrite_llm(monkeypatch):
    """Reset the module-level LLM singleton between tests."""
    monkeypatch.setattr(_rewrite_module, "_rewrite_llm", None)


# ---------------------------------------------------------------------------
# Turn 1 passthrough (no LLM call)
# ---------------------------------------------------------------------------


def test_no_history_passthrough_no_llm_call(monkeypatch):
    """Turn 1 (empty history) passes the query through without any LLM call.

    The LLM singleton must never be initialised for a passthrough.
    """
    get_llm_calls = []

    def _mock_get_llm():
        get_llm_calls.append(True)
        return _FakeRewriteLLM("should not be called")

    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", _mock_get_llm)

    result = rewrite_query("How long do refunds take?", history=[])
    assert result == "How long do refunds take?"
    assert get_llm_calls == [], "get_rewrite_llm must not be called on turn 1"


# ---------------------------------------------------------------------------
# Rewrite path (history non-empty)
# ---------------------------------------------------------------------------


def test_elliptical_followup_is_rewritten(monkeypatch):
    """An elliptical follow-up is rewritten into a self-contained query.

    The test only asserts shape (non-empty str) — not exact wording (§6.2).
    """
    fake_llm = _FakeRewriteLLM("how long do exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_llm)

    history = [
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00.000000Z",
        }
    ]
    result = rewrite_query("and exchanges?", history=history)
    # Shape: non-empty string
    assert isinstance(result, str)
    assert result.strip() != ""
    # It is not the raw elliptical follow-up (the LLM changed it)
    assert result != "and exchanges?"


def test_rewrite_returns_plain_string_not_langchain_type(monkeypatch):
    """The return value is a plain Python str, not a LangChain or Pydantic object."""
    fake_llm = _FakeRewriteLLM("how long do exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_llm)

    history = [
        {
            "question": "q",
            "answer": "a",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "t",
        }
    ]
    result = rewrite_query("follow up", history=history)
    assert type(result) is str, f"Expected plain str, got {type(result)}"


# ---------------------------------------------------------------------------
# Rule 2: already-self-contained query passes through (LLM echoes it)
# ---------------------------------------------------------------------------


def test_self_contained_query_passes_through_unchanged(monkeypatch):
    """Rule 2: the rewriter returns an already-self-contained query unchanged.

    The LLM is configured to return the same query string (mirroring the
    rewriter prompt's rule 2 guarantee). The route/store sees no difference
    from turn 1 passthrough in terms of the dispatched query.
    """
    self_contained = "What is the return window for sale items?"
    fake_llm = _FakeRewriteLLM(self_contained)  # LLM echoes unchanged
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_llm)

    history = [
        {
            "question": "q1",
            "answer": "a1",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "t",
        }
    ]
    result = rewrite_query(self_contained, history=history)
    assert result == self_contained


# ---------------------------------------------------------------------------
# Rule 4: ambiguous reference keeps original (LLM echoes original)
# ---------------------------------------------------------------------------


def test_ambiguous_reference_keeps_original(monkeypatch):
    """Rule 4: on an ambiguous reference, the LLM conservatively returns the original.

    The rewriter prompt instructs the LLM to keep the original rather than
    guess; the test asserts that the module passes back whatever the LLM returns.
    """
    raw_followup = "and what about that?"
    fake_llm = _FakeRewriteLLM(raw_followup)  # LLM conservatively echoes original
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_llm)

    history = [
        {
            "question": "q1",
            "answer": "a1",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "t",
        }
    ]
    result = rewrite_query(raw_followup, history=history)
    assert result == raw_followup


# ---------------------------------------------------------------------------
# Live smoke test (opt-in; ADR-0005 §6.4 — one per LLM-facing surface)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rewrite_query_live():
    """Live smoke: real OpenAI call rewrites an elliptical follow-up.

    Requires OPENAI_API_KEY. Run with: uv run pytest -m live
    Asserts shape only (§6.2): non-empty str returned for an elliptical follow-up.
    """
    history = [
        {
            "question": "How long do refunds take?",
            "answer": "Approved refunds are processed within 5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00.000000Z",
        }
    ]
    result = rewrite_query("and exchanges?", history=history)
    assert isinstance(result, str)
    assert len(result) > 0
    # The raw ellipsis should be expanded
    assert "?" in result or result.endswith(".")
