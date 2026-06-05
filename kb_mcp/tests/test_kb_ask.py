"""Integration tests for kb_ask_v1 using the FastMCP in-process harness.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.  This exercises FastMCP's argument
validation and the full dispatch path.

Three scenarios:
  AC-grounded: a grounded answer returns {stack, answer, citations, grounding}
               with grounding.passed=True.
  AC-cannot-confirm: a Cannot-Confirm result is a SUCCESS (not isError),
               with grounding.passed=False and grounding.reason surfaced.
  AC-llm-error: an LLMError surfaces as isError=True with {code, message};
               retryable=True → code='LLM_UNAVAILABLE'; False → 'LLM_ERROR'.

Mocking follows the project pattern (CODING_STANDARD §11 / implement.md §3.1):
  - LLM mocked at ``markdown_kb.app.retrieval.get_llm`` (not a deep entry point)
  - Grounding verifier mocked at
    ``markdown_kb.app.retrieval.grounding_module.verify``
  - Retrieval stack (indexer) mocked at ``markdown_kb.app.indexer.search``
    and ``kb_mcp.freshness.reload_if_stale`` (same pattern as test_kb_search.py)

The ``_isolate_module_state`` autouse fixture in conftest.py provides module-state
isolation; no duplication of path-redirect logic here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Stubs — retrieval / LLM / grounding
# ---------------------------------------------------------------------------


@dataclass
class _StubSection:
    """Minimal stub satisfying CitableContent Protocol."""

    id: str
    content: str
    heading_path: list[str] = field(default_factory=lambda: ["Test Heading"])
    metadata: dict = field(default_factory=dict)
    file: str = "stub.md"


class _FakeLLMResponse:
    """Minimal stub returned by a fake LLM's invoke()."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Fake LLM that returns a canned grounded answer."""

    def __init__(
        self, answer: str = "The refund takes 5-7 days. [Source: stub.md#heading]"
    ) -> None:
        self._answer = answer

    def invoke(self, messages: list) -> _FakeLLMResponse:  # noqa: ANN001
        return _FakeLLMResponse(content=self._answer)


class _ErrorLLM:
    """Fake LLM that raises a given exception on every invoke() call."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def invoke(self, messages: list) -> None:  # noqa: ANN001
        raise self._exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result.

    For success calls, FastMCP returns a list of TextContent items.
    For isError calls, FastMCP returns a CallToolResult directly.
    """
    from mcp.types import CallToolResult

    if isinstance(raw, CallToolResult):
        # isError path — extract the JSON text from the single TextContent
        return json.loads(raw.content[0].text)
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


def _is_error_result(raw: Any) -> bool:
    """Return True when the MCP call result has isError=True."""
    from mcp.types import CallToolResult

    return isinstance(raw, CallToolResult) and raw.isError is True


# ---------------------------------------------------------------------------
# Shared grounding outcome stubs
# ---------------------------------------------------------------------------


def _grounding_passed():
    """Return a GroundingOutcome(passed=True, reason='claim_supported')."""
    from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims supported.",
            claims=[
                GroundingClaim(
                    text="The refund takes 5-7 days.",
                    supported=True,
                    citing_section_ids=["stub.md#heading"],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


def _grounding_failed(reason: str = "claim_unsupported"):
    """Return a GroundingOutcome(passed=False) for a Cannot-Confirm path."""
    from markdown_kb.app.grounding import GroundingOutcome

    return GroundingOutcome(
        passed=False,
        reason=reason,  # type: ignore[arg-type]
        retries_attempted=0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_retrieval_stack(monkeypatch):
    """Patch the wiki indexer search + freshness so no real I/O occurs.

    The conftest _isolate_module_state autouse fixture redirects paths and
    resets module globals; this fixture patches the *leaf* search function
    so tests can call kb_ask_v1 without a populated index.
    """
    import markdown_kb.app.indexer as wiki_indexer

    import kb_mcp.freshness as freshness_mod

    stub_section = _StubSection(id="stub.md#heading", content="Refunds take 5-7 days.")
    monkeypatch.setattr(
        wiki_indexer,
        "search",
        lambda query, k=3: [(stub_section, 1.5)],
    )
    # expand_to_pages is called by _draft_and_verify; return the same stub
    monkeypatch.setattr(
        wiki_indexer,
        "expand_to_pages",
        lambda sections: sections,
    )
    monkeypatch.setattr(freshness_mod, "reload_if_stale", lambda *_a, **_kw: False)

    # Populate the in-process sections list so the pre-LLM gate (not indexed yet)
    # does not fire — the indexer checks `if not indexer.sections` before searching.
    monkeypatch.setattr(wiki_indexer, "sections", [stub_section])


# ---------------------------------------------------------------------------
# AC-grounded: grounded answer returns {stack, answer, citations, grounding}
# ---------------------------------------------------------------------------


def test_kb_ask_v1_grounded_answer_shape():
    """kb_ask_v1 grounded answer returns {stack, answer, citations, grounding}."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", fake_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
        mp.setattr(
            retrieval_mod.grounding_module,
            "verify",
            lambda draft, sections: _grounding_passed(),
        )
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "How long do refunds take?"}))

    assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
    result = _parse_result(raw)

    assert "stack" in result, f"Missing 'stack' key: {result}"
    assert "answer" in result, f"Missing 'answer' key: {result}"
    assert "citations" in result, f"Missing 'citations' key: {result}"
    assert "grounding" in result, f"Missing 'grounding' key: {result}"


def test_kb_ask_v1_grounded_answer_stack_default():
    """Default stack is 'wiki'."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", fake_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
        mp.setattr(
            retrieval_mod.grounding_module,
            "verify",
            lambda draft, sections: _grounding_passed(),
        )
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "refund question"}))

    result = _parse_result(raw)
    assert result["stack"] == "wiki"


def test_kb_ask_v1_grounded_grounding_passed_true():
    """Grounded answer has grounding.passed=True and grounding.reason='claim_supported'."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", fake_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
        mp.setattr(
            retrieval_mod.grounding_module,
            "verify",
            lambda draft, sections: _grounding_passed(),
        )
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "refund policy"}))

    result = _parse_result(raw)
    grounding = result["grounding"]
    assert grounding["passed"] is True
    assert grounding["reason"] == "claim_supported"


def test_kb_ask_v1_citations_is_list():
    """citations is a list (may be empty or populated)."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", fake_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
        mp.setattr(
            retrieval_mod.grounding_module,
            "verify",
            lambda draft, sections: _grounding_passed(),
        )
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "refund policy"}))

    result = _parse_result(raw)
    assert isinstance(result["citations"], list)


# ---------------------------------------------------------------------------
# AC-cannot-confirm: Cannot Confirm is a SUCCESS result, never isError
# ---------------------------------------------------------------------------


def test_kb_ask_v1_cannot_confirm_is_success_not_iserror():
    """Cannot Confirm is a SUCCESS result (not isError) — ADR-0015 / ADR-0016 invariant."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", fake_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
        # Force grounding to fail → Cannot Confirm path
        mp.setattr(
            retrieval_mod.grounding_module,
            "verify",
            lambda draft, sections: _grounding_failed("claim_unsupported"),
        )
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "claim not supported"}))

    assert not _is_error_result(raw), (
        "Cannot Confirm must be a SUCCESS result (isError=False), got isError result"
    )


def test_kb_ask_v1_cannot_confirm_grounding_passed_false():
    """Cannot Confirm has grounding.passed=False."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", fake_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
        mp.setattr(
            retrieval_mod.grounding_module,
            "verify",
            lambda draft, sections: _grounding_failed("claim_unsupported"),
        )
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "claim not supported"}))

    result = _parse_result(raw)
    assert result["grounding"]["passed"] is False


def test_kb_ask_v1_cannot_confirm_reason_surfaced():
    """Cannot Confirm surfaces grounding.reason (not just passed=False)."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", fake_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
        mp.setattr(
            retrieval_mod.grounding_module,
            "verify",
            lambda draft, sections: _grounding_failed("verifier_unavailable"),
        )
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "claim not supported"}))

    result = _parse_result(raw)
    grounding = result["grounding"]
    assert "reason" in grounding, f"Missing 'reason' in grounding: {grounding}"
    assert grounding["reason"] == "verifier_unavailable"


def test_kb_ask_v1_pre_llm_cannot_confirm_is_success():
    """Pre-LLM Cannot Confirm (below_threshold gate) is also a SUCCESS, not isError."""
    import asyncio

    import markdown_kb.app.indexer as wiki_indexer

    from kb_mcp.server import mcp

    # Patch search to return low-score results so the below_threshold gate fires
    stub_section = _StubSection(id="stub.md#heading", content="low score content")
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            wiki_indexer,
            "search",
            lambda query, k=3: [(stub_section, 0.0)],  # score=0 < threshold
        )
        mp.setattr(wiki_indexer, "sections", [stub_section])
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "very obscure query"}))

    assert not _is_error_result(raw), "Pre-LLM Cannot Confirm must be SUCCESS, not isError"
    result = _parse_result(raw)
    assert result["grounding"]["passed"] is False


# ---------------------------------------------------------------------------
# AC-llm-error: LLMError surfaces as isError with {code, message}
# ---------------------------------------------------------------------------


def test_kb_ask_v1_retryable_llm_error_iserror():
    """LLMError(retryable=True) → isError=True with code='LLM_UNAVAILABLE'."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod
    import openai

    from kb_mcp.server import mcp

    def _make_timeout():
        import httpx

        return openai.APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )

    error_llm = _ErrorLLM(_make_timeout())
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", error_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: error_llm)
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "will fail"}))

    assert _is_error_result(raw), f"Expected isError=True for retryable LLMError, got: {raw}"
    payload = _parse_result(raw)
    assert payload.get("code") == "LLM_UNAVAILABLE", (
        f"Expected code='LLM_UNAVAILABLE', got: {payload}"
    )
    assert "message" in payload, f"Missing 'message' key: {payload}"


def test_kb_ask_v1_non_retryable_llm_error_iserror():
    """LLMError(retryable=False) → isError=True with code='LLM_ERROR'."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod
    import openai

    from kb_mcp.server import mcp

    def _make_auth_error():
        import httpx

        return openai.AuthenticationError(
            "Incorrect API key",
            response=httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
            body={},
        )

    error_llm = _ErrorLLM(_make_auth_error())
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_llm", error_llm)
        mp.setattr(retrieval_mod, "get_llm", lambda: error_llm)
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "auth fail"}))

    assert _is_error_result(raw), f"Expected isError=True for non-retryable LLMError, got: {raw}"
    payload = _parse_result(raw)
    assert payload.get("code") == "LLM_ERROR", f"Expected code='LLM_ERROR', got: {payload}"
    assert "message" in payload, f"Missing 'message' key: {payload}"


def test_kb_ask_v1_llm_error_message_carried():
    """LLMError.message is carried through to the isError payload."""
    import asyncio

    import markdown_kb.app.retrieval as retrieval_mod
    from markdown_kb.app.errors import LLMError

    from kb_mcp.server import mcp

    expected_message = "LLM service temporarily unavailable, please retry."

    def _raise_with_message(question, prompt_text):  # noqa: ANN001
        raise LLMError(retryable=True, message=expected_message)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(retrieval_mod, "_call_llm_with_error_handling", _raise_with_message)
        raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "will fail"}))

    assert _is_error_result(raw)
    payload = _parse_result(raw)
    assert payload.get("message") == expected_message, (
        f"Expected message={expected_message!r}, got: {payload.get('message')!r}"
    )
