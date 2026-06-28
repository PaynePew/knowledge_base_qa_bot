"""MCP integration tests for kb_ask_v1 and kb_search_v1 with stack='hybrid'.

Mirrors the per-stack tests in test_kb_ask.py and test_kb_search.py.

Mocking strategy (implement.md / CODING_STANDARD §6.3 / trap #1):
  - LLM: mocked via ``hybrid_kb.app.query.get_llm`` (the lazy-singleton getter)
  - Grounding verifier: mocked at ``hybrid_kb.app.query.grounding_module.verify``
  - Dense arm: the offline ``_FakeEmbeddings`` (SHA-256-based deterministic vectors)
    so the real FAISS build / search path exercises the dense arm without network.
  - BM25 arm: populated in-place (sections.extend / rebuild_stats) to avoid
    any real load_index_json call; honours the conftest in-place restore contract.
  - NO mock of retrieve_and_gate, hybrid_query, or indexer.search — only the
    genuine network/LLM leaves are faked (trap #1).

The ``_isolate_module_state`` autouse fixture in conftest.py handles BM25 +
vector_rag cleanup; the ``_isolate_hybrid_state`` autouse fixture here extends
it to hybrid_kb state (DENSE_INDEX_DIR path, LOG_PATH, module-global teardown).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Offline embedding fake (mirrors hybrid_kb/tests/conftest.py pattern)
# ---------------------------------------------------------------------------


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings.

    SHA-256-based vectors so the real FAISS build / search path runs without
    any network call.  Stable across processes so a build-then-search roundtrip
    returns consistent neighbours.
    """

    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result."""
    from mcp.types import CallToolResult

    if isinstance(raw, CallToolResult):
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
# LLM stubs
# ---------------------------------------------------------------------------


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Fake LLM returning a canned grounded answer; records call count."""

    def __init__(
        self, content: str = "Refunds take 5-7 days. [Source: refund-policy#refund-policy]"
    ) -> None:
        self._content = content
        self.call_count = 0

    def invoke(self, messages: list) -> _FakeLLMResponse:  # noqa: ANN001
        self.call_count += 1
        return _FakeLLMResponse(content=self._content)


class _ErrorLLM:
    """Fake LLM that raises on every invoke."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def invoke(self, messages: list) -> None:  # noqa: ANN001
        raise self._exc


# ---------------------------------------------------------------------------
# Grounding stubs
# ---------------------------------------------------------------------------


def _grounding_passed():
    """GroundingOutcome(passed=True, reason='claim_supported')."""
    from markdown_kb.app.grounding import GroundingOutcome

    return GroundingOutcome(passed=True, reason="claim_supported")


def _grounding_failed(reason: str = "claim_unsupported"):
    """GroundingOutcome(passed=False)."""
    from markdown_kb.app.grounding import GroundingOutcome

    return GroundingOutcome(passed=False, reason=reason)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Synthetic wiki corpus (BM25 + dense — both arms use the same Section list)
# ---------------------------------------------------------------------------

REFUND_ID = "refund-policy#refund-policy"


def _mk_corpus():
    """Return a small synthetic wiki corpus for the wired-index fixtures."""
    import markdown_kb.app.indexer as bm25_indexer
    from markdown_kb.app.indexer import Section

    def _sec(section_id: str, content: str, heading: str) -> Section:
        return Section(
            id=section_id,
            file=section_id.split("#")[0],
            heading=heading,
            heading_path=[heading],
            content=content,
            tokens=bm25_indexer.tokenize(content),
            metadata={"lang": "en"},
        )

    return [
        _sec(
            REFUND_ID,
            "Refund policy: refunds are processed within seven business days "
            "after approval. How long a refund takes is usually about one week.",
            "Refund Policy",
        ),
        _sec(
            "shipping-policy#shipping-policy",
            "Shipping policy: standard shipping takes three to five business days.",
            "Shipping Policy",
        ),
    ]


# ---------------------------------------------------------------------------
# Autouse per-test fixture: isolate hybrid_kb state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_hybrid_state(tmp_path: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[type-arg]
    """Redirect hybrid_kb path constants to tmp and reset globals on teardown.

    Complements ``_isolate_module_state`` in conftest.py (which handles BM25 /
    vector_rag / markdown_kb state).  This fixture handles hybrid_kb-specific
    state:
      1. Redirects ``hybrid_kb.app.dense_index.DENSE_INDEX_DIR`` to tmp so
         ``build_index`` never writes to the committed ``.kb/hybrid_dense/`` seed.
      2. Redirects ``hybrid_kb.app.logger.LOG_PATH`` to tmp so ``log_event``
         calls never append to the real ``hybrid_kb/log.md``.
      3. On teardown: resets ``dense_index.vectorstore`` and ``sections_indexed``
         so a warm index from one test never leaks into another.
      4. On teardown: resets ``hybrid_kb.app.query._llm`` so a monkeypatched
         LLM singleton from one test cannot leak into another via the cache.
    """
    import hybrid_kb.app.dense_index as dense_index
    import hybrid_kb.app.logger as hk_logger
    import hybrid_kb.app.query as query_mod

    monkeypatch.setattr(dense_index, "DENSE_INDEX_DIR", tmp_path / ".kb" / "hybrid_dense")
    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")

    yield

    # Teardown: clear the in-memory dense index and the LLM singleton.
    dense_index.vectorstore = None
    dense_index.sections_indexed = 0
    query_mod._llm = None


# ---------------------------------------------------------------------------
# Wired-corpus fixture: builds BOTH arms offline (BM25 + dense with fake embs)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_embeddings_hybrid(monkeypatch: pytest.MonkeyPatch):
    """Patch hybrid_kb's embedding leaf to the deterministic offline fake."""
    import hybrid_kb.app.dense_index as dense_index

    fake = _FakeEmbeddings()
    monkeypatch.setattr(dense_index, "get_embeddings", lambda: fake)
    return fake


@pytest.fixture()
def wired_hybrid_corpus(fake_embeddings_hybrid, tmp_path):  # noqa: ANN001
    """Build BOTH arms over the synthetic corpus with no deep-module mocking.

    BM25: extend ``markdown_kb`` indexer sections IN PLACE (never rebind) and
    call rebuild_stats so the search finds the corpus.  In-place append is
    required for compatibility with conftest's teardown (clear + extend on the
    same list object).

    Dense: real FAISS build over the same Section list via the offline fake
    embeddings; index_dir points to tmp so the committed seed is never touched.
    """
    import hybrid_kb.app.dense_index as dense_index
    import markdown_kb.app.indexer as bm25_indexer

    corpus = _mk_corpus()
    # In-place append — never rebind (conftest teardown holds this list by identity).
    bm25_indexer.sections.extend(corpus)
    bm25_indexer.rebuild_stats()
    dense_index.build_index(sections=list(corpus), index_dir=tmp_path / ".kb" / "hybrid_dense")
    yield corpus
    # Teardown: remove our test sections in-place so conftest restore is a no-op.
    for sec in corpus:
        if sec in bm25_indexer.sections:
            bm25_indexer.sections.remove(sec)
    bm25_indexer.rebuild_stats()
    dense_index.vectorstore = None
    dense_index.sections_indexed = 0


# ===========================================================================
# Schema tests — stack Literal must include 'hybrid' in BOTH tools
# ===========================================================================


def test_kb_ask_v1_schema_includes_hybrid():
    """kb_ask_v1 stack Literal includes 'hybrid' in the JSON schema enum."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    ask_tool = next((t for t in tools if t.name == "kb_ask_v1"), None)
    assert ask_tool is not None, "kb_ask_v1 not registered"
    enum_vals = ask_tool.inputSchema["properties"]["stack"]["enum"]
    assert "hybrid" in enum_vals, f"'hybrid' missing from kb_ask_v1 stack enum: {enum_vals}"


def test_kb_search_v1_schema_includes_hybrid():
    """kb_search_v1 stack Literal includes 'hybrid' in the JSON schema enum."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    search_tool = next((t for t in tools if t.name == "kb_search_v1"), None)
    assert search_tool is not None, "kb_search_v1 not registered"
    enum_vals = search_tool.inputSchema["properties"]["stack"]["enum"]
    assert "hybrid" in enum_vals, f"'hybrid' missing from kb_search_v1 stack enum: {enum_vals}"


# ===========================================================================
# kb_ask_v1(stack='hybrid') — grounded answer
# ===========================================================================


def test_kb_ask_v1_hybrid_returns_stack_answer_citations_grounding(
    wired_hybrid_corpus, monkeypatch
):
    """kb_ask_v1(stack='hybrid') returns {stack, answer, citations, grounding}."""
    import asyncio

    import hybrid_kb.app.query as query_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(query_mod, "_llm", fake_llm)
    monkeypatch.setattr(query_mod, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        query_mod.grounding_module, "verify", lambda draft, sections: _grounding_passed()
    )
    raw = asyncio.run(
        mcp.call_tool("kb_ask_v1", {"query": "how long do refunds take", "stack": "hybrid"})
    )

    assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
    result = _parse_result(raw)
    assert result.get("stack") == "hybrid", f"Expected stack='hybrid': {result}"
    assert "answer" in result
    assert "citations" in result
    assert "grounding" in result


def test_kb_ask_v1_hybrid_grounding_passed_true(wired_hybrid_corpus, monkeypatch):
    """kb_ask_v1(stack='hybrid') grounded answer has grounding.passed=True."""
    import asyncio

    import hybrid_kb.app.query as query_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(query_mod, "_llm", fake_llm)
    monkeypatch.setattr(query_mod, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        query_mod.grounding_module, "verify", lambda draft, sections: _grounding_passed()
    )
    raw = asyncio.run(
        mcp.call_tool("kb_ask_v1", {"query": "how long do refunds take", "stack": "hybrid"})
    )
    result = _parse_result(raw)
    assert result["grounding"]["passed"] is True
    assert result["grounding"]["reason"] == "claim_supported"


def test_kb_ask_v1_hybrid_citations_is_list(wired_hybrid_corpus, monkeypatch):
    """kb_ask_v1(stack='hybrid') citations is a list."""
    import asyncio

    import hybrid_kb.app.query as query_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(query_mod, "_llm", fake_llm)
    monkeypatch.setattr(query_mod, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        query_mod.grounding_module, "verify", lambda draft, sections: _grounding_passed()
    )
    raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "refund policy", "stack": "hybrid"}))
    result = _parse_result(raw)
    assert isinstance(result["citations"], list)


# ===========================================================================
# kb_ask_v1(stack='hybrid') — Cannot Confirm
# ===========================================================================


def test_kb_ask_v1_hybrid_cannot_confirm_is_success(wired_hybrid_corpus, monkeypatch):
    """Sub-threshold hybrid query → Cannot Confirm is a SUCCESS (not isError).

    Forces both arms below threshold by clamping the dense ceiling to 0 and
    using a nonsense query with no keyword overlap — the same technique as
    test_query.py AC3 test.
    """
    import asyncio

    import hybrid_kb.app.query as query_mod

    from kb_mcp.server import mcp

    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    sentinel_llm = _FakeLLM("should never be called")
    monkeypatch.setattr(query_mod, "_llm", sentinel_llm)
    monkeypatch.setattr(query_mod, "get_llm", lambda: sentinel_llm)

    raw = asyncio.run(
        mcp.call_tool(
            "kb_ask_v1", {"query": "xylophone quokka zugzwang nonsense", "stack": "hybrid"}
        )
    )
    assert not _is_error_result(raw), "Cannot Confirm must be a SUCCESS result (not isError)"
    result = _parse_result(raw)
    assert result["grounding"]["passed"] is False


def test_kb_ask_v1_hybrid_cannot_confirm_exact_sentinel(wired_hybrid_corpus, monkeypatch):
    """Sub-threshold kb_ask_v1(stack='hybrid') returns the EXACT imported sentinel.

    AC: 'Cannot Confirm' is the shared sentinel imported from markdown_kb.app.retrieval,
    never paraphrased (implement.md trap #2 / CODING_STANDARD §3.3 define-once).
    """
    import asyncio

    import hybrid_kb.app.query as query_mod
    from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

    from kb_mcp.server import mcp

    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    sentinel_llm = _FakeLLM("should never be called")
    monkeypatch.setattr(query_mod, "_llm", sentinel_llm)
    monkeypatch.setattr(query_mod, "get_llm", lambda: sentinel_llm)

    raw = asyncio.run(
        mcp.call_tool(
            "kb_ask_v1", {"query": "xylophone quokka zugzwang nonsense", "stack": "hybrid"}
        )
    )
    result = _parse_result(raw)
    assert result["answer"] == CANNOT_CONFIRM_PHRASE, (
        f"Expected exact sentinel {CANNOT_CONFIRM_PHRASE!r}, got: {result['answer']!r}"
    )
    assert sentinel_llm.call_count == 0, "no LLM call may happen on the pre-LLM Cannot Confirm gate"


# ===========================================================================
# kb_ask_v1(stack='hybrid') — LLMError handling (ADR-0015)
# ===========================================================================


def test_kb_ask_v1_hybrid_llm_unavailable_is_iserror(wired_hybrid_corpus, monkeypatch):
    """LLMError(retryable=True) from hybrid path → isError=True, code='LLM_UNAVAILABLE'."""
    import asyncio

    import hybrid_kb.app.query as query_mod
    import openai

    from kb_mcp.server import mcp

    def _make_timeout():
        import httpx

        return openai.APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )

    error_llm = _ErrorLLM(_make_timeout())
    monkeypatch.setattr(query_mod, "_llm", error_llm)
    monkeypatch.setattr(query_mod, "get_llm", lambda: error_llm)

    raw = asyncio.run(
        mcp.call_tool("kb_ask_v1", {"query": "how long do refunds take", "stack": "hybrid"})
    )
    assert _is_error_result(raw), f"Expected isError=True, got: {raw}"
    payload = _parse_result(raw)
    assert payload.get("code") == "LLM_UNAVAILABLE", f"Expected LLM_UNAVAILABLE: {payload}"
    assert "message" in payload


def test_kb_ask_v1_hybrid_llm_error_non_retryable(wired_hybrid_corpus, monkeypatch):
    """LLMError(retryable=False) from hybrid path → isError=True, code='LLM_ERROR'."""
    import asyncio

    import hybrid_kb.app.query as query_mod
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
    monkeypatch.setattr(query_mod, "_llm", error_llm)
    monkeypatch.setattr(query_mod, "get_llm", lambda: error_llm)

    raw = asyncio.run(mcp.call_tool("kb_ask_v1", {"query": "refund policy", "stack": "hybrid"}))
    assert _is_error_result(raw), f"Expected isError=True, got: {raw}"
    payload = _parse_result(raw)
    assert payload.get("code") == "LLM_ERROR", f"Expected LLM_ERROR: {payload}"


# ===========================================================================
# kb_search_v1(stack='hybrid') — fused Sections, no LLM synthesis
# ===========================================================================


def test_kb_search_v1_hybrid_returns_stack_and_results(wired_hybrid_corpus):
    """kb_search_v1(stack='hybrid') returns {stack: 'hybrid', results: [...]}."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "refund policy", "stack": "hybrid"}))
    result = _parse_result(raw)
    assert result.get("stack") == "hybrid", f"Expected stack='hybrid': {result}"
    assert isinstance(result.get("results"), list), f"Expected results list: {result}"


def test_kb_search_v1_hybrid_result_items_shape(wired_hybrid_corpus):
    """Each hybrid result has {id, content, score: null} — RRF score not exposed."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "refund policy", "stack": "hybrid"}))
    result = _parse_result(raw)
    for item in result["results"]:
        assert "id" in item, f"Missing 'id': {item}"
        assert "content" in item, f"Missing 'content': {item}"
        assert "score" in item, f"Missing 'score': {item}"
        assert item["score"] is None, (
            f"Hybrid score must be null (RRF score not a calibrated magnitude): {item}"
        )


def test_kb_search_v1_hybrid_respects_k(wired_hybrid_corpus):
    """kb_search_v1(stack='hybrid') k=1 returns exactly 1 result."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(
        mcp.call_tool("kb_search_v1", {"query": "refund policy", "stack": "hybrid", "k": 1})
    )
    result = _parse_result(raw)
    assert len(result["results"]) == 1, f"Expected 1 result for k=1, got: {result['results']}"


def test_kb_search_v1_hybrid_no_llm_synthesis(wired_hybrid_corpus, monkeypatch):
    """kb_search_v1(stack='hybrid') makes NO LLM call — raw fused Sections only."""
    import asyncio

    import hybrid_kb.app.query as query_mod

    from kb_mcp.server import mcp

    tracker = _FakeLLM()
    monkeypatch.setattr(query_mod, "_llm", tracker)
    monkeypatch.setattr(query_mod, "get_llm", lambda: tracker)

    asyncio.run(mcp.call_tool("kb_search_v1", {"query": "refund policy", "stack": "hybrid"}))
    assert tracker.call_count == 0, "kb_search_v1 must not invoke the LLM for any stack"
