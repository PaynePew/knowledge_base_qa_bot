"""Hybrid Retrieval (Stack C) ``stream_query()`` surface — hermetic tests (S4, #314).

The streaming entry point the Gateway dispatches to for ``stack=hybrid``. It
composes S2's ``retrieve_and_gate`` with the SAME ``_draft_and_verify`` synthesis
``query()`` already uses (#313) and yields the two-dict generator contract the
shared SSE serializer (``markdown_kb.app.sse.events_for_result``) consumes for
every stack — sources-ready partial first, then the full result.

External behaviour only (CODING_STANDARD §0.2 / §6.2), all offline. Both arms run
FOR REAL over a small synthetic wiki corpus; only the two genuine network leaves
are faked (implement.md trap #1 — mock the LLM via its lazy getter, never a deep
module):

  * the answer-synthesis LLM → ``query.get_llm`` monkeypatched to a ``FakeLLM``
  * the dense embedding leaf  → the ``fake_embeddings`` fixture (#311 pattern)
  * the verifier (``grounding.verify``) is stubbed directly (no getter seam),
    exactly as every Wiki/RAG ``/chat`` test does.
"""

from __future__ import annotations

import pytest

import hybrid_kb.app.dense_index as dense_index
import hybrid_kb.app.query as query_module
import markdown_kb.app.indexer as bm25_indexer
from markdown_kb.app.grounding import GroundingOutcome
from markdown_kb.app.indexer import Section


# ---------------------------------------------------------------------------
# LLM stub (records calls so a "no LLM call" assertion is real)
# ---------------------------------------------------------------------------
class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Returns a canned grounded answer with a [Source: ...] token; records calls."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.call_count = 0

    def invoke(self, messages: list):
        self.call_count += 1
        return _FakeLLMResponse(content=self._content)


def _approved() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported")


# ---------------------------------------------------------------------------
# Synthetic wiki corpus + both arms built for real (no deep-module mocking)
# ---------------------------------------------------------------------------
REFUND_ID = "refund-policy#refund-policy"


def _wiki_section(
    section_id: str,
    content: str,
    heading_path: list[str],
    *,
    page_type: str | None = None,
) -> Section:
    """A BM25-ready English wiki Section (real tokens via the production tokenizer).

    ``page_type`` (entity/concept/qa) populates ``metadata["type"]`` so the wiki
    page path resolves — the signal that makes a citation clickable (#266 parity).
    """
    metadata: dict = {"lang": "en"}
    if page_type is not None:
        metadata["type"] = page_type
    return Section(
        id=section_id,
        file=section_id.split("#")[0],
        heading=heading_path[-1],
        heading_path=heading_path,
        content=content,
        tokens=bm25_indexer.tokenize(content),
        metadata=metadata,
    )


_CORPUS = [
    _wiki_section(
        REFUND_ID,
        "Refund policy: refunds are processed within seven business days after "
        "approval. How long a refund takes is usually about one week.",
        ["Refund Policy"],
        page_type="qa",
    ),
    _wiki_section(
        "shipping-policy#shipping-policy",
        "Shipping policy: standard shipping delivery takes three to five business "
        "days within the country.",
        ["Shipping Policy"],
        page_type="qa",
    ),
]


@pytest.fixture()
def wired_corpus(fake_embeddings):
    """Build BOTH arms over the synthetic corpus with no deep-module mocking."""
    bm25_indexer.sections = list(_CORPUS)
    bm25_indexer.rebuild_stats()
    dense_index.build_index(sections=list(_CORPUS))
    yield _CORPUS
    bm25_indexer.sections = []
    bm25_indexer.rebuild_stats()
    dense_index.vectorstore = None
    dense_index.sections_indexed = 0


def _patch_llm(monkeypatch, llm) -> None:
    """Swap hybrid's synthesis LLM via its lazy-singleton getter (trap #1)."""
    monkeypatch.setattr(query_module, "_llm", llm)
    monkeypatch.setattr(query_module, "get_llm", lambda: llm)


# ===========================================================================
# Generator contract — two dicts: sources-ready partial, then full result
# ===========================================================================
def test_stream_query_yields_sources_partial_then_full(wired_corpus, monkeypatch):
    """The first yield is the sources-ready partial; the second is the full result."""
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _approved()
    )

    gen = query_module.stream_query("how long do refunds take")

    partial = next(gen)
    assert partial["_phase"] == "sources_ready"
    assert partial["sources"], "the partial must carry the citation sources first"
    assert partial["early_exit"] is False
    # No LLM call may have happened yet — sources are emitted BEFORE synthesis.
    assert fake_llm.call_count == 0, (
        "sources-first: no LLM call before the partial yield"
    )

    full = next(gen)
    assert "[Source:" in full["answer"]
    assert full["grounding_outcome"].passed is True
    assert fake_llm.call_count == 1, "synthesis runs exactly once after the partial"

    with pytest.raises(StopIteration):
        next(gen)


def test_stream_query_full_result_shape_matches_query(wired_corpus, monkeypatch):
    """The full result dict mirrors query()'s shape so the shared serializer works."""
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _approved()
    )

    *_, full = list(query_module.stream_query("how long do refunds take"))

    assert {"answer", "sources", "grounding_outcome"} <= set(full)


# ===========================================================================
# Clickable citation — sources carry a resolvable wiki-page path (#266 parity)
# ===========================================================================
def test_stream_query_sources_carry_clickable_wiki_path(wired_corpus, monkeypatch):
    """An in-scope hybrid source carries a resolvable ``wiki/...`` path (clickable)."""
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _approved()
    )

    partial = next(query_module.stream_query("how long do refunds take"))

    src = next(s for s in partial["sources"] if s["source"] == REFUND_ID)
    assert src.get("path") == "wiki/qa/refund-policy.md", (
        "a wiki-typed Section must expose a resolvable wiki-page path so the "
        "reader UI renders a clickable citation (AC3 / #266 parity)"
    )
    assert "\\" not in src["path"], (
        "path must be forward-slashed (a /read/file relpath)"
    )


def test_query_sources_carry_clickable_wiki_path(wired_corpus, monkeypatch):
    """Non-streaming query() shares the same clickable-path source shape."""
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _approved()
    )

    result = query_module.query("how long do refunds take")

    src = next(s for s in result["sources"] if s["source"] == REFUND_ID)
    assert src.get("path") == "wiki/qa/refund-policy.md"


# ===========================================================================
# AC3 parity — pre-LLM OR-gate refuses → sentinel on the stream, NO LLM call
# ===========================================================================
def test_stream_query_sub_threshold_streams_cannot_confirm_without_llm(
    wired_corpus, monkeypatch
):
    """Pre-LLM gate refuses an out-of-scope query → sentinel, no synthesis call.

    The partial yield carries ``early_exit=True`` so the Gateway skips the
    verifying-status event; the second yield IS the full result and streams the
    exact Cannot Confirm sentinel (imported, not paraphrased — trap #2).
    """
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    sentinel = FakeLLM("should never be called")
    _patch_llm(monkeypatch, sentinel)

    frames = list(query_module.stream_query("xylophone quokka zugzwang nonsense"))

    partial, full = frames
    assert partial["early_exit"] is True
    assert full["answer"] == query_module.CANNOT_CONFIRM_PHRASE
    assert full["grounding_outcome"].passed is False
    assert full["grounding_outcome"].reason == "below_threshold"
    assert sentinel.call_count == 0, "no LLM call may happen on the pre-LLM gate (AC3)"


def test_stream_query_cannot_confirm_is_shared_sentinel(wired_corpus, monkeypatch):
    """The streamed Cannot Confirm text is the SAME literal the other stacks return."""
    import markdown_kb.app.retrieval as bm25_retrieval

    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    sentinel = FakeLLM("should never be called")
    _patch_llm(monkeypatch, sentinel)

    *_, full = list(query_module.stream_query("xylophone quokka zugzwang nonsense"))

    assert full["answer"] == bm25_retrieval.CANNOT_CONFIRM_PHRASE
