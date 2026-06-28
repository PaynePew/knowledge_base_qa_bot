"""Hybrid Retrieval (Stack C) query() surface — hermetic integration tests (S3, #313).

External behaviour only (CODING_STANDARD §0.2 / §6.2), all offline. The retrieval
deep module (``retrieve_and_gate``) and the dense FAISS path run FOR REAL over a
small synthetic wiki corpus — only the two genuine network leaves are faked
(implement.md trap #1: mock the LLM via its lazy-singleton getter, never a deep
module):

  * the answer-synthesis LLM   → ``query.get_llm`` monkeypatched to a ``FakeLLM``
  * the dense embedding leaf    → the ``fake_embeddings`` fixture (#311 pattern)
  * the verifier (``grounding.verify``) constructs its own ChatOpenAI inline with
    NO getter seam, so it is stubbed the same way every Wiki/RAG ``/chat`` test
    does (markdown_kb ``test_chat_grounded`` / vector_rag ``test_chat``).

Asserts the grounded-answer shape, the ``[Source:`` citation marker, the AC3
pre-LLM Cannot Confirm (no LLM call), and Cannot Confirm parity with the other
stacks (the imported sentinel, not a paraphrase).
"""

from __future__ import annotations

import pytest

import hybrid_kb.app.dense_index as dense_index
import hybrid_kb.app.query as query_module
import markdown_kb.app.indexer as bm25_indexer
import markdown_kb.app.retrieval as bm25_retrieval
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
        self.last_messages: list = []

    def invoke(self, messages: list):
        self.call_count += 1
        self.last_messages = messages
        return _FakeLLMResponse(content=self._content)


def _approved() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported")


def _rejected() -> GroundingOutcome:
    return GroundingOutcome(passed=False, reason="claim_unsupported")


# ---------------------------------------------------------------------------
# Synthetic wiki corpus + both arms built for real (no deep-module mocking)
# ---------------------------------------------------------------------------
REFUND_ID = "refund-policy#refund-policy"


def _wiki_section(section_id: str, content: str, heading_path: list[str]) -> Section:
    """A BM25-ready English wiki Section (real tokens via the production tokenizer)."""
    return Section(
        id=section_id,
        file=section_id.split("#")[0],
        heading=heading_path[-1],
        heading_path=heading_path,
        content=content,
        tokens=bm25_indexer.tokenize(content),
        metadata={"lang": "en"},
    )


_CORPUS = [
    _wiki_section(
        REFUND_ID,
        "Refund policy: refunds are processed within seven business days after "
        "approval. How long a refund takes is usually about one week.",
        ["Refund Policy"],
    ),
    _wiki_section(
        "shipping-policy#shipping-policy",
        "Shipping policy: standard shipping delivery takes three to five business "
        "days within the country.",
        ["Shipping Policy"],
    ),
    _wiki_section(
        "privacy-policy#privacy-policy",
        "Privacy policy: this policy explains how the company handles your personal "
        "account data and contact details.",
        ["Privacy Policy"],
    ),
]


@pytest.fixture()
def wired_corpus(fake_embeddings):
    """Build BOTH arms over the synthetic corpus with no deep-module mocking.

    BM25: populate ``markdown_kb`` indexer state directly (no build_index, so the
    committed ``.kb/index.json`` seed is never written). Dense: real FAISS build
    over the same Section list via the offline fake embeddings. Shared Section
    objects guarantee the 1:1 id alignment (the ADR-0018 same-corpus invariant).
    """
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
# AC1 / AC5 — grounded, cited answer reusing the shared synthesis + grounding
# ===========================================================================
def test_query_returns_grounded_cited_answer(wired_corpus, monkeypatch):
    """An in-scope query yields a grounded answer carrying a [Source: ...] citation."""
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _approved()
    )

    result = query_module.query("how long do refunds take")

    assert "[Source:" in result["answer"]
    assert result["grounding_outcome"].passed is True
    assert result["grounding_outcome"].reason == "claim_supported"
    # The synthesis LLM was actually invoked once with [SystemMessage, HumanMessage].
    assert fake_llm.call_count == 1
    assert len(fake_llm.last_messages) == 2


def test_query_sources_have_citation_shape(wired_corpus, monkeypatch):
    """Sources mirror the cross-stack citation shape: source + heading + content."""
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _approved()
    )

    result = query_module.query("how long do refunds take")

    sources = result["sources"]
    assert sources, "an in-scope query must return citation sources"
    assert all({"source", "heading", "content"} <= set(s) for s in sources)
    assert REFUND_ID in {s["source"] for s in sources}


def test_query_prompt_is_built_from_retrieved_wiki_sections(wired_corpus, monkeypatch):
    """The grounded prompt CONTEXT is filled from the fused wiki Sections (reuse)."""
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _approved()
    )

    query_module.query("how long do refunds take")

    prompt_text = fake_llm.last_messages[1].content
    assert "CONTEXT:" in prompt_text and "QUESTION:" in prompt_text
    assert f"[Source: {REFUND_ID}]" in prompt_text


# ===========================================================================
# AC3 — sub-threshold query → exact Cannot Confirm sentinel, NO LLM call
# ===========================================================================
def test_query_sub_threshold_returns_cannot_confirm_without_llm(
    wired_corpus, monkeypatch
):
    """Pre-LLM OR-gate refuses an out-of-scope query → sentinel, no synthesis call.

    Forcing the dense ceiling to 0 (dense can never clear) isolates the BM25 arm; a
    nonsense query with no keyword overlap leaves BOTH arms below threshold, so the
    gate — enforced inside the S2 deep module — returns Cannot Confirm and query()
    must short-circuit before the LLM.
    """
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    sentinel = FakeLLM("should never be called")
    _patch_llm(monkeypatch, sentinel)

    result = query_module.query("xylophone quokka zugzwang nonsense")

    assert result["answer"] == query_module.CANNOT_CONFIRM_PHRASE
    assert result["grounding_outcome"].passed is False
    assert result["grounding_outcome"].reason == "below_threshold"
    assert sentinel.call_count == 0, "no LLM call may happen on the pre-LLM gate (AC3)"


def test_query_cannot_confirm_phrase_is_the_shared_sentinel():
    """Hybrid returns the SAME Cannot Confirm literal the other stacks return (trap #2)."""
    assert query_module.CANNOT_CONFIRM_PHRASE == bm25_retrieval.CANNOT_CONFIRM_PHRASE


# ===========================================================================
# AC5 — post-LLM grounding rejection → Cannot Confirm parity with the stacks
# ===========================================================================
def test_query_grounding_rejection_replaces_with_cannot_confirm(
    wired_corpus, monkeypatch
):
    """A draft the verifier rejects is replaced by the Cannot Confirm sentinel.

    Parity with vector_rag/markdown_kb: the main LLM still runs once (the verifier
    is the gate), but an ungrounded draft never reaches the user.
    """
    fake_llm = FakeLLM("Refunds are instant and we also ship to Mars for free.")
    _patch_llm(monkeypatch, fake_llm)
    monkeypatch.setattr(
        query_module.grounding_module, "verify", lambda d, s: _rejected()
    )

    result = query_module.query("how long do refunds take")

    assert result["answer"] == query_module.CANNOT_CONFIRM_PHRASE
    assert result["grounding_outcome"].passed is False
    assert result["grounding_outcome"].reason == "claim_unsupported"
    assert fake_llm.call_count == 1, "the main LLM runs once; the verifier is the gate"


def test_query_llm_self_refusal_short_circuits_to_cannot_confirm(
    wired_corpus, monkeypatch
):
    """When the model itself emits the Cannot Confirm phrase, skip the verifier."""
    fake_llm = FakeLLM(query_module.CANNOT_CONFIRM_PHRASE)
    _patch_llm(monkeypatch, fake_llm)

    def _verify_must_not_run(draft, sections):  # pragma: no cover - asserts non-call
        raise AssertionError("verifier must be skipped on an LLM self-refusal")

    monkeypatch.setattr(query_module.grounding_module, "verify", _verify_must_not_run)

    result = query_module.query("how long do refunds take")

    assert result["answer"] == query_module.CANNOT_CONFIRM_PHRASE
    assert result["grounding_outcome"].passed is False
    assert result["grounding_outcome"].reason == "claim_unsupported"
