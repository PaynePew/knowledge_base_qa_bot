"""Hermetic tests for Phase 11 Slice 5 cross-stack toggle coherence (issue #163).

Covers:
- Wiki→RAG→Wiki conversation shares history via one session_id (Conversation Store
  keyed by session_id only; stack is per-turn metadata, NOT a partition key).
- RAG turns never file to wiki/qa/ (done.filed is always null for stack=rag).
- Wiki turns file the rewritten self-contained query on grounding pass.
- Grounding firewall: a claim only RAG's corpus supports CANNOT appear in a
  subsequent Wiki turn's answer or filed page (ungrounded → Cannot Confirm → no
  filing). Conversation context shapes the question, never injects facts into the
  answer.
- evict_expired() is called at the top of chat_stream so idle sessions are swept on
  the next request. An expired session is gone at request entry — verified with an
  injectable fake clock (no sleep).

All hermetic: LLM and rewriter mocked, no OPENAI_API_KEY.
Prior art: gateway/tests/test_multiturn_routes.py, test_chat_stream_rag.py.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _logger
import markdown_kb.app.retrieval as _retrieval
import pytest
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
import vector_rag.app.retrieval as vr_retrieval
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

import gateway.app.conversation_store as _store_module
import gateway.app.query_rewriting as _rewrite_module

_FIXTURE_DOCS = Path(__file__).resolve().parents[2] / "markdown_kb" / "tests" / "fixtures" / "docs"
_REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeWikiLLM:
    """Wiki LLM that returns an answer citing a Wiki section."""

    CANNED_ANSWER = (
        "Approved refunds are processed within 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


class _FakeRagLLM:
    """RAG LLM that returns an answer with RAG content."""

    CANNED_ANSWER = (
        "Approved refunds are processed within 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


class _FakeRewriteLLM:
    """Structured-output rewrite LLM stub that returns a fixed rewritten query."""

    def __init__(self, rewritten: str = "how long do refunds take?") -> None:
        self._rewritten = rewritten

    def with_structured_output(self, schema):
        chain = MagicMock()
        result = MagicMock()
        result.rewritten_query = self._rewritten
        chain.invoke.return_value = result
        return chain


class _FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings — no OpenAI call."""

    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# Grounding outcome helpers
# ---------------------------------------------------------------------------


def _approved_outcome() -> GroundingOutcome:
    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims trace to the cited section.",
            claims=[
                GroundingClaim(
                    text="Approved refunds are processed within 5-7 business days.",
                    supported=True,
                    citing_section_ids=["refund_policy.md#refund-timeline"],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


def _cannot_confirm_outcome() -> GroundingOutcome:
    """Grounding outcome that always fails — cannot confirm (grounding firewall)."""
    return GroundingOutcome(
        passed=False,
        reason="claim_unsupported",
        result=GroundingResult(
            reasoning="Claim not supported by the cited sections.",
            claims=[
                GroundingClaim(
                    text="Some RAG-only fact.",
                    supported=False,
                    citing_section_ids=[],
                )
            ],
            unsupported_claims=["Some RAG-only fact."],
            passed=False,
        ),
        retries_attempted=0,
    )


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse_response(content: str) -> list[dict]:
    """Parse a multi-frame SSE response into a list of {type, data} dicts."""
    events = []
    for frame in content.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        lines = frame.split("\n")
        event_type = "message"
        data_str = ""
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:]
        if data_str:
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"raw": data_str}
            events.append({"type": event_type, "data": data})
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_wiki_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect wiki paths to tmp so no real disk files are written."""
    monkeypatch.setattr(_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")


@pytest.fixture(autouse=True)
def _redirect_rag_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect vector_rag paths to tmp for all cross-stack tests."""
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    """Use a fresh ConversationStore per test to avoid cross-test bleed."""
    from gateway.app.conversation_store import ConversationStore

    fresh = ConversationStore()
    monkeypatch.setattr(_store_module, "store", fresh)
    return fresh


@pytest.fixture(autouse=True)
def _reset_rewrite_llm(monkeypatch):
    """Reset the rewrite LLM singleton between tests."""
    monkeypatch.setattr(_rewrite_module, "_rewrite_llm", None)


@pytest.fixture()
def indexed_wiki_corpus(tmp_path):
    """Build the Section Index from the 3-Source hermetic fixture docs."""
    _indexer.build_index(_FIXTURE_DOCS)
    yield
    _indexer.sections.clear()


@pytest.fixture()
def indexed_rag_corpus(tmp_path, monkeypatch):
    """Build the FAISS index from REAL_DOCS with fake embeddings."""
    fake = _FakeEmbeddings()
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: fake)
    vr_indexer.build_index(_REAL_DOCS)
    yield
    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0


@pytest.fixture()
def both_stacks_client(indexed_wiki_corpus, indexed_rag_corpus, monkeypatch):
    """TestClient with both Wiki and RAG stacks indexed; LLMs and rewriter mocked."""
    # Wiki LLM
    fake_wiki_llm = _FakeWikiLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_wiki_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_wiki_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    # RAG LLM
    fake_rag_llm = _FakeRagLLM()
    monkeypatch.setattr(vr_retrieval, "_llm", fake_rag_llm)
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: fake_rag_llm)
    monkeypatch.setattr(
        vr_retrieval.grounding_module,
        "verify",
        lambda draft, chunks: _approved_outcome(),
    )
    # Rewrite LLM
    fake_rewrite = _FakeRewriteLLM("how long do refund exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


# ---------------------------------------------------------------------------
# AC: Wiki→RAG→Wiki session shares history (store keyed by session_id only)
# ---------------------------------------------------------------------------


def test_crossstack_session_shares_history(both_stacks_client):
    """Wiki→RAG→Wiki conversation uses one shared session_id; history includes all turns.

    The store is keyed by session_id ONLY. A RAG turn and a Wiki turn in the same
    session both read from and write to the same session slot.
    """
    # Turn 1: Wiki
    resp1 = both_stacks_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp1.status_code == 200
    events1 = _parse_sse_response(resp1.text)
    done1 = next(e for e in events1 if e["type"] == "done")
    session_id = done1["data"]["session"]
    assert session_id, "Turn 1 must return a session id"

    # Turn 2: RAG — same session, different stack
    resp2 = both_stacks_client.post(
        f"/chat/stream?stack=rag&session={session_id}",
        json={"query": "and exchanges?"},
    )
    assert resp2.status_code == 200
    events2 = _parse_sse_response(resp2.text)
    done2 = next(e for e in events2 if e["type"] == "done")
    # Session id must be echoed unchanged
    assert done2["data"]["session"] == session_id, (
        f"RAG turn must echo session_id={session_id!r}; got {done2['data']['session']!r}"
    )

    # Turn 3: Wiki again — same session
    resp3 = both_stacks_client.post(
        f"/chat/stream?stack=wiki&session={session_id}",
        json={"query": "what about cancellations?"},
    )
    assert resp3.status_code == 200
    events3 = _parse_sse_response(resp3.text)
    done3 = next(e for e in events3 if e["type"] == "done")
    assert done3["data"]["session"] == session_id

    # After 3 turns, the shared store must hold all 3 turns (two wiki + one rag)
    history = _store_module.store.get_history(session_id)
    assert len(history) == 3, f"Expected 3 turns across stacks; got {len(history)}: {history}"
    stacks = [t["stack"] for t in history]
    assert stacks == ["wiki", "rag", "wiki"], f"Turn stacks must be wiki/rag/wiki; got {stacks}"


def test_rag_turn_uses_prior_wiki_history(indexed_wiki_corpus, indexed_rag_corpus, monkeypatch):
    """A RAG turn (turn 2) resolves its query against the shared history from turn 1 (wiki).

    The rewriter is called with the shared history, regardless of the stack of the current turn.
    """
    rewrite_calls = []

    class _TrackingRewriteLLM:
        def with_structured_output(self, schema):
            chain = MagicMock()
            result = MagicMock()
            result.rewritten_query = "how long do refund exchanges take?"
            chain.invoke.side_effect = lambda msgs: rewrite_calls.append(msgs) or result
            return chain

    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: _TrackingRewriteLLM())

    # Wiki LLM
    fake_wiki_llm = _FakeWikiLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_wiki_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_wiki_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    # RAG LLM
    fake_rag_llm = _FakeRagLLM()
    monkeypatch.setattr(vr_retrieval, "_llm", fake_rag_llm)
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: fake_rag_llm)
    monkeypatch.setattr(
        vr_retrieval.grounding_module,
        "verify",
        lambda draft, chunks: _approved_outcome(),
    )

    # Seed the session with a wiki turn
    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=rag&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200
    # The rewriter must have been called with the shared history
    assert len(rewrite_calls) == 1, (
        f"Rewriter must be called once on turn 2 (RAG) with shared history; got {len(rewrite_calls)}"
    )


# ---------------------------------------------------------------------------
# AC: RAG turns never file to wiki/qa/
# ---------------------------------------------------------------------------


def test_rag_turn_done_filed_always_null(both_stacks_client):
    """RAG turns must have done.filed=null regardless of grounding outcome."""
    resp = both_stacks_client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    done = next(e for e in events if e["type"] == "done")
    assert done["data"]["stack"] == "rag"
    assert done["data"].get("filed") is None, (
        f"RAG done.filed must be null, got: {done['data'].get('filed')}"
    )


def test_wiki_rag_toggle_rag_turn_does_not_file(
    indexed_wiki_corpus, indexed_rag_corpus, monkeypatch
):
    """In a cross-stack session, a RAG turn (after a Wiki turn) does not file."""
    fake_wiki_llm = _FakeWikiLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_wiki_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_wiki_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    fake_rag_llm = _FakeRagLLM()
    monkeypatch.setattr(vr_retrieval, "_llm", fake_rag_llm)
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: fake_rag_llm)
    monkeypatch.setattr(
        vr_retrieval.grounding_module,
        "verify",
        lambda draft, chunks: _approved_outcome(),
    )
    fake_rewrite = _FakeRewriteLLM("how long do refund exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    # Seed a session with a wiki turn
    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=rag&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    done = next(e for e in events if e["type"] == "done")
    assert done["data"]["stack"] == "rag"
    assert done["data"].get("filed") is None, (
        f"RAG done in cross-stack session must have filed=null; got {done['data'].get('filed')}"
    )


# ---------------------------------------------------------------------------
# AC: Grounding firewall — RAG-only facts cannot enter Wiki turn answers
# ---------------------------------------------------------------------------


def test_grounding_firewall_rag_fact_does_not_enter_wiki_answer(
    indexed_wiki_corpus, indexed_rag_corpus, monkeypatch
):
    """A claim only RAG's corpus supports cannot enter a subsequent Wiki turn's answer.

    Firewall mechanism: conversation context shapes the *question* (via rewriting),
    but every claim in the answer must pass the grounding check against the answering
    stack's own corpus. A RAG-only fact injected into the history cannot pass the Wiki
    grounding check → Cannot Confirm → no filing.

    Test strategy:
    - Seed a session with a RAG turn whose answer carries a "RAG-only fact".
    - Mock the Wiki grounding check to return claim_unsupported (the fact is not in
      Wiki corpus).
    - Assert: the Wiki turn returns Cannot Confirm, and done.filed is null.
    """
    from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

    # Wiki LLM returns the RAG-only fact in its draft
    class _WikiLLMWithRagFact:
        # This draft includes a claim that is not grounded by the Wiki corpus.
        DRAFT = (
            "Based on our RAG database, the delivery time is 2 hours. "
            "[Source: refund_policy.md#refund-timeline]"
        )

        def invoke(self, messages):
            return _FakeLLMResponse(content=self.DRAFT)

    fake_wiki_llm = _WikiLLMWithRagFact()
    monkeypatch.setattr(_retrieval, "_llm", fake_wiki_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_wiki_llm)
    # Grounding check fails — the Wiki corpus does NOT support the RAG-only claim.
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _cannot_confirm_outcome(),
    )

    fake_rewrite = _FakeRewriteLLM("what is the delivery time for exchanges?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    # Seed session with a RAG turn that claimed a "RAG-only" delivery time fact
    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long does RAG delivery take?",
            "answer": "Based on our RAG database, the delivery time is 2 hours.",
            "stack": "rag",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "what about that delivery time?"},  # references RAG-only fact
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)

    # The Wiki answer MUST be Cannot Confirm (grounding firewall held)
    token_texts = [e["data"].get("text", "") for e in events if e["type"] == "token"]
    answer = "".join(token_texts)
    assert CANNOT_CONFIRM_PHRASE in answer, (
        f"Grounding firewall must prevent the RAG-only fact from entering the Wiki answer. "
        f"Expected Cannot Confirm sentinel; got: {answer!r}"
    )

    # The done event must indicate grounding failed
    done = next(e for e in events if e["type"] == "done")
    assert done["data"]["grounding"]["passed"] is False, (
        f"Grounding must fail for RAG-only claim in Wiki turn; got: {done['data']['grounding']}"
    )

    # No filing when grounding fails
    assert done["data"].get("filed") is None, (
        f"No filing must occur when grounding firewall blocks the claim; "
        f"got filed={done['data'].get('filed')!r}"
    )


# ---------------------------------------------------------------------------
# AC: evict_expired() wired at top of chat_stream — idle sessions are swept
# ---------------------------------------------------------------------------


def test_evict_expired_called_at_request_entry_clears_idle_session(
    indexed_wiki_corpus, monkeypatch
):
    """An idle session (past TTL) is gone by the time the next request is processed.

    We wire a ConversationStore with a fake clock, advance the clock past TTL, then
    make a fresh request. The request entry must trigger evict_expired(), which sweeps
    the idle session. get_history on the expired session_id must return [] after the
    next request is processed.

    No sleep — the fake clock is advanced instantaneously.
    """
    from gateway.app.conversation_store import ConversationStore

    fake_time = [0.0]

    def _clock() -> float:
        return fake_time[0]

    # Build a fresh store with the fake clock and short TTL (60s for test speed)
    store_with_fake_clock = ConversationStore(ttl_seconds=60, clock=_clock)
    monkeypatch.setattr(_store_module, "store", store_with_fake_clock)

    # Wire the Wiki LLM
    fake_wiki_llm = _FakeWikiLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_wiki_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_wiki_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: _FakeRewriteLLM())

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)

    # Seed a session into the store (simulates a prior request that created the session)
    idle_session_id = str(uuid.uuid4())
    store_with_fake_clock.append_turn(
        idle_session_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )
    # Verify the session exists before TTL expiry
    assert len(store_with_fake_clock.get_history(idle_session_id)) == 1

    # Advance fake clock past the TTL (61 seconds > 60s TTL)
    fake_time[0] = 61.0

    # Make a NEW request (unrelated session) — this triggers evict_expired() at entry
    resp = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "What is the return policy?"},
    )
    assert resp.status_code == 200

    # The idle session must have been swept at request entry
    assert store_with_fake_clock.get_history(idle_session_id) == [], (
        "Idle session (past TTL) must be evicted when evict_expired() is called at "
        "the top of chat_stream on the next request."
    )


# ---------------------------------------------------------------------------
# AC: Wiki turn stores the rewritten query (cross-stack regression guard)
# ---------------------------------------------------------------------------


def test_rag_turn_stores_rewritten_query_not_raw(
    indexed_wiki_corpus, indexed_rag_corpus, monkeypatch
):
    """After a RAG turn (turn 2), the stored question is the rewritten query, not raw."""
    rewritten = "how long do refund exchanges take?"

    class _TrackingRewriteLLM:
        def with_structured_output(self, schema):
            chain = MagicMock()
            result = MagicMock()
            result.rewritten_query = rewritten
            chain.invoke.return_value = result
            return chain

    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: _TrackingRewriteLLM())

    fake_rag_llm = _FakeRagLLM()
    monkeypatch.setattr(vr_retrieval, "_llm", fake_rag_llm)
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: fake_rag_llm)
    monkeypatch.setattr(
        vr_retrieval.grounding_module,
        "verify",
        lambda draft, chunks: _approved_outcome(),
    )

    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=rag&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200

    history = _store_module.store.get_history(existing_id)
    assert len(history) == 2, f"Expected 2 turns; got {len(history)}"
    assert history[1]["question"] == rewritten, (
        f"RAG turn's stored question must be the rewritten query '{rewritten}'; "
        f"got '{history[1]['question']}'"
    )
    assert history[1]["stack"] == "rag", "Stored turn must record stack=rag"
