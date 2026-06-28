"""Gateway endpoint tests for POST /chat/stream?stack=hybrid (Phase 13 S4 / #314).

Mirrors ``test_chat_stream_rag.py`` — adds ``hybrid`` as the third stack branch
everywhere ``rag`` appears in the dispatch / validation / error path. Asserts the
hybrid SSE event sequence (sources → status → token(s) → done), the clickable
citation path, the unknown-stack error now naming ``hybrid``, uniform Cannot
Confirm rendering, and that multi-turn is inherited at the Gateway (Query
Rewriting is stack-agnostic, ADR-0013 — Hybrid gets it for free, no new code).

Both retrieval arms (BM25 + dense FAISS) run FOR REAL over a small synthetic wiki
corpus; only the genuine network leaves are faked — the synthesis LLM via its
lazy getter (trap #1), the embedding leaf via fake embeddings, and the verifier
stubbed directly (no getter seam). No OPENAI_API_KEY.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from unittest.mock import MagicMock

import hybrid_kb.app.dense_index as hk_dense
import hybrid_kb.app.logger as hk_logger
import hybrid_kb.app.query as hk_query
import markdown_kb.app.indexer as bm25_indexer
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from markdown_kb.app.grounding import GroundingOutcome
from markdown_kb.app.indexer import Section

import gateway.app.conversation_store as _store_module
import gateway.app.query_rewriting as _rewrite_module

REFUND_ID = "refund-policy#refund-policy"


# ---------------------------------------------------------------------------
# Fake embeddings (mirrors the hybrid_kb conftest fake)
# ---------------------------------------------------------------------------


class _FakeEmbeddings(Embeddings):
    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# Fake LLMs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    CANNED_ANSWER = (
        f"Approved refunds are processed within seven business days. [Source: {REFUND_ID}]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


class _FakeRewriteLLM:
    """Structured-output rewrite LLM stub returning a fixed rewritten query."""

    def __init__(self, rewritten: str = "how long does a refund take?") -> None:
        self._rewritten = rewritten

    def with_structured_output(self, schema):
        chain = MagicMock()
        result = MagicMock()
        result.rewritten_query = self._rewritten
        chain.invoke.return_value = result
        return chain


# ---------------------------------------------------------------------------
# Synthetic wiki corpus (type=qa so the citation path resolves → clickable)
# ---------------------------------------------------------------------------


def _wiki_section(section_id: str, content: str, heading: str) -> Section:
    return Section(
        id=section_id,
        file=section_id.split("#")[0],
        heading=heading,
        heading_path=[heading],
        content=content,
        tokens=bm25_indexer.tokenize(content),
        metadata={"lang": "en", "type": "qa"},
    )


_CORPUS = [
    _wiki_section(
        REFUND_ID,
        "Refund policy: refunds are processed within seven business days after "
        "approval. A refund usually takes about one week.",
        "Refund Policy",
    ),
    _wiki_section(
        "shipping-policy#shipping-policy",
        "Shipping policy: standard shipping delivery takes three to five business "
        "days within the country.",
        "Shipping Policy",
    ),
]


def _approved_outcome() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported", result=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_hybrid_paths_to_tmp(tmp_path, monkeypatch):
    """Keep the dense seed + hybrid log off the committed paths for every test."""
    monkeypatch.setattr(hk_dense, "DENSE_INDEX_DIR", tmp_path / ".kb" / "hybrid_dense")
    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")


@pytest.fixture()
def wired_hybrid_corpus(monkeypatch):
    """Build BOTH arms over the synthetic corpus with no deep-module mocking."""
    fake = _FakeEmbeddings()
    monkeypatch.setattr(hk_dense, "get_embeddings", lambda: fake)
    bm25_indexer.sections = list(_CORPUS)
    bm25_indexer.rebuild_stats()
    hk_dense.build_index(sections=list(_CORPUS))
    yield
    bm25_indexer.sections = []
    bm25_indexer.rebuild_stats()
    hk_dense.vectorstore = None
    hk_dense.sections_indexed = 0


@pytest.fixture()
def hybrid_gateway_client(wired_hybrid_corpus, monkeypatch):
    """TestClient for the Gateway with the hybrid synthesis LLM + verifier mocked."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(hk_query, "_llm", fake_llm)
    monkeypatch.setattr(hk_query, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        hk_query.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse_response(content: str) -> list[dict]:
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


# ===========================================================================
# Dispatch: stack=hybrid lands (parity with the rag dispatch test)
# ===========================================================================


def test_chat_stream_hybrid_returns_200(hybrid_gateway_client):
    """stack=hybrid dispatches to the hybrid streaming path (HTTP 200)."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert "text/event-stream" in resp.headers["content-type"]


# ===========================================================================
# AC4 — hybrid SSE event sequence: sources → status → token(s) → done
# ===========================================================================


def test_chat_stream_hybrid_event_order(hybrid_gateway_client):
    """POST /chat/stream?stack=hybrid emits: sources, status, token(s), done."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200

    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    assert types[0] == "sources", f"Expected first event 'sources', got {types[0]!r}"
    assert types[-1] == "done", f"Expected last event 'done', got {types[-1]!r}"
    middle = types[1:-1]
    assert middle and middle[0] == "status", f"Expected 'status' after sources: {types}"
    assert all(t == "token" for t in middle[1:]), f"Events after status must be 'token': {types}"


def test_chat_stream_hybrid_sources_non_empty(hybrid_gateway_client):
    """Hybrid sources event carries at least one source with the required fields."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "What is the refund policy?"},
    )
    sources = _parse_sse_response(resp.text)[0]["data"]["sources"]
    assert len(sources) >= 1
    item = sources[0]
    assert {"source", "heading", "content"} <= set(item)


def test_chat_stream_hybrid_sources_carry_clickable_wiki_path(hybrid_gateway_client):
    """The gateway forwards a resolvable wiki/ path so hybrid citations are clickable."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "What is the refund policy?"},
    )
    sources = _parse_sse_response(resp.text)[0]["data"]["sources"]
    refund = next(s for s in sources if s["source"] == REFUND_ID)
    assert refund.get("path") == "wiki/qa/refund-policy.md", (
        f"hybrid source must carry a clickable wiki-page path: {refund}"
    )
    assert "\\" not in refund["path"], "path must be forward-slashed"


def test_chat_stream_hybrid_tokens_form_answer(hybrid_gateway_client):
    """Joining token texts reconstructs the LLM-generated (verified) answer."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "What is the refund policy?"},
    )
    events = _parse_sse_response(resp.text)
    answer = "".join(e["data"]["text"] for e in events if e["type"] == "token")
    assert "seven business days" in answer


def test_chat_stream_hybrid_done_stack_and_passed(hybrid_gateway_client):
    """done carries grounding.passed=True and stack=hybrid (gateway-injected)."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "What is the refund policy?"},
    )
    done = _parse_sse_response(resp.text)[-1]
    assert done["type"] == "done"
    assert done["data"]["grounding"]["passed"] is True
    assert done["data"]["grounding"]["reason"] == "claim_supported"
    assert done["data"]["stack"] == "hybrid"


def test_chat_stream_hybrid_done_filed_null(hybrid_gateway_client):
    """Hybrid never files — done.filed is always null (parity with RAG)."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "What is the refund policy?"},
    )
    done = _parse_sse_response(resp.text)[-1]
    assert done["data"].get("filed") is None


# ===========================================================================
# §12.3 — uniform Cannot Confirm through the gateway (pre-LLM OR-gate)
# ===========================================================================


def test_chat_stream_hybrid_cannot_confirm_is_uniform(hybrid_gateway_client, monkeypatch):
    """An out-of-scope query streams the Cannot Confirm phrase, no status:verifying.

    Forcing the dense ceiling to 0 isolates the BM25 arm; a nonsense query leaves
    both arms below threshold, so the pre-LLM OR-gate refuses. The stream still
    emits sources first, then token(s) of the sentinel, then done{passed:false} —
    and skips the verifying-status event (no LLM ran).
    """
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=hybrid",
        json={"query": "xylophone quokka zugzwang nonsense"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    assert types[0] == "sources"
    assert types[-1] == "done"
    assert "status" not in types, f"early-exit CC must skip status:verifying: {types}"

    answer = "".join(e["data"]["text"] for e in events if e["type"] == "token")
    assert answer == hk_query.CANNOT_CONFIRM_PHRASE
    done = events[-1]
    assert done["data"]["grounding"]["passed"] is False
    assert done["data"]["grounding"]["reason"] == "below_threshold"
    assert done["data"]["stack"] == "hybrid"


# ===========================================================================
# Validation — unknown stack still 400, and the error names 'hybrid'
# ===========================================================================


def test_chat_stream_unknown_stack_error_names_hybrid(hybrid_gateway_client):
    """Unknown stack returns 400 and the error message now includes 'hybrid'."""
    resp = hybrid_gateway_client.post(
        "/chat/stream?stack=unknown",
        json={"query": "test"},
    )
    assert resp.status_code == 400
    assert "hybrid" in resp.json()["detail"], (
        f"unknown-stack error must name 'hybrid': {resp.json()['detail']!r}"
    )


# ===========================================================================
# AC5 — multi-turn is inherited at the Gateway (Query Rewriting, ADR-0013)
# ===========================================================================


@pytest.fixture()
def _fresh_store(monkeypatch):
    from gateway.app.conversation_store import ConversationStore

    fresh = ConversationStore()
    monkeypatch.setattr(_store_module, "store", fresh)
    return fresh


def test_chat_stream_hybrid_multiturn_resolves_context(
    hybrid_gateway_client, _fresh_store, monkeypatch
):
    """A Hybrid follow-up resolves conversational context via inherited rewriting.

    Seed a prior turn, then post an elliptical follow-up on stack=hybrid. The
    Gateway (not Hybrid) must: emit status:rewriting BEFORE sources, call the
    rewrite LLM, and store the REWRITTEN self-contained query — proving Hybrid
    inherits multi-turn with no stack-specific conversation code.
    """
    monkeypatch.setattr(_rewrite_module, "_rewrite_llm", None)
    rewritten = "how long does a refund take?"
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: _FakeRewriteLLM(rewritten))

    existing_id = str(uuid.uuid4())
    _fresh_store.append_turn(
        existing_id,
        {
            "question": "What is the refund policy?",
            "answer": "Refunds take about a week.",
            "stack": "hybrid",
            "grounding_reason": "claim_supported",
            "ts": "2026-06-28T10:00:00Z",
        },
    )

    resp = hybrid_gateway_client.post(
        f"/chat/stream?stack=hybrid&session={existing_id}",
        json={"query": "and how long does it take?"},  # raw elliptical follow-up
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    # status:rewriting precedes sources (Gateway Query Rewriting, ADR-0013).
    rewriting = [
        i
        for i, e in enumerate(events)
        if e["type"] == "status" and e.get("data", {}).get("phase") == "rewriting"
    ]
    assert rewriting, f"hybrid turn 2 must emit status:rewriting: {types}"
    assert rewriting[0] < types.index("sources"), f"status:rewriting must precede sources: {types}"

    # The stored turn-2 question is the REWRITTEN self-contained query.
    history = _fresh_store.get_history(existing_id)
    assert len(history) == 2
    assert history[1]["question"] == rewritten, (
        f"hybrid must store the rewritten query, got {history[1]['question']!r}"
    )
    # done echoes the same session id (continuity).
    assert events[-1]["data"]["session"] == existing_id
