"""Gateway endpoint tests for Answer Filing parity on POST /chat/stream?stack=wiki.

Phase 9 Slice 4 (issue #121) — AC: wiki stream files on grounding-pass exactly as
/chat does; Cannot Confirm streams do not file; RAG streams never file; /chat
filing behavior is unchanged.

Reuses Phase 6 filing test patterns from markdown_kb/tests/test_routes_filing.py:
- Mocked LLM (no OPENAI_API_KEY)
- Hermetic tmp_path via autouse _redirect_paths_to_tmp
- Same FakeLLM + _approved_outcome pattern as existing gateway tests
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _logger
import markdown_kb.app.retrieval as _retrieval
import pytest
from fastapi.testclient import TestClient
from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult
from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"
# Hermetic 3-doc corpus (account_help / refund_policy / shipping_faq) — same
# corpus used by test_chat_stream.py.  Gibberish queries reliably fall below
# the BM25 threshold against this small corpus, avoiding the fake-docs
# pollution that made test_wiki_stream_cannot_confirm_does_not_file flaky
# (issue #204 / same class as #145).
_FIXTURE_DOCS = Path(__file__).resolve().parents[2] / "markdown_kb" / "tests" / "fixtures" / "docs"
REFUND_SECTION_ID = "refund_policy.md#refund-timeline"


# ---------------------------------------------------------------------------
# Fake LLM stub (same pattern as test_chat_stream.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    CANNED_ANSWER = (
        f"Approved refunds are processed within 5-7 business days. [Source: {REFUND_SECTION_ID}]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


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
                    citing_section_ids=[REFUND_SECTION_ID],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect INDEX_PATH, LOG_PATH, WIKI_DIR to tmp for all filing tests."""
    monkeypatch.setattr(_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")


@pytest.fixture()
def indexed_wiki_corpus(tmp_path, monkeypatch):
    """Build the Section Index from the hermetic 3-doc fixture corpus.

    Switched from REAL_DOCS (full docs/ incl. fake-docs) to _FIXTURE_DOCS to
    prevent BM25 corpus pollution from the 20-doc fake-docs tree that caused
    ``test_wiki_stream_cannot_confirm_does_not_file`` to flake (~12% rate):
    a gibberish query would score above the threshold against the expanded
    corpus, reach the LLM, and have grounding.passed flip True (issue #204,
    same class as #145).
    """
    _indexer.build_index(_FIXTURE_DOCS)
    yield
    _indexer.sections.clear()


@pytest.fixture()
def grounded_stream_client(indexed_wiki_corpus, monkeypatch):
    """TestClient for the Gateway with a mocked LLM that always grounds."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


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
# AC1: A grounded Wiki stream files and populates done.filed
# ---------------------------------------------------------------------------


def test_wiki_stream_grounded_files_and_surfaces_done_filed(grounded_stream_client, tmp_path):
    """A grounding-passing Wiki stream creates a wiki/qa/<slug>.md draft
    and populates done.filed with slug / status / op / count.

    Phase 9 Slice 4 AC: filing happens at the post-verify point server-side
    (independent of client delivery — a disconnected client never causes
    partial/unfiled state, and a partial answer is never filed).
    """
    resp = grounded_stream_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200

    events = _parse_sse_response(resp.text)
    done_events = [e for e in events if e["type"] == "done"]
    assert done_events, f"Expected a done event; got types: {[e['type'] for e in events]}"

    done_data = done_events[-1]["data"]
    # PRD-locked shape: done.grounding.passed (not flat)
    assert done_data["grounding"]["passed"] is True

    # AC1: done.filed is populated on grounding-pass
    filed = done_data.get("filed")
    assert filed is not None, (
        f"done.filed must be non-null for grounded wiki stream; got {done_data}"
    )
    assert "slug" in filed, f"filed must have slug: {filed}"
    assert "status" in filed, f"filed must have status: {filed}"
    assert "op" in filed, f"filed must have op: {filed}"
    assert "count" in filed, f"filed must have count: {filed}"

    assert filed["status"] == "draft"
    assert filed["op"] == "created"
    assert filed["count"] == 1

    # Verify file was actually written to disk
    slug = filed["slug"]
    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    assert qa_path.exists(), f"Expected qa file at {qa_path}"
    content = qa_path.read_text(encoding="utf-8")
    assert "status: draft" in content
    assert "count: 1" in content


def test_wiki_stream_grounded_second_ask_touches_filed(grounded_stream_client, tmp_path):
    """Re-asking the same question bumps count and reports op=touched."""
    first_resp = grounded_stream_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert first_resp.status_code == 200
    first_events = _parse_sse_response(first_resp.text)
    first_done = next(e for e in first_events if e["type"] == "done")
    first_slug = first_done["data"]["filed"]["slug"]

    second_resp = grounded_stream_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert second_resp.status_code == 200
    second_events = _parse_sse_response(second_resp.text)
    second_done = next(e for e in second_events if e["type"] == "done")
    second_filed = second_done["data"]["filed"]

    assert second_filed is not None, "Second ask must also populate done.filed"
    assert second_filed["op"] == "touched"
    assert second_filed["count"] == 2
    assert second_filed["slug"] == first_slug, "Same Q must resolve to same slug"


# ---------------------------------------------------------------------------
# AC2: Cannot Confirm Wiki streams do NOT file
# ---------------------------------------------------------------------------


def test_wiki_stream_cannot_confirm_does_not_file(indexed_wiki_corpus, monkeypatch, tmp_path):
    """Cannot Confirm Wiki stream: done.filed is null and wiki/qa/ stays empty.

    Hermetic setup (issue #204): uses the 3-doc fixture corpus (not the full
    docs/ with fake-docs) so gibberish reliably scores below the BM25 threshold.
    Additionally mocks the LLM + grounding.verify at the get_llm/_llm seam to
    force a deterministic Cannot-Confirm even if BM25 were to pass: the test's
    AC is that CC streams do NOT file — it does not assert on the BM25 gate
    mechanism, so a mocked CC is a valid validator (no OPENAI_API_KEY needed).
    """

    # Belt-and-suspenders: mock LLM to return CANNOT_CONFIRM_PHRASE and mock
    # grounding.verify to return passed=False.  Together with the hermetic corpus
    # this guarantees a CC outcome regardless of BM25 scoring behaviour.
    class _FakeCCLLM:
        def invoke(self, messages):
            return _FakeLLMResponse(content=CANNOT_CONFIRM_PHRASE)

    fake_cc_llm = _FakeCCLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_cc_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_cc_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: GroundingOutcome(passed=False, reason="claim_unsupported"),
    )

    # Unrelated gibberish → BM25 scores fall below threshold → CC (pre-LLM gate).
    # The LLM mock above handles the rare case where BM25 scores above threshold.
    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "xyzzy banana orbit completely unrelated gibberish qqqq"},
    )
    assert resp.status_code == 200

    events = _parse_sse_response(resp.text)
    done_events = [e for e in events if e["type"] == "done"]
    assert done_events, "Expected a done event"

    done_data = done_events[-1]["data"]
    # PRD-locked shape: done.grounding.passed
    assert done_data["grounding"]["passed"] is False

    # AC2: no filing on CC path
    filed = done_data.get("filed")
    assert filed is None, f"done.filed must be null on CC paths; got: {filed}"

    qa_dir = tmp_path / "wiki" / "qa"
    qa_files = list(qa_dir.glob("*.md")) if qa_dir.exists() else []
    assert qa_files == [], f"wiki/qa must stay empty on CC streams; got: {qa_files}"


# ---------------------------------------------------------------------------
# AC3: RAG streams NEVER file (kept null even on grounding pass)
# ---------------------------------------------------------------------------


def test_rag_stream_done_filed_null_on_grounding_pass(tmp_path, monkeypatch):
    """RAG stream done.filed is always null even when grounding passes.

    Phase 9 Slice 4 AC: RAG never files — keep the constraint introduced in
    Slice 3 (#120) and confirm it is preserved after the filing extraction.
    """
    import hashlib

    import vector_rag.app.indexer as vr_indexer
    import vector_rag.app.logger as vr_logger
    import vector_rag.app.retrieval as vr_retrieval
    from langchain_core.embeddings import Embeddings
    from markdown_kb.app.grounding import GroundingOutcome

    class _FakeEmbeddings(Embeddings):
        _DIM = 16

        def _vec(self, text: str) -> list[float]:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            return [b / 255.0 for b in digest[: self._DIM]]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._vec(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._vec(text)

    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: _FakeEmbeddings())
    vr_indexer.build_index(REAL_DOCS)

    fake_llm = _FakeLLM()
    monkeypatch.setattr(vr_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(vr_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        vr_retrieval.grounding_module,
        "verify",
        lambda draft, chunks: GroundingOutcome(passed=True, reason="claim_supported"),
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        "/chat/stream?stack=rag",
        json={"query": "What is the refund policy?"},
    )
    assert resp.status_code == 200

    events = _parse_sse_response(resp.text)
    done_events = [e for e in events if e["type"] == "done"]
    assert done_events, "Expected a done event"
    done_data = done_events[-1]["data"]
    # PRD-locked shape: done.grounding.passed
    assert done_data["grounding"]["passed"] is True

    # AC3: RAG never files
    filed = done_data.get("filed")
    assert filed is None, f"RAG done.filed must always be null; got: {filed}"

    qa_dir = tmp_path / "wiki" / "qa"
    qa_files = list(qa_dir.glob("*.md")) if qa_dir.exists() else []
    assert qa_files == [], f"wiki/qa must stay empty for RAG streams; got: {qa_files}"

    # Clean up
    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0


# ---------------------------------------------------------------------------
# AC4 (regression): /chat filing behavior unchanged after extraction
# ---------------------------------------------------------------------------


def test_chat_route_filing_still_works_after_extraction(grounded_stream_client, tmp_path):
    """POST /chat (markdown_kb direct) still files on grounding pass.

    Regression check: extracting the filing dispatch into a shared helper
    must be behavior-preserving for the existing /chat endpoint.
    The grounded_stream_client fixture uses the same markdown_kb app instance,
    so we can reach /chat directly from the same TestClient.
    """
    # The grounded_stream_client points to the gateway app, but we need to
    # reach markdown_kb's /chat. Test via the markdown_kb app directly.
    from markdown_kb.app.main import app as _mkb_app

    mkb_client = TestClient(_mkb_app)

    resp = mkb_client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 200
    body = resp.json()

    # /chat must still return filed on grounding pass
    assert body.get("filed") is not None, (
        f"POST /chat must still populate filed after extraction; got: {body}"
    )
    assert body["filed"]["op"] in ("created", "touched")
    assert body["filed"]["status"] == "draft"
