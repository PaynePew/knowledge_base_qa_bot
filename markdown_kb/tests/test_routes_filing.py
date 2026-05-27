"""Integration tests for the Phase 6 Slice 6-2 ``/chat`` filing side-effect.

Verifies the cross-module wiring from ``routes.chat`` through
``qa.maybe_file_answer`` and back to ``ChatResponse.filed``. Mocking is
limited to the LLM (via the ``retrieval`` and ``grounding`` getters) — the
indexer, retrieval, qa, and filesystem layers are exercised end-to-end on
``tmp_path`` per the project's hermetic-test convention.

Coverage maps directly to issue #80 ``/chat`` integration AC block:

- Grounding-passing first ``/chat`` → ``filed.op=created``; second → ``filed.op=touched``
- Cannot-Confirm ``/chat`` → ``filed is None``; ``wiki/qa/`` empty
- Filing IOError (monkeypatched) → response 200 with ``filed: None``;
  ``qa_filing_error`` in log; ``answer`` still populated
- Concurrent same-Q ``/chat`` via threaded TestClient → exactly one file +
  ``count`` matches request count

NOTE on imports: this file uses LAZY imports of ``app.indexer``,
``app.retrieval``, etc. inside fixtures. ``test_persistence.py`` clears the
``app.*`` modules from ``sys.modules`` to simulate a server restart, so any
module-level reference taken before that point would become stale relative
to the post-restart sys.modules entries. Module-level imports here would
silently fail when this file runs after ``test_persistence.py`` (collection
order is alphabetical so this happens by default).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .conftest import FakeLLMResponse

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"
REFUND_SECTION_ID = "refund_policy.md#refund-timeline"


# ---------------------------------------------------------------------------
# FakeLLM stub (same shape as test_chat_grounded.py to stay idiomatic)
# ---------------------------------------------------------------------------


class FakeLLM:
    """Canned grounded answer for refund queries."""

    def __init__(self, source_id: str = REFUND_SECTION_ID):
        self.source_id = source_id
        self.last_messages: list = []

    def invoke(self, messages: list):
        self.last_messages = messages
        return FakeLLMResponse(
            content=(
                f"Approved refunds are processed within 5-7 business days. "
                f"[Source: {self.source_id}]"
            )
        )


def _approved_outcome(source_id: str = REFUND_SECTION_ID):
    """Build an approved GroundingOutcome — imported lazily for the same reason
    as the other ``app.*`` imports (see module docstring)."""
    from app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims trace to cited section.",
            claims=[
                GroundingClaim(
                    text="Approved refunds are processed within 5-7 business days.",
                    supported=True,
                    citing_section_ids=[source_id],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


# ---------------------------------------------------------------------------
# Built-index fixture — equivalent to conftest's ``indexed_corpus`` but uses
# the *current* ``app.indexer`` module so it survives ``test_persistence``'s
# sys.modules purge regardless of test collection order.
# ---------------------------------------------------------------------------


@pytest.fixture()
def built_corpus():
    """Build the real-docs corpus into ``tmp_path`` against the current
    indexer module, clearing sections on teardown."""
    import app.indexer as current_indexer

    current_indexer.build_index(REAL_DOCS)
    yield
    current_indexer.sections.clear()


@pytest.fixture()
def grounded_client(built_corpus, monkeypatch):
    """TestClient where grounding passes — exercises the filing dispatch."""
    import app.retrieval as retrieval_module

    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(REFUND_SECTION_ID),
    )

    from app.main import app

    return TestClient(app), fake_llm


# ---------------------------------------------------------------------------
# Grounding-passing path: first ask creates; second touches
# ---------------------------------------------------------------------------


def test_chat_grounded_first_ask_creates_filed(grounded_client, tmp_path):
    """A passing grounding check populates ``response.filed`` with op=created."""
    client, _ = grounded_client

    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["filed"] is not None, f"Expected response.filed populated, got: {body}"
    assert body["filed"]["op"] == "created"
    assert body["filed"]["status"] == "draft"
    assert body["filed"]["count"] == 1

    slug = body["filed"]["slug"]
    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    assert qa_path.exists(), f"Expected qa file at {qa_path}"
    content = qa_path.read_text(encoding="utf-8")
    assert "status: draft" in content
    assert "count: 1" in content


def test_chat_grounded_second_ask_touches_filed(grounded_client, tmp_path):
    """Re-asking the same Q bumps count and reports op=touched."""
    client, _ = grounded_client

    first_resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert first_resp.status_code == 200
    first_slug = first_resp.json()["filed"]["slug"]

    second_resp = client.post("/chat", json={"query": "How long do refunds take?"})
    assert second_resp.status_code == 200
    body = second_resp.json()
    assert body["filed"] is not None
    assert body["filed"]["op"] == "touched"
    assert body["filed"]["count"] == 2
    assert body["filed"]["slug"] == first_slug, "Same Q must resolve to same slug"


# ---------------------------------------------------------------------------
# Cannot Confirm: no filing, wiki/qa stays empty
# ---------------------------------------------------------------------------


def test_chat_cannot_confirm_does_not_file(built_corpus, monkeypatch, tmp_path):
    """When grounding does not pass, ``filed`` is None and wiki/qa stays empty."""
    import app.retrieval as retrieval_module
    from app.retrieval import CANNOT_CONFIRM_PHRASE

    # Sentinel LLM that would fail if called (we expect pre-LLM gate to fire
    # because the query has no BM25 matches in the indexed corpus).
    class _SentinelLLM:
        def invoke(self, messages):
            raise AssertionError("LLM must not be called for Cannot Confirm path")

    monkeypatch.setattr(retrieval_module, "_llm", _SentinelLLM())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: _SentinelLLM())

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == CANNOT_CONFIRM_PHRASE
    assert body["filed"] is None, f"Cannot Confirm must NOT file; got: {body['filed']}"

    qa_dir = tmp_path / "wiki" / "qa"
    qa_files = list(qa_dir.glob("*.md")) if qa_dir.exists() else []
    assert qa_files == [], f"wiki/qa must be empty on Cannot Confirm, got: {qa_files}"


# ---------------------------------------------------------------------------
# F3 fail-soft: filing IOError leaves answer intact, filed=None, log shows error
# ---------------------------------------------------------------------------


def test_chat_filing_io_error_failsoft(grounded_client, tmp_path, monkeypatch):
    """Monkeypatched ``os.replace`` → IOError; response is 200 with answer + filed=None."""
    import app.qa as qa_module

    monkeypatch.setattr(
        qa_module.os,
        "replace",
        lambda src, dst: (_ for _ in ()).throw(OSError("simulated disk full")),
    )

    client, _ = grounded_client
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, (
        f"F3 fail-soft must return 200 even when filing fails; got {resp.status_code}: {resp.text}"
    )
    body = resp.json()

    # Answer is still populated — primary value of /chat is the answer
    assert body["answer"], "Answer must be populated despite filing failure"
    assert "Approved refunds" in body["answer"]
    # filed must be None — caller knows filing didn't happen
    assert body["filed"] is None, f"Filing failure must surface as filed=None; got: {body['filed']}"

    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "qa_filing_error" in log
    assert "reason=io_error" in log


# ---------------------------------------------------------------------------
# L1 concurrency through the route: 8 threaded TestClient requests, one file
# ---------------------------------------------------------------------------


def test_chat_concurrent_same_query_creates_one_file(grounded_client, tmp_path):
    """8 parallel /chat calls with the same query → one file, count=8.

    TestClient is thread-safe for the FastAPI app (each ``post`` is independent);
    the contention surface under test is ``qa._filing_lock`` inside
    ``maybe_file_answer``. Verifies the lock-around-decision contract holds
    when the entry point is the route, not the qa module directly.
    """
    client, _ = grounded_client

    def post_once(_i):
        return client.post("/chat", json={"query": "How long do refunds take?"})

    with ThreadPoolExecutor(max_workers=8) as ex:
        responses = list(ex.map(post_once, range(8)))

    statuses = [r.status_code for r in responses]
    assert all(s == 200 for s in statuses), f"All 8 requests must return 200; got: {statuses}"

    bodies = [r.json() for r in responses]
    assert all(b["filed"] is not None for b in bodies), (
        f"All concurrent requests must report a filing outcome, got: {[b['filed'] for b in bodies]}"
    )

    ops = [b["filed"]["op"] for b in bodies]
    assert ops.count("created") == 1, f"Exactly one of 8 must be a create; got ops: {ops}"
    assert ops.count("touched") == 7

    slug = bodies[0]["filed"]["slug"]
    # Sanity: all 8 should resolve to the same slug
    assert all(b["filed"]["slug"] == slug for b in bodies), "All requests should share one slug"

    qa_dir = tmp_path / "wiki" / "qa"
    qa_files = list(qa_dir.glob("*.md"))
    assert len(qa_files) == 1, f"Expected exactly one qa file, got: {qa_files}"
    content = qa_files[0].read_text(encoding="utf-8")
    assert "count: 8" in content, f"Final count must equal request count (8). File:\n{content}"


# ---------------------------------------------------------------------------
# Sanity: existing /chat response signature unchanged
# (ensures the new ``filed`` field is purely additive — old clients still parse)
# ---------------------------------------------------------------------------


def test_chat_response_keeps_existing_keys(grounded_client):
    """``answer``, ``sources``, ``grounding`` keys must still be present and well-typed."""
    client, _ = grounded_client
    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert "grounding" in body
    assert "filed" in body, "filed field must be present (None or FiledStatus)"
    assert isinstance(body["sources"], list)
    assert isinstance(body["grounding"], dict)
