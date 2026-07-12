"""Degraded key-free serving at full budget cap (issue #598 Slice B).

At full ``KB_DAILY_USD_CAP``, ``POST /chat/stream?stack=wiki`` is admitted
past the budget gate with a ``degraded`` flag instead of 503'd — the wiki
stack's ``stream_query`` serves a no-LLM answer (see
``markdown_kb/tests/test_retrieval_degraded.py`` for that branch's own
coverage). Every other READ_PATHS surface (``stack=rag``, ``/wiki/chat``,
``/rag/chat``) has no such branch downstream, so it must keep the existing
hard 503 — admitting it would let a real (uncounted) LLM call through past
the budget ceiling.

All hermetic — no OPENAI_API_KEY, no real network; no wiki corpus is indexed,
so the wiki dispatch hits the ``index_missing`` early-exit either way (no LLM
call happens on either the degraded or the (hypothetical) non-degraded path
for this fixture), which is exactly what keeps this a middleware-focused test.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _fresh_app():
    """Reload the gateway app so module-level middleware/budget state is pristine."""
    import gateway.app.budget as budget_mod
    import gateway.app.main as main_mod
    import gateway.app.middleware as mw_mod
    import gateway.app.ratelimit as ratelimit_mod

    importlib.reload(budget_mod)
    importlib.reload(ratelimit_mod)
    importlib.reload(mw_mod)
    importlib.reload(main_mod)
    return main_mod.app


@pytest.fixture()
def exhausted_client(monkeypatch):
    """A client whose daily budget is exhausted from the very first request."""
    monkeypatch.setenv("KB_DAILY_USD_CAP", "0.0")
    monkeypatch.delenv("KB_ADMIN_TOKEN", raising=False)
    return TestClient(_fresh_app())


def _parse_sse_response(content: str) -> list[dict]:
    import json

    events = []
    for frame in content.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event_type = "message"
        data_str = ""
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:]
        if data_str:
            events.append({"type": event_type, "data": json.loads(data_str)})
    return events


# ---------------------------------------------------------------------------
# stack=wiki is admitted degraded instead of 503'd
# ---------------------------------------------------------------------------


def test_chat_stream_wiki_over_cap_is_not_503(exhausted_client):
    resp = exhausted_client.post("/chat/stream?stack=wiki", json={"query": "hi"})
    assert resp.status_code == 200, "an over-cap stack=wiki request must degrade, not 503"


def test_chat_stream_wiki_over_cap_done_event_carries_degraded_true(exhausted_client):
    resp = exhausted_client.post("/chat/stream?stack=wiki", json={"query": "hi"})
    events = _parse_sse_response(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    assert done["data"]["degraded"] is True
    assert done["data"]["grounding"]["reason"] == "degraded_budget_exhausted"


def test_chat_stream_wiki_over_cap_does_not_charge_budget(exhausted_client):
    import gateway.app.budget as budget_mod

    before = budget_mod.budget.day_total()
    exhausted_client.post("/chat/stream?stack=wiki", json={"query": "hi"})
    assert budget_mod.budget.day_total() == before, (
        "a degraded (no-LLM) admission must never be charged"
    )


def test_chat_stream_wiki_under_cap_done_event_carries_degraded_false(monkeypatch):
    """The additive ``degraded`` field is always present, False when not exhausted."""
    monkeypatch.delenv("KB_DAILY_USD_CAP", raising=False)
    client = TestClient(_fresh_app())
    resp = client.post("/chat/stream?stack=wiki", json={"query": "hi"})
    events = _parse_sse_response(resp.text)
    done = events[-1]
    assert done["data"]["degraded"] is False


# ---------------------------------------------------------------------------
# Every other READ_PATHS surface keeps the hard 503 -- no downstream no-LLM
# branch exists for it, so admitting it would let an uncounted LLM call through.
# ---------------------------------------------------------------------------


def test_chat_stream_rag_over_cap_still_503(exhausted_client):
    resp = exhausted_client.post("/chat/stream?stack=rag", json={"query": "hi"})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "daily demo budget reached"}


def test_chat_stream_hybrid_over_cap_still_503(exhausted_client):
    resp = exhausted_client.post("/chat/stream?stack=hybrid", json={"query": "hi"})
    assert resp.status_code == 503


def test_wiki_chat_over_cap_still_503(exhausted_client):
    resp = exhausted_client.post("/wiki/chat", json={"query": "hi"})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "daily demo budget reached"}


def test_rag_chat_over_cap_still_503(exhausted_client):
    resp = exhausted_client.post("/rag/chat", json={"query": "hi"})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "daily demo budget reached"}


# ---------------------------------------------------------------------------
# The other guards still apply to a degradable request -- degraded serving is
# an exception to the BUDGET gate only, not a bypass of every guard.
# ---------------------------------------------------------------------------


def test_chat_stream_wiki_over_cap_still_rate_limited(exhausted_client, monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "1")
    client = TestClient(_fresh_app())
    first = client.post("/chat/stream?stack=wiki", json={"query": "hi"})
    assert first.status_code == 200
    second = client.post("/chat/stream?stack=wiki", json={"query": "hi"})
    assert second.status_code == 429, "rate limiting must still gate a degraded request"


def test_chat_stream_wiki_over_cap_still_shed_when_read_sem_saturated(exhausted_client):
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.read_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = exhausted_client.post("/chat/stream?stack=wiki", json={"query": "hi"})
        assert resp.status_code == 503, "the concurrency cap must still shed a degraded request"
    finally:
        for _ in acquired:
            mw_mod.read_sem.release()
