"""Duplicate `stack` query-param resolution consistency (issue #689).

PR #685's verdict finding M3 (confirmed by live measurement): the ASGI budget
middleware and the Starlette-parsed FastAPI route disagreed on how a repeated
`stack` query param resolves. `middleware.py::_stack_param` used to take the
FIRST occurrence (`parse_qs(...)[0]`) while the route's own `stack: str = "rag"`
binding -- resolved off Starlette's `QueryParams`, whose `.get()` returns the
LAST occurrence of a repeated key -- took the LAST. For
`?stack=wiki&stack=rag` over an exhausted daily budget cap, the middleware
believed the request was degradable wiki (admitting it, uncharged) while the
route actually dispatched rag -- a real, uncounted LLM call past the cap the
middleware exists to enforce.

Fix: `_stack_param` now takes the LAST occurrence too, matching the route.
This file asserts the consistency invariant directly (not just in a comment)
and proves both duplicate-param orderings behave safely over an exhausted cap.

All hermetic -- no OPENAI_API_KEY, no real network; no wiki/RAG corpus is
indexed, so the wiki dispatch (when actually reached) hits the
`index_missing` early-exit (no LLM call either way, see
`test_budget_degraded_serving.py`'s own rationale), and the rag/hybrid
dispatch is never reached at all in the non-degradable direction -- the
request is rejected at the ASGI layer before the route runs.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import QueryParams


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
# The consistency invariant itself, asserted directly: whatever
# `_stack_param` resolves for a raw query string must match what Starlette's
# own `QueryParams.get()` resolves for the SAME string -- the exact
# resolution FastAPI's route-level `stack` binding uses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query_string",
    [
        b"stack=wiki&stack=rag",
        b"stack=rag&stack=wiki",
        b"stack=wiki&stack=rag&stack=hybrid",
        b"stack=wiki",
        b"",
    ],
)
def test_stack_param_resolution_matches_starlette_route_level_parsing(query_string):
    import gateway.app.middleware as mw_mod

    scope = {"query_string": query_string}
    expected = QueryParams(query_string.decode("latin-1")).get("stack") or "rag"
    assert mw_mod._stack_param(scope) == expected, (
        "middleware._stack_param must resolve a repeated 'stack' param exactly "
        f"like the route's own Starlette-backed query parsing does for "
        f"query_string={query_string!r}"
    )


# ---------------------------------------------------------------------------
# `?stack=wiki&stack=rag` resolves to rag (last value) -- must 503 over cap,
# never a bypass admission that lets an uncounted rag/LLM dispatch through.
# ---------------------------------------------------------------------------


def test_duplicate_stack_wiki_then_rag_over_cap_is_503_not_bypass(exhausted_client):
    resp = exhausted_client.post("/chat/stream?stack=wiki&stack=rag", json={"query": "hi"})
    assert resp.status_code == 503, (
        f"a duplicate stack param resolving to rag must hard-503 over cap, "
        f"same as a plain stack=rag request; got {resp.status_code}: {resp.text}"
    )
    assert resp.json() == {"detail": "daily demo budget reached"}


def test_duplicate_stack_wiki_then_rag_over_cap_does_not_charge_budget(exhausted_client):
    import gateway.app.budget as budget_mod

    before = budget_mod.budget.day_total()
    exhausted_client.post("/chat/stream?stack=wiki&stack=rag", json={"query": "hi"})
    assert budget_mod.budget.day_total() == before, "a rejected (503) request must never be charged"


# ---------------------------------------------------------------------------
# Reverse order: `?stack=rag&stack=wiki` resolves to wiki (last value) --
# must be admitted degraded, and the actual SSE dispatch must agree
# (done.stack == "wiki", matching the middleware's scope flag).
# ---------------------------------------------------------------------------


def test_duplicate_stack_rag_then_wiki_over_cap_is_degraded_not_503(exhausted_client):
    resp = exhausted_client.post("/chat/stream?stack=rag&stack=wiki", json={"query": "hi"})
    assert resp.status_code == 200, (
        f"a duplicate stack param resolving to wiki must degrade, not 503; "
        f"got {resp.status_code}: {resp.text}"
    )
    events = _parse_sse_response(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    assert done["data"]["degraded"] is True
    assert done["data"]["stack"] == "wiki", (
        "the actual dispatched stack must match the stack the middleware "
        f"believed was degradable; got done.stack={done['data']['stack']!r}"
    )


def test_duplicate_stack_rag_then_wiki_over_cap_does_not_charge_budget(exhausted_client):
    import gateway.app.budget as budget_mod

    before = budget_mod.budget.day_total()
    exhausted_client.post("/chat/stream?stack=rag&stack=wiki", json={"query": "hi"})
    assert budget_mod.budget.day_total() == before, (
        "a degraded (no-LLM) admission must never be charged, duplicate params included"
    )
