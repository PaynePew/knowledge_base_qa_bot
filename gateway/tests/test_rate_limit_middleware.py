"""ProdMiddleware per-IP rate-limit wiring (issue #598 Slice A / Q3).

Covers the middleware-level guard-order contract: token -> budget -> rate
limit -> semaphore (rate-limited requests never consume budget or a
concurrency slot). Unit-level RateLimiter/client_ip behaviour lives in
test_ratelimit.py; this file only exercises the wiring through the live ASGI
app, mirroring test_prod_middleware.py's fresh-app-per-test pattern.

All hermetic — no OPENAI_API_KEY, no real network. The sub-app heavy handlers
are monkeypatched away by the same TestClient setup test_prod_middleware.py
uses (no LLM call is ever reached — every request here is rejected by a
guard before the handler runs, or admitted and 200s against the stub app).
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _fresh_app():
    """Reload the gateway app so module-level middleware/ratelimit state is pristine."""
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
def client(monkeypatch):
    """Default-config client: default budget, default (30/5min) rate limit."""
    monkeypatch.delenv("KB_DAILY_USD_CAP", raising=False)
    monkeypatch.delenv("KB_READ_RESERVED_USD", raising=False)
    monkeypatch.delenv("KB_RATE_LIMIT_PER_IP", raising=False)
    monkeypatch.delenv("KB_MAX_INFLIGHT", raising=False)
    monkeypatch.delenv("KB_MAX_ADMIN", raising=False)
    return TestClient(_fresh_app())


# ---------------------------------------------------------------------------
# Basic admit / block
# ---------------------------------------------------------------------------


def test_read_path_under_limit_not_blocked(client):
    resp = client.post("/wiki/chat", json={"query": "hi"})
    assert resp.status_code != 429


def test_rate_limiter_lives_on_middleware_module_configured_from_ratelimit(monkeypatch):
    """The live singleton is owned by middleware.py (recreated on its reload —
    see ratelimit.py's trailing comment) but configured from ratelimit.py's
    env-derived constants."""
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "7")
    _fresh_app()
    import gateway.app.middleware as mw_mod
    import gateway.app.ratelimit as ratelimit_mod

    assert mw_mod.rate_limiter.limit == 7 == ratelimit_mod.RATE_LIMIT_PER_IP
    assert mw_mod.rate_limiter.window_sec == pytest.approx(ratelimit_mod.RATE_LIMIT_WINDOW_SEC)


def test_read_path_over_limit_returns_429(monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "1")
    client = TestClient(_fresh_app())
    first = client.post("/wiki/chat", json={"query": "hi"})
    assert first.status_code != 429
    second = client.post("/wiki/chat", json={"query": "hi"})
    assert second.status_code == 429
    assert second.json() == {"detail": "rate limited, please retry later"}


def test_admin_path_is_also_rate_limited(monkeypatch):
    """Heavy paths includes ADMIN_PATHS, not just READ_PATHS (issue #598 Q3)."""
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "1")
    client = TestClient(_fresh_app())
    first = client.post("/wiki/index")
    assert first.status_code != 429
    second = client.post("/wiki/index")
    assert second.status_code == 429


def test_non_heavy_path_bypasses_rate_limit(monkeypatch):
    """A non-heavy path is never gated, even once the caller's IP is over limit."""
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "1")
    client = TestClient(_fresh_app())
    client.post("/wiki/chat", json={"query": "hi"})  # consumes the one slot
    blocked = client.post("/wiki/chat", json={"query": "hi"})
    assert blocked.status_code == 429
    resp = client.get("/")
    assert resp.status_code == 200


def test_distinct_ips_via_x_forwarded_for_tracked_independently(monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "1")
    client = TestClient(_fresh_app())
    first = client.post("/wiki/chat", json={"query": "hi"}, headers={"X-Forwarded-For": "1.1.1.1"})
    assert first.status_code != 429
    other_ip = client.post(
        "/wiki/chat", json={"query": "hi"}, headers={"X-Forwarded-For": "2.2.2.2"}
    )
    assert other_ip.status_code != 429  # a different IP has its own window
    same_ip_again = client.post(
        "/wiki/chat", json={"query": "hi"}, headers={"X-Forwarded-For": "1.1.1.1"}
    )
    assert same_ip_again.status_code == 429


def test_rate_limit_zero_disables_the_gate(monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "0")
    client = TestClient(_fresh_app())
    for _ in range(10):
        resp = client.post("/wiki/chat", json={"query": "hi"})
        assert resp.status_code != 429


# ---------------------------------------------------------------------------
# Guard order: token -> budget -> rate limit -> semaphore
# ---------------------------------------------------------------------------


def test_budget_gate_short_circuits_before_rate_limit(monkeypatch):
    """An exhausted budget returns 503, not 429 — budget is checked first."""
    monkeypatch.setenv("KB_DAILY_USD_CAP", "0.0")
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "30")  # plenty of rate-limit headroom
    client = TestClient(_fresh_app())
    resp = client.post("/wiki/chat", json={"query": "hi"})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "daily demo budget reached"}


def test_rate_limit_gate_short_circuits_before_semaphore(monkeypatch):
    """A rate-limited request returns 429, not the semaphore's 503 — even
    while the semaphore is fully saturated — proving rate limit runs BEFORE
    the concurrency gate."""
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "1")
    client = TestClient(_fresh_app())
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.read_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        first = client.post("/wiki/chat", json={"query": "hi"})
        assert (
            first.status_code == 503
        )  # shed by the (drained) semaphore, consumes rate-limit slot 1/1
        second = client.post("/wiki/chat", json={"query": "hi"})
        assert (
            second.status_code == 429
        )  # rate limit trips before ever touching the semaphore again
    finally:
        for _ in acquired:
            mw_mod.read_sem.release()


def test_rate_limited_request_does_not_consume_budget(monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "1")
    client = TestClient(_fresh_app())
    import gateway.app.budget as budget_mod

    client.post("/wiki/chat", json={"query": "hi"})  # consumes the one rate-limit slot
    before = budget_mod.budget.day_total()
    blocked = client.post("/wiki/chat", json={"query": "hi"})
    assert blocked.status_code == 429
    assert budget_mod.budget.day_total() == before, "a rate-limited request must not be charged"
