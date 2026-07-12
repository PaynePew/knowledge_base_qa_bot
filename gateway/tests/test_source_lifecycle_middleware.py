"""Prod-middleware wiring for the Source lifecycle mutation endpoints (issue
#604, ADR-0041).

``POST /sources/retire`` / ``POST /sources/restore`` mutate the live corpus
(a Source file move) even though neither calls an LLM (Confirmed retire /
Direct restore), so — like every sibling heavy endpoint (precedent: ``DELETE
/wiki/pages/{slug}``, issue #381) — they must be classified in ADMIN_PATHS
and priced in ``_COST_ESTIMATES``. An unclassified path bypasses ALL five
production guards (budget cap, concurrency shed, kill-switch, quota->503
mapping), which on the public demo deploy is an unauthenticated-mutation
vector. Neither path is parameterized (relpath/timestamp travel in the
request body), so — unlike ``PAGES_DELETE_TEMPLATE`` — no canonicalisation
is needed; this file pins the plain string membership + an end-to-end
kill-switch check instead.

The read-only siblings (``GET /sources/{relpath}/impact``, ``GET
/sources/trash``) are deliberately UNCLASSIFIED (mirrors ``GET /read/*`` /
``GET /pages/resolution-map``) — pinned here too so a future change cannot
silently admin-gate a pure read.
"""

from __future__ import annotations

import pytest

from gateway.app.middleware import ADMIN_PATHS, READ_PATHS

# ---------------------------------------------------------------------------
# Classification + budget: both mutation endpoints in ADMIN_PATHS and priced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/sources/retire", "/sources/restore"])
def test_mutation_endpoint_in_admin_paths(path):
    assert path in ADMIN_PATHS, (
        f"{path} must be in ADMIN_PATHS or it bypasses every prod guard (ADR-0041 / issue #604)"
    )


@pytest.mark.parametrize("path", ["/sources/retire", "/sources/restore"])
def test_mutation_endpoint_cost_estimate_is_explicit_zero(path):
    """No LLM call anywhere in retire/restore — priced at an explicit $0.00,
    not left to the un-tabulated default-heavy fallback."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert path in _COST_ESTIMATES, f"{path} must have an explicit cost estimate (issue #604)"
    assert estimate_cost(path) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Read-only siblings stay unclassified (a pure read must never be gated)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/sources/trash", "/sources/x.md/impact"])
def test_read_only_endpoints_are_unclassified(path):
    assert path not in ADMIN_PATHS
    assert path not in READ_PATHS


# ---------------------------------------------------------------------------
# End-to-end through the ASGI stack: kill-switch fires on both mutation paths
# (proves classification is wired into __call__, not just unit-true)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/sources/retire", "/sources/restore"])
def test_mutation_endpoint_blocked_by_admin_token_gate(monkeypatch, path):
    """When KB_ADMIN_TOKEN is set, POSTing without a Bearer token -> 401 from
    the middleware, before the handler (so no filesystem fixture is needed;
    the token is read at request time via os.getenv)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(path, json={"relpath": "anything.md"})
    assert resp.status_code == 401, (
        f"Expected 401 for {path} without a Bearer token; got {resp.status_code}"
    )
