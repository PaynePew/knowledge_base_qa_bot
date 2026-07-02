"""Prod-middleware wiring for the C4 collision endpoints (issue #378, ADR-0028).

All four endpoints call the LLM (generate: draft + grounding; apply:
grounding re-check) and the /apply endpoints additionally mutate wiki pages
and trigger a reindex, so — like the sibling C5 reconcile endpoints
(precedent: issue #376) — they must be classified in ADMIN_PATHS and priced
in _COST_ESTIMATES. An unclassified path bypasses ALL five production guards
(budget cap, concurrency shed, kill-switch, quota->503 mapping), which on the
public demo deploy is an uncounted-LLM-spend / cost-DoS vector.
"""

import pytest

COLLISION_PATHS = (
    "/wiki/pages/collision/merge",
    "/wiki/pages/collision/merge/apply",
    "/wiki/pages/collision/differentiate",
    "/wiki/pages/collision/differentiate/apply",
)


# ---------------------------------------------------------------------------
# Classification: all four endpoints are heavy admin paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", COLLISION_PATHS)
def test_collision_paths_in_admin_paths(path):
    """Every collision endpoint is in ADMIN_PATHS — admin semaphore +
    kill-switch + budget gate (issue #378; /wiki/pages/reconcile precedent)."""
    from gateway.app.middleware import ADMIN_PATHS

    assert path in ADMIN_PATHS, (
        f"{path} must be in ADMIN_PATHS or it bypasses every prod guard (ADR-0028 / issue #378)"
    )


# ---------------------------------------------------------------------------
# Budget: all four endpoints carry an explicit conservative estimate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/wiki/pages/collision/merge", 0.05),
        ("/wiki/pages/collision/merge/apply", 0.02),
        ("/wiki/pages/collision/differentiate", 0.08),
        ("/wiki/pages/collision/differentiate/apply", 0.03),
    ],
)
def test_collision_cost_estimates(path, expected):
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert path in _COST_ESTIMATES, (
        f"{path} must have an explicit cost estimate (ADR-0028 / issue #378)"
    )
    assert estimate_cost(path) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Kill-switch: with KB_ADMIN_TOKEN set, every endpoint demands the Bearer token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", COLLISION_PATHS)
def test_collision_blocked_by_admin_token_gate(monkeypatch, path):
    """When KB_ADMIN_TOKEN is set, POST without Bearer → 401.

    The middleware rejects before the handler runs, so no LLM stubbing is
    needed; the token is read at request time (os.getenv), so
    monkeypatch.setenv works without reloading the app.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(path, json={})
    assert resp.status_code == 401, (
        f"Expected 401 for {path} when KB_ADMIN_TOKEN is set and no Bearer; got {resp.status_code}"
    )
