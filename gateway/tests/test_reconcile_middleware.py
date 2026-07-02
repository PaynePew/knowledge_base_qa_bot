"""Prod-middleware wiring for the C5 reconcile endpoints (issue #376, ADR-0028).

Both reconcile endpoints call the LLM (generate: draft + grounding; apply:
grounding re-check) and /apply additionally mutates two wiki pages and triggers
a reindex, so — like every sibling heavy endpoint (precedent: /hybrid/index,
issue #348) — they must be classified in ADMIN_PATHS and priced in
_COST_ESTIMATES.  An unclassified path bypasses ALL five production guards
(budget cap, concurrency shed, kill-switch, quota->503 mapping), which on the
public demo deploy is an uncounted-LLM-spend / cost-DoS vector.
"""

import pytest

RECONCILE_PATHS = ("/wiki/pages/reconcile", "/wiki/pages/reconcile/apply")


# ---------------------------------------------------------------------------
# Classification: both endpoints are heavy admin paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", RECONCILE_PATHS)
def test_reconcile_paths_in_admin_paths(path):
    """Both reconcile endpoints are in ADMIN_PATHS — admin semaphore +
    kill-switch + budget gate (issue #376; /hybrid/index precedent)."""
    from gateway.app.middleware import ADMIN_PATHS

    assert path in ADMIN_PATHS, (
        f"{path} must be in ADMIN_PATHS or it bypasses every prod guard (ADR-0028 / issue #376)"
    )


# ---------------------------------------------------------------------------
# Budget: both endpoints carry an explicit conservative estimate
# ---------------------------------------------------------------------------


def test_reconcile_generate_cost_estimate():
    """/wiki/pages/reconcile is priced explicitly (draft over two pages'
    Source union + grounding check)."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert "/wiki/pages/reconcile" in _COST_ESTIMATES, (
        "/wiki/pages/reconcile must have an explicit cost estimate (ADR-0028 / issue #376)"
    )
    assert estimate_cost("/wiki/pages/reconcile") == pytest.approx(0.05)


def test_reconcile_apply_cost_estimate():
    """/wiki/pages/reconcile/apply is priced explicitly (grounding re-check
    on the submitted content)."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert "/wiki/pages/reconcile/apply" in _COST_ESTIMATES, (
        "/wiki/pages/reconcile/apply must have an explicit cost estimate (ADR-0028 / issue #376)"
    )
    assert estimate_cost("/wiki/pages/reconcile/apply") == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Kill-switch: with KB_ADMIN_TOKEN set, both endpoints demand the Bearer token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", RECONCILE_PATHS)
def test_reconcile_blocked_by_admin_token_gate(monkeypatch, path):
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
