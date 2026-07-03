"""Prod-middleware wiring for the tier-B S6 batch-promote endpoint (issue
#382, ADR-0023 Consequences).

``POST /wiki/qa/promote-batch`` calls no LLM (Direct Remediation), but it
still flips potentially many pages ``draft -> live`` and triggers one BM25
reindex — like ``DELETE /wiki/pages/{slug}`` (ADR-0025, issue #381), it must
be classified in ADMIN_PATHS and priced in _COST_ESTIMATES or it bypasses
every production guard (budget cap, concurrency shed, kill-switch). Unlike
the parameterized C9/C11 endpoints, this path carries no slug, so it needs
no ``_canonical_path`` template collapsing — it is an exact-match ADMIN_PATHS
member directly.
"""

import pytest

PROMOTE_BATCH_PATH = "/wiki/qa/promote-batch"


# ---------------------------------------------------------------------------
# Classification + budget
# ---------------------------------------------------------------------------


def test_promote_batch_in_admin_paths():
    """/wiki/qa/promote-batch is in ADMIN_PATHS — admin semaphore +
    kill-switch + budget gate (issue #382; PAGES_DELETE_TEMPLATE precedent)."""
    from gateway.app.middleware import ADMIN_PATHS

    assert PROMOTE_BATCH_PATH in ADMIN_PATHS, (
        f"{PROMOTE_BATCH_PATH} must be in ADMIN_PATHS or it bypasses every "
        "prod guard (ADR-0023 / issue #382)"
    )


def test_promote_batch_cost_estimate():
    """No LLM call anywhere on this path — priced at $0.00, explicit rather
    than left to the default-heavy fallback (mirrors PAGES_DELETE_TEMPLATE)."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert PROMOTE_BATCH_PATH in _COST_ESTIMATES, (
        f"{PROMOTE_BATCH_PATH} must have an explicit cost estimate (ADR-0023 / issue #382)"
    )
    assert estimate_cost(PROMOTE_BATCH_PATH) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end through the ASGI stack
# ---------------------------------------------------------------------------


def test_promote_batch_blocked_by_admin_token_gate(monkeypatch):
    """When KB_ADMIN_TOKEN is set, POST without Bearer → 401, before the
    handler runs (the token is read at request time via os.getenv)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(PROMOTE_BATCH_PATH, json={"slugs": []})
    assert resp.status_code == 401, (
        f"Expected 401 for {PROMOTE_BATCH_PATH} when KB_ADMIN_TOKEN is set "
        f"and no Bearer; got {resp.status_code}"
    )


def test_promote_batch_blocked_by_exhausted_budget(monkeypatch):
    """With the daily cap already reached, the batch endpoint gets the
    budget 503 before the handler runs — even at a $0.00 estimate, an
    ADMIN_PATHS member is still subject to the cap gate."""
    from fastapi.testclient import TestClient

    import gateway.app.middleware as mw_mod

    monkeypatch.setattr(mw_mod._budget.budget, "cap_usd", 0.0)

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(PROMOTE_BATCH_PATH, json={"slugs": []})
    assert resp.status_code == 503, (
        f"Expected budget 503 for {PROMOTE_BATCH_PATH} at cap; got {resp.status_code}"
    )
    assert resp.json() == {"detail": "daily demo budget reached"}
