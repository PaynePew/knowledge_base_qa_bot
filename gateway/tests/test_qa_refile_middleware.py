"""Prod-middleware wiring for the C9 re-file endpoint (issue #380, ADR-0026).

``POST /wiki/qa/{slug}/refile`` runs a full grounded answer round through the
chat pipeline (LLM draft + verify) and demotes a live page in place, so — like
the C5 reconcile (#376) and C4 collision (#378) endpoints — it must be
classified in ADMIN_PATHS and priced in _COST_ESTIMATES. An unclassified path
bypasses ALL five production guards (budget cap, concurrency shed,
kill-switch, quota->503 mapping), which on the public demo deploy is an
uncounted-LLM-spend / cost-DoS vector — and here also an unauthenticated
demote of every live Filed Answer.

Unlike its siblings this path is parameterized (the slug is in the path, per
ADR-0026's literal endpoint shape), so the middleware canonicalises concrete
``/wiki/qa/<slug>/refile`` paths to ``QA_REFILE_TEMPLATE`` before its
exact-match classification; the canonicalisation itself is pinned here too.
"""

import pytest

from gateway.app.middleware import QA_REFILE_TEMPLATE, _canonical_path

# ---------------------------------------------------------------------------
# Canonicalisation: concrete slug paths collapse to the template; nothing
# else does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/qa/some-slug/refile",
        "/wiki/qa/some-slug/refile/",  # trailing slash normalised first
        "/wiki/qa/x/refile",
    ],
)
def test_concrete_refile_paths_canonicalise_to_template(raw):
    assert _canonical_path(raw) == QA_REFILE_TEMPLATE


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/qa/some-slug/promote",  # sibling lifecycle endpoints stay exact-match
        "/wiki/qa/some-slug",  # PUT/DELETE qa page — LLM-free, unclassified
        "/wiki/qa//refile",  # empty slug segment never reaches the handler (404)
        "/wiki/qa/a/b/refile",  # slug is a single segment; two never routes
        "/wiki/qa/refile",  # no slug at all
        "/wiki/chat",  # existing exact-match paths pass through unchanged
    ],
)
def test_non_refile_paths_pass_through_unchanged(raw):
    assert _canonical_path(raw) != QA_REFILE_TEMPLATE


# ---------------------------------------------------------------------------
# Classification + budget: template key in ADMIN_PATHS and the estimate table
# ---------------------------------------------------------------------------


def test_refile_template_in_admin_paths():
    """The refile template is in ADMIN_PATHS — admin semaphore + kill-switch +
    budget gate (issue #380; reconcile/collision precedent)."""
    from gateway.app.middleware import ADMIN_PATHS

    assert QA_REFILE_TEMPLATE in ADMIN_PATHS, (
        f"{QA_REFILE_TEMPLATE} must be in ADMIN_PATHS or every concrete "
        "/wiki/qa/<slug>/refile bypasses every prod guard (ADR-0026 / issue #380)"
    )


def test_refile_cost_estimate():
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert QA_REFILE_TEMPLATE in _COST_ESTIMATES, (
        f"{QA_REFILE_TEMPLATE} must have an explicit cost estimate (ADR-0026 / issue #380)"
    )
    assert estimate_cost(QA_REFILE_TEMPLATE) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# End-to-end through the ASGI stack: kill-switch and budget cap both fire on a
# CONCRETE slug path (proves canonicalisation is wired into __call__, not just
# unit-true)
# ---------------------------------------------------------------------------


def test_refile_blocked_by_admin_token_gate(monkeypatch):
    """When KB_ADMIN_TOKEN is set, POST to a concrete slug path without a
    Bearer token → 401 from the middleware, before the handler (so no LLM
    stubbing is needed; the token is read at request time via os.getenv)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/wiki/qa/any-slug/refile")
    assert resp.status_code == 401, (
        f"Expected 401 for /wiki/qa/any-slug/refile when KB_ADMIN_TOKEN is set "
        f"and no Bearer; got {resp.status_code}"
    )


def test_refile_blocked_by_exhausted_budget(monkeypatch):
    """With the daily cap already reached, a concrete slug path gets the
    budget 503 before the handler runs (ADR-0026 Consequences: 're-file burns
    LLM tokens under the daily cap')."""
    from fastapi.testclient import TestClient

    import gateway.app.middleware as mw_mod

    monkeypatch.setattr(mw_mod._budget.budget, "cap_usd", 0.0)

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/wiki/qa/any-slug/refile")
    assert resp.status_code == 503, (
        f"Expected budget 503 for /wiki/qa/any-slug/refile at cap; got {resp.status_code}"
    )
    assert resp.json() == {"detail": "daily demo budget reached"}
