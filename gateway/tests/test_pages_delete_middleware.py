"""Prod-middleware wiring for the C11 orphan-delete endpoint (issue #381, ADR-0025).

``DELETE /wiki/pages/{slug}`` mutates the live corpus and triggers a reindex
even though it calls no LLM (Confirmed Remediation, ADR-0024 Invariant), so —
like every sibling heavy endpoint (precedent: /wiki/qa/{slug}/refile, issue
#380) — it must be classified in ADMIN_PATHS and priced in _COST_ESTIMATES.
An unclassified path bypasses ALL five production guards (budget cap,
concurrency shed, kill-switch, quota->503 mapping), which on the public demo
deploy is an unauthenticated-delete-of-corpus-content vector.

Unlike its non-parameterized siblings this path is parameterized (the slug is
in the path), so the middleware canonicalises concrete ``/wiki/pages/<slug>``
paths to ``PAGES_DELETE_TEMPLATE`` before its exact-match classification; the
canonicalisation itself — including that it must NOT swallow the sibling
``/wiki/pages/reconcile`` / ``/wiki/pages/collision/...`` paths — is pinned
here too.
"""

import pytest

from gateway.app.middleware import PAGES_DELETE_TEMPLATE, _canonical_path

# ---------------------------------------------------------------------------
# Canonicalisation: concrete slug paths collapse to the template; sibling
# /pages/ endpoints never do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/pages/some-orphan-slug",
        "/wiki/pages/some-orphan-slug/",  # trailing slash normalised first
        "/wiki/pages/x",
    ],
)
def test_concrete_delete_paths_canonicalise_to_template(raw):
    assert _canonical_path(raw) == PAGES_DELETE_TEMPLATE


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/pages/reconcile",  # sibling C5 endpoint, must stay its own key
        "/wiki/pages/reconcile/apply",
        "/wiki/pages/collision/merge",  # sibling C4 endpoints, two segments
        "/wiki/pages/collision/merge/apply",
        "/wiki/pages/collision/differentiate",
        "/wiki/pages/collision/differentiate/apply",
        "/wiki/pages/",  # empty slug segment never reaches the handler (404)
        "/wiki/chat",  # existing exact-match paths pass through unchanged
    ],
)
def test_sibling_and_non_delete_paths_pass_through_unchanged(raw):
    assert _canonical_path(raw) != PAGES_DELETE_TEMPLATE


# ---------------------------------------------------------------------------
# Classification + budget: template key in ADMIN_PATHS and the estimate table
# ---------------------------------------------------------------------------


def test_delete_template_in_admin_paths():
    """The delete template is in ADMIN_PATHS — admin semaphore + kill-switch +
    budget gate (issue #381; refile precedent) even though it costs $0."""
    from gateway.app.middleware import ADMIN_PATHS

    assert PAGES_DELETE_TEMPLATE in ADMIN_PATHS, (
        f"{PAGES_DELETE_TEMPLATE} must be in ADMIN_PATHS or every concrete "
        "/wiki/pages/<slug> delete bypasses every prod guard (ADR-0025 / issue #381)"
    )


def test_delete_cost_estimate_is_explicit_zero():
    """No LLM call anywhere in the delete path (ADR-0024 Invariant) — priced
    at an explicit $0.00, not left to the un-tabulated default-heavy fallback."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert PAGES_DELETE_TEMPLATE in _COST_ESTIMATES, (
        f"{PAGES_DELETE_TEMPLATE} must have an explicit cost estimate (ADR-0025 / issue #381)"
    )
    assert estimate_cost(PAGES_DELETE_TEMPLATE) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end through the ASGI stack: kill-switch fires on a CONCRETE slug path
# (proves canonicalisation is wired into __call__, not just unit-true)
# ---------------------------------------------------------------------------


def test_delete_blocked_by_admin_token_gate(monkeypatch):
    """When KB_ADMIN_TOKEN is set, DELETE on a concrete slug path without a
    Bearer token → 401 from the middleware, before the handler (so no
    filesystem/wiki fixture is needed; the token is read at request time via
    os.getenv)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.delete("/wiki/pages/any-slug")
    assert resp.status_code == 401, (
        f"Expected 401 for /wiki/pages/any-slug when KB_ADMIN_TOKEN is set "
        f"and no Bearer; got {resp.status_code}"
    )
