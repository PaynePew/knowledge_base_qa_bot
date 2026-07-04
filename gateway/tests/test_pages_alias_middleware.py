"""Prod-middleware wiring for the assign-alias endpoint (issue #409, ADR-0030
decision 3).

``POST /wiki/pages/{slug}/aliases`` mutates the live corpus (a frontmatter
field write) even though it calls no LLM (Direct-class assign-alias,
ADR-0030 Invariant), so — like every sibling heavy endpoint (precedent:
``DELETE /wiki/pages/{slug}``, issue #381) — it must be classified in
ADMIN_PATHS and priced in _COST_ESTIMATES. An unclassified path bypasses ALL
five production guards (budget cap, concurrency shed, kill-switch,
quota->503 mapping), which on the public demo deploy is an
unauthenticated-corpus-mutation vector.

This path is parameterized (the slug is in the path) AND has a third path
segment ("/aliases") that its sibling ``/wiki/pages/{slug}`` (DELETE) does
not, so the middleware canonicalises concrete
``/wiki/pages/<slug>/aliases`` paths to ``ALIAS_ASSIGN_TEMPLATE`` before its
exact-match classification; the canonicalisation itself — including that it
must NOT swallow the sibling ``/wiki/pages/{slug}`` (2-segment DELETE) or
``/wiki/pages/reconcile`` / ``/wiki/pages/collision/...`` paths — is pinned
here too.
"""

import pytest

from gateway.app.middleware import ALIAS_ASSIGN_TEMPLATE, PAGES_DELETE_TEMPLATE, _canonical_path

# ---------------------------------------------------------------------------
# Canonicalisation: concrete slug paths collapse to the template; sibling
# /pages/ endpoints never do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/pages/some-page/aliases",
        "/wiki/pages/some-page/aliases/",  # trailing slash normalised first
        "/wiki/pages/x/aliases",
    ],
)
def test_concrete_alias_paths_canonicalise_to_template(raw):
    assert _canonical_path(raw) == ALIAS_ASSIGN_TEMPLATE


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/pages/some-page",  # sibling DELETE endpoint, 2 segments only
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
def test_sibling_and_non_alias_paths_pass_through_unchanged(raw):
    assert _canonical_path(raw) != ALIAS_ASSIGN_TEMPLATE


def test_alias_assign_never_collapses_onto_the_delete_template():
    """A 3-segment alias-assign path must classify as its OWN template, not
    accidentally fall through to the 2-segment delete template."""
    assert _canonical_path("/wiki/pages/some-page/aliases") != PAGES_DELETE_TEMPLATE


# ---------------------------------------------------------------------------
# Classification + budget: template key in ADMIN_PATHS and the estimate table
# ---------------------------------------------------------------------------


def test_alias_assign_template_in_admin_paths():
    """The alias-assign template is in ADMIN_PATHS — admin semaphore +
    kill-switch + budget gate (issue #409; delete-path precedent) even
    though it costs $0."""
    from gateway.app.middleware import ADMIN_PATHS

    assert ALIAS_ASSIGN_TEMPLATE in ADMIN_PATHS, (
        f"{ALIAS_ASSIGN_TEMPLATE} must be in ADMIN_PATHS or every concrete "
        "/wiki/pages/<slug>/aliases assign bypasses every prod guard (ADR-0030 / issue #409)"
    )


def test_alias_assign_cost_estimate_is_explicit_zero():
    """No LLM call anywhere in the assign-alias path (ADR-0030 Invariant) —
    priced at an explicit $0.00, not left to the un-tabulated default-heavy
    fallback."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert ALIAS_ASSIGN_TEMPLATE in _COST_ESTIMATES, (
        f"{ALIAS_ASSIGN_TEMPLATE} must have an explicit cost estimate (ADR-0030 / issue #409)"
    )
    assert estimate_cost(ALIAS_ASSIGN_TEMPLATE) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end through the ASGI stack: kill-switch fires on a CONCRETE slug path
# (proves canonicalisation is wired into __call__, not just unit-true)
# ---------------------------------------------------------------------------


def test_alias_assign_blocked_by_admin_token_gate(monkeypatch):
    """When KB_ADMIN_TOKEN is set, POST on a concrete slug path without a
    Bearer token → 401 from the middleware, before the handler (so no
    filesystem/wiki fixture is needed; the token is read at request time via
    os.getenv)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/wiki/pages/any-slug/aliases", json={"alias": "x"})
    assert resp.status_code == 401, (
        f"Expected 401 for /wiki/pages/any-slug/aliases when KB_ADMIN_TOKEN is set "
        f"and no Bearer; got {resp.status_code}"
    )
