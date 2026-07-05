"""Prod-middleware wiring for the remove-alias endpoint (issue #491, ADR-0030
extension).

``DELETE /wiki/pages/{slug}/aliases/{alias}`` mutates the live corpus (a
frontmatter field write) even though it calls no LLM (Direct-class
remove-alias), so — like its sibling ``POST /wiki/pages/{slug}/aliases``
(issue #409) and ``DELETE /wiki/pages/{slug}`` (issue #381) — it must be
classified in ADMIN_PATHS and priced in _COST_ESTIMATES. An unclassified
path bypasses ALL five production guards (budget cap, concurrency shed,
kill-switch, quota->503 mapping), which on the public demo deploy is an
unauthenticated-corpus-mutation vector (the #376 bug class).

This path has a FOURTH path segment (the alias) that its sibling
``/wiki/pages/{slug}/aliases`` (POST, three segments) does not, so the
middleware canonicalises concrete ``/wiki/pages/<slug>/aliases/<alias>``
paths to ``ALIAS_REMOVE_TEMPLATE`` before its exact-match classification —
mirroring ``test_pages_alias_middleware.py``'s own coverage for the sibling
assign-alias path.
"""

import pytest

from gateway.app.middleware import (
    ALIAS_ASSIGN_TEMPLATE,
    ALIAS_REMOVE_TEMPLATE,
    PAGES_DELETE_TEMPLATE,
    _canonical_path,
)

# ---------------------------------------------------------------------------
# Canonicalisation: concrete slug/alias paths collapse to the template;
# sibling /pages/ endpoints never do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/pages/some-page/aliases/PayPal",
        "/wiki/pages/some-page/aliases/PayPal/",  # trailing slash normalised first
        "/wiki/pages/x/aliases/y",
    ],
)
def test_concrete_alias_remove_paths_canonicalise_to_template(raw):
    assert _canonical_path(raw) == ALIAS_REMOVE_TEMPLATE


@pytest.mark.parametrize(
    "raw",
    [
        "/wiki/pages/some-page/aliases",  # sibling POST endpoint, 3 segments only
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
def test_sibling_and_non_alias_remove_paths_pass_through_unchanged(raw):
    assert _canonical_path(raw) != ALIAS_REMOVE_TEMPLATE


def test_alias_remove_never_collapses_onto_the_assign_or_delete_templates():
    """A 4-segment alias-remove path must classify as its OWN template, not
    accidentally fall through to the 3-segment assign or 2-segment delete
    template."""
    canonical = _canonical_path("/wiki/pages/some-page/aliases/PayPal")
    assert canonical != ALIAS_ASSIGN_TEMPLATE
    assert canonical != PAGES_DELETE_TEMPLATE


# ---------------------------------------------------------------------------
# Classification + budget: template key in ADMIN_PATHS and the estimate table
# ---------------------------------------------------------------------------


def test_alias_remove_template_in_admin_paths():
    """The alias-remove template is in ADMIN_PATHS — admin semaphore +
    kill-switch + budget gate (issue #491; assign-alias / delete-path
    precedent) even though it costs $0."""
    from gateway.app.middleware import ADMIN_PATHS

    assert ALIAS_REMOVE_TEMPLATE in ADMIN_PATHS, (
        f"{ALIAS_REMOVE_TEMPLATE} must be in ADMIN_PATHS or every concrete "
        "/wiki/pages/<slug>/aliases/<alias> removal bypasses every prod guard "
        "(ADR-0030 extension / issue #491)"
    )


def test_alias_remove_cost_estimate_is_explicit_zero():
    """No LLM call anywhere in the remove-alias path — priced at an explicit
    $0.00, not left to the un-tabulated default-heavy fallback."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert ALIAS_REMOVE_TEMPLATE in _COST_ESTIMATES, (
        f"{ALIAS_REMOVE_TEMPLATE} must have an explicit cost estimate (ADR-0030 extension / issue #491)"
    )
    assert estimate_cost(ALIAS_REMOVE_TEMPLATE) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end through the ASGI stack: kill-switch fires on a CONCRETE
# slug/alias path (proves canonicalisation is wired into __call__, not just
# unit-true)
# ---------------------------------------------------------------------------


def test_alias_remove_blocked_by_admin_token_gate(monkeypatch):
    """When KB_ADMIN_TOKEN is set, DELETE on a concrete slug/alias path
    without a Bearer token → 401 from the middleware, before the handler (so
    no filesystem/wiki fixture is needed; the token is read at request time
    via os.getenv)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.delete("/wiki/pages/any-slug/aliases/PayPal")
    assert resp.status_code == 401, (
        f"Expected 401 for /wiki/pages/any-slug/aliases/PayPal when KB_ADMIN_TOKEN is set "
        f"and no Bearer; got {resp.status_code}"
    )
