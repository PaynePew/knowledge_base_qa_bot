"""Admin-path coverage audit (issue #583).

An unclassified mutating route bypasses ALL of ``ProdMiddleware``'s guards —
the admin-token kill-switch, the admin concurrency semaphore, AND the daily
budget cap (the "#376 bug class", already hit twice before: issue #459's
``/wiki/transcribe/batch`` and issue #497's ``/wiki/import/jobs``). Both of
those were caught by hand, after the fact. This module instead enumerates
the routes DYNAMICALLY off the live composed Gateway app, so any future
mutating route added without a matching ``ADMIN_PATHS`` (or documented
exemption) entry fails CI immediately instead of shipping ungated.

Running this audit against pre-#583 code fails with exactly the gap the
issue reported: ``POST /wiki/qa/{slug}/promote``, ``POST
/wiki/qa/{slug}/demote``, ``DELETE /wiki/qa/{slug}``, and ``PUT
/wiki/qa/{slug}`` were all mounted (Phase 6 / ADR-0012 / ADR-0026 decision 2
/ ADR-0037) but absent from ``ADMIN_PATHS`` — reachable with no token, no
concurrency cap, and no budget charge even when ``KB_ADMIN_TOKEN`` was set.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import gateway.app.main as main_mod
import gateway.app.middleware as mw_mod

MUTATING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

# Mounted mutating paths deliberately absent from ADMIN_PATHS / READ_PATHS,
# each with the reason a human already decided is sufficient (mirrors
# gateway/app/middleware.py's own end-of-ADMIN_PATHS exemption comment).
EXEMPT_MUTATING_PATHS: frozenset[str] = frozenset(
    {
        # Reader Feedback is opinion data ABOUT the corpus that never enters
        # it (CONTEXT.md "Reader Feedback") — stays a public, ungated surface
        # with its own payload/store-size caps (gateway/app/feedback.py).
        "/feedback",
    }
)


def _mounted_mutating_paths() -> set[str]:
    """Every mounted route+method pair with a mutating HTTP verb, as a set of
    full template-shaped paths (``/wiki/qa/{slug}/promote``, not a concrete
    slug) — the same shape ``ADMIN_PATHS``/``READ_PATHS`` and the middleware's
    own ``_canonical_path`` templates use, so no further canonicalisation is
    needed here.

    Walks ``app.routes`` and recurses into any ``Mount`` (Starlette's
    ``app.mount("/wiki", sub_app)`` / ``app.mount("/rag", sub_app)``) that
    itself carries `.routes` — a plain ASGI mount with no ``.routes``
    attribute (``StaticFiles`` at ``/static``) is skipped, since it only ever
    serves GET/HEAD.
    """
    paths: set[str] = set()

    def _walk(routes, prefix: str) -> None:
        for route in routes:
            sub_routes = getattr(route, "routes", None)
            if sub_routes is not None:
                mount_path = getattr(route, "path", "")
                _walk(sub_routes, prefix + mount_path)
                continue
            methods = getattr(route, "methods", None) or set()
            if methods & MUTATING_METHODS:
                route_path = getattr(route, "path", "")
                paths.add(prefix + route_path)

    _walk(main_mod.app.routes, "")
    return paths


def test_every_mutating_route_is_classified_or_exempt():
    """No mounted POST/PUT/DELETE/PATCH route falls through ADMIN_PATHS,
    READ_PATHS, and the documented exemption set — regression guard for the
    #376 bug class (issue #583)."""
    mutating = _mounted_mutating_paths()
    classified = mw_mod.ADMIN_PATHS | mw_mod.READ_PATHS | EXEMPT_MUTATING_PATHS
    unclassified = mutating - classified
    assert not unclassified, (
        "mutating route(s) mounted but not classified in ADMIN_PATHS, "
        "READ_PATHS, or EXEMPT_MUTATING_PATHS (see gateway/app/middleware.py "
        f"and this file): {sorted(unclassified)}"
    )


def test_audit_actually_sees_mutating_routes():
    """Guards the audit itself against a silently-empty enumeration (a bug in
    ``_mounted_mutating_paths`` that returned nothing would make the test
    above vacuously pass)."""
    mutating = _mounted_mutating_paths()
    assert "/upload" in mutating
    assert "/wiki/qa/{slug}/promote" in mutating
    assert "/wiki/qa/{slug}/demote" in mutating
    assert "/wiki/qa/{slug}" in mutating


# ---------------------------------------------------------------------------
# Regression: the four specific routes the audit found unclassified are now
# admin-gated end to end (kill-switch 401, not just present in ADMIN_PATHS).
# ---------------------------------------------------------------------------


def test_qa_promote_is_admin_gated():
    assert "/wiki/qa/{slug}/promote" in mw_mod.ADMIN_PATHS


def test_qa_demote_is_admin_gated():
    assert "/wiki/qa/{slug}/demote" in mw_mod.ADMIN_PATHS


def test_qa_item_template_is_admin_gated():
    assert "/wiki/qa/{slug}" in mw_mod.ADMIN_PATHS


def test_qa_promote_requires_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(main_mod.app)
    resp = client.post("/wiki/qa/some-slug/promote")
    assert resp.status_code == 401


def test_qa_demote_requires_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(main_mod.app)
    resp = client.post("/wiki/qa/some-slug/demote")
    assert resp.status_code == 401


def test_qa_delete_requires_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(main_mod.app)
    resp = client.delete("/wiki/qa/some-slug")
    assert resp.status_code == 401


def test_qa_edit_requires_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(main_mod.app)
    resp = client.put("/wiki/qa/some-slug", json={"question": "q", "body": "b"})
    assert resp.status_code == 401


def test_qa_promote_open_when_token_unset(monkeypatch):
    """Unset KB_ADMIN_TOKEN (the demo default) — never 401, current behaviour
    preserved (may 404/500 depending on sub-app state, but not 401)."""
    monkeypatch.delenv("KB_ADMIN_TOKEN", raising=False)
    client = TestClient(main_mod.app)
    resp = client.post("/wiki/qa/nonexistent-slug/promote")
    assert resp.status_code != 401


def test_canonical_path_collapses_promote_demote_and_item():
    """`_canonical_path` maps concrete per-slug paths to the stable template
    keys ADMIN_PATHS/the budget table use — and never confuses the fixed
    literal `/wiki/qa/promote-batch` route for a slug named "promote-batch"."""
    assert mw_mod._canonical_path("/wiki/qa/refund-policy/promote") == mw_mod.QA_PROMOTE_TEMPLATE
    assert mw_mod._canonical_path("/wiki/qa/refund-policy/demote") == mw_mod.QA_DEMOTE_TEMPLATE
    assert mw_mod._canonical_path("/wiki/qa/refund-policy") == mw_mod.QA_ITEM_TEMPLATE
    assert mw_mod._canonical_path("/wiki/qa/promote-batch") == "/wiki/qa/promote-batch"
