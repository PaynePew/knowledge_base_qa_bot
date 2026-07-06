"""Prod-middleware wiring for the async Import job endpoints (issue #497).

``POST /wiki/import/jobs`` starts the SAME billed work as ``/wiki/import``
(mechanical conversion flat floor + Transcribe's per-page hook on top when a
scan auto-routes), just asynchronously — so it must be classified in
ADMIN_PATHS and priced in _COST_ESTIMATES, exactly as ``POST
/wiki/transcribe/batch`` was classified alongside ``/wiki/transcribe``
(issue #447's precedent; an unclassified path bypasses ALL five production
guards — the #376 bug class).

Its poll sibling ``GET /wiki/import/jobs/{job_id}`` is deliberately left OUT
of both sets: a read-only, in-memory job-status lookup with no LLM call and
no corpus mutation — the same rationale that leaves ``GET
/wiki/transcribe/jobs/{job_id}`` unclassified.
"""

import pytest

from gateway.app.middleware import ADMIN_PATHS, READ_PATHS, _canonical_path

# ---------------------------------------------------------------------------
# Classification + budget: submit gated, poll deliberately not
# ---------------------------------------------------------------------------


def test_import_jobs_submit_in_admin_paths():
    assert "/wiki/import/jobs" in ADMIN_PATHS, (
        "/wiki/import/jobs must be in ADMIN_PATHS or the async submit starts "
        "the same billed Import work with every prod guard bypassed (issue #497)"
    )


def test_import_jobs_submit_priced_like_sync_import():
    """Same surface, same flat floor as /wiki/import — the per-page Transcribe
    hook still charges on top when a scan auto-routes."""
    from gateway.app.budget import _COST_ESTIMATES, estimate_cost

    assert "/wiki/import/jobs" in _COST_ESTIMATES, (
        "/wiki/import/jobs must have an explicit cost estimate (issue #497)"
    )
    assert estimate_cost("/wiki/import/jobs") == pytest.approx(estimate_cost("/wiki/import"))


def test_import_jobs_poll_is_deliberately_unclassified():
    """The poll path stays out of BOTH sets (read-only in-memory lookup) —
    same rationale as GET /wiki/transcribe/jobs/{job_id}. The concrete path
    passes through canonicalisation unchanged, so an exact-match set can
    never see it."""
    concrete = "/wiki/import/jobs/0123456789abcdef"
    assert _canonical_path(concrete) == concrete
    assert concrete not in ADMIN_PATHS
    assert concrete not in READ_PATHS


def test_concrete_poll_path_never_collapses_onto_the_submit_key():
    """Trailing-slash normalisation must not turn a poll GET into the
    classified submit key, nor vice versa."""
    assert _canonical_path("/wiki/import/jobs/") == "/wiki/import/jobs"
    assert _canonical_path("/wiki/import/jobs/abc123") != "/wiki/import/jobs"


# ---------------------------------------------------------------------------
# End-to-end through the ASGI stack: kill-switch fires on the submit path
# ---------------------------------------------------------------------------


def test_import_jobs_submit_blocked_by_admin_token_gate(monkeypatch):
    """When KB_ADMIN_TOKEN is set, POST /wiki/import/jobs without a Bearer
    token → 401 from the middleware, before the handler (no job is ever
    scheduled)."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/wiki/import/jobs")
    assert resp.status_code == 401, (
        f"Expected 401 for POST /wiki/import/jobs when KB_ADMIN_TOKEN is set "
        f"and no Bearer; got {resp.status_code}"
    )
