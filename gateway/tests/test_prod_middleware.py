"""Gateway production overload + cost-protection middleware tests (issue #269).

Covers the six acceptance criteria for the VPS deploy hardening:

1. GET /healthz always 200 (liveness) — even when budget exhausted / shed active.
2. GET /healthz/shed 200 normally, 503 when the READ semaphore is saturated.
3. Two concurrency semaphores (read / admin) → 503 over-limit, full-path match.
4. Daily USD budget guard → 503 with {"detail":"daily demo budget reached"} at cap.
5. Graceful provider failure: OpenAI insufficient_quota / 429 → friendly 503.
6. Optional KB_ADMIN_TOKEN kill-switch: when set, admin paths need Bearer; off by default.

All hermetic — no OPENAI_API_KEY, no real network. The sub-app heavy handlers are
monkeypatched so the middleware behaviour is exercised in isolation.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fresh app per test — the middleware holds in-process counters / semaphores,
# so each test reloads the modules to start from a clean slate (and so env
# vars set via monkeypatch are read at construction time).
# ---------------------------------------------------------------------------


def _fresh_app():
    """Reload the gateway app so module-level middleware state is pristine."""
    import gateway.app.budget as budget_mod
    import gateway.app.main as main_mod
    import gateway.app.middleware as mw_mod

    importlib.reload(budget_mod)
    importlib.reload(mw_mod)
    importlib.reload(main_mod)
    return main_mod.app


@pytest.fixture()
def client(monkeypatch):
    """Default-config client: token OFF, default caps, default budget."""
    monkeypatch.delenv("KB_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("KB_MAX_INFLIGHT", raising=False)
    monkeypatch.delenv("KB_MAX_ADMIN", raising=False)
    monkeypatch.delenv("KB_DAILY_USD_CAP", raising=False)
    return TestClient(_fresh_app())


# ---------------------------------------------------------------------------
# AC1: GET /healthz always 200
# ---------------------------------------------------------------------------


def test_healthz_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_stays_200_when_budget_exhausted(monkeypatch):
    """Liveness must not flip even when the daily budget is exhausted."""
    monkeypatch.setenv("KB_DAILY_USD_CAP", "0.0")  # exhausted from the first request
    client = TestClient(_fresh_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_stays_200_when_shed_active(client):
    """Liveness must not flip even when the read semaphore is fully saturated."""
    import gateway.app.middleware as mw_mod

    # Drain the read semaphore to simulate saturation.
    while mw_mod.read_sem.acquire(blocking=False):
        pass
    try:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
    finally:
        importlib.reload(mw_mod)  # restore for later tests


# ---------------------------------------------------------------------------
# AC2: GET /healthz/shed reflects ONLY read-semaphore saturation
# ---------------------------------------------------------------------------


def test_shed_ok_normally(client):
    resp = client.get("/healthz/shed")
    assert resp.status_code == 200


def test_shed_503_when_read_sem_saturated(client):
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.read_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.get("/healthz/shed")
        assert resp.status_code == 503
    finally:
        for _ in acquired:
            mw_mod.read_sem.release()


def test_shed_not_flipped_by_admin_saturation(client):
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.admin_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.get("/healthz/shed")
        assert resp.status_code == 200, "admin saturation must NOT flip shed"
    finally:
        for _ in acquired:
            mw_mod.admin_sem.release()


# ---------------------------------------------------------------------------
# AC3: concurrency caps → 503 over-limit, full mounted-path match
# ---------------------------------------------------------------------------


def test_read_path_over_limit_returns_503(client):
    """A read heavy path returns 503 once the read semaphore is fully held."""
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.read_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.post("/wiki/chat", json={"query": "hi"})
        assert resp.status_code == 503
    finally:
        for _ in acquired:
            mw_mod.read_sem.release()


def test_admin_path_over_limit_returns_503(client):
    """An admin heavy path returns 503 once the admin semaphore is fully held."""
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.admin_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.post("/wiki/index")
        assert resp.status_code == 503
    finally:
        for _ in acquired:
            mw_mod.admin_sem.release()


def test_trailing_slash_is_stripped_for_match(client):
    """`/wiki/chat/` matches the read set the same as `/wiki/chat`."""
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.read_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.post("/wiki/chat/", json={"query": "hi"})
        assert resp.status_code == 503
    finally:
        for _ in acquired:
            mw_mod.read_sem.release()


def test_non_heavy_path_not_gated_by_read_sem(client):
    """A non-heavy path (static UI) is unaffected by read-sem saturation."""
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.read_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.get("/")
        assert resp.status_code == 200
    finally:
        for _ in acquired:
            mw_mod.read_sem.release()


# ---------------------------------------------------------------------------
# AC4: daily USD budget guard
# ---------------------------------------------------------------------------


def test_budget_blocks_heavy_path_at_cap(monkeypatch):
    monkeypatch.setenv("KB_DAILY_USD_CAP", "0.0")
    client = TestClient(_fresh_app())
    resp = client.post("/wiki/chat", json={"query": "hi"})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "daily demo budget reached"}


def test_budget_does_not_block_non_heavy_path(monkeypatch):
    monkeypatch.setenv("KB_DAILY_USD_CAP", "0.0")
    client = TestClient(_fresh_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_shed_request_does_not_consume_budget(client):
    """A request shed by the concurrency cap must NOT consume daily budget.

    Regression (issue #269 review): charging before the semaphore gate let a
    burst of shed traffic drain the $/day ceiling on zero real spend and then
    503 every heavy path for the rest of the UTC day. Budget is now charged only
    AFTER a request is admitted past the concurrency gate.
    """
    import gateway.app.budget as budget_mod
    import gateway.app.middleware as mw_mod

    before = budget_mod.budget.day_total()
    acquired = []
    while mw_mod.read_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.post("/wiki/chat", json={"query": "hi"})
        assert resp.status_code == 503  # shed by the read semaphore
    finally:
        for _ in acquired:
            mw_mod.read_sem.release()
    assert budget_mod.budget.day_total() == before, "a shed request must not be charged"


def test_budget_accumulates_then_blocks(monkeypatch):
    """A tiny cap allows the first heavy charge, then blocks once at the cap."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    # Cap just above one chat estimate so the first charge stays under, the
    # second charge crosses the cap (over_cap → the next request is rejected).
    estimate = budget_mod.estimate_cost("/wiki/chat")
    b = budget_mod.DailyBudget(cap_usd=estimate * 1.5)

    b.charge("/wiki/chat")
    assert b.over_cap() is False
    b.charge("/wiki/chat")
    assert b.over_cap() is True


def test_budget_resets_on_new_utc_day(monkeypatch):
    """The accumulator is keyed by UTC day; a day rollover resets the total."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    b = budget_mod.DailyBudget(cap_usd=10.0)
    b.charge("/wiki/chat", day="2026-06-14")
    first_day_total = b.day_total(day="2026-06-14")
    assert first_day_total > 0
    # A different UTC day starts fresh.
    assert b.day_total(day="2026-06-15") == 0.0


def test_estimate_overestimates_admin_reindex(monkeypatch):
    """Re-embedding the whole corpus must cost more than a single chat answer."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    assert budget_mod.estimate_cost("/rag/index") > budget_mod.estimate_cost("/wiki/chat")


# ---------------------------------------------------------------------------
# Issue #460: Transcribe per-page budget charging
# ---------------------------------------------------------------------------


def test_transcribe_path_is_admin_gated():
    """POST /wiki/transcribe is now classified as an admin heavy path.

    Regression guard: before issue #460 this path was absent from
    ADMIN_PATHS entirely, so it bypassed every guard (admin token,
    concurrency, budget) — the same "#376 bug class" the middleware
    docstring warns about, just for Transcribe's forced entry.
    """
    import gateway.app.middleware as mw_mod

    assert "/wiki/transcribe" in mw_mod.ADMIN_PATHS


def test_transcribe_path_over_admin_limit_returns_503(client):
    """The admin semaphore also shields /wiki/transcribe once fully held."""
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.admin_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.post("/wiki/transcribe", json={"source": "x.pdf"})
        assert resp.status_code == 503
    finally:
        for _ in acquired:
            mw_mod.admin_sem.release()


def test_transcribe_path_requires_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(_fresh_app())
    resp = client.post("/wiki/transcribe", json={"source": "x.pdf"})
    assert resp.status_code == 401


def test_transcribe_estimate_is_zero_true_cost_is_per_page(monkeypatch):
    """No flat admission estimate for /wiki/transcribe — its real cost is
    charged per-page by the transcriber hook, so a flat estimate on top
    would double-count."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    assert budget_mod.estimate_cost("/wiki/transcribe") == 0.0


def test_charge_pages_accumulates_page_count_times_rate(monkeypatch):
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    b = budget_mod.DailyBudget(cap_usd=10.0)
    total = b.charge_pages(63)
    assert total == pytest.approx(63 * budget_mod.TRANSCRIBE_PAGE_USD)


def test_charge_pages_shares_the_same_ledger_as_charge(monkeypatch):
    """A flat charge() and a per-page charge_pages() accumulate into one total."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    b = budget_mod.DailyBudget(cap_usd=10.0)
    b.charge("/wiki/import")
    b.charge_pages(10)
    expected = budget_mod.estimate_cost("/wiki/import") + 10 * budget_mod.TRANSCRIBE_PAGE_USD
    assert b.day_total() == pytest.approx(expected)


def test_main_registers_transcribe_budget_hook(monkeypatch):
    """The Gateway composition root wires a page-budget hook into markdown_kb.app.transcriber."""
    _fresh_app()
    from markdown_kb.app import transcriber as transcriber_mod

    assert transcriber_mod.get_page_budget_hook() is not None


def test_transcribe_budget_hook_charges_ledger_then_trips_cap(monkeypatch):
    """The wired hook charges the shared ledger per page and rejects once over cap.

    Proves the multi-scan batch scenario from the issue: a batch charges
    proportionally to pages transcribed (not a flat $0.10), and once the
    running total reaches the cap the NEXT file is rejected before any
    vision-model call for it.
    """
    monkeypatch.setenv("KB_DAILY_USD_CAP", "0.5")
    _fresh_app()

    from markdown_kb.app import transcriber as transcriber_mod

    import gateway.app.budget as budget_mod

    hook = transcriber_mod.get_page_budget_hook()
    assert hook is not None

    before = budget_mod.budget.day_total()
    # 63-page scan at the recalibrated default $0.005/page (issue #510) =
    # $0.315 — under the $0.50 cap.
    hook(63)
    after_one = budget_mod.budget.day_total()
    assert after_one == pytest.approx(before + 63 * budget_mod.TRANSCRIBE_PAGE_USD)
    assert budget_mod.budget.over_cap() is False

    # A second 63-page scan pushes the running total to $0.63 — over the cap.
    hook(63)
    assert budget_mod.budget.over_cap() is True

    # A third file must be rejected BEFORE its pages are charged/billed.
    total_before_third = budget_mod.budget.day_total()
    with pytest.raises(transcriber_mod.TranscribeBudgetExceeded):
        hook(63)
    assert budget_mod.budget.day_total() == total_before_third, (
        "a rejected file must not be charged"
    )


# ---------------------------------------------------------------------------
# Issue #447: /wiki/transcribe/batch — the async submit path was absent from
# ADMIN_PATHS entirely (issue #459 added the route after #460 classified only
# the synchronous /wiki/transcribe), so it bypassed the admin token,
# concurrency cap, and budget-cap check — the same "#376 bug class" the
# middleware docstring warns about. GET /wiki/transcribe/jobs/{job_id} and
# GET /wiki/transcribe/page-count are deliberately NOT covered here — they are
# read-only/mechanical (job-status poll; PDF page count), same rationale as
# GET /read/* and GET /healthz* staying unclassified.
# ---------------------------------------------------------------------------


def test_transcribe_batch_path_is_admin_gated():
    import gateway.app.middleware as mw_mod

    assert "/wiki/transcribe/batch" in mw_mod.ADMIN_PATHS


def test_transcribe_batch_path_over_admin_limit_returns_503(client):
    """The admin semaphore also shields /wiki/transcribe/batch once fully held."""
    import gateway.app.middleware as mw_mod

    acquired = []
    while mw_mod.admin_sem.acquire(blocking=False):
        acquired.append(True)
    try:
        resp = client.post("/wiki/transcribe/batch", json={"sources": ["x.pdf"]})
        assert resp.status_code == 503
    finally:
        for _ in acquired:
            mw_mod.admin_sem.release()


def test_transcribe_batch_path_requires_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(_fresh_app())
    resp = client.post("/wiki/transcribe/batch", json={"sources": ["x.pdf"]})
    assert resp.status_code == 401


def test_transcribe_batch_estimate_is_zero_true_cost_is_per_page(monkeypatch):
    """No flat admission estimate for /wiki/transcribe/batch either — it is
    the SAME force-transcribe surface as /wiki/transcribe, metered by the
    same per-page hook (issue #460), not a second flat charge."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    assert budget_mod.estimate_cost("/wiki/transcribe/batch") == 0.0


# ---------------------------------------------------------------------------
# AC6: optional KB_ADMIN_TOKEN kill-switch
# ---------------------------------------------------------------------------


def test_admin_open_when_token_unset(client):
    """Admin paths are reachable (not 401) when KB_ADMIN_TOKEN is unset."""
    # Saturate nothing; just confirm we do not get 401 (the path may 500/200
    # depending on sub-app state, but never 401 in open mode).
    resp = client.post("/wiki/index")
    assert resp.status_code != 401


def test_admin_requires_bearer_when_token_set(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(_fresh_app())
    resp = client.post("/wiki/index")
    assert resp.status_code == 401


def test_admin_accepts_correct_bearer(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(_fresh_app())
    resp = client.post("/wiki/index", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code != 401


def test_admin_rejects_wrong_bearer(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(_fresh_app())
    resp = client.post("/wiki/index", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_token_does_not_gate_read_paths(monkeypatch):
    """The kill-switch protects admin paths only; read paths stay open."""
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(_fresh_app())
    # No Authorization header on a READ path: must not be 401 (token gates admin).
    resp = client.post("/wiki/chat", json={"query": "hi"})
    assert resp.status_code != 401


# ---------------------------------------------------------------------------
# AC5: graceful provider failure (insufficient_quota / 429) → 503
# ---------------------------------------------------------------------------


def test_quota_error_maps_to_503(monkeypatch):
    """An OpenAI RateLimitError raised inside a heavy handler → friendly 503."""
    import openai

    import gateway.app.middleware as mw_mod

    importlib.reload(mw_mod)

    # Build a RateLimitError the way the SDK does (insufficient_quota is a 429).
    def _boom():
        raise openai.RateLimitError(
            message="insufficient_quota",
            response=_FakeResponse(429),
            body={"error": {"code": "insufficient_quota"}},
        )

    # is_provider_quota_error must recognise it.
    try:
        _boom()
    except Exception as exc:  # noqa: BLE001
        assert mw_mod.is_provider_quota_error(exc) is True


class _FakeResponse:
    """Minimal stand-in for httpx.Response used to build an SDK error."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.request = None
        self.headers = {}
