"""Budget estimate calibration + GET /healthz/budget visibility (issue #510).

PRD #508 grill finding: the daily-budget ledger's per-endpoint estimates ran
5-50x over real cost, so a normal demo day (6 Lint runs + 3 Ingests + one
Import + one Upload + two small-corpus re-embeds + a 63-page Transcribe batch
+ several chats) tripped the $3.00 ``KB_DAILY_USD_CAP`` on well under $0.40 of
real spend, and the ledger was invisible until that 503 happened. This file
covers:

  - The three recalibrated numbers (lint, rag/index + hybrid/index,
    KB_TRANSCRIBE_PAGE_USD default) land at their new values and
    KB_DAILY_USD_CAP is unchanged.
  - GET /healthz/budget: shape, thread-safety (reflects a worker-thread
    charge_pages() call), and that it NEVER charges the ledger itself.
  - The grill-day ledger-arithmetic scenario totals well under $3.00
    post-calibration.

All hermetic — no OPENAI_API_KEY, no real network (mirrors
test_prod_middleware.py's fresh-app-per-test pattern).
"""

from __future__ import annotations

import importlib
import threading

import pytest
from fastapi.testclient import TestClient


def _fresh_app():
    """Reload the gateway app so module-level budget/middleware state is pristine."""
    import gateway.app.budget as budget_mod
    import gateway.app.main as main_mod
    import gateway.app.middleware as mw_mod

    importlib.reload(budget_mod)
    importlib.reload(mw_mod)
    importlib.reload(main_mod)
    return main_mod.app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("KB_DAILY_USD_CAP", raising=False)
    monkeypatch.delenv("KB_TRANSCRIBE_PAGE_USD", raising=False)
    return TestClient(_fresh_app())


# ---------------------------------------------------------------------------
# Recalibrated estimates (issue #510 AC1)
# ---------------------------------------------------------------------------


def test_lint_estimate_recalibrated_to_three_cents():
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    assert budget_mod.estimate_cost("/wiki/lint") == pytest.approx(0.03)


def test_rag_index_estimate_recalibrated_to_ten_cents():
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    assert budget_mod.estimate_cost("/rag/index") == pytest.approx(0.10)


def test_hybrid_index_estimate_recalibrated_to_ten_cents():
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    assert budget_mod.estimate_cost("/hybrid/index") == pytest.approx(0.10)


def test_transcribe_page_usd_default_recalibrated_to_half_cent(monkeypatch):
    import gateway.app.budget as budget_mod

    monkeypatch.delenv("KB_TRANSCRIBE_PAGE_USD", raising=False)
    importlib.reload(budget_mod)
    assert pytest.approx(0.005) == budget_mod.TRANSCRIBE_PAGE_USD


def test_daily_cap_unchanged_at_three_dollars(monkeypatch):
    import gateway.app.budget as budget_mod

    monkeypatch.delenv("KB_DAILY_USD_CAP", raising=False)
    importlib.reload(budget_mod)
    assert pytest.approx(3.0) == budget_mod.DAILY_USD_CAP


def test_unrecalibrated_estimates_untouched():
    """Sanity check the recalibration didn't touch endpoints the issue didn't name."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    assert budget_mod.estimate_cost("/wiki/chat") == pytest.approx(0.02)
    assert budget_mod.estimate_cost("/wiki/ingest") == pytest.approx(0.10)
    assert budget_mod.estimate_cost("/wiki/index") == pytest.approx(0.05)
    assert budget_mod.estimate_cost("/upload") == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# GET /healthz/budget (issue #510 AC2)
# ---------------------------------------------------------------------------


def test_healthz_budget_shape(client):
    resp = client.get("/healthz/budget")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"day", "spent_estimate", "cap", "remaining"}
    assert isinstance(body["day"], str) and len(body["day"]) == 10  # YYYY-MM-DD
    assert body["spent_estimate"] == pytest.approx(0.0)
    assert body["cap"] == pytest.approx(3.0)
    assert body["remaining"] == pytest.approx(3.0)


def test_healthz_budget_never_charges(client):
    """Repeated GETs must not move the ledger — this is a read-only probe."""
    import gateway.app.budget as budget_mod

    before = budget_mod.budget.day_total()
    for _ in range(5):
        resp = client.get("/healthz/budget")
        assert resp.status_code == 200
    assert budget_mod.budget.day_total() == before == pytest.approx(0.0)


def test_healthz_budget_unauthenticated_even_with_admin_token(monkeypatch):
    """The probe stays open even when KB_ADMIN_TOKEN is set (mirrors /healthz)."""
    monkeypatch.setenv("KB_ADMIN_TOKEN", "s3cret")
    client = TestClient(_fresh_app())
    resp = client.get("/healthz/budget")
    assert resp.status_code == 200


def test_healthz_budget_reflects_event_loop_charge(client):
    """A charge() from a heavy request shows up in the next snapshot."""
    import gateway.app.budget as budget_mod

    client.post("/wiki/chat", json={"query": "hi"})
    expected = budget_mod.estimate_cost("/wiki/chat")

    resp = client.get("/healthz/budget")
    body = resp.json()
    assert body["spent_estimate"] == pytest.approx(expected)
    assert body["remaining"] == pytest.approx(body["cap"] - expected)


def test_healthz_budget_reflects_worker_thread_charge_pages(client):
    """A charge_pages() call from a worker thread (Transcribe's real path) is
    visible to the NEXT snapshot — proves the endpoint reads the SAME shared,
    lock-protected ledger the middleware/hook charge into (issue #472's
    thread-safety guarantee extended to this new read path)."""
    import gateway.app.budget as budget_mod

    done = threading.Event()

    def worker():
        budget_mod.budget.charge_pages(10)
        done.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    assert done.is_set(), "worker thread did not complete in time"

    resp = client.get("/healthz/budget")
    body = resp.json()
    assert body["spent_estimate"] == pytest.approx(10 * budget_mod.TRANSCRIBE_PAGE_USD)


def test_healthz_budget_remaining_clamps_at_zero_over_cap(monkeypatch):
    """remaining never goes negative even once the ledger has crossed the cap
    (charge-after-admit can push spent_estimate slightly past cap_usd)."""
    monkeypatch.setenv("KB_DAILY_USD_CAP", "0.01")
    client = TestClient(_fresh_app())
    import gateway.app.budget as budget_mod

    # Directly charge past the tiny cap (bypassing the gate) to simulate the
    # admitted-then-over-cap moment the middleware itself can produce.
    budget_mod.budget.charge("/wiki/ingest")  # 0.10 > 0.01 cap
    resp = client.get("/healthz/budget")
    body = resp.json()
    assert body["spent_estimate"] > body["cap"]
    assert body["remaining"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Ledger arithmetic: the grill-day scenario totals well under $3.00 (issue
# #510 AC4)
# ---------------------------------------------------------------------------


def test_grill_day_scenario_totals_well_under_cap():
    """Replays the PRD #508 grill-day operation sequence against the
    recalibrated table: 6 Lint + 3 Ingest + 1 Import + 1 Upload + 2 re-embeds
    + one 63-page Transcribe batch + several chats. Pre-calibration this
    sequence charged well over $3.00 in accounting fiction on <$0.40 of real
    spend; post-calibration it must stay comfortably under the $3.00 cap."""
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    b = budget_mod.DailyBudget(cap_usd=budget_mod.DAILY_USD_CAP)

    for _ in range(6):
        b.charge("/wiki/lint")
    for _ in range(3):
        b.charge("/wiki/ingest")
    b.charge("/wiki/import")
    b.charge("/upload")
    b.charge("/rag/index")
    b.charge("/hybrid/index")
    b.charge_pages(63)  # the 63-page Transcribe batch
    for _ in range(10):  # "several" chats — a generous reading of the word
        b.charge("/wiki/chat")

    total = b.day_total()
    assert total < 2.0, (
        f"recalibrated grill-day total ${total:.2f} should be comfortably "
        f"under the $3.00 cap (issue #510 AC4)"
    )
    assert b.over_cap() is False
