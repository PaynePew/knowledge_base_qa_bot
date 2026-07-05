"""Thread-safety regression tests for the daily USD budget ledger (issue #472).

`DailyBudget._totals` gained a second class of writer once Transcribe's
per-page hook (issue #460) started running on worker threads
(``asyncio.to_thread`` batch runs, anyio threadpool sync handlers) alongside
`ProdMiddleware.charge()` on the event loop. A non-atomic read-modify-write
loses updates under concurrent callers, and a separate over_cap-then-charge
sequence lets two concurrent callers both pass the check before either
charges. These tests fail on the pre-#472 (unlocked) implementation and pass
once `DailyBudget` holds a lock around every mutation and around the atomic
`reserve_pages` admission check.
"""

from __future__ import annotations

import importlib
import sys
import threading

import pytest

import gateway.app.budget as budget_mod


def _reload() -> None:
    importlib.reload(budget_mod)


@pytest.fixture(autouse=True)
def _force_thread_interleaving():
    """Shrink the GIL's thread-switch interval for this module's tests.

    CPython only switches between threads roughly every 5ms (the default
    ``sys.getswitchinterval()``) or on I/O — a plain tight loop of dict
    read-modify-writes may simply not get preempted mid-operation often
    enough to reliably surface a lost update inside a short-lived test, even
    on the unlocked implementation. Shrinking the interval makes the race
    surface reliably (verified: on the pre-#472 unlocked implementation this
    reproduces a ~30-70% lost-update loss every run; on the locked
    implementation the total stays exact every run) rather than being a rare
    flake either way. Restored after each test so it does not affect the
    rest of the suite.
    """
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(original)


def test_charge_concurrent_no_lost_update():
    """N threads each calling charge() on the same day-key must sum exactly."""
    _reload()
    b = budget_mod.DailyBudget(cap_usd=1_000_000.0)
    n_threads = 50
    calls_per_thread = 200
    day = "2026-01-01"
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        for _ in range(calls_per_thread):
            b.charge("/upload", day=day)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * calls_per_thread * budget_mod.estimate_cost("/upload")
    assert b.day_total(day=day) == pytest.approx(expected)


def test_charge_pages_concurrent_no_lost_update():
    """N threads each calling charge_pages() on the same day-key must sum exactly.

    This is the exact scenario from the issue: concurrent transcribe batches
    charging per-page cost into the same ledger.
    """
    _reload()
    b = budget_mod.DailyBudget(cap_usd=1_000_000.0)
    n_threads = 50
    calls_per_thread = 200
    day = "2026-01-01"
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        for _ in range(calls_per_thread):
            b.charge_pages(1, day=day)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * calls_per_thread * budget_mod.TRANSCRIBE_PAGE_USD
    assert b.day_total(day=day) == pytest.approx(expected)


def test_reserve_pages_admits_exactly_up_to_cap_under_concurrency(monkeypatch):
    """Concurrent reserve_pages() calls cannot both succeed past the cap.

    Sets the cap to exactly `max_admits` single-page reservations. Under a
    non-atomic "check over_cap, then charge_pages" sequence, many of the
    concurrent callers would observe "under cap" before any of them charges,
    so far more than `max_admits` would be admitted. With reserve_pages()
    atomic under one lock hold, exactly `max_admits` succeed regardless of
    thread interleaving.

    Uses a binary-exact per-page cost (0.125 = 2^-3) so the cap boundary is
    exact in floating point — the default $0.01/page is not exactly
    representable and accumulation order (irrelevant to correctness, but
    real) could otherwise nudge the total a float-epsilon either side of the
    cap and make the expected admit count ambiguous.
    """
    monkeypatch.setenv("KB_TRANSCRIBE_PAGE_USD", "0.125")
    _reload()
    page_cost = budget_mod.TRANSCRIBE_PAGE_USD
    max_admits = 10
    b = budget_mod.DailyBudget(cap_usd=page_cost * max_admits)
    n_threads = 100
    day = "2026-01-01"
    barrier = threading.Barrier(n_threads)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        admitted = b.reserve_pages(1, day=day)
        with results_lock:
            results.append(admitted)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == max_admits
    assert b.day_total(day=day) == pytest.approx(page_cost * max_admits)
