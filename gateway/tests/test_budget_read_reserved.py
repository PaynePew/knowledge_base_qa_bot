"""Read-reserved daily budget floor for ADMIN_PATHS (issue #598 Slice A / Q2).

A visitor burst (or abuse — the box is public and key-free) on ingest/index/
transcribe paths could otherwise drain the whole ``KB_DAILY_USD_CAP`` before
any reader ever gets a grounded answer. ``KB_READ_RESERVED_USD`` (default
$1.00) carves out a floor of the daily cap that ADMIN_PATHS cannot spend into
— ``DailyBudget.over_admin_cap()`` trips at ``cap_usd - read_reserved_usd``
while READ_PATHS keep gating on the unchanged ``over_cap()`` (full cap).

All hermetic — no OPENAI_API_KEY, no real network (mirrors
test_budget_calibration.py's fresh-module-per-test pattern).
"""

from __future__ import annotations

import importlib

import pytest


def _reload():
    import gateway.app.budget as budget_mod

    importlib.reload(budget_mod)
    return budget_mod


# ---------------------------------------------------------------------------
# KB_READ_RESERVED_USD env parsing
# ---------------------------------------------------------------------------


def test_read_reserved_usd_default_is_one_dollar(monkeypatch):
    monkeypatch.delenv("KB_READ_RESERVED_USD", raising=False)
    budget_mod = _reload()
    assert pytest.approx(1.0) == budget_mod.READ_RESERVED_USD


def test_read_reserved_usd_reads_env_override(monkeypatch):
    monkeypatch.setenv("KB_READ_RESERVED_USD", "0.25")
    budget_mod = _reload()
    assert pytest.approx(0.25) == budget_mod.READ_RESERVED_USD


def test_read_reserved_usd_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("KB_READ_RESERVED_USD", "not-a-number")
    budget_mod = _reload()
    assert pytest.approx(1.0) == budget_mod.READ_RESERVED_USD


def test_module_singleton_carries_read_reserved(monkeypatch):
    monkeypatch.delenv("KB_READ_RESERVED_USD", raising=False)
    budget_mod = _reload()
    assert budget_mod.budget.read_reserved_usd == pytest.approx(budget_mod.READ_RESERVED_USD)


# ---------------------------------------------------------------------------
# DailyBudget.over_admin_cap() — the reduced admin ceiling
# ---------------------------------------------------------------------------


def test_over_admin_cap_false_under_reserved_floor():
    """Admin gate stays open while spend is below cap - reserved."""
    budget_mod = _reload()
    b = budget_mod.DailyBudget(cap_usd=3.0, read_reserved_usd=1.0)
    b.charge("/wiki/ingest")  # 0.10, well under the 2.0 admin ceiling (3.0 - 1.0)
    assert b.over_admin_cap() is False


def test_over_admin_cap_true_once_spend_crosses_cap_minus_reserved():
    budget_mod = _reload()
    b = budget_mod.DailyBudget(cap_usd=1.0, read_reserved_usd=0.4)
    # Admin ceiling = 0.6; six 0.10 charges land exactly on the boundary.
    for _ in range(6):
        b.charge("/wiki/ingest")
    assert b.over_admin_cap() is True


def test_over_admin_cap_defaults_to_zero_reserved_when_unset():
    """A DailyBudget built without read_reserved_usd behaves like over_cap()."""
    budget_mod = _reload()
    b = budget_mod.DailyBudget(cap_usd=1.0)
    b.charge("/wiki/ingest")
    assert b.over_admin_cap() == b.over_cap()


def test_read_path_still_gated_on_full_cap_not_reserved_floor():
    """over_cap() (used by READ_PATHS) is unaffected by read_reserved_usd."""
    budget_mod = _reload()
    b = budget_mod.DailyBudget(cap_usd=1.0, read_reserved_usd=0.9)
    for _ in range(5):
        b.charge("/wiki/ingest")  # total 0.50: under the 1.0 full cap...
    assert b.over_cap() is False
    assert b.over_admin_cap() is True  # ...but over the 0.1 admin ceiling


# ---------------------------------------------------------------------------
# snapshot() gains an additive read_reserved field (GET /healthz/budget)
# ---------------------------------------------------------------------------


def test_snapshot_gains_additive_read_reserved_field():
    budget_mod = _reload()
    b = budget_mod.DailyBudget(cap_usd=3.0, read_reserved_usd=1.0)
    snap = b.snapshot()
    assert set(snap.keys()) == {"day", "spent_estimate", "cap", "remaining", "read_reserved"}
    assert snap["read_reserved"] == pytest.approx(1.0)
