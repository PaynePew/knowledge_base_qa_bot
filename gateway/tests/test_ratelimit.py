"""Per-IP fixed-window rate limiter (issue #598 Slice A / Q3).

Unit-level coverage for ``gateway.app.ratelimit``: env parsing, the
``RateLimiter`` fixed-window admit/prune behaviour, and ``client_ip``
extraction. Middleware wiring (guard order, 429 response shape, budget
non-consumption) is covered separately in test_rate_limit_middleware.py.

All hermetic — no OPENAI_API_KEY, no real network, no real clock (``allow``
takes an injectable ``now`` so window-expiry is deterministic).
"""

from __future__ import annotations

import importlib


def _reload():
    import gateway.app.ratelimit as ratelimit_mod

    importlib.reload(ratelimit_mod)
    return ratelimit_mod


# ---------------------------------------------------------------------------
# KB_RATE_LIMIT_PER_IP env parsing
# ---------------------------------------------------------------------------


def test_rate_limit_per_ip_default_is_thirty(monkeypatch):
    monkeypatch.delenv("KB_RATE_LIMIT_PER_IP", raising=False)
    ratelimit_mod = _reload()
    assert ratelimit_mod.RATE_LIMIT_PER_IP == 30


def test_rate_limit_per_ip_reads_env_override(monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "5")
    ratelimit_mod = _reload()
    assert ratelimit_mod.RATE_LIMIT_PER_IP == 5


def test_rate_limit_per_ip_zero_is_preserved_not_defaulted(monkeypatch):
    """0 is a valid, meaningful value (disables rate limiting) — not a fallback trigger."""
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "0")
    ratelimit_mod = _reload()
    assert ratelimit_mod.RATE_LIMIT_PER_IP == 0


def test_rate_limit_per_ip_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "not-a-number")
    ratelimit_mod = _reload()
    assert ratelimit_mod.RATE_LIMIT_PER_IP == 30


def test_rate_limit_per_ip_falls_back_on_negative_value(monkeypatch):
    monkeypatch.setenv("KB_RATE_LIMIT_PER_IP", "-1")
    ratelimit_mod = _reload()
    assert ratelimit_mod.RATE_LIMIT_PER_IP == 30


# ---------------------------------------------------------------------------
# RateLimiter — fixed-window admit / block / reset
# ---------------------------------------------------------------------------


def test_allow_admits_up_to_the_limit_then_blocks():
    ratelimit_mod = _reload()
    rl = ratelimit_mod.RateLimiter(limit=3, window_sec=300.0)
    assert rl.allow("1.2.3.4", now=0.0) is True
    assert rl.allow("1.2.3.4", now=0.0) is True
    assert rl.allow("1.2.3.4", now=0.0) is True
    assert rl.allow("1.2.3.4", now=0.0) is False  # 4th request in the same window


def test_allow_resets_after_the_window_elapses():
    ratelimit_mod = _reload()
    rl = ratelimit_mod.RateLimiter(limit=1, window_sec=300.0)
    assert rl.allow("1.2.3.4", now=0.0) is True
    assert rl.allow("1.2.3.4", now=100.0) is False  # still inside the same 300s window
    assert rl.allow("1.2.3.4", now=300.0) is True  # a full window has elapsed


def test_allow_tracks_distinct_ips_independently():
    ratelimit_mod = _reload()
    rl = ratelimit_mod.RateLimiter(limit=1, window_sec=300.0)
    assert rl.allow("1.2.3.4", now=0.0) is True
    assert rl.allow("1.2.3.4", now=0.0) is False
    assert rl.allow("5.6.7.8", now=0.0) is True  # a different IP has its own window


def test_limit_zero_always_allows_and_records_nothing():
    """limit=0 disables rate limiting entirely — the store must stay empty."""
    ratelimit_mod = _reload()
    rl = ratelimit_mod.RateLimiter(limit=0, window_sec=300.0)
    for _ in range(50):
        assert rl.allow("1.2.3.4", now=0.0) is True
    assert len(rl._windows) == 0  # never recorded — a disabled limiter must not grow the store


def test_expired_windows_are_pruned_bounding_the_store():
    """A fully expired IP's window is dropped on the next allow() call from any IP."""
    ratelimit_mod = _reload()
    rl = ratelimit_mod.RateLimiter(limit=5, window_sec=300.0)
    rl.allow("1.2.3.4", now=0.0)
    assert "1.2.3.4" in rl._windows
    rl.allow("5.6.7.8", now=300.0)  # a full window later — 1.2.3.4's entry has expired
    assert "1.2.3.4" not in rl._windows


# ---------------------------------------------------------------------------
# client_ip — X-Forwarded-For RIGHTMOST hop (issue #598 scope addendum point
# 4), else scope["client"]
# ---------------------------------------------------------------------------


def test_client_ip_uses_rightmost_x_forwarded_for_hop():
    ratelimit_mod = _reload()
    scope = {
        "headers": [(b"x-forwarded-for", b"9.9.9.9, 10.0.0.1, 10.0.0.2")],
        "client": ("127.0.0.1", 12345),
    }
    assert ratelimit_mod.client_ip(scope) == "10.0.0.2"


def test_client_ip_strips_whitespace_around_rightmost_hop():
    ratelimit_mod = _reload()
    scope = {"headers": [(b"x-forwarded-for", b"9.9.9.9,  10.0.0.1  ")], "client": None}
    assert ratelimit_mod.client_ip(scope) == "10.0.0.1"


def test_client_ip_single_hop_is_both_leftmost_and_rightmost():
    ratelimit_mod = _reload()
    scope = {"headers": [(b"x-forwarded-for", b"9.9.9.9")], "client": None}
    assert ratelimit_mod.client_ip(scope) == "9.9.9.9"


def test_client_ip_rightmost_hop_cannot_be_spoofed_by_a_forged_leading_hop():
    """Security property (issue #598 scope addendum point 4): a client that
    sets its OWN X-Forwarded-For header can only prepend forged hops ahead of
    whatever the trusted edge appends -- the edge-appended rightmost hop is
    what must be trusted, never the client-controlled first one."""
    ratelimit_mod = _reload()
    forged_first_hop = "6.6.6.6"  # attacker-controlled, spoofed to evade the limiter
    edge_appended_real_client = "9.9.9.9"  # appended by the trusted edge, last
    scope = {
        "headers": [
            (b"x-forwarded-for", f"{forged_first_hop}, {edge_appended_real_client}".encode())
        ],
        "client": None,
    }
    assert ratelimit_mod.client_ip(scope) == edge_appended_real_client


def test_client_ip_falls_back_to_scope_client_when_header_absent():
    ratelimit_mod = _reload()
    scope = {"headers": [], "client": ("203.0.113.7", 54321)}
    assert ratelimit_mod.client_ip(scope) == "203.0.113.7"


def test_client_ip_falls_back_to_unknown_when_nothing_present():
    ratelimit_mod = _reload()
    scope = {"headers": [], "client": None}
    assert ratelimit_mod.client_ip(scope) == "unknown"
