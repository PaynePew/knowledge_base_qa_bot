"""Gateway unhandled-exception logging tests (issue #648).

Covers the two acceptance criteria that exercise ``ErrorLoggingMiddleware``
end to end:

(a) A route that raises an unmapped exception -> the response stays a 500
    AND exactly one ``unhandled_error`` line lands in ``gateway/log.md``.
(b) The existing OpenAI-quota path still emits ONLY ``provider_quota_503`` —
    no double-logging via the new middleware.

Hermetic: no OPENAI_API_KEY, no real network. Uses ``POST /hybrid/index``
(an ADMIN_PATHS heavy path routed directly on the Gateway's own app — see
``gateway/app/routes.py::hybrid_index`` — with no per-route try/except
around the re-embed call) as the raising handler for both scenarios,
monkeypatching ``gateway.app.routes._hybrid_build_index``.

Deliberately NOT a route under the mounted ``/wiki`` or ``/rag`` sub-apps:
each of those is its OWN FastAPI instance with its own default
``ServerErrorMiddleware``, which sends a generic 500 (and re-raises) BEFORE
the exception ever reaches the Gateway's ``ProdMiddleware`` — so by the time
``_call_with_quota_guard`` sees it, the response has already started and its
quota-mapping branch can never fire. A route on the Gateway's own top-level
router has no such inner middleware in the way.
"""

from __future__ import annotations

import openai
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def gw_log_path(tmp_path, monkeypatch):
    """Redirect gateway/log.md to a tmp file and return its path."""
    import gateway.app.logger as gw_logger_module

    log_path = tmp_path / "gateway" / "log.md"
    monkeypatch.setattr(gw_logger_module, "LOG_PATH", log_path)
    return log_path


def _client() -> TestClient:
    from gateway.app.main import app

    # raise_server_exceptions=False so an unhandled 500 comes back as a
    # response instead of re-raising inside the test process.
    return TestClient(app, raise_server_exceptions=False)


class _FakeResponse:
    """Minimal stand-in for httpx.Response used to build an SDK error."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.request = None
        self.headers = {}


# ---------------------------------------------------------------------------
# AC (a): unmapped exception -> 500 + exactly one unhandled_error line
# ---------------------------------------------------------------------------


def test_unhandled_exception_returns_500_and_logs_once(gw_log_path, monkeypatch):
    import gateway.app.routes as gw_routes_module

    def _boom() -> int:
        raise RuntimeError("index build exploded")

    monkeypatch.setattr(gw_routes_module, "_hybrid_build_index", _boom)

    resp = _client().post("/hybrid/index")

    assert resp.status_code == 500

    assert gw_log_path.exists(), "gateway/log.md must exist after an unhandled exception"
    lines = [
        ln for ln in gw_log_path.read_text(encoding="utf-8").splitlines() if "unhandled_error" in ln
    ]
    assert len(lines) == 1, f"expected exactly one unhandled_error line, found: {lines}"
    line = lines[0]
    assert "path=/hybrid/index" in line
    assert "exc=RuntimeError" in line
    assert "index build exploded" in line


def test_unhandled_exception_message_truncated_to_200_chars(gw_log_path, monkeypatch):
    import gateway.app.routes as gw_routes_module

    long_message = "x" * 500

    def _boom() -> int:
        raise RuntimeError(long_message)

    monkeypatch.setattr(gw_routes_module, "_hybrid_build_index", _boom)

    resp = _client().post("/hybrid/index")
    assert resp.status_code == 500

    content = gw_log_path.read_text(encoding="utf-8")
    line = next(ln for ln in content.splitlines() if "unhandled_error" in ln)
    # Only the message portion (after "<ExcClass>: ") is bounded to <=200
    # chars per the issue's AC — the exc= field as a whole also carries the
    # exception class name.
    message_part = line.split("exc=RuntimeError: ", 1)[1]
    assert len(message_part) <= 200, (
        f"message must be bounded to <=200 chars, got {len(message_part)}"
    )


# ---------------------------------------------------------------------------
# AC (b): provider-quota path keeps its existing kind, no double-logging
# ---------------------------------------------------------------------------


def test_provider_quota_error_logs_only_provider_quota_503(gw_log_path, monkeypatch):
    import gateway.app.routes as gw_routes_module

    def _boom() -> int:
        raise openai.RateLimitError(
            message="insufficient_quota",
            response=_FakeResponse(429),
            body={"error": {"code": "insufficient_quota"}},
        )

    monkeypatch.setattr(gw_routes_module, "_hybrid_build_index", _boom)

    resp = _client().post("/hybrid/index")

    assert resp.status_code == 503
    assert resp.json() == {"detail": "LLM provider quota exhausted, please retry later."}

    content = gw_log_path.read_text(encoding="utf-8") if gw_log_path.exists() else ""
    quota_lines = [ln for ln in content.splitlines() if "provider_quota_503" in ln]
    unhandled_lines = [ln for ln in content.splitlines() if "unhandled_error" in ln]

    assert len(quota_lines) == 1, (
        f"expected exactly one provider_quota_503 line, found: {quota_lines}"
    )
    assert unhandled_lines == [], (
        f"provider_quota_503 must NOT also emit unhandled_error; found: {unhandled_lines}"
    )
