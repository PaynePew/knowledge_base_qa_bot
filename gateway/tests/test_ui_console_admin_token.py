"""Structural tests for the Operator Console's admin-token UX (issue #583).

Following the pattern in ``test_ui_console_lint_fast_default.py`` /
``test_ui_console_lint_remediation_batch.py``, these tests inspect the
production ``gateway/static/console.html`` file's text to assert the
structural invariants of issue #583's console half:

- The token is stored under localStorage key ``kb-admin-token`` (the
  ``kb-console-lang`` precedent).
- ``adminFetch()`` attaches ``Authorization: Bearer <token>`` when a token
  is stored, and surfaces a 401 by opening the token panel with a clear
  "required / invalid" message and focusing the token field — no silent
  failure.
- The token itself is never echoed into DOM text (§12.4): the input is
  ``type="password"``, it never pre-fills from the stored value, the field
  is cleared after Save, and the status line only ever carries fixed
  human-readable strings.
- The bare-fetch audit: no string-literal ``fetch("...")`` call site in the
  console targets an ADMIN_PATHS-classified endpoint — every admin-mutating
  call goes through ``adminFetch``. This is derived DYNAMICALLY from
  ``gateway.app.middleware.ADMIN_PATHS`` (the same enumerate-don't-hardcode
  discipline as ``test_admin_path_coverage.py``), so classifying a new
  endpoint into ADMIN_PATHS makes a bare-fetch console call site fail here
  without any test edit.

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). The actual click -> 401 -> panel-focus loop is verified manually
per §12.7.
"""

from __future__ import annotations

import re
from pathlib import Path

import gateway.app.middleware as mw_mod

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the S4/S5 test files' helper of the same name)."""
    marker = f"function {name}("
    start = text.index(marker)
    depth = 0
    started = False
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
            started = True
        elif text[i] == "}":
            depth -= 1
            if started and depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unbalanced braces extracting function {name}")


# ---------------------------------------------------------------------------
# Storage key + indicator
# ---------------------------------------------------------------------------


def test_storage_key_is_kb_admin_token():
    text = _console_text()
    assert 'ADMIN_TOKEN_STORAGE_KEY = "kb-admin-token"' in text, (
        "the token must live under localStorage key kb-admin-token "
        "(the kb-console-lang precedent — issue #583 scope decision)"
    )


def test_indicator_reflects_boolean_state_only():
    """The masthead lock icon toggles a CSS class from a boolean — the token
    value itself must never reach classList/DOM through this path."""
    fn = _extract_function(_console_text(), "refreshAdminTokenIndicator")
    assert '"token-set", !!getAdminToken()' in fn


# ---------------------------------------------------------------------------
# adminFetch: Bearer header + non-silent 401
# ---------------------------------------------------------------------------


def test_admin_fetch_attaches_bearer_header_when_token_stored():
    fn = _extract_function(_console_text(), "adminFetch")
    assert 'headers["Authorization"] = "Bearer " + token' in fn, (
        "adminFetch must attach Authorization: Bearer <token> for "
        "ADMIN_PATHS endpoints when a token is stored (issue #583)"
    )
    assert "if (token)" in fn, (
        "no token stored -> no Authorization header (unset = open, the "
        "current demo behaviour must be byte-identical)"
    )


def test_admin_fetch_surfaces_401_not_silently():
    fn = _extract_function(_console_text(), "adminFetch")
    assert "resp.status === 401" in fn
    assert "showAdminTokenRequired()" in fn, (
        "a 401 must open the token panel with a clear message — never a "
        "silent failure (issue #583 scope decision)"
    )


def test_401_message_is_clear_and_focuses_the_token_field():
    text = _console_text()
    required_fn = _extract_function(text, "showAdminTokenRequired")
    assert "Admin token required or invalid" in required_fn
    panel_fn = _extract_function(text, "showAdminTokenPanel")
    assert "input.focus()" in panel_fn, (
        "the 401 surface must focus the token field so the operator can "
        "type immediately (issue #583 scope decision)"
    )
    assert "status.textContent =" in panel_fn, (
        "panel status messages render via textContent only (§12.4)"
    )


# ---------------------------------------------------------------------------
# No-echo invariants (§12.4)
# ---------------------------------------------------------------------------


def test_token_input_is_a_password_field():
    text = _console_text()
    m = re.search(r'<input id="admin-token-input"[^>]*>', text)
    assert m is not None
    assert 'type="password"' in m.group(0)
    assert 'autocomplete="off"' in m.group(0)


def test_token_input_never_prefills_from_the_stored_value():
    text = _console_text()
    assert "input.value = getAdminToken()" not in text, (
        "re-opening the panel must not echo the stored token back into the field (§12.4 — no echo)"
    )


def test_save_clears_the_field_and_status_never_carries_the_token():
    text = _console_text()
    init_fn = _extract_function(text, "initAdminTokenPanel")
    assert 'input.value = ""' in init_fn, (
        "Save must clear the field — the token must not sit in the DOM "
        "after storing it (§12.4 — no echo)"
    )
    # Status strings are fixed literals; the token variable is never
    # concatenated into any panel message.
    assert (
        'showAdminTokenPanel(value ? "Admin token saved." : "Admin token cleared.", false)'
        in init_fn
    )


def test_console_never_logs_the_token():
    """No console.log anywhere near the token plumbing — the whole file is
    checked because a single log call would be enough to leak it."""
    text = _console_text()
    for fn_name in (
        "getAdminToken",
        "setAdminToken",
        "adminFetch",
        "showAdminTokenPanel",
        "showAdminTokenRequired",
        "initAdminTokenPanel",
    ):
        assert "console.log" not in _extract_function(text, fn_name)


# ---------------------------------------------------------------------------
# Bare-fetch audit: every ADMIN_PATHS call site goes through adminFetch
# ---------------------------------------------------------------------------

# String-literal bare fetch call sites: fetch("<url>"...). Case-sensitive on
# purpose — adminFetch( contains a capital F, so its call sites never match.
_BARE_FETCH_RE = re.compile(r'(?<![A-Za-z])fetch\(\s*\n?\s*"([^"]+)"')


def test_no_bare_fetch_targets_an_admin_classified_endpoint():
    """Enumerates every string-literal bare ``fetch("...")`` call site and
    asserts none of them resolve into ADMIN_PATHS — mirrors the dynamic
    coverage audit in ``test_admin_path_coverage.py`` so a future endpoint
    classified into ADMIN_PATHS immediately fails this test if the console
    still calls it with a bare fetch (no Authorization header)."""
    text = _console_text()
    offenders: list[str] = []
    for m in _BARE_FETCH_RE.finditer(text):
        literal = m.group(1).split("?")[0]
        canonical = mw_mod._canonical_path(literal)
        if canonical in mw_mod.ADMIN_PATHS:
            offenders.append(m.group(1))
            continue
        # Concatenated URLs (fetch("/wiki/qa/" + encodeURIComponent(slug))):
        # a literal prefix ending in "/" that some ADMIN_PATHS template
        # extends is an admin call site too.
        if literal.endswith("/") and any(p.startswith(literal) for p in mw_mod.ADMIN_PATHS):
            offenders.append(m.group(1))
    assert not offenders, (
        "bare fetch() call site(s) target admin-classified endpoints — they "
        f"must go through adminFetch (issue #583): {sorted(offenders)}"
    )


def test_bare_fetch_audit_actually_sees_call_sites():
    """Guards the audit above against a silently-empty enumeration — the
    console legitimately keeps bare fetch for read paths, so the regex must
    find at least those."""
    text = _console_text()
    literals = {m.group(1).split("?")[0] for m in _BARE_FETCH_RE.finditer(text)}
    assert "/wiki/chat" in literals  # read path — deliberately NOT admin-gated
    assert any(lit.startswith("/read/") for lit in literals)


# ---------------------------------------------------------------------------
# Deploy passthrough (issue #583 AC 3)
# ---------------------------------------------------------------------------


def test_prod_compose_passes_kb_admin_token_through_from_host_env():
    """docker-compose.prod.yml must pass KB_ADMIN_TOKEN through from the
    host env with an empty default — unset host env = empty string = the
    middleware treats the gate as OFF (current demo behaviour preserved)."""
    compose = _CONSOLE_HTML.parents[2] / "docker-compose.prod.yml"
    text = compose.read_text(encoding="utf-8")
    assert "KB_ADMIN_TOKEN: ${KB_ADMIN_TOKEN:-}" in text
