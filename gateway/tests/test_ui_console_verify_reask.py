"""Structural tests for the C1 "Verify: re-ask" control (tier-B S8, issue
#384, ADR-0027 decision 2).

Following the pattern in ``test_ui_console_routed_coverage_fill.py``, these
tests inspect the production ``gateway/static/console.html`` file's text —
no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- The C1 row wires BOTH "Fill via Import" (navigation, never fetches) and
  "Verify: re-ask" (a real POST /wiki/chat + re-lint) — C2 keeps only the
  former (its closure signal is the existing honest-miss check).
- "Verify: re-ask" reuses a sample query from the finding (falling back to
  the canonical query) and goes through the shared runLintRemediation +
  allButtons in-flight guard, exactly like every other executing
  remediation on this card.
- chatVerifyRequest posts to the existing POST /wiki/chat endpoint — no
  new server route.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _function_body(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace-depth
    scanning — robust to nested braces inside the body."""
    m = re.search(rf"function {name}\([^)]*\)\s*\{{", text)
    assert m is not None, f"console.html must define function {name}(...)"
    start = m.end() - 1  # index of the opening brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unbalanced braces scanning function {name}")


def _row_renderer_body(text: str, code: str) -> str:
    row = re.search(rf"{code}:\s*function\(i, f\)\s*\{{(.*?)\n    \}},", text, re.DOTALL)
    assert row is not None, f"could not isolate the {code} row renderer"
    return row.group(1)


# ---------------------------------------------------------------------------
# C1 row wires both actions; C2 keeps only Fill via Import
# ---------------------------------------------------------------------------


def test_c1_row_wires_fill_via_import_and_verify_reask():
    text = _console_text()
    c1_row = _row_renderer_body(text, "C1")
    assert "fillViaImportAction(" in c1_row
    assert "verifyReaskAction(" in c1_row


def test_c2_row_does_not_get_verify_reask():
    """Verify: re-ask is C1's own closure signal (ADR-0027 decision 2) —
    C2's closure is the existing honest-miss check, unaffected by this slice."""
    text = _console_text()
    c2_row = _row_renderer_body(text, "C2")
    assert "fillViaImportAction(" in c2_row
    assert "verifyReaskAction(" not in c2_row


# ---------------------------------------------------------------------------
# verifyReaskAction: real request, shared in-flight guard, bilingual label
# ---------------------------------------------------------------------------


def test_verify_reask_action_issues_a_chat_verify_request():
    """Unlike fillViaImportAction (which must never fetch), this control DOES
    issue a request — the point of the AC is a real re-ask."""
    text = _console_text()
    body = _function_body(text, "verifyReaskAction")
    assert "chatVerifyRequest(" in body
    assert "runLintRemediation(" in body


def test_verify_reask_action_joins_the_shared_in_flight_guard():
    """Every executing remediation pushes its button onto allButtons so the
    in-flight guard (issue #364) disables it alongside its siblings — unlike
    fillViaImportAction, which is deliberately excluded."""
    text = _console_text()
    body = _function_body(text, "verifyReaskAction")
    assert "allButtons.push(btn)" in body


def test_verify_reask_uses_a_sample_query_with_canonical_fallback():
    text = _console_text()
    body = _function_body(text, "verifyReaskAction")
    assert "finding.sample_raw_queries" in body
    assert "finding.query_canonical" in body


def test_verify_reask_button_class_and_bilingual_label():
    text = _console_text()
    assert "verify-reask-btn" in text
    en_match = re.search(r"en:\s*\{(.*?)\n  \},", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},", text, re.DOTALL)
    assert en_match is not None and zh_match is not None
    assert re.search(r'verifyReask:\s*"[^"]+"', en_match.group(1)), "en.verifyReask missing"
    assert re.search(r'verifyReask:\s*"[^"]+"', zh_match.group(1)), "zh.verifyReask missing"


# ---------------------------------------------------------------------------
# chatVerifyRequest: existing POST /wiki/chat, no new endpoint
# ---------------------------------------------------------------------------


def test_chat_verify_request_posts_to_the_existing_chat_endpoint():
    text = _console_text()
    body = _function_body(text, "chatVerifyRequest")
    assert '"/wiki/chat"' in body
    assert '"POST"' in body
    assert "query" in body


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
