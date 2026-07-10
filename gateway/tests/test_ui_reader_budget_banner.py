"""Structural tests for the reader UI's budget-exhausted banner (issue #560).

The reader UI lives in ``gateway/static/index.html`` as a single vanilla
HTML/CSS/JS file (CODING_STANDARD §12.1). Following the established pattern in
``test_ui_reader_feedback.py``, these tests inspect the production UI file's
text — no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- ``startRequest``'s ``!resp.ok`` branch reads the body of a 503 and, when the
  detail matches the budget-middleware's exact contract string
  (``"daily demo budget reached"``), renders ``onBudgetExhausted()`` instead
  of the generic ``onError`` HTTP-status box (AC1).
- Any other non-ok status/detail (including other 503s, e.g. load-shed) falls
  through unchanged to the existing generic ``"HTTP " + status + ": " +
  statusText`` path (AC2, regression).
- A ``resp.json()`` parse failure on a non-ok 503 falls back to the generic
  path via a ``.catch`` — no uncaught rejection (AC3).
- ``onBudgetExhausted`` renders a ``.notice`` block (not ``.err``), has no
  "Try again" button, and never reuses the red error styling.
- ``CHROME.en`` / ``CHROME.zh`` carry the banner's bilingual copy; zh values
  are real Chinese text (i18n coverage, AC4).
- textContent-only (§12.4); no client-side re-implementation of the budget
  policy beyond matching the literal contract string the gateway documents.
"""

from __future__ import annotations

import re
from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling test helper of the same name)."""
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
    raise AssertionError(f"unterminated function body for {name}")


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


# ---------------------------------------------------------------------------
# AC1: exact budget detail on a 503 -> onBudgetExhausted(), no generic error
# ---------------------------------------------------------------------------


def test_start_request_detects_budget_503_and_renders_banner():
    text = _ui_text()
    fn = _extract_function(text, "startRequest")
    assert "resp.status === 503" in fn
    assert '"daily demo budget reached"' in fn
    assert "onBudgetExhausted()" in fn


def test_onbudgetexhausted_function_exists_and_renders_notice_block():
    text = _ui_text()
    fn = _extract_function(text, "onBudgetExhausted")
    assert '"notice"' in fn
    assert 't("budgetBody")' in fn


# ---------------------------------------------------------------------------
# AC2: any other non-ok status/detail keeps the existing generic error path
# ---------------------------------------------------------------------------


_GENERIC_HTTP_ERROR = 'onError({ detail: "HTTP " + resp.status + ": " + resp.statusText });'


def test_non_budget_503_and_non_503_non_ok_keep_generic_error_path_unchanged():
    """The unchanged generic error string is defined once (genericHttpError)
    and called from three sites: (1) a 503 whose body detail isn't the exact
    budget contract (e.g. the load-shed 503 {"detail": "server busy, please
    retry"}), (2) a 503 whose body fails to parse, and (3) any non-503
    non-ok status (e.g. 400 unknown stack). All three are regressions of the
    pre-existing generic path."""
    text = _ui_text()
    fn = _extract_function(text, "startRequest")
    assert fn.count(_GENERIC_HTTP_ERROR) == 1, (
        "expected the generic HTTP-error string defined exactly once "
        f"(genericHttpError) — found {fn.count(_GENERIC_HTTP_ERROR)}"
    )
    assert fn.count("genericHttpError()") == 3, (
        "expected exactly 3 call sites for genericHttpError() (non-budget "
        "503 detail, 503 body-parse failure, non-503 non-ok) — found "
        f"{fn.count('genericHttpError()')}"
    )


# ---------------------------------------------------------------------------
# AC3: body-parse failure on a non-ok 503 -> generic path, no uncaught rejection
# ---------------------------------------------------------------------------


def test_json_parse_failure_on_503_has_catch_falling_back_to_generic_error():
    text = _ui_text()
    fn = _extract_function(text, "startRequest")
    branch_503 = fn[fn.index("resp.status === 503") :]
    catch_block = branch_503[branch_503.index(".catch(function() {") :]
    assert "genericHttpError();" in catch_block[:300]


# ---------------------------------------------------------------------------
# Banner styling: informational .notice, not the red .err treatment, no button
# ---------------------------------------------------------------------------


def test_onbudgetexhausted_never_uses_err_class_or_try_again_button():
    text = _ui_text()
    fn = _extract_function(text, "onBudgetExhausted")
    assert '"err"' not in fn
    assert "tryAgain" not in fn
    assert 'el("button"' not in fn


def test_notice_css_class_does_not_reuse_fail_color():
    text = _ui_text()
    notice_block = text[text.index(".notice {") : text.index(".notice {") + 400]
    assert "var(--fail)" not in notice_block


# ---------------------------------------------------------------------------
# Bilingual chrome (i18n coverage, AC4)
# ---------------------------------------------------------------------------


def test_chrome_defines_budget_banner_keys_in_both_languages():
    text = _ui_text()
    en_block = text.split("var CHROME = {", 1)[1].split("zh: {", 1)[0]
    zh_block = text.split("zh: {", 1)[1].split("};", 1)[0]
    keys = ("noticeLabel", "budgetBody")
    for key in keys:
        assert f"{key}:" in en_block, f"CHROME.en must define {key}"
        assert f"{key}:" in zh_block, f"CHROME.zh must define {key}"

    zh_body = re.search(r'budgetBody:\s*"([^"]+)"', zh_block)
    assert zh_body is not None and _has_cjk(zh_body.group(1))

    en_body = re.search(r'budgetBody:\s*"([^"]+)"', en_block)
    assert en_body is not None and "budget" in en_body.group(1).lower()


# ---------------------------------------------------------------------------
# §12.4: textContent only, no new innerHTML introduced
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_after_change():
    text = _ui_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
