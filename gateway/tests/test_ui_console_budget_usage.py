"""Console daily budget usage display (issue #510).

Following the pattern in ``test_ui_console_i18n_coverage.py``: inspects the
production ``gateway/static/console.html`` file's text — no DOM, no fetch, no
browser, no OPENAI_API_KEY (fully hermetic, §6.3 / §12.7). Verifies:

  - The static banner markup + its bilingual LINT_CHROME keys exist and are
    real Chinese in the zh half.
  - ``renderBudgetUsage`` renders via ``textContent`` only (never innerHTML,
    §12.4) and templates all three placeholders.
  - Boot fetches the read-only ``GET /healthz/budget`` probe exactly once and
    wires the result into ``renderBudgetUsage``.
  - ``applyConsoleLang`` re-renders the budget banner on language toggle (so
    switching zh/en updates the banner without a re-fetch).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling ``test_ui_console_*`` helper of the same
    name — robust to nested braces inside the body)."""
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
# Static markup
# ---------------------------------------------------------------------------


def test_budget_usage_banner_markup_present():
    text = _console_text()
    assert '<div id="budget-usage-banner"' in text
    assert '<span id="budget-usage-text"' in text


# ---------------------------------------------------------------------------
# Bilingual chrome
# ---------------------------------------------------------------------------


def test_budget_usage_chrome_keys_present_and_zh_is_real_chinese():
    text = _console_text()
    for key in ("budgetUsageLoading", "budgetUsageError", "budgetUsageText"):
        assert f"{key}:" in text, f"LINT_CHROME must define {key}"

    zh_match = re.search(r'budgetUsageText:\s*"([^"]+)"', text.split("zh: {", 1)[1])
    assert zh_match is not None
    assert _has_cjk(zh_match.group(1)), "zh budgetUsageText must contain real Chinese text"
    assert "{spent}" in zh_match.group(1)
    assert "{cap}" in zh_match.group(1)
    assert "{remaining}" in zh_match.group(1)


def test_budget_usage_text_template_has_all_three_placeholders_in_en():
    text = _console_text()
    en_match = re.search(
        r'budgetUsageText:\s*"([^"]+)"', text.split("en: {", 1)[1].split("zh: {", 1)[0]
    )
    assert en_match is not None
    template = en_match.group(1)
    assert "{spent}" in template
    assert "{cap}" in template
    assert "{remaining}" in template


# ---------------------------------------------------------------------------
# renderBudgetUsage: textContent-only, all placeholders replaced
# ---------------------------------------------------------------------------


def test_render_budget_usage_uses_textcontent_only():
    text = _console_text()
    fn = _extract_function(text, "renderBudgetUsage")
    assert ".innerHTML" not in fn, "§12.4: renderBudgetUsage must not use innerHTML"
    assert "textEl.textContent" in fn


def test_render_budget_usage_handles_null_and_false_sentinels():
    """null = not-yet-loaded, false = fetch failed — both render distinct chrome."""
    text = _console_text()
    fn = _extract_function(text, "renderBudgetUsage")
    assert "snapshot === null" in fn
    assert "budgetUsageLoading" in fn
    assert "snapshot === false" in fn
    assert "budgetUsageError" in fn


def test_render_budget_usage_replaces_all_placeholders():
    text = _console_text()
    fn = _extract_function(text, "renderBudgetUsage")
    assert '.replace("{spent}"' in fn
    assert '.replace("{cap}"' in fn
    assert '.replace("{remaining}"' in fn


# ---------------------------------------------------------------------------
# Boot: fetches GET /healthz/budget exactly once, wires into renderBudgetUsage
# ---------------------------------------------------------------------------


def test_boot_fetches_healthz_budget_exactly_once():
    text = _console_text()
    assert text.count('fetch("/healthz/budget")') == 1


def test_boot_budget_fetch_updates_last_snapshot_and_renders():
    text = _console_text()
    fn = _extract_function(text, "loadBudgetUsage")
    assert "lastBudgetSnapshot = data;" in fn
    assert "renderBudgetUsage(lastBudgetSnapshot);" in fn
    assert "lastBudgetSnapshot = false;" in fn, "a fetch failure must set the false sentinel"


# ---------------------------------------------------------------------------
# Language toggle re-renders the budget banner (no re-fetch needed)
# ---------------------------------------------------------------------------


def test_apply_console_lang_rerenders_budget_usage():
    text = _console_text()
    fn = _extract_function(text, "applyConsoleLang")
    assert "renderBudgetUsage(lastBudgetSnapshot);" in fn
