"""Console Ingest card: sections_count / uncarried_chars / enriched_chars
(issue #511, ADR-0033 "Ingest observability" decision).

Following the pattern in ``test_ui_console_budget_usage.py``: inspects the
production ``gateway/static/console.html`` file's text — no DOM, no fetch, no
browser, no OPENAI_API_KEY (fully hermetic, §6.3 / §12.7). Verifies:

  - The three new ``LINT_CHROME`` keys exist in both ``en`` and ``zh``, and
    the zh strings are real Chinese with the ``{n}`` placeholder intact.
  - ``renderIngestCard`` renders the Section count on the first page row of
    each Source, and only renders the uncarried/enriched crumbs when the
    corresponding count is non-zero (anomaly signal, not routine noise).
  - The hash-match skipped-source row also renders the Section count.
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
# Bilingual chrome
# ---------------------------------------------------------------------------


def test_ingest_observability_chrome_keys_present():
    text = _console_text()
    for key in (
        "ingestCardSectionsCrumb",
        "ingestCardUncarriedCharsCrumb",
        "ingestCardEnrichedCharsCrumb",
    ):
        assert f"{key}:" in text, f"LINT_CHROME must define {key}"


def test_ingest_observability_zh_strings_are_real_chinese_with_placeholder():
    text = _console_text()
    zh_block = text.split("zh: {", 1)[1]
    for key in (
        "ingestCardSectionsCrumb",
        "ingestCardUncarriedCharsCrumb",
        "ingestCardEnrichedCharsCrumb",
    ):
        match = re.search(rf'{key}:\s*"([^"]+)"', zh_block)
        assert match is not None, f"LINT_CHROME.zh.{key} missing"
        assert _has_cjk(match.group(1)), f"zh {key} must contain real Chinese text"
        assert "{n}" in match.group(1), f"zh {key} must keep the {{n}} placeholder"


# ---------------------------------------------------------------------------
# renderIngestCard: Section count always shown, anomaly crumbs conditional
# ---------------------------------------------------------------------------


def test_render_ingest_card_shows_sections_count_on_first_page_row():
    text = _console_text()
    fn = _extract_function(text, "renderIngestCard")
    assert "ingestCardSectionsCrumb" in fn
    assert "pageIdx === 0" in fn, (
        "Section count must render once per Source (first page row), not once per page"
    )


def test_render_ingest_card_uncarried_and_enriched_crumbs_are_conditional():
    text = _console_text()
    fn = _extract_function(text, "renderIngestCard")
    assert "if (r.uncarried_chars)" in fn, (
        "uncarried_chars crumb must be conditional — anomaly signal, not routine noise"
    )
    assert "if (r.enriched_chars)" in fn, (
        "enriched_chars crumb must be conditional — anomaly signal, not routine noise"
    )
    assert "ingestCardUncarriedCharsCrumb" in fn
    assert "ingestCardEnrichedCharsCrumb" in fn


def test_render_ingest_card_skipped_rows_also_show_sections_count():
    text = _console_text()
    fn = _extract_function(text, "renderIngestCard")
    skipped_block = fn.split("skipped.forEach(function(r) {", 1)[1]
    assert "ingestCardSectionsCrumb" in skipped_block


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
