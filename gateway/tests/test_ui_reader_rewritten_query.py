"""Structural tests for surfacing the rewritten query in the reader UI (issue #579).

Following the established pattern (``test_ui_reader_feedback.py``,
``test_ui_reader_budget_banner.py``), these tests inspect the production
``gateway/static/index.html`` file's text — no DOM, no fetch, no browser, no
OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- ``dispatch`` forwards the full ``sources`` event payload to ``onSources``
  (not just the ``sources`` array) so it has access to ``rewritten_query``.
- ``onSources`` renders ``data.rewritten_query`` via ``textContent`` only
  (§12.4), gated on presence — turn 1 (no ``rewritten_query`` key) renders
  nothing extra.
- ``CHROME.en`` / ``CHROME.zh`` define the caption's bilingual copy (i18n
  coverage); the zh value is real Chinese text.
- No business logic: the client renders the server-decided string verbatim,
  it does not reconstruct or guess at the rewrite (§12.5).
"""

from __future__ import annotations

import re
from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace matching."""
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
# dispatch forwards the full sources payload
# ---------------------------------------------------------------------------


def test_dispatch_passes_full_data_to_on_sources():
    text = _ui_text()
    fn = _extract_function(text, "dispatch")
    assert "onSources(data)" in fn, (
        f"dispatch must forward the full sources event payload (not just "
        f".sources) so onSources can read rewritten_query, got:\n{fn}"
    )
    assert "onSources(data.sources)" not in fn


# ---------------------------------------------------------------------------
# onSources renders rewritten_query, gated + via textContent
# ---------------------------------------------------------------------------


def test_on_sources_reads_sources_from_data():
    text = _ui_text()
    fn = _extract_function(text, "onSources")
    assert "var sources = data.sources" in fn


def test_on_sources_renders_rewritten_query_gated_and_via_textcontent():
    text = _ui_text()
    fn = _extract_function(text, "onSources")
    assert "data.rewritten_query" in fn, "onSources must read data.rewritten_query"
    assert 'class: "rewritten-query"' in fn
    assert 'text: t("searchedForLabel") + data.rewritten_query' in fn, (
        "rewritten_query must render via el()'s text prop (textContent, §12.4), never innerHTML"
    )
    # Gated: the rewritten-query block construction must be inside a
    # truthiness check on data.rewritten_query so turn 1 (key absent) renders
    # nothing extra.
    gate_idx = fn.index("if (data.rewritten_query)")
    block_idx = fn.index('class: "rewritten-query"')
    assert gate_idx < block_idx, "rendering must be gated on data.rewritten_query being present"


# ---------------------------------------------------------------------------
# Bilingual chrome (i18n coverage)
# ---------------------------------------------------------------------------


def test_chrome_defines_searched_for_label_in_both_languages():
    text = _ui_text()
    en_block = text.split("var CHROME = {", 1)[1].split("zh: {", 1)[0]
    zh_block = text.split("zh: {", 1)[1].split("};", 1)[0]
    assert "searchedForLabel:" in en_block, "CHROME.en must define searchedForLabel"
    assert "searchedForLabel:" in zh_block, "CHROME.zh must define searchedForLabel"

    zh_label = re.search(r'searchedForLabel:\s*"([^"]+)"', zh_block)
    assert zh_label is not None and _has_cjk(zh_label.group(1))

    en_label = re.search(r'searchedForLabel:\s*"([^"]+)"', en_block)
    assert en_label is not None and "search" in en_label.group(1).lower()


# ---------------------------------------------------------------------------
# No innerHTML anywhere near the new render site (§12.4 belt-and-suspenders)
# ---------------------------------------------------------------------------


def test_on_sources_never_uses_innerhtml():
    text = _ui_text()
    fn = _extract_function(text, "onSources")
    assert "innerHTML" not in fn
