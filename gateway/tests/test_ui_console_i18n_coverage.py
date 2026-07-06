"""Structural tests for whole-console i18n coverage (issue #498).

Before this slice, ``LINT_CHROME`` only covered the Lint card's chrome
(issue #365, ADR-0023 tier-A S5). This slice extends bilingual coverage to
the WHOLE Operator Console — pipeline stepper, Upload/Import/Ingest/Index
result cards, RAG/Hybrid tracks, Curation Queue, draft/refile/flag cards,
the Browse file browser + viewer, and the static banners — while keeping
``LINT_CHROME`` as the single dictionary (issue #498 AC: reuse the existing
name, new keys namespaced by region prefix).

Following the pattern in ``test_ui_console_lint_bilingual.py``, these tests
inspect the production ``gateway/static/console.html`` file's text — no DOM,
no fetch, no browser, no OPENAI_API_KEY (fully hermetic, §6.3 / §12.7).

Covers:
- Parity guard: every key in ``LINT_CHROME.en`` has a matching key in
  ``LINT_CHROME.zh`` and vice versa — a structural safety net so a future
  region can add English chrome without silently forgetting the zh half.
- Spot checks: Curation Queue header/empty state, a step description, the
  edit-rejected message, and the C8/C9/C10 zh headers are real (non-English)
  Chinese text.
- The ``bindText`` / ``I18N_BOUND`` boot-time registry exists and
  ``applyConsoleLang`` resets every bound element on toggle.
- The header language toggle handler also re-renders the Curation Queue
  (not just the Lint card), guarded the same way the Lint-card re-render is
  (skipped while a remediation is in flight, issue #364).
- A handful of English literals used as regression anchors elsewhere in the
  test suite are still present verbatim (e.g. "Promotion Candidates (C8)",
  "No curation items", "Drag and drop files here").

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). Visual toggle-and-look verification is out of scope for unit tests.
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
    name — robust to nested braces inside the body, unlike a lazy regex)."""
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


def _extract_object_keys(block: str) -> set[str]:
    """Extract bare-identifier object keys (``key: ...``) at the top level of
    a ``{ ... }`` object-literal body. Matches the ``key:`` shape LINT_CHROME
    uses throughout (no quoted keys) — a simple line-anchored regex is robust
    here because every entry in the dictionary is on its own line. Values are
    almost always double-quoted, but a handful (e.g. draftPromoteSuccess)
    single-quote the JS string because the text itself contains a double
    quote — match either delimiter."""
    return set(re.findall(r'^\s*([A-Za-z_][A-Za-z0-9_]*):\s*[\'"]', block, re.MULTILINE))


def _lint_chrome_en_zh() -> tuple[str, str]:
    """Return the (en_block, zh_block) source text of LINT_CHROME's two
    language objects, isolated by brace-depth scanning from the
    ``var LINT_CHROME = {`` declaration (robust to the large number of
    nested braces/comments inside each language block — a lazy regex like
    the sibling test files use for smaller objects breaks here)."""
    text = _console_text()
    marker = "var LINT_CHROME = {"
    start = text.index(marker)
    body_start = start + len(marker) - 1  # index of the opening brace
    depth = 0
    body_end = None
    for i in range(body_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body_end = i
                break
    assert body_end is not None, "LINT_CHROME object literal is unterminated"
    full_body = text[body_start + 1 : body_end]  # strips the outer { }

    # Now split into the en: { ... } and zh: { ... } sub-objects the same way.
    def _sub_object(label: str) -> str:
        sub_marker = f"{label}: {{"
        sub_start = full_body.index(sub_marker)
        sub_body_start = sub_start + len(sub_marker) - 1
        sub_depth = 0
        for j in range(sub_body_start, len(full_body)):
            if full_body[j] == "{":
                sub_depth += 1
            elif full_body[j] == "}":
                sub_depth -= 1
                if sub_depth == 0:
                    return full_body[sub_body_start + 1 : j]
        raise AssertionError(f"LINT_CHROME.{label} object literal is unterminated")

    return _sub_object("en"), _sub_object("zh")


# ---------------------------------------------------------------------------
# Parity guard: en/zh key sets match exactly
# ---------------------------------------------------------------------------


def test_lint_chrome_en_and_zh_key_sets_are_identical():
    en_block, zh_block = _lint_chrome_en_zh()
    en_keys = _extract_object_keys(en_block)
    zh_keys = _extract_object_keys(zh_block)

    assert en_keys, "LINT_CHROME.en must define at least one key"
    assert zh_keys, "LINT_CHROME.zh must define at least one key"

    missing_in_zh = en_keys - zh_keys
    missing_in_en = zh_keys - en_keys
    assert not missing_in_zh, f"keys defined in en but missing from zh: {sorted(missing_in_zh)}"
    assert not missing_in_en, f"keys defined in zh but missing from en: {sorted(missing_in_en)}"


def test_lint_chrome_has_grown_well_beyond_the_lint_card(request):
    """Sanity check that this slice actually widened LINT_CHROME (issue #498)
    rather than the parity guard above trivially passing on an empty/small
    dictionary. #365 shipped ~35 keys for the Lint card alone; #498 covers
    the whole console, so the key count should be substantially larger."""
    en_block, _ = _lint_chrome_en_zh()
    en_keys = _extract_object_keys(en_block)
    assert len(en_keys) >= 100, (
        f"expected LINT_CHROME.en to cover the whole console (100+ keys), "
        f"found only {len(en_keys)} — issue #498 scope regression?"
    )


# ---------------------------------------------------------------------------
# Spot checks: representative zh strings exist and are real Chinese text
# ---------------------------------------------------------------------------


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def test_curation_queue_header_and_empty_state_are_bilingual():
    _, zh_block = _lint_chrome_en_zh()
    header_match = re.search(r'queueHeaderLabel:\s*"([^"]+)"', zh_block)
    empty_match = re.search(r'queueEmpty:\s*"([^"]+)"', zh_block)
    assert header_match is not None, "LINT_CHROME.zh.queueHeaderLabel missing"
    assert empty_match is not None, "LINT_CHROME.zh.queueEmpty missing"
    assert _has_cjk(empty_match.group(1)), "zh queueEmpty must contain real Chinese text"


def test_step_description_is_bilingual():
    _, zh_block = _lint_chrome_en_zh()
    match = re.search(r'stepIngestDesc:\s*"([^"]+)"', zh_block)
    assert match is not None, "LINT_CHROME.zh.stepIngestDesc missing"
    assert _has_cjk(match.group(1)), "zh stepIngestDesc must contain real Chinese text"


def test_edit_rejected_message_is_bilingual():
    _, zh_block = _lint_chrome_en_zh()
    match = re.search(r'editRejectedHead:\s*"([^"]+)"', zh_block)
    assert match is not None, "LINT_CHROME.zh.editRejectedHead missing"
    assert _has_cjk(match.group(1)), "zh editRejectedHead must contain real Chinese text"


def test_c8_c9_c10_queue_headers_align_with_lint_check_label_zh_vocabulary():
    """The Curation Queue's own C8/C9/C10 section headers must use the SAME
    zh vocabulary as LINT_CHECK_LABEL_ZH (issue #498 design decision 7):
    Promotion Candidates -> 待升級草稿, Stale Filed Answers -> 資料過舊的已歸檔回答,
    Invalid QA Schema -> QA 格式錯誤."""
    _, zh_block = _lint_chrome_en_zh()
    c8 = re.search(r'queueC8Header:\s*"([^"]+)"', zh_block)
    c9 = re.search(r'queueC9Header:\s*"([^"]+)"', zh_block)
    c10 = re.search(r'queueC10Header:\s*"([^"]+)"', zh_block)
    assert c8 is not None and "待升級" in c8.group(1) and "C8" in c8.group(1)
    assert c9 is not None and "過舊" in c9.group(1) and "C9" in c9.group(1)
    assert c10 is not None and "格式錯誤" in c10.group(1) and "C10" in c10.group(1)


# ---------------------------------------------------------------------------
# bindText / I18N_BOUND boot-time registry
# ---------------------------------------------------------------------------


def test_i18n_bound_registry_and_bind_text_helper_exist():
    text = _console_text()
    assert "var I18N_BOUND = [];" in text, "console.html must define the I18N_BOUND registry (issue #498)"
    fn = _extract_function(text, "bindText")
    assert "I18N_BOUND.push(" in fn
    assert "elem.textContent = LINT_CHROME[consoleLang][key];" in fn
    assert "return elem;" in fn


def test_apply_console_lang_resets_every_bound_element():
    text = _console_text()
    fn = _extract_function(text, "applyConsoleLang")
    assert "I18N_BOUND.forEach(function(entry)" in fn
    assert "entry.el.textContent = LINT_CHROME[consoleLang][entry.key];" in fn


def test_bind_text_is_used_at_more_than_a_handful_of_boot_time_sites():
    """Guards against a token bindText call that satisfies the letter of the
    registry pattern without actually wiring up the boot-time chrome this
    issue targets (pipeline stepper, upload drop zone, RAG/Hybrid panels,
    Browse label, banners)."""
    text = _console_text()
    call_count = text.count("bindText(")
    # 1 definition + step track/panel labels+descs (STEP_DEFS has 5 steps,
    # each contributing at least a track pill + panel title + panel desc) +
    # upload drop-zone strings + RAG/Hybrid panel/track chrome + banners +
    # Browse label + Run/Deep-audit buttons — comfortably more than 20.
    assert call_count >= 20, f"expected 20+ bindText call sites, found {call_count - 1}"


# ---------------------------------------------------------------------------
# Toggle handler also re-renders the Curation Queue (design decision #4)
# ---------------------------------------------------------------------------


def test_toggle_handler_also_rerenders_curation_queue():
    text = _console_text()
    init_fn = re.search(r"function initConsoleLangToggle\(\) \{.*?\n\}\)\(\);", text, re.DOTALL)
    assert init_fn is not None
    body = init_fn.group(0)
    assert "lintResultEl.replaceChildren(renderLintCard(lastLintData, lintResultEl));" in body, (
        "the pre-existing Lint-card re-render (issue #365) must still be present"
    )
    queue_rerender = re.search(
        r"if \(lastLintData && !remediationInFlight\) \{\s*renderCurationQueue\(", body
    )
    assert queue_rerender is not None, (
        "the toggle handler must also re-render the Curation Queue via "
        "renderCurationQueue(...), guarded the same way as the Lint card "
        "(skipped while remediationInFlight, issue #498 design decision #4)"
    )


# ---------------------------------------------------------------------------
# Regression anchors: representative English literals other tests / users
# rely on are still present verbatim after this slice's refactor
# ---------------------------------------------------------------------------


def test_representative_english_literals_still_present():
    text = _console_text()
    for literal in (
        "Promotion Candidates (C8)",
        "No curation items",
        "Drag and drop files here",
    ):
        assert literal in text, f"expected English literal {literal!r} to still be present verbatim"


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
