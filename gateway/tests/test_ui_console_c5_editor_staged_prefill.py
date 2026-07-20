"""Structural tests for the C5 in-modal Source editor's staged-prefill rider
(issue #634, riding on ADR-0043's openC5SourceEditor).

Following the pattern in ``test_ui_console_c5_two_view_modal.py``, these
tests inspect the production ``gateway/static/console.html`` file's text —
no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Bug fixed: ``openC5SourceEditor`` always fetched the on-disk file fresh and
started the Stage button enabled. Re-opening the editor after staging a
correction, then clicking Stage again without editing, silently replaced
the staged text with the unmodified original (an accidental-replace path
ADR-0043 decision 2's "re-staging replaces" never intended to create).

Covers:
- A staged correction for the section's ``source_path`` is preferred over
  fetching ``/read/file`` (checked before the fetch call).
- The fetch fallback still runs when no staged correction exists for this
  Source.
- The Stage button starts disabled in both paths, arming only on the
  textarea's first ``input`` event.
- A visible "showing staged edit" badge renders only on the staged-prefill
  path, using the existing ``c5-staged-badge`` class; the string is a new
  bilingual LINT_CHROME key.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling test files' helper of the same name)."""
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


# ---------------------------------------------------------------------------
# Prefill from an already-staged correction, checked before the fetch
# ---------------------------------------------------------------------------


def test_editor_looks_up_a_staged_target_before_fetching():
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    staged_pos = fn.find("pendingC5FixSourceTargets.filter(")
    fetch_pos = fn.find('fetch("/read/file?"')
    assert staged_pos != -1, "must check pendingC5FixSourceTargets for a staged match"
    assert fetch_pos != -1, "the fetch fallback must still exist"
    assert staged_pos < fetch_pos, (
        "the staged-target lookup must run BEFORE the fetch, so a staged "
        "correction is never clobbered by an on-disk fetch"
    )


def test_staged_match_keys_on_source_path_and_requires_content():
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert "t.sourcePath === section.source_path && t.content != null" in fn


def test_staged_target_short_circuits_and_never_fetches():
    """The staged-prefill branch must return before reaching the fetch call
    — mirrors ADR-0043's own invariant that staging/prefill never issues a
    request; only an explicit upload run does."""
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    m = re.search(
        r"if \(stagedTarget\) \{\s*buildEditor\(stagedTarget\.content, true\);\s*return;\s*\}",
        fn,
    )
    assert m is not None, "staged branch must call buildEditor(..., true) then return"
    assert fn.index(m.group(0)) < fn.find('fetch("/read/file?"')


def test_fetch_fallback_still_loads_the_whole_file_verbatim():
    """Unchanged ADR-0043 invariant: when no staged correction exists, the
    editor still fetches the WHOLE file via /read/file, section.source_path
    verbatim (no client-side path reconstruction, §12.5)."""
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert '"/read/file?"' in fn
    assert "section.source_path" in fn
    assert '"docs/" + ' not in fn
    assert "buildEditor(data.content, false)" in fn


# ---------------------------------------------------------------------------
# Stage button starts disabled; arms only on the first real edit
# ---------------------------------------------------------------------------


def test_stage_button_starts_disabled():
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert re.search(r'text:\s*chrome\.c5EditSourceStage,\s*disabled:\s*""\s*\}', fn), (
        "Stage must be created disabled — a bare re-open + click must not write anything"
    )


def test_stage_button_arms_on_first_input_event():
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert 'textarea.addEventListener("input", function() {' in fn
    assert "stageBtn.disabled = false;" in fn


def test_stage_disabled_at_creation_applies_to_both_the_staged_and_fetch_paths():
    """buildEditor is the single shared constructor for both the staged-
    prefill and fetch-fallback content — the disabled-until-edit behaviour
    must not be duplicated (and therefore cannot drift) between the two."""
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert fn.count("function buildEditor(") == 1
    assert fn.count('disabled: "" }') == 1
    assert "buildEditor(stagedTarget.content, true)" in fn
    assert "buildEditor(data.content, false)" in fn


# ---------------------------------------------------------------------------
# Visible "showing staged edit" indicator on the staged-prefill path only
# ---------------------------------------------------------------------------


def test_staged_badge_rendered_only_on_the_staged_path():
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert (
        'showingStaged ? el("span", { class: "c5-staged-badge", '
        "text: chrome.c5EditSourceShowingStaged }) : null" in fn
    )


def test_staged_prefill_indicator_chrome_defined_bilingually():
    text = _console_text()
    assert len(re.findall(r'c5EditSourceShowingStaged:\s*"[^"]+"', text)) == 2, (
        "c5EditSourceShowingStaged must be defined in BOTH language blocks"
    )
    assert "顯示暫存修正版本" in text


def test_staged_badge_reuses_the_existing_badge_css_class():
    """AC: reuse the c5-staged-badge string/pattern rather than inventing a
    new visual treatment — the class already exists for the batch panel's
    staged marker (ADR-0043 decision 4)."""
    text = _console_text()
    assert re.search(r"\.c5-staged-badge\s*\{[^}]*\}", text), (
        "c5-staged-badge CSS rule must already exist (reused, not redefined)"
    )
    fn = _extract_function(text, "openC5SourceEditor")
    assert 'class: "c5-staged-badge"' in fn


# ---------------------------------------------------------------------------
# Regression: the editor still never writes anything itself (ADR-0043 §2)
# ---------------------------------------------------------------------------


def test_editor_still_only_stages_client_side_never_writes():
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert "addC5FixSourceTarget(section.source_path, textarea.value)" in fn
    assert "adminFetch(" not in fn
    assert '"/upload"' not in fn


def test_no_inner_html_regression():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
