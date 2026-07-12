"""Structural tests for the Operator Console Source lifecycle surface
(issue #606, ADR-0041): per-file-row Rename/Retire actions on the EXISTING
Browse rail, and a dedicated Source Trash section with Restore.

Following the pattern in ``test_ui_console_orphan_delete.py``, these tests
inspect the production ``gateway/static/console.html`` file's text — no DOM,
no fetch, no browser, no OPENAI_API_KEY (fully hermetic, §6.3 / §12.7).

Covers:
- ``sourceRelpathFromEntry`` strips the docs/ root prefix and returns null
  for directories / non-docs files (only Sources are lifecycle-governed).
- Browse rows under docs/ wire Rename (outline chrome) + Retire (the same
  red ``discard-btn`` chrome the C11 orphan-delete trigger uses), both
  stopPropagation-guarded so a button click never also opens the file.
- The Retire confirm dialog names the operation + target, loads the
  server-computed impact preview before enabling Confirm, posts
  ``POST /sources/retire`` via ``adminFetch``, and on success refreshes both
  the Browse listing and the Source Trash section.
- The Rename dialog posts ``POST /sources/rename`` via ``adminFetch`` and
  refuses locally only on a blank new name.
- The Source Trash section fetches ``GET /sources/trash``, renders a Restore
  button per entry that posts ``POST /sources/restore`` via ``adminFetch``,
  and reuses ``openFile`` over the read-only ``.trash`` root for content
  preview.
- Both the confirm-preview GET and the trash-listing GET use plain
  ``fetch`` (never ``adminFetch``) — they are ungated read paths, mirroring
  ``GET /read/*``.
- No ``innerHTML`` assignment is introduced (§12.4).
- LINT_CHROME en/zh parity for the new keys (belt-and-suspenders on top of
  the whole-file parity guard in ``test_ui_console_i18n_coverage.py``).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the click -> confirm -> retire -> refresh loop is
verified manually per §12.7 (visual rendering is out of scope for unit
tests).
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


# ---------------------------------------------------------------------------
# sourceRelpathFromEntry — docs/ files only, prefix stripped
# ---------------------------------------------------------------------------


def test_source_relpath_from_entry_returns_null_for_directories_and_non_docs():
    text = _console_text()
    fn = _extract_function(text, "sourceRelpathFromEntry")
    assert "entry.is_dir" in fn
    assert 'entry.relpath.indexOf("docs/") !== 0' in fn
    assert "return null" in fn


def test_source_relpath_from_entry_strips_docs_prefix():
    text = _console_text()
    fn = _extract_function(text, "sourceRelpathFromEntry")
    assert 'entry.relpath.slice("docs/".length)' in fn


# ---------------------------------------------------------------------------
# Browse rail: per-file-row Rename/Retire actions
# ---------------------------------------------------------------------------


def test_render_entries_wires_rename_and_retire_for_docs_files():
    text = _console_text()
    fn = _extract_function(text, "renderEntries")
    assert "var sourceRelpath = sourceRelpathFromEntry(entry);" in fn
    assert "openRenameModal(sourceRelpath)" in fn
    assert "openRetireConfirm(sourceRelpath)" in fn


def test_rename_and_retire_buttons_use_distinct_button_chrome():
    text = _console_text()
    fn = _extract_function(text, "renderEntries")
    rename_btn = re.search(r'var renameBtn = el\("button", \{([^}]*)\}', fn)
    retire_btn = re.search(r'var retireBtn = el\("button", \{([^}]*)\}', fn)
    assert rename_btn is not None and '"fix-source-btn"' in rename_btn.group(1)
    assert retire_btn is not None and '"discard-btn"' in retire_btn.group(1)


def test_action_buttons_stop_propagation_before_opening_their_modal():
    text = _console_text()
    fn = _extract_function(text, "renderEntries")
    rename_click = re.search(
        r"renameBtn\.addEventListener\(\"click\", function\(e\) \{(.*?)\}\);", fn, re.DOTALL
    )
    retire_click = re.search(
        r"retireBtn\.addEventListener\(\"click\", function\(e\) \{(.*?)\}\);", fn, re.DOTALL
    )
    assert rename_click is not None
    assert "e.stopPropagation();" in rename_click.group(1)
    assert retire_click is not None
    assert "e.stopPropagation();" in retire_click.group(1)


def test_lang_toggle_rerenders_browse_entries_and_source_trash_from_cache():
    text = _console_text()
    init_fn = re.search(r"function initConsoleLangToggle\(\) \{.*?\n\}\)\(\);", text, re.DOTALL)
    assert init_fn is not None
    body = init_fn.group(0)
    assert "if (lastBrowserEntries && !sourceLifecycleInFlight) {" in body
    assert "renderEntries(lastBrowserEntries);" in body
    assert "if (lastSourceTrashEntries && !sourceLifecycleInFlight) {" in body
    assert "renderSourceTrash(lastSourceTrashEntries);" in body


# ---------------------------------------------------------------------------
# Retire confirm dialog
# ---------------------------------------------------------------------------


def test_retire_confirm_names_operation_and_target_before_impact_loads():
    text = _console_text()
    fn = _extract_function(text, "openRetireConfirm")
    assert "chrome.sourceRetireModalTitle" in fn
    assert '": docs/" + relpath' in fn
    assert "chrome.sourceRetireModalBodyPrefix + relpath + chrome.sourceRetireModalBodySuffix" in fn


def test_retire_confirm_button_starts_disabled_until_impact_preview_resolves():
    text = _console_text()
    fn = _extract_function(text, "openRetireConfirm")
    confirm_decl = re.search(r'var confirmBtn = el\("button", \{([^}]*)\}', fn)
    assert confirm_decl is not None
    assert 'disabled: ""' in confirm_decl.group(1)
    assert "confirmBtn.disabled = false;" in fn


def test_retire_confirm_loads_impact_preview_via_plain_fetch_not_admin_fetch():
    text = _console_text()
    fn = _extract_function(text, "sourceImpactRequest")
    assert '"/sources/" + encodeSourceRelpath(relpath) + "/impact"' in fn
    assert "fetch(url)" in fn
    assert "adminFetch" not in fn


def test_retire_confirm_posts_retire_via_admin_fetch():
    text = _console_text()
    fn = _extract_function(text, "retireSourceRequest")
    assert 'adminFetch("/sources/retire"' in fn
    assert 'method: "POST"' in fn
    assert "JSON.stringify({ relpath: relpath })" in fn


def test_retire_confirm_success_refreshes_browse_and_source_trash():
    text = _console_text()
    fn = _extract_function(text, "openRetireConfirm")
    success_branch = re.search(
        r"retireSourceRequest\(relpath\)\s*\.then\(function\(\)\s*\{(.*?)\n      \}\)",
        fn,
        re.DOTALL,
    )
    assert success_branch is not None
    body = success_branch.group(1)
    assert "closeModal();" in body
    assert "navigateTo(browserPath);" in body
    assert "loadSourceTrash();" in body


def test_retire_confirm_click_guards_against_double_submit():
    text = _console_text()
    fn = _extract_function(text, "openRetireConfirm")
    click_fn = re.search(r"confirmBtn\.addEventListener\(\"click\".*?\n  \}\);", fn, re.DOTALL)
    assert click_fn is not None
    assert "if (confirmBtn.disabled || sourceLifecycleInFlight) return;" in click_fn.group(0)


# ---------------------------------------------------------------------------
# Rename dialog
# ---------------------------------------------------------------------------


def test_rename_modal_posts_rename_via_admin_fetch():
    text = _console_text()
    fn = _extract_function(text, "renameSourceRequest")
    assert 'adminFetch("/sources/rename"' in fn
    assert 'method: "POST"' in fn
    assert "JSON.stringify({ relpath: relpath, new_basename: newBasename })" in fn


def test_rename_modal_refuses_blank_input_client_side_before_any_request():
    text = _console_text()
    fn = _extract_function(text, "openRenameModal")
    click_fn = re.search(r"confirmBtn\.addEventListener\(\"click\".*?\n  \}\);", fn, re.DOTALL)
    assert click_fn is not None
    body = click_fn.group(0)
    blank_check = re.search(r"if \(!newBasename\) \{(.*?)\n    \}", body, re.DOTALL)
    assert blank_check is not None
    assert "chrome.sourceRenameBlankError" in blank_check.group(1)
    assert "return;" in blank_check.group(1)
    # The blank check must precede the request call.
    assert body.index("if (!newBasename)") < body.index("renameSourceRequest(relpath, newBasename)")


def test_rename_modal_success_refreshes_browse_listing():
    text = _console_text()
    fn = _extract_function(text, "openRenameModal")
    success_branch = re.search(
        r"renameSourceRequest\(relpath, newBasename\)\s*\.then\(function\(\)\s*\{(.*?)\n      \}\)",
        fn,
        re.DOTALL,
    )
    assert success_branch is not None
    body = success_branch.group(1)
    assert "closeModal();" in body
    assert "navigateTo(browserPath);" in body


def test_rename_modal_click_guards_against_double_submit():
    text = _console_text()
    fn = _extract_function(text, "openRenameModal")
    click_fn = re.search(r"confirmBtn\.addEventListener\(\"click\".*?\n  \}\);", fn, re.DOTALL)
    assert click_fn is not None
    assert "if (confirmBtn.disabled || sourceLifecycleInFlight) return;" in click_fn.group(0)


# ---------------------------------------------------------------------------
# Source Trash section
# ---------------------------------------------------------------------------


def test_source_trash_list_uses_plain_fetch_not_admin_fetch():
    text = _console_text()
    fn = _extract_function(text, "sourceTrashListRequest")
    assert 'fetch("/sources/trash")' in fn
    assert "adminFetch" not in fn


def test_render_source_trash_shows_empty_state():
    text = _console_text()
    fn = _extract_function(text, "renderSourceTrash")
    assert "if (entries.length === 0) {" in fn
    assert "chrome.sourceTrashEmpty" in fn


def test_render_source_trash_restore_button_posts_restore_via_admin_fetch():
    text = _console_text()
    fn = _extract_function(text, "renderSourceTrash")
    assert "restoreSourceRequest(entry.timestamp, entry.relpath)" in fn
    restore_fn = _extract_function(text, "restoreSourceRequest")
    assert 'adminFetch("/sources/restore"' in restore_fn
    assert 'method: "POST"' in restore_fn
    assert "JSON.stringify({ timestamp: timestamp, relpath: relpath })" in restore_fn


def test_render_source_trash_restore_click_guards_against_double_submit():
    text = _console_text()
    fn = _extract_function(text, "renderSourceTrash")
    click_fn = re.search(
        r"restoreBtn\.addEventListener\(\"click\", function\(e\) \{(.*?)\n    \}\);", fn, re.DOTALL
    )
    assert click_fn is not None
    assert "if (restoreBtn.disabled || sourceLifecycleInFlight) return;" in click_fn.group(1)


def test_render_source_trash_row_click_opens_file_over_trash_root():
    text = _console_text()
    fn = _extract_function(text, "renderSourceTrash")
    assert 'var trashPath = ".trash/" + entry.timestamp + "/docs/" + entry.relpath;' in fn
    assert "openFile(trashPath, entry.relpath," in fn


def test_load_source_trash_caches_entries_and_renders():
    text = _console_text()
    fn = _extract_function(text, "loadSourceTrash")
    assert "sourceTrashListRequest()" in fn
    assert "lastSourceTrashEntries = data.entries || [];" in fn
    assert "renderSourceTrash(lastSourceTrashEntries);" in fn


def test_boot_calls_load_source_trash():
    text = _console_text()
    assert 'navigateTo("");\nloadSourceTrash();' in text


# ---------------------------------------------------------------------------
# LINT_CHROME en/zh parity for the new Source lifecycle keys
# ---------------------------------------------------------------------------


_SOURCE_LIFECYCLE_KEYS = (
    "sourceRenameBtn",
    "sourceRetireBtn",
    "sourceRetireModalTitle",
    "sourceRetireModalBodyPrefix",
    "sourceRetireModalBodySuffix",
    "sourceRetireImpactFullLabel",
    "sourceRetireImpactPartialLabel",
    "sourceRetireImpactNone",
    "sourceRetireImpactLoadFailedPrefix",
    "sourceRetireConfirmBtn",
    "sourceRetireCancelBtn",
    "sourceRetireWorking",
    "sourceRenameModalTitlePrefix",
    "sourceRenameInputLabel",
    "sourceRenameConfirmBtn",
    "sourceRenameCancelBtn",
    "sourceRenameWorking",
    "sourceRenameBlankError",
    "sourceTrashSectionLabel",
    "sourceTrashLoadFailedPrefix",
    "sourceTrashEmpty",
    "sourceTrashRestoreBtn",
    "sourceTrashRestoreWorking",
    "sourceTrashRestoreFailedPrefix",
)


def test_every_source_lifecycle_key_defined_in_both_languages():
    text = _console_text()
    for key in _SOURCE_LIFECYCLE_KEYS:
        assert re.search(rf"\b{key}:\s*[\"']", text), f"LINT_CHROME key {key!r} not found anywhere"
        assert len(re.findall(rf"\b{key}:\s*[\"']", text)) == 2, (
            f"expected exactly 2 definitions of {key!r} (en + zh), "
            f"found {len(re.findall(rf'{key}:', text))}"
        )


def test_source_trash_zh_label_is_real_chinese_text():
    text = _console_text()
    match = re.search(r'sourceTrashSectionLabel:\s*"([^"]+)"', text)
    assert match is not None
    zh_candidates = [m for m in re.findall(r'sourceTrashSectionLabel:\s*"([^"]+)"', text)]
    assert any(any("一" <= ch <= "鿿" for ch in s) for s in zh_candidates), (
        "expected a real Chinese sourceTrashSectionLabel among the two definitions"
    )


# ---------------------------------------------------------------------------
# No innerHTML / textContent-only discipline (§12.4)
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_in_source_lifecycle_code():
    text = _console_text()
    for name in (
        "renderEntries",
        "openRetireConfirm",
        "openRenameModal",
        "loadSourceTrash",
        "renderSourceTrash",
    ):
        body = _extract_function(text, name)
        assert ".innerHTML =" not in body and ".innerHTML=" not in body


def test_no_inner_html_assignment_still_holds_console_wide():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
