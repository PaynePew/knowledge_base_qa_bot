"""Structural tests for the Operator Console C5 stale-relint preservation
(issue #489).

Following the pattern in ``test_ui_console_lint_remediation_batch.py`` /
``test_ui_console_lint_fast_default.py``, these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of issue #489:

- After ANY remediation, ``runLintRemediation`` (and every other
  Direct/Routed/Authored remediation flow) re-lints via
  ``POST /wiki/lint?include_c5=false`` — a fast, LLM-free pass that always
  returns ``page_pairs: []`` because C5 is never judged on that path. Before
  this slice, ``renderLintCard`` re-rendered that empty result verbatim,
  making a previously-populated C5 Contradiction section disappear — reading
  as "contradictions resolved" when nothing was re-judged.
- ``renderLintCard`` now falls back to the last genuinely-judged
  ``page_pairs`` (``lastDeepC5Pairs``) when the current response's
  ``page_pairs`` is empty and a non-empty baseline exists, and renders that
  fallback with a bilingual stale marker instead of dropping the section.
- The Deep audit control (the ONLY call site that requests
  ``include_c5=true``) marks the upcoming render as a fresh judgment via a
  one-shot module flag (``pendingDeepC5Audit``) — NOT via a 3rd
  ``renderLintCard`` argument, so the existing pinned 2-arg call
  (``test_ui_console_lint_fast_default.py``'s
  ``test_run_and_deep_audit_share_the_same_completion_chain``) stays intact.
- The fallback never mutates ``data.findings`` (the fetch response object,
  also stashed in ``lastLintData`` and handed to ``renderCurationQueue``) —
  it renders from a shallow copy.
- The top badge's finding count folds the stale C5 pairs back in, so it does
  not undercount against what the card body actually renders.
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the actual remediation -> fast-relint -> stale-C5
render loop is verified manually per §12.7 (visual rendering is out of scope
for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors test_ui_console_lint_remediation_batch.py's helper of
    the same name), so assertions don't over/under-match past the
    function's own closing brace."""
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


def _extract_lint_step_run_body(text: str) -> str:
    """Mirrors test_ui_console_lint_fast_default.py's helper of the same
    name, so this file stays consistent with that pinned structural test."""
    match = re.search(
        r'id:\s*"lint",.*?run:\s*function\(resultEl, setStepBusy\)\s*\{(.*?)\n    \},',
        text,
        re.DOTALL,
    )
    assert match is not None, "could not locate the lint STEP_DEFS run() body"
    return match.group(1)


def _extract_lint_step_deep_audit_body(text: str) -> str:
    match = re.search(
        r'id:\s*"lint",.*?deepAudit:\s*function\(resultEl, setStepBusy\)\s*\{(.*?)\n    \},',
        text,
        re.DOTALL,
    )
    assert match is not None, "lint STEP_DEFS entry must define deepAudit()"
    return match.group(1)


# ---------------------------------------------------------------------------
# renderLintCard: fallback to the last judged page_pairs, flagged stale
# ---------------------------------------------------------------------------


def test_render_lint_card_falls_back_to_last_deep_c5_pairs_when_empty():
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    assert "lastDeepC5Pairs && lastDeepC5Pairs.length > 0" in render_fn, (
        "renderLintCard must only fall back when a non-empty judged baseline exists (issue #489)"
    )
    assert "findingsWithStaleC5.page_pairs = lastDeepC5Pairs;" in render_fn
    assert "c5Stale = true;" in render_fn


def test_render_lint_card_fallback_does_not_mutate_the_response_object():
    """The fallback must build a shallow copy, never assign directly onto
    ``findings`` (== ``data.findings``, the fetch response also stashed in
    lastLintData / handed to renderCurationQueue by every caller)."""
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    assert "findings.page_pairs = lastDeepC5Pairs;" not in render_fn, (
        "must not assign onto the original findings/data.findings object directly"
    )
    assert "var findingsWithStaleC5 = {};" in render_fn
    assert "findingsWithStaleC5[fKey] = findings[fKey];" in render_fn


def test_render_lint_card_updates_baseline_only_on_a_genuine_deep_judgment():
    """A genuine include_c5=true response (c5Deep true) must become the new
    baseline unconditionally — including an honest empty result, which must
    NOT be treated as 'never judged'."""
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    match = re.search(
        r"if \(c5Deep\) \{\s*lastDeepC5Pairs = findings\.page_pairs \|\| \[\];\s*\}",
        render_fn,
    )
    assert match is not None, (
        "renderLintCard must reset lastDeepC5Pairs from findings.page_pairs "
        "whenever c5Deep is true, even when that result is empty"
    )


# ---------------------------------------------------------------------------
# One-shot pendingDeepC5Audit signal — not a renderLintCard argument
# ---------------------------------------------------------------------------


def test_pending_deep_c5_audit_flag_declared_at_module_level():
    text = _console_text()
    assert re.search(r"^var pendingDeepC5Audit = false;", text, re.MULTILINE), (
        "pendingDeepC5Audit must be a module-level, persistent (not re-initialised per render) flag"
    )


def test_render_lint_card_consumes_pending_deep_c5_audit_as_one_shot():
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    assert "var c5Deep = pendingDeepC5Audit;" in render_fn
    assert "pendingDeepC5Audit = false;" in render_fn
    # consumed (reset) immediately, before it could leak into a later render
    assert render_fn.index("pendingDeepC5Audit = false;") < render_fn.index("var c5Stale = false;")


def test_deep_audit_step_sets_pending_flag_before_rendering():
    """The Deep audit control is the ONLY call site that requests
    include_c5=true — it must set pendingDeepC5Audit BEFORE renderLintCard
    runs, and must NOT grow renderLintCard's own argument list (the existing
    test_ui_console_lint_fast_default.py pins the literal 2-arg call in both
    the Run and Deep audit bodies)."""
    text = _console_text()
    deep_audit_body = _extract_lint_step_deep_audit_body(text)
    set_pos = deep_audit_body.find("pendingDeepC5Audit = true;")
    render_pos = deep_audit_body.find("renderLintCard(data, resultEl)")
    assert set_pos != -1, "deepAudit() must set pendingDeepC5Audit = true before rendering"
    assert render_pos != -1, (
        "deepAudit() must still call the plain 2-arg renderLintCard(data, resultEl)"
    )
    assert set_pos < render_pos, "pendingDeepC5Audit must be set BEFORE renderLintCard runs"
    assert "renderLintCard(data, resultEl, true)" not in deep_audit_body, (
        "the deep-judgment signal must not be threaded through as a 3rd renderLintCard argument"
    )


def test_lint_step_run_never_sets_pending_deep_c5_audit():
    """Regression guard: the fast (include_c5=false) pipeline Run must never
    mark its own response as a fresh deep judgment."""
    text = _console_text()
    run_body = _extract_lint_step_run_body(text)
    assert "pendingDeepC5Audit" not in run_body


# ---------------------------------------------------------------------------
# Stale marker: bilingual chrome, wired into the C5 check-group heading only
# ---------------------------------------------------------------------------


def test_c5_stale_note_chrome_is_bilingual():
    text = _console_text()
    assert re.search(r'c5StaleNote:\s*"stale — re-run Deep audit \(C5\) to refresh"', text), (
        "LINT_CHROME.en.c5StaleNote must exist (issue #489 AC 'stale marker')"
    )
    assert re.search(r'c5StaleNote:\s*"[^"]*已過時[^"]*深度稽核[^"]*"', text), (
        "LINT_CHROME.zh.c5StaleNote must exist and reference the deep-audit re-run"
    )


def test_check_head_accepts_an_optional_note_appended_to_the_label():
    text = _console_text()
    check_head_fn = _extract_function(text, "checkHead")
    assert "function checkHead(code, count, batchBtnEl, noteText)" in check_head_fn
    assert 'noteText ? " — " + noteText : ""' in check_head_fn


def test_axis_loop_wires_the_stale_note_only_for_c5():
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    assert 'code === "C5" && c5Stale' in render_fn, (
        "the stale marker must be gated to the C5 check specifically — no "
        "other check's heading should ever pick up this note"
    )
    assert "checkHead(code, items.length, checkBatchBtnEl, noteText)" in render_fn


# ---------------------------------------------------------------------------
# Badge total: fold the stale pairs back into the top count
# ---------------------------------------------------------------------------


def test_badge_total_folds_in_stale_c5_pairs_count():
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    assert (
        'if (c5Stale && typeof total === "number") total += lastDeepC5Pairs.length;' in render_fn
    ), (
        "the fast relint's own total_findings never counted the stale C5 "
        "pairs — the badge must fold them back in so it does not undercount "
        "against what the body renders (issue #489's exact reported illusion)"
    )


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
