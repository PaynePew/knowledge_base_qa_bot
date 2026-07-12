"""Structural tests for the Operator Console's fast-by-default pipeline Lint
(issue #441).

Following the pattern in ``test_ui_console_lint_remediation_batch.py`` /
``test_ui_console_lint_bilingual.py``, these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of issue #441:

- The pipeline Lint step's own ``Run`` button now posts
  ``/wiki/lint?include_c5=false`` (zero LLM calls) instead of the old
  parameter-less ``/wiki/lint`` (which defaulted to the full, LLM-backed
  audit server-side).
- A separate, clearly-labelled "Deep audit (C5)" control exists next to the
  Run button and posts ``/wiki/lint?include_c5=true`` on demand, with a
  visible busy/running state (indeterminate spinner, §12.8 — no fake %).
- ``run()`` and ``deepAudit()`` share the exact same completion chain
  (outcome banners -> ``renderLintCard`` -> ``lastLintData`` -> Curation
  Queue), so a deep-audit response renders through the identical pipeline a
  default Lint response does — no special-cased C5 rendering was
  introduced. (The two bodies stay independently inlined, not factored into
  a shared helper: ``test_ui_console_routed_coverage_fill.py`` and
  ``test_ui_console_c3_routed_fix_source.py`` already pin the exact call
  ordering inside the lint step's own ``run()`` body, so this file verifies
  parity instead of refactoring that anchor away.)
- Running either control disables both buttons (an in-flight guard, mirrors
  the existing remediation-button convention) so Run and Deep audit can
  never race the same underlying lint pass.
- The Deep audit button's label is bilingual (``LINT_CHROME``), consistent
  with every other persistent button label already sourced from that map.
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the actual click -> fetch -> render loop is verified
manually per §12.7 (visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

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
    raise AssertionError(f"unterminated function body for {name}")


def _extract_lint_step_run_body(text: str) -> str:
    """Mirrors the anchor already used by test_ui_console_routed_coverage_fill.py
    and test_ui_console_c3_routed_fix_source.py so this file stays consistent
    with those pinned structural tests."""
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
# Pipeline Lint's own Run button is LLM-free by default
# ---------------------------------------------------------------------------


def test_lint_step_run_posts_include_c5_false():
    body = _extract_lint_step_run_body(_console_text())
    assert 'adminFetch("/wiki/lint?include_c5=false", { method: "POST" })' in body, (
        "the pipeline Lint step's Run button must call POST /wiki/lint with "
        "include_c5=false (issue #441 AC 'zero LLM calls') through the "
        "admin-token wrapper (issue #583)"
    )


def test_lint_step_no_longer_posts_bare_wiki_lint():
    """Regression guard: the old parameter-less POST /wiki/lint (which
    defaulted to include_c5=true server-side) must be gone — every call site
    is now explicit about include_c5 (issue #441)."""
    body = _extract_lint_step_run_body(_console_text())
    assert 'fetch("/wiki/lint", { method: "POST" })' not in body, (
        "the Lint step's Run must not fetch the bare /wiki/lint endpoint"
    )
    assert 'adminFetch("/wiki/lint", { method: "POST" })' not in body, (
        "the bare /wiki/lint call must not sneak back in via the admin-token "
        "wrapper either (issues #441/#583)"
    )


# ---------------------------------------------------------------------------
# Deep audit (C5) control exists, on demand, and is LLM-backed
# ---------------------------------------------------------------------------


def test_lint_step_defines_deep_audit_posting_include_c5_true():
    body = _extract_lint_step_deep_audit_body(_console_text())
    assert 'adminFetch("/wiki/lint?include_c5=true", { method: "POST" })' in body, (
        "Deep audit must call POST /wiki/lint with include_c5=true (issue #441 "
        "AC 'runs the contradiction audit on demand') through the admin-token "
        "wrapper (issue #583)"
    )


def test_run_and_deep_audit_share_the_same_completion_chain():
    """Both triggers must call the SAME sequence of downstream functions —
    the issue #441 AC 'same card, same downstream flows' is satisfied by
    parity between the two bodies, not by special-casing C5 anywhere."""
    text = _console_text()
    run_body = _extract_lint_step_run_body(text)
    deep_audit_body = _extract_lint_step_deep_audit_body(text)
    shared_calls = [
        "checkCoverageFillOutcome(data);",
        "checkFixSourceOutcome(data);",
        "resultEl.replaceChildren(renderLintCard(data, resultEl));",
        "lastLintData = data;",
        'renderCurationQueue(document.getElementById("curation-queue-root"), data);',
    ]
    for call in shared_calls:
        assert call in run_body, f"lint run() must call {call!r}"
        assert call in deep_audit_body, f"deepAudit() must call {call!r}"
    # Ordering must match too (coverage-fill / fix-source outcomes resolved
    # BEFORE renderLintCard, mirroring the existing pinned ordering).
    for body in (run_body, deep_audit_body):
        assert body.find("checkCoverageFillOutcome(") < body.find("renderLintCard(data, resultEl)")
        assert body.find("checkFixSourceOutcome(") < body.find("renderLintCard(data, resultEl)")


def test_deep_audit_shows_a_busy_state():
    """No fake progress (§12.8) — an indeterminate busy card while the C5
    audit runs, mirroring every other pipeline step's Run busy card."""
    body = _extract_lint_step_deep_audit_body(_console_text())
    assert "setStepBusy(true);" in body
    assert "resultEl.replaceChildren(makeBusyCard(" in body
    assert ".finally(function()   { setStepBusy(false); });" in body


def test_deep_audit_button_wired_only_when_step_defines_it():
    text = _console_text()
    build_pipeline = _extract_function(text, "buildPipeline")
    assert "if (step.deepAudit) {" in build_pipeline, (
        "the Deep audit button must be conditionally added, not hardcoded for "
        "every pipeline step (issue #441 — Lint-only control)"
    )
    assert 'class: "deep-audit-btn"' in build_pipeline
    assert "text: LINT_CHROME[consoleLang].deepAudit," in build_pipeline
    assert "step.deepAudit(resultEl, setStepBusy);" in build_pipeline


def test_deep_audit_busy_state_disables_both_buttons():
    """An in-flight Run or Deep audit call disables BOTH buttons — otherwise
    a user could fire two overlapping /wiki/lint calls against the same
    underlying lint pass (issue #441 AC 'visible progress/running state')."""
    text = _console_text()
    build_pipeline = _extract_function(text, "buildPipeline")
    set_step_busy = re.search(
        r"function setStepBusy\(busy\) \{.*?\n    \}", build_pipeline, re.DOTALL
    )
    assert set_step_busy is not None
    body = set_step_busy.group(0)
    assert "if (runBtn) runBtn.disabled = busy;" in body
    assert "if (deepAuditBtn) deepAuditBtn.disabled = busy;" in body


# ---------------------------------------------------------------------------
# Bilingual chrome for the new control (issue #441 AC)
# ---------------------------------------------------------------------------


def test_deep_audit_label_is_bilingual():
    text = _console_text()
    assert re.search(r'deepAudit:\s*"Deep audit \(C5\)"', text), (
        "LINT_CHROME.en.deepAudit must exist (issue #441 AC 'bilingual chrome strings')"
    )
    assert "深度稽核（C5）" in text, (
        "LINT_CHROME.zh.deepAudit must exist (issue #441 AC 'bilingual chrome strings')"
    )


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
