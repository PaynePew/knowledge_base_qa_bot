"""Structural tests for the Operator Console C12 alias-collision row
(issue #488).

Following the pattern in ``test_ui_console_collision.py`` /
``test_ui_console_assign_alias.py``, these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of issue #488:

- C12 is a real, wired check: present in ``LINT_CHECK_META`` (with the
  ``alias_collisions`` findings key, matching ``LintFindings.alias_collisions``
  in ``markdown_kb/app/schemas.py``) and in the Coherence axis's
  ``AXIS_CHECK_CODES`` — before this slice neither existed, so a C12 finding
  was silently dropped by the axis-loop's ``if (!items ...) return`` guard
  (issue #488 "no row" symptom).
- C12 is NOT one of the Curation-Queue-owned Lifecycle checks (issue #438) —
  it must render its own per-item rows, not a count + pointer line.
- ``ROW_RENDERERS.C12`` renders per-row detail: the alias, the collision
  kind, and the colliding page(s)/slug (``claimed_by`` / ``slug_owner``).
- The row's remediation reuses the EXISTING ``assignAliasAction`` picker
  (issue #409/ADR-0030 decision 3) — no new endpoint, no bespoke action.
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the click -> assign -> relint loop is verified
manually per §12.7 (visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C12 is wired into the taxonomy/axis maps — the root cause of issue #488
# ---------------------------------------------------------------------------


def test_c12_is_in_lint_check_meta_with_alias_collisions_findings_key():
    text = _console_text()
    match = re.search(r"var LINT_CHECK_META\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    assert match is not None, "console.html must define LINT_CHECK_META"
    body = match.group(1)
    c12_line = re.search(r"C12:\s*\{([^}]*)\}", body)
    assert c12_line is not None, "LINT_CHECK_META must define a C12 entry (issue #488)"
    assert '"alias_collisions"' in c12_line.group(1), (
        "C12's findingsKey must be 'alias_collisions' — matches "
        "LintFindings.alias_collisions in markdown_kb/app/schemas.py"
    )


def test_c12_is_in_the_coherence_axis():
    text = _console_text()
    match = re.search(r"var AXIS_CHECK_CODES\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    assert match is not None, "console.html must define AXIS_CHECK_CODES"
    coherence = re.search(r"Coherence:\s*\[(.*?)\]", match.group(1))
    assert coherence is not None
    codes = [c.strip().strip('"') for c in coherence.group(1).split(",")]
    assert "C12" in codes, (
        "C12 must be listed under the Coherence axis (matching "
        "markdown_kb/app/lint.py's LINT_CHECK_TAXONOMY) or its findings are "
        "silently skipped by the axis-loop's empty-items guard (issue #488)"
    )


def test_c12_is_not_a_curation_queue_owned_check():
    text = _console_text()
    match = re.search(r"var QUEUE_OWNED_LIFECYCLE_CHECKS\s*=\s*\{(.*?)\};", text, re.DOTALL)
    assert match is not None
    assert "C12" not in match.group(1), (
        "C12 findings render their own per-row detail on the Lint card — they "
        "are not Curation-Queue-owned like C8/C9/C10"
    )


# ---------------------------------------------------------------------------
# ROW_RENDERERS.C12 renders alias, collision kind, and colliding pages/slug
# ---------------------------------------------------------------------------


def test_row_renderers_c12_is_defined():
    text = _console_text()
    match = re.search(r"C12:\s*function\(i,\s*f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert match is not None, "console.html must define ROW_RENDERERS.C12 (issue #488 AC)"
    body = match.group(1)
    assert "f.alias" in body, "the row must show the colliding alias"
    assert "f.kind" in body, "the row must show the collision type (alias_vs_slug / alias_vs_alias)"
    assert "f.claimed_by" in body, "the row must show the colliding page(s)"
    assert "f.slug_owner" in body, "the row must show the colliding real page slug when present"
    assert "f.resolved_to" in body


# ---------------------------------------------------------------------------
# Remediation reuses the EXISTING Assign-Alias picker — no new endpoint
# ---------------------------------------------------------------------------


def test_c12_row_reuses_assign_alias_action_not_a_new_control():
    text = _console_text()
    match = re.search(r"C12:\s*function\(i,\s*f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert match is not None
    body = match.group(1)
    assert "assignAliasAction(" in body, (
        "C12 must reuse the existing Assign-Alias flow (issue #409/ADR-0030 "
        "decision 3) — the issue explicitly scopes this as reusing, not "
        "inventing, a remediation control"
    )
    assert "tierBAffordance()" not in body, (
        "C12 has a real Direct-tier remediation available (assign-alias) — it "
        "must not fall back to the disabled Authored-tier placeholder"
    )


def test_c12_assign_alias_action_targets_the_colliding_alias():
    text = _console_text()
    match = re.search(r"C12:\s*function\(i,\s*f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert match is not None
    assert "assignAliasAction({ slug: f.alias })" in match.group(1), (
        "the picker must assign the collided alias string (f.alias) to "
        "whichever existing page the operator picks — the same shape "
        "assignAliasAction already expects via finding.slug"
    )


def test_assign_alias_action_itself_is_unmodified_by_this_slice():
    """assignAliasAction is reused verbatim (duck-typed { slug: ... } shim at
    the C12 call site) — its own body must be untouched so the existing C2
    coverage in test_ui_console_assign_alias.py keeps pinning it."""
    text = _console_text()
    action_src = text.split("function assignAliasAction(finding)")[1].split(
        "function coverageFillOutcomeBanner"
    )[0]
    assert "assignAliasRequest(select.value, finding.slug)" in action_src


# ---------------------------------------------------------------------------
# Console-wide invariants extended by this slice
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_introduced():
    text = _console_text()
    assert ".innerHTML" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
