"""Structural tests for the Operator Console C12 alias-collision row
(issue #488, upgraded to a real Remove control by issue #491).

Following the pattern in ``test_ui_console_collision.py`` /
``test_ui_console_assign_alias.py``, these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of issue #488 / #491:

- C12 is a real, wired check: present in ``LINT_CHECK_META`` (with the
  ``alias_collisions`` findings key, matching ``LintFindings.alias_collisions``
  in ``markdown_kb/app/schemas.py``) and in the Coherence axis's
  ``AXIS_CHECK_CODES`` — before issue #488 neither existed, so a C12 finding
  was silently dropped by the axis-loop's ``if (!items ...) return`` guard
  (issue #488 "no row" symptom).
- C12 is NOT one of the Curation-Queue-owned Lifecycle checks (issue #438) —
  it must render its own per-item rows, not a count + pointer line.
- ``ROW_RENDERERS.C12`` renders per-row detail: the alias, the collision
  kind, and the colliding page(s)/slug (``claimed_by`` / ``slug_owner``).
- The row's remediation is ``removeAliasAction`` (issue #491, ADR-0030
  extension) wired to the real ``DELETE /pages/{slug}/aliases/{alias}``
  endpoint — NOT the add-only ``assignAliasAction`` picker (issue
  #409/ADR-0030 decision 3), which can never resolve a collision (verdict on
  the first #488 build).
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the click -> remove -> relint loop is verified
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
# Remediation is a real Remove control (issue #491) — not the add-only
# assign-alias picker, and not a disabled tier-B affordance
# ---------------------------------------------------------------------------


def _row_renderer_c12_body() -> str:
    match = re.search(r"C12:\s*function\(i,\s*f\)\s*\{(.*?)\n    \},", _console_text(), re.DOTALL)
    assert match is not None, "console.html must define ROW_RENDERERS.C12"
    return match.group(1)


def test_c12_row_wires_remove_alias_action_not_assign_alias():
    """The C12 row must NOT wire the add-only assign-alias picker: adding
    can never resolve a collision (same-page add is an idempotent no-op;
    any other page 409s — verdict on the first #488 build). Issue #491 wires
    the real fix, ``removeAliasAction`` (``DELETE
    /pages/{slug}/aliases/{alias}``), instead."""
    body = _row_renderer_c12_body()
    assert "assignAliasAction(" not in body, (
        "C12 must not reuse the add-only assign-alias picker - it cannot "
        "resolve a collision (verdict on the first #488 build)"
    )
    assert "removeAliasAction(" in body, (
        "C12 row must wire the real remove-alias control (issue #491)"
    )
    assert "tierBAffordance()" not in body


def test_c12_remove_labels_exist_in_both_language_maps():
    text = _console_text()
    for key in ("c12Remove:", "c12RemoveChooseKeeper:", "c12RemoveKeepAndClear:"):
        assert text.count(key) >= 2, f"{key} label must exist in BOTH LINT_CHROME maps (en + zh)"


def test_remove_alias_action_handles_both_collision_kinds():
    """``removeAliasAction`` (issue #491 AC) must branch on both C12 shapes:
    alias_vs_slug (no valid keeper — every claimant loses the alias) and
    alias_vs_alias (curator picks a keeper; the rest lose it)."""
    text = _console_text()
    match = re.search(r"function removeAliasAction\(finding\)\s*\{(.*?)\n  \}", text, re.DOTALL)
    assert match is not None, "console.html must define removeAliasAction"
    body = match.group(1)
    assert 'finding.kind === "alias_vs_slug"' in body
    assert "removeAliasFromClaimantsRequest" in body
