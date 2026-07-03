"""Structural tests for the Operator Console Promote-all batch action
(tier-B S6, issue #382, ADR-0023 Consequences).

Following the pattern in ``test_ui_console_lint_remediation_batch.py``
(tier-A S4, issue #364), these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of issue #382:

- The C8 "Promotion Candidates" section header offers a "Promote all (N)"
  batch action, wired to a single ``POST /wiki/qa/promote-batch`` call
  carrying exactly the slugs rendered as candidates (never re-resolved as
  "all drafts").
- The batch action is guarded by the shared ``remediationInFlight`` flag
  (double-submit guard, issue #364 convention) and reuses the shared
  ``finishBatchRemediation`` re-lint-and-rerender path.
- The batch completion renders a per-item skipped list (slug + reason),
  not just an aggregate count.
- The batch affordance is wired ONLY inside the C8 candidates branch of
  ``renderCurationQueue`` — never inside any Authored/Confirmed/Routed
  finding renderer (guards the ADR-0024 "batch is Direct-only" invariant).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7).
"""

from __future__ import annotations

from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors test_ui_console_lint_remediation_batch.py)."""
    marker = f"function {name}("
    start = text.index(marker)
    depth = 0
    started = False
    i = start
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
# The batch action exists, posts to the right endpoint, with the right body
# ---------------------------------------------------------------------------


def test_promote_all_action_posts_to_promote_batch_endpoint():
    text = _console_text()
    fn = _extract_function(text, "promoteAllAction")
    assert '"/wiki/qa/promote-batch"' in fn
    assert '"POST"' in fn
    assert "slugs: slugs" in fn, "the request body must carry the exact submitted slugs"


def test_promote_all_button_wired_into_c8_section_head_only():
    """promoteAllAction is called exactly once, inside the C8 candidates
    branch of renderCurationQueue — never from a Gated/Routed finding
    renderer (ADR-0024 invariant: batch is Direct-only)."""
    text = _console_text()
    call_sites = text.count("promoteAllAction(")
    # One definition site (`function promoteAllAction(slugs) {`) + exactly
    # one call site inside renderCurationQueue's C8 branch.
    assert text.count("function promoteAllAction(") == 1
    assert call_sites == 2, f"expected exactly one call site, found {call_sites - 1}"

    render_queue_fn = _extract_function(text, "renderCurationQueue")
    assert "promoteAllAction(" in render_queue_fn, (
        "the call site must live inside renderCurationQueue (the C8 block)"
    )

    # Gated/Routed finding renderers must never reference the batch action.
    for gated_fn_name in ("reconcileAction", "collisionAction"):
        gated_fn = _extract_function(text, gated_fn_name)
        assert "promoteAllAction" not in gated_fn, (
            f"{gated_fn_name} (a Gated finding renderer) must never wire a batch "
            "action — ADR-0024 invariant: batch is Direct-only"
        )


# ---------------------------------------------------------------------------
# Double-submit guard + shared batch-completion path
# ---------------------------------------------------------------------------


def test_promote_all_guarded_by_remediation_in_flight():
    text = _console_text()
    fn = _extract_function(text, "promoteAllAction")
    assert "remediationInFlight" in fn, (
        "promoteAllAction must consult the shared in-flight guard (issue #364 "
        "convention) so a click mid-batch is a no-op"
    )
    assert "btn.disabled = true" in fn, "the button must disable itself while in flight"


def test_promote_all_reuses_shared_batch_completion_path():
    """On success, promoteAllAction hands off to the SAME finishBatchRemediation
    helper the Lint card's own batch actions use — one re-lint, one re-render
    of both cards, one durability story (not a bespoke re-render path)."""
    text = _console_text()
    fn = _extract_function(text, "promoteAllAction")
    assert "finishBatchRemediation(" in fn


# ---------------------------------------------------------------------------
# Per-item skipped list rendering (issue #382 AC)
# ---------------------------------------------------------------------------


def test_promote_batch_summary_renders_skipped_reason_per_item():
    text = _console_text()
    fn = _extract_function(text, "renderPromoteBatchSummary")
    assert "result.skipped" in fn
    assert "s.slug + " in fn and "s.reason" in fn, (
        "each skipped item must render both its slug and its reason"
    )


def test_promote_batch_summary_is_a_one_shot_banner():
    """lastPromoteBatchResult is cleared immediately after being rendered so
    it does not linger across an unrelated later re-render (mirrors
    lastBatchSummary's one-shot convention, issue #364)."""
    text = _console_text()
    render_queue_fn = _extract_function(text, "renderCurationQueue")
    assert "lastPromoteBatchResult = null" in render_queue_fn
