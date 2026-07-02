"""Structural tests for the Operator Console Lint Remediation UI (issue #363).

Following the pattern in ``test_ui_console_legibility.py``, these tests
inspect the production ``gateway/static/console.html`` file's text to assert
the structural invariants of issue #363 (Lint Remediation tier-A S3):

- The Lint card groups findings under the four Lint Axis headers, reusing
  S1's stable order (Freshness -> Coherence -> Coverage -> Lifecycle).
- Each of the ten wired checks is labelled by code (issue #361 pattern).
- C6/C3 findings get a Re-ingest / Re-ingest (retry) button wired to the
  existing ``POST /wiki/ingest``; C3's request MUST include ``force:true``
  (ADR-0023 Invariant — CODING_STANDARD §11/§12.8 drift signal guard).
- C10 findings get a Discard button wired to the existing
  ``DELETE /wiki/qa/{slug}``.
- Authored-tier findings (C5/C4/C1/C2) render a disabled tier-B affordance.
- A successful remediation re-runs ``POST /wiki/lint?include_c5=false``.
- No ``innerHTML`` assignment is introduced (§12.4 — extends the existing
  console-wide guard with a remediation-specific anchor).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the click -> fix -> relint loop is verified manually
per §12.7 (visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Axis grouping (reuses S1's taxonomy/order)
# ---------------------------------------------------------------------------


def test_lint_axis_order_matches_s1_taxonomy():
    """The Console's axis order mirrors LINT_AXIS_ORDER from markdown_kb."""
    text = _console_text()
    match = re.search(r"var LINT_AXIS_ORDER\s*=\s*\[(.*?)\];", text)
    assert match is not None, "console.html must define LINT_AXIS_ORDER"
    axes = [a.strip().strip('"') for a in match.group(1).split(",")]
    assert axes == ["Freshness", "Coherence", "Coverage", "Lifecycle"]


def test_lint_card_groups_findings_under_axis_headers():
    """renderLintCard renders one axis-head element per LINT_AXIS_ORDER entry."""
    text = _console_text()
    assert "lint-axis-head" in text, "Lint card must render an axis-head element (issue #363)"
    assert "axisHead(axis)" in text or "axisHead(" in text


def test_every_wired_check_is_labelled_by_code():
    """All ten wired checks (C1-C11, minus C7) appear in the check-metadata map."""
    text = _console_text()
    match = re.search(r"var LINT_CHECK_META\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    assert match is not None, "console.html must define LINT_CHECK_META"
    body = match.group(1)
    for code in ["C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10", "C11"]:
        assert re.search(rf"\b{code}\s*:", body), f"LINT_CHECK_META missing {code}"


# ---------------------------------------------------------------------------
# Direct Remediation buttons: C6/C3 Re-ingest, C10 Discard
# ---------------------------------------------------------------------------


def test_c6_stale_gets_reingest_button():
    text = _console_text()
    assert "reingestAction(f.source, false)" in text, (
        "C6 stale rows must wire a Re-ingest action (issue #363 AC)"
    )


def test_c3_failed_grounding_gets_reingest_retry_with_force_true():
    """C3's request MUST send force:true — omitting it is a §11 drift signal
    (hash-skip #93 would no-op the retry into a false fix)."""
    text = _console_text()
    assert "reingestAction(f.source, true)" in text, (
        "C3 failed-grounding rows must wire a Re-ingest (retry) action (issue #363 AC)"
    )
    # The reingestAction implementation must forward force through to the request.
    reingest_fn = re.search(r"function reingestAction\(.*?\n\}", text, re.DOTALL)
    assert reingest_fn is not None
    assert "ingestRemediationRequest(source, retry" in reingest_fn.group(0)
    request_fn = re.search(r"function ingestRemediationRequest\(.*?\n\}", text, re.DOTALL)
    assert request_fn is not None
    assert "payload.force = true" in request_fn.group(0), (
        "ingestRemediationRequest must send force:true when force is truthy (ADR-0023 Invariant)"
    )


def test_c10_invalid_schema_gets_discard_button_in_lint_card():
    text = _console_text()
    assert "discardAction(f.page_slug)" in text, (
        "C10 invalid-schema rows must wire a Discard action (issue #363 AC)"
    )


def test_remediation_requests_hit_existing_endpoints_only():
    """No new backend endpoint — Direct Remediation reuses /wiki/ingest and
    /wiki/qa/{slug} (issue #363 AC 'No new backend endpoint')."""
    text = _console_text()
    assert '"/wiki/ingest"' in text
    assert '"/wiki/qa/"' in text


# ---------------------------------------------------------------------------
# Authored-tier disabled affordance
# ---------------------------------------------------------------------------


def test_authored_tier_findings_render_disabled_affordance():
    text = _console_text()
    assert "tier-b-btn" in text
    assert "Authored (tier B)" in text
    # The disabled button must actually carry a disabled attribute.
    tier_b_fn = re.search(r"function tierBAffordance\(\).*?\n\}", text, re.DOTALL)
    assert tier_b_fn is not None
    assert "disabled" in tier_b_fn.group(0)


def test_c8_promotion_controls_stay_in_curation_queue_only():
    """C8's ROW_RENDERERS entry must NOT wire a remediation action — its
    Promote/Discard controls remain the dedicated Curation Queue block
    (issue #363 AC 'C8 promotion controls remain ... unchanged')."""
    text = _console_text()
    c8_row = re.search(r"C8:\s*function\(i, f\)\s*\{(.*?)\},", text, re.DOTALL)
    assert c8_row is not None
    assert "Action(" not in c8_row.group(1)


# ---------------------------------------------------------------------------
# Auto-relint on success
# ---------------------------------------------------------------------------


def test_remediation_success_re_runs_fast_lint():
    text = _console_text()
    runner_fn = re.search(r"function runLintRemediation\(.*?\n\}", text, re.DOTALL)
    assert runner_fn is not None
    assert '"/wiki/lint?include_c5=false"' in runner_fn.group(0), (
        "A successful remediation must re-run POST /lint?include_c5=false (issue #363 AC)"
    )
    assert "renderLintCard(data, resultEl)" in runner_fn.group(0)
    assert "renderCurationQueue(" in runner_fn.group(0)


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
