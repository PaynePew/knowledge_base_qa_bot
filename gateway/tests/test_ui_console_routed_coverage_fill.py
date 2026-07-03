"""Structural tests for the Routed Coverage Fill UI (tier-B S7, issue #383,
ADR-0027).

Following the pattern in ``test_ui_console_lint_remediation.py``, these
tests inspect the production ``gateway/static/console.html`` file's text —
no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- C1/C2 flip from the disabled Authored tier-B placeholder to a real
  "Fill via Import" navigation control (never an execute button).
- The control's click handler makes no request of its own (ADR-0027
  Invariant: a Routed remediation commits nothing itself).
- The context banner carries the finding's fields for both C1 and C2.
- The honest-miss report path: a fill that does not resolve the red link
  reports the new pages the fill created plus the still-unresolved target.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _function_body(text: str, name: str, *, indent: str = "") -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace-depth
    scanning — robust to nested braces inside the body (regex-based extraction
    like the sibling test files use breaks as soon as a function contains an
    ``if``/object-literal brace, which several of these do)."""
    m = re.search(rf"function {name}\([^)]*\)\s*\{{", text)
    assert m is not None, f"console.html must define function {name}(...)"
    start = m.end() - 1  # index of the opening brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unbalanced braces scanning function {name}")


# ---------------------------------------------------------------------------
# Taxonomy: C1/C2 render as routed, not authored
# ---------------------------------------------------------------------------


def test_c1_and_c2_are_routed_in_client_taxonomy():
    """LINT_CHECK_META mirrors markdown_kb.app.lint's routed tier for C1/C2
    (ADR-0027) — the disabled tier-B placeholder no longer applies to them."""
    text = _console_text()
    match = re.search(r"var LINT_CHECK_META\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    assert match is not None
    body = match.group(1)
    c1 = re.search(r"C1:\s*\{([^}]*)\}", body)
    c2 = re.search(r"C2:\s*\{([^}]*)\}", body)
    assert c1 is not None and c2 is not None
    assert 'tier: "routed"' in c1.group(1), f"C1 tier must be routed: {c1.group(1)}"
    assert 'tier: "routed"' in c2.group(1), f"C2 tier must be routed: {c2.group(1)}"


def test_c1_and_c2_rows_no_longer_use_the_disabled_tier_b_affordance():
    text = _console_text()
    c1_row = re.search(r"C1:\s*function\(i, f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    c2_row = re.search(r"C2:\s*function\(i, f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert c1_row is not None and c2_row is not None
    assert "tierBAffordance()" not in c1_row.group(1)
    assert "tierBAffordance()" not in c2_row.group(1)
    assert "fillViaImportAction(" in c1_row.group(1)
    assert "fillViaImportAction(" in c2_row.group(1)


# ---------------------------------------------------------------------------
# Navigation control: real button, but no execution
# ---------------------------------------------------------------------------


def test_fill_via_import_button_class_and_bilingual_label():
    text = _console_text()
    assert "fill-import-btn" in text
    assert "fillViaImport" in text
    # LINT_CHROME carries a real (non-empty) label for both languages.
    en_match = re.search(r"en:\s*\{(.*?)\n  \},", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},", text, re.DOTALL)
    assert en_match is not None and zh_match is not None
    assert re.search(r'fillViaImport:\s*"[^"]+"', en_match.group(1)), "en.fillViaImport missing"
    assert re.search(r'fillViaImport:\s*"[^"]+"', zh_match.group(1)), "zh.fillViaImport missing"


def test_fill_via_import_action_never_makes_a_request():
    """ADR-0027 Invariant: a Routed remediation commits nothing itself — the
    control's own click handler must contain no fetch call anywhere."""
    text = _console_text()
    body = _function_body(text, "fillViaImportAction")
    assert "fetch(" not in body, f"fillViaImportAction must not fetch: {body}"


def test_fill_via_import_action_sets_pending_state_and_shows_banner():
    text = _console_text()
    body = _function_body(text, "fillViaImportAction")
    assert "pendingCoverageFill" in body
    assert "showCoverageFillBanner(" in body


# ---------------------------------------------------------------------------
# Context banner: carries the finding's fields for both C1 and C2
# ---------------------------------------------------------------------------


def test_context_banner_element_exists_and_hidden_by_default():
    text = _console_text()
    assert 'id="coverage-fill-banner"' in text
    css_match = re.search(r"\.coverage-fill-banner\s*\{([^}]*)\}", text, re.DOTALL)
    assert css_match is not None
    assert "display: none" in css_match.group(1)
    visible_match = re.search(r"\.coverage-fill-banner\.visible\s*\{([^}]*)\}", text)
    assert visible_match is not None


def test_context_text_carries_c2_fields():
    """C2 banner text must carry slug + referenced_by + sample_context (AC)."""
    text = _console_text()
    body = _function_body(text, "coverageFillContextText")
    c2_branch = body.split('code === "C2"')[1] if 'code === "C2"' in body else body
    assert "finding.slug" in c2_branch
    assert "finding.referenced_by" in c2_branch
    assert "finding.sample_context" in c2_branch


def test_context_text_carries_c1_fields():
    """C1 banner text must carry sample_raw_queries + hit_count (AC)."""
    text = _console_text()
    body = _function_body(text, "coverageFillContextText")
    assert "finding.sample_raw_queries" in body
    assert "finding.hit_count" in body


# ---------------------------------------------------------------------------
# Honest-miss report path (ADR-0027 decision 3)
# ---------------------------------------------------------------------------


def test_honest_miss_check_exists_and_is_one_shot():
    text = _console_text()
    body = _function_body(text, "checkCoverageFillOutcome")
    assert "pendingCoverageFill = null" in body, "pendingCoverageFill must be consumed (one-shot)"
    assert "hideCoverageFillBanner()" in body


def test_honest_miss_message_reports_new_pages_and_unresolved_target():
    text = _console_text()
    body = _function_body(text, "checkCoverageFillOutcome")
    assert "lastIngestedPageSlugs" in body
    assert "still unresolved" in body
    assert "lastCoverageFillOutcome" in body


def test_ingest_batches_capture_new_page_slugs():
    """runIngestBatches must record the pages an Ingest run actually wrote,
    so the honest-miss report can name them (ADR-0027 decision 3)."""
    text = _console_text()
    body = _function_body(text, "runIngestBatches")
    assert "lastIngestedPageSlugs" in body


def test_lint_step_checks_coverage_fill_outcome_before_rendering():
    """The Lint step's own Run must call checkCoverageFillOutcome(data)
    BEFORE renderLintCard, so a pending fill's outcome banner is already
    queued when the card renders (mirrors the lastBatchSummary ordering)."""
    text = _console_text()
    lint_step_match = re.search(
        r'id:\s*"lint",.*?run:\s*function\(resultEl, setStepBusy\)\s*\{(.*?)\n    \},',
        text,
        re.DOTALL,
    )
    assert lint_step_match is not None, "could not locate the lint STEP_DEFS entry"
    body = lint_step_match.group(1)
    check_pos = body.find("checkCoverageFillOutcome(")
    render_pos = body.find("renderLintCard(data, resultEl)")
    assert check_pos != -1, f"lint step must call checkCoverageFillOutcome: {body}"
    assert render_pos != -1
    assert check_pos < render_pos, "checkCoverageFillOutcome must run before renderLintCard"


def test_render_lint_card_shows_and_clears_the_one_shot_outcome_banner():
    text = _console_text()
    body = _function_body(text, "renderLintCard")
    assert "lastCoverageFillOutcome" in body
    # One-shot: cleared after being read, mirroring lastBatchSummary immediately below it.
    assert re.search(r"lastCoverageFillOutcome\s*=\s*null", body), (
        "lastCoverageFillOutcome must be cleared after rendering (one-shot)"
    )


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
