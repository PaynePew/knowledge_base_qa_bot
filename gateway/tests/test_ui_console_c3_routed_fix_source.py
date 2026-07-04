"""Structural tests for C3's Routed fix-the-Source UI (issue #408, ADR-0029
decisions 2-4).

Following the pattern in ``test_ui_console_routed_coverage_fill.py``, these
tests inspect the production ``gateway/static/console.html`` file's text —
no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- C3 is the first finding carrying TWO remediation classes: the existing
  Direct "Re-ingest (retry)" action stays, and a new "Fix Source" Routed
  navigation control is added ALONGSIDE it (never replacing it).
- The "Fix Source" control's click handler makes no request of its own
  (ADR-0029 Invariant: a Routed remediation commits nothing itself).
- The context banner carries the finding's Source filename + unsupported
  claims, plus a read-only View Source control (GET /read/file, via the
  existing openFile() viewer).
- The give-up path is card copy (no button) — no delete affordance appears
  on a C3 finding (ADR-0029 Invariant).
- The post-fix-source outcome check hides the banner once the finding
  clears on a re-lint, mirroring checkCoverageFillOutcome's ordering.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _function_body(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace-depth
    scanning — robust to nested braces inside the body (mirrors the sibling
    test files' helper)."""
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
# Taxonomy: C3 stays direct, gains the Fix Source control alongside it
# ---------------------------------------------------------------------------


def test_c3_row_renders_both_remediation_classes():
    """ROW_RENDERERS.C3 must render the existing Direct reingestAction AND
    the new Routed fixSourceAction — a strict addition, never a replacement."""
    text = _console_text()
    c3_row = re.search(r"C3:\s*function\(i, f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert c3_row is not None
    body = c3_row.group(1)
    assert "reingestAction(f.source, true)" in body, f"C3 must keep Re-ingest: {body}"
    assert "fixSourceAction(f)" in body, f"C3 must add Fix Source: {body}"
    assert "c3GiveUpNote(f)" in body, f"C3 must render the give-up advisory: {body}"


def test_c3_is_still_direct_tier_in_client_taxonomy():
    """LINT_CHECK_META.C3 is unaffected — C3's PRIMARY tier stays direct."""
    text = _console_text()
    match = re.search(r"var LINT_CHECK_META\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    assert match is not None
    c3 = re.search(r"C3:\s*\{([^}]*)\}", match.group(1))
    assert c3 is not None
    assert 'tier: "direct"' in c3.group(1), f"C3 tier must stay direct: {c3.group(1)}"


# ---------------------------------------------------------------------------
# Navigation control: real button, but no execution
# ---------------------------------------------------------------------------


def test_fix_source_button_class_and_bilingual_label():
    text = _console_text()
    assert "fix-source-btn" in text
    en_match = re.search(r"en:\s*\{(.*?)\n  \},", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},", text, re.DOTALL)
    assert en_match is not None and zh_match is not None
    assert re.search(r'fixSource:\s*"[^"]+"', en_match.group(1)), "en.fixSource missing"
    assert re.search(r'fixSource:\s*"[^"]+"', zh_match.group(1)), "zh.fixSource missing"


def test_fix_source_action_never_makes_a_request():
    """ADR-0029 Invariant: a Routed remediation commits nothing itself — the
    control's own click handler must contain no fetch call anywhere."""
    text = _console_text()
    body = _function_body(text, "fixSourceAction")
    assert "fetch(" not in body, f"fixSourceAction must not fetch: {body}"


def test_fix_source_action_sets_pending_state_and_shows_banner():
    text = _console_text()
    body = _function_body(text, "fixSourceAction")
    assert "pendingFixSource" in body
    assert "showFixSourceBanner(" in body


def test_fix_source_action_degrades_gracefully_with_no_source():
    """Mirrors reingestAction's noSource guard (§12.5 no false affordance)."""
    text = _console_text()
    body = _function_body(text, "fixSourceAction")
    assert "finding.source" in body
    assert "btn.disabled = true" in body


# ---------------------------------------------------------------------------
# Context banner: Source filename + unsupported claims + view link
# ---------------------------------------------------------------------------


def test_context_banner_element_exists_and_hidden_by_default():
    text = _console_text()
    assert 'id="fix-source-banner"' in text
    css_match = re.search(r"\.fix-source-banner\s*\{([^}]*)\}", text, re.DOTALL)
    assert css_match is not None
    assert "display: none" in css_match.group(1)
    visible_match = re.search(r"\.fix-source-banner\.visible\s*\{([^}]*)\}", text)
    assert visible_match is not None


def test_banner_text_carries_filename_and_claims():
    """fixSourceBannerText is a top-level function (called from the
    fixSourceAction click handler, not from inside renderLintCard's
    closure), so it must compute the claims summary itself rather than
    reach for c3ClaimsSummary — a private helper scoped inside
    renderLintCard that is NOT visible here (would be a ReferenceError at
    click time). issue #445: the Source location comes from
    c3SourceLabel(finding), which renders the resolved source_path (or a
    distinct missing/ambiguous message) instead of a client-built
    "docs/" + basename guess."""
    text = _console_text()
    body = _function_body(text, "fixSourceBannerText")
    assert "c3SourceLabel(finding)" in body
    assert "finding.unsupported_claims" in body
    assert "c3ClaimsSummary" not in body, (
        "fixSourceBannerText must not call renderLintCard's private "
        "c3ClaimsSummary helper — it is out of scope from top level"
    )


def test_banner_offers_a_view_source_control_wired_to_read_file():
    """AC: 'a /read/file view link' — implemented as a button reusing the
    existing openFile() resource-browser viewer (GET /read/file), never a
    second fetch implementation. issue #445: the button opens the finding's
    own server-resolved source_path — never a client-built "docs/" +
    basename guess, which 404'd for Sources nested under a docs/
    subdirectory."""
    text = _console_text()
    body = _function_body(text, "showFixSourceBanner")
    assert "view-source-btn" in body
    assert "openFile(" in body
    assert "openFile(finding.source_path" in body
    assert '"docs/" + ' not in body, (
        "showFixSourceBanner must not rebuild the Source path client-side "
        "(issue #445) — it must use finding.source_path verbatim"
    )


def test_show_fix_source_banner_never_fetches_directly():
    """The banner itself is pure navigation — only openFile() (a read-only,
    existing GET) may be reached from inside it; no POST/mutating call."""
    text = _console_text()
    body = _function_body(text, "showFixSourceBanner")
    assert "fetch(" not in body


# ---------------------------------------------------------------------------
# Give-up path: card copy, no button (ADR-0029 Invariant — no delete)
# ---------------------------------------------------------------------------


def test_give_up_note_chrome_exists_bilingually():
    text = _console_text()
    en_match = re.search(r"en:\s*\{(.*?)\n  \},", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},", text, re.DOTALL)
    assert en_match is not None and zh_match is not None
    assert re.search(r'c3GiveUpPrefix:\s*"[^"]+"', en_match.group(1))
    assert re.search(r'c3GiveUpSuffix:\s*"[^"]+"', en_match.group(1))
    assert re.search(r'c3GiveUpPrefix:\s*"[^"]+"', zh_match.group(1))
    assert re.search(r'c3GiveUpSuffix:\s*"[^"]+"', zh_match.group(1))


def test_zh_give_up_copy_matches_the_issue_verbatim():
    """The zh copy is quoted verbatim from issue #408 / ADR-0029 decision 4 —
    pin the exact string so it cannot drift silently."""
    text = _console_text()
    assert "若此 Source 不該產生 wiki 頁:移除 docs/" in text
    assert "後此頁轉為 C11 full orphan,可用 Confirmed delete 移除。" in text


def test_give_up_note_has_no_button():
    """Card copy only — no delete affordance appears on a C3 finding
    (ADR-0029 Invariant)."""
    text = _console_text()
    body = _function_body(text, "c3GiveUpNote")
    assert 'el("button"' not in body
    assert "<button" not in body


def test_c3_row_offers_no_delete_or_discard_affordance():
    """ADR-0029 Invariant, structural check at the row level too: none of
    C3's per-row actions may be a delete/discard control."""
    text = _console_text()
    c3_row = re.search(r"C3:\s*function\(i, f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert c3_row is not None
    body = c3_row.group(1)
    assert "deleteOrphanAction(" not in body
    assert "discardAction(" not in body


# ---------------------------------------------------------------------------
# Outcome check: one-shot, ordered before render (mirrors C1/C2's pattern)
# ---------------------------------------------------------------------------


def test_check_fix_source_outcome_exists_and_is_one_shot():
    text = _console_text()
    body = _function_body(text, "checkFixSourceOutcome")
    assert "pendingFixSource = null" in body, "pendingFixSource must be consumed (one-shot)"
    assert "hideFixSourceBanner()" in body
    assert "failed_grounding" in body, "must re-check against the SAME findings key C3 reads"


def test_lint_step_checks_fix_source_outcome_before_rendering():
    """The Lint step's own Run must call checkFixSourceOutcome(data) BEFORE
    renderLintCard, mirroring checkCoverageFillOutcome's ordering."""
    text = _console_text()
    lint_step_match = re.search(
        r'id:\s*"lint",.*?run:\s*function\(resultEl, setStepBusy\)\s*\{(.*?)\n    \},',
        text,
        re.DOTALL,
    )
    assert lint_step_match is not None, "could not locate the lint STEP_DEFS entry"
    body = lint_step_match.group(1)
    check_pos = body.find("checkFixSourceOutcome(")
    render_pos = body.find("renderLintCard(data, resultEl)")
    assert check_pos != -1, f"lint step must call checkFixSourceOutcome: {body}"
    assert render_pos != -1
    assert check_pos < render_pos, "checkFixSourceOutcome must run before renderLintCard"


def test_run_lint_remediation_and_batch_remediation_also_check_fix_source_outcome():
    """The two Direct-remediation auto-relint call sites (single-row and
    batch) must also resolve a pending Fix Source, mirroring
    checkCoverageFillOutcome's existing wiring at both sites."""
    text = _console_text()
    run_lint_remediation_body = _function_body(text, "runLintRemediation")
    finish_batch_body = _function_body(text, "finishBatchRemediation")
    assert "checkFixSourceOutcome(data)" in run_lint_remediation_body
    assert "checkFixSourceOutcome(data)" in finish_batch_body


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
