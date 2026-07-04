"""Structural tests for the Operator Console Assign-Alias UI (issue #409,
ADR-0030 decision 3).

Following the pattern in ``test_ui_console_lint_remediation.py`` /
``test_ui_console_lint_remediation_batch.py``, these tests inspect the
production ``gateway/static/console.html`` file's text to assert the
structural invariants of issue #409:

- C2 red-link rows render TWO exits: the existing Routed
  ``fillViaImportAction`` (unchanged) and a new Direct-class
  ``assignAliasAction`` picker, mirroring the C3 two-remediation-classes
  precedent (issue #408).
- The Assign-Alias write reuses the shared ``runLintRemediation`` path (the
  SAME re-lint ``include_c5=false`` + re-render every other Direct
  remediation on this card uses), not a bespoke fetch/relint loop.
- The picker's page list is sourced from the EXISTING ``GET /read/tree``
  resource browser — no second new listing endpoint invented.
- No batch affordance anywhere (ADR-0030 Invariant) — no
  "assign-alias-all"/"assign all" control exists.
- The post-fill honest-miss report (ADR-0027 decision 3) is upgraded into an
  executable per-new-page offer, reusing the same remediation path.
- Bilingual chrome (issue #365 convention) exists for both new strings.

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
# C2 gets a second, Direct-class exit alongside the existing Routed one
# ---------------------------------------------------------------------------


def test_c2_red_link_renders_both_fill_via_import_and_assign_alias():
    text = _console_text()
    match = re.search(r"C2:\s*function\(i,\s*f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert match is not None, "console.html must define ROW_RENDERERS.C2"
    body = match.group(1)
    assert 'fillViaImportAction("C2", f)' in body, (
        "C2 must keep its existing Routed Fill via Import control (issue #383)"
    )
    assert "assignAliasAction(f)" in body, (
        "C2 must gain a second, Direct-class Assign-Alias control (issue #409)"
    )


def test_assign_alias_action_is_defined():
    text = _console_text()
    assert "function assignAliasAction(finding)" in text


# ---------------------------------------------------------------------------
# The write reuses the shared remediation runner — no bespoke fetch/relint
# ---------------------------------------------------------------------------


def test_assign_alias_action_posts_to_the_new_endpoint():
    text = _console_text()
    assert "function assignAliasRequest(pageSlug, alias)" in text
    assert '"/wiki/pages/" + encodeURIComponent(pageSlug) + "/aliases"' in text
    assert '"POST"' in text.split("function assignAliasRequest")[1][:400]


def test_assign_alias_action_reuses_run_lint_remediation():
    text = _console_text()
    action_src = text.split("function assignAliasAction(finding)")[1].split(
        "function coverageFillOutcomeBanner"
    )[0]
    assert "runLintRemediation(" in action_src, (
        "Assign-Alias must reuse the shared Direct-remediation runner "
        "(the SAME include_c5=false re-lint + re-render every other row action uses)"
    )
    assert "assignAliasRequest(select.value, finding.slug)" in action_src


# ---------------------------------------------------------------------------
# Picker page list reuses the EXISTING GET /read/tree resource browser
# ---------------------------------------------------------------------------


def test_alias_picker_reuses_existing_read_tree_endpoint():
    text = _console_text()
    fetch_src = text.split("function fetchAliasPickerPages()")[1][:800]
    assert "/read/tree?" in fetch_src, (
        "the picker's page list must come from the EXISTING GET /read/tree "
        "endpoint, not a new listing endpoint (ADR-0030 adds exactly one new endpoint)"
    )
    assert '"wiki/" + subdir' in fetch_src


def test_alias_picker_select_element_is_wired():
    text = _console_text()
    assert 'class: "assign-alias-select"' in text
    assert "aliasPickerPages()" in text


# ---------------------------------------------------------------------------
# No batch affordance anywhere (ADR-0030 Invariant)
# ---------------------------------------------------------------------------


def test_assign_alias_never_batches():
    text = _console_text()
    assert "assign-alias-all" not in text
    assert "assignAliasAllAction" not in text
    assert "aliases: [" not in text, "the request body must carry exactly one alias, never a list"


# ---------------------------------------------------------------------------
# Honest-miss offer upgrade (ADR-0027 decision 3 + ADR-0030 decision 3)
# ---------------------------------------------------------------------------


def test_coverage_fill_outcome_is_a_structured_object_not_a_plain_string():
    text = _console_text()
    assert "lastCoverageFillOutcome = {" in text, (
        "the one-shot outcome must carry {target, newPages} so the honest-miss "
        "report can render an executable offer, not just a message string"
    )


def test_coverage_fill_outcome_banner_offers_one_button_per_new_page():
    text = _console_text()
    banner_src = text.split("function coverageFillOutcomeBanner(outcome)")[1].split(
        "\n  /* C1 Verify: re-ask"
    )[0]
    assert "outcome.newPages.forEach(" in banner_src
    assert '"assign-alias-offer-btn"' in banner_src
    assert "runLintRemediation(" in banner_src
    assert "assignAliasRequest(pageSlug, outcome.target)" in banner_src


def test_render_lint_card_invokes_the_offer_banner():
    text = _console_text()
    assert "coverageFillOutcomeBanner(lastCoverageFillOutcome)" in text


# ---------------------------------------------------------------------------
# Bilingual chrome (issue #365 convention)
# ---------------------------------------------------------------------------


def test_assign_alias_chrome_is_bilingual():
    text = _console_text()
    en_match = re.search(r"en:\s*\{(.*?)\n  \},", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},", text, re.DOTALL)
    assert en_match is not None and zh_match is not None
    assert "assignAlias:" in en_match.group(1)
    assert "assignAliasChoosePage:" in en_match.group(1)
    assert "assignAlias:" in zh_match.group(1)
    assert "assignAliasChoosePage:" in zh_match.group(1)
