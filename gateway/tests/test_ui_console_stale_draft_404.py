"""Structural tests for the Curation Queue stale-draft self-heal (issue #442).

Curation Queue cards capture their slug at render time and never revalidate
before an action fires. The qa drafts under ``wiki/qa/`` are ephemeral demo
data (wiped by the periodic demo reset or a redeploy), so an operator
holding an older tab who clicks Save/Promote/Discard against a draft whose
file is already gone previously got a raw
``Error: HTTP 404 wiki/qa/<slug>.md not found``.

Following the pattern in ``test_ui_console_qa_edit.py`` / ``test_ui_console_
lint_bilingual.py``, these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of the self-heal:

- A shared ``handleStaleDraftGone`` helper marks the card as gone and
  refreshes the queue via the shared ``refreshCurationQueueFast`` helper —
  the queue actually drops the card because the next lint response no
  longer contains its finding, not because the client deletes a DOM node
  itself (§12.5: no client-side business logic re-deriving queue state).
- Promote, Save (PUT), and Discard each branch on an HTTP 404 status
  *before* falling through to the generic "Error: HTTP ..." string
  construction, so the raw error text never renders for this case.
- The notice shown is the bilingual ``LINT_CHROME`` string, in both en/zh.
- Non-404 errors (409 Discard-refused, 422 grounding failure, network
  errors, other HTTP statuses) are untouched — they still surface via the
  existing "Error: ..." / validation-list paths.
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). The actual click -> 404 -> card-fades -> queue-refreshes loop is
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
    matching (mirrors test_ui_console_lint_bilingual.py's helper of the
    same name)."""
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


def _build_draft_card_body() -> str:
    text = _console_text()
    match = re.search(r"function buildDraftCard\(f\)\s*\{(.*?)\n\}\n", text, re.DOTALL)
    assert match is not None, "buildDraftCard function not found in console.html"
    return match.group(1)


# ---------------------------------------------------------------------------
# Shared self-heal helper + shared queue-refresh helper exist
# ---------------------------------------------------------------------------


def test_stale_draft_gone_helper_exists():
    text = _console_text()
    fn = _extract_function(text, "handleStaleDraftGone")
    assert "LINT_CHROME[consoleLang].staleDraftGone" in fn, (
        "handleStaleDraftGone must show the bilingual notice (issue #442 AC "
        "'explanatory notice (en/zh)')"
    )
    assert "cardEl.style.opacity" in fn, "the card must be visually marked gone"
    assert "refreshCurationQueueFast()" in fn, (
        "the self-heal must trigger a queue refresh (issue #442 AC 'refreshes the queue')"
    )


def test_stale_draft_gone_helper_tolerates_null_status_msg():
    """The C10 flag-card Discard path calls doDiscard without a statusMsgEl —
    the helper must not assume one is always present."""
    text = _console_text()
    fn = _extract_function(text, "handleStaleDraftGone")
    assert "if (statusMsgEl)" in fn


def test_refresh_curation_queue_fast_helper_exists_and_is_shared():
    text = _console_text()
    fn = _extract_function(text, "refreshCurationQueueFast")
    assert '"/wiki/lint?include_c5=false"' in fn
    assert "renderCurationQueue(" in fn

    # The Refresh button and the initial page-load fetch both route through
    # the same helper now, rather than duplicating the fetch+render call —
    # a single fetch+render path is what makes the self-heal's "auto-refresh
    # the queue" behave identically to a manual Refresh click.
    refresh_btn_fn = _extract_function(text, "makeRefreshBtn")
    assert "refreshCurationQueueFast()" in refresh_btn_fn

    boot_fn = re.search(r"\(function loadCurationQueueFast\(\) \{.*?\n\}\)\(\);", text, re.DOTALL)
    assert boot_fn is not None, "loadCurationQueueFast boot IIFE not found"
    assert "refreshCurationQueueFast()" in boot_fn.group(0)


# ---------------------------------------------------------------------------
# Promote self-heals on 404, before the generic error path
# ---------------------------------------------------------------------------


def test_promote_self_heals_on_404():
    body = _build_draft_card_body()
    promote_fn = re.search(r"promoteBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert promote_fn is not None
    promote_body = promote_fn.group(0)

    status_check = re.search(
        r"resp\.status === 404.{0,80}handleStaleDraftGone\(", promote_body, re.DOTALL
    )
    assert status_check is not None, (
        "Promote must branch on 404 into handleStaleDraftGone before the generic !resp.ok error path"
    )

    # The 404 branch must come before the generic "!resp.ok" throw so a
    # stale draft never reaches the raw "HTTP 404: ..." Error string.
    idx_404 = promote_body.index("resp.status === 404")
    idx_generic = promote_body.index("!resp.ok")
    assert idx_404 < idx_generic, "the 404 check must short-circuit before the generic error throw"


# ---------------------------------------------------------------------------
# Save (PUT) self-heals on 404, before the generic error path
# ---------------------------------------------------------------------------


def test_save_self_heals_on_404():
    body = _build_draft_card_body()
    save_fn = re.search(r"saveBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert save_fn is not None
    save_body = save_fn.group(0)

    status_check = re.search(
        r"result\.status === 404.{0,400}handleStaleDraftGone\(", save_body, re.DOTALL
    )
    assert status_check is not None, (
        "Save must branch on 404 into handleStaleDraftGone before the generic error string"
    )

    idx_404 = save_body.index("result.status === 404")
    idx_generic = save_body.index('"Error: HTTP " + result.status')
    assert idx_404 < idx_generic, (
        "the 404 check must come before the generic 'Error: HTTP ...' fallback"
    )

    # 422 (grounding-check failure) stays completely untouched by this slice.
    assert "result.status === 422" in save_body
    idx_422 = save_body.index("result.status === 422")
    assert idx_422 < idx_404, "422 handling must remain intact and precede the new 404 branch"


# ---------------------------------------------------------------------------
# Discard (shared doDiscard helper, used by both C8 draft cards and C10
# invalid-schema flag cards) self-heals on 404, alongside the existing 409
# live-page refusal
# ---------------------------------------------------------------------------


def test_discard_self_heals_on_404():
    text = _console_text()
    fn = _extract_function(text, "doDiscard")
    assert "resp.status === 404" in fn
    assert "handleStaleDraftGone(cardEl, statusMsgEl)" in fn

    # The existing 409 "page is live" refusal is untouched (unchanged AC:
    # "Non-404 errors still surface as errors").
    assert "resp.status === 409" in fn
    assert "Cannot discard: page is live" in fn

    idx_409 = fn.index("resp.status === 409")
    idx_404 = fn.index("resp.status === 404")
    idx_generic = fn.index("!resp.ok")
    assert idx_409 < idx_404 < idx_generic, (
        "409 and 404 must both be checked before the generic !resp.ok error branch"
    )


# ---------------------------------------------------------------------------
# Bilingual notice text (issue #442 AC: "explanatory notice (en/zh)")
# ---------------------------------------------------------------------------


def test_stale_draft_notice_is_bilingual():
    text = _console_text()
    en_block = re.search(r"var LINT_CHROME = \{\s*en: \{(.*?)\n  \},\n  zh: \{", text, re.DOTALL)
    zh_block = re.search(r"zh: \{(.*?)\n  \},\n\};", text, re.DOTALL)
    assert en_block is not None and zh_block is not None
    assert "staleDraftGone:" in en_block.group(1)
    assert "staleDraftGone:" in zh_block.group(1)

    # zh string is real Chinese copy, not a re-use of the English literal.
    zh_match = re.search(r'staleDraftGone:\s*"([^"]+)"', zh_block.group(1))
    assert zh_match is not None
    assert any("一" <= ch <= "鿿" for ch in zh_match.group(1)), (
        "zh staleDraftGone must contain actual CJK text"
    )


# ---------------------------------------------------------------------------
# No raw HTTP error string surfaces for the stale-draft case
# ---------------------------------------------------------------------------


def test_no_raw_http_string_reachable_for_stale_draft_promote():
    """Once the 404 branch returns early, the generic 'Error: HTTP ...'
    construction in the .catch() must never run for this case — the first
    .then() short-circuits with `return null` and the second .then() checks
    for it before touching statusMsgEl again."""
    body = _build_draft_card_body()
    promote_fn = re.search(r"promoteBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert promote_fn is not None
    promote_body = promote_fn.group(0)
    assert "handleStaleDraftGone(cardEl, statusMsgEl); return null;" in promote_body
    assert "if (data === null) return;" in promote_body, (
        "the success .then() must skip its body when the 404 branch already handled the response"
    )


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
