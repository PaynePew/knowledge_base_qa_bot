"""Structural tests for the Transcribe confirm-modal state machine (issue #476).

The #447 modal had a cluster of state bugs invisible to static-HTML tests'
usual assertions — so these tests pin the STRUCTURE of the fix (which code
path owns which state transition), and the real click→fail→recover loops are
proven in a driven browser against a live gateway (issue #476 AC):

- M1: a failure path must NOT re-enable the row's trigger button while the
  confirm modal is still open — re-enabling is the modal exit's job
  (``cancelAndClose``), otherwise a re-click stacks a second confirm modal
  over the first (the exact bug 37d122f claimed to fix).
- M2: ``pollTranscribeJob`` must retry a bounded number of consecutive
  transient failures on the SAME job_id instead of abandoning a still-running
  paid job on the first blip; after giving up it must not hand the trigger
  back (the job may still be running server-side — a re-click would start a
  duplicate paid job).
- M3: the overlay backdrop must not dismiss the modal while a job is in
  flight — after dismissal ``fail()`` would render into a detached modal and
  the terminal failure would surface nowhere.
- L1: while one confirm modal is open, a sibling row's Transcribe click must
  not stack a second modal (module-level open-modal guard).
- L4: a failure must hide the progress bar — a frozen bar behind an error
  reads as "still working".
- L5: an unrecognised job status must not poll forever (bounded backstop).

Following the ``test_ui_console_lint_remediation.py`` pattern: text-level
assertions against the production console.html — no DOM, no fetch, no
browser.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _function_body(name: str, args_pattern: str = r"[^)]*") -> str:
    """Return the source of ``function <name>(...) { ... }`` (brace-naive:
    match up to the first line-anchored closing brace, same convention as the
    sibling UI test files)."""
    text = _console_text()
    match = re.search(rf"function {name}\({args_pattern}\) \{{(.*?)\n\}}", text, re.DOTALL)
    assert match is not None, f"console.html must define function {name}"
    return match.group(1)


def _fail_body() -> str:
    """The fail() helper nested inside openTranscribeConfirm's confirm handler."""
    body = _function_body("openTranscribeConfirm")
    match = re.search(r"function fail\((.*?)\) \{(.*?)\n    \}", body, re.DOTALL)
    assert match is not None, "openTranscribeConfirm must define a fail() helper"
    return match.group(2)


# ---------------------------------------------------------------------------
# M1 — failure paths leave the trigger disabled; modal exit owns re-enabling
# ---------------------------------------------------------------------------


def test_fail_does_not_reenable_trigger():
    """fail() must not touch triggerBtn — re-enabling it with the modal still
    open lets a re-click stack a second confirm modal for the same source."""
    assert "triggerBtn.disabled = false" not in _fail_body(), (
        "fail() re-enables the trigger while the modal is open (issue #476 M1) — "
        "move re-enabling to cancelAndClose"
    )


def test_modal_exit_owns_trigger_reenable():
    """cancelAndClose re-enables the trigger only when nothing can still be
    running: never-submitted, or a server-terminal failure."""
    body = _function_body("openTranscribeConfirm")
    match = re.search(r"function cancelAndClose\(\) \{(.*?)\n  \}", body, re.DOTALL)
    assert match is not None, "openTranscribeConfirm must define cancelAndClose"
    cac = match.group(1)
    assert "triggerBtn.disabled = false" in cac, (
        "cancelAndClose must own re-enabling the trigger (issue #476 M1)"
    )
    assert "jobOutcome" in cac, (
        "cancelAndClose must gate re-enabling on the job outcome — an abandoned "
        "poll means the job may still be running server-side (issue #476 M2)"
    )


# ---------------------------------------------------------------------------
# M2 — bounded poll retry on the same job_id; no early abandon
# ---------------------------------------------------------------------------


def test_poll_retries_transient_failures():
    """The shared job poller retries a bounded number of consecutive failures
    on the same job_id before surfacing onError. (The retry discipline moved
    from pollTranscribeJob into the generic pollJobStatus when the Import job
    joined the same submit/poll pattern — issue #497; the M2 guarantee is
    unchanged, transcribe polls THROUGH this function.)"""
    body = _function_body("pollJobStatus")
    assert "consecutiveFailures" in body, (
        "pollJobStatus must count consecutive poll failures (issue #476 M2)"
    )
    catch_match = re.search(r"\.catch\(function\(err\) \{(.*?)\n      \}\);", body, re.DOTALL)
    assert catch_match is not None
    catch_body = catch_match.group(1)
    assert "setTimeout(tick" in catch_body, (
        "the poll .catch must re-schedule the SAME job's poll (bounded retry), "
        "not abandon a still-running paid job on the first blip (issue #476 M2)"
    )
    assert "onError" in catch_body, (
        "after retries are exhausted the poll must still surface onError honestly"
    )


def test_abandoned_poll_reports_honestly():
    """The poll-abandon path must flag the may-still-be-running state (fail's
    abandoned branch) and both language maps must carry the honest message."""
    text = _console_text()
    body = _function_body("openTranscribeConfirm")
    assert re.search(r"fail\(.*transcribePollLost.*,\s*true\)", body), (
        "the poll onError path must mark the failure as abandoned "
        "(job may still be running server-side) — issue #476 M2"
    )
    assert text.count("transcribePollLost") >= 3, (
        "transcribePollLost label must exist in BOTH LINT_CHROME maps (en+zh) "
        "and be used by the poll-abandon path"
    )


# ---------------------------------------------------------------------------
# M3 — backdrop is inert while the job is in flight
# ---------------------------------------------------------------------------


def test_backdrop_guarded_while_in_flight():
    body = _function_body("openTranscribeConfirm")
    match = re.search(
        r'overlay\.addEventListener\("click", function\(e\) \{(.*?)\}\);',
        body,
        re.DOTALL,
    )
    assert match is not None, "openTranscribeConfirm must wire a backdrop handler"
    assert "transcribeInFlight" in match.group(1), (
        "the backdrop handler must be inert while the job is in flight "
        "(issue #476 M3) — otherwise fail() renders into a detached modal"
    )


# ---------------------------------------------------------------------------
# L1 — sibling rows cannot stack a second confirm modal
# ---------------------------------------------------------------------------


def test_sibling_transcribe_click_cannot_stack_modal():
    text = _console_text()
    assert "transcribeModalOpen" in text, (
        "a module-level open-modal flag must exist (issue #476 L1)"
    )
    body = _function_body("transcribeAction")
    assert "transcribeModalOpen" in body, (
        "transcribeAction's click handler must refuse to open a second confirm "
        "modal while one is already open (issue #476 L1)"
    )


# ---------------------------------------------------------------------------
# L4 — failure hides the progress bar
# ---------------------------------------------------------------------------


def test_failure_hides_progress_bar():
    fail_body = _fail_body()
    assert "progressTrackEl.style.display" in fail_body, (
        "fail() must hide the progress track — a frozen bar behind an error "
        "reads as still-working/finished (issue #476 L4)"
    )


# ---------------------------------------------------------------------------
# L5 — unknown job status cannot poll forever
# ---------------------------------------------------------------------------


def test_unknown_status_has_bounded_backstop():
    # Lives in the generic pollJobStatus since issue #497 (see the M2 test's
    # note); transcribe and import both poll through it.
    body = _function_body("pollJobStatus")
    assert "unknownStatusTicks" in body, (
        "pollJobStatus must bound polling on an unrecognised status (issue #476 L5)"
    )
    assert '"submitted"' in body and '"working"' in body, (
        "the non-terminal statuses the poller recognises are the server's "
        'actual Literal["submitted", "working", ...] — not queued/running'
    )
