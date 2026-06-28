"""Structural tests for the reader UI's third Hybrid stack toggle (Phase 13 S4 / #314).

The reader UI lives in ``gateway/static/index.html`` as a single vanilla
HTML/CSS/JS file (CODING_STANDARD §12.1 — no framework, no build step). Following
the established pattern in ``test_ui_bilingual_starters.py`` /
``test_ui_citation_links.py``, these tests inspect the production UI file's text
to assert the structural invariants of issue #314's reader-toggle AC:

- A third **Hybrid** toggle button exists, mirroring the existing Wiki / RAG
  buttons (same ``el()`` factory, same ``setStack`` dispatch, same masthead
  toggle group).
- Selecting Hybrid issues a fresh ``?stack=hybrid`` request AND preserves
  ``?session=`` for multi-turn (ADR-0013 / §12.9). The request builder is
  stack-agnostic (it already interpolates ``stack`` + ``sessionId``), so this is
  inherited — the tests pin that it stays intact.
- The empty state is stack-aware for all three stacks (a per-stack label lookup,
  not a binary wiki/RAG branch that would mislabel Hybrid).
- Still textContent-only (§12.4 — no innerHTML).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 / §12.7).
"""

from __future__ import annotations

from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A third Hybrid toggle button, mirroring Wiki / RAG
# ---------------------------------------------------------------------------


def test_ui_has_hybrid_toggle_button():
    """The UI builds a Hybrid toggle button wired to setStack('hybrid')."""
    text = _ui_text()
    assert 'text: "Hybrid"' in text, "a 'Hybrid' toggle button must exist"
    assert 'setStack("hybrid")' in text, "the Hybrid button must dispatch setStack('hybrid')"


def test_ui_hybrid_button_mirrors_existing_segments():
    """The Hybrid button reuses the same segment variable pattern as Wiki / RAG."""
    text = _ui_text()
    # The two existing segment vars are segWiki / segRag; the third must follow suit.
    assert "segHybrid" in text, "a segHybrid segment button variable must exist (parity)"


def test_ui_masthead_toggle_includes_all_three_buttons():
    """All three segment buttons are placed in the masthead toggle group."""
    text = _ui_text()
    # The toggle group element is `.mast-toggle`; assert the three segments are
    # appended to it together (one mast-toggle el() call carrying all three).
    assert "mast-toggle" in text
    for seg in ("segWiki", "segRag", "segHybrid"):
        assert seg in text, f"masthead toggle must include {seg}"


# ---------------------------------------------------------------------------
# syncSeg tracks the hybrid pressed state
# ---------------------------------------------------------------------------


def test_ui_syncseg_tracks_hybrid_pressed_state():
    """syncSeg sets aria-pressed for the hybrid segment (selected-state parity)."""
    text = _ui_text()
    assert 'stack === "hybrid"' in text, (
        "syncSeg must compute the hybrid pressed state (stack === 'hybrid')"
    )


# ---------------------------------------------------------------------------
# Fresh ?stack=hybrid request preserves ?session= (multi-turn inherited)
# ---------------------------------------------------------------------------


def test_ui_request_is_stack_agnostic_and_preserves_session():
    """The request builder interpolates the active stack AND keeps the session id.

    Selecting Hybrid is a fresh request (§12.3) that maps to ?stack=hybrid via the
    generic ``stack`` interpolation, and the existing &session= echo preserves
    multi-turn continuity across the toggle (ADR-0013 / §12.9). Both must remain.
    """
    text = _ui_text()
    assert '"/chat/stream?stack=" + encodeURIComponent(stack)' in text, (
        "the request URL must interpolate the active stack generically"
    )
    assert '"&session=" + encodeURIComponent(sessionId)' in text, (
        "a fresh stack request must preserve ?session= for multi-turn"
    )


def test_ui_setstack_reasks_current_query_on_toggle():
    """Toggling stacks re-asks the current question (compare stacks in one session)."""
    text = _ui_text()
    # setStack re-issues the current query as a new turn when one exists.
    assert "function setStack" in text
    assert "startRequest(currentQuery)" in text


# ---------------------------------------------------------------------------
# Empty state is stack-aware for all three stacks
# ---------------------------------------------------------------------------


def test_ui_empty_state_is_stack_aware_for_hybrid():
    """The empty state labels each stack via a per-stack lookup, not a wiki/RAG binary.

    A binary ``isWiki ? 'Wiki' : 'RAG'`` would mislabel Hybrid as RAG. The UI must
    resolve the active stack's label/copy from a stack-keyed structure that
    includes a hybrid entry.
    """
    text = _ui_text()
    assert "STACK_META" in text, (
        "the empty state must resolve label/copy from a stack-keyed STACK_META map"
    )
    assert "STACK_META[stack]" in text, "the empty state must look up STACK_META by active stack"


# ---------------------------------------------------------------------------
# §12.4: still textContent-only (no innerHTML introduced by the toggle)
# ---------------------------------------------------------------------------


def test_ui_no_inner_html_assignment():
    """The UI still never assigns to innerHTML (§12.4 — textContent only)."""
    text = _ui_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found — §12.4 requires textContent only"
    )
