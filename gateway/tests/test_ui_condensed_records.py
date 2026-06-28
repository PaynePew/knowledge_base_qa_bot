"""Structural tests for the condensed RECORDS layout in the reader UI.

Follow-up to #266 (clickable citations). The retrieved-record cards used to render
the full section ``content`` snippet inline, *and* clicking the citation opened the
same source in the #266 viewer — the same body of text shown twice (once before the
answer, once on click). This was redundant ("畫蛇添足") and made the chat noisy.

The condensed design (direction A): each record is a single compact, clickable
reference — ``num · citation-id · heading · provenance`` — with NO always-visible
content snippet. The full source is reached on demand through the existing #266
citation viewer (``openCitation``), which becomes the single content surface.

Like ``test_ui_citation_links.py`` these are pure text assertions on the production
UI source (``gateway/static/index.html``) — no DOM, no browser, fully hermetic
(§6.3 / §12.7).
"""

from __future__ import annotations

from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Core: no detailed content is shown by default ("不顯示詳細內容")
# ---------------------------------------------------------------------------


def test_record_has_no_inline_content_snippet():
    """A record no longer renders the section ``content`` as an inline snippet.

    The ``.snip`` element was the duplicated content surface; it must be gone so
    the body text lives only behind the citation viewer.
    """
    text = _ui_text()
    assert 'class: "snip"' not in text, (
        "records must not render an inline content snippet — the full content "
        "belongs to the citation viewer (#266), shown on demand only"
    )


def test_record_does_not_render_source_content_field():
    """The per-record body must not bind ``s.content`` (the section text) at all."""
    text = _ui_text()
    assert "s.content" not in text, (
        "s.content (the retrieved section body) must not be rendered into the "
        "condensed record — it is reached via the citation viewer instead"
    )


# ---------------------------------------------------------------------------
# The condensed record is itself the citation trigger (whole-row clickable)
# ---------------------------------------------------------------------------


def test_record_row_is_a_clickable_citation():
    """The whole record row is the clickable citation affordance, not just the id.

    The row carries the ``cite-link`` class (so the #266 affordance/test contract
    holds) and is a real ``<button>`` for native focus + Enter/Space semantics.
    """
    text = _ui_text()
    assert "rec cite-link" in text, (
        "the condensed record row must be a single clickable .cite-link element"
    )


def test_record_click_opens_server_supplied_path():
    """Clicking the row opens the server-supplied path via the in-page viewer (§12.5)."""
    text = _ui_text()
    assert "openCitation(s.path" in text, (
        "the record row click must open the source via openCitation(s.path, …) — "
        "the resolvable path is a server decision, not a client-built slug"
    )


# ---------------------------------------------------------------------------
# Provenance survives the condensing (derived_from still shown)
# ---------------------------------------------------------------------------


def test_record_keeps_provenance():
    """Condensing drops the snippet but keeps the ``derived_from`` provenance line."""
    text = _ui_text()
    assert "s.derived_from" in text, (
        "the condensed record must still surface derived_from provenance"
    )
