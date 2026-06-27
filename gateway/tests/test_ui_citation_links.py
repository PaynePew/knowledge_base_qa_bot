"""Structural tests for clickable citation links in the reader UI (#266).

The reader UI lives in ``gateway/static/index.html`` as a single vanilla
HTML/CSS/JS file (CODING_STANDARD §12.1 — no framework, no build step). Mirroring
``test_ui_bilingual_starters.py`` / ``test_sse_parser.py``, these tests inspect
the production UI file's text to assert the structural invariants of issue #266:

- A retrieved record's citation renders as a CLICKABLE element (``.cite-link``)
  that opens the grounding source IN-PAGE — not a plain ``<a href>`` that
  navigates the browser away to the raw ``/read/file`` JSON (AC1).
- The viewer fetches ``/read/file`` and shows the Markdown body with the YAML
  frontmatter stripped, rendered via ``textContent`` (§12.4 — never innerHTML).
- The link target is the server-supplied ``source.path`` (§12.5 — the resolvable
  path is a server decision; the client does not construct ``docs/``/``wiki/``
  paths from bare slugs/filenames, which would 404 on sub-foldered corpora).
- RWD: desktop renders an inline card; ``≤540px`` promotes it to a bottom sheet
  (``position: fixed`` + ``.cite-scrim`` + ``dvh`` height + safe-area inset).

No DOM, no fetch, no browser — fully hermetic (§6.3 / §12.7): pure text assertions
on the production UI source.
"""

from __future__ import annotations

from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC1: the citation is clickable and opens an in-page viewer (not a navigation)
# ---------------------------------------------------------------------------


def test_citation_renders_clickable_link_class():
    """A record's citation renders with the ``.cite-link`` affordance class (AC1)."""
    text = _ui_text()
    assert "cite-link" in text, "citation must render as a clickable .cite-link element"


def test_citation_opens_inpage_viewer_not_navigation():
    """Clicking a citation opens an in-page viewer via openCitation(), not <a href>.

    /read/file returns JSON ({relpath, content}), so a bare ``<a href>`` would
    dump raw JSON in a new tab and leave the chat. The fix must route through an
    in-page fetch+render helper instead.
    """
    text = _ui_text()
    assert "function openCitation" in text, "an openCitation() in-page viewer helper must exist"
    assert "openCitation(" in text, "the citation click must call openCitation()"
    assert 'fetch("/read/file' in text, "the viewer must fetch /read/file in-page"


def test_citation_viewer_strips_frontmatter():
    """The viewer shows the Markdown body with YAML frontmatter stripped."""
    text = _ui_text()
    assert "function stripFrontmatter" in text, "a stripFrontmatter() helper must exist"


def test_citation_link_target_is_server_supplied_path():
    """The link target is the server-supplied source.path, not a client-built path (§12.5)."""
    text = _ui_text()
    assert "s.path" in text, (
        "the citation link must use the server-supplied source.path "
        "(client must not construct docs/wiki paths from bare slugs)"
    )


# ---------------------------------------------------------------------------
# §12.4: still textContent-only (no innerHTML introduced by the citation viewer)
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment():
    """The UI still never assigns to innerHTML (§12.4)."""
    text = _ui_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found — §12.4 requires textContent only"
    )


# ---------------------------------------------------------------------------
# RWD: inline card on desktop, bottom sheet on mobile (≤540px)
# ---------------------------------------------------------------------------


def test_rwd_viewer_and_scrim_styled():
    """The viewer (.cite-viewer) and its mobile scrim (.cite-scrim) are styled."""
    text = _ui_text()
    assert ".cite-viewer" in text, "missing .cite-viewer styles"
    assert ".cite-scrim" in text, "missing .cite-scrim (mobile overlay) styles"
    assert ".cite-content" in text, "missing .cite-content styles"


def test_rwd_mobile_bottom_sheet():
    """≤540px promotes the viewer to a bottom sheet (fixed, dvh height, safe-area)."""
    text = _ui_text()
    assert "@keyframes cite-slide" in text, "bottom-sheet slide-in animation missing"
    assert "position: fixed" in text, "the mobile bottom sheet must be position: fixed"
    assert "dvh" in text, "viewer height must use dvh units (mobile browser chrome safe)"
    assert "env(safe-area-inset-bottom)" in text, (
        "the bottom sheet must respect the iOS home-indicator safe area"
    )


def test_reduced_motion_respected():
    """The global prefers-reduced-motion guard still neutralises animations."""
    text = _ui_text()
    assert "prefers-reduced-motion" in text, "prefers-reduced-motion guard missing"
