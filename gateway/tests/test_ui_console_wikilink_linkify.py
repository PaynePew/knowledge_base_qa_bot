"""Structural tests for wikilink linkify in the Operator Console ``/read/file``
viewer (issue #410, ADR-0030 decision 5 — curator surface).

Following the pattern in ``test_ui_console_assign_alias.py`` / ``test_ui_
citation_links.py``, these tests inspect the production ``gateway/static/
console.html`` file's text to assert the structural invariants of issue #410:

- ``openFile`` fetches the shared ``GET /wiki/pages/resolution-map`` endpoint
  alongside ``GET /read/file`` (one resolver, ADR-0030 Invariant — no surface
  builds its own slug set).
- A resolved wikilink renders as a ``.wikilink`` navigable link; an
  unresolved one renders as ``.wikilink-unresolved`` — red-link styling
  (``var(--fail)``), the SAME judgment C2 reports for the same corpus.
- The client never constructs a wiki path from a bare slug (§12.5) — the
  navigate target is always the server-supplied relpath from ``slugs``.
- No ``innerHTML`` is introduced (§12.4) — linkify builds nodes via ``el()``
  / ``document.createTextNode``, not string concatenation into markup.

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 / §12.7).
DOM rendering / the click -> navigate loop is verified manually per §12.7
(visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The shared resolution-map endpoint is consulted, not a locally-built slug set
# ---------------------------------------------------------------------------


def test_fetch_resolution_map_hits_the_one_shared_endpoint():
    text = _console_text()
    assert "function fetchResolutionMap()" in text
    fn_src = text.split("function fetchResolutionMap()")[1][:400]
    assert '"/wiki/pages/resolution-map"' in fn_src


def test_open_file_fetches_resolution_map_alongside_read_file():
    text = _console_text()
    fn_src = text.split("function openFile(relpath, name)")[1]
    assert "fetchResolutionMap()" in fn_src
    assert 'fetch("/read/file?' in fn_src
    assert "Promise.all(" in fn_src, "both fetches must resolve before rendering the viewer"


# ---------------------------------------------------------------------------
# Resolved -> navigable link; unresolved -> red-link styling (curator surface)
# ---------------------------------------------------------------------------


def test_render_wikilink_nodes_distinguishes_resolved_from_unresolved():
    text = _console_text()
    assert "function renderWikilinkNodes(content, resolutionMap)" in text
    fn_src = text.split("function renderWikilinkNodes(content, resolutionMap)")[1].split("\n}")[0]
    assert 'class: "wikilink"' in fn_src, "resolved wikilinks must get the .wikilink class"
    assert '"wikilink-unresolved"' in fn_src, (
        "unresolved wikilinks must get the .wikilink-unresolved red-link class"
    )


def test_unresolved_wikilink_css_uses_the_shared_fail_token():
    text = _console_text()
    assert ".wikilink-unresolved" in text
    css_block = text.split(".wikilink-unresolved")[1][:200]
    assert "var(--fail)" in css_block, (
        "unresolved wikilinks must render in the SAME red used for C2 findings"
    )


def test_resolved_wikilink_click_reopens_the_file_viewer():
    text = _console_text()
    fn_src = text.split("function renderWikilinkNodes(content, resolutionMap)")[1].split("\n}")[0]
    assert "openFile(relpath" in fn_src, "clicking a resolved wikilink must navigate in-viewer"


# ---------------------------------------------------------------------------
# Client never constructs a wiki path from a bare slug (§12.5)
# ---------------------------------------------------------------------------


def test_resolve_wikilink_path_reads_server_supplied_relpath_not_building_one():
    text = _console_text()
    assert "function resolveWikilinkPath(slug, resolutionMap)" in text
    fn_src = text.split("function resolveWikilinkPath(slug, resolutionMap)")[1].split("\n}")[0]
    assert '"wiki/' not in fn_src, (
        "the resolver must return the server-supplied relpath verbatim, "
        "never construct a wiki/<subdir>/<slug>.md path itself"
    )


# ---------------------------------------------------------------------------
# §12.4: still textContent/safe-DOM-construction only (no innerHTML introduced)
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found — §12.4 requires textContent / safe DOM construction"
    )
