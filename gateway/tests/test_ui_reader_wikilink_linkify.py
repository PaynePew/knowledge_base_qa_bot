"""Structural tests for wikilink linkify in the reader UI (issue #410,
ADR-0030 decision 5 — reader surface: chat answer bodies + the chat-side
citation viewer).

Following the pattern in ``test_ui_citation_links.py``, these tests inspect
the production ``gateway/static/index.html`` file's text to assert the
structural invariants of issue #410:

- Resolvable wikilinks (chat answers AND the citation viewer body) render as
  a ``.wikilink`` link that opens the page view by reusing ``openCitation``
  — the SAME in-page viewer citations already use, not a second navigation
  mechanism.
- Unresolvable wikilinks degrade to PLAIN TEXT on the reader surface — ZERO
  red styling, no wrapper element at all (a reader can fix nothing; red
  would only be noise).
- The chat answer body linkifies once, at ``onDone`` — never mid-stream,
  since a ``[[wikilink]]`` can split across SSE token boundaries.
- The client never constructs a wiki path from a bare slug (§12.5) — the
  navigate target is always the server-supplied relpath from the shared
  resolution map.
- No ``innerHTML`` is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 / §12.7).
"""

from __future__ import annotations

from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The shared resolution-map endpoint is consulted, not a locally-built slug set
# ---------------------------------------------------------------------------


def test_fetch_resolution_map_hits_the_one_shared_endpoint():
    text = _ui_text()
    assert "function fetchResolutionMap()" in text
    fn_src = text.split("function fetchResolutionMap()")[1][:400]
    assert '"/wiki/pages/resolution-map"' in fn_src


def test_open_citation_fetches_resolution_map_alongside_read_file():
    text = _ui_text()
    fn_src = text.split("function openCitation(path, label, triggerEl)")[1]
    assert "fetchResolutionMap()" in fn_src
    assert 'fetch("/read/file?' in fn_src
    assert "Promise.all(" in fn_src


# ---------------------------------------------------------------------------
# Resolved -> a .wikilink that opens the page view via the EXISTING viewer
# ---------------------------------------------------------------------------


def test_render_wikilink_nodes_defined():
    text = _ui_text()
    assert "function renderWikilinkNodes(content, resolutionMap)" in text


def test_resolved_wikilink_reuses_open_citation_not_a_new_navigation():
    text = _ui_text()
    fn_src = text.split("function renderWikilinkNodes(content, resolutionMap)")[1].split("\n}")[0]
    assert 'class: "wikilink"' in fn_src
    assert "openCitation(relpath" in fn_src, (
        "a resolved wikilink must reuse the existing in-page citation viewer, "
        "not invent a second navigation mechanism"
    )


# ---------------------------------------------------------------------------
# AC: unresolved wikilinks are PLAIN TEXT on the reader surface — zero red
# ---------------------------------------------------------------------------


def test_unresolved_wikilink_returns_plain_text_node_no_wrapper():
    text = _ui_text()
    fn_src = text.split("function renderWikilinkNodes(content, resolutionMap)")[1].split("\n}")[0]
    assert "return document.createTextNode(seg.display);" in fn_src, (
        "an unresolved wikilink must degrade to a plain Text node — no span, no class, no styling"
    )


def test_reader_surface_never_uses_the_red_link_class():
    """Reader surface must carry ZERO red/unresolved styling (curator-only concept)."""
    text = _ui_text()
    assert "wikilink-unresolved" not in text, (
        "the reader UI must not define/use the red-link class — that is the "
        "Console (curator surface) affordance only"
    )


# ---------------------------------------------------------------------------
# Chat answer body linkifies ONCE at onDone, never mid-stream
# ---------------------------------------------------------------------------


def test_on_token_accumulates_raw_text_without_linkifying():
    text = _ui_text()
    # The token param is ``tok`` (not ``t``) so it never shadows the ``t()``
    # chrome-lookup helper the reader now uses inside the same function.
    fn_src = text.split("function onToken(tok)")[1].split("\nfunction ")[0]
    assert "answerRawText +=" in fn_src
    assert "parseWikilinks" not in fn_src, "linkify must not run per-token (mid-stream)"


def test_on_done_linkifies_the_complete_answer_once():
    text = _ui_text()
    fn_src = text.split("function onDone(d)")[1].split("\nfunction ")[0]
    assert "linkifyAnswerBody(answerBody, answerRawText)" in fn_src


def test_linkify_answer_body_captures_target_by_argument_not_shared_state():
    """A stale fetch must never overwrite a LATER turn's answer (race guard)."""
    text = _ui_text()
    assert "function linkifyAnswerBody(targetBody, rawText)" in text
    fn_src = text.split("function linkifyAnswerBody(targetBody, rawText)")[1].split("\n}")[0]
    assert "targetBody.replaceChildren" in fn_src


# ---------------------------------------------------------------------------
# Client never constructs a wiki path from a bare slug (§12.5)
# ---------------------------------------------------------------------------


def test_resolve_wikilink_path_reads_server_supplied_relpath_not_building_one():
    text = _ui_text()
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
    text = _ui_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found — §12.4 requires textContent / safe DOM construction"
    )
