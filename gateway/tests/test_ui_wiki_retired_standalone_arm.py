"""Structural tests for retiring stack=wiki as the reader's standalone arm
(issue #681, ADR-0045 kill clause).

Following the established pattern in ``test_ui_hybrid_toggle.py`` /
``test_ui_bilingual_starters.py``, these tests inspect the production UI file
(``gateway/static/index.html``) as text -- no DOM, no fetch, no browser, no
OPENAI_API_KEY (§6.3 / §12.7).

Scope (issue #681 AC):
  - The reader's stack toggle no longer offers Wiki as a selectable arm --
    only RAG and Hybrid remain (``segWiki`` and its ``setStack("wiki")``
    dispatch are gone entirely, not merely hidden).
  - The default active stack is ``rag`` (mirrors the server-side default
    flip in ``gateway/app/routes.py``).
  - Hybrid's empty-state copy carries no retrieval/grounding superiority
    claim (ADR-0045 demote clause).
  - The wiki layer is repositioned as the KB-governance workflow
    demonstration (curation, provenance, Console), not a measured quality
    win; retrieval quality is attributed to RAG's dense search.

Out of scope (untouched by this file): the Console page (``console.html``),
governance/import endpoints, and stack=wiki's server-side dispatch --
covered by ``gateway/tests/test_chat_stream.py`` et al.
"""

from __future__ import annotations

from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Wiki is gone as a standalone toggle option -- not merely hidden
# ---------------------------------------------------------------------------


def test_ui_no_setstack_wiki_dispatch():
    """No control in the UI dispatches setStack('wiki') anymore."""
    text = _ui_text()
    assert 'setStack("wiki")' not in text, (
        "the reader must no longer offer a control that selects stack=wiki"
    )


def test_ui_no_segwiki_identifier():
    """The segWiki toggle-button variable is removed entirely (not just unused)."""
    text = _ui_text()
    assert "segWiki" not in text, "segWiki must be fully removed, not merely hidden"


def test_ui_masthead_toggle_has_exactly_rag_and_hybrid():
    """The masthead toggle group wires up only the two surviving segments."""
    text = _ui_text()
    assert "mast-toggle" in text
    assert "segRag" in text and "segHybrid" in text
    # The toggle group's own el() call must not still list a third (wiki) segment.
    assert 'el("div", { class: "mast-toggle", role: "tablist" }, segRag, segHybrid)' in text, (
        "the mast-toggle group must be built from exactly [segRag, segHybrid]"
    )


# ---------------------------------------------------------------------------
# Default active stack is rag (mirrors the server-side default, issue #681)
# ---------------------------------------------------------------------------


def test_ui_default_stack_is_rag():
    """The client-side ``stack`` state variable now defaults to 'rag'."""
    text = _ui_text()
    assert 'var stack = "rag";' in text, (
        "the reader's default active stack must be rag, matching the server default"
    )


def test_ui_stack_meta_has_no_wiki_entry():
    """STACK_META (empty-state per-stack copy) drops the wiki entry."""
    text = _ui_text()
    assert "STACK_META" in text
    # rag and hybrid entries remain (still selectable stacks).
    assert "rag: {" in text
    assert "hybrid: {" in text
    assert "wiki: {" not in text, "STACK_META must not still carry a wiki entry"
    # The fallback for an unrecognised/absent stack now resolves to rag, not wiki.
    assert "STACK_META[stack] || STACK_META.rag" in text


# ---------------------------------------------------------------------------
# Hybrid carries no superiority claim (ADR-0045 demote clause)
# ---------------------------------------------------------------------------

_SUPERIORITY_PHRASES_EN = (
    "outperforms",
    "superior to",
    "better than rag",
    "better than dense",
    "higher quality than",
    "quality win",
    "wins over",
    "more accurate than",
    "beats rag",
)
_SUPERIORITY_PHRASES_ZH = ("勝出", "優於", "更佳", "更準確", "領先", "品質更高")


def test_ui_hybrid_copy_has_no_superiority_claim():
    """No narrative claim of retrieval/grounding superiority survives for hybrid
    (ADR-0045 demote clause: 'every narrative claim of retrieval or grounding
    superiority is dropped').

    Scoped to actual marketing-style superiority phrasing (multi-word, e.g.
    "superior to") rather than a bare "superior"/"win" substring, since that
    would also false-positive on THIS TEST FILE's own explanatory prose
    (and any future code comment documenting the absence of such a claim).
    """
    text = _ui_text().lower()
    for kw in _SUPERIORITY_PHRASES_EN:
        assert kw not in text, f"superiority phrase found in demo copy: {kw!r}"
    raw_text = _ui_text()
    for kw in _SUPERIORITY_PHRASES_ZH:
        assert kw not in raw_text, f"superiority phrase found in demo copy: {kw!r}"


# ---------------------------------------------------------------------------
# The wiki layer is repositioned as the governance-workflow demonstration
# ---------------------------------------------------------------------------


def test_ui_wiki_layer_repositioned_as_governance_demo():
    """The empty-state curation-loop copy names the wiki layer as this demo's
    governance/curation surface, and attributes retrieval quality to RAG --
    per the ADR-0045 demote clause wording, not a measured quality win."""
    text = _ui_text()
    assert "curated wiki layer" in text, (
        "the loop-note copy must still reference the curated wiki layer"
    )
    assert "governance" in text.lower(), (
        "demo copy must reposition the wiki layer as a governance-workflow demonstration"
    )
    # English loop-note copy: retrieval quality is attributed to RAG, not wiki.
    assert (
        "Retrieval quality comes from RAG" in text
        or "retrieval quality comes from rag" in text.lower()
    )
