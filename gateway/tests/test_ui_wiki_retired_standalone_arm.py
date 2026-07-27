"""Structural tests for the reader UI's stack toggle + Governance-mode demo entry.

Two eras of the same ADR-0045 intent are pinned here, in one file, because
issue #687 explicitly authorises editing this file rather than replacing it:

  - Issue #681 (kill clause): stack=wiki is retired as an unlabeled PEER
    inside the retrieval-quality toggle. The masthead's quality toggle group
    (segRag / segHybrid) never grows a third segment, and the default active
    stack is rag.
  - Issue #687 (demote clause): stack=wiki is restored as a distinct,
    explicitly-labeled Governance-mode DEMO entry -- a separate control with
    its own copy, reachable outside the quality toggle, carrying zero
    retrieval/grounding superiority claim. What stays banned is an unlabeled
    wiki PEER inside the quality toggle group; what's now required is an
    out-of-band governance control that reaches the same stack=wiki dispatch
    under an honest "this demonstrates curation, not retrieval quality" frame.

Following the established pattern in ``test_ui_hybrid_toggle.py`` /
``test_ui_bilingual_starters.py``, these tests inspect the production UI file
(``gateway/static/index.html``) as text -- no DOM, no fetch, no browser, no
OPENAI_API_KEY (§6.3 / §12.7).

Scope (issue #681 AC, still pinned):
  - The quality toggle group is built from exactly [segRag, segHybrid] --
    segWiki never comes back.
  - The default active stack is ``rag`` (mirrors the server-side default
    flip in ``gateway/app/routes.py``).
  - Hybrid's empty-state copy carries no retrieval/grounding superiority
    claim (ADR-0045 demote clause).

Scope (issue #687 AC, new):
  - A Governance-mode control exists, distinct from the quality toggle
    (separate variable + CSS class, not a member of the mast-toggle group),
    and dispatches setStack("wiki").
  - Governance copy (en + zh) frames the wiki stack as a KB-governance
    workflow demonstration -- curation, provenance, filing -- and references
    the corpus v3 verdict for where retrieval quality is actually measured.
  - No superiority claim survives anywhere in the file (grep-level check,
    both languages), now that a wiki entry exists again.

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
# The quality toggle stays exactly [segRag, segHybrid] -- no wiki PEER
# ---------------------------------------------------------------------------


def test_ui_no_segwiki_identifier():
    """The segWiki toggle-button variable stays removed (governance uses govBtn)."""
    text = _ui_text()
    assert "segWiki" not in text, "segWiki must not come back as a quality-toggle segment"


def test_ui_masthead_toggle_has_exactly_rag_and_hybrid():
    """The masthead quality-toggle group wires up only the two ranking segments."""
    text = _ui_text()
    assert "mast-toggle" in text
    assert "segRag" in text and "segHybrid" in text
    # The toggle group's own el() call must not list a third (wiki) segment --
    # the governance control is appended as a sibling, not a group member.
    assert 'el("div", { class: "mast-toggle", role: "tablist" }, segRag, segHybrid)' in text, (
        "the mast-toggle group must be built from exactly [segRag, segHybrid]"
    )


# ---------------------------------------------------------------------------
# Default active stack is rag (mirrors the server-side default, issue #681)
# ---------------------------------------------------------------------------


def test_ui_default_stack_is_rag():
    """The client-side ``stack`` state variable defaults to 'rag'."""
    text = _ui_text()
    assert 'var stack = "rag";' in text, (
        "the reader's default active stack must be rag, matching the server default"
    )


# ---------------------------------------------------------------------------
# Governance-mode control: exists, is distinct from the quality toggle,
# and dispatches setStack("wiki") (issue #687)
# ---------------------------------------------------------------------------


def test_ui_governance_control_exists_and_dispatches_wiki():
    """A dedicated govBtn control drives stack=wiki, outside the quality toggle."""
    text = _ui_text()
    assert "govBtn" in text, "a govBtn governance-mode control variable must exist"
    assert 'setStack(stack === "wiki" ? "rag" : "wiki")' in text, (
        'the governance control must dispatch setStack("wiki") (toggling back to rag)'
    )


def test_ui_governance_control_is_visually_and_semantically_separate():
    """govBtn is its own CSS class, appended as a masthead SIBLING of mast-toggle,
    not a member of the toggle group's own el() children list (ADR-0045 demote
    clause: "own control or badge... not a third segment in the same group")."""
    text = _ui_text()
    assert 'class: "mast-governance"' in text, (
        "the governance control must carry its own class, distinct from mast-toggle button"
    )
    # The masthead appends govBtn as a sibling after the toggle group, not inside it.
    assert (
        'el("div", { class: "mast-toggle", role: "tablist" }, segRag, segHybrid),\n    govBtn,'
    ) in text, "govBtn must be appended to the masthead as a sibling of mast-toggle, not inside it"


def test_ui_governance_control_tracks_its_own_pressed_state():
    """syncSeg syncs govBtn's aria-pressed to stack === 'wiki' (parity with the
    quality-toggle segments' own pressed-state tracking)."""
    text = _ui_text()
    assert 'govBtn.setAttribute("aria-pressed", String(stack === "wiki"))' in text


# ---------------------------------------------------------------------------
# No superiority claim survives for hybrid OR wiki (ADR-0045 demote clause)
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
    OR wiki (ADR-0045 demote clause: 'every narrative claim of retrieval or
    grounding superiority is dropped').

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
# STACK_META carries a governance-framed (not quality-framed) wiki entry
# ---------------------------------------------------------------------------


def test_ui_stack_meta_wiki_entry_is_governance_framed():
    """STACK_META's wiki entry exists again (issue #687), but its label and copy
    frame it as a governance demo, never as a retrieval-quality option.

    Issue #687 explicitly authorises this reversal of the #681-era assertion
    that STACK_META must NOT carry a wiki entry: the demote clause allows (and
    requires) the entry's return under an honest governance-only frame."""
    text = _ui_text()
    assert "STACK_META" in text
    assert "rag: {" in text
    assert "hybrid: {" in text
    assert "wiki: {" in text, "STACK_META must carry a governance-framed wiki entry (issue #687)"
    # The fallback for an unrecognised/absent stack still resolves to rag.
    assert "STACK_META[stack] || STACK_META.rag" in text
    # The label reads as a governance concept, not a retrieval-quality brand
    # name (contrast STACK_META.rag.label: "RAG", STACK_META.hybrid.label: "Hybrid").
    assert 'label: "Governance"' in text


def test_ui_governance_copy_frames_curation_workflow_bilingual():
    """The wiki entry's sub copy names the governance workflow -- curation,
    filing, the console -- in BOTH languages, not just one."""
    text = _ui_text()
    assert "KB-governance workflow" in text, (
        "English governance copy must name the KB-governance workflow"
    )
    assert "治理工作流程" in text, "Chinese governance copy must name the governance workflow"
    assert "draft QA page" in text and "草稿 QA 頁面" in text, (
        "governance copy must describe the filing outcome (draft QA page) bilingually"
    )


def test_ui_governance_copy_references_corpus_v3_verdict():
    """Governance copy points to the corpus v3 verdict as where retrieval
    quality is actually measured (issue #687 AC: 'links or references the
    corpus v3 verdict for retrieval quality'), in both languages."""
    text = _ui_text()
    assert text.count("corpus v3 verdict") >= 2, (
        "both the English and Chinese governance copy must reference the corpus v3 verdict"
    )


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
