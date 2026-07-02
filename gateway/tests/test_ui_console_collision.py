"""Structural tests for the Operator Console C4 collision dual-resolution UI
(tier-B S2, issue #378, ADR-0028).

Following the pattern in ``test_ui_console_reconcile.py``, these tests
inspect the production ``gateway/static/console.html`` file's text to assert
the structural invariants of the C4 Merge / Differentiate flow:

- The C4 row renders BOTH real "Merge" and "Differentiate" buttons (not the
  disabled tier-b-btn every other Authored-tier check still renders —
  C1/C2 unchanged, see ``test_ui_console_reconcile.py``).
- Each choice opens a generate -> preview/edit -> apply flow wired to its
  own pair of new endpoints, never recomputing the content hash client-side
  (CODING_STANDARD §12.5 — hash values are forwarded verbatim from the
  generate response).
- A 409 inbound-reference guard refusal on merge renders the referrers
  HONESTLY (who references what), distinct from a generic hash-mismatch
  refusal.
- A successful apply re-runs ``POST /wiki/lint?include_c5=false`` and
  re-renders both the Lint card and the Curation Queue (matching every
  other remediation's success path).
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the click -> preview -> apply -> re-lint loop is
verified manually per §12.7 (visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C4 row wires BOTH real collision actions, not the disabled tier-b-btn
# ---------------------------------------------------------------------------


def test_c4_row_wires_collision_action_not_tier_b():
    text = _console_text()
    c4_row = re.search(r"C4:\s*function\(i, f\)\s*\{(.*?)\},", text, re.DOTALL)
    assert c4_row is not None
    assert "collisionAction(f.base_slug, f.pages_in_group)" in c4_row.group(1), (
        "C4 finding rows must wire the real collision dual-resolution action (issue #378 AC)"
    )
    assert "tierBAffordance()" not in c4_row.group(1), (
        "C4 no longer renders the disabled tier-B placeholder — it ships both real affordances"
    )


def test_collision_action_renders_both_choices():
    text = _console_text()
    fn = re.search(
        r"function collisionAction\(baseSlug, pagesInGroup\)\s*\{(.*?)\n  \}", text, re.DOTALL
    )
    assert fn is not None
    body = fn.group(1)
    assert '"collision-merge-btn"' in body
    assert '"collision-differentiate-btn"' in body
    assert 'openCollisionModal("merge"' in body
    assert 'openCollisionModal("differentiate"' in body


def test_collision_chrome_defined():
    text = _console_text()
    assert re.search(r"collisionMerge:\s*\"Merge\"", text)
    assert re.search(r"collisionDifferentiate:\s*\"Differentiate\"", text)
    assert re.search(r"collisionApply:\s*\"Apply\"", text)
    assert re.search(r"collisionCancel:\s*\"Cancel\"", text)


# ---------------------------------------------------------------------------
# Endpoint wiring — the four new tier-B endpoints, hash forwarded verbatim
# ---------------------------------------------------------------------------


def test_generate_and_apply_endpoints_are_wired():
    text = _console_text()
    for path in (
        "/wiki/pages/collision/merge",
        "/wiki/pages/collision/merge/apply",
        "/wiki/pages/collision/differentiate",
        "/wiki/pages/collision/differentiate/apply",
    ):
        assert f'"{path}"' in text, f"{path} must be wired somewhere in the console"


def test_merge_apply_forwards_hash_tokens_verbatim_never_recomputed():
    """CODING_STANDARD §12.5 — the client forwards the server-issued hash
    tokens unchanged, never computes its own hash of the (possibly edited)
    textarea content."""
    text = _console_text()
    fn = re.search(r"function renderCollisionMergePreview\(.*?\n\}", text, re.DOTALL)
    assert fn is not None
    body = fn.group(0)
    assert "hash_base: data.hash_base" in body
    assert "hash_variants: data.hash_variants" in body
    assert "sha256" not in body.lower(), "the client must never compute its own content hash"


def test_differentiate_apply_forwards_hash_tokens_verbatim_never_recomputed():
    text = _console_text()
    fn = re.search(r"function renderCollisionDifferentiatePreview\(.*?\n\}", text, re.DOTALL)
    assert fn is not None
    body = fn.group(0)
    assert "hashes: data.hashes" in body
    assert "sha256" not in body.lower(), "the client must never compute its own content hash"


# ---------------------------------------------------------------------------
# Guard-refusal rendering — honest, not generic
# ---------------------------------------------------------------------------


def test_merge_apply_renders_guard_refusal_referrers_honestly():
    """A 409 with a dict `detail.referrers` (the inbound-reference guard) is
    distinguished from a plain-string hash-mismatch 409 and rendered as a
    per-variant referrer list, not a generic error (ADR-0028 AC)."""
    text = _console_text()
    fn = re.search(r"function renderCollisionMergePreview\(.*?\n\}", text, re.DOTALL)
    assert fn is not None
    body = fn.group(0)
    assert "detail.referrers" in body
    assert "collision-referrers" in body
    assert "r.wiki_referrers" in body
    assert "r.qa_referrers" in body


# ---------------------------------------------------------------------------
# Success path — re-lint (fast) + re-render, same convention as every other
# remediation (issue #363/#376)
# ---------------------------------------------------------------------------


def test_collision_apply_success_relints_fast_and_rerenders():
    text = _console_text()
    fn = re.search(r"function _finishCollisionApply\(.*?\n\}", text, re.DOTALL)
    assert fn is not None
    body = fn.group(0)
    assert '"/wiki/lint?include_c5=false"' in body
    assert "renderLintCard(lintData" in body
    assert "renderCurationQueue(" in body


# ---------------------------------------------------------------------------
# Security — no innerHTML in the new collision code (§12.4)
# ---------------------------------------------------------------------------


def test_no_inner_html_in_collision_code():
    text = _console_text()
    start = text.index("function openCollisionModal(")
    end = text.index("/* Generic error card")
    collision_block = text[start:end]
    assert "innerHTML" not in collision_block
