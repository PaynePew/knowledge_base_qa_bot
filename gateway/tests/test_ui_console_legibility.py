"""Structural tests for the Operator Console demo legibility (#347, ADR-0021).

The Operator Console lives in ``gateway/static/console.html`` as a single
vanilla HTML/CSS/JS file (CODING_STANDARD §12.1 — no framework, no build
step). Following the pattern in ``test_ui_bilingual_starters.py``, these
tests inspect the production UI file's text to assert the structural
invariants of issue #347:

- A persistent reset banner is present and visible (not hidden by default),
  explaining that this is a demo environment whose writes are ephemeral
  (ADR-0021 AC1).
- Every lifecycle action (Upload / Import / Ingest / Index / Lint / Promote /
  Discard / RAG rebuild) has an always-visible what-it-does one-liner
  grounded in ADR-0021's example copy (AC2).
- The Promote one-liner uses the exact ADR-0021 wording describing the
  ADR-0020 human gate (AC2 Promote requirement).
- No new ``innerHTML`` assignment or ``EventSource`` is introduced (AC3 /
  CODING_STANDARD §12.4 and §12.2 invariants).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 / §12.7).
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Path to the production Operator Console file
# ---------------------------------------------------------------------------

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC1: persistent reset banner is present and always-visible
# ---------------------------------------------------------------------------


def test_console_has_reset_banner_element():
    """A reset banner element exists in the static HTML (AC1)."""
    text = _console_text()
    assert "demo-reset-banner" in text, (
        "console.html must contain a .demo-reset-banner element (ADR-0021 AC1)"
    )


def test_console_reset_banner_ephemeral_copy():
    """The reset banner mentions ephemeral writes and the reset cadence (AC1).

    ADR-0021: 'a persistent banner … stating the environment is a demo whose
    changes are real but ephemeral — it resets to the seeded knowledge base
    periodically (~2 days) and on each deploy'.
    """
    text = _console_text()
    assert "ephemeral" in text, "reset banner must use the word 'ephemeral' (ADR-0021 AC1)"
    assert "~2 days" in text, "reset banner must mention the ~2-day reset cadence (ADR-0021 AC1)"
    assert "on each deploy" in text, "reset banner must mention 'on each deploy' (ADR-0021 AC1)"


def test_console_reset_banner_not_hidden_by_default():
    """The reset banner must NOT use display:none or require a JS .visible toggle (AC1).

    Unlike the staleness guard (which defaults to hidden and is shown by JS),
    the reset banner must be always-visible: no 'display: none' on the banner
    element itself and no 'visible' class toggle needed to show it.
    """
    text = _console_text()
    # The banner must be present as static HTML markup in the body (not injected
    # or toggled by JS the way the staleness guard is).
    assert 'class="demo-reset-banner"' in text, (
        "demo-reset-banner must be a static HTML element in the body (AC1)"
    )
    # Its OWN base CSS rule must not hide it by default. Extract the
    # `.demo-reset-banner { ... }` block (not the descendant icon/text rules) and
    # assert it contains no `display: none` — unlike `.staleness-banner`, which
    # defaults to `display: none` in its own block and is revealed later by JS.
    match = re.search(r"\.demo-reset-banner\s*\{([^}]*)\}", text)
    assert match is not None, ".demo-reset-banner CSS rule must exist (AC1)"
    banner_css = match.group(1)
    assert "display: none" not in banner_css and "display:none" not in banner_css, (
        "reset banner base CSS must not set display:none — it is always-visible (AC1)"
    )


# ---------------------------------------------------------------------------
# AC2: every lifecycle action has an always-visible what-it-does one-liner
# ---------------------------------------------------------------------------


def test_console_upload_step_has_description():
    """Upload step has a what-it-does one-liner (AC2)."""
    text = _console_text()
    # Must contain a description explaining Upload's role in the pipeline.
    assert "Upload" in text, "Upload label must exist"
    # The one-liner must explain the step's principle (stage files for processing).
    assert "stage" in text.lower() or "entry point" in text.lower(), (
        "Upload step must have a description explaining it stages files for processing (AC2)"
    )


def test_console_import_step_has_description():
    """Import step has a what-it-does one-liner (AC2)."""
    text = _console_text()
    assert "Import" in text, "Import label must exist"
    # Import converts raw files to Markdown.
    assert "convert" in text.lower() or "normalise" in text.lower() or "normaliz" in text.lower(), (
        "Import step must have a description explaining it converts/normalises sources (AC2)"
    )


def test_console_ingest_step_has_description():
    """Ingest step has the ADR-0021 example one-liner (AC2)."""
    text = _console_text()
    # ADR-0021 exact example: "Ingest: an LLM synthesises a curated wiki page from a Source, grounding-checked"
    assert "synthesises" in text or "synthesizes" in text or "LLM" in text, (
        "Ingest step must mention LLM synthesis (ADR-0021 example copy, AC2)"
    )
    assert "grounding" in text.lower(), (
        "Ingest step must mention grounding-checked (ADR-0021 example copy, AC2)"
    )


def test_console_index_step_has_description():
    """Index step has a what-it-does one-liner (AC2)."""
    text = _console_text()
    # Index rebuilds the BM25 search index.
    assert "BM25" in text or "search index" in text.lower(), (
        "Index step must have a description mentioning BM25 or search index (AC2)"
    )
    assert "retrievable" in text.lower() or "retrieve" in text.lower(), (
        "Index step description must explain knowledge becomes retrievable (AC2)"
    )


def test_console_lint_step_has_description():
    """Lint step has a what-it-does one-liner (AC2)."""
    text = _console_text()
    # Lint audits the wiki for structural issues.
    assert "orphan" in text.lower(), "Lint step must mention orphans in its description (AC2)"
    assert "coherent" in text.lower() or "audit" in text.lower(), (
        "Lint step description must explain the coherence audit purpose (AC2)"
    )


def test_console_promote_line_exact_adr0021_copy():
    """Promote has the EXACT one-liner mandated by issue #347 / ADR-0021 (AC2).

    The issue states the Promote line MUST be:
    'Promote: a curator approves a draft Filed Answer so it becomes retrievable
    — the human gate of ADR-0020's validated write-back.'
    """
    text = _console_text()
    required = "a curator approves a draft Filed Answer so it becomes retrievable"
    assert required in text, (
        f"Promote one-liner must contain: {required!r} (issue #347 / ADR-0021 AC2)"
    )
    assert "ADR-0020" in text, (
        "Promote one-liner must reference ADR-0020 (issue #347 / ADR-0021 AC2)"
    )
    assert "human gate" in text, (
        "Promote one-liner must say 'human gate' (issue #347 / ADR-0021 AC2)"
    )


def test_console_discard_step_has_description():
    """Discard has a what-it-does one-liner (AC2)."""
    text = _console_text()
    # Discard removes an inert draft from the queue; server refuses live answers.
    assert "Discard" in text or "discard" in text, "Discard label must exist"
    assert "inert" in text.lower() or "draft" in text.lower(), (
        "Discard description must mention inert/draft entries (AC2)"
    )


def test_console_rag_rebuild_has_description():
    """RAG rebuild step has a what-it-does one-liner (AC2)."""
    text = _console_text()
    # RAG rebuild re-embeds docs/ into the vector index.
    assert "vector" in text.lower() or "embed" in text.lower(), (
        "RAG rebuild must have a description mentioning vector embeddings (AC2)"
    )
    assert "semantic" in text.lower() or "parallel" in text.lower(), (
        "RAG rebuild description must mention semantic search or parallel execution (AC2)"
    )


# ---------------------------------------------------------------------------
# AC3: no new innerHTML or EventSource (CODING_STANDARD §12.4 / §12.2)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment():
    """console.html never assigns to innerHTML (§12.4 / AC3 — textContent only)."""
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 / #347 AC3 requires textContent only"
    )


def test_console_no_event_source():
    """console.html does not use EventSource (§12.2 / AC3)."""
    text = _console_text()
    assert "new EventSource" not in text, (
        "new EventSource found in console.html — §12.2 / #347 AC3 prohibits EventSource"
    )
