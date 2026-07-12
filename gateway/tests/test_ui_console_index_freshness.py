"""Console Index node: state-derived fresh/stale badge (issue #559 A2 —
the follow-up to A1's artifact nodes, scoped to coloring the Index node
only).

Design (frozen in the 2026-07-11 grill, split 2026-07-12): state coloring
derives from artifact state (``GET /read/index-freshness`` ->
``markdown_kb.app.indexer.index_stale()`` -> the CONTEXT.md Section Index
staleness semantic), NEVER from click history — contrast the pre-existing
#173 ``indexStale`` flag (Ingest-click sets it, Index-click clears it),
which this slice does not touch. Every other artifact node (raw/docs/wiki)
stays flat/neutral; only the kb node gets a badge.

Following the pattern in ``test_ui_console_artifact_nodes.py``: inspects the
production ``gateway/static/console.html`` file's text — no DOM, no fetch,
no browser, no OPENAI_API_KEY (fully hermetic, §6.3 / §12.7).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling ``test_ui_console_*`` helper of the same
    name — robust to nested braces inside the body)."""
    marker = f"function {name}("
    start = text.index(marker)
    depth = 0
    started = False
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
            started = True
        elif text[i] == "}":
            depth -= 1
            if started and depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unterminated function body for {name}")


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


# ---------------------------------------------------------------------------
# The badge element is scoped to the kb node only
# ---------------------------------------------------------------------------


def test_freshness_badge_element_exists_and_is_kb_scoped():
    text = _console_text()
    assert 'id: "artifact-freshness-kb"' in text
    # No sibling badge id for raw/docs/wiki — the design is Index-node-only.
    for other_id in ("raw", "docs", "wiki"):
        assert f'id: "artifact-freshness-{other_id}"' not in text


def test_build_pipeline_only_adds_the_badge_for_the_kb_artifact():
    text = _console_text()
    fn = _extract_function(text, "buildPipeline")
    assert 'artifact.id === "kb"' in fn
    assert 'id: "artifact-freshness-kb"' in fn


def test_build_pipeline_artifact_pill_object_literal_still_classlist_free():
    """Re-asserts the A1 contract this slice must not disturb: the
    artifact-pill's own {class:...} construction never hardcodes a state
    class (mirrors test_ui_console_artifact_nodes.py's own guard)."""
    text = _console_text()
    fn = _extract_function(text, "buildPipeline")
    start = fn.index('class: "artifact-pill"')
    end = fn.index(")", start)
    snippet = fn[start:end]
    assert "classList" not in snippet
    assert "fresh" not in snippet.lower()
    assert "stale" not in snippet.lower()


# ---------------------------------------------------------------------------
# renderIndexFreshness — the ONLY function allowed to toggle a state class
# ---------------------------------------------------------------------------


def test_render_index_freshness_function_exists_and_is_textcontent_based():
    text = _console_text()
    fn = _extract_function(text, "renderIndexFreshness")
    assert "textContent" in fn
    assert ".innerHTML" not in fn


def test_render_index_freshness_toggles_a_state_class():
    text = _console_text()
    fn = _extract_function(text, "renderIndexFreshness")
    assert "classList" in fn
    assert "state-fresh" in fn
    assert "state-stale" in fn


def test_render_artifact_counts_still_has_no_classlist():
    """A2 must not smuggle state-coloring into A1's renderArtifactCounts —
    re-asserts the sibling guard in test_ui_console_artifact_nodes.py."""
    text = _console_text()
    fn = _extract_function(text, "renderArtifactCounts")
    assert "classList" not in fn


# ---------------------------------------------------------------------------
# refreshIndexFreshness — fetches GET /read/index-freshness
# ---------------------------------------------------------------------------


def test_refresh_index_freshness_fetches_the_endpoint():
    text = _console_text()
    fn = _extract_function(text, "refreshIndexFreshness")
    assert 'fetch("/read/index-freshness")' in fn


def test_refresh_index_freshness_called_after_build_pipeline_at_boot():
    text = _console_text()
    build_call = text.index('buildPipeline(document.getElementById("pipeline-root"));')
    refresh_call = text.index("refreshIndexFreshness();", build_call)
    assert refresh_call > build_call


def test_refresh_index_freshness_called_after_ingest_and_index_steps():
    """wiki/ changes on Ingest completion; .kb rebuilds on Index completion —
    both are the two mutations that can flip the staleness signal."""
    text = _console_text()
    ingest_run = text.index('id: "ingest",')
    index_run = text.index('id: "index",', ingest_run)
    lint_run = text.index('id: "lint",', index_run)

    ingest_block = text[ingest_run:index_run]
    index_block = text[index_run:lint_run]

    assert "refreshIndexFreshness();" in ingest_block
    assert "refreshIndexFreshness();" in index_block


def test_apply_console_lang_rerenders_index_freshness_from_cached_value():
    """Mirrors renderArtifactCounts(lastArtifactCounts) — re-renders from the
    last-fetched value on a language toggle rather than re-fetching."""
    text = _console_text()
    fn = _extract_function(text, "applyConsoleLang")
    assert "renderIndexFreshness(lastIndexFreshness)" in fn


# ---------------------------------------------------------------------------
# LINT_CHROME — bilingual coverage for the badge copy
# ---------------------------------------------------------------------------


def test_index_freshness_badge_keys_exist_bilingually():
    text = _console_text()
    for key in ("indexFreshBadge", "indexStaleBadge"):
        assert text.count(f"{key}:") == 2, f"expected exactly one en + one zh {key} entry"

    zh_matches = re.findall(r'indexStaleBadge:\s*"([^"]+)"', text)
    assert len(zh_matches) == 2, "expected one en + one zh indexStaleBadge entry"
    zh_text = zh_matches[1]
    assert _has_cjk(zh_text), "zh indexStaleBadge must contain real Chinese text"


# ---------------------------------------------------------------------------
# No new innerHTML / EventSource (§12.4 / §12.2 — still holds)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
