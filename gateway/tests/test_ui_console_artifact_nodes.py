"""Console pipeline: artifact nodes + live counts + input-state lines
(issue #559 A1 — "pipeline semantics, not wizard semantics").

Design (frozen in the 2026-07-11 grill, scoped 2026-07-12): interleave
artifact nodes into the step track (``[Upload] -> raw/ -> [Import] -> docs/
-> [Ingest] -> wiki/ -> [Index] -> .kb -> [Lint]``), each showing a live
count fetched from ``GET /read/counts``; each operation card (except Upload)
shows a plain input-state line ("inform, never gate" — the Run button is
NEVER disabled by it). This slice (A1) explicitly does NOT add any
state-derived coloring — that is a separate follow-up slice scoped to the
Index node's fresh/stale badge only.

Following the pattern in ``test_ui_console_ingest_observability.py``:
inspects the production ``gateway/static/console.html`` file's text — no
DOM, no fetch, no browser, no OPENAI_API_KEY (fully hermetic, §6.3 / §12.7).
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
# ARTIFACT_DEFS — the four artifact nodes, in chain order
# ---------------------------------------------------------------------------


def test_artifact_defs_declares_four_nodes_in_chain_order():
    text = _console_text()
    match = re.search(r"var ARTIFACT_DEFS = \[(.*?)\n\];", text, re.DOTALL)
    assert match is not None, "console.html must declare var ARTIFACT_DEFS = [...]"
    ids = re.findall(r'id:\s*"([^"]+)"', match.group(1))
    assert ids == ["raw", "docs", "wiki", "kb"], (
        f"ARTIFACT_DEFS must be raw/docs/wiki/kb in chain order, got {ids}"
    )


# ---------------------------------------------------------------------------
# buildPipeline interleaves artifact nodes into the step track
# ---------------------------------------------------------------------------


def test_build_pipeline_interleaves_artifact_pills_between_steps():
    text = _console_text()
    fn = _extract_function(text, "buildPipeline")
    assert "ARTIFACT_DEFS[i - 1]" in fn, (
        "buildPipeline must interleave one ARTIFACT_DEFS entry between each "
        "pair of STEP_DEFS entries"
    )
    assert 'class: "artifact-pill"' in fn


def test_build_pipeline_never_toggles_a_state_class_on_artifact_pills():
    """A1 explicitly excludes state-derived coloring — the artifact pill is
    flat/neutral, never gets .classList.add('fresh'/'stale'/...) from
    buildPipeline (that is a separate, later slice scoped to the Index node)."""
    text = _console_text()
    fn = _extract_function(text, "buildPipeline")
    # Isolate the artifact-pill construction snippet specifically.
    start = fn.index('class: "artifact-pill"')
    end = fn.index(")", start)
    snippet = fn[start:end]
    assert "classList" not in snippet
    assert "fresh" not in snippet.lower()
    assert "stale" not in snippet.lower()


# ---------------------------------------------------------------------------
# Each artifact node carries a live count element, updated from GET /read/counts
# ---------------------------------------------------------------------------


def test_artifact_pill_has_a_count_element_per_node():
    text = _console_text()
    fn = _extract_function(text, "buildPipeline")
    assert '"artifact-count-" + artifact.id' in fn, (
        "each artifact pill must mint an id='artifact-count-<id>' element "
        "for renderArtifactCounts to update"
    )


def test_refresh_artifact_counts_fetches_read_counts():
    text = _console_text()
    fn = _extract_function(text, "refreshArtifactCounts")
    assert 'fetch("/read/counts")' in fn


def test_refresh_artifact_counts_called_after_build_pipeline():
    """The fetch must run after buildPipeline() has minted the artifact-pill /
    input-state DOM elements, else renderArtifactCounts has nothing to fill."""
    text = _console_text()
    build_call = text.index('buildPipeline(document.getElementById("pipeline-root"));')
    refresh_call = text.index("refreshArtifactCounts();", build_call)
    assert refresh_call > build_call


def test_render_artifact_counts_updates_pill_and_input_state_text_only():
    """renderArtifactCounts is pure presentation — textContent only, never
    disables anything (inform, never gate) and never toggles a class."""
    text = _console_text()
    fn = _extract_function(text, "renderArtifactCounts")
    assert "countEl.textContent" in fn
    assert "lineEl.textContent" in fn
    assert "disabled" not in fn
    assert "classList" not in fn


# ---------------------------------------------------------------------------
# Input-state lines: which step reads which upstream artifact
# ---------------------------------------------------------------------------


def test_input_artifact_wiring_matches_the_frozen_chain():
    """[Upload] -> raw/ -> [Import] -> docs/ -> [Ingest] -> wiki/ -> [Index]
    -> .kb -> [Lint] — each step's inputArtifact is the node immediately
    upstream of it; Upload has none (nothing precedes it in the chain)."""
    text = _console_text()
    step_defs = text[
        text.index("var STEP_DEFS = [") : text.index("\n];", text.index("var STEP_DEFS = ["))
    ]

    step_ids = ["upload", "import", "ingest", "index", "lint"]

    def _input_artifact_for(step_id: str) -> str | None:
        marker = f'id: "{step_id}",'
        start = step_defs.index(marker)
        # Bound the window to just this step object: up to the next
        # sibling's own `id: "..."` marker (or end of STEP_DEFS for lint).
        next_ids = [
            step_defs.index(f'id: "{other}",')
            for other in step_ids
            if other != step_id and step_defs.index(f'id: "{other}",') > start
        ]
        end = min(next_ids) if next_ids else len(step_defs)
        window = step_defs[start:end]
        match = re.search(r'inputArtifact:\s*"([^"]+)"', window)
        return match.group(1) if match else None

    assert _input_artifact_for("upload") is None, "Upload has no upstream artifact"
    assert _input_artifact_for("import") == "raw"
    assert _input_artifact_for("ingest") == "docs"
    assert _input_artifact_for("index") == "wiki"
    assert _input_artifact_for("lint") == "kb"


def test_input_state_line_element_only_built_for_steps_with_input_artifact():
    text = _console_text()
    fn = _extract_function(text, "buildPipeline")
    assert "if (step.inputArtifact)" in fn
    assert '"step-input-state"' in fn


# ---------------------------------------------------------------------------
# LINT_CHROME — bilingual coverage for the new copy
# ---------------------------------------------------------------------------


def test_input_state_keys_exist_bilingually_with_placeholder_intact():
    text = _console_text()
    for key in ("inputStateRaw", "inputStateDocs", "inputStateWiki", "inputStateKb"):
        en_match = re.search(rf'{key}:\s*"([^"]+)"', text)
        assert en_match is not None, f"LINT_CHROME.en.{key} missing"
        assert "{n}" in en_match.group(1), f"{key} must carry the {{n}} count placeholder"

    zh_matches = re.findall(r'inputStateRaw:\s*"([^"]+)"', text)
    assert len(zh_matches) == 2, "expected one en + one zh inputStateRaw entry"
    zh_text = zh_matches[1]
    assert _has_cjk(zh_text), "zh inputStateRaw must contain real Chinese text"
    assert "{n}" in zh_text


def test_artifact_node_name_keys_exist_bilingually():
    text = _console_text()
    for key in ("artifactNodeRaw", "artifactNodeDocs", "artifactNodeWiki", "artifactNodeKb"):
        assert text.count(f"{key}:") == 2, f"expected exactly one en + one zh {key} entry"


# ---------------------------------------------------------------------------
# No new innerHTML / EventSource (§12.4 / §12.2 — still holds)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
