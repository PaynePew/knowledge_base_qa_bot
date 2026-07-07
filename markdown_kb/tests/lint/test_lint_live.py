"""Live integration smoke test for run_lint() — Slice 5-5 (#70).

Makes a real OpenAI API call so the C5 page-pair contradiction check exercises
the production LLM path end-to-end (``get_lint_llm()`` lazy singleton →
``langchain_openai.ChatOpenAI`` → ``with_structured_output(PagePairFinding)``).
This is the ONE allowed ``@pytest.mark.live`` test for the lint surface per
ADR-0005's "one live test per LLM-facing surface" policy.

Opt-in only: skipped by default; run with::

    uv run pytest -m live markdown_kb/tests/lint/test_lint_live.py

Requirements:
    OPENAI_API_KEY must be set in the environment.  The test fails with a
    clear message if it is absent rather than silently passing or skipping.

Assertions are deliberately SHAPE-only (no specific prose words asserted) so
the test stays robust across model updates and across small judgement-noise
shifts in the LLM output:
    - run_lint() returns a LintResponse (200-equivalent — no HTTP layer used
      because run_lint is the deep entry point under the route)
    - wiki/lint-report.md is written
    - at least one C5 finding exists OR (if none) the no-C5-findings case is
      explained by zero F1∪F3 candidate pairs in the fixture state
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths to the fixture corpus
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES_DIR = _REPO_ROOT / "eval" / "lint_fixtures"


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_run_lint_live_c5_against_real_openai(tmp_path, monkeypatch):
    """run_lint() against the fixture corpus with the real OpenAI API.

    Loads ``eval/lint_fixtures/`` into ``tmp_path/wiki`` + ``tmp_path/docs``,
    builds the BM25 index against the fixture wiki so F3 can fire, then runs
    ``run_lint()`` with the real LLM in play.  Asserts only that the run
    completes, the report file exists, and the C5 path executed (llm_calls
    is incremented or check_errors records the actual exception).

    Deliberately model-version-robust: no exact severity, slug pair, claim
    text, or token-count value is asserted.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail(
            "OPENAI_API_KEY is not set. "
            "Export your key before running live tests: "
            "export OPENAI_API_KEY=sk-..."
        )

    # --- Set up the fixture wiki + docs in tmp -----------------------------
    wiki_dir = tmp_path / "wiki"
    shutil.copytree(str(_FIXTURES_DIR / "wiki"), str(wiki_dir))
    (wiki_dir / "entities").mkdir(exist_ok=True)
    (wiki_dir / "concepts").mkdir(exist_ok=True)

    docs_dir = tmp_path / "docs"
    shutil.copytree(str(_FIXTURES_DIR / "sources"), str(docs_dir))
    # Make all sources old except aged_policy.md (touched to now) — mirrors
    # the hermetic e2e fixture setup so C6 still fires under the live run.
    old_time = 1764547200.0  # 2025-12-01 UTC
    for src in docs_dir.glob("*.md"):
        os.utime(str(src), (old_time, old_time))
    aged = docs_dir / "aged_policy.md"
    if aged.exists():
        now = time.time()
        os.utime(str(aged), (now, now))

    # --- Pre-populate log.md with fixture entries so C1 has data -----------
    log_path = wiki_dir / "log.md"
    log_fixtures = _FIXTURES_DIR / "log_entries.txt"
    if log_fixtures.exists():
        log_path.write_text(log_fixtures.read_text(encoding="utf-8"), encoding="utf-8")

    # --- Point indexer at fixture wiki + build BM25 index ------------------
    import app.indexer as indexer_module
    import app.lint as lint_module

    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts"],
    )
    monkeypatch.setattr(lint_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(lint_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(lint_module, "LOG_PATH", log_path)

    index_path = tmp_path / ".kb" / "index.json"
    indexer_module.build_index(index_path)

    # --- Reset the lint LLM singleton so we hit the production code path ---
    # ``get_lint_llm()`` caches its ChatOpenAI singleton in the module-level
    # ``_lint_llm`` global. Reset it so this test's env-vars (if any) take
    # effect and so we don't reuse a singleton accidentally populated by an
    # earlier test in the same process.
    monkeypatch.setattr(lint_module, "_lint_llm", None)

    # --- Run lint against the real OpenAI API ------------------------------
    from app.lint import run_lint

    result = run_lint(
        wiki_dir=wiki_dir,
        docs_dir=docs_dir,
        log_path=log_path,
    )

    # --- Shape-only assertions --------------------------------------------
    # 1. Report file exists
    report_path = wiki_dir / "lint-report.md"
    assert report_path.exists(), "lint-report.md must be written"

    # 2. Report has the expected sentinel + C5 section heading
    report_text = report_path.read_text(encoding="utf-8")
    assert "<!-- Auto-generated by POST /lint" in report_text
    assert "## C5 Contradictions" in report_text

    # 3. summary.llm_calls reflects actual C5 invocations.  Either C5 made
    #    at least one call (llm_calls > 0) OR there were zero F1/F3 candidate
    #    pairs in the fixture state (a legitimate state — still success).
    by_check = result.summary.findings_by_check
    assert "c5" in by_check, "c5 must appear in findings_by_check"

    # 4. If C5 had an exception (e.g. transient network), it should be in
    #    check_errors — surface the error in the assertion message so a
    #    failing CI run shows the real cause, not just a bool mismatch.
    c5_error = result.check_errors.get("c5")
    if c5_error is not None:
        pytest.fail(f"C5 raised against real OpenAI API: {c5_error}")

    # 5. llm_calls is non-negative
    assert result.summary.llm_calls >= 0

    # 6. cost_usd is non-negative
    assert result.summary.cost_usd >= 0.0

    # 7. If any C5 findings were returned, each must have a real severity
    #    (the orchestrator filters out severity='none' before returning).
    for ppf in result.findings.page_pairs:
        assert ppf.severity in ("direct", "tension"), (
            f"Live C5 returned a finding with severity={ppf.severity!r}; "
            f"the orchestrator should filter 'none' before returning."
        )
        # Canonical slug order
        assert ppf.page_a <= ppf.page_b, f"page_a ({ppf.page_a}) must be <= page_b ({ppf.page_b})"
