"""Unit tests for C5 lint scaling (issue #194).

The full ``POST /wiki/lint`` (``include_c5=True``) made one LLM call per
candidate page-pair. On the ~60-page wiki the F1 "shares a source" filter
over-generated ~1,770 pairs → ~1,770 serial ``gpt-4o-mini`` calls → ~200s.

This module locks the two scaling fixes:

  1. **Similarity pre-filter (top-K cap).** Candidate pairs are ranked by a
     cheap, deterministic lexical token-overlap (Jaccard) signal and only the
     top ``KB_LINT_C5_MAX_PAIRS`` (default 30) are sent to the LLM judge. Pairs
     beyond the cap are surfaced as "not judged (capped)", never dropped
     silently.
  2. **Bounded concurrency.** The surviving (≤K) ``_judge_page_pair`` calls run
     concurrently with a ``KB_LINT_C5_CONCURRENCY`` (default 5) worker limit.

All tests are hermetic — the LLM is mocked via the lazy-singleton getter
(monkeypatch ``get_lint_llm``); no OPENAI_API_KEY is required.

Every fixture wiki writes pages that **all share one source** so the F1
candidate set is exactly C(N, 2) regardless of the in-memory BM25 index state
(F3 can only re-surface pairs already in that set, which dedupe). This keeps
the candidate count deterministic without per-test index isolation.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SHARED_SOURCE = "shared_source.md#section"


def _write_wiki_page(wiki_dir: Path, slug: str, body: str, *, subdir: str = "concepts") -> Path:
    """Write a wiki page citing the single shared source, with the given body."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-05-26T00:00:00Z",
        "updated": "2026-05-26T00:00:00Z",
        "sources": [_SHARED_SOURCE],
        "status": "live",
        "open_questions": [],
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _spy_llm(monkeypatch, severity: str = "none"):
    """Install a mock C5 LLM that records each judged pair and returns ``severity``.

    Returns the ``judged`` list — one ``(page_a, page_b)`` canonical tuple per
    LLM invocation, in call order — so tests can assert *how many* and *which*
    pairs were sent to the judge.
    """
    import re

    import app.lint as lint_module
    from app.schemas import PagePairFinding

    judged: list[tuple[str, str]] = []

    def mock_invoke(messages):
        content = str(messages)
        slugs = re.findall(r"slug: `([^`]+)`", content)
        a, b = (min(slugs[0], slugs[1]), max(slugs[0], slugs[1])) if len(slugs) >= 2 else ("?", "?")
        judged.append((a, b))
        return PagePairFinding(
            severity=severity,
            page_a=a,
            page_b=b,
            page_a_claim=f"claim from {a}",
            page_b_claim=f"claim from {b}",
            summary=f"{severity} between {a} and {b}",
            suggested_action="review",
        )

    from unittest.mock import MagicMock

    mock_chain = MagicMock()
    mock_chain.invoke = mock_invoke
    monkeypatch.setattr(
        lint_module,
        "get_lint_llm",
        lambda: MagicMock(with_structured_output=lambda s: mock_chain),
    )
    return judged


class TestC5TopKCap:
    """KB_LINT_C5_MAX_PAIRS caps the number of pairs sent to the LLM judge."""

    def test_cap_limits_judged_pair_count(self, tmp_wiki_dir, monkeypatch):
        """With 4 shared-source pages (6 F1 pairs) and cap=2, only 2 pairs are judged."""
        for slug in ("alpha", "beta", "gamma", "delta"):
            _write_wiki_page(tmp_wiki_dir, slug, f"Body about {slug} topic.")

        monkeypatch.setenv("KB_LINT_C5_MAX_PAIRS", "2")
        judged = _spy_llm(monkeypatch)

        import app.lint as lint_module

        lint_module._check_c5_page_pair(tmp_wiki_dir)

        assert len(judged) == 2, f"Expected exactly 2 judged pairs under cap=2; judged {judged}"

    def test_most_similar_pair_judged_first(self, tmp_wiki_dir, monkeypatch):
        """The similarity pre-filter judges the highest-token-overlap pair under cap=1."""
        # Two refund pages share almost all body tokens; the shipping page is disjoint.
        _write_wiki_page(
            tmp_wiki_dir,
            "refund-timeline",
            "refund window cancellation policy timeline business days processed details",
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "refund-window",
            "refund window cancellation policy timeline business days processed",
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "shipping-carrier",
            "shipping carrier tracking delivery courier dispatch warehouse",
        )

        monkeypatch.setenv("KB_LINT_C5_MAX_PAIRS", "1")
        judged = _spy_llm(monkeypatch)

        import app.lint as lint_module

        lint_module._check_c5_page_pair(tmp_wiki_dir)

        assert judged == [("refund-timeline", "refund-window")], (
            f"Expected only the most-similar (refund-timeline, refund-window) pair; judged {judged}"
        )

    def test_chinese_corpus_ranks_by_shared_characters(self, tmp_wiki_dir, monkeypatch):
        """CJK bodies rank by shared-bigram overlap (the real 退款 corpus is Chinese)."""
        # Two refund pages share most characters; the shipping page is disjoint.
        _write_wiki_page(tmp_wiki_dir, "tuikuan-a", "退款需要五個工作天才能完成處理")
        _write_wiki_page(tmp_wiki_dir, "tuikuan-b", "退款需要五個工作天即可完成")
        _write_wiki_page(tmp_wiki_dir, "yunsong", "運送由貨運商負責配送與追蹤包裹")

        monkeypatch.setenv("KB_LINT_C5_MAX_PAIRS", "1")
        judged = _spy_llm(monkeypatch)

        import app.lint as lint_module

        lint_module._check_c5_page_pair(tmp_wiki_dir)

        assert judged == [("tuikuan-a", "tuikuan-b")], (
            f"Expected the two 退款 pages to rank highest; judged {judged}"
        )


class TestC5CappedSurfacing:
    """Capped (not-judged) pairs are surfaced in the response summary + report."""

    def test_summary_reports_judged_and_capped_counts(self, lint_env, monkeypatch):
        """run_lint().summary carries llm_calls (judged) and c5_pairs_capped (skipped)."""
        wiki_dir = lint_env["wiki_dir"]
        # 5 shared-source pages → C(5,2) = 10 candidate pairs.
        for i in range(5):
            _write_wiki_page(wiki_dir, f"page-{i}", f"Body number {i} about a topic.")

        monkeypatch.setenv("KB_LINT_C5_MAX_PAIRS", "3")
        _spy_llm(monkeypatch)

        from app.lint import run_lint

        result = run_lint(**lint_env)

        assert result.summary.llm_calls == 3, (
            f"Expected 3 judged pairs (== LLM calls) under cap=3; got {result.summary.llm_calls}"
        )
        assert result.summary.c5_pairs_capped == 7, (
            f"Expected 10 candidates − 3 judged = 7 capped; got {result.summary.c5_pairs_capped}"
        )

    def test_report_notes_capped_pairs(self, lint_env, monkeypatch):
        """The C5 report section surfaces capped pairs as 'not judged', never silently drops them."""
        wiki_dir = lint_env["wiki_dir"]
        for i in range(5):
            _write_wiki_page(wiki_dir, f"page-{i}", f"Body number {i} about a topic.")

        monkeypatch.setenv("KB_LINT_C5_MAX_PAIRS", "3")
        _spy_llm(monkeypatch)

        from app.lint import run_lint

        run_lint(**lint_env)

        report = (wiki_dir / "lint-report.md").read_text(encoding="utf-8")
        assert "not judged" in report and "capped" in report, (
            "C5 report section must surface capped pairs as 'not judged (capped)'"
        )
        assert "7" in report, "The capped-pair count (7) should appear in the report note"

    def test_no_cap_when_candidates_below_limit(self, lint_env, monkeypatch):
        """A small wiki (candidates ≤ cap) judges every pair; nothing is capped."""
        wiki_dir = lint_env["wiki_dir"]
        # 3 shared-source pages → C(3,2) = 3 candidate pairs, all under the default cap.
        for slug in ("alpha", "beta", "gamma"):
            _write_wiki_page(wiki_dir, slug, f"Body about {slug}.")

        _spy_llm(monkeypatch)

        from app.lint import run_lint

        result = run_lint(**lint_env)

        assert result.summary.llm_calls == 3
        assert result.summary.c5_pairs_capped == 0


class TestC5BoundedConcurrency:
    """The surviving (≤K) LLM calls run concurrently under a KB_LINT_C5_CONCURRENCY bound."""

    def test_calls_run_in_parallel_within_the_bound(self, tmp_wiki_dir, monkeypatch):
        """Judge calls overlap (parallel) but never exceed the configured worker count."""
        import threading
        import time
        from unittest.mock import MagicMock

        import app.lint as lint_module
        from app.schemas import PagePairFinding

        # 6 shared-source pages → C(6,2) = 15 pairs; cap covers all.
        for i in range(6):
            _write_wiki_page(tmp_wiki_dir, f"page-{i}", f"Body number {i} about a topic.")

        monkeypatch.setenv("KB_LINT_C5_MAX_PAIRS", "100")
        monkeypatch.setenv("KB_LINT_C5_CONCURRENCY", "3")

        lock = threading.Lock()
        state = {"in_flight": 0, "max_in_flight": 0}

        def mock_invoke(messages):
            with lock:
                state["in_flight"] += 1
                state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            time.sleep(0.05)  # hold the worker slot so genuine overlap is observable
            with lock:
                state["in_flight"] -= 1
            return PagePairFinding(
                severity="none",
                page_a="a",
                page_b="b",
                page_a_claim="x",
                page_b_claim="y",
                summary="s",
                suggested_action="r",
            )

        mock_chain = MagicMock()
        mock_chain.invoke = mock_invoke
        monkeypatch.setattr(
            lint_module,
            "get_lint_llm",
            lambda: MagicMock(with_structured_output=lambda s: mock_chain),
        )

        lint_module._check_c5_page_pair(tmp_wiki_dir)

        # ≥2 proves the calls genuinely overlapped (serial execution would peak at 1).
        assert state["max_in_flight"] >= 2, (
            f"Expected concurrent judge calls; peak in-flight was {state['max_in_flight']}"
        )
        # ≤3 proves the KB_LINT_C5_CONCURRENCY bound is honored (would exceed if ignored).
        assert state["max_in_flight"] <= 3, (
            f"Concurrency exceeded the bound of 3; peak in-flight was {state['max_in_flight']}"
        )

    def test_continue_on_error_under_concurrency(self, tmp_wiki_dir, monkeypatch):
        """One pair raising does not sink the others — findings from successful pairs survive.

        Order-independent (keyed by slug content, not call count) so it holds
        under non-deterministic concurrent completion order.
        """
        from unittest.mock import MagicMock

        import app.lint as lint_module
        from app.schemas import PagePairFinding

        _write_wiki_page(tmp_wiki_dir, "alpha", "Alpha body about refunds.")
        _write_wiki_page(tmp_wiki_dir, "beta", "Beta body about refunds.")
        _write_wiki_page(tmp_wiki_dir, "gamma", "Gamma body about refunds.")

        monkeypatch.setenv("KB_LINT_C5_CONCURRENCY", "3")

        import re

        def mock_invoke(messages):
            slugs = re.findall(r"slug: `([^`]+)`", str(messages))
            a, b = min(slugs), max(slugs)
            # The pair that touches gamma always fails; alpha-beta succeeds.
            if "gamma" in (a, b):
                raise RuntimeError("LLM error for a gamma pair")
            return PagePairFinding(
                severity="direct",
                page_a=a,
                page_b=b,
                page_a_claim="x",
                page_b_claim="y",
                summary="conflict",
                suggested_action="fix",
            )

        mock_chain = MagicMock()
        mock_chain.invoke = mock_invoke
        monkeypatch.setattr(
            lint_module,
            "get_lint_llm",
            lambda: MagicMock(with_structured_output=lambda s: mock_chain),
        )

        results = lint_module._check_c5_page_pair(tmp_wiki_dir)

        # alpha-beta must survive despite the two gamma pairs raising.
        assert [(f.page_a, f.page_b) for f in results] == [("alpha", "beta")], (
            f"Expected the non-failing alpha-beta finding retained; got {results}"
        )
