"""Hermetic tests for Slice 5-4: C1 coverage gap aggregation from chat_fallback log.

AC coverage (issue #69):
  - _canonicalise() pure helper: lowercase, strip leading/trailing punctuation,
    collapse internal whitespace, strip outer whitespace
  - retrieval_empty entries grouped by _canonicalise(q) key
  - below_threshold entries grouped by (_canonicalise(q), top_section) key
  - claim_unsupported entries grouped by (_canonicalise(q), tuple(sorted(cited_pages))) key
  - wiki_layer_empty entries are silently ignored
  - CoverageGapFinding shape: reason, query_canonical, sample_raw_queries,
    hit_count, first_seen, last_seen, top_section, cited_pages, suggested_action
  - KB_LINT_MIN_HITS env var: clusters with hit_count < threshold are skipped
  - Canonicalisation correctly clusters "How do I cancel?" + "how do i cancel"
  - Sort: hit_count descending within each reason group, alphabetical for ties;
    groups in fixed order (retrieval_empty, below_threshold, claim_unsupported)
  - Report rendered with ## C1 Coverage gaps section, sub-headings per reason
  - Malformed log lines are skipped silently (no crash)
  - Malformed lines count is logged as lint_check_error if > 0
  - run_lint() integrates C1 into LintResponse.findings.coverage_gaps

All tests are hermetic (no OPENAI_API_KEY required).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Log-writing helpers
# ---------------------------------------------------------------------------


def _write_log(log_path: Path, lines: list[str]) -> None:
    """Write log lines (already formatted as ## [...] kind | summary) to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fallback_line(ts: str, kind: str, summary: str) -> str:
    return f"## [{ts}] {kind} | {summary}"


# ---------------------------------------------------------------------------
# _canonicalise tests (pure function, no I/O)
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_lowercases(self):
        from app.lint import _canonicalise

        assert _canonicalise("How Do I Cancel") == "how do i cancel"

    def test_strips_leading_trailing_punctuation(self):
        from app.lint import _canonicalise

        assert _canonicalise("?how do i cancel?") == "how do i cancel"

    def test_collapses_internal_whitespace(self):
        from app.lint import _canonicalise

        assert _canonicalise("how  do   i  cancel") == "how do i cancel"

    def test_strips_outer_whitespace(self):
        from app.lint import _canonicalise

        assert _canonicalise("  how do i cancel  ") == "how do i cancel"

    def test_clusters_question_variations(self):
        """'How do I cancel?' and 'how do i cancel' must canonicalise identically."""
        from app.lint import _canonicalise

        assert _canonicalise("How do I cancel?") == _canonicalise("how do i cancel")

    def test_no_stop_word_removal(self):
        from app.lint import _canonicalise

        # "do" and "i" are stop-words but must NOT be removed per AC
        result = _canonicalise("how do i cancel")
        assert "do" in result
        assert "i" in result

    def test_no_token_sorting(self):
        from app.lint import _canonicalise

        # token order must be preserved (no anagram normalisation)
        assert _canonicalise("cancel how") != _canonicalise("how cancel")

    def test_empty_string(self):
        from app.lint import _canonicalise

        assert _canonicalise("") == ""

    def test_only_punctuation(self):
        from app.lint import _canonicalise

        # stripping all punctuation leaves empty string
        assert _canonicalise("???") == ""


# ---------------------------------------------------------------------------
# _check_c1_coverage_gaps tests
# ---------------------------------------------------------------------------


class TestC1CoverageGaps:
    def test_retrieval_empty_produces_finding(self, lint_env, monkeypatch):
        """A single retrieval_empty entry produces one CoverageGapFinding."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"what is the refund policy" reason=retrieval_empty top_score=0.0',
                )
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert len(findings) == 1
        f = findings[0]
        assert f.reason == "retrieval_empty"
        assert f.query_canonical == "what is the refund policy"
        assert f.hit_count == 1
        assert "what is the refund policy" in f.sample_raw_queries

    def test_below_threshold_produces_finding(self, lint_env, monkeypatch):
        """A below_threshold entry produces one CoverageGapFinding with top_section."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"how long does shipping take" reason=below_threshold top_score=0.42 top_section=shipping#overview',
                )
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert len(findings) == 1
        f = findings[0]
        assert f.reason == "below_threshold"
        assert f.top_section == "shipping#overview"
        assert f.suggested_action.startswith("Extend page")

    def test_claim_unsupported_produces_finding(self, lint_env, monkeypatch):
        """A claim_unsupported entry produces one CoverageGapFinding with cited_pages."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_grounding_fallback",
                    '"can i get a full refund" reason=claim_unsupported cited=refund-policy#overview,refund-policy#window',
                )
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert len(findings) == 1
        f = findings[0]
        assert f.reason == "claim_unsupported"
        assert f.cited_pages is not None
        assert "refund-policy#overview" in f.cited_pages
        assert f.suggested_action.startswith("Review:")

    def test_wiki_layer_empty_is_ignored(self, lint_env, monkeypatch):
        """wiki_layer_empty log entries must be silently skipped."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "wiki_layer_empty",
                    "entities=0 concepts=0",
                )
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert findings == []

    def test_canonicalisation_clusters_query_variations(self, lint_env, monkeypatch):
        """'How do I cancel?' and 'how do i cancel' cluster into one finding."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"How do I cancel?" reason=retrieval_empty top_score=0.0',
                ),
                _fallback_line(
                    "2026-05-27T10:01:00.000000Z",
                    "chat_fallback",
                    '"how do i cancel" reason=retrieval_empty top_score=0.0',
                ),
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert len(findings) == 1
        f = findings[0]
        assert f.hit_count == 2
        # Both raw queries should be in sample_raw_queries (up to 3 unique)
        assert len(f.sample_raw_queries) == 2

    def test_kb_lint_min_hits_filters_below_threshold_clusters(self, lint_env, monkeypatch):
        """With KB_LINT_MIN_HITS=2, single-hit clusters are excluded."""
        monkeypatch.setenv("KB_LINT_MIN_HITS", "2")
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                # This cluster has 2 hits → should surface
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"what is the refund window" reason=retrieval_empty top_score=0.0',
                ),
                _fallback_line(
                    "2026-05-27T10:01:00.000000Z",
                    "chat_fallback",
                    '"what is the refund window" reason=retrieval_empty top_score=0.0',
                ),
                # This cluster has 1 hit → should NOT surface
                _fallback_line(
                    "2026-05-27T10:02:00.000000Z",
                    "chat_fallback",
                    '"rare one-off question" reason=retrieval_empty top_score=0.0',
                ),
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert len(findings) == 1
        assert "refund window" in findings[0].query_canonical

    def test_sample_raw_queries_deduplicated_up_to_3(self, lint_env, monkeypatch):
        """sample_raw_queries holds up to first 3 unique raw queries per cluster."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        # 5 hits, 4 unique query strings, same canonical
        queries = [
            "how do i cancel",
            "How do I cancel?",
            "HOW DO I CANCEL",
            "how  do  i  cancel",
            "how do i cancel",  # duplicate of first
        ]
        lines = [
            _fallback_line(
                f"2026-05-27T10:0{i}:00.000000Z",
                "chat_fallback",
                f'"{q}" reason=retrieval_empty top_score=0.0',
            )
            for i, q in enumerate(queries)
        ]
        _write_log(log_path, lines)
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert len(findings) == 1
        f = findings[0]
        assert f.hit_count == 5
        assert len(f.sample_raw_queries) <= 3

    def test_first_seen_and_last_seen_populated(self, lint_env, monkeypatch):
        """first_seen and last_seen must be parsed from log timestamps."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"cancel subscription" reason=retrieval_empty top_score=0.0',
                ),
                _fallback_line(
                    "2026-05-27T11:00:00.000000Z",
                    "chat_fallback",
                    '"cancel subscription" reason=retrieval_empty top_score=0.0',
                ),
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        f = findings[0]
        assert f.first_seen is not None
        assert f.last_seen is not None
        assert f.first_seen <= f.last_seen

    def test_sort_hit_count_descending_within_reason_group(self, lint_env, monkeypatch):
        """Findings sorted hit_count descending, alphabetical for ties."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        # 3 queries: aaa(1 hit), bbb(3 hits), ccc(2 hits)
        lines = (
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"aaa question" reason=retrieval_empty top_score=0.0',
                )
            ]
            + [
                _fallback_line(
                    f"2026-05-27T10:0{i}:00.000000Z",
                    "chat_fallback",
                    '"bbb question" reason=retrieval_empty top_score=0.0',
                )
                for i in range(1, 4)
            ]
            + [
                _fallback_line(
                    f"2026-05-27T10:0{i}:00.000000Z",
                    "chat_fallback",
                    '"ccc question" reason=retrieval_empty top_score=0.0',
                )
                for i in range(4, 6)
            ]
        )
        _write_log(log_path, lines)
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        hit_counts = [f.hit_count for f in findings]
        assert hit_counts == sorted(hit_counts, reverse=True), (
            f"Not sorted by hit_count desc: {hit_counts}"
        )

    def test_sort_alphabetical_for_ties(self, lint_env, monkeypatch):
        """Findings with equal hit_count sorted alphabetically by query_canonical."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        lines = [
            _fallback_line(
                "2026-05-27T10:00:00.000000Z",
                "chat_fallback",
                '"zebra topic" reason=retrieval_empty top_score=0.0',
            ),
            _fallback_line(
                "2026-05-27T10:01:00.000000Z",
                "chat_fallback",
                '"alpha topic" reason=retrieval_empty top_score=0.0',
            ),
        ]
        _write_log(log_path, lines)
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        canonicals = [f.query_canonical for f in findings]
        assert canonicals == sorted(canonicals), f"Not sorted alphabetically for ties: {canonicals}"

    def test_groups_in_fixed_order(self, lint_env, monkeypatch):
        """Groups appear in fixed order: retrieval_empty, below_threshold, claim_unsupported."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_grounding_fallback",
                    '"claim q" reason=claim_unsupported cited=pg#s',
                ),
                _fallback_line(
                    "2026-05-27T10:01:00.000000Z",
                    "chat_fallback",
                    '"bt q" reason=below_threshold top_score=0.3 top_section=pg#s',
                ),
                _fallback_line(
                    "2026-05-27T10:02:00.000000Z",
                    "chat_fallback",
                    '"re q" reason=retrieval_empty top_score=0.0',
                ),
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        reasons = [f.reason for f in findings]
        # retrieval_empty must come before below_threshold, below_threshold before claim_unsupported
        assert reasons.index("retrieval_empty") < reasons.index("below_threshold")
        assert reasons.index("below_threshold") < reasons.index("claim_unsupported")

    def test_malformed_line_skipped_silently(self, lint_env, monkeypatch):
        """Malformed log lines do not crash; they are silently skipped."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                "## this is not a valid log line at all",
                "garbage",
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"valid query" reason=retrieval_empty top_score=0.0',
                ),
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        # Only the valid line contributes a finding
        assert len(findings) == 1

    def test_empty_log_returns_empty_list(self, lint_env, monkeypatch):
        """Empty log file returns empty findings list."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert findings == []

    def test_missing_log_returns_empty_list(self, lint_env, monkeypatch):
        """Missing log file returns empty findings list (no crash)."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        # Do not create the file
        assert not log_path.exists()
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert findings == []

    def test_retrieval_empty_suggested_action(self, lint_env, monkeypatch):
        """suggested_action for retrieval_empty mentions 'Create a new wiki page'."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"refund timeline" reason=retrieval_empty top_score=0.0',
                )
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert "Create a new wiki page" in findings[0].suggested_action

    def test_below_threshold_suggested_action(self, lint_env, monkeypatch):
        """suggested_action for below_threshold mentions 'Extend page'."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"shipping estimate" reason=below_threshold top_score=0.31 top_section=shipping#eta',
                )
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        assert "Extend page" in findings[0].suggested_action
        assert "shipping#eta" in findings[0].suggested_action

    def test_claim_unsupported_suggested_action(self, lint_env, monkeypatch):
        """suggested_action for claim_unsupported starts with 'Review:'."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_grounding_fallback",
                    '"can i get free returns" reason=claim_unsupported cited=returns#policy',
                )
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        action = findings[0].suggested_action
        assert action.startswith("Review:")
        assert "returns#policy" in action

    def test_out_of_scope_reasons_are_silent(self, lint_env, monkeypatch):
        """chat_fallback lines with out-of-scope reasons (e.g. not_indexed) do NOT
        fire a lint_check_error and are silently ignored.

        Fix 1 (concern #2): production-emitted reasons like ``not_indexed`` and
        ``verifier_unavailable`` are legitimately not C1's concern and must be
        counted as ``out_of_scope_reasons``, not ``malformed_lines``.  Only true
        parse failures should trigger a ``lint_check_error``.
        """
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                # out-of-scope reason — clean parse, but C1 doesn't handle it
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"some query" reason=not_indexed top_score=0.0',
                ),
                # another out-of-scope reason
                _fallback_line(
                    "2026-05-27T10:01:00.000000Z",
                    "chat_fallback",
                    '"another query" reason=verifier_unavailable top_score=0.0',
                ),
                # one valid C1 line so we have something to assert on
                _fallback_line(
                    "2026-05-27T10:02:00.000000Z",
                    "chat_fallback",
                    '"valid query" reason=retrieval_empty top_score=0.0',
                ),
            ],
        )
        from app.lint import _check_c1_coverage_gaps

        findings = _check_c1_coverage_gaps(log_path)
        # Only the valid line produces a finding
        assert len(findings) == 1
        assert findings[0].reason == "retrieval_empty"

        # No lint_check_error should have been written (out_of_scope_reasons != malformed)
        log_content = log_path.read_text(encoding="utf-8")
        assert "lint_check_error" not in log_content


# ---------------------------------------------------------------------------
# Integration with run_lint()
# ---------------------------------------------------------------------------


class TestRunLintC1Integration:
    def test_run_lint_includes_c1_findings_in_response(self, lint_env, monkeypatch):
        """run_lint() populates findings.coverage_gaps from chat_fallback log entries."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"what is the return window" reason=retrieval_empty top_score=0.0',
                )
            ],
        )
        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert len(result.findings.coverage_gaps) == 1
        assert result.findings.coverage_gaps[0].reason == "retrieval_empty"

    def test_run_lint_c1_count_in_summary(self, lint_env, monkeypatch):
        """run_lint() includes 'c1' key in summary.findings_by_check."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"some query" reason=retrieval_empty top_score=0.0',
                )
            ],
        )
        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert "c1" in result.summary.findings_by_check
        assert result.summary.findings_by_check["c1"] == 1

    def test_run_lint_report_has_c1_section(self, lint_env, monkeypatch):
        """lint-report.md has '## C1 Coverage gaps' section."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"test query" reason=retrieval_empty top_score=0.0',
                )
            ],
        )
        from app.lint import run_lint

        run_lint(**lint_env)
        content = (lint_env["wiki_dir"] / "lint-report.md").read_text(encoding="utf-8")
        assert "## C1 Coverage gaps" in content

    def test_run_lint_c1_zero_findings_section_still_present(self, lint_env, monkeypatch):
        """Report includes ## C1 Coverage gaps even when zero C1 findings."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        # Empty log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        from app.lint import run_lint

        run_lint(**lint_env)
        content = (lint_env["wiki_dir"] / "lint-report.md").read_text(encoding="utf-8")
        assert "## C1 Coverage gaps" in content

    def test_run_lint_c1_continue_on_error(self, lint_env, monkeypatch):
        """If _check_c1_coverage_gaps raises, error recorded in check_errors; other checks run."""
        import app.lint as lint_module

        def _bad_c1(*_args, **_kwargs):
            raise RuntimeError("simulated C1 failure")

        monkeypatch.setattr(lint_module, "_check_c1_coverage_gaps", _bad_c1)
        from app.lint import run_lint

        result = run_lint(**lint_env)
        assert "c1" in result.check_errors
        # C11 still ran — check_errors only has c1
        assert "c11" not in result.check_errors

    def test_run_lint_total_findings_includes_c1(self, lint_env, monkeypatch):
        """summary.total_findings counts both C11 orphans and C1 coverage gaps."""
        monkeypatch.delenv("KB_LINT_MIN_HITS", raising=False)
        log_path = lint_env["log_path"]
        _write_log(
            log_path,
            [
                _fallback_line(
                    "2026-05-27T10:00:00.000000Z",
                    "chat_fallback",
                    '"coverage question" reason=retrieval_empty top_score=0.0',
                )
            ],
        )
        # Also plant an orphan
        from tests.lint.test_lint_scaffold import _write_wiki_page

        _write_wiki_page(lint_env["wiki_dir"], "orphan-for-c1-test", ["gone.md#sec"])
        from app.lint import run_lint

        result = run_lint(**lint_env)
        # 1 C1 finding + 1 C11 finding = 2 total
        assert result.summary.total_findings == 2
