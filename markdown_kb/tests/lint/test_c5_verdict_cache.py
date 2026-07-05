"""Hermetic tests for the C5 content-hash verdict cache (issue #446).

AC coverage:
  - A second deep audit over an unchanged corpus makes zero LLM calls and
    returns the same findings.
  - Editing either page of a pair re-judges only the affected pairs.
  - Cache hits/misses and estimated cost are visible in logs
    (`lint_completed`'s `c5_cache_hits=` / `llm_calls=` / `cost_usd=`).
  - Tests with a fake judge LLM cover hit, miss, and invalidation paths.

All tests are hermetic — the LLM is mocked via the lazy-singleton getter
(monkeypatch ``get_lint_llm``); no OPENAI_API_KEY is required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    sources: list[str],
    body: str,
    *,
    subdir: str = "concepts",
) -> Path:
    """Write a wiki page with the given frontmatter sources and body content."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-05-26T00:00:00Z",
        "updated": "2026-05-26T00:00:00Z",
        "sources": sources,
        "status": "live",
        "open_questions": [],
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _spy_llm(monkeypatch, severity: str = "direct"):
    """Install a mock C5 LLM that records each judged pair and returns ``severity``.

    Returns the ``judged`` list — one canonical ``(page_a, page_b)`` tuple per
    LLM invocation, in call order — mirroring test_c5_scaling.py's ``_spy_llm``.
    """
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

    mock_chain = MagicMock()
    mock_chain.invoke = mock_invoke
    monkeypatch.setattr(
        lint_module,
        "get_lint_llm",
        lambda: MagicMock(with_structured_output=lambda s: mock_chain),
    )
    return judged


# ---------------------------------------------------------------------------
# Cache path / hash / key helper unit tests
# ---------------------------------------------------------------------------


class TestCacheHelpers:
    def test_cache_path_is_colocated_with_kb_state(self, tmp_wiki_dir):
        from app.lint import _c5_verdict_cache_path

        path = _c5_verdict_cache_path(tmp_wiki_dir)
        assert path == tmp_wiki_dir.parent / ".kb" / "c5_verdict_cache.json"

    def test_content_hash_stable_for_same_body(self):
        from app.lint import _c5_content_hash

        assert _c5_content_hash("hello") == _c5_content_hash("hello")

    def test_content_hash_differs_for_different_body(self):
        from app.lint import _c5_content_hash

        assert _c5_content_hash("hello") != _c5_content_hash("goodbye")

    def test_cache_key_is_order_sensitive_to_call_args(self):
        """The key is the two hashes joined in call order (callers always pass
        the slug-canonical order, so this is stable pair-to-pair)."""
        from app.lint import _c5_cache_key, _c5_content_hash

        key = _c5_cache_key("body a", "body b")
        assert key == f"{_c5_content_hash('body a')}:{_c5_content_hash('body b')}"

    def test_load_missing_cache_returns_empty_dict(self, tmp_path):
        from app.lint import _load_c5_verdict_cache

        assert _load_c5_verdict_cache(tmp_path / ".kb" / "c5_verdict_cache.json") == {}

    def test_load_corrupt_cache_raises(self, tmp_path):
        """Fail-fast on unparseable JSON (CODING_STANDARD §4.1)."""
        from app.lint import _load_c5_verdict_cache

        cache_path = tmp_path / ".kb" / "c5_verdict_cache.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            _load_c5_verdict_cache(cache_path)

    def test_load_non_object_cache_raises(self, tmp_path):
        """A JSON array (not an object) is also treated as corrupt."""
        from app.lint import _load_c5_verdict_cache

        cache_path = tmp_path / ".kb" / "c5_verdict_cache.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ValueError, match="not a JSON object"):
            _load_c5_verdict_cache(cache_path)

    def test_write_then_load_round_trips(self, tmp_path):
        from app.lint import _load_c5_verdict_cache, _write_c5_verdict_cache

        cache_path = tmp_path / ".kb" / "c5_verdict_cache.json"
        payload = {"deadbeef:cafef00d": {"severity": "direct"}}
        _write_c5_verdict_cache(cache_path, payload)
        assert _load_c5_verdict_cache(cache_path) == payload


# ---------------------------------------------------------------------------
# AC1 — a second audit over an unchanged corpus makes zero LLM calls and
# returns the same findings.
# ---------------------------------------------------------------------------


class TestCacheHitUnchangedCorpus:
    def test_second_run_makes_zero_llm_calls(self, tmp_wiki_dir, monkeypatch):
        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        judged = _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        first = lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1, "First run must judge the one candidate pair"

        second = lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1, "Second run over an unchanged corpus must make zero new LLM calls"

        assert len(first) == 1
        assert len(second) == 1

    def test_second_run_returns_the_same_finding(self, tmp_wiki_dir, monkeypatch):
        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        first = lint_module._check_c5_page_pair(tmp_wiki_dir)
        second = lint_module._check_c5_page_pair(tmp_wiki_dir)

        assert first[0].model_dump() == second[0].model_dump()

    def test_cache_hit_counter_reflects_reused_verdicts(self, tmp_wiki_dir, monkeypatch):
        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert lint_module._c5_cache_hit_counter[0] == 0, "First run is all misses"

        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert lint_module._c5_cache_hit_counter[0] == 1, "Second run reuses the cached verdict"
        assert lint_module._c5_llm_call_counter[0] == 0, "Second run makes zero LLM calls"

    def test_frontmatter_only_edit_still_hits_cache(self, tmp_wiki_dir, monkeypatch):
        """Verdicts are keyed on body content only — a frontmatter-only re-ingest
        (e.g. ``updated`` bumped, prose unchanged) must not invalidate the cache."""
        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        judged = _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1

        # Re-write page-a with a bumped `updated` timestamp but identical body.
        page_dir = tmp_wiki_dir / "concepts"
        frontmatter = {
            "id": "page-a",
            "type": "concept",
            "created": "2026-05-26T00:00:00Z",
            "updated": "2026-07-01T00:00:00Z",  # bumped
            "sources": ["shared.md#s"],
            "status": "live",
            "open_questions": [],
        }
        content = (
            f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\nRefund takes 5 days.\n"
        )
        (page_dir / "page-a.md").write_text(content, encoding="utf-8")

        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1, "A frontmatter-only edit must not invalidate the verdict cache"


# ---------------------------------------------------------------------------
# AC2 — editing either page of a pair re-judges only the affected pairs.
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_editing_one_page_reinvalidates_only_its_pairs(self, tmp_wiki_dir, monkeypatch):
        """3 pages sharing one source -> 3 candidate pairs (A-B, A-C, B-C).

        After the first run judges all 3, editing only page A's body must
        re-judge exactly the two pairs containing A; the B-C pair (untouched)
        stays a cache hit.
        """
        _write_wiki_page(tmp_wiki_dir, "alpha", ["shared.md#s"], "Alpha says yes.")
        _write_wiki_page(tmp_wiki_dir, "beta", ["shared.md#s"], "Beta says no.")
        _write_wiki_page(tmp_wiki_dir, "gamma", ["shared.md#s"], "Gamma says maybe.")

        judged = _spy_llm(monkeypatch, severity="tension")

        import app.lint as lint_module

        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 3, "First run judges all 3 candidate pairs"

        # Edit only alpha's body.
        _write_wiki_page(tmp_wiki_dir, "alpha", ["shared.md#s"], "Alpha now says definitely yes.")

        judged.clear()
        lint_module._check_c5_page_pair(tmp_wiki_dir)

        assert set(judged) == {("alpha", "beta"), ("alpha", "gamma")}, (
            f"Expected only the two alpha-containing pairs re-judged; got {judged}"
        )
        assert lint_module._c5_cache_hit_counter[0] == 1, (
            "The untouched beta-gamma pair is a cache hit"
        )
        assert lint_module._c5_llm_call_counter[0] == 2

    def test_unrelated_pair_finding_is_unaffected_by_a_sibling_edit(
        self, tmp_wiki_dir, monkeypatch
    ):
        """The beta-gamma finding itself (not just the call count) survives
        untouched across a run that only edits alpha."""
        _write_wiki_page(tmp_wiki_dir, "alpha", ["shared.md#s"], "Alpha says yes.")
        _write_wiki_page(tmp_wiki_dir, "beta", ["shared.md#s"], "Beta says no.")
        _write_wiki_page(tmp_wiki_dir, "gamma", ["shared.md#s"], "Gamma says maybe.")

        _spy_llm(monkeypatch, severity="tension")

        import app.lint as lint_module

        first = lint_module._check_c5_page_pair(tmp_wiki_dir)
        first_beta_gamma = next(f for f in first if (f.page_a, f.page_b) == ("beta", "gamma"))

        _write_wiki_page(tmp_wiki_dir, "alpha", ["shared.md#s"], "Alpha now says definitely yes.")

        second = lint_module._check_c5_page_pair(tmp_wiki_dir)
        second_beta_gamma = next(f for f in second if (f.page_a, f.page_b) == ("beta", "gamma"))

        assert first_beta_gamma.model_dump() == second_beta_gamma.model_dump()


# ---------------------------------------------------------------------------
# A single stale/incompatible cache entry degrades to a miss, not a poisoned
# whole-cache failure.
# ---------------------------------------------------------------------------


class TestStaleCacheEntry:
    def test_one_incompatible_entry_is_treated_as_a_miss(self, tmp_wiki_dir, monkeypatch):
        from app.lint import _c5_verdict_cache_path, _write_c5_verdict_cache

        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        cache_path = _c5_verdict_cache_path(tmp_wiki_dir)
        # An entry shaped like an old/incompatible schema (missing required fields).
        _write_c5_verdict_cache(cache_path, {"deadbeef:cafef00d": {"severity": "direct"}})

        judged = _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        results = lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1, "The incompatible entry must be treated as a cache miss, not raise"
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Generation salt (issue #473) — a model or system-prompt change invalidates
# every existing entry instead of serving a stale verdict at cost=0.
# ---------------------------------------------------------------------------


class TestGenerationSalt:
    def test_salt_changes_when_resolved_model_changes(self, monkeypatch):
        from app.lint import _c5_generation_salt

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
        salt_a = _c5_generation_salt()

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4.1")
        salt_b = _c5_generation_salt()

        assert salt_a != salt_b

    def test_salt_changes_when_system_prompt_changes(self, monkeypatch):
        import app.lint as lint_module

        salt_a = lint_module._c5_generation_salt()
        monkeypatch.setattr(lint_module, "_C5_SYSTEM_PROMPT", "a different judge prompt")
        salt_b = lint_module._c5_generation_salt()

        assert salt_a != salt_b

    def test_salt_stable_for_unchanged_model_and_prompt(self, monkeypatch):
        from app.lint import _c5_generation_salt

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
        assert _c5_generation_salt() == _c5_generation_salt()


class TestGenerationSaltInvalidation:
    def test_model_change_does_not_serve_stale_verdict(self, tmp_wiki_dir, monkeypatch):
        """FAILS on pre-#473 main: the cache key omitted the model, so a model
        swap between two runs still served the first run's cached verdict."""
        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        judged = _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
        first = lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1, "First run must judge the one candidate pair"
        assert len(first) == 1

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4.1")
        second = lint_module._check_c5_page_pair(tmp_wiki_dir)

        assert len(judged) == 2, (
            "A model change must re-judge the pair, not reuse the prior model's verdict"
        )
        assert lint_module._c5_llm_call_counter[0] == 1, "The post-swap run is a fresh miss"
        assert lint_module._c5_cache_hit_counter[0] == 0, (
            "Nothing carries over across a model change"
        )
        assert len(second) == 1

    def test_system_prompt_change_does_not_serve_stale_verdict(self, tmp_wiki_dir, monkeypatch):
        """FAILS on pre-#473 main for the same reason as the model-change test
        above, but via a ``_C5_SYSTEM_PROMPT`` edit instead of an env var."""
        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        judged = _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1

        monkeypatch.setattr(
            lint_module, "_C5_SYSTEM_PROMPT", lint_module._C5_SYSTEM_PROMPT + "\nrevised."
        )
        second = lint_module._check_c5_page_pair(tmp_wiki_dir)

        assert len(judged) == 2, (
            "A system-prompt change must re-judge the pair, not reuse the prior prompt's verdict"
        )
        assert lint_module._c5_llm_call_counter[0] == 1
        assert lint_module._c5_cache_hit_counter[0] == 0
        assert len(second) == 1

    def test_unchanged_model_and_prompt_still_hits_cache(self, tmp_wiki_dir, monkeypatch):
        """Same content + same model + same prompt preserves the #446 cost win."""
        _write_wiki_page(tmp_wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(tmp_wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        judged = _spy_llm(monkeypatch, severity="direct")

        import app.lint as lint_module

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1

        lint_module._check_c5_page_pair(tmp_wiki_dir)
        assert len(judged) == 1, "Same generation salt must still hit the cache"
        assert lint_module._c5_cache_hit_counter[0] == 1
        assert lint_module._c5_llm_call_counter[0] == 0

    def test_run_lint_metrics_correct_after_model_change(self, lint_env, monkeypatch):
        """Metrics (llm_calls, cost_usd) stay accurate through a salt-driven
        invalidation surfaced via the full ``run_lint`` orchestrator."""
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        _spy_llm(monkeypatch, severity="direct")

        from app.lint import run_lint

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
        first = run_lint(**lint_env)
        assert first.summary.llm_calls == 1
        assert first.summary.cost_usd > 0.0

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4.1")
        second = run_lint(**lint_env)
        assert second.summary.llm_calls == 1, (
            "A model change must register as a real re-judge call, not a cache hit"
        )
        assert second.summary.cost_usd > 0.0, "Cost must reflect the real re-judge, not zero"


# ---------------------------------------------------------------------------
# LLM singleton rebuild on model change (issue #483) — the salt (#473) makes a
# model swap re-judge, but the pre-#483 singleton kept serving the OLD model's
# client under the NEW salt key unless the whole process restarted.
# ---------------------------------------------------------------------------


class _FakeChatOpenAI:
    """Records the ``model=`` kwarg it was constructed with; no network calls."""

    def __init__(self, *, model, **_kwargs):
        self.model = model


class TestLintLlmSingletonModelReset:
    def _reset_singleton(self, monkeypatch):
        import app.lint as lint_module

        monkeypatch.setattr(lint_module, "_lint_llm", None)
        monkeypatch.setattr(lint_module, "_lint_llm_model", None)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
        return lint_module

    def test_rebuilds_when_resolved_model_changes(self, monkeypatch):
        lint_module = self._reset_singleton(monkeypatch)

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
        first = lint_module.get_lint_llm()
        assert first.model == "gpt-4o-mini"

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4.1")
        second = lint_module.get_lint_llm()

        assert second.model == "gpt-4.1", (
            "An in-process OPENAI_LINT_MODEL swap must rebuild the singleton with "
            "the new model, not keep judging with the stale client under the new "
            "cache salt (issue #483)"
        )
        assert second is not first

    def test_does_not_rebuild_when_model_is_unchanged(self, monkeypatch):
        lint_module = self._reset_singleton(monkeypatch)

        monkeypatch.setenv("OPENAI_LINT_MODEL", "gpt-4o-mini")
        first = lint_module.get_lint_llm()
        second = lint_module.get_lint_llm()

        assert second is first, (
            "Steady state (unchanged model) must reuse the existing client — "
            "no extra client construction per AC"
        )


# ---------------------------------------------------------------------------
# AC3 — cache hits/misses and estimated cost visible in logs.
# ---------------------------------------------------------------------------


class TestCacheVisibleInLogs:
    def test_run_lint_cost_usd_reflects_only_real_calls(self, lint_env, monkeypatch):
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        _spy_llm(monkeypatch, severity="direct")

        from app.lint import run_lint

        first = run_lint(**lint_env)
        assert first.summary.llm_calls == 1
        assert first.summary.cost_usd > 0.0

        second = run_lint(**lint_env)
        assert second.summary.llm_calls == 0, "Second run is a full cache hit — zero LLM calls"
        assert second.summary.cost_usd == 0.0, (
            "Cost must reflect zero real calls, not the cached count"
        )

    def test_log_completed_line_reports_cache_hits(self, lint_env, monkeypatch):
        wiki_dir = lint_env["wiki_dir"]
        _write_wiki_page(wiki_dir, "page-a", ["shared.md#s"], "Refund takes 5 days.")
        _write_wiki_page(wiki_dir, "page-b", ["shared.md#s"], "Refund takes 14 days.")

        _spy_llm(monkeypatch, severity="direct")

        from app.lint import run_lint

        run_lint(**lint_env)
        run_lint(**lint_env)

        log_text = lint_env["log_path"].read_text(encoding="utf-8")
        completed_lines = [line for line in log_text.splitlines() if "lint_completed" in line]
        assert len(completed_lines) == 2
        assert "c5_cache_hits=0" in completed_lines[0]
        assert "c5_cache_hits=1" in completed_lines[1]
        assert "llm_calls=0" in completed_lines[1]


# ---------------------------------------------------------------------------
# A corrupt cache file degrades gracefully via run_lint's continue-on-error.
# ---------------------------------------------------------------------------


class TestCorruptCacheDegradesGracefully:
    def test_corrupt_cache_surfaces_in_check_errors_other_checks_unaffected(self, lint_env):
        from app.lint import _c5_verdict_cache_path

        wiki_dir = lint_env["wiki_dir"]

        # Plant an orphan page so C11 has something to find independently of C5.
        _write_wiki_page(wiki_dir, "orphan-page", ["missing.md#s"], "Orphan body.")

        cache_path = _c5_verdict_cache_path(wiki_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{not valid json", encoding="utf-8")

        from app.lint import run_lint

        result = run_lint(**lint_env)

        assert "c5" in result.check_errors
        assert result.findings.page_pairs == []
        # C11 still ran cleanly despite the C5 cache being corrupt.
        assert len(result.findings.orphans) == 1
