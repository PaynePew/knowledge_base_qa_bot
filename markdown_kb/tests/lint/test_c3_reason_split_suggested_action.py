"""Hermetic tests for C3's reason-split ``suggested_action`` (issue #407, ADR-0029
decision 3).

AC coverage:
  - ``claim_unsupported`` names at least one offending sentence and does not
    recommend a bare Re-ingest.
  - ``claim_unsupported`` with an empty ``unsupported_claims`` list (defensive /
    legacy page) degrades gracefully to an honest "claims not recorded" note
    instead of crashing or rendering a blank clause.
  - ``verifier_unavailable`` recommends Re-ingest (transient failure).

All tests are hermetic (no OPENAI_API_KEY required).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_failed_grounding_page(
    wiki_dir: Path,
    slug: str,
    source: str,
    *,
    reason: str = "claim_unsupported",
    unsupported_claims: list[str] | None = None,
) -> Path:
    """Write a wiki page with status=failed_grounding and grounding_failure block."""
    page_dir = wiki_dir / "concepts"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    grounding_failure: dict = {"reason": reason, "unsupported_claims": unsupported_claims or []}

    frontmatter = {
        "id": slug,
        "type": "concept",
        "created": "2026-07-04T00:00:00Z",
        "updated": "2026-07-04T00:00:00Z",
        "sources": [source],
        "status": "failed_grounding",
        "open_questions": [],
        "grounding_failure": grounding_failure,
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n# {slug}\n\nFailed content.\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


class TestC3ReasonSplitSuggestedAction:
    """suggested_action splits by reason instead of one undifferentiated template."""

    def test_claim_unsupported_names_the_offending_sentence(self, lint_env):
        """suggested_action for claim_unsupported must name at least one
        recorded unsupported claim verbatim."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "broken",
            "source.md#sec",
            reason="claim_unsupported",
            unsupported_claims=["The refund window is 90 days."],
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        assert "The refund window is 90 days." in findings[0].suggested_action

    def test_claim_unsupported_does_not_recommend_bare_reingest(self, lint_env):
        """The claim_unsupported action must not read as a plain retry
        suggestion — re-ingesting feeds the same unchanged Source to the
        same verifier and fails identically (ADR-0029 decision 3)."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "broken",
            "source.md#sec",
            reason="claim_unsupported",
            unsupported_claims=["fake claim"],
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        action = findings[0].suggested_action
        assert "not suggested" in action.lower() or "plain re-ingest" in action.lower(), (
            f"suggested_action must explicitly withhold a bare Re-ingest recommendation: {action!r}"
        )
        assert "amend the source" in action.lower()

    def test_claim_unsupported_empty_claims_degrades_gracefully(self, lint_env):
        """An empty unsupported_claims list on a claim_unsupported finding
        (defensive / legacy page) must not crash and must produce an honest
        'not recorded' note rather than a blank clause."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "broken",
            "source.md#sec",
            reason="claim_unsupported",
            unsupported_claims=[],
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        assert findings[0].unsupported_claims == []
        assert "not recorded" in findings[0].suggested_action.lower()

    def test_verifier_unavailable_recommends_reingest(self, lint_env):
        """verifier_unavailable is a transient failure — Re-ingest IS the
        correct suggested fix (ADR-0029 decision 3)."""
        wiki_dir = lint_env["wiki_dir"]
        _write_failed_grounding_page(
            wiki_dir,
            "broken",
            "source.md#sec",
            reason="verifier_unavailable",
            unsupported_claims=[],
        )

        from app.lint import _check_c3_failed_grounding

        findings = _check_c3_failed_grounding(wiki_dir)
        action_lower = findings[0].suggested_action.lower()
        assert "re-ingest" in action_lower
        assert "source.md#sec" in findings[0].suggested_action
