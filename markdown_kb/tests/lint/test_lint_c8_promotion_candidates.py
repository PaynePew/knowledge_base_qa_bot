"""Unit tests for the Slice 6-5 C8 check: promotion-candidate surfacing.

Phase 6 PRD #78 §"Phase 5 lint amendment" — C8.

C8 scans ``wiki/qa/*.md`` for ``status: draft`` Filed Answers, ranks them by
``count`` desc then ``updated`` desc (tiebreak), and caps the surfaced list via
the ``KB_LINT_PROMOTION_TOP_N`` env var (default 10).

These tests verify:
  - Ranking order (count desc, updated desc tiebreak).
  - ``status: draft`` filter (live / stale / superseded pages are excluded).
  - Top-N truncation via the env var.

Hermetic — no LLM invocation; the check is a pure filesystem scan.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_qa_page(
    wiki_dir: Path,
    slug: str,
    *,
    status: str = "draft",
    count: int = 1,
    question: str = "Sample question?",
    created: str = "2026-03-15T09:00:00Z",
    updated: str = "2026-03-15T09:00:00Z",
    sources: list[str] | None = None,
) -> Path:
    """Write a qa page under ``wiki/qa/`` with the requested frontmatter shape."""
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    page_path = qa_dir / f"{slug}.md"
    frontmatter: dict = {
        "id": slug,
        "type": "qa",
        "created": created,
        "updated": updated,
        "sources": sources or ["pricing.md#tier"],
        "status": status,
        "question": question,
        "count": count,
        "open_questions": [],
    }
    body = f"# {question}\n\nMocked filed answer body for `{slug}`.\n"
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


class TestC8PromotionCandidates:
    """C8 surfaces ranked, capped draft Filed Answers from wiki/qa/."""

    def test_only_draft_status_is_surfaced(self, tmp_wiki_dir):
        """Live / stale / superseded pages must NOT appear in the candidate list."""
        _write_qa_page(tmp_wiki_dir, "qa-draft", status="draft", count=3)
        _write_qa_page(tmp_wiki_dir, "qa-live", status="live", count=99)
        _write_qa_page(tmp_wiki_dir, "qa-stale", status="stale", count=42)
        _write_qa_page(tmp_wiki_dir, "qa-superseded", status="superseded", count=17)

        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        slugs = {f.slug for f in findings}
        assert slugs == {"qa-draft"}, f"Only the draft page should be surfaced; got {slugs}"

    def test_ranking_count_desc(self, tmp_wiki_dir):
        """Findings are ordered by ``count`` desc."""
        _write_qa_page(tmp_wiki_dir, "qa-low", status="draft", count=1)
        _write_qa_page(tmp_wiki_dir, "qa-high", status="draft", count=10)
        _write_qa_page(tmp_wiki_dir, "qa-mid", status="draft", count=5)

        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        assert [f.slug for f in findings] == ["qa-high", "qa-mid", "qa-low"]

    def test_ranking_updated_desc_tiebreak(self, tmp_wiki_dir):
        """Equal ``count`` -> ``updated`` desc tiebreak."""
        _write_qa_page(
            tmp_wiki_dir,
            "qa-old",
            status="draft",
            count=5,
            updated="2026-01-01T00:00:00Z",
        )
        _write_qa_page(
            tmp_wiki_dir,
            "qa-newer",
            status="draft",
            count=5,
            updated="2026-04-01T00:00:00Z",
        )
        _write_qa_page(
            tmp_wiki_dir,
            "qa-middle",
            status="draft",
            count=5,
            updated="2026-02-15T00:00:00Z",
        )

        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        assert [f.slug for f in findings] == ["qa-newer", "qa-middle", "qa-old"]

    def test_top_n_truncation_via_env_var(self, tmp_wiki_dir, monkeypatch):
        """``KB_LINT_PROMOTION_TOP_N`` caps the surfaced count."""
        for i in range(7):
            _write_qa_page(tmp_wiki_dir, f"qa-{i:02d}", status="draft", count=10 - i)

        monkeypatch.setenv("KB_LINT_PROMOTION_TOP_N", "3")
        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        assert len(findings) == 3
        # Top 3 by count desc: qa-00 (10), qa-01 (9), qa-02 (8)
        assert [f.slug for f in findings] == ["qa-00", "qa-01", "qa-02"]

    def test_top_n_default_is_10(self, tmp_wiki_dir, monkeypatch):
        """Default cap is 10 when the env var is unset."""
        monkeypatch.delenv("KB_LINT_PROMOTION_TOP_N", raising=False)
        for i in range(15):
            _write_qa_page(tmp_wiki_dir, f"qa-{i:02d}", status="draft", count=20 - i)

        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        assert len(findings) == 10

    def test_finding_carries_required_fields(self, tmp_wiki_dir):
        """Each finding records slug, question, count, age_days, cited_count."""
        _write_qa_page(
            tmp_wiki_dir,
            "qa-fields",
            status="draft",
            count=4,
            question="What is the answer?",
            sources=["a.md#x", "b.md#y", "c.md#z"],
        )

        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        assert len(findings) == 1
        f = findings[0]
        assert f.slug == "qa-fields"
        assert f.question == "What is the answer?"
        assert f.count == 4
        assert isinstance(f.age_days, float)
        assert f.cited_count == 3

    def test_question_is_truncated_in_finding(self, tmp_wiki_dir):
        """Long question text is truncated for the report column."""
        long_q = "Q" + ("u" * 200) + "?"
        _write_qa_page(tmp_wiki_dir, "qa-long", status="draft", count=1, question=long_q)

        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        assert len(findings) == 1
        # Should be much shorter than the original 200-char question.
        assert len(findings[0].question) < len(long_q)
        # Truncated representations end with the horizontal ellipsis character.
        assert findings[0].question.endswith("…")

    def test_empty_qa_dir_returns_empty_list(self, tmp_wiki_dir):
        """A wiki with no qa/ subdir or no qa pages yields no findings."""
        from app.lint import _check_c8_promotion_candidates

        findings = _check_c8_promotion_candidates(tmp_wiki_dir)
        assert findings == []
