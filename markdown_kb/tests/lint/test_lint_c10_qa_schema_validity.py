"""Unit tests for the Slice 6-5 C10 check: qa-schema validity.

Phase 6 PRD #78 §"Phase 5 lint amendment" — C10.

C10 scans every ``wiki/qa/*.md`` page and validates four frontmatter properties:

  - ``status``    ∈ ``{live, draft, stale, superseded}``
  - ``type``      == ``"qa"``
  - ``question``  is a non-empty string
  - ``count``     is a positive integer

Each invalidity produces a single finding identifying the property and the
offending value. Closes the curator-typo orphan zombie failure mode
(PRD #78 Q8d) — Layer 3 of the orphan-visibility defence.

Hermetic — no LLM invocation.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_qa_page(wiki_dir: Path, slug: str, frontmatter: dict) -> Path:
    """Write a qa page with the provided frontmatter dict verbatim.

    The body is constant — tests focus on schema validation, not content.
    """
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    page_path = qa_dir / f"{slug}.md"
    body = f"# {slug}\n\nQa body."
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _valid_frontmatter(slug: str) -> dict:
    """Return a known-valid qa frontmatter dict — tests mutate one property."""
    return {
        "id": slug,
        "type": "qa",
        "created": "2026-03-15T09:00:00Z",
        "updated": "2026-03-15T09:00:00Z",
        "sources": ["a.md#x"],
        "status": "draft",
        "question": "Question?",
        "count": 1,
        "open_questions": [],
    }


class TestC10QaSchemaValidity:
    """Each of the four invalidity classes produces exactly one finding."""

    def test_invalid_status_capital_L(self, tmp_wiki_dir):
        """Status `Live` (capital L) is not in the valid set."""
        fm = _valid_frontmatter("qa-typo")
        fm["status"] = "Live"
        _write_qa_page(tmp_wiki_dir, "qa-typo", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert len(findings) == 1
        assert findings[0].page_slug == "qa-typo"
        assert findings[0].property_name == "status"
        assert findings[0].offending_value == "Live"

    def test_invalid_status_unknown_value(self, tmp_wiki_dir):
        """An unknown status value is flagged."""
        fm = _valid_frontmatter("qa-unknown-status")
        fm["status"] = "archived"
        _write_qa_page(tmp_wiki_dir, "qa-unknown-status", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert any(
            f.property_name == "status" and f.offending_value == "archived" for f in findings
        )

    def test_wrong_type_for_qa_page(self, tmp_wiki_dir):
        """A page under wiki/qa/ with ``type != "qa"`` is flagged."""
        fm = _valid_frontmatter("qa-wrong-type")
        fm["type"] = "entity"
        _write_qa_page(tmp_wiki_dir, "qa-wrong-type", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert any(f.property_name == "type" and f.offending_value == "entity" for f in findings)

    def test_missing_question(self, tmp_wiki_dir):
        """Missing ``question`` property is flagged."""
        fm = _valid_frontmatter("qa-missing-q")
        fm.pop("question")
        _write_qa_page(tmp_wiki_dir, "qa-missing-q", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert any(f.property_name == "question" for f in findings)

    def test_empty_question_is_flagged(self, tmp_wiki_dir):
        """Empty-string ``question`` value is flagged."""
        fm = _valid_frontmatter("qa-empty-q")
        fm["question"] = "   "
        _write_qa_page(tmp_wiki_dir, "qa-empty-q", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert any(f.property_name == "question" for f in findings)

    def test_missing_count(self, tmp_wiki_dir):
        """Missing ``count`` property is flagged."""
        fm = _valid_frontmatter("qa-missing-count")
        fm.pop("count")
        _write_qa_page(tmp_wiki_dir, "qa-missing-count", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert any(f.property_name == "count" for f in findings)

    def test_nonpositive_count(self, tmp_wiki_dir):
        """``count`` of 0 (non-positive) is flagged."""
        fm = _valid_frontmatter("qa-zero-count")
        fm["count"] = 0
        _write_qa_page(tmp_wiki_dir, "qa-zero-count", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert any(f.property_name == "count" for f in findings)

    def test_string_count_is_flagged(self, tmp_wiki_dir):
        """``count`` typed as string is flagged (not a positive int)."""
        fm = _valid_frontmatter("qa-string-count")
        fm["count"] = "many"
        _write_qa_page(tmp_wiki_dir, "qa-string-count", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert any(f.property_name == "count" for f in findings)

    def test_valid_page_produces_no_findings(self, tmp_wiki_dir):
        """A schema-valid qa page produces zero findings."""
        fm = _valid_frontmatter("qa-clean")
        _write_qa_page(tmp_wiki_dir, "qa-clean", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        assert findings == []

    def test_multiple_invalidities_each_produce_a_finding(self, tmp_wiki_dir):
        """A page with multiple broken properties yields one finding per property."""
        fm = _valid_frontmatter("qa-multi")
        fm["status"] = "Live"
        fm["type"] = "concept"
        fm.pop("question")
        fm["count"] = -1
        _write_qa_page(tmp_wiki_dir, "qa-multi", fm)

        from app.lint import _check_c10_qa_schema_validity

        findings = _check_c10_qa_schema_validity(tmp_wiki_dir)
        properties = {f.property_name for f in findings if f.page_slug == "qa-multi"}
        assert {"status", "type", "question", "count"}.issubset(properties)
