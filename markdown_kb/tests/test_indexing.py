"""Component tests for parse_markdown, build_index, and related helpers.

These tests cover Slice 1 acceptance criteria:
- test_parse_markdown_sample_docs: exact Section IDs match the real docs/
- test_parse_markdown_body_bearing_rule: H1 with intro + H2 child → two Sections
- test_parse_markdown_fenced_code: # inside fenced block is content, not heading
- test_parse_markdown_slug_collision: two ##Overview in one Source → -2 suffix
- test_build_index_counts: POST /index returns 3 files / 9 sections
- test_write_and_load_index_json: round-trip lossless through load_index_json
"""

import json
import re
import tempfile
from pathlib import Path

import pytest

import app.indexer as indexer
from app.indexer import (
    Section,
    build_index,
    load_index_json,
    parse_markdown,
    write_index_json,
)
from app.indexer import (
    sections as _sections,
)

from .conftest import REAL_DOCS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_dir: Path, filename: str, content: str) -> Path:
    p = tmp_dir / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Acceptance criterion: exact Section IDs from real docs/
# ---------------------------------------------------------------------------


def test_parse_markdown_sample_docs(tmp_path):
    """parse_markdown on each real doc produces the expected Section IDs."""
    # refund_policy.md
    refund_sections = parse_markdown(REAL_DOCS / "refund_policy.md")
    refund_ids = [s.id for s in refund_sections]
    assert "refund_policy.md#cancellation-window" in refund_ids
    assert "refund_policy.md#refund-timeline" in refund_ids
    assert "refund_policy.md#non-refundable-items" in refund_ids
    # The H1 "Refund Policy" has no body — it should NOT produce a Section
    assert not any("refund-policy" in sid for sid in refund_ids), (
        "Top-level H1 with no body should not produce a Section"
    )

    # account_help.md
    account_sections = parse_markdown(REAL_DOCS / "account_help.md")
    account_ids = [s.id for s in account_sections]
    assert "account_help.md#change-email-address" in account_ids
    assert "account_help.md#reset-password" in account_ids
    assert "account_help.md#delete-account" in account_ids

    # shipping_faq.md
    shipping_sections = parse_markdown(REAL_DOCS / "shipping_faq.md")
    shipping_ids = [s.id for s in shipping_sections]
    assert "shipping_faq.md#standard-shipping" in shipping_ids
    assert "shipping_faq.md#expedited-shipping" in shipping_ids
    assert "shipping_faq.md#tracking-number" in shipping_ids


# ---------------------------------------------------------------------------
# Acceptance criterion: body-bearing rule
# ---------------------------------------------------------------------------


def test_parse_markdown_body_bearing_rule(tmp_path):
    """H1 with intro body + H2 child → exactly two Sections."""
    md = "# H1\nIntro.\n## Child\nDetail.\n"
    p = _write(tmp_path, "test.md", md)
    result = parse_markdown(p)
    assert len(result) == 2, (
        f"Expected 2 sections (body-bearing H1 + leaf H2), got {len(result)}: "
        f"{[s.id for s in result]}"
    )
    ids = {s.id for s in result}
    assert "test.md#h1" in ids, f"H1 Section missing, got ids: {ids}"
    assert "test.md#child" in ids, f"H2 Section missing, got ids: {ids}"


# ---------------------------------------------------------------------------
# Acceptance criterion: fenced code
# ---------------------------------------------------------------------------


def test_parse_markdown_fenced_code(tmp_path):
    """A '# bash comment' inside a fenced block is content, not a heading."""
    md = (
        "# Real Heading\n"
        "Some intro.\n"
        "```bash\n"
        "# this is NOT a heading\n"
        "echo hello\n"
        "```\n"
        "More content.\n"
    )
    p = _write(tmp_path, "fenced.md", md)
    result = parse_markdown(p)
    # Only "Real Heading" is a real heading — fenced '# ...' must NOT produce a Section
    ids = [s.id for s in result]
    assert len(result) == 1, f"Expected 1 section (the real heading), got {len(result)}: {ids}"
    assert ids[0] == "fenced.md#real-heading"
    # Content should include the fenced code lines
    assert "echo hello" in result[0].content


# ---------------------------------------------------------------------------
# Acceptance criterion: slug collision
# ---------------------------------------------------------------------------


def test_parse_markdown_slug_collision(tmp_path):
    """Two '## Overview' in one Source → #overview and #overview-2."""
    md = "# Doc\n## Overview\nFirst overview content.\n## Overview\nSecond overview content.\n"
    p = _write(tmp_path, "collision.md", md)
    result = parse_markdown(p)
    ids = [s.id for s in result]
    assert "collision.md#overview" in ids, f"First overview missing: {ids}"
    assert "collision.md#overview-2" in ids, f"Second overview (with -2 suffix) missing: {ids}"


# ---------------------------------------------------------------------------
# Acceptance criterion: build_index counts 3 files / 9 sections
# ---------------------------------------------------------------------------


def test_build_index_counts(tmp_path, monkeypatch):
    """build_index on the real docs/ produces 3 files and 9 sections."""
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"

    # Patch INDEX_PATH to a temp location so we don't pollute the real .kb/
    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)

    files_count, sections_count = build_index(REAL_DOCS)
    assert files_count == 3, f"Expected 3 files, got {files_count}"
    assert sections_count == 9, f"Expected 9 sections, got {sections_count}"


# ---------------------------------------------------------------------------
# Acceptance criterion: .kb/index.json round-trip
# ---------------------------------------------------------------------------


def test_write_and_load_index_json(tmp_path, monkeypatch):
    """write_index_json then load_index_json round-trips losslessly."""
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"

    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)

    # Build from real docs
    build_index(REAL_DOCS)

    # Verify it's pretty-printed JSON
    assert index_path.exists(), ".kb/index.json must exist after build_index"
    raw = index_path.read_text(encoding="utf-8")
    # Pretty-printed means it has newlines beyond a single line
    assert "\n" in raw, "index.json should be pretty-printed (multi-line)"
    parsed = json.loads(raw)
    assert "sections" in parsed, "index.json must have a 'sections' key"
    assert "stats" in parsed, "index.json must have a 'stats' key"

    # Round-trip: reload from disk, check counts
    files_loaded, sections_loaded = load_index_json(index_path)
    assert files_loaded == 3
    assert sections_loaded == 9

    # Verify Section objects are fully restored
    loaded = indexer.sections
    original_ids = {s.id for s in loaded}
    assert "refund_policy.md#refund-timeline" in original_ids
    assert "account_help.md#change-email-address" in original_ids
