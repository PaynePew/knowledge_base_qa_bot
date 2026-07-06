"""Tests for issue #509 / ADR-0033 decision 1: parse_markdown preamble becomes
a Section (``#intro``) instead of being silently dropped.

Root cause covered: before this fix, body content preceding a heading-bearing
Source's first heading was accumulated onto ``stack[-1]`` — but the stack is
empty until the first heading is pushed, so those lines were discarded with no
Section and no ``parse_warning``. A transcribed 63-page book with a single
stray heading at ~60% depth lost 60% of its text this way.

Tests:
- test_preamble_becomes_intro_section: preamble + headings -> #intro Section
  carrying exactly the pre-heading body, plus a `parse_warning`.
- test_whitespace_only_preamble_no_section_no_warning: regression — no Section,
  no warning, byte-identical to today when the preamble is blank.
- test_headed_source_with_no_preamble_unaffected: regression — a Source whose
  body starts directly with a heading is byte-identical (no #intro Section).
- test_frontmatter_then_preamble: frontmatter is stripped first; the preamble
  Section's content excludes it but inherits the frontmatter metadata.
- test_fenced_code_block_in_preamble: a fenced block containing `#` lines
  before the first real heading is preamble content, not a heading.
- test_intro_slug_collision_with_real_intro_heading: a real "Intro" heading
  later in the same Source collides with the synthesized #intro slug and
  gets the existing -2 suffix — never a silent overwrite.
- test_socrates_transcript_shape_preamble_plus_single_heading: fixture
  mirroring the 蘇格拉底 transcript (single stray heading at ~60% depth) ->
  2 Sections with 100% body-character coverage.
"""

from __future__ import annotations

from pathlib import Path

import app.logger as logger_module
from app.indexer import parse_markdown


def _write(tmp_dir: Path, filename: str, content: str) -> Path:
    p = tmp_dir / filename
    p.write_text(content, encoding="utf-8")
    return p


def _log_content() -> str:
    return (
        logger_module.LOG_PATH.read_text(encoding="utf-8")
        if logger_module.LOG_PATH.exists()
        else ""
    )


# ---------------------------------------------------------------------------
# AC: preamble + headings -> #intro Section carrying exactly the pre-heading body
# ---------------------------------------------------------------------------


def test_preamble_becomes_intro_section(tmp_path):
    md = "Some intro text before any heading.\n\n# First Heading\nBody one.\n"
    p = _write(tmp_path, "intro.md", md)

    result = parse_markdown(p)

    assert len(result) == 2, f"Expected 2 sections, got {len(result)}: {[s.id for s in result]}"
    ids = [s.id for s in result]
    assert "intro.md#intro" in ids, f"#intro Section missing: {ids}"
    assert "intro.md#first-heading" in ids, f"heading Section missing: {ids}"

    intro_section = next(s for s in result if s.id == "intro.md#intro")
    assert intro_section.content == "Some intro text before any heading."
    assert intro_section.heading == "intro", (
        f"heading must be the file stem, got {intro_section.heading!r}"
    )
    assert intro_section.heading_path == ["intro"]

    log = _log_content()
    assert "parse_warning" in log, f"preamble Section must emit parse_warning.\nLog:\n{log}"
    assert "intro.md#intro" in log, f"parse_warning must reference the Section id.\nLog:\n{log}"


# ---------------------------------------------------------------------------
# AC: whitespace-only preamble -> no Section, no warning
# ---------------------------------------------------------------------------


def test_whitespace_only_preamble_no_section_no_warning(tmp_path):
    md = "\n\n   \n\n# Heading\nBody.\n"
    p = _write(tmp_path, "blank_preamble.md", md)

    result = parse_markdown(p)

    ids = [s.id for s in result]
    assert len(result) == 1, f"Expected 1 section (no preamble Section), got {len(result)}: {ids}"
    assert "blank_preamble.md#intro" not in ids

    log = _log_content()
    assert "preamble" not in log, f"whitespace-only preamble must not log a warning.\nLog:\n{log}"


# ---------------------------------------------------------------------------
# Regression: a heading-bearing Source with no preamble is unaffected
# ---------------------------------------------------------------------------


def test_headed_source_with_no_preamble_unaffected(tmp_path):
    md = "# H1\nIntro.\n## Child\nDetail.\n"
    p = _write(tmp_path, "no_preamble.md", md)

    result = parse_markdown(p)

    ids = [s.id for s in result]
    assert len(result) == 2, f"Expected 2 sections (unchanged), got {len(result)}: {ids}"
    assert "no_preamble.md#intro" not in ids


# ---------------------------------------------------------------------------
# AC: frontmatter + preamble
# ---------------------------------------------------------------------------


def test_frontmatter_then_preamble(tmp_path):
    md = "---\nkey: value\n---\nPreamble text after frontmatter.\n\n# Heading\nBody.\n"
    p = _write(tmp_path, "fm_preamble.md", md)

    result = parse_markdown(p)

    ids = [s.id for s in result]
    assert "fm_preamble.md#intro" in ids, f"#intro Section missing: {ids}"
    intro_section = next(s for s in result if s.id == "fm_preamble.md#intro")

    # Frontmatter must not leak into the preamble body...
    assert intro_section.content == "Preamble text after frontmatter."
    # ...but frontmatter metadata still attaches to every Section, incl. #intro.
    assert intro_section.metadata.get("key") == "value", intro_section.metadata


# ---------------------------------------------------------------------------
# AC: a fenced code block containing `#` lines before the first real heading
# ---------------------------------------------------------------------------


def test_fenced_code_block_in_preamble(tmp_path):
    md = (
        "Intro before fence.\n"
        "```bash\n"
        "# not a heading\n"
        "echo hi\n"
        "```\n"
        "More intro.\n\n"
        "# Real Heading\n"
        "Body.\n"
    )
    p = _write(tmp_path, "fenced_preamble.md", md)

    result = parse_markdown(p)

    ids = [s.id for s in result]
    # Exactly 2 Sections — the fenced '#' lines must NOT be treated as headings.
    assert len(result) == 2, f"Expected 2 sections, got {len(result)}: {ids}"
    assert "fenced_preamble.md#intro" in ids
    assert "fenced_preamble.md#real-heading" in ids

    intro_section = next(s for s in result if s.id == "fenced_preamble.md#intro")
    assert "echo hi" in intro_section.content
    assert "# not a heading" in intro_section.content
    assert "More intro." in intro_section.content


# ---------------------------------------------------------------------------
# AC: #intro / real "intro" heading slug collision -> -2 suffix, never overwrite
# ---------------------------------------------------------------------------


def test_intro_slug_collision_with_real_intro_heading(tmp_path):
    md = "Preamble content.\n\n# Intro\nReal heading body.\n"
    p = _write(tmp_path, "collision.md", md)

    result = parse_markdown(p)

    ids = [s.id for s in result]
    assert "collision.md#intro" in ids, f"synthesized preamble Section missing: {ids}"
    assert "collision.md#intro-2" in ids, f"real 'Intro' heading must get -2 suffix: {ids}"

    preamble_section = next(s for s in result if s.id == "collision.md#intro")
    heading_section = next(s for s in result if s.id == "collision.md#intro-2")
    assert preamble_section.content == "Preamble content."
    assert heading_section.content == "Real heading body."


# ---------------------------------------------------------------------------
# AC: fixture mirroring the 蘇格拉底 transcript shape
# ---------------------------------------------------------------------------


def test_socrates_transcript_shape_preamble_plus_single_heading(tmp_path):
    """Single stray heading at ~60% depth -> 2 Sections, 100% body coverage.

    Mirrors the failure that motivated ADR-0033 decision 1: a transcribed book
    with exactly one heading (a mid-file page running-header) lost every
    character before it. Both halves here must survive intact.
    """
    preamble = ("蘇格拉底的申辯記錄了蘇格拉底在雅典法庭上為自己辯護的言論。" * 6).strip()
    tail = ("他畢生追求智慧與美德，即便面對死刑判決仍拒絕逃亡或屈服。" * 4).strip()
    md = f"{preamble}\n\n# 柏拉图对话集 43\n\n{tail}\n"
    p = _write(tmp_path, "socrates.md", md)

    result = parse_markdown(p)

    ids = [s.id for s in result]
    assert len(result) == 2, f"Expected 2 sections, got {len(result)}: {ids}"
    assert "socrates.md#intro" in ids

    intro_section = next(s for s in result if s.id == "socrates.md#intro")
    chapter_section = next(s for s in result if s.id != "socrates.md#intro")

    assert intro_section.content == preamble
    assert chapter_section.content == tail
    assert chapter_section.heading == "柏拉图对话集 43"

    # 100% body-character coverage: the two Sections together account for
    # every non-heading, non-blank-line character in the original body.
    combined_len = len(intro_section.content) + len(chapter_section.content)
    assert combined_len == len(preamble) + len(tail), (
        "Sections must cover the full body with no characters dropped"
    )
