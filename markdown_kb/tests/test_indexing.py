"""Component tests for parse_markdown, build_index, and related helpers.

Original Slice 1 tests (parse_markdown behaviour) are preserved verbatim.
Slice 4-2 tests replace the obsolete docs/-scoped build_index tests with
wiki-whitelist scope + meta-file exclusion + wiki_layer_empty emission tests.
Slice 4-3a tests assert bare-slug Section.id and Section.file for wiki-derived Sections.

Tests:
- test_parse_markdown_sample_docs: exact Section IDs match the real docs/
- test_parse_markdown_body_bearing_rule: H1 with intro + H2 child → two Sections
- test_parse_markdown_fenced_code: # inside fenced block is content, not heading
- test_parse_markdown_slug_collision: two ##Overview in one Source → -2 suffix
- test_parse_markdown_with_source_id: source_id param overrides filename in Section.id/file
- test_build_index_scans_wiki_subdirs: wiki entities/ + concepts/ are indexed
- test_build_index_wiki_sections_use_bare_slug: Section.id and .file use bare slug
- test_build_index_excludes_wiki_meta_files: index.md, log.md, hot.md, README.md excluded
- test_build_index_meta_files_not_in_index_json: meta-files absent from .kb/index.json
- test_write_and_load_index_json: round-trip lossless through load_index_json
- test_wiki_layer_empty_emitted_when_both_subdirs_empty: new log kind emitted
- test_wiki_layer_empty_not_emitted_when_entities_has_page: no emission with content
- test_wiki_layer_empty_not_emitted_when_concepts_has_page: no emission with content
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

# Path to hand-written wiki fixtures (deterministic, not LLM-generated)
WIKI_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wiki"


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
# Slice 4-3a AC: parse_markdown source_id param overrides filename in Section.id/file
# ---------------------------------------------------------------------------


def test_parse_markdown_with_source_id(tmp_path):
    """When source_id is supplied to parse_markdown, Section.id uses that prefix
    instead of the filename. Section.file also uses the bare source_id.

    This is the mechanism build_index uses to produce bare-slug IDs for
    wiki-derived Sections (e.g. 'refund-policy' instead of 'refund-policy.md').
    """
    md = "## Cancellation Window\nContent about cancellations.\n"
    p = _write(tmp_path, "refund-policy.md", md)
    result = parse_markdown(p, source_id="refund-policy")
    assert len(result) == 1, f"Expected 1 section, got {len(result)}: {[s.id for s in result]}"
    sec = result[0]
    # id must use bare slug, not filename
    assert sec.id == "refund-policy#cancellation-window", (
        f"Expected 'refund-policy#cancellation-window', got {sec.id!r}"
    )
    # file must also be the bare slug
    assert sec.file == "refund-policy", f"Expected file='refund-policy', got {sec.file!r}"


# ---------------------------------------------------------------------------
# Slice 4-3a AC: build_index produces bare-slug Section.id and .file for wiki
# ---------------------------------------------------------------------------


def test_build_index_wiki_sections_use_bare_slug(tmp_path, monkeypatch):
    """Sections from SOURCE_DIRS use the bare slug form: no type subdir, no .md.

    A wiki page at wiki/entities/acme-shop.md with heading '## Standard Shipping'
    produces Section.id = 'acme-shop#standard-shipping', Section.file = 'acme-shop'.
    """
    import app.indexer as indexer_module
    import app.logger as logger_module

    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    entities_dir.mkdir(parents=True)
    (wiki_dir / "concepts").mkdir(parents=True)

    # Copy acme-shop fixture which has heading 'Acme Shop'
    src = WIKI_FIXTURES / "entities" / "acme-shop.md"
    (entities_dir / "acme-shop.md").write_bytes(src.read_bytes())

    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"

    monkeypatch.setattr(indexer_module, "INDEX_PATH", index_path)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", wiki_dir / "log.md")
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [entities_dir, wiki_dir / "concepts"],
    )

    build_index()

    secs = indexer_module.sections
    assert secs, "Expected at least one section indexed"

    for sec in secs:
        # Section.file must be bare slug — no .md extension
        assert not sec.file.endswith(".md"), (
            f"Section.file must NOT end with .md for wiki-derived sections, got {sec.file!r}"
        )
        # Section.id prefix must match Section.file
        id_prefix = sec.id.split("#")[0] if "#" in sec.id else sec.id
        assert id_prefix == sec.file, (
            f"Section.id prefix '{id_prefix}' must match Section.file '{sec.file}'"
        )

    # Specifically: acme-shop.md's sections should use 'acme-shop' as slug
    acme_secs = [s for s in secs if s.file == "acme-shop"]
    assert acme_secs, (
        f"Expected sections with file='acme-shop', got files: {[s.file for s in secs]}"
    )
    # Check the Section.id form: 'acme-shop#<heading-slug>'
    acme_id_prefixes = {s.id.split("#")[0] if "#" in s.id else s.id for s in acme_secs}
    assert all(p == "acme-shop" for p in acme_id_prefixes), (
        f"Expected all acme-shop Section.id to start with 'acme-shop', got: "
        f"{[s.id for s in acme_secs]}"
    )


# ---------------------------------------------------------------------------
# Slice 4-2 AC: build_index scans wiki/entities/ and wiki/concepts/ (not docs/)
# ---------------------------------------------------------------------------


def test_build_index_scans_wiki_subdirs(tmp_path, monkeypatch):
    """build_index (default SOURCE_DIRS) indexes wiki/entities/ and wiki/concepts/.

    Uses hand-written fixtures under tests/fixtures/wiki/.
    """
    import app.indexer as indexer_module
    import app.logger as logger_module

    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"
    wiki_dir = tmp_path / "wiki"

    # Copy fixtures into the tmp wiki
    entities_src = WIKI_FIXTURES / "entities"
    concepts_src = WIKI_FIXTURES / "concepts"

    entities_dst = wiki_dir / "entities"
    concepts_dst = wiki_dir / "concepts"
    entities_dst.mkdir(parents=True)
    concepts_dst.mkdir(parents=True)

    for f in entities_src.iterdir():
        (entities_dst / f.name).write_bytes(f.read_bytes())
    for f in concepts_src.iterdir():
        (concepts_dst / f.name).write_bytes(f.read_bytes())

    monkeypatch.setattr(indexer_module, "INDEX_PATH", index_path)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    # Rebuild SOURCE_DIRS to use the patched WIKI_DIR
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts"],
    )

    files_count, sections_count = build_index()

    assert files_count >= 1, f"Expected at least 1 file indexed, got {files_count}"
    assert sections_count >= 1, f"Expected at least 1 section indexed, got {sections_count}"

    # Sections should come from the wiki fixtures, not docs/.
    # Slice 4-3a: Section.file uses bare slug (no .md extension).
    sec_files = {s.file for s in indexer_module.sections}
    assert "acme-shop" in sec_files, (
        f"Expected acme-shop fixture to be indexed with bare slug, got files: {sec_files}"
    )
    # No section file should end with .md for wiki-derived sources.
    for sf in sec_files:
        assert not sf.endswith(".md"), (
            f"Wiki section file must NOT end with .md (bare slug expected), got: {sf!r}"
        )


# ---------------------------------------------------------------------------
# Slice 4-2 AC: meta-files directly under wiki/ are NOT in the index
# ---------------------------------------------------------------------------


def test_build_index_excludes_wiki_meta_files(tmp_path, monkeypatch):
    """Meta-files (index.md, log.md, hot.md, README.md) directly under wiki/
    must NOT appear in the section index even when present on disk.

    The whitelist semantics (only entities/ and concepts/ subdirs) naturally
    exclude them — this test verifies the invariant explicitly.
    """
    import app.indexer as indexer_module
    import app.logger as logger_module

    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    entities_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)

    # Write meta-files directly under wiki/ — they must NOT be indexed
    for meta_name in ("index.md", "log.md", "hot.md", "README.md"):
        (wiki_dir / meta_name).write_text(
            f"# {meta_name}\n\nThis is a meta-file that must not be indexed.\n",
            encoding="utf-8",
        )

    # Copy one real fixture so the index is non-empty
    src_fixture = WIKI_FIXTURES / "concepts" / "cancellation-window.md"
    (concepts_dir / "cancellation-window.md").write_bytes(src_fixture.read_bytes())

    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", wiki_dir / "log.md")

    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [entities_dir, concepts_dir],
    )

    build_index()

    indexed_files = {s.file for s in indexer_module.sections}
    for meta_name in ("index.md", "log.md", "hot.md", "README.md"):
        assert meta_name not in indexed_files, (
            f"Meta-file '{meta_name}' must NOT appear in the section index but was found.\n"
            f"Indexed files: {indexed_files}"
        )


# ---------------------------------------------------------------------------
# Slice 4-2 AC: meta-files absent from .kb/index.json
# ---------------------------------------------------------------------------


def test_build_index_meta_files_not_in_index_json(tmp_path, monkeypatch):
    """Verify .kb/index.json does not contain any meta-file Section after build_index."""
    import app.indexer as indexer_module
    import app.logger as logger_module

    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    entities_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)

    for meta_name in ("index.md", "log.md", "hot.md", "README.md"):
        (wiki_dir / meta_name).write_text(
            f"# {meta_name}\n\nMeta content.\n",
            encoding="utf-8",
        )

    src_fixture = WIKI_FIXTURES / "concepts" / "cancellation-window.md"
    (concepts_dir / "cancellation-window.md").write_bytes(src_fixture.read_bytes())

    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"

    monkeypatch.setattr(indexer_module, "INDEX_PATH", index_path)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", wiki_dir / "log.md")

    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [entities_dir, concepts_dir],
    )

    build_index()

    assert index_path.exists(), ".kb/index.json must exist after build_index"
    raw = index_path.read_text(encoding="utf-8")
    parsed = json.loads(raw)

    indexed_file_names = {s["file"] for s in parsed["sections"]}
    for meta_name in ("index.md", "log.md", "hot.md", "README.md"):
        assert meta_name not in indexed_file_names, (
            f"Meta-file '{meta_name}' must NOT appear in index.json sections but was found.\n"
            f"Indexed files: {indexed_file_names}"
        )


# ---------------------------------------------------------------------------
# Acceptance criterion: .kb/index.json round-trip (preserved from Slice 1)
# ---------------------------------------------------------------------------


def test_write_and_load_index_json(tmp_path, monkeypatch):
    """write_index_json then load_index_json round-trips losslessly.

    Uses the 3-Source hermetic fixture dir (tests/fixtures/docs/) rather than
    the real docs/ so the count assertions remain stable regardless of how many
    files live under docs/ (issue #142: docs/fake-docs/ is now part of docs/).
    """
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"

    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)

    # Use the hermetic 3-Source fixture, not the real docs/
    fixture_docs = Path(__file__).resolve().parent / "fixtures" / "docs"
    build_index(fixture_docs)

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


# ---------------------------------------------------------------------------
# Slice S2 (#300): deterministic serialization — stats.doc_freq keys are sorted
# ---------------------------------------------------------------------------


def test_write_index_json_doc_freq_keys_are_sorted(tmp_path, monkeypatch):
    """The persisted ``stats.doc_freq`` must have keys in canonical sorted order.

    ``doc_freq`` is a Counter built by iterating ``set(sec.tokens)``, whose
    iteration order is non-deterministic across processes (hash randomisation),
    so persisting ``dict(doc_freq)`` produced a spurious full-file reorder diff
    on every re-bake of the committed ``.kb/index.json`` seed. Sorting the keys
    makes the serialization deterministic. This is metadata-only: BM25 reads
    doc_freq by key, never by position, so scores are unchanged.
    """
    kb_dir = tmp_path / ".kb"
    index_path = kb_dir / "index.json"
    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)

    # Index the hermetic 3-Source fixture so doc_freq has many distinct tokens.
    fixture_docs = Path(__file__).resolve().parent / "fixtures" / "docs"
    build_index(fixture_docs)

    raw = index_path.read_text(encoding="utf-8")
    # Preserve JSON object key order from the file (default dict preserves it).
    parsed = json.loads(raw)
    doc_freq_keys = list(parsed["stats"]["doc_freq"].keys())

    assert len(doc_freq_keys) > 1, "fixture corpus must yield multiple tokens"
    assert doc_freq_keys == sorted(doc_freq_keys), (
        "stats.doc_freq keys must be persisted in sorted order for a stable "
        f"re-bake diff; got: {doc_freq_keys}"
    )


# ---------------------------------------------------------------------------
# Slice 4-2 AC: wiki_layer_empty emitted when both subdirs have zero sections
# ---------------------------------------------------------------------------


def test_wiki_layer_empty_emitted_when_both_subdirs_empty(tmp_path, monkeypatch):
    """wiki_layer_empty log kind is emitted when both entities/ and concepts/
    scan to zero sections (both directories empty).
    """
    import app.indexer as indexer_module
    import app.logger as logger_module

    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    entities_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)

    log_path = wiki_dir / "log.md"
    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [entities_dir, concepts_dir],
    )

    build_index()

    assert log_path.exists(), "log.md must exist after build_index"
    content = log_path.read_text(encoding="utf-8")
    assert "wiki_layer_empty" in content, (
        f"Expected 'wiki_layer_empty' in log when both wiki subdirs are empty.\n"
        f"Log content:\n{content}"
    )


# ---------------------------------------------------------------------------
# Slice 4-2 AC: wiki_layer_empty NOT emitted when entities/ has at least one page
# ---------------------------------------------------------------------------


def test_wiki_layer_empty_not_emitted_when_entities_has_page(tmp_path, monkeypatch):
    """wiki_layer_empty must NOT be emitted when entities/ contains at least one page."""
    import app.indexer as indexer_module
    import app.logger as logger_module

    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    entities_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)

    # One entity fixture — concepts/ is empty
    src = WIKI_FIXTURES / "entities" / "acme-shop.md"
    (entities_dir / "acme-shop.md").write_bytes(src.read_bytes())

    log_path = wiki_dir / "log.md"
    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [entities_dir, concepts_dir],
    )

    build_index()

    content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    assert "wiki_layer_empty" not in content, (
        f"wiki_layer_empty must NOT be emitted when entities/ has a page.\nLog content:\n{content}"
    )


# ---------------------------------------------------------------------------
# Slice 4-2 AC: wiki_layer_empty NOT emitted when concepts/ has at least one page
# ---------------------------------------------------------------------------


def test_wiki_layer_empty_not_emitted_when_concepts_has_page(tmp_path, monkeypatch):
    """wiki_layer_empty must NOT be emitted when concepts/ contains at least one page."""
    import app.indexer as indexer_module
    import app.logger as logger_module

    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    entities_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)

    # One concept fixture — entities/ is empty
    src = WIKI_FIXTURES / "concepts" / "cancellation-window.md"
    (concepts_dir / "cancellation-window.md").write_bytes(src.read_bytes())

    log_path = wiki_dir / "log.md"
    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [entities_dir, concepts_dir],
    )

    build_index()

    content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    assert "wiki_layer_empty" not in content, (
        f"wiki_layer_empty must NOT be emitted when concepts/ has a page.\nLog content:\n{content}"
    )
