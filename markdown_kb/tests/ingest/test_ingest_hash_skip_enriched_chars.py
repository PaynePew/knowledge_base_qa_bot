"""Tests for issue #523: the hash-skip exit reads persisted enriched_chars.

Both ``ingest_sources`` (sync) and ``aingest_sources`` (async) build the
skipped-source ``IngestSourceResult`` BEFORE the classify step that reads
frontmatter via ``_frontmatter_enriched_chars`` — so a hash-skipped enriched
Source previously fell through to the schema default ``enriched_chars: 0``,
contradicting the schema docstring ("0 for any Source that was never
enriched"). ``_skip_path_enriched_chars`` (ingest.py) now reads the same
frontmatter on the skip exit.

No fake LLM is needed: the hash-skip decision short-circuits before any LLM
call, so these tests plant a docs Source + a matching wiki page with
``source_hashes`` already set (same technique as
``test_ingest_hash_skip.py``) and assert directly on the skip result.

Tests:
- test_hash_skip_reports_persisted_enriched_chars: a hash-skipped Source
  carrying ``structure: enriched`` / ``enriched_chars: 27`` reports 27 on the
  skipped result (sync).
- test_hash_skip_non_enriched_source_still_reports_zero: a hash-skipped
  Source with no enrichment marker still reports 0 (sync).
- test_aingest_hash_skip_reports_persisted_enriched_chars: same first case
  through the async sibling (``aingest_sources``).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import yaml

import app.indexer as indexer_module
import app.ingest as ingest_module

_ENRICHED_CHARS = 27
_ENRICHED_SOURCE_TEXT = (
    "---\n"
    "structure: enriched\n"
    f"enriched_chars: {_ENRICHED_CHARS}\n"
    "---\n"
    "\n"
    "## Chapter One\n\nThe opening chapter.\n\n"
    "## Chapter Two\n\nThe second chapter.\n"
)
_PLAIN_SOURCE_TEXT = "## Only Section\n\nSome body text, never enriched.\n"


def _plant_hash_skip_fixture(
    docs_dir: Path, wiki_dir: Path, source_name: str, source_text: str
) -> None:
    """Write a Source file plus a wiki page whose ``source_hashes`` already
    matches it, so ``_should_skip_source`` returns True with no LLM call."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    source_path = docs_dir / source_name
    source_path.write_text(source_text, encoding="utf-8")
    docs_body_hash = hashlib.sha256(source_text.encode()).hexdigest()

    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    fm = {
        "id": "existing-page",
        "type": "concept",
        "sources": [f"{source_name}#existing-page"],
        "source_hashes": {source_name: {"docs_body": docs_body_hash, "raw": None}},
    }
    fm_yaml = yaml.dump(fm, default_flow_style=False, allow_unicode=True)
    (concepts_dir / "existing-page.md").write_text(
        f"---\n{fm_yaml}---\n\nBody text.\n", encoding="utf-8"
    )


def test_hash_skip_reports_persisted_enriched_chars(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    _plant_hash_skip_fixture(docs_dir, wiki_dir, "book.md", _ENRICHED_SOURCE_TEXT)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    batch = ingest_module.ingest_sources(["book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert batch.results == [], "expected a hash-skip, not a fresh ingest"
    assert len(batch.skipped_sources) == 1
    skipped = batch.skipped_sources[0]
    assert skipped.status == "skipped"
    assert skipped.enriched_chars == _ENRICHED_CHARS, skipped


def test_hash_skip_non_enriched_source_still_reports_zero(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    _plant_hash_skip_fixture(docs_dir, wiki_dir, "plain.md", _PLAIN_SOURCE_TEXT)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    batch = ingest_module.ingest_sources(["plain.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert len(batch.skipped_sources) == 1
    assert batch.skipped_sources[0].enriched_chars == 0


def test_aingest_hash_skip_reports_persisted_enriched_chars(tmp_path, monkeypatch):
    """The async sibling's hand-duplicated skip exit reads the same value."""
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    _plant_hash_skip_fixture(docs_dir, wiki_dir, "book.md", _ENRICHED_SOURCE_TEXT)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    batch = asyncio.run(
        ingest_module.aingest_sources(["book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    )

    assert batch.results == [], "expected a hash-skip, not a fresh ingest"
    assert len(batch.skipped_sources) == 1
    skipped = batch.skipped_sources[0]
    assert skipped.status == "skipped"
    assert skipped.enriched_chars == _ENRICHED_CHARS, skipped
