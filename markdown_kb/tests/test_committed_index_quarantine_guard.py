"""Committed BM25 seed guard — the REAL artifact must stay quarantined (ADR-0029, #405).

The #307 / hybrid_kb #311 lesson applied here: the running app serves the
COMMITTED ``.kb/index.json`` seed, not a freshly-built one. A fresh
``build_index()`` unit test proves the *filter* works, but cannot catch a
seed that was never re-baked after the filter shipped — exactly the drift
this ADR exists to close (the live corpus had 6 ``status: failed_grounding``
pages serving from the stale seed at grill time). This guard reads the
COMMITTED seed directly, independent of any indexer in-memory state or
autouse tmp-path redirect, so a stale seed fails CI and forces a re-bake.

Hermetic: pure file reads (``json.loads`` on the committed index,
``app.indexer.split_frontmatter`` on the real wiki pages) — no ``build_index()``
call, no fixtures mutated, no LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.indexer as indexer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_INDEX = _REPO_ROOT / ".kb" / "index.json"
_WIKI_DIRS = [
    _REPO_ROOT / "wiki" / "entities",
    _REPO_ROOT / "wiki" / "concepts",
    _REPO_ROOT / "wiki" / "qa",
]

_seed_present = _COMMITTED_INDEX.exists()
_skip_reason = "committed BM25 seed (.kb/index.json) not present in this checkout"


def _quarantined_slugs() -> set[str]:
    """Slugs (bare filename stem) of every real wiki page on disk whose
    frontmatter ``status`` is ``failed_grounding``.

    Scanned directly off disk via ``indexer.split_frontmatter`` — the same
    parser + sentinel-comment handling ``build_index()`` uses in production —
    so this stays independent of whatever the committed seed currently says.
    """
    slugs: set[str] = set()
    for wiki_dir in _WIKI_DIRS:
        if not wiki_dir.exists():
            continue
        for md_file in wiki_dir.glob("*.md"):
            metadata, _body = indexer.split_frontmatter(md_file.read_text(encoding="utf-8"))
            if metadata.get("status") == "failed_grounding":
                slugs.add(md_file.stem)
    return slugs


def _committed_section_files() -> set[str]:
    payload = json.loads(_COMMITTED_INDEX.read_text(encoding="utf-8"))
    return {s["file"] for s in payload["sections"]}


@pytest.mark.skipif(not _seed_present, reason=_skip_reason)
def test_committed_index_excludes_every_failed_grounding_page():
    """ADR-0029 invariant on the shipped artifact: no quarantined page's
    Sections may appear in the seed the running app actually loads."""
    quarantined = _quarantined_slugs()
    if not quarantined:
        # Vacuously true: nothing on disk is quarantined, so nothing can leak.
        # This is the expected end-state once the C3 Sources are remediated
        # (#407/#408 flow) — a skip, not a failure, keeps CI honest then.
        pytest.skip("no status: failed_grounding page on disk — invariant vacuously true")

    served = _committed_section_files()
    leaked = quarantined & served
    assert not leaked, (
        f"committed .kb/index.json still serves {sorted(leaked)}, which carry "
        "status: failed_grounding on disk — ADR-0029 quarantine violated on the "
        "REAL artifact. Re-bake the seed: build_index() then git add -f .kb/index.json."
    )


@pytest.mark.skipif(not _seed_present, reason=_skip_reason)
def test_committed_index_not_empty():
    """Sanity check the guard above isn't vacuously true because the whole
    seed is empty (e.g. a bad rebuild wiped everything)."""
    assert _committed_section_files(), "committed .kb/index.json must not be empty"
