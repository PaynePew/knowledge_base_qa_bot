"""Committed dense-seed guard — the REAL artifact must stay re-baked (#311 AC5).

This is the #307 lesson applied to the Hybrid stack. The running app loads the
COMMITTED ``.kb/hybrid_dense/`` seed, not a freshly-built one. A seed that drifts
from the wiki corpus (a Section added/removed/renamed without a re-bake) would
break Hybrid fusion in production — the dense ids would no longer align 1:1 with
the BM25 ids — while every fresh-build unit test (which embeds the current wiki)
stays green. This guard reads the committed seed directly so a stale seed fails
CI and forces a re-bake.

Hermetic: the dense seed is loaded read-only via the explicit committed dir
(bypassing the autouse tmp redirect); ``fake_embeddings`` keeps it offline —
inspecting docstore metadata needs no real vectors, and no search is run. The
BM25 id set is read straight from the committed ``.kb/index.json`` (no
markdown_kb state is mutated).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hybrid_kb.app.dense_index as dense_index

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_DENSE_DIR = _REPO_ROOT / ".kb" / "hybrid_dense"
_COMMITTED_BM25_INDEX = _REPO_ROOT / ".kb" / "index.json"

_seed_present = (_COMMITTED_DENSE_DIR / "index.faiss").exists()
_skip_reason = "committed dense seed not present in this checkout"


def _bm25_ids() -> set[str]:
    """The Section id set of the committed BM25 wiki index (.kb/index.json)."""
    payload = json.loads(_COMMITTED_BM25_INDEX.read_text(encoding="utf-8"))
    return {s["id"] for s in payload["sections"]}


@pytest.mark.skipif(not _seed_present, reason=_skip_reason)
def test_committed_seed_metadata(fake_embeddings):
    """The committed seed carries the expected dense-over-wiki metadata."""
    metadata = json.loads(
        (_COMMITTED_DENSE_DIR / dense_index.METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["embedding_model"] == dense_index.EMBEDDING_MODEL, (
        "committed seed must be embedded with text-embedding-3-small (model parity)"
    )
    assert metadata["granularity"] == "section", (
        "committed seed must be Section-granular (not char-Chunk)"
    )
    assert metadata["sections_indexed"] == len(_bm25_ids()), (
        "committed seed sections_indexed must equal the BM25 wiki Section count "
        "— a mismatch means the seed needs re-baking (#311 AC5)"
    )


@pytest.mark.skipif(not _seed_present, reason=_skip_reason)
def test_committed_seed_ids_align_1to1_with_bm25_wiki_index(fake_embeddings):
    """The committed dense seed's ids align 1:1 with the committed BM25 wiki ids.

    This is the ADR-0018 same-corpus invariant enforced on the shipped artifact:
    fusion is only meaningful when a dense entry and its BM25 counterpart share
    an id. A stale seed (built before a wiki change) breaks this and must fail.
    """
    dense_index.load_dense_index(_COMMITTED_DENSE_DIR)
    docs = list(dense_index.vectorstore.docstore._dict.values())
    assert docs, "committed dense seed must not be empty"
    dense_ids = {d.metadata["id"] for d in docs}

    bm25_ids = _bm25_ids()
    assert bm25_ids, "committed BM25 wiki index must not be empty"

    only_dense = sorted(dense_ids - bm25_ids)
    only_bm25 = sorted(bm25_ids - dense_ids)
    assert dense_ids == bm25_ids, (
        "committed dense seed is stale — its ids do not align 1:1 with the BM25 "
        f"wiki index (re-bake needed, see #311 AC5). only_in_dense={only_dense[:5]} "
        f"only_in_bm25={only_bm25[:5]}"
    )


@pytest.mark.skipif(not _seed_present, reason=_skip_reason)
def test_committed_seed_has_no_duplicate_ids(fake_embeddings):
    """Every committed dense entry has a distinct id (true 1:1, no double-embed)."""
    dense_index.load_dense_index(_COMMITTED_DENSE_DIR)
    ids = [d.metadata["id"] for d in dense_index.vectorstore.docstore._dict.values()]
    assert len(ids) == len(set(ids)), "committed dense seed must not double-embed an id"
