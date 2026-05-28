"""FAISS persistence tests — save/load roundtrip, fail-fast, reload (issue #103).

External behaviour only (CODING_STANDARD §0.2 / §6.2): we assert that a built
index persists to ``.kb/faiss_index/`` (FAISS payload + metadata.json), that a
fresh load reconstructs an index that returns the same neighbours WITHOUT
re-embedding, and that a corrupt / incomplete persisted index fails fast on
load instead of silently serving empty (§4.1).

Runs offline via the ``fake_embeddings`` fixture (real FAISS save/load path,
deterministic embedding leaf).
"""

from __future__ import annotations

import json

import pytest

import vector_rag.app.indexer as indexer

from .conftest import REAL_DOCS

QUERY = "How long do refunds take?"


def test_build_index_persists_faiss_and_metadata(indexed_corpus):
    """build_index writes the FAISS payload + metadata.json to .kb/faiss_index/."""
    index_dir = indexer.FAISS_INDEX_DIR
    assert index_dir.exists(), "FAISS index dir must exist after build_index"
    assert (index_dir / "index.faiss").exists(), (
        "FAISS payload (index.faiss) must be persisted"
    )
    assert (index_dir / "index.pkl").exists(), (
        "FAISS payload (index.pkl) must be persisted"
    )

    metadata_path = index_dir / indexer.METADATA_FILENAME
    assert metadata_path.exists(), "metadata.json must be persisted"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["files_indexed"] == indexer.files_indexed
    assert metadata["chunks_indexed"] == indexer.chunks_indexed
    assert metadata["embedding_model"] == indexer.EMBEDDING_MODEL


def test_save_load_roundtrip_returns_same_neighbours(indexed_corpus):
    """A reload reconstructs an index returning the same top neighbours.

    The reload runs through real FAISS.load_local — no re-embedding of the
    corpus, only the query is embedded at search time (the persistence point of
    the PROMPT.md contract).
    """
    before = [c.source for c in indexer.search(QUERY, k=3)]
    assert before, "the indexed corpus must return results for a refund query"

    # Drop the in-memory index entirely, then reload purely from disk.
    indexer.vectorstore = None
    files, chunks = indexer.load_vector_index()

    assert files == indexer.files_indexed
    assert chunks == indexer.chunks_indexed
    assert indexer.vectorstore is not None, (
        "load_vector_index must repopulate the index"
    )

    after = [c.source for c in indexer.search(QUERY, k=3)]
    assert after == before, (
        f"reloaded index must return the same neighbours; before={before} after={after}"
    )


def test_reload_does_not_reembed(indexed_corpus, monkeypatch):
    """load_vector_index reconstructs the index without calling embed_documents.

    Only the corpus embedding (embed_documents) is the expensive re-embed we
    must avoid on restart; embed_query is still allowed (search embeds the query).
    """
    indexer.vectorstore = None

    embeddings = indexer.get_embeddings()
    calls = {"docs": 0}
    original_embed_documents = embeddings.embed_documents

    def _counting_embed_documents(texts):
        calls["docs"] += 1
        return original_embed_documents(texts)

    monkeypatch.setattr(embeddings, "embed_documents", _counting_embed_documents)

    indexer.load_vector_index()

    assert calls["docs"] == 0, (
        "reload must NOT re-embed the corpus (embed_documents called)"
    )


def test_load_missing_index_returns_zero(tmp_path):
    """load_vector_index on an absent index dir is a clean (0, 0), not an error."""
    files, chunks = indexer.load_vector_index(tmp_path / "nope" / "faiss_index")
    assert (files, chunks) == (0, 0)
    assert indexer.vectorstore is None


def test_load_corrupt_metadata_fails_fast(indexed_corpus):
    """A corrupt metadata.json raises on load (§4.1 fail-fast), not silent-empty."""
    metadata_path = indexer.FAISS_INDEX_DIR / indexer.METADATA_FILENAME
    metadata_path.write_text("{ this is not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        indexer.load_vector_index()


def test_load_index_missing_metadata_fails_fast(indexed_corpus):
    """A present index dir with no metadata.json raises rather than serving empty."""
    metadata_path = indexer.FAISS_INDEX_DIR / indexer.METADATA_FILENAME
    metadata_path.unlink()

    with pytest.raises(RuntimeError):
        indexer.load_vector_index()


def test_save_overwrites_previous_index(indexed_corpus):
    """A second build_index atomically replaces the persisted index dir."""
    index_dir = indexer.FAISS_INDEX_DIR
    first_metadata = json.loads(
        (index_dir / indexer.METADATA_FILENAME).read_text(encoding="utf-8")
    )

    # Re-build against the same corpus; the dir must still be a valid, loadable index.
    indexer.build_index(REAL_DOCS)
    indexer.vectorstore = None
    files, chunks = indexer.load_vector_index()
    assert files == first_metadata["files_indexed"]
    assert chunks == first_metadata["chunks_indexed"]
