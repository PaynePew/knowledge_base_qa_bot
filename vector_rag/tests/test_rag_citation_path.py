"""#307 — RAG-stack citations carry a resolvable ``docs/`` path (clickable UI).

The browser UI (gateway/static/index.html ``onSources``) makes a citation
clickable iff its source dict carries a ``path``; the gateway forwards ``path``
generically (routes.py: ``if "path" in s``). Wiki citations already do this
(#266). This slice gives RAG sources the same field so the UX is identical —
ZERO UI / gateway-forwarding changes.

The contract proven here (behavioural, CODING_STANDARD §6.2 / §6.5):
  AC1  a RAG source dict carries ``path`` == ``docs/<relpath-from-repo-root>``
       (forward slashes, ends ``.md``).
  AC2  that path is GENUINELY openable — it resolves through the SAME whitelist
       (``markdown_kb.app.read.read_file``) that the gateway ``GET /read/file``
       route delegates to, returning the real on-disk docs content.
  AC3  back-compat: a Chunk without ``file`` metadata (an older persisted FAISS
       index) emits NO ``path`` key — never ``path: None``.
  #120 the RAG-no-score / no-derived_from invariant is preserved alongside path.

Offline + hermetic: the embedding leaf is the ``fake_embeddings`` fixture; the
autouse path redirect (conftest) keeps the persisted FAISS index in tmp. The
docs/-path AC builds over the REAL docs/ corpus (not the 3-file hermetic
fixture) because the path is only genuinely resolvable when the indexed files
actually live under the real ``docs/`` whitelist root.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import vector_rag.app.indexer as indexer
import vector_rag.app.retrieval as retrieval
from markdown_kb.app import read as kb_read

from .conftest import REAL_DOCS

_EN_QUERY = "How long do refunds take?"


@pytest.fixture()
def indexed_real_corpus(fake_embeddings):
    """Build the FAISS index over the REAL docs/ corpus with offline fake embeddings.

    Uses REAL_DOCS (not the 3-file fixture) so the emitted ``docs/<relpath>``
    path points at a file that truly exists under the ``docs/`` whitelist root
    and can be opened by ``read_file`` (AC2). Relies on the autouse path redirect
    so the persisted index lands in tmp.
    """
    indexer.build_index(REAL_DOCS)
    yield
    indexer.vectorstore = None


def _sources_for(question: str) -> list[dict]:
    """Run the real pre-LLM retrieve+gate and return its RAG source dicts."""
    gate = retrieval._retrieve_and_gate(question)
    assert not gate["early_exit"], (
        "expected retrieval to pass the pre-LLM gates, got "
        f"{gate['grounding_outcome'].reason}"
    )
    assert gate["sources"], "expected at least one retrieved source"
    return gate["sources"]


# ---------------------------------------------------------------------------
# AC1 — RAG source carries a docs/-relative, forward-slash, .md path
# ---------------------------------------------------------------------------
def test_rag_source_carries_docs_relative_path(indexed_real_corpus):
    """Every retrieved RAG source dict carries path == docs/<relpath> (#307)."""
    for s in _sources_for(_EN_QUERY):
        assert "path" in s, f"RAG source must carry a clickable path: {s}"
        p = s["path"]
        assert p.startswith("docs/"), f"path must be repo-root-relative under docs/: {p!r}"
        assert "\\" not in p, f"path must use forward slashes (Windows-safe): {p!r}"
        assert p.endswith(".md"), f"path must point at a markdown Source: {p!r}"


# ---------------------------------------------------------------------------
# AC2 — the emitted path resolves via the SAME whitelist GET /read/file uses
# ---------------------------------------------------------------------------
def test_rag_path_resolves_through_read_whitelist(indexed_real_corpus):
    """The emitted path opens the real docs file through read_file (#307 AC2).

    ``markdown_kb.app.read.read_file`` IS the function the gateway ``GET
    /read/file`` route delegates to (gateway/app/routes.py imports it as
    ``_read_file``); resolving through it proves the path is genuinely openable,
    not merely well-formed.
    """
    path = _sources_for(_EN_QUERY)[0]["path"]
    content = kb_read.read_file(path)
    assert content.strip(), f"resolved file {path!r} must have real, non-empty content"
    on_disk = (indexer.DOCS_DIR.parent / path).read_text(encoding="utf-8")
    assert content == on_disk, "read_file content must equal the on-disk docs file"


# ---------------------------------------------------------------------------
# #120 invariant preserved — no score / derived_from leak alongside the path
# ---------------------------------------------------------------------------
def test_rag_sources_still_have_no_score_or_derived_from(indexed_real_corpus):
    """Adding path does NOT reintroduce score / derived_from (issue #120 spec)."""
    for s in _sources_for(_EN_QUERY):
        assert "score" not in s, f"RAG source must not carry 'score': {s}"
        assert "derived_from" not in s, f"RAG source must not carry 'derived_from': {s}"


# ---------------------------------------------------------------------------
# AC3 — back-compat: a chunk with no file metadata emits NO path key
# ---------------------------------------------------------------------------
def test_source_without_file_emits_no_path(monkeypatch):
    """A Chunk whose ``file`` is empty (old persisted index) omits ``path`` entirely."""
    chunk = indexer.Chunk(
        id="x.md#h", source="x.md#h", heading_path=["X"], content="body"
    )
    assert chunk.file == "", "Chunk.file must default to '' for back-compat"

    monkeypatch.setattr(indexer, "vectorstore", object())
    monkeypatch.setattr(indexer, "search_with_distance", lambda q, k=3: [(chunk, 0.5)])

    gate = retrieval._retrieve_and_gate("anything")
    assert not gate["early_exit"]
    assert gate["sources"], "expected a source"
    for s in gate["sources"]:
        assert "path" not in s, f"a chunk without file metadata must omit path: {s}"


def test_source_with_file_emits_that_path(monkeypatch):
    """A Chunk carrying ``file`` surfaces it verbatim as the source ``path``."""
    chunk = indexer.Chunk(
        id="x.md#h",
        source="x.md#h",
        heading_path=["X"],
        content="body",
        file="docs/refund_policy.md",
    )
    monkeypatch.setattr(indexer, "vectorstore", object())
    monkeypatch.setattr(indexer, "search_with_distance", lambda q, k=3: [(chunk, 0.5)])

    gate = retrieval._retrieve_and_gate("anything")
    assert gate["sources"][0]["path"] == "docs/refund_policy.md"


def test_source_with_non_whitelisted_file_emits_no_path(monkeypatch):
    """A Chunk whose ``file`` is not under a /read/file whitelist root omits path.

    Boundary hardening (#307): the emit gate is the whitelist ROOT prefix, not mere
    non-emptiness — so a corpus indexed outside ``docs/``/``raw/``/``wiki/`` (a
    repo-relative fixture path, an out-of-repo tmp corpus whose file degraded to a
    bare basename, or an absolute path) can never surface a clickable-but-404
    citation. Keeps "clickable iff openable" true at the source.
    """
    for bad_file in (
        "markdown_kb/tests/fixtures/docs/x.md",  # repo-relative, not a whitelist root
        "x.md",                                   # bare basename (out-of-repo fallback)
        "/abs/x.md",                              # absolute
    ):
        chunk = indexer.Chunk(
            id="x.md#h", source="x.md#h", heading_path=["X"], content="body",
            file=bad_file,
        )
        monkeypatch.setattr(indexer, "vectorstore", object())
        monkeypatch.setattr(
            indexer, "search_with_distance", lambda q, k=3, _c=chunk: [(_c, 0.5)]
        )
        gate = retrieval._retrieve_and_gate("anything")
        for s in gate["sources"]:
            assert "path" not in s, f"non-whitelisted file {bad_file!r} must omit path: {s}"


# ---------------------------------------------------------------------------
# _chunk_from_document plumbs the file metadata (with back-compat default)
# ---------------------------------------------------------------------------
def test_chunk_from_document_reads_file_metadata():
    """_chunk_from_document maps Document metadata['file'] onto Chunk.file."""
    doc = Document(
        page_content="body", metadata={"source": "x.md#h", "file": "docs/x.md"}
    )
    assert indexer._chunk_from_document(doc).file == "docs/x.md"


def test_chunk_from_document_defaults_file_when_absent():
    """A Document without 'file' metadata reconstructs a Chunk with file=''."""
    doc = Document(page_content="body", metadata={"source": "x.md#h"})
    assert indexer._chunk_from_document(doc).file == ""
