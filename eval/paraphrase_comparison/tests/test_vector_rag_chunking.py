"""vector_rag chunk-metadata tests (external behaviour only, CODING_STANDARD §0.2).

Asserts the heading-split-then-char-split contract and that a Chunk's ``source``
is a single docs Section id under the canonical slug convention. The embedding
layer is the only network leaf and is swapped for a deterministic fake
(fixture ``fake_vector_index``) so the real chunking path is exercised offline
(§6.3: mock the leaf, not the deep module).
"""

from __future__ import annotations

import vector_rag.app.indexer as vr_indexer
from eval.paraphrase_comparison.stacks import FIXTURES

CORPUS = FIXTURES["corpus"]


def test_chunk_source_is_a_single_docs_section_id():
    documents = vr_indexer._load_documents(CORPUS)
    assert documents, "corpus should produce chunk documents"
    for doc in documents:
        source = doc.metadata["source"]
        # Canonical slug convention: '<source-filename>#<heading-slug>'.
        assert source.count("#") == 1
        filename, slug = source.split("#")
        assert filename.endswith(".md")
        assert slug == slug.lower()
        assert " " not in slug


def test_multi_subfact_section_yields_multiple_chunks_same_source():
    documents = vr_indexer._load_documents(CORPUS)
    long_section = "returns_policy.md#refund-processing-time"
    matching = [d for d in documents if d.metadata["source"] == long_section]
    # The Refund Processing Time section body exceeds chunk_size=500, so the
    # recursive char splitter must produce more than one Chunk, each tagged with
    # that single Section id.
    assert len(matching) > 1
    assert all(d.metadata["source"] == long_section for d in matching)


def test_chunk_sources_cover_all_body_sections():
    documents = vr_indexer._load_documents(CORPUS)
    sources = {d.metadata["source"] for d in documents}
    assert "returns_policy.md#return-window" in sources
    assert "shipping_options.md#expedited-delivery" in sources
    assert "account_management.md#password-reset" in sources


def test_search_returns_domain_chunk_not_langchain_document(fake_vector_index):
    vr_indexer.build_index(CORPUS)
    results = vr_indexer.search("refund processing business days", k=3)
    assert results
    for chunk in results:
        # Domain type — no LangChain Document leaks past the module (§2.4).
        assert isinstance(chunk, vr_indexer.Chunk)
        assert chunk.source == chunk.id
        assert chunk.source.count("#") == 1
        assert isinstance(chunk.heading_path, list)


def test_search_returns_empty_when_index_not_built():
    vr_indexer.vectorstore = None
    assert vr_indexer.search("anything", k=3) == []
