"""Index-time ``lang`` tagging correctness for the FAISS (RAG/Stack B) build.

Issue #285: building the FAISS index must tag every Chunk's document metadata
with a ``lang`` value, derived from the Chunk's CONTENT (never filename/folder)
via ``detect_lang``. This slice only ADDS the tag — retrieval results and answers
are unchanged (nothing filters on ``lang`` yet).

Behavioural test (CODING_STANDARD §6.2 — assert external state, mock only the
embedding leaf via ``fake_embeddings``): build over a tiny mixed-language fixture
corpus whose filenames are language-misleading, then assert every retrieved
Chunk carries the correct content-derived ``lang``. The persisted Document
metadata is checked too, since that is what a future metadata filter queries.
"""

from __future__ import annotations

import vector_rag.app.indexer as indexer

# Chinese body under an *English* filename — proves content (not filename) drives
# the tag.
_ZH_DOC = """# 退款政策

## 退款時間

退款會在七個工作天內處理完成，款項退回原付款方式，請耐心等候銀行作業時間。

## 取消訂單

訂單在出貨前都可以免費取消，出貨之後請依照退貨流程辦理退款手續。
"""

# English body under a *Chinese* filename — the mirror case.
_EN_DOC = """# Refund Policy

## Refund Timeline

Refunds are processed within seven business days to the original payment method,
so please allow time for your bank to post the credit to your account.

## Cancellation

Orders can be cancelled free of charge any time before they ship out.
"""


def _write_corpus(docs_dir):
    docs_dir.mkdir(parents=True, exist_ok=True)
    # English filename, Chinese content.
    (docs_dir / "refund_policy.md").write_text(_ZH_DOC, encoding="utf-8")
    # Chinese filename, English content.
    (docs_dir / "退款.md").write_text(_EN_DOC, encoding="utf-8")


def test_faiss_chunks_tagged_with_content_language(tmp_path, fake_embeddings):
    """Every FAISS Chunk carries a content-derived ``lang`` on its metadata."""
    docs_dir = tmp_path / "docs"
    _write_corpus(docs_dir)

    indexer.build_index(docs_dir)

    # The persisted LangChain Document metadata is what a future metadata filter
    # queries — assert the tag lives there, on every chunk, with a valid value.
    docstore = indexer.vectorstore.docstore._dict
    assert docstore, "Expected chunks in the FAISS docstore after build"
    for doc in docstore.values():
        assert "lang" in doc.metadata, (
            f"Chunk from {doc.metadata.get('file')!r} missing 'lang' metadata"
        )
        assert doc.metadata["lang"] in ("zh", "en")

    # Group by source file and assert content (not filename) drove each tag.
    # vector_rag's build_index calls parse_markdown without a source_id, so
    # Section.file is the full filename (with .md), unlike the BM25 bare-slug.
    zh_docs = [d for d in docstore.values() if d.metadata["file"] == "refund_policy.md"]
    en_docs = [d for d in docstore.values() if d.metadata["file"] == "退款.md"]
    assert zh_docs, "Expected chunks from the Chinese-content file"
    assert en_docs, "Expected chunks from the English-content file"
    assert all(d.metadata["lang"] == "zh" for d in zh_docs), (
        "Chinese content must tag zh regardless of the English filename"
    )
    assert all(d.metadata["lang"] == "en" for d in en_docs), (
        "English content must tag en regardless of the Chinese filename"
    )


def test_chunk_exposes_lang_after_search(tmp_path, fake_embeddings):
    """A retrieved domain Chunk surfaces its ``lang`` (no LangChain Document leak)."""
    docs_dir = tmp_path / "docs"
    _write_corpus(docs_dir)

    indexer.build_index(docs_dir)

    chunks = indexer.search("退款", k=5)
    assert chunks, "Expected search to return chunks"
    for chunk in chunks:
        assert chunk.lang in ("zh", "en")


def test_faiss_lang_tag_survives_persistence_roundtrip(tmp_path, fake_embeddings):
    """The lang tag persists through save_local + load_local."""
    docs_dir = tmp_path / "docs"
    _write_corpus(docs_dir)

    indexer.build_index(docs_dir)
    before = {
        k: d.metadata.get("lang") for k, d in indexer.vectorstore.docstore._dict.items()
    }

    indexer.vectorstore = None
    indexer.load_vector_index()
    after = {
        k: d.metadata.get("lang") for k, d in indexer.vectorstore.docstore._dict.items()
    }

    assert before == after, "lang tag must round-trip through the persisted index"
    assert all(v in ("zh", "en") for v in after.values())
