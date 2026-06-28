"""Hybrid Retrieval (Stack C) dense-over-wiki index — hermetic behaviour (S1, #311).

External behaviour only (CODING_STANDARD §0.2 / §6.2): we assert that the dense
index builds at Section granularity from the BM25-aligned wiki Section list,
persists atomically to its own ``.kb/hybrid_dense/`` seed and reloads without
re-embedding, fails fast on a corrupt / incomplete seed (§4.1), and that the
dense retrieval call returns ``(Section, distance)`` ranked results filtered to
the query's language exactly like the BM25 path.

Runs offline via the ``fake_embeddings`` fixture (real FAISS save/load/search
path, deterministic embedding leaf). The S2 OR-gate, distance calibration, and
RRF fusion are out of scope here (#312).
"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

import hybrid_kb.app.dense_index as dense_index
from markdown_kb.app.indexer import Section

from .conftest import make_section

# A small bilingual synthetic corpus: two English Sections, two Chinese ones.
# Content drives the index-time language tag via ``_section_lang`` (no explicit
# lang pin) so the tests exercise the SAME content-derived classifier the BM25
# path uses.
_EN_REFUND = make_section(
    "refund-policy#refund-policy",
    "Refunds are processed within 7 business days of approval.",
    heading_path=["Refund Policy"],
)
_EN_SHIPPING = make_section(
    "standard-shipping#standard-shipping",
    "Standard shipping takes three to five business days to arrive.",
    heading_path=["Standard Shipping"],
)
_ZH_REFUND = make_section(
    "退款政策#退款政策",
    "退款會在核准後的七個工作天內處理完成。",
    heading_path=["退款政策"],
)
_ZH_SHIPPING = make_section(
    "標準配送#標準配送",
    "標準配送需要三到五個工作天才會送達。",
    heading_path=["標準配送"],
)
_SYNTHETIC = [_EN_REFUND, _EN_SHIPPING, _ZH_REFUND, _ZH_SHIPPING]


@pytest.fixture()
def indexed_synthetic(fake_embeddings):
    """Build the dense index over the 4-Section synthetic corpus (offline)."""
    n = dense_index.build_index(sections=_SYNTHETIC)
    assert n == len(_SYNTHETIC)
    yield
    dense_index.vectorstore = None


# ---------------------------------------------------------------------------
# AC2 — Section-granular build; ids equal the BM25-aligned Section ids
# ---------------------------------------------------------------------------
def test_build_is_section_granular_one_entry_per_section(indexed_synthetic):
    """Exactly one dense entry per input Section, keyed by the Section id."""
    docs = list(dense_index.vectorstore.docstore._dict.values())
    assert len(docs) == len(_SYNTHETIC), "one embedding per Section (no char-chunking)"
    ids = {d.metadata["id"] for d in docs}
    assert ids == {s.id for s in _SYNTHETIC}, "dense ids must equal the Section ids 1:1"


def test_build_covers_the_full_filtered_wiki_list(fake_embeddings):
    """Building the default corpus covers EVERY filtered wiki Section, 1:1.

    The dense build with no explicit ``sections`` derives the corpus from
    ``filtered_wiki_sections()`` — the same scan + status-live qa filter BM25
    uses. The resulting dense id set must equal that filtered list's id set with
    no drops and no duplicates, the ADR-0018 same-corpus invariant at build time.
    """
    expected = {s.id for s in dense_index.filtered_wiki_sections()}
    assert expected, "the real wiki corpus must yield a non-empty filtered list"

    n = dense_index.build_index()
    assert n == len(expected), "dense entry count must equal the filtered Section count"

    docs = list(dense_index.vectorstore.docstore._dict.values())
    dense_ids = {d.metadata["id"] for d in docs}
    assert dense_ids == expected, "dense ids must align 1:1 with the filtered wiki list"


def test_build_embeds_non_empty_text_for_every_section(fake_embeddings):
    """An empty-body (heading-only) Section still embeds a non-empty text.

    Rule-8 empty-body leaves exist in the BM25 corpus; to keep the dense id set a
    1:1 superset-free match, every Section must produce a non-degenerate embed
    text. ``_embed_text`` falls back to the heading-path breadcrumb.
    """
    leaf = make_section(
        "faq#empty-leaf", "", heading="Empty Leaf", heading_path=["FAQ", "Empty Leaf"]
    )
    text = dense_index._embed_text(leaf)
    assert text.strip(), "an empty-body Section must still embed a non-empty text"


# ---------------------------------------------------------------------------
# AC3 — atomic persist + fail-fast load
# ---------------------------------------------------------------------------
def test_build_persists_faiss_and_metadata(indexed_synthetic):
    """build_index writes the FAISS payload + metadata.json to .kb/hybrid_dense/."""
    index_dir = dense_index.DENSE_INDEX_DIR
    assert (index_dir / "index.faiss").exists(), "FAISS payload (index.faiss) persisted"
    assert (index_dir / "index.pkl").exists(), "FAISS payload (index.pkl) persisted"

    metadata_path = index_dir / dense_index.METADATA_FILENAME
    assert metadata_path.exists(), "metadata.json must be persisted"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sections_indexed"] == len(_SYNTHETIC)
    assert metadata["embedding_model"] == dense_index.EMBEDDING_MODEL
    assert metadata["granularity"] == "section"


def test_save_load_roundtrip_returns_same_neighbours(indexed_synthetic):
    """A reload reconstructs an index returning the same top neighbour ids."""
    before = [s.id for s in dense_index.search("refund", k=2)]
    assert before, "the indexed corpus must return results for a refund query"

    dense_index.vectorstore = None
    n = dense_index.load_dense_index()
    assert n == len(_SYNTHETIC)
    assert dense_index.vectorstore is not None, "load must repopulate the index"

    after = [s.id for s in dense_index.search("refund", k=2)]
    assert after == before, (
        f"reload must return the same neighbours; {before=} {after=}"
    )


def test_reload_does_not_reembed_corpus(
    indexed_synthetic, fake_embeddings, monkeypatch
):
    """load_dense_index reconstructs the index without calling embed_documents."""
    dense_index.vectorstore = None
    calls = {"docs": 0}
    original = fake_embeddings.embed_documents

    def _counting(texts):
        calls["docs"] += 1
        return original(texts)

    monkeypatch.setattr(fake_embeddings, "embed_documents", _counting)
    dense_index.load_dense_index()
    assert calls["docs"] == 0, "reload must NOT re-embed the corpus"


def test_load_missing_index_returns_zero(tmp_path):
    """load_dense_index on an absent index dir is a clean 0, not an error."""
    n = dense_index.load_dense_index(tmp_path / "nope" / "hybrid_dense")
    assert n == 0
    assert dense_index.vectorstore is None


def test_load_corrupt_metadata_fails_fast(indexed_synthetic):
    """A corrupt metadata.json raises on load (§4.1 fail-fast), not silent-empty."""
    metadata_path = dense_index.DENSE_INDEX_DIR / dense_index.METADATA_FILENAME
    metadata_path.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        dense_index.load_dense_index()


def test_load_missing_metadata_fails_fast(indexed_synthetic):
    """A present index dir with no metadata.json raises rather than serving empty."""
    metadata_path = dense_index.DENSE_INDEX_DIR / dense_index.METADATA_FILENAME
    metadata_path.unlink()
    with pytest.raises(RuntimeError):
        dense_index.load_dense_index()


def test_empty_corpus_clears_index_without_persisting(fake_embeddings):
    """An empty Section list clears the in-memory index and persists nothing."""
    n = dense_index.build_index(sections=[])
    assert n == 0
    assert dense_index.vectorstore is None
    assert not dense_index.DENSE_INDEX_DIR.exists(), "empty build must not write a seed"


# ---------------------------------------------------------------------------
# AC4 — dense retrieval returns (Section, distance), language-filtered
# ---------------------------------------------------------------------------
def test_search_with_distance_returns_section_and_float(indexed_synthetic):
    """search_with_distance returns ranked (Section, distance) pairs."""
    results = dense_index.search_with_distance("refund", k=2)
    assert results, "expected at least one dense hit for an English refund query"
    for section, distance in results:
        assert isinstance(section, Section), "must return domain Section, not Document"
        assert isinstance(distance, float), "distance must be a float (the dense score)"


def test_search_returns_no_langchain_document_leak(indexed_synthetic):
    """The no-distance wrapper returns Section objects only (§2.4 no leak)."""
    for section in dense_index.search("refund", k=4):
        assert isinstance(section, Section)
        assert not isinstance(section, Document)


def test_english_query_returns_only_english_sections(indexed_synthetic):
    """An English query is scored only against en-tagged Sections (BM25 parity)."""
    results = dense_index.search_with_distance("how long do refunds take", k=4)
    assert results, "expected English hits"
    langs = {s.metadata.get("lang") for s, _ in results}
    assert langs == {"en"}, f"English query must return only en Sections; got {langs}"
    ids = {s.id for s, _ in results}
    assert ids <= {_EN_REFUND.id, _EN_SHIPPING.id}, "no Chinese Section may leak in"


def test_chinese_query_returns_only_chinese_sections(indexed_synthetic):
    """A Chinese query is scored only against zh-tagged Sections (BM25 parity)."""
    results = dense_index.search_with_distance("退款需要多久", k=4)
    assert results, "expected Chinese hits"
    langs = {s.metadata.get("lang") for s, _ in results}
    assert langs == {"zh"}, f"Chinese query must return only zh Sections; got {langs}"
    ids = {s.id for s, _ in results}
    assert ids <= {_ZH_REFUND.id, _ZH_SHIPPING.id}, "no English Section may leak in"


def test_search_on_unbuilt_index_is_empty(fake_embeddings):
    """search before any build returns [] (no crash, no silent stale state)."""
    dense_index.vectorstore = None
    assert dense_index.search_with_distance("anything", k=3) == []
    assert dense_index.search("anything", k=3) == []
