"""#290 — RAG language-filtered retrieval (the essential cross-language fix), PRD #284.

The RAG/FAISS retrieval path must restrict a Chinese query to ``zh``-tagged Chunks
and an English query to ``en``-tagged Chunks, keyed on the QUERY language. The probe
in PRD #284 showed cross-language retrieval lives ONLY on the RAG stack (the
multilingual embeddings cross-retrieve), so this filter — not the BM25 belt-and-
suspenders one (#287) — is the load-bearing change.

Mirrors #287's BM25 filter: the query-language predicate is the consolidated
``markdown_kb.app.indexer.detect_lang`` helper (#285) — the same one that tags each
Chunk's ``metadata['lang']`` at index time — so query-time routing and index-time
tagging can never drift. No third predicate is forked.

CRITICAL fail-closed behaviour (PRD #284 user story 3 / 14): when the query's
language has NO covering Chunk, the existing pre-LLM gate must return the existing
``CANNOT_CONFIRM_PHRASE`` sentinel — NEVER a wrong-language answer. The distance gate
is preserved unchanged; the filter narrows the candidate set BEFORE the gate.

Behavioural, not implementation-detail (test README §"Why integration-first"): the
fixture builds a REAL mixed-language FAISS index with the deterministic offline
embeddings and asserts on the retrieved Chunks' languages — no deep-module mock.
"""

from __future__ import annotations

import pytest

import vector_rag.app.indexer as vr_indexer
import vector_rag.app.retrieval as retrieval

# A Source carrying parallel English and Chinese Sections on the SAME topic
# (refunds). Because the topic is shared, the multilingual embeddings cross-retrieve
# (the PRD probe: a zh refund query pulls the en refund chunk into the top-k) — so
# this fixture is the honest test of the language filter on the RAG stack: the zh and
# en chunks are about the same thing, differing only in language.
_MIXED_MD = """\
# Policies

## Refund Policy

Approved refunds are processed within 5 to 7 business days to the original payment method.

## Shipping Policy

Standard shipping takes 3 to 5 business days for all domestic orders.

## 退款政策

核准的退款會在我們收到退回商品後的 5 至 7 個工作天內處理，款項將退回原付款方式。

## 運送政策

國內訂單的標準運送時間為 3 至 5 個工作天。
"""

# An English-ONLY corpus (no Chinese coverage at all). A Chinese query against this
# index has NO covering zh Chunk in ANY topic — the filter strips every en Chunk and
# the existing pre-LLM gate must return Cannot Confirm rather than a wrong-language
# answer (PRD #284 story 3 / 14: out-of-coverage = no Source in the query LANGUAGE).
_EN_ONLY_MD = """\
# Store Policies

## Gift Card Balance

You can check your gift card balance online at any time using the card number printed on the back.

## Refund Policy

Approved refunds are processed within 5 to 7 business days to the original payment method.
"""


@pytest.fixture()
def mixed_index(fake_embeddings, tmp_path):
    """Build a real FAISS index over the mixed-language fixture using fake embeddings.

    Writes the bilingual Source to a tmp docs dir and runs the REAL ``build_index``
    (which tags each Chunk's ``metadata['lang']`` via ``detect_lang``, #285). Relies
    on the autouse path redirect so the persisted index lands in tmp.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policies.md").write_text(_MIXED_MD, encoding="utf-8")
    vr_indexer.build_index(docs_dir)
    yield
    vr_indexer.vectorstore = None


@pytest.fixture()
def en_only_index(fake_embeddings, tmp_path):
    """Build a real FAISS index over an ENGLISH-ONLY corpus (no zh coverage).

    A Chinese query against this index has no covering zh Chunk, so the language
    filter strips everything and the pre-LLM gate fails closed to Cannot Confirm.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "store.md").write_text(_EN_ONLY_MD, encoding="utf-8")
    vr_indexer.build_index(docs_dir)
    yield
    vr_indexer.vectorstore = None


def _langs_of(chunks: list) -> set[str]:
    """Return the set of ``lang`` tags carried by the retrieved Chunks."""
    return {chunk.lang for chunk in chunks}


# ---------------------------------------------------------------------------
# Fixture sanity — the index really is mixed-language
# ---------------------------------------------------------------------------


def test_fixture_index_is_genuinely_mixed_language(mixed_index):
    """Sanity: the index carries both zh and en Chunks (guards a trivial green)."""
    en_only = vr_indexer.search_with_distance("refund policy", k=10)
    zh_only = vr_indexer.search_with_distance("退款政策", k=10)
    assert _langs_of([c for c, _ in en_only]) == {"en"}
    assert _langs_of([c for c, _ in zh_only]) == {"zh"}


# ---------------------------------------------------------------------------
# AC1 — language-restricted retrieval
# ---------------------------------------------------------------------------


def test_chinese_query_retrieves_only_zh_chunks(mixed_index):
    """A Chinese RAG query returns ONLY zh-tagged Chunks (AC1)."""
    results = vr_indexer.search_with_distance("退款需要多久", k=5)
    assert results, "Chinese query should retrieve at least one Chunk"
    assert _langs_of([c for c, _ in results]) == {"zh"}, (
        "Chinese query must return only zh Chunks, got "
        f"{[(c.source, c.lang) for c, _ in results]}"
    )


def test_english_query_retrieves_only_en_chunks(mixed_index):
    """An English RAG query returns ONLY en-tagged Chunks (AC1)."""
    results = vr_indexer.search_with_distance("how long do refunds take", k=5)
    assert results, "English query should retrieve at least one Chunk"
    assert _langs_of([c for c, _ in results]) == {"en"}, (
        "English query must return only en Chunks, got "
        f"{[(c.source, c.lang) for c, _ in results]}"
    )


def test_search_wrapper_also_filters_by_query_language(mixed_index):
    """The plain ``search`` wrapper (no distance) inherits the language filter."""
    zh_chunks = vr_indexer.search("退款需要多久", k=5)
    assert zh_chunks
    assert _langs_of(zh_chunks) == {"zh"}


# ---------------------------------------------------------------------------
# AC2 + AC3 — out-of-coverage query fails closed (Cannot Confirm), gate preserved
# ---------------------------------------------------------------------------


def test_zh_query_on_en_only_index_returns_cannot_confirm(en_only_index, monkeypatch):
    """A Chinese query against an EN-only index returns Cannot Confirm, never the
    English answer (AC2 — the deliberate behaviour change on RAG, PRD #284 story 3).

    No zh Chunk exists in ANY topic, so the zh filter strips every en Chunk; the
    existing pre-LLM gate then returns ``retrieval_empty`` Cannot Confirm.
    """
    # Permissive distance ceiling so the test isolates the LANGUAGE filter, not the
    # distance gate (the autouse fixture already sets 1000.0, but be explicit).
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "1000.0")
    gate = retrieval._retrieve_and_gate("禮品卡餘額怎麼查")
    assert gate["early_exit"] is True
    assert gate["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert gate["chunks"] == []
    # No en Chunk leaked into the sources of a Chinese query (fail-closed).
    assert gate["sources"] == []


def test_public_query_on_en_only_index_in_zh_is_cannot_confirm(
    en_only_index, monkeypatch
):
    """End-to-end parity: the public ``query`` returns Cannot Confirm for a zh
    query against an en-only index, with NO LLM call (the gate early-exits)."""
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "1000.0")
    result = retrieval.query("禮品卡餘額怎麼查")
    assert result["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert result["sources"] == []


def test_distance_gate_still_fires_within_language(mixed_index, monkeypatch):
    """AC3: the distance gate is preserved — a tight ceiling still refuses an
    in-language query whose closest Chunk is too far (no regression of the gate)."""
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    gate = retrieval._retrieve_and_gate("退款需要多久")
    assert gate["early_exit"] is True
    assert gate["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert gate["grounding_outcome"].reason == "below_threshold"


# ---------------------------------------------------------------------------
# Predicate-reuse guard — no forked language predicate (#285 / PRD #284 story 16)
# ---------------------------------------------------------------------------


def test_query_language_predicate_is_detect_lang():
    """The RAG query-language decision reuses ``markdown_kb`` ``detect_lang`` (#285),
    the same classifier the indexer uses to tag Chunks — so they cannot drift."""
    from markdown_kb.app.indexer import detect_lang

    assert detect_lang("退款需要多久") == "zh"
    assert detect_lang("how long do refunds take") == "en"
    # The vector_rag indexer imports the identical function (not a fork).
    assert vr_indexer.detect_lang is detect_lang


def test_no_regression_search_path_signature_is_unchanged():
    """``search_with_distance`` keeps its public arity — the language filter is an
    internal narrowing, not a contract change; it still returns (Chunk, float)
    pairs with no LangChain Document leak (CODING_STANDARD §2.4)."""
    import inspect

    sig = inspect.signature(vr_indexer.search_with_distance)
    assert list(sig.parameters) == ["query", "k"]


def test_retrieved_units_are_domain_chunks_not_documents(mixed_index):
    """No LangChain Document leaks past the indexer despite the new metadata filter."""
    results = vr_indexer.search_with_distance("退款需要多久", k=5)
    assert results
    for chunk, distance in results:
        assert isinstance(chunk, vr_indexer.Chunk)
        assert isinstance(distance, float)
