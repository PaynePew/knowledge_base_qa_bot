"""Hybrid Retrieval (Stack C) cross-encoder reranker — ADR-0019 / #310.

External behaviour only (CODING_STANDARD §0.2 / §6.2). The reranker is the
precision step ADR-0018 deferred: it re-scores the RRF-fused candidate pool with
a cross-encoder (query + Section scored jointly) and reorders it before the final
top_k cut. Default-OFF (``KB_HYBRID_RERANK``), eval-only — never loaded on the
512m VPS tenant (ADR-0019).

The only heavy seam is ``get_cross_encoder`` (it loads ``sentence-transformers``
+ the ~2.3 GB ``bge-reranker-v2-m3`` model). Hermetic tests monkeypatch it to a
deterministic token-overlap fake so the reorder / flag / truncation logic runs
with NO torch and NO model download; one ``@pytest.mark.live`` smoke exercises
the real multilingual model (EN + ZH) and is skipped unless ``-m live`` (and the
optional ``rerank`` dependency group) is present.
"""

from __future__ import annotations

import sys

import pytest

import hybrid_kb.app.rerank as rerank
from markdown_kb.app.indexer import Section

from .conftest import make_section


class _TokenOverlapEncoder:
    """Deterministic offline stand-in for a ``CrossEncoder``.

    Scores each ``(query, passage)`` pair by the count of shared lowercased word
    tokens — higher overlap = higher relevance — so a reorder test reads like a
    real reranker without any model. ``predict`` mirrors sentence-transformers'
    signature: a list of ``[query, passage]`` pairs → a list of float scores.
    Records the pairs it was handed so a test can assert the call shape.
    """

    def __init__(self) -> None:
        self.seen_pairs: list | None = None

    def predict(self, pairs):
        self.seen_pairs = [list(p) for p in pairs]
        scores = []
        for query, passage in pairs:
            q = set(query.lower().split())
            p = set(passage.lower().split())
            scores.append(float(len(q & p)))
        return scores


@pytest.fixture()
def fake_encoder(monkeypatch):
    """Swap the heavy ``get_cross_encoder`` leaf for the deterministic fake.

    Mirrors the dense suite's ``fake_embeddings`` pattern: the network/model leaf
    is faked, the rerank reorder/truncate logic runs unchanged and offline.
    """
    enc = _TokenOverlapEncoder()
    monkeypatch.setattr(rerank, "get_cross_encoder", lambda: enc)
    return enc


def _sec(section_id: str, content: str) -> Section:
    return make_section(section_id, content=content)


# ===========================================================================
# is_enabled — env flag, read THROUGH the module at call time (monkeypatchable)
# ===========================================================================
def test_is_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv(rerank.RERANK_ENABLED_ENV, raising=False)
    assert rerank.is_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "yes", "Yes"])
def test_is_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv(rerank.RERANK_ENABLED_ENV, value)
    assert rerank.is_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "maybe"])
def test_is_enabled_falsy(monkeypatch, value):
    monkeypatch.setenv(rerank.RERANK_ENABLED_ENV, value)
    assert rerank.is_enabled() is False


# ===========================================================================
# rerank() — reorders by cross-encoder score, truncates, preserves identity
# ===========================================================================
def test_rerank_reorders_by_cross_encoder_score(fake_encoder):
    """The candidate with the most query overlap is promoted to the front.

    The input order deliberately buries the relevant Section last, so a pass
    proves rerank reordered the pool (not just passed it through).
    """
    query = "how long does a refund take"
    candidates = [
        _sec("shipping", "shipping delivery business days within the country"),
        _sec("privacy", "privacy personal account data contact details"),
        _sec("refund", "a refund take how long does it usually about one week"),
    ]
    ranked = rerank.rerank(query, candidates, top_n=3)
    assert ranked[0].id == "refund"


def test_rerank_truncates_to_top_n(fake_encoder):
    """A deep pool collapses to the final top_n (the precision-step cut)."""
    candidates = [_sec(f"s{i}", f"body {i} refund") for i in range(10)]
    ranked = rerank.rerank("refund", candidates, top_n=3)
    assert len(ranked) == 3


def test_rerank_preserves_section_identity_only_reorders(fake_encoder):
    """Output Sections are the SAME objects (ids/content), only the order changes.

    Rerank is pure reordering — it must never mint, mutate, or drop Section
    content (the 1:1 id alignment + CitableContent invariants, ADR-0018/0019).
    """
    candidates = [_sec("a", "alpha refund refund"), _sec("b", "beta")]
    ranked = rerank.rerank("refund", candidates, top_n=2)
    assert {s.id for s in ranked} == {"a", "b"}
    assert all(isinstance(s, Section) for s in ranked)
    assert set(map(id, ranked)) == set(map(id, candidates)), "same objects, reordered"


def test_rerank_passes_query_passage_pairs_to_encoder(fake_encoder):
    """The encoder is scored on (query, Section-passage) pairs — the cross-encoder contract."""
    candidates = [_sec("a", "alpha body"), _sec("b", "beta body")]
    rerank.rerank("my query", candidates, top_n=2)
    assert fake_encoder.seen_pairs is not None
    assert all(pair[0] == "my query" for pair in fake_encoder.seen_pairs)
    passages = [pair[1] for pair in fake_encoder.seen_pairs]
    assert any("alpha body" in p for p in passages), (
        "passage must carry Section content"
    )


def test_rerank_empty_candidates_returns_empty(fake_encoder):
    assert rerank.rerank("anything", [], top_n=3) == []


# ===========================================================================
# get_cross_encoder — the optional heavy dep fails CLEARLY when absent
# ===========================================================================
def test_get_cross_encoder_missing_dep_raises_actionable_error(monkeypatch):
    """Absent ``sentence-transformers`` → a RuntimeError naming the fix, not a raw ImportError.

    Forcing ``sys.modules['sentence_transformers'] = None`` makes the lazy import
    raise ImportError regardless of whether the optional dep is installed, so this
    is deterministic in CI (torch-free) AND on a dev box that synced the rerank
    group — and it never loads the 2.3 GB model.
    """
    monkeypatch.setattr(rerank, "_cross_encoder", None, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match=r"rerank"):
        rerank.get_cross_encoder()


# ===========================================================================
# Live smoke — the REAL multilingual cross-encoder (EN + ZH); opt-in only
# ===========================================================================
@pytest.mark.live
def test_rerank_real_model_promotes_relevant_passage_en_and_zh():
    """The real ``bge-reranker-v2-m3`` ranks a relevant passage above an irrelevant one.

    Exercises the actual model load + scoring bilingually (the corpus is EN+ZH),
    the one live smoke a new model-backed surface earns (CODING_STANDARD §6.4).
    Opt-in: skipped unless ``-m live`` AND the optional ``rerank`` dep is synced.
    """
    pytest.importorskip("sentence_transformers")
    rerank._cross_encoder = None
    try:
        refund_en = make_section(
            "refund-en",
            content="Refunds are processed within seven business days after approval.",
        )
        shipping_en = make_section(
            "shipping-en",
            content="Standard shipping delivery takes three to five business days.",
        )
        ranked_en = rerank.rerank(
            "how long does a refund take", [shipping_en, refund_en], top_n=2
        )
        assert ranked_en[0].id == "refund-en"

        refund_zh = make_section(
            "refund-zh", content="退款會在核准後七個工作天內處理完成。"
        )
        shipping_zh = make_section(
            "shipping-zh", content="標準運送通常需要三到五個工作天送達。"
        )
        ranked_zh = rerank.rerank("退款要多久才會到", [shipping_zh, refund_zh], top_n=2)
        assert ranked_zh[0].id == "refund-zh"
    finally:
        rerank._cross_encoder = None
