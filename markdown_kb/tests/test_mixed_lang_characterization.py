"""#582 — mixed-language / code-switched query detection: characterization
fixtures (tests-only, no runtime change).

The pre-LLM Cannot-Confirm gate (``retrieval._retrieve_and_gate``) decides two
things about a query using TWO INDEPENDENTLY MAINTAINED classifiers:

  1. Which threshold to apply — ``retrieval._is_cjk_query`` (#261): True when
     the query contains ANY CJK character, however rare.
  2. Which corpus slice to search — ``indexer.detect_lang`` via
     ``indexer.search`` (#285/#287): the DOMINANT script by CJK-letter ratio
     (>= 0.20 -> "zh", else "en").

For a code-switched query that is CJK-dominant (a stray English word inside a
mostly-Chinese sentence), both classifiers agree: "zh". For a query that is
Latin-dominant but quotes a single CJK token (an English question naming a
Chinese product), they DISAGREE: ``_is_cjk_query`` fires on the lone CJK
character and routes the gate to the zh threshold (4.0 by default), while
``detect_lang`` stays "en" and restricts BM25 scoring to the en-tagged corpus
slice (BM25-scale calibrated for the en threshold, 0.5). The zh threshold
applied to an en-scale score refuses queries that would otherwise clear the
gate — a false Cannot Confirm on a genuinely in-scope question.

These tests PIN the current behaviour (including the mismatch) as evidence for
the follow-up routing-policy slice (dominant-script / dual-gate / union
corpus) named in the issue's scope decision. They intentionally do not assert
which behaviour is "correct" — that policy call has not been made yet. No
production code changes in this slice.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import app.indexer as idx
import app.retrieval as ret
from app.indexer import parse_markdown

# ---------------------------------------------------------------------------
# Fixture corpora
# ---------------------------------------------------------------------------

# Parallel English/Chinese refund policy — used for the CJK-dominant
# code-switch contrast case (both classifiers agree "zh").
_POLICIES_MD = """\
# Policies

## Refund Policy

Approved refunds are processed within 5 to 7 business days to the original
payment method. Store credit refunds are issued immediately after approval.

## 退款政策

核准的退款會在我們收到退回商品後的 5 至 7 個工作天內處理，款項將退回原付款方式，
經核准的商店禮金退款會立即發放。
"""

# Parallel English/Chinese gift-card policy — used for the Latin-dominant
# code-switch case that quotes a single CJK product name (classifiers
# disagree: _is_cjk_query -> True, detect_lang -> "en").
_PRODUCTS_MD = """\
# Products

## Gift Card Policy

Gift cards can be redeemed online or in store within one year of purchase.
Balances never expire and can be combined with a promotional discount code
at checkout.

## 禮品卡政策

禮品卡可以在一年內於網路或實體店兌換，餘額永不過期，並可與促銷折扣碼一併使用。
"""


def _index_markdown(markdown_text: str, source_id: str) -> list:
    """Build a BM25 index over ``markdown_text`` and return the Sections."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8", delete=False) as fh:
        fh.write(markdown_text)
        tmp = Path(fh.name)
    try:
        secs = parse_markdown(tmp, source_id=source_id)
    finally:
        tmp.unlink(missing_ok=True)
    with idx._index_lock:
        idx.sections = secs
        idx.rebuild_stats()
    return secs


def _teardown_index() -> None:
    with idx._index_lock:
        idx.sections = []
        idx.rebuild_stats()


def _langs_of(hits: list) -> set[str]:
    """Return the set of ``lang`` tags carried by the retrieved Sections."""
    return {sec.metadata.get(idx.LANG_METADATA_KEY) for sec, _score in hits}


# ---------------------------------------------------------------------------
# 1. Predicate disagreement — the raw signal (pure, no corpus needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_is_cjk_query, expected_detect_lang",
    [
        # CJK-dominant code-switch: a stray English word inside a mostly-
        # Chinese sentence. Both classifiers agree "zh" -> consistent routing.
        ("退款 take 幾天?", True, "zh"),
        ("我要 refund 這個訂單，謝謝", True, "zh"),
        # Latin-dominant code-switch quoting a single CJK token (a product
        # name, a stray character): _is_cjk_query fires on the lone CJK
        # character present; detect_lang's ratio gate (>= 0.20) stays "en".
        # DISAGREEMENT — this is the condition issue #582 flags as untested.
        ("Can I redeem the 熊貓 gift card online?", True, "en"),
        ("Does the return policy cover the 拿鐵 mug bundle?", True, "en"),
        ("Is the 熊 plush toy available in gift wrap?", True, "en"),
        # Pure English / pure Chinese controls, included so the table reads
        # as a complete before/after picture.
        ("How long does a refund take?", False, "en"),
        ("退款需要多久", True, "zh"),
    ],
)
def test_predicate_disagreement_on_code_switched_queries(
    query, expected_is_cjk_query, expected_detect_lang
):
    assert ret._is_cjk_query(query) is expected_is_cjk_query
    assert idx.detect_lang(query) == expected_detect_lang


# ---------------------------------------------------------------------------
# 2. Gate consequence — corpus slice vs threshold routing can diverge
#
# Thresholds are monkeypatched to extreme, unambiguous values (0.0 always
# clears, 9999.0 always refuses) so the assertions isolate WHICH threshold
# got applied, independent of real BM25 score calibration (CODING_STANDARD
# §6.2 — no absolute-score assertions). Mirrors test_zh_score_threshold.py.
# ---------------------------------------------------------------------------


def test_cjk_dominant_code_switch_routes_consistently(monkeypatch):
    """Contrast case: a CJK-dominant code-switched query does NOT trigger the
    #582 mismatch. Both classifiers say "zh", so the zh threshold is applied
    to a zh-filtered corpus slice — routing and corpus slice agree.
    """
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 9999.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 0.0)
    _index_markdown(_POLICIES_MD, source_id="policies")
    try:
        query = "退款 take 幾天?"
        hits = idx.search(query, k=5)
        assert _langs_of(hits) == {"zh"}, (
            "corpus slice should be the zh-tagged Sections (detect_lang), "
            f"got {[(sec.id, sec.metadata.get(idx.LANG_METADATA_KEY)) for sec, _ in hits]}"
        )
        gate = ret._retrieve_and_gate(query)
        # zh threshold (0.0, always clears) applied to the zh-filtered
        # corpus -> the gate passes.
        assert gate["early_exit"] is False
    finally:
        _teardown_index()


def test_latin_dominant_code_switch_applies_zh_threshold_to_en_corpus_slice(monkeypatch):
    """The #582 mismatch, pinned end to end via ``_retrieve_and_gate``.

    ``indexer.search`` (called inside ``_retrieve_and_gate``) filters the
    corpus using ``detect_lang`` -> "en", so only the en-tagged Section is
    ever scored. But ``_is_cjk_query`` -> True (the query quotes "熊貓"), so
    the zh threshold is the one applied to that en-scale score. With the zh
    threshold set to always-refuse and the en threshold set to always-clear,
    the CURRENT routing produces a refusal on a query whose retrieved corpus
    slice is entirely English — evidence that the threshold and the corpus
    slice can be gated by disagreeing language calls.
    """
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 9999.0)
    _index_markdown(_PRODUCTS_MD, source_id="products")
    try:
        query = "Can I redeem the 熊貓 gift card online?"
        hits = idx.search(query, k=5)
        assert _langs_of(hits) == {"en"}, (
            "corpus slice should be the en-tagged Sections (detect_lang), "
            f"got {[(sec.id, sec.metadata.get(idx.LANG_METADATA_KEY)) for sec, _ in hits]}"
        )
        gate = ret._retrieve_and_gate(query)
        # zh threshold (9999.0, always refuses) applied even though the
        # scored corpus slice is en-only -> the gate refuses.
        assert gate["early_exit"] is True
        assert gate["grounding_outcome"].reason == "below_threshold"
    finally:
        _teardown_index()
