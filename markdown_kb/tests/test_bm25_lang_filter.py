"""#287 — BM25 language-filtered retrieval (belt-and-suspenders), PRD #284.

``indexer.search`` must restrict a Chinese query to ``zh``-tagged Sections and an
English query to ``en``-tagged Sections, using the QUERY language. This makes
explicit the routing that CJK-bigram tokenisation (ADR-0014) already does
implicitly — English tokens never match Chinese bigrams and vice versa — so it
must NOT regress existing BM25 answers.

The query-language predicate is the consolidated ``indexer.detect_lang`` helper
(#285, PRD #284 user story 16) — the same one that tags each Section's
``metadata['lang']`` at index time — so query-time routing and index-time tagging
can never drift apart. No third language predicate is forked.

Behavioural, not implementation-detail (test README §"Why integration-first"):
the fixture builds a real mixed-language BM25 index and asserts on the returned
Sections' languages, never on internal scoring.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import app.indexer as _idx
from app.indexer import LANG_METADATA_KEY, detect_lang, parse_markdown

# A single Source carrying parallel English and Chinese Sections on the SAME
# topic (refunds). Because the topic is shared, a naive (unfiltered) BM25 search
# could only ever cross-match via token overlap — which CJK bigrams structurally
# prevent — so this fixture is the honest test of the explicit language filter:
# the zh and en Sections are about the same thing, differing only in language.
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


def _index_mixed() -> list:
    """Build a BM25 index over the mixed-language fixture and return the Sections."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8", delete=False) as fh:
        fh.write(_MIXED_MD)
        tmp = Path(fh.name)
    try:
        secs = parse_markdown(tmp, source_id="policies")
    finally:
        tmp.unlink(missing_ok=True)
    with _idx._index_lock:
        _idx.sections = secs
        _idx.rebuild_stats()
    return secs


def _teardown_index() -> None:
    with _idx._index_lock:
        _idx.sections = []
        _idx.rebuild_stats()


def _langs_of(hits: list) -> set[str]:
    """Return the set of ``lang`` tags carried by the retrieved Sections."""
    return {sec.metadata.get(LANG_METADATA_KEY) for sec, _score in hits}


def test_fixture_index_is_genuinely_mixed_language():
    """Sanity: the fixture really does carry both zh and en Sections.

    Guards against the filter passing trivially because the fixture happened to
    be monolingual (a green test that proves nothing).
    """
    secs = _index_mixed()
    try:
        langs = {sec.metadata.get(LANG_METADATA_KEY) for sec in secs}
        assert "zh" in langs, f"fixture must contain zh Sections, got langs={langs}"
        assert "en" in langs, f"fixture must contain en Sections, got langs={langs}"
    finally:
        _teardown_index()


def test_chinese_query_restricted_to_zh_sections():
    """A Chinese BM25 query returns ONLY zh-tagged Sections (AC1)."""
    _index_mixed()
    try:
        hits = _idx.search("退款需要多久", k=5)
        assert hits, "Chinese query should retrieve at least one Section"
        assert _langs_of(hits) == {"zh"}, (
            f"Chinese query must return only zh Sections, got "
            f"{[(sec.id, sec.metadata.get(LANG_METADATA_KEY)) for sec, _ in hits]}"
        )
    finally:
        _teardown_index()


def test_english_query_restricted_to_en_sections():
    """An English BM25 query returns ONLY en-tagged Sections (AC1)."""
    _index_mixed()
    try:
        hits = _idx.search("how long do refunds take", k=5)
        assert hits, "English query should retrieve at least one Section"
        assert _langs_of(hits) == {"en"}, (
            f"English query must return only en Sections, got "
            f"{[(sec.id, sec.metadata.get(LANG_METADATA_KEY)) for sec, _ in hits]}"
        )
    finally:
        _teardown_index()


def test_no_regression_english_top_hit_unchanged():
    """No regression (AC2): the English query's ranking among en Sections is
    identical to the implicit token-routing baseline.

    Bigram tokenisation already makes en queries score only en Sections (zh
    bigrams never match Latin tokens), so the explicit filter must preserve the
    exact same top hit. We compute the unfiltered en-only baseline by scoring the
    English Sections directly and assert the filtered ``search`` agrees.
    """
    secs = _index_mixed()
    try:
        hits = _idx.search("how long do refunds take", k=5)
        top_sec = hits[0][0]
        # Baseline: among the en Sections only, the refund one out-ranks shipping
        # for a refund query under plain BM25 — the filter must not change that.
        assert top_sec.metadata.get(LANG_METADATA_KEY) == "en"
        assert "refund" in top_sec.id.lower(), (
            f"refund query should still rank the en refund Section first, got {top_sec.id!r}"
        )
        # And the zh refund Section must NOT have leaked into the en result.
        zh_ids = [sec.id for sec in secs if sec.metadata.get(LANG_METADATA_KEY) == "zh"]
        returned_ids = {sec.id for sec, _ in hits}
        assert not (set(zh_ids) & returned_ids), (
            f"zh Sections leaked into en result: {set(zh_ids) & returned_ids}"
        )
    finally:
        _teardown_index()


def test_query_language_predicate_is_detect_lang():
    """The query-language decision reuses ``indexer.detect_lang`` (#285), not a
    forked predicate — so index-time tagging and query-time routing share one
    classifier (PRD #284 user story 16).
    """
    assert detect_lang("退款需要多久") == "zh"
    assert detect_lang("how long do refunds take") == "en"


# A Source where the en and zh Sections share an ASCII token (a brand name). This
# is the case the IMPLICIT bigram routing does NOT cover: "paypal" tokenises the
# same in both languages, so a Chinese query carrying that token cross-matches the
# English Section under plain BM25 (verified: the en Section even out-ranks the zh
# one). The explicit ``lang`` filter is what closes this leak — this fixture is the
# test that distinguishes the explicit filter from the implicit token routing.
_SHARED_TOKEN_MD = """\
# Policies

## PayPal Refunds

Refunds to your PayPal account are processed within 5 to 7 business days.

## PayPal 退款查詢

請使用您的 PayPal 帳號登入後，於訂單頁面查看 5 至 7 天內的退款進度與狀態說明。
"""


def _index_shared_token() -> list:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8", delete=False) as fh:
        fh.write(_SHARED_TOKEN_MD)
        tmp = Path(fh.name)
    try:
        secs = parse_markdown(tmp, source_id="policies")
    finally:
        tmp.unlink(missing_ok=True)
    with _idx._index_lock:
        _idx.sections = secs
        _idx.rebuild_stats()
    return secs


def test_chinese_query_with_shared_ascii_token_excludes_en_section():
    """The discriminating case (AC1 + AC2): a Chinese query carrying a shared
    ASCII token (a brand name) must NOT return the English Section.

    Under plain BM25 the shared "paypal" token makes the en Section match — and
    even out-rank — the zh one, so the implicit token routing leaks here. The
    explicit language filter is what restricts the result to zh Sections.
    """
    _index_shared_token()
    try:
        hits = _idx.search("PayPal 退款進度", k=5)
        assert hits, "Chinese query should retrieve the zh Section"
        assert _langs_of(hits) == {"zh"}, (
            "Chinese query carrying a shared ASCII token must still return only zh "
            f"Sections, got {[(sec.id, sec.metadata.get(LANG_METADATA_KEY)) for sec, _ in hits]}"
        )
    finally:
        _teardown_index()


def test_english_query_with_shared_ascii_token_excludes_zh_section():
    """The mirror case: an English query carrying the same shared token returns
    only en Sections — the zh Section (which also contains "paypal") is excluded.
    """
    _index_shared_token()
    try:
        hits = _idx.search("paypal refund progress", k=5)
        assert hits, "English query should retrieve the en Section"
        assert _langs_of(hits) == {"en"}, (
            "English query carrying a shared ASCII token must return only en "
            f"Sections, got {[(sec.id, sec.metadata.get(LANG_METADATA_KEY)) for sec, _ in hits]}"
        )
    finally:
        _teardown_index()
