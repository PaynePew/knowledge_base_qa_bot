"""#291 — 1:1 parallel EN/ZH corpus guard (PRD #284, the final content pass).

Phase B (#288) brought Traditional-Chinese coverage to topic parity — 20 ZH Sources for the
20 EN topics — and re-baked the committed seed; #287/#290 added query-language routing on the
BM25 and RAG stacks. This test LOCKS IN the resulting invariant over the SHIPPED baked seed
(``.kb/index.json``, force-committed), so a future corpus/index change cannot silently regress
1:1 bilingual coverage:

  * source-level 1:1 — ``docs/demo-zh`` carries a Source for every ``docs/fake-docs`` topic, and
  * bidirectional same-language retrieval — a same-language query returns a same-language Source
    for every sampled topic, in BOTH languages.

Offline / BM25 only (no ``OPENAI_API_KEY``). The RAG/FAISS side of the same property is verified
live (a 20/20 bidirectional probe + grounded smoke, recorded on the PR); one live test per
surface (ADR-0005) is deliberately not expanded here. Behavioural per the test README — asserts
on the languages of the retrieved Sections over the real seed, never on internal scoring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.indexer as _idx
from app._paths import DOCS_DIR
from app.indexer import LANG_METADATA_KEY

# The force-committed BM25 seed whose bilingual invariants this test locks in.
_COMMITTED_INDEX = Path(__file__).resolve().parents[2] / ".kb" / "index.json"

# (topic, zh query, en query): single-intent, well-covered queries spanning the original 8 ZH
# Sources and the 12 added in #288 — including the about/contact cluster the parity audit
# flagged (customer_support / acme_shop_about) and the shipping cluster split across files.
_TOPICS = [
    ("payment_methods", "有哪些付款方式可以選擇？", "What payment methods do you accept?"),
    ("returns_policy", "退款政策的條件是什麼？", "What is your refund policy?"),
    ("warranty", "商品的保固期有多久？", "How long is the product warranty?"),
    ("customer_support", "我要如何聯絡客服？", "How do I contact customer support?"),
    ("acme_shop_about", "ACME 商店的營業時間是？", "What are Acme Shop's operating hours?"),
    ("international_shipping", "你們有提供國際配送嗎？", "Do you ship internationally?"),
    ("store_pickup", "可以選擇門市自取嗎？", "Can I pick up my order in store?"),
    ("gift_cards", "禮物卡要如何使用？", "How do I redeem a gift card?"),
    ("subscription_orders", "如何設定定期訂閱配送？", "How do I set up a subscription order?"),
    ("promo_codes", "促銷折扣碼如何使用？", "How do I apply a promo code?"),
]


@pytest.fixture(scope="module")
def baked_index(tmp_path_factory):
    """Load the real, force-committed ``.kb/index.json`` — the shipped bilingual seed.

    ``load_index_json`` emits an ``index_loaded`` entry via ``app.logger.log_event``.
    This fixture is module-scoped, so it sets up *before* the function-scoped
    ``_redirect_paths_to_tmp`` autouse redirect — and would otherwise append that
    entry to the committed ``wiki/log.md`` (a §6.5 isolation leak the repo-root guard
    now flags — #303). Redirect ``LOG_PATH`` to a tmp file for the load, and read the
    seed from the explicit committed path so the probe is independent of redirect
    ordering and writes nothing under ``wiki/``.

    Resets the module-global index on teardown so fixture-based tests elsewhere are unaffected.
    """
    import app.logger as _logger

    log_tmp = tmp_path_factory.mktemp("baked_seed") / "log.md"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_logger, "LOG_PATH", log_tmp)
        _idx.load_index_json(_COMMITTED_INDEX)
    try:
        yield
    finally:
        with _idx._index_lock:
            _idx.sections = []
            _idx.rebuild_stats()


def _top_lang(query: str) -> tuple[str | None, str]:
    hits = _idx.search(query, k=1)
    if not hits:
        return None, "<no hits>"
    sec = hits[0][0]
    return sec.metadata.get(LANG_METADATA_KEY), sec.id


def test_source_level_one_to_one_parity():
    """``docs/demo-zh`` has a Source for every ``docs/fake-docs`` topic (1:1 at the Source level)."""
    en = sorted(p.name for p in (DOCS_DIR / "fake-docs").glob("*.md"))
    zh = sorted(p.name for p in (DOCS_DIR / "demo-zh").glob("*.md"))
    assert len(en) == len(zh), (
        f"EN/ZH Source counts must match for 1:1 parity: {len(en)} EN vs {len(zh)} ZH"
    )
    assert len(zh) >= 20, f"expected the full parallel ZH corpus (>=20 Sources), got {len(zh)}"


def test_baked_seed_is_genuinely_bilingual(baked_index):
    """The shipped BM25 seed carries a meaningful body of BOTH zh and en Sections.

    Guards against the bidirectional assertions below passing trivially over a monolingual seed.
    """
    langs = [s.metadata.get(LANG_METADATA_KEY) for s in _idx.sections]
    n_zh = sum(1 for x in langs if x == "zh")
    n_en = sum(1 for x in langs if x == "en")
    assert n_zh >= 20 and n_en >= 20, f"seed is not balanced-bilingual: zh={n_zh} en={n_en}"


@pytest.mark.parametrize("topic,zh_q,en_q", _TOPICS, ids=[t[0] for t in _TOPICS])
def test_zh_query_returns_zh_source(topic, zh_q, en_q, baked_index):
    """A Traditional-Chinese query returns a Chinese Source for every sampled topic (AC3, zh→zh)."""
    lang, sec_id = _top_lang(zh_q)
    assert lang == "zh", f"[{topic}] zh query top hit is not zh: lang={lang} section={sec_id!r}"


@pytest.mark.parametrize("topic,zh_q,en_q", _TOPICS, ids=[t[0] for t in _TOPICS])
def test_en_query_returns_en_source(topic, zh_q, en_q, baked_index):
    """An English query returns an English Source for every sampled topic (AC3, en→en)."""
    lang, sec_id = _top_lang(en_q)
    assert lang == "en", f"[{topic}] en query top hit is not en: lang={lang} section={sec_id!r}"
