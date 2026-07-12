"""#582 (slice 2 of 2) — dominant-script gate routing policy.

Slice 1 (#601, ``test_mixed_lang_characterization.py``) pinned the CURRENT
mismatch as characterization fixtures: the pre-LLM gate used two
independently maintained classifiers — ``retrieval._is_cjk_query`` (any CJK
char present -> the zh threshold) for the threshold, and
``indexer.detect_lang`` (CJK ratio >= 0.20 -> "zh") for the corpus slice.  A
Latin-dominant query naming a single CJK product name made them disagree: the
zh threshold got applied to an en-only corpus slice's score, producing a
false Cannot Confirm.

This slice replaces both call sites in ``_retrieve_and_gate`` with ONE
classifier, ``retrieval._dominant_script`` (a wider CJK-vs-Latin codepoint
ratio band than ``indexer.detect_lang``'s corpus-tagging ratio), so the
threshold and the corpus slice always agree outside the near-50/50 band, and
fall back to union (both slices searched, pass if either clears its own
threshold) inside it.

Tests here cover the new classifier and the gate-level routing decision
(``_gate_route``); the #601 fixtures whose pinned assertions flip under this
policy are updated in ``test_mixed_lang_characterization.py`` with a comment
naming this issue.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import app.indexer as idx
import app.retrieval as ret
from app.indexer import parse_markdown

# ---------------------------------------------------------------------------
# 1. _script_ratio / _dominant_script — pure classifier, no corpus needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, expected_ratio",
    [
        ("hello world", 0.0),
        ("退款政策", 1.0),
        ("", 0.0),
        ("12345 !?", 0.0),  # digits/punctuation only -> no letters -> 0.0
        ("退款 abc", 0.4),  # 2 CJK / (2 CJK + 3 Latin)
        ("退款政 ab", 0.6),  # 3 CJK / (3 CJK + 2 Latin)
        ("退 a", 0.5),  # 1 CJK / (1 CJK + 1 Latin)
    ],
)
def test_script_ratio(question, expected_ratio):
    assert ret._script_ratio(question) == pytest.approx(expected_ratio)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("hello world", "en"),
        ("How long does a refund take?", "en"),
        ("退款政策", "zh"),
        ("退款需要多久", "zh"),
        ("", "en"),  # no signal -> fail-closed default (mirrors indexer._DEFAULT_LANG)
        ("退款 abc", "en"),  # ratio 0.4 -> boundary, low end inclusive
        ("退款政 ab", "zh"),  # ratio 0.6 -> boundary, high end inclusive
        ("退 a", "mixed"),  # ratio 0.5 -> strictly inside the band
        # Reuses the two #601 gate-level fixtures directly, so the
        # classification driving their (possibly updated) outcomes is pinned
        # here too.
        ("Can I redeem the 熊貓 gift card online?", "en"),
        ("退款 take 幾天?", "mixed"),
    ],
)
def test_dominant_script(question, expected):
    assert ret._dominant_script(question) == expected


# ---------------------------------------------------------------------------
# 2. indexer.search(lang=...) override — corpus-slice forcing
# ---------------------------------------------------------------------------

_BILINGUAL_MD = """\
# Policies

## Return Policy

Return requests are accepted within 14 days of delivery for a full refund.

## 退貨政策

退貨申請需於收到商品後十四天內提出，審核通過後將辦理退款。
"""


def _index_bilingual() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8", delete=False) as fh:
        fh.write(_BILINGUAL_MD)
        tmp = Path(fh.name)
    try:
        secs = parse_markdown(tmp, source_id="policies")
    finally:
        tmp.unlink(missing_ok=True)
    with idx._index_lock:
        idx.sections = secs
        idx.rebuild_stats()


def _teardown_index() -> None:
    with idx._index_lock:
        idx.sections = []
        idx.rebuild_stats()


def _langs_of(hits: list) -> set[str]:
    return {sec.metadata.get(idx.LANG_METADATA_KEY) for sec, _score in hits}


def test_search_lang_override_forces_corpus_slice(monkeypatch):
    """``lang=`` wins over the query's own ``detect_lang`` pick.

    ``bm25_score`` is pinned to a constant positive value so the assertion
    isolates the corpus-slice filter from real term overlap (CODING_STANDARD
    §6.2 — no absolute-score assertions; real scoring is exercised elsewhere).
    """
    monkeypatch.setattr(idx, "bm25_score", lambda _tokens, _section, **_kwargs: 1.0)
    _index_bilingual()
    try:
        # A pure-English query's own detect_lang pick is "en" (default,
        # no override) -> only the en Section is scored.
        default_hits = idx.search("return policy", k=5)
        assert _langs_of(default_hits) == {"en"}

        # Same query, forced to the zh slice: the override wins even though
        # the query itself has zero CJK signal.
        forced_hits = idx.search("return policy", k=5, lang="zh")
        assert _langs_of(forced_hits) == {"zh"}
    finally:
        _teardown_index()


def test_search_lang_default_none_preserves_prior_behaviour():
    """Omitting ``lang`` (the default) is byte-identical to the pre-#582 call shape."""
    _index_bilingual()
    try:
        hits_no_kwarg = idx.search("退貨政策", k=5)
        hits_explicit_none = idx.search("退貨政策", k=5, lang=None)
        assert [(sec.id, score) for sec, score in hits_no_kwarg] == [
            (sec.id, score) for sec, score in hits_explicit_none
        ]
        assert _langs_of(hits_no_kwarg) == {"zh"}
    finally:
        _teardown_index()


# ---------------------------------------------------------------------------
# 3. _search_for_lang call-shape preservation — the kb_cli stub regression
#    this precedent (mirrored from `exclude_qa`) exists to protect.
# ---------------------------------------------------------------------------


def test_search_for_lang_omits_override_when_it_would_be_a_no_op(monkeypatch):
    """No ``lang=`` kwarg reaches ``indexer.search`` when the override is a no-op.

    Several test doubles across the suite (e.g. kb_cli's CLI tests) replace
    ``indexer.search`` with a fixed ``lambda query, k=3: ...`` that TypeErrors
    on any unexpected kwarg. A pure-English query routed to "en" must not
    widen the call.
    """
    calls: list[dict] = []
    real_search = idx.search

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(ret.indexer, "search", _spy)
    _index_bilingual()
    try:
        ret._search_for_lang("return policy", "en", exclude_qa=False)
        assert calls == [{"k": 3}], f"expected no lang kwarg, got kwargs={calls}"
    finally:
        _teardown_index()


def test_search_for_lang_passes_override_when_it_changes_the_pick(monkeypatch):
    """A genuine override (target differs from ``detect_lang``'s own pick) is forwarded."""
    calls: list[dict] = []
    real_search = idx.search

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(ret.indexer, "search", _spy)
    _index_bilingual()
    try:
        ret._search_for_lang("return policy", "zh", exclude_qa=False)
        assert calls == [{"k": 3, "lang": "zh"}], (
            f"expected an explicit lang override, got kwargs={calls}"
        )
    finally:
        _teardown_index()


# ---------------------------------------------------------------------------
# 4. _gate_route / _retrieve_and_gate — union fallback for the near-50/50 band
# ---------------------------------------------------------------------------

# A corpus where a genuinely mixed query ("退貨政策為 return", ratio 5/11 ~
# 0.4545, inside the band) scores non-zero against BOTH slices, so pass/fail
# per route is controlled purely by the extreme-threshold pattern used
# elsewhere in the suite (0.0 always clears, 9999.0 always refuses) —
# CODING_STANDARD §6.2, no absolute-score assertions.
_UNION_MD = """\
# Policies

## Return Policy

Return requests are accepted within 14 days of delivery for a full refund to
the original payment method.

## 退貨政策

退貨政策為顧客提供收到商品後十四天內申請退貨的權利，退貨將於審核後三個工作天內完成。
"""


def _index_union() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8", delete=False) as fh:
        fh.write(_UNION_MD)
        tmp = Path(fh.name)
    try:
        secs = parse_markdown(tmp, source_id="policies")
    finally:
        tmp.unlink(missing_ok=True)
    with idx._index_lock:
        idx.sections = secs
        idx.rebuild_stats()


_MIXED_QUERY = "退貨政策為 return"


def test_mixed_query_is_in_the_union_band():
    # Guards the fixture itself: if this stops being "mixed" the tests below
    # would silently stop exercising the union path.
    assert ret._dominant_script(_MIXED_QUERY) == "mixed"


def test_union_passes_when_only_zh_route_clears(monkeypatch):
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 0.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 9999.0)
    _index_union()
    try:
        gate = ret._retrieve_and_gate(_MIXED_QUERY)
        assert gate["early_exit"] is False
        assert _langs_of(gate["ranked"]) == {"zh"}
    finally:
        _teardown_index()


def test_union_passes_when_only_en_route_clears(monkeypatch):
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 9999.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 0.0)
    _index_union()
    try:
        gate = ret._retrieve_and_gate(_MIXED_QUERY)
        assert gate["early_exit"] is False
        assert _langs_of(gate["ranked"]) == {"en"}
    finally:
        _teardown_index()


def test_union_refuses_when_neither_route_clears(monkeypatch):
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 9999.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 9999.0)
    _index_union()
    try:
        gate = ret._retrieve_and_gate(_MIXED_QUERY)
        assert gate["early_exit"] is True
        assert gate["grounding_outcome"].reason == "below_threshold"
    finally:
        _teardown_index()


def test_union_tie_break_prefers_query_own_lean_when_both_clear(monkeypatch):
    """Both routes pass -> the tie-break picks the query's own script lean.

    ``_MIXED_QUERY`` has a CJK ratio ~0.4545 (< 0.5), so the Latin lean wins.
    """
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 0.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 0.0)
    assert ret._script_ratio(_MIXED_QUERY) < 0.5
    _index_union()
    try:
        gate = ret._retrieve_and_gate(_MIXED_QUERY)
        assert gate["early_exit"] is False
        assert _langs_of(gate["ranked"]) == {"en"}
    finally:
        _teardown_index()
