"""#261 — per-language ``KB_SCORE_THRESHOLD_ZH``: call-time CJK-aware gate routing.

The pre-LLM Cannot Confirm gate (``retrieval._retrieve_and_gate``) must apply
``_SCORE_THRESHOLD_ZH`` to CJK queries and ``_SCORE_THRESHOLD`` to Latin queries.
Chinese bigram BM25 scores sit in a higher band than English (#256/#261
calibration: in-scope min ~4.9 vs English ~1.4), so the English-calibrated 0.5
leaks Chinese adjacent-absent queries; a Chinese-calibrated value (4.0) gates them.

Gate parity (CODING_STANDARD §4.3): the routing lives in the deep ``retrieval``
module so CLI / MCP / Browser / Gateway all inherit it. Interim — superseded by
the Phase 13 reranker (ADR-0014 / roadmap Phase 13).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import app.indexer as _idx
import app.retrieval as ret
from app.indexer import parse_markdown

# One Source carrying both a Latin and a CJK Section so a Latin query and a CJK
# query each retrieve a non-zero top score from the same index.
_MIXED_MD = """\
# Policies

## Refund Policy

Approved refunds are processed within 5 to 7 business days to the original method.

## 退款政策

核准的退款會在我們收到退回商品後的 5 至 7 個工作天內處理，款項將退回原付款方式。
"""


def _index_mixed() -> None:
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


def _teardown_index() -> None:
    with _idx._index_lock:
        _idx.sections = []
        _idx.rebuild_stats()


def test_zh_threshold_default_is_calibrated_value():
    """The shipped Chinese default is the #261 re-swept value (4.0), distinct from EN 0.5."""
    assert ret._KB_SCORE_THRESHOLD_ZH_DEFAULT == 4.0
    # En default is unchanged at 0.5 — the two are genuinely separate knobs.
    assert ret._KB_SCORE_THRESHOLD_DEFAULT == 0.5


def test_cjk_query_routes_to_zh_threshold(monkeypatch):
    """A CJK query is gated by ``_SCORE_THRESHOLD_ZH``, not the English threshold.

    EN threshold set absurdly high (would refuse everything) and ZH threshold set
    to 0.0 (would refuse nothing): the CJK query must clear the gate (routed to ZH),
    while the Latin query is refused (routed to EN).
    """
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 9999.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 0.0)
    _index_mixed()
    try:
        cjk_gate = ret._retrieve_and_gate("退款政策")
        latin_gate = ret._retrieve_and_gate("refund policy")
        # CJK routed to ZH (0.0) → not refused.
        assert cjk_gate["early_exit"] is False, (
            "CJK query should route to _SCORE_THRESHOLD_ZH (0.0) and clear the gate"
        )
        # Latin routed to EN (9999) → refused.
        assert latin_gate["early_exit"] is True
        assert latin_gate["grounding_outcome"].reason == "below_threshold"
    finally:
        _teardown_index()


def test_latin_query_routes_to_en_threshold(monkeypatch):
    """The reverse: ZH threshold high, EN threshold 0.0 — Latin clears, CJK refused."""
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(ret, "_SCORE_THRESHOLD_ZH", 9999.0)
    _index_mixed()
    try:
        cjk_gate = ret._retrieve_and_gate("退款政策")
        latin_gate = ret._retrieve_and_gate("refund policy")
        # Latin routed to EN (0.0) → not refused.
        assert latin_gate["early_exit"] is False
        # CJK routed to ZH (9999) → refused.
        assert cjk_gate["early_exit"] is True
        assert cjk_gate["grounding_outcome"].reason == "below_threshold"
    finally:
        _teardown_index()


def test_is_cjk_query_detects_any_cjk_character():
    """``_is_cjk_query`` is True when the query contains any CJK char, else False."""
    assert ret._is_cjk_query("退款政策") is True
    assert ret._is_cjk_query("How long do refunds take?") is False
    # Mixed scripts → CJK present → routes to the zh threshold.
    assert ret._is_cjk_query("refund 退款") is True
    assert ret._is_cjk_query("") is False
