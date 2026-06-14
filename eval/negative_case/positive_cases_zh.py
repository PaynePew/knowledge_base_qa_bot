"""Shallow module per Ousterhout. Committed Traditional-Chinese in-scope set (#256).

The 繁中 mirror of ``positive_cases.POSITIVE_CASES``: every query HAS an answer in
``corpus_zh`` (refund / shipping / account), so refusing one is an *over-refusal* —
the cost of setting ``KB_SCORE_THRESHOLD`` too high for Chinese. The first query is
reused verbatim from the committed CJK fixtures (``test_cjk_retrieval`` /
``test_chinese_ingest_e2e``) as a proven-retrievable anchor.
"""

from __future__ import annotations

from .models import PositiveCase

POSITIVE_CASES_ZH: list[PositiveCase] = [
    PositiveCase("退款需要多少時間？", "refund timeline (reused fixture query)"),
    PositiveCase("退款什麼時候會退回？", "refund timeline"),
    PositiveCase("哪些商品不可退款？", "non-refundable items"),
    PositiveCase("標準運送需要幾天？", "delivery estimates"),
    PositiveCase("有提供快遞運送嗎？", "delivery estimates / expedited"),
    PositiveCase("你們有提供國際運送嗎？", "international shipping"),
    PositiveCase("國際訂單的關稅由誰負擔？", "international shipping"),
    PositiveCase("如何重設密碼？", "password reset"),
    PositiveCase("如何關閉我的帳號？", "closing account"),
    PositiveCase("關閉帳號後我的資料會怎樣？", "closing account"),
]
