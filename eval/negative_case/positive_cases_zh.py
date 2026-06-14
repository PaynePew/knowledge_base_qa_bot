"""Shallow module per Ousterhout. Committed Traditional-Chinese in-scope set (#256, enlarged #261).

The 繁中 mirror of ``positive_cases.POSITIVE_CASES``: every query HAS an answer in
``corpus_zh`` (refund / shipping / account / warranty / payment / damaged items /
gift cards / order management / customer support / product info), so refusing one is
an *over-refusal* — the cost of setting ``KB_SCORE_THRESHOLD_ZH`` too high for Chinese.
The first query is reused verbatim from the committed CJK fixtures
(``test_cjk_retrieval`` / ``test_chinese_ingest_e2e``) as a proven-retrievable anchor.

#261 enlarged the original 10-case set (3-topic corpus) to cover the 10-topic corpus,
so the re-sweep that fixes the production ``KB_SCORE_THRESHOLD_ZH`` default rests on
BM25 statistics from a corpus large enough to be representative — not the 3-file
illustrative set behind the original ~1.875 recommendation.
"""

from __future__ import annotations

from .models import PositiveCase

POSITIVE_CASES_ZH: list[PositiveCase] = [
    # --- refund / shipping / account (original 3-topic corpus)
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
    # --- warranty / payment / damaged / gift cards / orders / support / product (#261)
    PositiveCase("商品的保固期限多長？", "warranty period"),
    PositiveCase("保固理賠要怎麼申請？", "warranty claim process"),
    PositiveCase("你們接受哪些付款方式？", "payment methods"),
    PositiveCase("付款失敗的話該怎麼辦？", "failed payments"),
    PositiveCase("收到的商品損壞了怎麼辦？", "reporting damage"),
    PositiveCase("如何兌換禮品卡？", "redeeming a gift card"),
    PositiveCase("如何追蹤我的訂單？", "order tracking"),
    PositiveCase("出貨前可以取消訂單嗎？", "cancel an order"),
    PositiveCase("客服的服務時間是什麼時候？", "customer support hours"),
    PositiveCase("怎麼查詢商品的尺寸？", "size guide"),
]
