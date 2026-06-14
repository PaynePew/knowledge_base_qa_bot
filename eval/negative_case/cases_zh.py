"""Shallow module per Ousterhout. Committed Traditional-Chinese out-of-scope set (#256).

The 繁中 mirror of ``cases.NEGATIVE_CASES``: the same 10 clearly-out-of-scope + 5
adjacent-absent split, over the same refund / shipping / account topics
(``corpus_zh``). Category strings stay the canonical English values so the zh
report's by-category table lines up with the English baseline.
"""

from __future__ import annotations

from .models import NegativeCase

NEGATIVE_CASES_ZH: list[NegativeCase] = [
    # --- clearly out of scope: no commerce-vocab overlap, the gate should always fire
    NegativeCase("附近有哪些餐廳？", "local/geo search", "clearly_out_of_scope"),
    NegativeCase("明天天氣如何？", "weather", "clearly_out_of_scope"),
    NegativeCase("如何投資股票市場？", "finance", "clearly_out_of_scope"),
    NegativeCase(
        "幫我寫一首關於貓的詩。", "open-ended generation", "clearly_out_of_scope"
    ),
    NegativeCase("法國的首都是哪裡？", "general trivia", "clearly_out_of_scope"),
    NegativeCase("如何製作酸種麵包？", "cooking", "clearly_out_of_scope"),
    NegativeCase("推薦一部好看的動作片。", "entertainment", "clearly_out_of_scope"),
    NegativeCase("聖母峰有多高？", "geography trivia", "clearly_out_of_scope"),
    NegativeCase("把 hello 翻譯成日文。", "translation", "clearly_out_of_scope"),
    NegativeCase("生命的意義是什麼？", "open-ended", "clearly_out_of_scope"),
    # --- adjacent-absent: shares 繁中 commerce vocab, but the specific answer is
    #     absent from the corpus (refund timeline / non-refundable / shipping
    #     estimates / international / password reset / account closing only)
    NegativeCase(
        "你們有跟競品比價嗎？", "no price-match policy in KB", "adjacent_absent"
    ),
    NegativeCase(
        "我的訂單可以包裝成禮物嗎？", "no gift-wrap info in KB", "adjacent_absent"
    ),
    NegativeCase(
        "每次購買可以累積多少會員點數？", "no loyalty program in KB", "adjacent_absent"
    ),
    NegativeCase("有提供學生折扣嗎？", "no discount policy in KB", "adjacent_absent"),
    NegativeCase(
        "下單後可以變更收件地址嗎？",
        "no address-change policy in KB",
        "adjacent_absent",
    ),
]
