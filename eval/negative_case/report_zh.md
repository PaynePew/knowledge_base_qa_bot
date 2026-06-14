# Negative-case eval — fallback rate · Traditional Chinese (#256)

Measures whether the bot correctly **refuses** (Cannot Confirm) out-of-scope
queries the KB cannot answer. The refusal decision is the production pre-LLM
gate (`retrieval._retrieve_and_gate`: BM25 + `KB_SCORE_THRESHOLD`), so this is
deterministic and LLM-free. A *low* rate means the threshold is too permissive
(the bot answers things it should refuse).

**Correct-refusal rate: 95%** (20/21 refused)

## By category

| Category | Refusal rate |
|---|---|
| adjacent_absent | 88% |
| clearly_out_of_scope | 100% |

## Per-case detail

| Query | Category | Refused? | Reason | Top BM25 score |
|---|---|---|---|---|
| 附近有哪些餐廳？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 明天天氣如何？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 如何投資股票市場？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 幫我寫一首關於貓的詩。 | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 法國的首都是哪裡？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 如何製作酸種麵包？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 推薦一部好看的動作片。 | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 聖母峰有多高？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 把 hello 翻譯成日文。 | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 生命的意義是什麼？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 太陽系有幾顆行星？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 初學者要怎麼學會游泳？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 鋼琴要怎麼調音？ | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| 你們有跟競品比價嗎？ | adjacent_absent | ✅ | retrieval_empty | 0.000 |
| 我的訂單可以包裝成禮物嗎？ | adjacent_absent | ❌ leaked | answered | 8.771 |
| 每次購買可以累積多少會員點數？ | adjacent_absent | ✅ | below_threshold | 2.093 |
| 有提供學生折扣嗎？ | adjacent_absent | ✅ | below_threshold | 1.889 |
| 下單後可以變更收件地址嗎？ | adjacent_absent | ✅ | below_threshold | 2.849 |
| 你們有提供訂閱制方案嗎？ | adjacent_absent | ✅ | below_threshold | 1.889 |
| 可以到實體門市自取嗎？ | adjacent_absent | ✅ | retrieval_empty | 0.000 |
| 大量採購有優惠嗎？ | adjacent_absent | ✅ | retrieval_empty | 0.000 |

> A `❌ leaked` row is an out-of-scope query that cleared the threshold — the
> raw material for calibrating `KB_SCORE_THRESHOLD` (the `top_score` column
> shows how far over the 0.5 default it landed).
