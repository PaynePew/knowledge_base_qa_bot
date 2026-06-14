# Negative-case eval — fallback rate (Week 6 FM4)

Measures whether the bot correctly **refuses** (Cannot Confirm) out-of-scope
queries the KB cannot answer. The refusal decision is the production pre-LLM
gate (`retrieval._retrieve_and_gate`: BM25 + `KB_SCORE_THRESHOLD`), so this is
deterministic and LLM-free. A *low* rate means the threshold is too permissive
(the bot answers things it should refuse).

**Correct-refusal rate: 87%** (13/15 refused)

## By category

| Category | Refusal rate |
|---|---|
| adjacent_absent | 60% |
| clearly_out_of_scope | 100% |

## Per-case detail

| Query | Category | Refused? | Reason | Top BM25 score |
|---|---|---|---|---|
| Which restaurants are nearby? | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| What's the weather tomorrow? | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| How do I invest in the stock market? | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| Write me a poem about cats. | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| What is the capital of France? | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| How do I bake sourdough bread? | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| Recommend a good action movie. | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| How tall is Mount Everest? | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| Translate hello into Japanese. | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| What is the meaning of life? | clearly_out_of_scope | ✅ | retrieval_empty | 0.000 |
| Do you price match competitors? | adjacent_absent | ✅ | retrieval_empty | 0.000 |
| Can I gift wrap my order? | adjacent_absent | ❌ leaked | answered | 1.703 |
| How many loyalty points do I earn per purchase? | adjacent_absent | ✅ | retrieval_empty | 0.000 |
| Is there a student discount? | adjacent_absent | ✅ | retrieval_empty | 0.000 |
| Can I change my delivery address after ordering? | adjacent_absent | ❌ leaked | answered | 1.489 |

> A `❌ leaked` row is an out-of-scope query that cleared the threshold — the
> raw material for calibrating `KB_SCORE_THRESHOLD` (the `top_score` column
> shows how far over the 0.5 default it landed).
