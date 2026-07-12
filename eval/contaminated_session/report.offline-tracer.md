⚠️ PLACEHOLDER — NOT REAL DATA. The rewrite step below used a deterministic stand-in, not the real gateway.app.query_rewriting LLM call (no OPENAI_API_KEY, or --fake was passed). Do not interpret these numbers as real drift measurements.

# Contaminated-session rewrite drift (#608)

An earlier WRONG answer sits in a session's history; a later
on-topic follow-up is re-asked. Two LLM-free, deterministic
measurements per case (only the contaminated rewrite itself can be
a real LLM call — see module docstring):

- **Rewrite drift** — token-overlap Jaccard + length ratio between
  the CONTAMINATED rewrite and the user's literal follow-up (lower
  overlap / higher length ratio means the rewrite pulled in more
  than what was actually asked).
- **Answer flip** — does the retrieval gate's top Section / outcome
  reason differ between the contaminated rewrite and the
  clean-history control (no prior turn) for the SAME literal
  follow-up?

**0/2 case(s) flipped** the retrieval outcome under contamination.

## Per-case detail

| Case | Literal follow-up | Contaminated rewrite | Token overlap | Length ratio | Flipped? |
|---|---|---|---|---|---|
| wrong_topic_contamination | And what if I don't have the receipt? | And what if I don't have the receipt? [How long do refunds take?] | 57% | 1.75 | no |
| wrong_fact_same_topic | Does that apply if I return it as store credit too? | Does that apply if I return it as store credit too? [What's your refund window?] | 70% | 1.43 | no |

## Gate outcomes

| Case | Contaminated top source | Contaminated reason | Clean top source | Clean reason |
|---|---|---|---|---|
| wrong_topic_contamination | refund_policy#store-credit-refunds | claim_supported | refund_policy#store-credit-refunds | claim_supported |
| wrong_fact_same_topic | refund_policy#store-credit-refunds | claim_supported | refund_policy#store-credit-refunds | claim_supported |

## Case notes

- **wrong_topic_contamination**: Turn 1's WRONG answer conflates the refund timeline with the Shipping Policy's carrier delivery timeline (the real refund window is 14 days and is unrelated to the carrier). The literal follow-up is on-topic for Refund Policy > Store Credit Refunds — a topic-drift probe: does the shipping-flavoured wrong turn steer the rewrite (and therefore retrieval) toward Shipping Policy instead?
- **wrong_fact_same_topic**: Turn 1's WRONG answer states 60 days (the real policy is 14 days). The literal follow-up stays on-topic (still Refund Policy) — a fact-drift probe: does the wrong '60 days' figure get baked into the rewritten query text even when the retrieved Section does not flip?
