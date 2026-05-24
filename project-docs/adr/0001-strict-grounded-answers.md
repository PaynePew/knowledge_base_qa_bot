# Strict grounded answers, never loose

The bot answers strictly from cited Sections in the retrieved CONTEXT. The LLM may not draw on outside world knowledge, may not infer beyond what is written, and must reply with the literal phrase `"I cannot confirm from the knowledge base."` whenever retrieval is empty, sub-threshold, or only partially related. Synthesis across multiple cited Sections is the one form of reasoning allowed.

We chose strict over loose because the project's headline use cases (customer-support FAQs today, personal knowledge wiki later) make hallucination cost asymmetric: a confidently wrong refund timeline or medical note is far worse than an honest "cannot confirm." Strict also matches the project's grounding contract in `PROMPT.md` (the `"Which restaurants are nearby?"` verification case must explicitly fail to confirm), makes the system behavior automatable to test, and aligns with the most-upvoted critique of Karpathy's LLM Wiki pattern (@laphilosophia: source-grounded and citation-first is the discipline that keeps the pattern from rotting).

## Consequences

- The score-threshold fallback runs *before* the LLM call when retrieval is weak — the model is never handed weak context and asked to police itself.
- The system prompt must explicitly frame `"cannot confirm"` as a *good* answer, otherwise the model's "be helpful" prior produces plausible-sounding fabrications.
- Partial-match cases ("the KB mentions card provider delays but does not specify Visa") are answered by quoting the partial Section and naming what is missing, never by completing the gap with general knowledge.
- Cross-Section synthesis is allowed and expected (e.g. combining `cancellation-window` and `refund-timeline` to answer one question), provided every claim still traces to a cited Section.
