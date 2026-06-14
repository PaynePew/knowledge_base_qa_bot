"""Negative-case / fallback-rate eval (Week 6 FM4 — Knowledge Gap, issue #249).

Measures whether the bot correctly REFUSES (Cannot Confirm) out-of-scope queries
that the KB cannot answer — the complement to ``paraphrase_comparison`` (which
measures positive retrieval robustness). The refusal decision is made entirely at
the pre-LLM Cannot Confirm gate (``retrieval._retrieve_and_gate``), so the whole
eval is deterministic and LLM-free.
"""
