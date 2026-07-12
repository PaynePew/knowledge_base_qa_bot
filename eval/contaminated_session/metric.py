"""Shallow module. Deterministic drift metrics for contaminated-session rewrites.

Token-overlap / length-ratio only — no LLM judge, no absolute-score assertion.
Both metrics are pure functions of two strings via the production tokenizer
(``markdown_kb.app.indexer.tokenize``), so this module stays LLM-free and its
outputs are exact-comparable in tests (CODING_STANDARD §6.2 bars asserting
*LLM output text* or *corpus-sensitive BM25 scores*; a deterministic
token-overlap ratio over fixed fixture strings is neither).
"""

from __future__ import annotations

from dataclasses import dataclass

from markdown_kb.app.indexer import tokenize


@dataclass(frozen=True)
class DriftMetrics:
    """One rewrite's drift from the user's literal ask.

    ``token_overlap`` is the Jaccard index of the two token sets (1.0 =
    identical vocabulary, 0.0 = fully disjoint). ``length_ratio`` is
    ``len(tokens(rewritten)) / len(tokens(literal))`` — a rewrite that is
    much longer than the literal question is the over-specification shape
    #579 reported. ``added_tokens`` are the tokens present in the rewrite but
    absent from the literal question, sorted for determinism — the
    concrete vocabulary a reader can eyeball to see WHAT got baked in.
    """

    literal_question: str
    rewritten_query: str
    token_overlap: float
    length_ratio: float
    added_tokens: tuple[str, ...]


def compute_drift(literal_question: str, rewritten_query: str) -> DriftMetrics:
    """Compare a rewritten query against the user's literal follow-up.

    Both inputs are tokenized with the same production tokenizer retrieval
    itself uses, so ``token_overlap`` reflects actual BM25-relevant
    vocabulary drift, not surface string difference.
    """
    literal_tokens = tokenize(literal_question)
    rewritten_tokens = tokenize(rewritten_query)
    literal_set = set(literal_tokens)
    rewritten_set = set(rewritten_tokens)
    union = literal_set | rewritten_set
    overlap = len(literal_set & rewritten_set) / len(union) if union else 1.0
    literal_len = len(literal_tokens) or 1
    return DriftMetrics(
        literal_question=literal_question,
        rewritten_query=rewritten_query,
        token_overlap=round(overlap, 3),
        length_ratio=round(len(rewritten_tokens) / literal_len, 3),
        added_tokens=tuple(sorted(rewritten_set - literal_set)),
    )
