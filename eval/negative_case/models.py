"""Data models for the negative-case eval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NegativeCase:
    """An out-of-scope query the bot SHOULD refuse (no answer exists in the KB).

    ``category`` separates ``clearly_out_of_scope`` (no vocabulary overlap with
    the corpus — the gate should always fire) from ``adjacent_absent`` (shares
    commerce vocabulary but the specific answer is absent — the harder case where
    a too-low threshold leaks a confident wrong answer).
    """

    query: str
    note: str
    category: str


@dataclass(frozen=True)
class RefusalOutcome:
    """The system's pre-LLM response to one negative case.

    ``refused`` is True iff the Cannot Confirm gate fired (the correct behaviour).
    ``reason`` carries the gate reason (``retrieval_empty`` / ``below_threshold``)
    when refused, or ``answered`` when the query slipped past the gate.
    ``top_score`` is the BM25 score of the top hit (0.0 when nothing ranked) — the
    signal you tune ``KB_SCORE_THRESHOLD`` against.
    """

    query: str
    refused: bool
    reason: str
    top_score: float
