"""Deep module per Ousterhout. Public surface: ``HitRateAtK``, ``is_hit``, ``hit_at_k``, ``reciprocal_rank_at_k``.

C5c deterministic hit metric for the Phase 8 retrieval comparison (CONTEXT.md
§ Phase 8 > Key Tokens, PRD #100). This is the L1 metric — the source of the
headline numbers (the optional L2 Spot-check, a later slice, only reports an
agreement rate).

A retrieved item is a **hit** for a Paraphrase iff BOTH hold:

  1. its ``source_section_id`` equals the Paraphrase's ``gold_docs_section_id``
     (the retrieved unit points at the correct docs Gold Section); AND
  2. its ``content`` shares at least one token with the union of the dual-side
     Key Tokens (``key_tokens_docs`` ∪ ``key_tokens_wiki``), so a correct-id
     item whose body does NOT actually answer the question is a miss.

Condition 2 is what makes the metric measure *retrieval that answers* rather
than mere id bookkeeping: an item with the right ``source_section_id`` but
content sharing no Key Token (the correct-metadata-wrong-content case) scores a
miss.

``HitRateAtK`` wraps this as a DeepEval ``BaseMetric`` scoring one Paraphrase
at a time (1.0 hit / 0.0 miss over the top-k retrieved items). The per-type
hit_rate@k headline is the mean of these per-Paraphrase scores, computed by the
runner.

Alongside hit_rate@k the metric also exposes the **reciprocal rank** of the
first hit within the top-k (``reciprocal_rank_at_k``): 1/rank where rank is the
1-based position of the first hitting item, or 0.0 if no top-k item hits. The
per-type **MRR** (PRD #100) is the mean of these per-Paraphrase reciprocal
ranks, computed by the runner. RR rewards ranking quality (a hit at rank 1 beats
the same hit at rank 3); hit_rate@k only sees the binary "any hit in top-k".
"""

from __future__ import annotations

from collections.abc import Iterable

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

# ADR-0002: reuse markdown_kb's tokeniser so content-overlap uses the same
# alphanumeric-lowercase-stopword convention as the BM25 corpus, keeping
# "shares a token" consistent across both Stacks.
from markdown_kb.app.indexer import tokenize

from .models import RetrievedItem

DEFAULT_K = 3


# ---------------------------------------------------------------------------
# Pure hit predicate (the deterministic core)
# ---------------------------------------------------------------------------
def is_hit(
    item: RetrievedItem,
    gold_section_id: str,
    key_tokens: Iterable[str],
) -> bool:
    """Return True iff ``item`` satisfies both C5c conditions for the gold section.

    Source-match AND non-empty Key-Token overlap (see module docstring). The
    Key-Token overlap is computed on the tokenised content so casing and
    punctuation never cause a spurious miss.
    """
    if item.source_section_id != gold_section_id:
        return False
    wanted = {t.lower() for t in key_tokens}
    if not wanted:
        return False
    content_tokens = set(tokenize(item.content))
    return bool(content_tokens & wanted)


def hit_at_k(
    items: list[RetrievedItem],
    gold_section_id: str,
    key_tokens: Iterable[str],
    k: int = DEFAULT_K,
) -> float:
    """1.0 if any of the top-``k`` items is a hit, else 0.0 (hit_rate@k per Paraphrase)."""
    wanted = list(key_tokens)
    return 1.0 if any(is_hit(it, gold_section_id, wanted) for it in items[:k]) else 0.0


def reciprocal_rank_at_k(
    items: list[RetrievedItem],
    gold_section_id: str,
    key_tokens: Iterable[str],
    k: int = DEFAULT_K,
) -> float:
    """Reciprocal rank of the first hit in the top-``k`` (1/rank), else 0.0.

    ``rank`` is the 1-based position of the first item satisfying both C5c
    conditions; a hit at rank 1 scores 1.0, rank 2 scores 0.5, rank 3 scores
    ~0.333. No top-k hit scores 0.0. The per-Paraphrase reciprocal rank averaged
    over a Paraphrase Type is that type's MRR (PRD #100).
    """
    wanted = list(key_tokens)
    for rank, item in enumerate(items[:k], start=1):
        if is_hit(item, gold_section_id, wanted):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# DeepEval BaseMetric wrapper
# ---------------------------------------------------------------------------
class HitRateAtK(BaseMetric):
    """C5c hit metric as a DeepEval BaseMetric — deterministic, no LLM judge.

    Scores one Paraphrase per ``LLMTestCase``. The runner packs each test case
    so that:
      - ``expected_output`` carries the gold_section_id,
      - ``retrieval_context`` carries the retrieved items' ``source_section_id``
        (top-k order preserved),
      - ``metadata`` carries the per-item contents and the Paraphrase's Key
        Tokens.

    The metric is the headline (L1) source per PRD #100 — no model is invoked,
    so the verdict is reproducible offline.

    ``score`` carries hit_rate@k (DeepEval's success semantics key off it);
    ``reciprocal_rank`` carries the parallel RR-at-k of the same retrieval, read
    by the runner to aggregate per-type MRR. Both are recomputed on every
    ``measure`` call from the identical (items, gold, key_tokens) inputs, so they
    can never disagree about whether a hit occurred.
    """

    _required_params: list[SingleTurnParams] = [
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(self, k: int = DEFAULT_K):
        self.k = k
        self.threshold = 1.0
        self.async_mode = False
        self.score = 0.0
        self.reciprocal_rank = 0.0
        self.success = False
        self.reason: str | None = None
        self.error: str | None = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        gold_section_id = test_case.expected_output or ""
        meta = test_case.metadata or {}
        items: list[RetrievedItem] = list(meta.get("retrieved_items", []))
        key_tokens: list[str] = list(meta.get("key_tokens", []))

        self.score = hit_at_k(items, gold_section_id, key_tokens, k=self.k)
        self.reciprocal_rank = reciprocal_rank_at_k(
            items, gold_section_id, key_tokens, k=self.k
        )
        self.success = self.score >= self.threshold
        self.reason = (
            f"hit@{self.k}=1: a top-{self.k} item matched gold "
            f"'{gold_section_id}' with Key-Token overlap"
            if self.success
            else f"hit@{self.k}=0: no top-{self.k} item matched gold "
            f"'{gold_section_id}' with Key-Token overlap"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # DeepEval displays this in its report tables
        return f"C5c HitRate@{self.k}"
