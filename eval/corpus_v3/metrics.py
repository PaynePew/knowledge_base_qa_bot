"""Deep module per Ousterhout. Public surface: ``is_hit``, ``hit_at_k``,
``reciprocal_rank_at_k``, ``HIT_AT_1``, ``HIT_AT_3``.

Stratum-agnostic retrieval-hit metrics for the corpus v3 fair experiment
(issue #659, ADR-0045 Prerequisite 4: "hit@1 and MRR reported alongside
hit@3"). Mirrors the C5c hit predicate from
``eval.paraphrase_comparison.metric`` (source-id match AND non-empty
Key-Token content overlap, PRD #100) but generalises the gold side to a SET
of Section ids — ADR-0045 Prerequisite 3's 1:N entity-page mapping (issue
#658) lets one query have several equally-correct gold Sections — and stays
independent of that package (PRD #654: corpus v3 is "a new eval package,
sibling to the existing paraphrase-comparison eval ... with its own
committed fixtures ... and production isolation").

``hit_at_k`` and ``reciprocal_rank_at_k`` take an explicit ``k`` rather than
each owning a single cutoff constant, so the same functions produce hit@1,
hit@3, and MRR-at-3 (reciprocal rank at the hit@3 cutoff) without duplicating
the hit predicate three times.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# ADR-0002: reuse markdown_kb's tokeniser so content-overlap uses the same
# alphanumeric-lowercase-stopword convention as the BM25 corpus, keeping
# "shares a token" consistent with the v2 eval's C5c metric.
from markdown_kb.app.indexer import tokenize

from .models import RetrievedItem

# The two required cutoffs (ADR-0045 Prerequisite 4). MRR is reciprocal rank
# at the HIT_AT_3 cutoff — there is no separate "MRR cutoff" constant.
HIT_AT_1 = 1
HIT_AT_3 = 3


def is_hit(
    item: RetrievedItem,
    gold_section_ids: Iterable[str],
    key_tokens: Iterable[str],
) -> bool:
    """True iff ``item`` resolves to one of ``gold_section_ids`` AND its
    content shares at least one (lower-cased) token with ``key_tokens``.

    Both conditions are required, mirroring the v2 C5c metric: an item whose
    id matches a gold Section but whose content shares no Key Token is a miss
    (correct-metadata-wrong-content case), and an empty ``gold_section_ids``
    or ``key_tokens`` can never be satisfied (an unanswerable query's empty
    gold set means every retrieval scores a miss on this metric, by design —
    correct-refusal is a separate axis).
    """
    gold = set(gold_section_ids)
    if not gold or item.source_section_id not in gold:
        return False
    wanted = {t.lower() for t in key_tokens}
    if not wanted:
        return False
    return bool(set(tokenize(item.content)) & wanted)


def hit_at_k(
    items: Sequence[RetrievedItem],
    gold_section_ids: Iterable[str],
    key_tokens: Iterable[str],
    k: int,
) -> float:
    """1.0 if any of the top-``k`` items is a hit, else 0.0."""
    gold = list(gold_section_ids)
    wanted = list(key_tokens)
    return 1.0 if any(is_hit(it, gold, wanted) for it in items[:k]) else 0.0


def reciprocal_rank_at_k(
    items: Sequence[RetrievedItem],
    gold_section_ids: Iterable[str],
    key_tokens: Iterable[str],
    k: int,
) -> float:
    """Reciprocal rank (1/rank) of the first hit within the top-``k``, else 0.0.

    ``rank`` is the 1-based position of the first item satisfying
    :func:`is_hit`. The mean of this value over a stratum, at ``k =
    HIT_AT_3``, is that stratum's MRR (ADR-0045 Prerequisite 4).
    """
    gold = list(gold_section_ids)
    wanted = list(key_tokens)
    for rank, item in enumerate(items[:k], start=1):
        if is_hit(item, gold, wanted):
            return 1.0 / rank
    return 0.0
