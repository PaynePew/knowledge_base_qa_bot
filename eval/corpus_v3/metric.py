"""Deep module per Ousterhout. Public surface: ``is_hit``, ``hit_at_k``.

De-biased hit metric for the corpus v3 fair experiment (ADR-0045
Prerequisite 3, PRD #654 user stories 3-4). Extends the v2 C5c hit condition
(``eval.paraphrase_comparison.metric.is_hit``) along the two axes ADR-0045
names as measured tilts, both favoring Stack B
(``eval/fairness_review/verdict.md``):

  1. **Gold-label symmetry** — condition 1 is no longer equality against a
     single docs-native id; it is membership in the gold Section SET
     ``eval.corpus_v3.gold.resolve_gold_sections`` returns for that gold
     answer (docs id or wiki id, concept or entity, 1:N allowed). A wiki
     entity-page hit that v2 could never map to a docs-native id now resolves
     through the same table as everything else, so a correct hit is no longer
     a structural miss.
  2. **Dual-corpus Key Tokens** — condition 2 takes the UNION of Key Tokens
     collected from BOTH corpora (``key_tokens_docs`` and ``key_tokens_wiki``)
     rather than the docs-body-only side, so a correctly retrieved wiki
     Section that reworded the docs' phrasing still shares a token with its
     OWN corpus's Key Tokens instead of scoring a miss.

A retrieved item is a hit for a gold answer iff BOTH hold, exactly like C5c:
gold-set membership AND non-empty tokenised content overlap with the unioned
Key Tokens. Callers pass the already-resolved gold Section set (``gold``
computes it); this module stays pure and corpus-mapping-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable

# ADR-0002: reuse markdown_kb's tokeniser so content-overlap uses the same
# alphanumeric-lowercase-stopword convention as the BM25 corpus (mirrors
# eval.paraphrase_comparison.metric's tokenisation convention).
from markdown_kb.app.indexer import tokenize

from .models import RetrievedItem

DEFAULT_K = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def is_hit(
    item: RetrievedItem,
    gold_section_ids: Iterable[str],
    key_tokens_docs: Iterable[str],
    key_tokens_wiki: Iterable[str],
) -> bool:
    """Return True iff ``item`` satisfies both de-biased hit conditions.

    1. ``item.source_section_id`` is a member of ``gold_section_ids`` (the
       resolved gold equivalence class — see ``eval.corpus_v3.gold``).
    2. The tokenised ``item.content`` shares at least one token with the
       UNION of ``key_tokens_docs`` and ``key_tokens_wiki``, so casing and
       punctuation never cause a spurious miss.
    """
    if item.source_section_id not in set(gold_section_ids):
        return False
    wanted = {t.lower() for t in (*key_tokens_docs, *key_tokens_wiki)}
    if not wanted:
        return False
    content_tokens = set(tokenize(item.content))
    return bool(content_tokens & wanted)


def hit_at_k(
    items: list[RetrievedItem],
    gold_section_ids: Iterable[str],
    key_tokens_docs: Iterable[str],
    key_tokens_wiki: Iterable[str],
    k: int = DEFAULT_K,
) -> float:
    """1.0 if any of the top-``k`` items is a hit, else 0.0 (hit_rate@k per query)."""
    gold_set = set(gold_section_ids)
    docs = list(key_tokens_docs)
    wiki = list(key_tokens_wiki)
    return 1.0 if any(is_hit(it, gold_set, docs, wiki) for it in items[:k]) else 0.0
