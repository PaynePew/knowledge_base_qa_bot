"""Key-Token QC gate tests (external behaviour only, CODING_STANDARD §0.2).

The gate is deterministic — given a token set and an IDF table the verdict is
fixed — so these assert the verdict directly, covering both the hard
all-stopword rejection and the soft low-distinctiveness flag the AC calls out.
"""

from __future__ import annotations

from eval.paraphrase_comparison.generation.qc import (
    QcVerdict,
    build_idf,
    check_key_tokens,
)

# A tiny corpus where 'refund' is ubiquitous (low IDF) and 'restocking' is rare
# (high IDF). 'the'/'a' never appear in the IDF table — the tokeniser drops them.
_CORPUS = [
    "refund within thirty days for a full refund",
    "refund issued to the original payment method",
    "refund applied to store credit instantly",
    "large appliances carry a restocking fee when returned",
]


def test_all_stopword_key_tokens_are_rejected():
    idf = build_idf(_CORPUS)
    verdict = check_key_tokens("p1", ["the", "a", "of", "to"], idf)
    assert isinstance(verdict, QcVerdict)
    assert verdict.rejected is True
    assert verdict.reasons  # carries a human-readable reason


def test_distinctive_tokens_pass_clean():
    idf = build_idf(_CORPUS)
    verdict = check_key_tokens("p2", ["restocking", "appliances"], idf)
    assert verdict.rejected is False
    assert verdict.flagged_tokens == []


def test_ubiquitous_token_is_flagged_not_rejected():
    idf = build_idf(_CORPUS)
    # 'refund' appears in 3 of 4 docs → IDF ≈ 1.22, below this generous bar → it
    # is flagged for human review, but the entry is still admitted (not rejected).
    assert idf["refund"] < 1.5
    verdict = check_key_tokens("p3", ["refund"], idf, min_idf=1.5)
    assert verdict.rejected is False
    assert "refund" in verdict.flagged_tokens


def test_token_absent_from_corpus_is_flagged():
    idf = build_idf(_CORPUS)
    # A token that never appears in any body cannot be matched against retrieved
    # content; flag it for review rather than silently admitting it.
    verdict = check_key_tokens("p4", ["zzqx"], idf)
    assert verdict.rejected is False
    assert "zzqx" in verdict.flagged_tokens


def test_mixed_set_with_one_distinctive_token_is_admitted():
    idf = build_idf(_CORPUS)
    # 'the' is a stop-word (dropped) but 'restocking' survives → not all-stopword.
    verdict = check_key_tokens("p5", ["the", "restocking"], idf)
    assert verdict.rejected is False


def test_build_idf_excludes_stopwords_and_is_empty_for_no_docs():
    idf = build_idf(_CORPUS)
    assert "the" not in idf  # stop-words never enter the table
    assert "restocking" in idf
    assert build_idf([]) == {}
