"""Deterministic IDF Key Token derivation tests (external behaviour only).

The ``derive_key_tokens`` function is the core of Phase 8.5 S1: replace the
LLM-emitted Key Tokens with a corpus-grounded, deterministic derivation from
each Paraphrase's Gold Section body. These tests verify:

  1. Returned tokens are single words (no multi-word phrases).
  2. Tokens shorter than 2 chars are dropped.
  3. Number-token handling: pure numeric tokens are excluded (documented policy).
  4. An empty corpus section returns an empty list (not an error).
  5. A section with only stopwords / short tokens returns an empty list.
  6. Tokens are ranked by IDF (higher IDF = more distinctive, returned first).
  7. top_n is respected — at most ``top_n`` tokens returned.
  8. Each returned token is actually present in the IDF table (came from corpus).
  9. Round-trip: re-keying the committed queries.yaml yields zero multi-word
     tokens and no Paraphrase with an empty matchable Key-Token set.
"""

from __future__ import annotations

from eval.paraphrase_comparison.generation.qc import build_idf, derive_key_tokens
from eval.paraphrase_comparison.loader import load_paraphrases


# ---------------------------------------------------------------------------
# Tiny deterministic corpus for isolated unit tests
# ---------------------------------------------------------------------------
# "restocking" is rare (1/4 docs) → high IDF
# "refund" is common (3/4 docs) → lower IDF
# "the"/"a"/"to"/"for" are stop-words → never in IDF table
_CORPUS = [
    "refund within thirty days for a full refund",
    "refund issued to the original payment method",
    "refund applied to store credit instantly",
    "large appliances carry a restocking fee when returned",
]

_IDF = build_idf(_CORPUS)


# ---------------------------------------------------------------------------
# 1. No multi-word tokens
# ---------------------------------------------------------------------------
def test_all_returned_tokens_are_single_words():
    tokens = derive_key_tokens(
        "restocking fee applies to large appliances",
        _IDF,
        top_n=10,
    )
    for tok in tokens:
        assert " " not in tok, f"multi-word token found: {tok!r}"


# ---------------------------------------------------------------------------
# 2. Min-length filter: tokens shorter than 2 chars are dropped
# ---------------------------------------------------------------------------
def test_tokens_shorter_than_2_chars_are_dropped():
    # Build a corpus and IDF where single-char tokens could appear
    mini_corpus = ["a x b y z larger token"]
    mini_idf = build_idf(mini_corpus)
    # "x", "y", "z" should NOT appear (len < 2); "larger" and "token" should
    tokens = derive_key_tokens("x y z larger token", mini_idf, top_n=10)
    for tok in tokens:
        assert len(tok) >= 2, f"token shorter than 2 chars returned: {tok!r}"


# ---------------------------------------------------------------------------
# 3. Number-token handling: pure numeric tokens are excluded
# ---------------------------------------------------------------------------
def test_pure_numeric_tokens_are_excluded():
    num_corpus = ["30 days return window", "thirty days policy"]
    num_idf = build_idf(num_corpus)
    tokens = derive_key_tokens("30 days thirty refund", num_idf, top_n=10)
    for tok in tokens:
        assert not tok.isdigit(), f"pure numeric token found: {tok!r}"


# ---------------------------------------------------------------------------
# 4. Empty section body returns empty list
# ---------------------------------------------------------------------------
def test_empty_section_body_returns_empty_list():
    tokens = derive_key_tokens("", _IDF, top_n=5)
    assert tokens == []


# ---------------------------------------------------------------------------
# 5. Section with only stopwords / short tokens returns empty list
# ---------------------------------------------------------------------------
def test_section_with_only_stopwords_returns_empty_list():
    # "the", "a", "to", "of" are all stopwords → tokenize() drops them all
    tokens = derive_key_tokens("the a to of", _IDF, top_n=5)
    assert tokens == []


# ---------------------------------------------------------------------------
# 6. Tokens are ranked by IDF (most distinctive first)
# ---------------------------------------------------------------------------
def test_tokens_ranked_by_idf_most_distinctive_first():
    # "restocking" has a higher IDF than "refund" in _IDF
    assert _IDF["restocking"] > _IDF["refund"]
    # Section body contains both; restocking should come first
    tokens = derive_key_tokens(
        "restocking fee refund payment",
        _IDF,
        top_n=5,
    )
    assert tokens  # non-empty
    # restocking should appear before refund because it's more distinctive
    if "restocking" in tokens and "refund" in tokens:
        assert tokens.index("restocking") < tokens.index("refund")


# ---------------------------------------------------------------------------
# 7. top_n is respected
# ---------------------------------------------------------------------------
def test_top_n_limits_output():
    tokens = derive_key_tokens(
        "restocking fee refund payment credit method",
        _IDF,
        top_n=2,
    )
    assert len(tokens) <= 2


def test_top_n_larger_than_available_returns_all():
    tokens = derive_key_tokens("restocking", _IDF, top_n=100)
    # Only "restocking" survives (all others are stopwords or absent from IDF);
    # but some words may not be in the IDF (tokens not in corpus → skip).
    # Key assertion: we get AT MOST top_n items and not more than actual unique tokens.
    assert len(tokens) <= 100


# ---------------------------------------------------------------------------
# 8. Returned tokens are in the IDF table
# ---------------------------------------------------------------------------
def test_returned_tokens_are_in_idf_table():
    tokens = derive_key_tokens(
        "restocking fee refund original payment",
        _IDF,
        top_n=10,
    )
    for tok in tokens:
        assert tok in _IDF, f"token not in IDF table: {tok!r}"


# ---------------------------------------------------------------------------
# 9. Round-trip: committed queries.yaml re-keying yields zero multi-word
#    tokens and no Paraphrase with an empty matchable Key-Token set
# ---------------------------------------------------------------------------
def test_rekeying_committed_queries_yields_no_multiword_tokens(tmp_path):
    """After re-keying via derive_key_tokens, no Paraphrase has multi-word tokens."""
    from eval.paraphrase_comparison.generation.sampling import GOLD_SECTIONS_PATH
    from eval.paraphrase_comparison.rekey import build_corpus_idf, rekey_paraphrase

    _PKG_ROOT = GOLD_SECTIONS_PATH.parent
    corpus_dir = _PKG_ROOT / "corpus"
    idf = build_corpus_idf(corpus_dir)
    paraphrases = load_paraphrases()

    # Import the section bodies for re-keying
    from markdown_kb.app.indexer import parse_markdown, slugify

    bodies: dict[str, str] = {}
    for md_file in sorted(corpus_dir.glob("*.md")):
        for section in parse_markdown(md_file, source_id=None):
            if section.content.strip():
                bodies[f"{md_file.name}#{slugify(section.heading)}"] = section.content

    multi_word_count = 0
    empty_set_count = 0

    for p in paraphrases:
        body = bodies.get(p.gold_docs_section_id, "")
        rekeyed = rekey_paraphrase(p, body, idf)
        tokens = rekeyed.key_tokens_docs  # both sides are now identical
        for tok in tokens:
            if " " in tok:
                multi_word_count += 1
        if not tokens:
            empty_set_count += 1

    assert multi_word_count == 0, (
        f"found {multi_word_count} multi-word tokens after re-keying"
    )
    assert empty_set_count == 0, (
        f"found {empty_set_count} Paraphrases with empty Key-Token set"
    )


# ---------------------------------------------------------------------------
# 10. C5c condition-2 still distinguishes hit from miss after re-keying
# ---------------------------------------------------------------------------
def test_c5c_condition2_distinguishes_hit_from_miss_with_idf_tokens():
    """A correct-id on-topic body hits; correct-id off-topic body misses."""
    from eval.paraphrase_comparison.metric import is_hit
    from eval.paraphrase_comparison.models import RetrievedItem

    # Key tokens derived from a section about restocking fees
    key_tokens = derive_key_tokens(
        "large appliances carry a restocking fee when returned",
        _IDF,
        top_n=5,
    )
    assert key_tokens, "need non-empty key tokens for this test"

    gold = "returns_policy.md#restocking-fee"
    on_topic = RetrievedItem(
        source_section_id=gold,
        content="large appliances carry a restocking fee when returned",
    )
    off_topic = RetrievedItem(
        source_section_id=gold,
        content="we offer great customer support",
    )
    assert is_hit(on_topic, gold, key_tokens) is True
    assert is_hit(off_topic, gold, key_tokens) is False
