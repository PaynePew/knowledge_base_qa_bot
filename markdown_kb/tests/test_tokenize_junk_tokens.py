"""Tokenizer junk-token regression tests (issue #252).

The negative-case eval (#249) showed that ``tokenize`` emitted junk tokens that
created spurious BM25 hits, letting out-of-scope queries clear the Cannot Confirm
threshold:
  - ``"What's ..."`` → bare ``'s'`` matched ``customer's`` in the corpus.
  - ``"... invest in ..."`` → unfiltered ``'in'`` matched body text.

Fix: drop length-1 Latin tokens (contraction/possessive debris) in the non-CJK
path, and add the zero-signal prepositions to STOP_WORDS. Single-digit tokens and
CJK unigrams must be preserved.
"""

from __future__ import annotations

from app.indexer import tokenize


def test_possessive_s_dropped():
    """Possessive 's leaves no bare 's' junk token."""
    assert "s" not in tokenize("What's the refund policy?")


def test_contraction_clitic_debris_dropped():
    """Length-1 Latin contraction fragments are filtered."""
    toks = tokenize("I don't think it's ready, we'd wait")
    assert "t" not in toks
    assert "s" not in toks
    assert "d" not in toks


def test_common_prepositions_are_stopwords():
    """Zero-signal prepositions carry no retrieval signal and are filtered."""
    toks = tokenize("invest in the stock market on Monday at noon by Tuesday with cash")
    for prep in ("in", "on", "at", "by", "with"):
        assert prep not in toks


def test_single_digit_preserved():
    """Length-1 DIGITS stay (quantities/years carry signal); only single letters go."""
    assert "5" in tokenize("refund within 5 days")


def test_multichar_content_words_preserved():
    """Real content words are untouched — no over-filtering."""
    toks = tokenize("refund shipping account password")
    assert {"refund", "shipping", "account", "password"} <= set(toks)


def test_cjk_single_char_preserved():
    """The length-1-Latin filter must not touch the CJK unigram fallback."""
    assert "退" in tokenize("退")
