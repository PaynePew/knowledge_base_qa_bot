"""Deep module per Ousterhout. Public surface: ``QcVerdict``, ``build_idf``, ``check_key_tokens``, ``derive_key_tokens``.

Key-Token QC gate for generated Paraphrases (PRD #100, issue #102, #139).

A Paraphrase's dual-side Key Tokens (``key_tokens_docs`` ∪ ``key_tokens_wiki``)
are what the C5c hit metric overlaps against retrieved content — so a Paraphrase
whose Key Tokens are all stop-words, or all corpus-ubiquitous filler, would score
a "hit" on almost any retrieved Section and silently corrupt the comparison. This
gate is the programmatic half of the QC step; the flagged entries it surfaces are
the input to the human PR review (the issue's "place for human review").

Two checks, in order:

  1. **All-stopword rejection** (hard fail): after the markdown_kb tokeniser
     strips stop-words, an empty token set means every Key Token was a stop-word.
     The metric can never confirm content with such a set — reject outright.
  2. **Low-distinctiveness flag** (soft, human-review): a token whose IDF over the
     corpus is below ``min_idf`` is so common it barely discriminates one Section
     from another. These are flagged (not rejected) so a human decides on PR
     whether the Key Token set still has enough distinctive signal.

The tokeniser is markdown_kb's so "is a stop-word" and "is a token" use the exact
same convention as the BM25 corpus and the C5c metric (ADR-0002 shared tokeniser).

Issue #139 adds ``derive_key_tokens``, which replaces the LLM-emitted Key Tokens
with a deterministic derivation from the Gold Section body: tokenise the body,
rank surviving tokens by corpus IDF, and take the top-N most distinctive single
tokens. Number-token policy: pure numeric tokens (e.g. "30") are excluded because
they appear identically in many sections ("30 days", "30%") and are unreliable as
section discriminators. Tokens shorter than 2 chars are also dropped.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

# ADR-0002: reuse markdown_kb's tokeniser so QC's notion of "token" / "stop-word"
# matches the BM25 corpus and the C5c metric exactly.
from markdown_kb.app.indexer import tokenize

# Below this IDF a token is "corpus-ubiquitous filler" — present in so many
# Sections it barely discriminates. Tuned for the eval corpus's ~40 Sections:
# a token in >~60% of Sections falls under this bar. Surfaced for human review,
# never auto-rejected.
DEFAULT_MIN_IDF = 0.5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QcVerdict:
    """Outcome of the Key-Token QC gate for one Paraphrase.

    ``rejected`` is True when the gate hard-fails (all-stopword Key Tokens) — the
    Paraphrase must not enter ``queries.yaml``. ``flagged_tokens`` lists the
    low-distinctiveness Key Tokens a human should review on PR; a non-empty list
    with ``rejected=False`` means "admit, but look at these". ``reasons`` carries
    human-readable notes for the PR-review surface.
    """

    paraphrase_id: str
    rejected: bool
    flagged_tokens: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IDF model
# ---------------------------------------------------------------------------
def build_idf(documents: list[str]) -> dict[str, float]:
    """Compute a smoothed IDF score per token over ``documents``.

    Each document is one Section body (or Wiki Page body). IDF uses the standard
    ``log((N + 1) / (df + 1)) + 1`` smoothing so a token present in every document
    still scores a small positive value rather than zero. The token set per
    document is the markdown_kb tokeniser output (stop-words already removed), so
    stop-words never appear in the IDF table.
    """
    n = len(documents)
    if n == 0:
        return {}
    doc_freq: Counter[str] = Counter()
    for doc in documents:
        for tok in set(tokenize(doc)):
            doc_freq[tok] += 1
    return {tok: math.log((n + 1) / (df + 1)) + 1.0 for tok, df in doc_freq.items()}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def check_key_tokens(
    paraphrase_id: str,
    key_tokens: list[str],
    idf: dict[str, float],
    *,
    min_idf: float = DEFAULT_MIN_IDF,
) -> QcVerdict:
    """Run the two-stage Key-Token QC gate for one Paraphrase's token set.

    ``key_tokens`` is the union of the dual-side Key Tokens. ``idf`` comes from
    ``build_idf`` over the corpus. Check order (see module docstring):

      1. All-stopword rejection — tokenise the Key Tokens; an empty result means
         every Key Token was a stop-word. Hard reject.
      2. Low-distinctiveness flag — any surviving token whose IDF is below
         ``min_idf`` (or absent from the IDF table, i.e. it never appears in the
         corpus body) is flagged for human review, not rejected.
    """
    surviving = [t for t in (tok.lower() for tok in key_tokens) if tokenize(t)]
    if not surviving:
        return QcVerdict(
            paraphrase_id=paraphrase_id,
            rejected=True,
            reasons=["all Key Tokens are stop-words after tokenisation"],
        )

    flagged: list[str] = []
    reasons: list[str] = []
    for tok in surviving:
        # A token absent from the IDF table never appears in any Section body —
        # maximally distinctive in one sense, but it also means the metric can
        # never match it against retrieved content, so flag it for review.
        score = idf.get(tok)
        if score is None:
            flagged.append(tok)
            reasons.append(f"'{tok}' absent from corpus bodies (cannot match content)")
        elif score < min_idf:
            flagged.append(tok)
            reasons.append(f"'{tok}' low distinctiveness (idf={score:.3f} < {min_idf})")

    return QcVerdict(
        paraphrase_id=paraphrase_id,
        rejected=False,
        flagged_tokens=flagged,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Deterministic Key Token derivation (issue #139)
# ---------------------------------------------------------------------------
_MIN_TOKEN_LEN = 2  # single chars are never informative Key Tokens


def derive_key_tokens(
    section_body: str,
    idf: dict[str, float],
    top_n: int = 10,
) -> list[str]:
    """Derive the top-N most distinctive single tokens from a Gold Section body.

    Pipeline (issue #139):

      1. Tokenise ``section_body`` with the markdown_kb tokeniser (ADR-0002) —
         stop-words and non-alphanumeric chars are stripped automatically.
      2. Filter out tokens shorter than ``_MIN_TOKEN_LEN`` (2) — single chars
         carry no retrieval signal.
      3. Filter out pure numeric tokens (e.g. "30", "15") — numbers appear
         identically across many sections ("30 days", "30%", "fifteen percent as
         "15") and are unreliable section discriminators. This is the documented
         number-token policy (issue #139 AC-4).
      4. Retain only tokens present in the IDF table — tokens absent from the
         corpus body cannot match against retrieved content (QC gate logic).
      5. Rank by IDF descending (higher IDF = more distinctive for this corpus).
      6. De-duplicate while preserving rank order.
      7. Return the top-N.

    Returns an empty list when ``section_body`` has no surviving tokens after
    filtering, rather than raising — the caller (``rekey_paraphrase``) surfaces
    this via the QC gate.
    """
    tokens = tokenize(section_body)

    seen: set[str] = set()
    candidates: list[tuple[str, float]] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        # Drop single-char tokens
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        # Drop pure numeric tokens (documented number-token policy)
        if tok.isdigit():
            continue
        score = idf.get(tok)
        if score is None:
            # Token not in corpus IDF table → cannot match retrieved content; skip.
            continue
        candidates.append((tok, score))

    # Sort by IDF descending (most distinctive first); stable for equal scores.
    candidates.sort(key=lambda t: -t[1])
    return [tok for tok, _ in candidates[:top_n]]
