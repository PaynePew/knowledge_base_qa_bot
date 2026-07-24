"""Deep module per Ousterhout. Public surface: ``DEFAULT_OVERLAP_THRESHOLD``,
``lexical_overlap_ratio``, ``classify_overlap_stratum``.

Query-document lexical-overlap stratum computation, done **at generation
time** rather than left for a downstream pass (issue #660, ADR-0045
Prerequisite 3: "query provenance stratified by query-document lexical
overlap — LLM-paraphrased queries depress overlap and favor dense (Ren et
al. 2022, DPR's SQuAD finding)"). Computing this at generation time, from
the query text as actually generated, is what prevents the v2 eval's failure
mode: paraphrase-only same-family generation silently produced low-overlap
queries without ever labelling them as such, so the overlap-predicts-winner
effect stayed hidden inside the pooled numbers (PRD #654 user story 6).

Uses the shared ``markdown_kb`` tokeniser (ADR-0002: one tokeniser
convention for BM25, the C5c hit metric, and now this stratum) so "overlap"
means the same thing here as it means to the BM25 corpus and
``eval.corpus_v3.metrics.is_hit`` — CJK text tokenises via the same
character-bigram convention as English word tokens, so the ratio is
comparable across both the en and zh slices without a separate code path.
"""

from __future__ import annotations

from markdown_kb.app.indexer import tokenize

from ..query_schema import OverlapStratum

# Empirically-motivated split point: a query sharing at least half its
# distinct tokens with its gold reference text(s) is "high overlap" (the
# regime dense retrieval was tuned to look easy in); below that, "low
# overlap" (an LLM-paraphrased or genuinely reworded query — the regime the
# v2 eval's paraphrase-only generation under-sampled). Not calibrated against
# a labelled fixture (no corpus v3 data exists yet, per this issue's scope);
# a downstream corpus-build slice may recalibrate once real generated query
# text exists to inspect.
DEFAULT_OVERLAP_THRESHOLD = 0.5


def lexical_overlap_ratio(query_text: str, reference_texts: list[str]) -> float:
    """Fraction of ``query_text``'s distinct tokens also present in ``reference_texts``.

    A containment ratio (query tokens found in the reference union, divided
    by the query's own distinct token count) rather than a symmetric Jaccard
    ratio — this is the DPR/SQuAD-style "does the query's vocabulary survive
    in the passage" measure the overlap-predicts-winner literature uses,
    which stays sensitive to a fully-paraphrased query (all query tokens
    replaced) even against a long reference passage.

    ``reference_texts`` is normally the query's gold Section body/bodies; for
    an ``unanswerable`` query (no gold Sections) the caller should pass the
    near-miss distractor Section(s) the query was written to resemble, so the
    stratum still carries signal instead of trivially resolving to 0.0 — see
    the generation spec (``SPEC.md``).

    Raises ``ValueError`` if ``query_text`` tokenises to nothing (fail-fast
    per CODING_STANDARD §4.1 — an empty-token query cannot be scored, and
    silently returning 0.0 would be indistinguishable from "real, measured
    zero overlap").
    """
    query_tokens = set(tokenize(query_text))
    if not query_tokens:
        raise ValueError(
            "lexical_overlap_ratio: query_text has no tokens after tokenisation"
        )
    reference_tokens: set[str] = set()
    for text in reference_texts:
        reference_tokens.update(tokenize(text))
    if not reference_tokens:
        return 0.0
    return len(query_tokens & reference_tokens) / len(query_tokens)


def classify_overlap_stratum(
    query_text: str,
    reference_texts: list[str],
    *,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> OverlapStratum:
    """Classify a query's overlap stratum from its text and reference passage(s).

    ``>= threshold`` is ``"high_overlap"``; below it, ``"low_overlap"`` — the
    two values ``eval.corpus_v3.query_schema.OVERLAP_STRATA`` defines. Callers
    assign the result directly to ``Query.overlap_stratum``
    (``generation.gen_schema.to_query`` does this).
    """
    ratio = lexical_overlap_ratio(query_text, reference_texts)
    return "high_overlap" if ratio >= threshold else "low_overlap"
