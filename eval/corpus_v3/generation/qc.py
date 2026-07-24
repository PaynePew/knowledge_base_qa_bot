"""Deep module per Ousterhout. Public surface: ``GenerationQcVerdict``,
``check_generated_query``.

Generation-time QC gate for corpus v3 queries (issue #660, ADR-0045
Prerequisite 4). Unlike ``eval.paraphrase_comparison.generation.qc``'s Key
Token gate — which hard-rejects only all-stopword sets and *flags*
low-distinctiveness tokens for a human PR reviewer — every check here is a
hard reject. This gate runs immediately after ``gen_schema.to_query``
stitches a draft into a ``Query``, before any query ever reaches a human or
a committed file: a failure means "regenerate this query", not "flag for
review" (there is no corpus v3 human-review surface for individual queries
the way the v2 eval's PR review was).

Three checks, independent of each other (a query can fail more than one; all
failing reasons are collected before returning, so a batch generator can
report every problem at once rather than iterating one rejection at a time):

1. **generating_family recorded** — PRD #654 user story 5 / ADR-0045
   Prerequisite 3 requires "generating family recorded per query" so the
   verdict report can audit for model-family bias the way v2 could not.
2. **Key Tokens survive tokenisation** — an all-stopword Key Token set can
   never confirm a hit against retrieved content (same failure mode
   ``eval.paraphrase_comparison.generation.qc.check_key_tokens`` guards).
3. **Language matches detected script** — the zh slice's "own gate"
   (ADR-0045 Prerequisite 3: "a zh query slice with its own gates"). A query
   labelled ``language="zh"`` whose text ``detect_lang``s as ``"en"`` (or the
   reverse) is a generation-time labelling bug: the per-language report
   would silently misattribute it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from markdown_kb.app.indexer import detect_lang, tokenize

from ..query_schema import Query


@dataclass(frozen=True)
class GenerationQcVerdict:
    """Outcome of the generation QC gate for one stitched ``Query``.

    ``rejected`` is True when any check failed — the query must not enter a
    committed query file. ``reasons`` lists every failing check (not just the
    first), human-readable, for the generator's batch log.
    """

    query_id: str
    rejected: bool
    reasons: list[str] = field(default_factory=list)


def check_generated_query(query: Query) -> GenerationQcVerdict:
    """Run the three-check generation QC gate over one stitched ``Query``."""
    reasons: list[str] = []

    if not query.generating_family.strip():
        reasons.append(
            "generating_family is empty — every generated query must record its source"
        )

    if query.key_tokens:
        surviving = [tok for tok in query.key_tokens if tokenize(tok)]
        if not surviving:
            reasons.append("all key_tokens are stop-words after tokenisation")

    detected = detect_lang(query.text)
    if detected != query.language:
        reasons.append(
            f"language={query.language!r} but query text detect_lang's as {detected!r}"
        )

    return GenerationQcVerdict(
        query_id=query.query_id,
        rejected=bool(reasons),
        reasons=reasons,
    )
