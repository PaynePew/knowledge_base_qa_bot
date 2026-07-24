"""Shallow module per Ousterhout. Public surface: ``QueryDraft``, ``to_query``.

Structured-output schema for the corpus v3 query generator (issue #660,
ADR-0045 Prerequisite 4). ``QueryDraft`` is the Pydantic schema an LLM (or a
human author, for the human-written slice ADR-0045 Prerequisite 3 allows in
place of a second model family) fills in: the query text and the Key Tokens
its answer must contain. The generator owns every bookkeeping field a model
could hallucinate wrong — ``query_id``, ``scenario_stratum``, ``language``,
``gold_section_ids``, ``generating_family`` — plus ``overlap_stratum``, which
is *computed*, not authored (``generation.overlap.classify_overlap_stratum``)
so it reflects the query as actually written rather than a model's own
(unreliable) self-assessment. ``to_query`` stitches these together into the
canonical ``Query`` (mirrors
``eval.paraphrase_comparison.generation.gen_schema``'s
``ParaphraseDraft``/``to_paraphrase`` split, independently defined here per
the corpus v3 package's production-isolation precedent —
``eval/corpus_v3/models.py``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..query_schema import Language, OverlapStratum, Query, ScenarioStratum


class QueryDraft(BaseModel):
    """The LLM- (or human-) supplied portion of a corpus v3 Query.

    ``key_tokens`` has no ``min_length`` here — an ``unanswerable`` query
    legitimately has none, and ``Query.__post_init__`` already enforces "gold
    Section ids require Key Tokens" as the single source of truth for that
    invariant (``to_query`` does not duplicate it).
    """

    text: str = Field(description="The generated query text, in the target language.")
    key_tokens: list[str] = Field(
        default_factory=list,
        description="Distinctive tokens the correct answer must contain; empty for unanswerable queries.",
    )
    generation_notes: str = Field(
        default="",
        description="Optional note: targeted sub-fact, contradiction pair, version pinned, etc.",
    )


def to_query(
    draft: QueryDraft,
    *,
    query_id: str,
    scenario_stratum: ScenarioStratum,
    language: Language,
    overlap_stratum: OverlapStratum,
    gold_section_ids: list[str],
    generating_family: str,
) -> Query:
    """Stitch a generator-owned id/strata/gold/family onto an LLM ``QueryDraft``.

    ``overlap_stratum`` is expected to already be the output of
    ``generation.overlap.classify_overlap_stratum`` run against ``draft.text``
    — this function does not compute it, keeping the deterministic-derivation
    step visible at the call site rather than hidden inside the stitcher.
    Raises whatever ``Query.__post_init__`` raises (ValueError) if the
    resulting combination is invalid (e.g. an answerable ``scenario_stratum``
    with empty ``gold_section_ids``) — no separate validation here.
    """
    return Query(
        query_id=query_id,
        text=draft.text,
        scenario_stratum=scenario_stratum,
        overlap_stratum=overlap_stratum,
        language=language,
        gold_section_ids=list(gold_section_ids),
        key_tokens=list(draft.key_tokens),
        generating_family=generating_family,
        generation_notes=draft.generation_notes,
    )
