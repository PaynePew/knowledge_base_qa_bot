"""Deep module per Ousterhout. Public surface: ``Query``, ``ScenarioStratum``,
``SCENARIO_STRATA``, ``OverlapStratum``, ``OVERLAP_STRATA``, ``Language``,
``LANGUAGES``, ``load_queries``, ``dump_queries``, ``query_to_dict``,
``query_from_dict``.

Stratified query file schema for the corpus v3 fair experiment (issue #659,
ADR-0045 Prerequisite 4: "query provenance stratified by query-document
lexical overlap ... and a zh query slice"; Prerequisite 4 also names hit@1 /
MRR alongside hit@3, which the sibling ``metrics`` module reports from these
strata). Every query carries three mandatory labels — scenario stratum,
lexical-overlap stratum, and language — so aggregation (``aggregation``
module) can run per stratum before any macro rollup, instead of hiding the
overlap-predicts-winner effect and per-language behaviour inside one pooled
number (PRD #654 user stories 6-7, 13).

An unlabeled or malformed query is a fail-fast data-corruption condition
(CODING_STANDARD §4.1), not a silently-dropped row: a query missing a stratum
label would otherwise either crash deep inside aggregation with a confusing
KeyError, or worse, silently be excluded from every stratum it should have
counted toward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import yaml

# ---------------------------------------------------------------------------
# Stratum vocabularies (ADR-0045 Prerequisite 4)
# ---------------------------------------------------------------------------
ScenarioStratum = Literal["factoid", "cross_doc", "version_conflict", "unanswerable"]
SCENARIO_STRATA: tuple[str, ...] = get_args(ScenarioStratum)

OverlapStratum = Literal["high_overlap", "low_overlap"]
OVERLAP_STRATA: tuple[str, ...] = get_args(OverlapStratum)

Language = Literal["en", "zh"]
LANGUAGES: tuple[str, ...] = get_args(Language)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Query:
    """One corpus v3 query with its three mandatory strata labels.

    ``gold_section_ids`` is a list rather than a single id — ADR-0045
    Prerequisite 3's corpus-neutral mapping table (issue #658) allows a wiki
    entity page to map 1:N onto Source Sections, and any one of the mapped
    Sections should count as a hit. It may be empty ONLY when
    ``scenario_stratum == "unanswerable"``: there is no correct retrieval for
    an unanswerable query, so nothing to check hit_at_k against. ``key_tokens``
    follows the same rule — required whenever ``gold_section_ids`` is
    non-empty, since a gold id without a Key Token cannot be scored by the
    content-overlap half of the hit condition (``metrics.is_hit``).
    """

    query_id: str
    text: str
    scenario_stratum: ScenarioStratum
    overlap_stratum: OverlapStratum
    language: Language
    gold_section_ids: list[str] = field(default_factory=list)
    key_tokens: list[str] = field(default_factory=list)
    generating_family: str = ""
    generation_notes: str = ""

    def __post_init__(self) -> None:
        _validate(self)


def _validate(query: Query) -> None:
    """Rule 1: every stratum label is one of the fixed, known values — Python's
    ``Literal`` is a type-checker hint only and enforces nothing at runtime, so
    this is the actual gate.
    Rule 2: an answerable query (any scenario other than "unanswerable")
    carries at least one gold Section id.
    Rule 3: a query with gold Section ids carries at least one Key Token.
    """
    if query.scenario_stratum not in SCENARIO_STRATA:
        raise ValueError(
            f"query {query.query_id!r}: scenario_stratum must be one of "
            f"{SCENARIO_STRATA}, got {query.scenario_stratum!r}"
        )
    if query.overlap_stratum not in OVERLAP_STRATA:
        raise ValueError(
            f"query {query.query_id!r}: overlap_stratum must be one of "
            f"{OVERLAP_STRATA}, got {query.overlap_stratum!r}"
        )
    if query.language not in LANGUAGES:
        raise ValueError(
            f"query {query.query_id!r}: language must be one of "
            f"{LANGUAGES}, got {query.language!r}"
        )
    if query.scenario_stratum != "unanswerable" and not query.gold_section_ids:
        raise ValueError(
            f"query {query.query_id!r}: scenario_stratum "
            f"{query.scenario_stratum!r} requires at least one gold_section_id"
        )
    if query.gold_section_ids and not query.key_tokens:
        raise ValueError(
            f"query {query.query_id!r}: has gold_section_ids but no key_tokens"
        )


# ---------------------------------------------------------------------------
# Public API — (de)serialisation
# ---------------------------------------------------------------------------
def query_to_dict(query: Query) -> dict:
    """Serialise ``query`` to a plain dict (the YAML entry shape)."""
    return {
        "query_id": query.query_id,
        "text": query.text,
        "scenario_stratum": query.scenario_stratum,
        "overlap_stratum": query.overlap_stratum,
        "language": query.language,
        "gold_section_ids": list(query.gold_section_ids),
        "key_tokens": list(query.key_tokens),
        "generating_family": query.generating_family,
        "generation_notes": query.generation_notes,
    }


def query_from_dict(entry: dict) -> Query:
    """Parse one YAML entry into a validated ``Query``.

    Raises ``ValueError`` naming every missing required field at once (rather
    than a bare ``KeyError`` on the first one) so a batch-authored query file
    with several malformed entries reports each cleanly.
    """
    required = {"query_id", "text", "scenario_stratum", "overlap_stratum", "language"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"query entry missing required field(s): {sorted(missing)}")
    return Query(
        query_id=entry["query_id"],
        text=entry["text"],
        scenario_stratum=entry["scenario_stratum"],
        overlap_stratum=entry["overlap_stratum"],
        language=entry["language"],
        gold_section_ids=list(entry.get("gold_section_ids", [])),
        key_tokens=list(entry.get("key_tokens", [])),
        generating_family=entry.get("generating_family", ""),
        generation_notes=entry.get("generation_notes", ""),
    )


def dump_queries(queries: list[Query], path: Path) -> None:
    """Serialise ``queries`` to YAML at ``path`` (round-trips with :func:`load_queries`)."""
    data = {"queries": [query_to_dict(q) for q in queries]}
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def load_queries(path: Path) -> list[Query]:
    """Parse a stratified query file into validated ``Query`` objects.

    Raises ``ValueError`` on a missing required field or an invalid/missing
    stratum label (fail-fast per CODING_STANDARD §4.1) rather than silently
    dropping or mis-bucketing a malformed query.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [query_from_dict(entry) for entry in data.get("queries", [])]
