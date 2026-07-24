"""Deep module per Ousterhout. Public surface: ``StratumCount``,
``build_deviations``, ``build_metadata``, ``render_query_artifact``.

Committed query-set artifact assembly for the corpus v3 query generator
(issue #672 AC 1: "deviations (if any) stated in the artifact header, not
silent"). Kept separate from ``generate_queries.py``'s orchestration so the
header-building and deviation logic is unit-testable on canned counts,
without ever invoking an LLM (mirrors ``content_axes.py``'s pure-math /
``run_verdict.py``'s orchestration split already established in this
package).
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from ..query_schema import Query, query_to_dict


@dataclass(frozen=True)
class StratumCount:
    """Target vs actual query count for one (scenario stratum, language)
    cell, plus how many drafts the QC gate rejected while reaching it."""

    scenario_stratum: str
    language: str
    target: int
    actual: int
    qc_rejected: int


def build_deviations(counts: list[StratumCount]) -> list[str]:
    """One human-readable line per cell where ``actual != target`` — never
    silent (issue #672 AC 1). Returns an empty list when every cell hit its
    target exactly."""
    return [
        f"{c.scenario_stratum}/{c.language}: target={c.target}, actual={c.actual} "
        f"({'short by ' + str(c.target - c.actual) if c.actual < c.target else 'over by ' + str(c.actual - c.target)})"
        for c in counts
        if c.actual != c.target
    ]


def build_metadata(
    *,
    counts: list[StratumCount],
    family_a_model: str,
    family_b_source: str,
    embedding_family: str,
    generated_at: str,
    cost_usd: float | None,
    prompt_template_version: int,
) -> dict:
    """Assemble the artifact's metadata block.

    ``family_b_source`` documents Family B's identity per query set (a second
    model family name, or ``"human"`` for the human-written-slice
    alternative ``generation/SPEC.md`` sanctions) — never omitted, per
    ADR-0045 Prerequisite 3's "generating family recorded per query"
    requirement applied at the artifact level too. ``embedding_family`` is
    recorded for audit (arm B's embedding model) but is NOT compared against
    ``family_a_model`` here — ``generation/SPEC.md`` explicitly names Family A
    as "an OpenAI model, the family already in use ... elsewhere", the same
    family arm B's embeddings use; the multi-family requirement is that
    Family B (or a human slice) differs from Family A, not that Family A
    itself must avoid the embedding family.

    Raises ``ValueError`` if ``family_b_source`` equals ``family_a_model``
    (case-insensitively) — ADR-0045 Prerequisite 3's whole point is a second,
    DIFFERENT source; the one documented exception is ``family_b_source ==
    "human"``, always distinct from a model family name by construction — or
    if ``family_b_source`` is blank.
    """
    if not family_b_source.strip():
        raise ValueError("build_metadata: family_b_source must not be empty")
    if family_b_source.strip().lower() == family_a_model.strip().lower():
        raise ValueError(
            f"build_metadata: family_b_source {family_b_source!r} matches "
            f"family_a_model {family_a_model!r} — ADR-0045 Prerequisite 3 "
            "requires a second, different generating family (or 'human')"
        )
    return {
        "generator_family_a": family_a_model,
        "generator_family_b": family_b_source,
        "embedding_family_avoided": embedding_family,
        "generated_at": generated_at,
        "prompt_template_version": prompt_template_version,
        "cost_usd": cost_usd,
        "counts": [
            {
                "scenario_stratum": c.scenario_stratum,
                "language": c.language,
                "target": c.target,
                "actual": c.actual,
                "qc_rejected": c.qc_rejected,
            }
            for c in counts
        ],
        "deviations": build_deviations(counts),
    }


def render_query_artifact(queries: list[Query], *, metadata: dict) -> str:
    """Serialise ``queries`` + ``metadata`` to the committed YAML shape.

    Round-trips through ``query_schema.load_queries`` (which reads only the
    ``queries`` key and ignores unknown top-level keys), so this stays
    compatible with the existing loader without changing its contract.

    Raises ``ValueError`` on duplicate ``query_id``s: downstream joins key on
    ``query_id`` and ``load_queries`` performs no uniqueness check, so a
    duplicate written here would silently inflate n and corrupt pairing.
    """
    seen: dict[str, int] = {}
    for q in queries:
        seen[q.query_id] = seen.get(q.query_id, 0) + 1
    duplicates = sorted(qid for qid, n in seen.items() if n > 1)
    if duplicates:
        raise ValueError(
            f"render_query_artifact: duplicate query_ids {duplicates[:5]}"
            f"{' …' if len(duplicates) > 5 else ''} "
            f"({len(duplicates)} id(s) appear more than once)"
        )
    data = {
        "metadata": metadata,
        "queries": [query_to_dict(q) for q in queries],
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
