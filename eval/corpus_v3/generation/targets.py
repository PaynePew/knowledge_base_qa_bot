"""Deep module per Ousterhout. Public surface: ``GenerationTarget``,
``derive_generation_targets``, ``sha256_order``, ``sample_targets``.

Deterministic generation-target derivation for the corpus v3 query generator
(issue #672, ``generation/SPEC.md``). A ``GenerationTarget`` names WHICH gold
Section(s) (or, for ``unanswerable``, which near-miss distractor Section) one
generated query should be written against, and which scenario stratum that
target belongs to — the seam the generator script iterates over before ever
calling an LLM.

Targets are derived from ``build_corpus.ADVERSARIAL_GROUPS`` (issue #661) —
the corpus's single source of truth — rather than re-authored here, per
``CORPUS.md``'s "Power-analysis reconciliation": 9 groups is a floor of
independent topical seeds a generator multiplies into many queries via
paraphrase/family/overlap variation, not a claim that one query per group
reaches n=909. ``sample_targets`` is how the generator does that multiplying:
cycling deterministically through a stratum's (small) target list as many
times as the configured count requires.

Class-to-stratum mapping (the corpus's own design intent, ``CORPUS.md``):

- ``redundancy``   -> ``factoid`` (one target per individual gold Section)
                      AND ``cross_doc`` (one target unifying the group's
                      near-duplicate Sections — the dedup synthesis case).
- ``contradiction`` -> ``factoid`` (per individual Section) AND ``cross_doc``
                      (the conflict-comparison case — this is the
                      contradiction-leak axis's home ground).
- ``version_evolution`` -> ``version_conflict`` only. The group's own
                      ``gold_section_ids`` already names only the newest
                      version (``build_corpus`` enforces this), so no
                      additional filtering happens here.
- every group also contributes one ``unanswerable`` target: its non-gold
  Section(s) (the superseded versions for ``version_evolution``, or any
  Section not itself the group's sole gold anchor) stand in as the near-miss
  distractor text a generated unanswerable query is written to resemble
  (``generation/SPEC.md`` § Overlap stratum) without actually answering.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..build_corpus import AdversarialGroup
from ..query_schema import ScenarioStratum


@dataclass(frozen=True)
class GenerationTarget:
    """One (scenario stratum, corpus anchor) pair a generated query is written
    against. ``gold_section_ids`` is empty only for ``unanswerable`` — mirrors
    ``Query``'s own invariant (``query_schema._validate``). ``reference_ids`` /
    ``reference_text`` are always populated (gold text for answerable strata,
    distractor text for ``unanswerable``) so the overlap stratum
    (``overlap.classify_overlap_stratum``) always has something to measure
    against, per ``generation/SPEC.md``.
    """

    scenario_stratum: ScenarioStratum
    group_id: str
    heading: str
    gold_section_ids: list[str] = field(default_factory=list)
    reference_ids: list[str] = field(default_factory=list)
    reference_text: str = ""


# ---------------------------------------------------------------------------
# Public API — derivation
# ---------------------------------------------------------------------------
def derive_generation_targets(
    groups: list[AdversarialGroup],
) -> dict[ScenarioStratum, list[GenerationTarget]]:
    """Bucket every ``AdversarialGroup`` into its scenario-stratum target(s).

    Returns a dict with all four ``ScenarioStratum`` keys always present
    (possibly with an empty list) so a caller never hits a ``KeyError`` for a
    stratum this corpus happens not to feed.
    """
    buckets: dict[ScenarioStratum, list[GenerationTarget]] = {
        "factoid": [],
        "cross_doc": [],
        "version_conflict": [],
        "unanswerable": [],
    }
    for group in groups:
        by_id = {s.source_id: s for s in group.sections}

        if group.adversarial_class in ("redundancy", "contradiction"):
            for gold_id in group.gold_section_ids:
                section = by_id[gold_id]
                buckets["factoid"].append(
                    GenerationTarget(
                        scenario_stratum="factoid",
                        group_id=group.group_id,
                        heading=section.heading,
                        gold_section_ids=[gold_id],
                        reference_ids=[gold_id],
                        reference_text=section.body,
                    )
                )
            if len(group.gold_section_ids) >= 2:
                buckets["cross_doc"].append(
                    GenerationTarget(
                        scenario_stratum="cross_doc",
                        group_id=group.group_id,
                        heading=group.wiki_title,
                        gold_section_ids=list(group.gold_section_ids),
                        reference_ids=list(group.gold_section_ids),
                        reference_text="\n\n".join(
                            by_id[gid].body for gid in group.gold_section_ids
                        ),
                    )
                )
        elif group.adversarial_class == "version_evolution":
            newest_ids = list(group.gold_section_ids)
            buckets["version_conflict"].append(
                GenerationTarget(
                    scenario_stratum="version_conflict",
                    group_id=group.group_id,
                    heading=group.wiki_title,
                    gold_section_ids=newest_ids,
                    reference_ids=newest_ids,
                    reference_text="\n\n".join(by_id[gid].body for gid in newest_ids),
                )
            )
        else:  # pragma: no cover - AdversarialClass is a closed Literal
            raise ValueError(f"unknown adversarial_class: {group.adversarial_class!r}")

        distractor_ids = [
            sid for sid in by_id if sid not in group.gold_section_ids
        ] or list(group.gold_section_ids)
        buckets["unanswerable"].append(
            GenerationTarget(
                scenario_stratum="unanswerable",
                group_id=group.group_id,
                heading=by_id[distractor_ids[0]].heading,
                gold_section_ids=[],
                reference_ids=distractor_ids,
                reference_text="\n\n".join(by_id[sid].body for sid in distractor_ids),
            )
        )
    return buckets


# ---------------------------------------------------------------------------
# Public API — deterministic ordering / sampling (production-isolation
# precedent, ``gen_schema.py``'s docstring: independently defined here rather
# than importing ``eval.paraphrase_comparison``'s sha256_order, whose type is
# pinned to that package's own ``GoldSection``).
# ---------------------------------------------------------------------------
def sha256_order(targets: list[GenerationTarget], seed: str) -> list[GenerationTarget]:
    """Return ``targets`` ordered by ``sha256(seed:group_id:scenario_stratum)``
    — a byte-stable shuffle independent of Python's salted ``hash()`` /
    ``PYTHONHASHSEED``, mirroring
    ``eval.paraphrase_comparison.generation.sampling.sha256_order``'s pattern.
    """
    return sorted(targets, key=lambda t: _digest(seed, t))


def sample_targets(
    targets: list[GenerationTarget], seed: str, count: int
) -> list[GenerationTarget]:
    """Deterministically draw exactly ``count`` targets, CYCLING through the
    sha256-ordered pool when ``count`` exceeds its size.

    Unlike ``eval.paraphrase_comparison``'s ``sample_sections`` (which caps at
    the pool size — one query per Paraphrase Type per section), corpus v3's
    per-stratum target pool is intentionally small (``CORPUS.md``'s "3
    distinct instances per class" floor) while the power-sized count is large
    (n=909) — the generator draws MANY variants per target. Cycling is
    order-stable: variant ``k`` of a full cycle always lands on
    ``ordered[k % len(ordered)]``, so a re-run reproduces the exact same
    (target, variant-index) assignment for every query id.

    Raises ``ValueError`` if ``targets`` is empty and ``count > 0`` (no pool
    to cycle through) or if ``count`` is negative.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count!r}")
    if count > 0 and not targets:
        raise ValueError("sample_targets: targets is empty but count > 0")
    ordered = sha256_order(targets, seed)
    if not ordered:
        return []
    return [ordered[i % len(ordered)] for i in range(count)]


def _digest(seed: str, target: GenerationTarget) -> str:
    # ``reference_ids`` is included, not just ``group_id``/``scenario_stratum``
    # -- a redundancy/contradiction group contributes MULTIPLE ``factoid``
    # targets sharing the same (group_id, scenario_stratum) pair (one per
    # gold Section), and without a per-target-unique component in the key,
    # Python's stable sort would leave those ties in their original list
    # order regardless of ``seed`` -- silently defeating the "different seed,
    # different order" contract ``sample_targets`` relies on for variety.
    key = (
        f"{seed}:{target.group_id}:{target.scenario_stratum}:"
        f"{','.join(target.reference_ids)}"
    )
    return hashlib.sha256(key.encode()).hexdigest()
