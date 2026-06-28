"""Deep module per Ousterhout. Public surface: ``retrieve_and_gate``, ``reciprocal_rank_fusion``, ``evaluate_or_gate``, ``RRF_K``, ``DEFAULT_CANDIDATE_DEPTH``, ``DEFAULT_TOP_K``.

Hybrid Retrieval (Stack C) — the retrieval core (slice S2, ADR-0018 / #312).

This is the recall-union heart of the Phase 13 Hybrid stack: it runs BOTH arms
over the SAME curated ``wiki/`` Section corpus — BM25 (``markdown_kb``) and the
dense-over-wiki index (S1, ``hybrid_kb.dense_index``) — overfetching a deep
candidate pool per arm, applies the pre-LLM Cannot Confirm **OR-gate** on each
arm's NATIVE score before fusion, then fuses the two ranked lists with
**Reciprocal Rank Fusion (RRF, K=60)** and returns the top-k wiki Sections.

There is **no LLM call at this layer** — the output is ranked ``Section`` objects
(which satisfy the ``CitableContent`` protocol) plus a gate verdict. The
LLM-facing ``hybrid_kb.query()`` surface that consumes this is S3 (#313).

Three pure pieces, each unit-tested with synthetic inputs (no embeddings, no
LLM):

  * :func:`reciprocal_rank_fusion` — pure RRF over two ranked Section lists. A
    Section id present in both arms sums its reciprocal-rank contributions
    ``1/(K+rank)``; ids are deduped over the shared id space; the fused list is
    truncated to the final ``top_k``. RRF uses RANK only — never an arm's raw
    score — so the two arms' incomparable native scales (BM25 magnitude vs FAISS
    distance) never need reconciling.

  * :func:`evaluate_or_gate` — pure pre-LLM OR-gate over the two arms' native top
    scores + the query language. Proceed if the BM25 arm clears its per-language
    ``KB_SCORE_THRESHOLD`` **OR** the dense arm clears its calibrated distance
    ceiling; otherwise Cannot Confirm (``below_threshold``). OR is required — AND
    would defeat the recall-union purpose (a Section one arm ranks low but the
    other ranks high must still surface). The RRF fused score is **never** used
    as a relevance threshold — it is not a calibrated relevance magnitude
    (ADR-0018, §4.3 gate-parity extended).

  * :func:`retrieve_and_gate` — composes the two arms + gate + fusion into the
    deep module's public retrieval surface.

ADR-0018 blessed cross-app reuse (the SAME recorded coupling ``vector_rag`` and
``hybrid_kb.dense_index`` already have on ``markdown_kb`` leaves): the two
eval-calibrated thresholds are reused **by reference**, NOT redefined here —

  * BM25 arm → ``markdown_kb.app.retrieval._SCORE_THRESHOLD`` (en) and
    ``_SCORE_THRESHOLD_ZH`` (zh, #261), selected per query language via the
    consolidated ``detect_lang`` classifier (#285) so routing never drifts from
    the dense arm's language filter.
  * dense arm → ``vector_rag.app.retrieval._max_rag_distance()`` (the accessor
    for the calibrated ``_KB_RAG_DISTANCE_THRESHOLD_DEFAULT`` = 1.1, honouring
    the ``KB_RAG_DISTANCE_THRESHOLD`` override exactly as Stack B's own gate
    does).

Both are read THROUGH their owning module at call time so a test that
monkeypatches them (the established ``_SCORE_THRESHOLD`` / env patterns) is
honoured, and **no new threshold is introduced** (ADR-0018 invariant).
"""

from __future__ import annotations

from collections.abc import Sequence

# ADR-0018 blessed cross-app reuse. Imported as modules (not as bound values) so
# the per-language BM25 thresholds and the dense distance ceiling are resolved
# THROUGH their owning module at gate-call time — honouring test monkeypatching
# and the import-time / env-time resolution each owner already does. No threshold
# is redefined here (ADR-0018 "no new threshold" invariant).
import markdown_kb.app.indexer as _bm25_indexer
import markdown_kb.app.retrieval as _bm25_gate
import vector_rag.app.retrieval as _dense_gate
from markdown_kb.app.grounding import GroundingOutcome
from markdown_kb.app.indexer import Section, detect_lang

from . import dense_index

__all__ = [
    "retrieve_and_gate",
    "reciprocal_rank_fusion",
    "evaluate_or_gate",
    "RRF_K",
    "DEFAULT_CANDIDATE_DEPTH",
    "DEFAULT_TOP_K",
]

# ---------------------------------------------------------------------------
# Constants (ADR-0018 / #312)
# ---------------------------------------------------------------------------
# RRF smoothing constant. K=60 is the canonical Cormack et al. value the ADR
# fixes; it damps the influence of any single arm's top rank so the fusion is
# robust to one arm's spurious #1.
RRF_K = 60

# Per-arm overfetch depth — the deep candidate pool each arm retrieves BEFORE
# fusion. Decoupled from the final cutoff (DEFAULT_TOP_K): RRF needs a deep pool
# to rescue a Section one arm ranked low (the recall-union / 補漏 purpose). A
# generous default effectively scans the small wiki corpus.
DEFAULT_CANDIDATE_DEPTH = 50

# Final cutoff — how many fused Sections flow downstream to synthesis. Matches
# the top-3 the Wiki and RAG stacks return, so prompt/citation behaviour is
# identical across the three stacks.
DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# RRF fusion (pure function over two ranked lists — AC2)
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_a: Sequence[Section],
    ranked_b: Sequence[Section],
    *,
    k: int = RRF_K,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[Section, float]]:
    """Fuse two rank-ordered Section lists with Reciprocal Rank Fusion.

    Pure function — no I/O, no embeddings, no LLM. Each input list is assumed
    already rank-ordered (best first). A Section's fused score is the sum over
    the arms in which it appears of ``1 / (k + rank)`` with ``rank`` 1-based
    (so the #1 item contributes ``1/(k+1)``). A Section id present in BOTH arms
    therefore sums its two contributions — the natural RRF dedup over the shared
    Section-id space (ADR-0018 same-corpus invariant makes the ids comparable).

    Only RANK is used, never an arm's raw score: BM25 magnitude and FAISS
    distance are incomparable scales, and RRF deliberately needs neither.

    Returns the fused ``(Section, fused_score)`` pairs sorted by fused score
    (descending), truncated to ``top_k``. Ties keep first-seen order (arm A
    before arm B, each in its own rank order) for determinism. When a Section id
    appears in both arms the Section object kept is arm A's (they are equal under
    the 1:1 id invariant).
    """
    fused_scores: dict[str, float] = {}
    section_by_id: dict[str, Section] = {}

    for ranked in (ranked_a, ranked_b):
        for rank, section in enumerate(ranked, start=1):
            fused_scores[section.id] = fused_scores.get(section.id, 0.0) + 1.0 / (
                k + rank
            )
            section_by_id.setdefault(section.id, section)

    # ``section_by_id`` iterates in first-seen (insertion) order; ``sorted`` is
    # stable, so equal fused scores keep that order — deterministic tie-breaking.
    ordered_ids = sorted(section_by_id, key=lambda sid: fused_scores[sid], reverse=True)
    return [(section_by_id[sid], fused_scores[sid]) for sid in ordered_ids[:top_k]]


# ---------------------------------------------------------------------------
# Pre-LLM OR-gate (pure function over native scores + language — AC3 / AC4)
# ---------------------------------------------------------------------------
def _bm25_arm_clears(top_score: float | None, lang: str) -> bool:
    """True when the BM25 arm's native top score clears its per-language gate.

    Mirrors ``markdown_kb``'s gate semantics (``top_score >= threshold``; higher
    BM25 score = stronger keyword match). The threshold is selected per query
    language — ``_SCORE_THRESHOLD_ZH`` (zh, #261) vs ``_SCORE_THRESHOLD`` (en) —
    and read through ``markdown_kb.app.retrieval`` at call time, so it is reused
    by reference (no redefinition) and honours the established
    ``_SCORE_THRESHOLD`` monkeypatch pattern.
    """
    if top_score is None:
        return False
    threshold = (
        _bm25_gate._SCORE_THRESHOLD_ZH if lang == "zh" else _bm25_gate._SCORE_THRESHOLD
    )
    return top_score >= threshold


def _dense_arm_clears(best_distance: float | None) -> bool:
    """True when the dense arm's closest hit clears its calibrated distance gate.

    Mirrors ``vector_rag``'s gate semantics exactly: lower FAISS distance = closer,
    so the arm clears when its BEST (minimum) distance ``<= ceiling``.  The ceiling
    is read through ``vector_rag.app.retrieval._max_rag_distance()`` — the same
    accessor Stack B's own gate uses, returning the calibrated
    ``_KB_RAG_DISTANCE_THRESHOLD_DEFAULT`` (1.1) and honouring the
    ``KB_RAG_DISTANCE_THRESHOLD`` override.

    ``None`` ceiling semantics (ADR-0018 §4.3 gate-parity, #327):
    ``_max_rag_distance()`` returns ``None`` when the gate is *disabled*.
    ``vector_rag``'s own gate disables on ``None`` (proceeds rather than refuses);
    hybrid must match — a *present* dense hit clears when the gate is disabled.
    The ``best_distance is None`` branch is orthogonal: no hit at all never clears
    regardless of whether the ceiling gate is enabled.
    """
    if best_distance is None:
        return False
    ceiling = _dense_gate._max_rag_distance()
    if ceiling is None:
        # Gate disabled → parity with vector_rag's None semantics: a present
        # dense hit clears (ADR-0018 §4.3, #327).
        return True
    return best_distance <= ceiling


def evaluate_or_gate(
    bm25_top_score: float | None,
    dense_best_distance: float | None,
    lang: str,
) -> GroundingOutcome:
    """Pure pre-LLM OR-gate over the two arms' native top scores + language.

    Proceed (``GroundingOutcome(passed=True, reason="claim_supported")``) when the
    BM25 arm clears its per-language threshold **OR** the dense arm clears its
    distance ceiling. Otherwise Cannot Confirm
    (``GroundingOutcome(passed=False, reason="below_threshold")``) — the same
    pre-LLM ``below_threshold`` reason both existing stacks emit, validated here
    against the shared ``GroundingOutcome.reason`` Literal rather than a bare
    string.

    OR — not AND — is load-bearing: AND would refuse whenever EITHER arm is weak,
    defeating the recall-union purpose of fusing two complementary methods (the
    whole reason Hybrid exists). The RRF fused score is never consulted here; the
    gate is on each arm's calibrated NATIVE score, before fusion (ADR-0018 §4.3
    gate-parity).
    """
    if _bm25_arm_clears(bm25_top_score, lang) or _dense_arm_clears(dense_best_distance):
        return GroundingOutcome(passed=True, reason="claim_supported")
    return GroundingOutcome(passed=False, reason="below_threshold")


# ---------------------------------------------------------------------------
# Retrieval core — overfetch both arms, gate, fuse (AC1 / AC4 / AC6)
# ---------------------------------------------------------------------------
def retrieve_and_gate(
    question: str,
    *,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Hybrid retrieval core: overfetch both arms, apply the OR-gate, fuse.

    Flow (no LLM call at this layer):
      1. Classify the query language once via the consolidated ``detect_lang``
         (#285) — the same classifier both arms route on.
      2. Overfetch a deep candidate pool from EACH arm at ``candidate_depth``
         (default 50), decoupled from the final ``top_k`` (default 3):
           * BM25  — ``markdown_kb.app.indexer.search`` → (Section, score)
           * dense — ``hybrid_kb.dense_index.search_with_distance`` → (Section, distance)
         Neither deep module is mocked; both run over their real (possibly
         test-built) indexes.
      3. Apply the pre-LLM OR-gate on each arm's NATIVE top score BEFORE fusion
         (BM25 top score / dense minimum distance).
      4. Fuse the two ranked lists with RRF (K=60) and truncate to ``top_k``.

    Returns a dict (mirroring the ``_retrieve_and_gate`` shape the Wiki/RAG
    stacks expose, so the S3 ``query()`` can compose it the same way):

        sections          — fused top-k ``Section`` objects (satisfy
                            ``CitableContent``), ready for downstream synthesis.
                            Populated even on a Cannot Confirm verdict (the weak
                            candidates), exactly as the existing stacks return
                            their below-threshold sources.
        grounding_outcome — the OR-gate verdict (provisional pass / Cannot Confirm).
        early_exit        — ``True`` when the gate refused (S3 skips the LLM).

    The gate lives HERE, inside the retrieval deep module — never in an adapter
    or route (ADR-0018 §4.3 gate-parity invariant).
    """
    lang = detect_lang(question)

    bm25_ranked = _bm25_indexer.search(question, k=candidate_depth)
    dense_ranked = dense_index.search_with_distance(question, k=candidate_depth)

    # Each arm's NATIVE top score: BM25's highest score (search returns
    # score-descending), the dense arm's minimum distance (closest hit). These —
    # not the fused score — drive the gate.
    bm25_top_score = bm25_ranked[0][1] if bm25_ranked else None
    dense_best_distance = min((distance for _, distance in dense_ranked), default=None)

    outcome = evaluate_or_gate(bm25_top_score, dense_best_distance, lang)

    fused = reciprocal_rank_fusion(
        [section for section, _ in bm25_ranked],
        [section for section, _ in dense_ranked],
        k=RRF_K,
        top_k=top_k,
    )
    sections = [section for section, _ in fused]

    return {
        "sections": sections,
        "grounding_outcome": outcome,
        "early_exit": not outcome.passed,
    }
