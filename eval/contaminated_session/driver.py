"""Deep module per Ousterhout. Public surface: ``index_corpus``,
``evaluate_case``, ``RewriteFn``, ``GateOutcome``, ``ContaminatedSessionOutcome``.

Drive one contaminated-session case through an injected rewrite seam and the
REAL LLM-free retrieval gate (``markdown_kb.app.retrieval._retrieve_and_gate``
— mirrors ``eval.negative_case.driver``: never re-derive the gate's logic,
drive the production deep module directly so this eval moves with it).

``rewrite_fn`` is dependency-injected rather than imported directly so the
caller controls the seam:
  - the real production seam, ``gateway.app.query_rewriting.rewrite_query``
    (an LLM call — wired in only by ``runner.main``'s live path; CODING_STANDARD
    §6.4 caps a surface at one live test, already spent by
    ``gateway/tests/test_query_rewriting.py::test_rewrite_query_live``, so this
    eval never adds a second one and never calls it from pytest);
  - a deterministic stand-in for the offline-tracer path
    (``runner._offline_rewrite_stub``) and for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import markdown_kb.app.indexer as indexer
import markdown_kb.app.retrieval as retrieval

from .metric import DriftMetrics, compute_drift
from .sessions import ContaminatedSessionCase

_PKG_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = _PKG_ROOT / "corpus"


class RewriteFn(Protocol):
    """``(raw_query, *, history) -> rewritten_query`` — mirrors
    ``gateway.app.query_rewriting.rewrite_query``'s signature exactly
    (``history`` is KEYWORD-ONLY there, hence the ``*`` here too — a plain
    ``Callable[[str, list[dict]], str]`` alias cannot express that and would
    silently accept a stub that takes ``history`` positionally, which is
    exactly the #608 seam-drift bug) so the real function is a drop-in
    ``RewriteFn``.
    """

    def __call__(self, raw_query: str, *, history: list[dict]) -> str: ...


def index_corpus(corpus_dir: Path | None = None) -> tuple[int, int]:
    """Build markdown_kb's Section Index over the committed characterization corpus.

    Mirrors ``eval.negative_case.driver.index_corpus``: points ``SOURCE_DIRS``
    at the corpus and runs the production build path. The caller is
    responsible for redirecting ``INDEX_PATH`` / ``WIKI_DIR`` to a tmp
    directory first (test isolation, CODING_STANDARD §6.5) — see the test
    conftest / ``runner._isolate_production_paths``.
    """
    corpus = corpus_dir or CORPUS_DIR
    indexer.SOURCE_DIRS = [corpus]
    return indexer.build_index()


@dataclass(frozen=True)
class GateOutcome:
    """The LLM-free retrieval gate's outcome for one rewritten query."""

    query: str
    #: ``ranked[0]``'s Section id, or ``None`` when nothing ranked at all.
    top_source: str | None
    #: ``GroundingOutcome.reason``: "claim_supported" (gate passed) |
    #: "below_threshold" | "retrieval_empty" (early-exit Cannot Confirm).
    reason: str


@dataclass(frozen=True)
class ContaminatedSessionOutcome:
    """One case's full result: both rewrites, drift, and the flip verdict."""

    case: ContaminatedSessionCase
    contaminated_rewrite: str
    clean_rewrite: str
    drift: DriftMetrics
    contaminated_gate: GateOutcome
    clean_gate: GateOutcome

    @property
    def flipped(self) -> bool:
        """True iff contamination changed the retrieval outcome vs. the clean
        control for the SAME literal follow-up: a different top Section, or a
        different gate reason (e.g. clean answers, contaminated falls to
        Cannot Confirm, or vice versa)."""
        return (
            self.contaminated_gate.top_source != self.clean_gate.top_source
            or self.contaminated_gate.reason != self.clean_gate.reason
        )


def _gate(query: str) -> GateOutcome:
    result = retrieval._retrieve_and_gate(query)
    sources = result["sources"]
    top_source = sources[0]["source"] if sources else None
    return GateOutcome(
        query=query, top_source=top_source, reason=result["grounding_outcome"].reason
    )


def evaluate_case(
    case: ContaminatedSessionCase, rewrite_fn: RewriteFn
) -> ContaminatedSessionOutcome:
    """Run one case: rewrite the same literal follow-up under both histories,
    gate both rewrites, and compute drift against the user's literal ask.

    Assumes the caller already built the corpus index (see ``index_corpus``).
    The clean-history arm is the control: ``case.clean_history`` is normally
    empty, so ``rewrite_fn`` takes the turn-1-passthrough branch there and the
    ONLY LLM call (when ``rewrite_fn`` is the real ``rewrite_query``) is the
    contaminated arm.
    """
    contaminated_rewrite = rewrite_fn(
        case.followup_question, history=case.contaminated_history
    )
    clean_rewrite = rewrite_fn(case.followup_question, history=case.clean_history)
    drift = compute_drift(case.followup_question, contaminated_rewrite)
    return ContaminatedSessionOutcome(
        case=case,
        contaminated_rewrite=contaminated_rewrite,
        clean_rewrite=clean_rewrite,
        drift=drift,
        contaminated_gate=_gate(contaminated_rewrite),
        clean_gate=_gate(clean_rewrite),
    )
