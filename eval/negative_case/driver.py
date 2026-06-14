"""Index the in-scope corpus and drive the REAL Cannot Confirm gate (LLM-free).

``evaluate_case`` calls ``retrieval._retrieve_and_gate`` so the eval measures the
production pre-LLM gate (BM25 + ``KB_SCORE_THRESHOLD``) directly, rather than
re-deriving the threshold logic — if production changes how it gates, this eval
moves with it. No LLM is ever called: for out-of-scope queries the gate
early-exits before synthesis, and for in-scope queries we stop at the gate.
"""

from __future__ import annotations

from pathlib import Path

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.retrieval as retrieval

from .models import RefusalOutcome

_PKG_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = _PKG_ROOT / "corpus"


def index_corpus(corpus_dir: Path | None = None) -> tuple[int, int]:
    """Build markdown_kb's Section Index over the committed in-scope corpus.

    Points ``SOURCE_DIRS`` at the corpus and runs the production build path. The
    caller (runner / test conftest) is responsible for redirecting ``INDEX_PATH``
    / ``WIKI_DIR`` to a tmp directory so production ``.kb/`` / ``wiki/`` are never
    touched (production isolation, mirroring ``stacks.index_stack_a``).
    """
    corpus = corpus_dir or CORPUS_DIR
    mk_indexer.SOURCE_DIRS = [corpus]
    return mk_indexer.build_index()


def evaluate_case(query: str) -> RefusalOutcome:
    """Run one out-of-scope query through the pre-LLM Cannot Confirm gate.

    ``refused`` is True iff the gate early-exited with the Cannot Confirm phrase
    (reason ``retrieval_empty`` / ``below_threshold``). The top BM25 score is
    surfaced so a report can show how close each leak was to the threshold.
    """
    gate = retrieval._retrieve_and_gate(query)
    refused = bool(gate["early_exit"]) and gate["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    reason = gate["grounding_outcome"].reason if refused else "answered"
    sources = gate.get("sources") or []
    top_score = float(sources[0]["score"]) if sources else 0.0
    return RefusalOutcome(query=query, refused=refused, reason=reason, top_score=top_score)
