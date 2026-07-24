"""Deep module per Ousterhout. Public surface: ``parse_cited_source_ids``,
``dense_over_wiki_query``, ``ARM_QUERY_FNS``, ``build_answer_fn``.

Wires the corpus v3 live-mode answering seam (``run_verdict.AnswerFn``,
``(query_id, arm, retrieved_items) -> AnswerRecord``) through each arm's
real, in-process public query surface -- no HTTP (PRD #654: "through each
app's public query function, not HTTP"). Issue #662's ``run_verdict.py``
module docstring named this module's job as an explicit, unwired follow-up
rather than faking a live run; this module (issue #673) is that follow-up.

Three arms already own a full retrieval + synthesis + Grounding Check query
surface that this module drives UNCHANGED (nothing here reimplements it):

    wiki    -> markdown_kb.app.retrieval.query
    rag     -> vector_rag.app.retrieval.query
    hybrid  -> hybrid_kb.app.query.query

``dense_over_wiki`` has no such surface of its own -- ADR-0045 Prerequisite 1
names it only as a RETRIEVAL cell (``eval.corpus_v3.stacks
.dense_over_wiki_retrieval``: hybrid_kb's dense arm searched WITHOUT RRF).
:func:`dense_over_wiki_query` composes ``hybrid_kb.app.dense_index.search``
(the retrieval leg, no RRF) with ``hybrid_kb.app.query.answer_over_sections``
(a small additive public seam this issue adds to ``hybrid_kb.app.query`` --
the SAME answer-synthesis leg ``query()`` itself uses, over a caller-supplied
Section pool). Composing at that seam, rather than re-invoking LangChain
directly here, keeps LangChain message/client types confined to
``hybrid_kb.app.query`` (CODING_STANDARD §2.4 / ADR-0005) and reuses
``query()``'s own error handling, self-refusal short-circuit, and Grounding
Check verbatim instead of re-implementing them.

Citation extraction: every stack's shared ``SYSTEM_PROMPT``
(``markdown_kb.app.prompt_builder``) instructs the LLM to mark each factual
claim with the literal ``[Source: <id>]`` token.
``content_axes.AnswerRecord.cited_source_ids`` is populated by parsing those
tokens back out of the answer text -- the one piece
``content_axes.AnswerRecord``'s own docstring names as "not this module's
concern how"; :func:`parse_cited_source_ids` is that concern, owned here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

import hybrid_kb.app.dense_index as hk_dense
import hybrid_kb.app.query as hybrid_query
import markdown_kb.app.retrieval as mk_retrieval
import vector_rag.app.retrieval as vr_retrieval
from hybrid_kb.app.retrieval import DEFAULT_TOP_K

from eval.cost_ledger.hooks import instrument_invoke
from eval.cost_ledger.ledger import CostLedger

from .content_axes import AnswerRecord
from .models import RetrievedItem

# ---------------------------------------------------------------------------
# Citation extraction (issue #673 -- the one piece content_axes leaves open)
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[Source:\s*([^\]]+)\]")


def parse_cited_source_ids(answer_text: str) -> frozenset[str]:
    """Parse every literal ``[Source: <id>]`` citation token out of ``answer_text``.

    A refusal (the Cannot Confirm sentinel) carries no citation token and
    parses to an empty ``frozenset``, matching ``AnswerRecord``'s own default
    and ``content_axes.grounding_pass``'s "no citation -> not grounded" rule.
    """
    return frozenset(
        match.group(1).strip() for match in _CITATION_RE.finditer(answer_text)
    )


# ---------------------------------------------------------------------------
# dense_over_wiki -- the missing synthesis leg (ADR-0045 Prerequisite 1)
# ---------------------------------------------------------------------------
def dense_over_wiki_query(question: str, k: int = DEFAULT_TOP_K) -> dict:
    """Answer via hybrid_kb's dense arm alone, retrieval WITHOUT RRF fusion.

    Assumes ``stacks.index_wiki_corpus()`` and ``stacks.index_dense_over_wiki()``
    have already populated the wiki Section list and the dense index (mirrors
    ``stacks.dense_over_wiki_retrieval``'s own precondition). Composes the
    retrieval leg (``hk_dense.search``, no RRF) with the synthesis leg
    (``hybrid_query.answer_over_sections``, this issue's small additive
    public seam on ``hybrid_kb.app.query`` -- the SAME answer-synthesis
    ``query()`` itself uses). Returns the same ``{answer, sources,
    grounding_outcome}`` shape every other arm's ``query()`` returns, so
    :data:`ARM_QUERY_FNS` can register all four arms uniformly.

    No pre-LLM relevance gate is applied here beyond ``answer_over_sections``'s
    own empty-pool check (unlike the other three arms' calibrated OR-gate /
    threshold gates) -- calibrating this arm's own gate is ADR-0045
    Prerequisite 2's job, verified separately in the pilot issue (#674).
    """
    sections = hk_dense.search(question, k=k)
    return hybrid_query.answer_over_sections(question, sections)


# ---------------------------------------------------------------------------
# Per-arm public query surfaces (issue #673 AC 1: "no HTTP anywhere")
# ---------------------------------------------------------------------------
ARM_QUERY_FNS: dict[str, Callable[[str], dict]] = {
    "wiki": mk_retrieval.query,
    "rag": vr_retrieval.query,
    "hybrid": hybrid_query.query,
    "dense_over_wiki": dense_over_wiki_query,
}

# Where each arm's answer-synthesis LLM singleton getter lives (issue #657's
# cost ledger hooks wrap exactly this seam). ``dense_over_wiki`` shares
# ``hybrid_kb``'s getter (see ``dense_over_wiki_query``); :func:`_instrumented`
# always wraps-then-restores per call so the shared getter never carries a
# stale (arm, ledger) wrap across arms.
_LLM_GETTER_TARGETS: dict[str, tuple[object, str]] = {
    "wiki": (mk_retrieval, "get_llm"),
    "rag": (vr_retrieval, "get_llm"),
    "hybrid": (hybrid_query, "get_llm"),
    "dense_over_wiki": (hybrid_query, "get_llm"),
}


@contextmanager
def _instrumented(arm: str, ledger: CostLedger | None) -> Iterator[None]:
    """Temporarily wrap ``arm``'s answer-synthesis LLM getter so every
    ``.invoke()`` call made during the ``with`` block is recorded into
    ``ledger`` under ``(stack=arm, phase="query")``, then restore the
    module's original getter.

    Wrap-then-restore (rather than a permanent monkeypatch) matters because
    ``dense_over_wiki`` and ``hybrid`` share the SAME underlying getter
    (``hybrid_kb.app.query.get_llm``) -- restoring after every call is what
    keeps one arm's calls from being mis-attributed to the other, and keeps
    repeated calls from double-wrapping an already-wrapped getter.

    A ``None`` ledger (the default -- a caller that doesn't want cost
    accounting) is a no-op.
    """
    if ledger is None:
        yield
        return
    module, attr_name = _LLM_GETTER_TARGETS[arm]
    original_getter = getattr(module, attr_name)
    setattr(
        module,
        attr_name,
        instrument_invoke(original_getter, ledger, stack=arm, phase="query"),
    )
    try:
        yield
    finally:
        setattr(module, attr_name, original_getter)


# ---------------------------------------------------------------------------
# Public API — the AnswerFn factory (run_verdict.AnswerFn's shape)
# ---------------------------------------------------------------------------
def build_answer_fn(
    query_text_by_id: Mapping[str, str], *, ledger: CostLedger | None = None
) -> Callable[[str, str, list[RetrievedItem]], AnswerRecord]:
    """Build the real, in-process ``AnswerFn``: ``(query_id, arm,
    retrieved_items) -> AnswerRecord`` (``run_verdict.AnswerFn``'s fixed
    shape; issue #662's module docstring named wiring it as this issue's
    job).

    ``retrieved_items`` is accepted -- the seam's contract is fixed by
    ``run_verdict.py`` -- but UNUSED: each arm answers through its REAL
    public query function (:data:`ARM_QUERY_FNS`), which performs its own
    retrieval internally. Accepting externally-supplied retrieved items
    instead would decouple the measured answer from what the app actually
    serves end-to-end (PRD #654: "through each app's public query function,
    not HTTP").

    ``query_text_by_id`` resolves a query id to its text (the stratified
    ``query_schema.Query.text`` the caller's loaded query set already
    carries) -- a missing id is a caller programming error and raises
    ``KeyError`` immediately rather than silently answering the wrong
    question.

    ``ledger`` (default ``None``, no cost accounting) records the
    answer-synthesis call's tokens under ``(stack=arm, phase="query")`` for
    every call the returned function makes (issue #657's ``CostLedger``,
    wired here per its own ``hooks.py`` docstring: "Call
    record_usage_from_response directly at those call sites when the
    harness wires this ledger into the corpus v3 runner").
    """

    def answer_fn(
        query_id: str, arm: str, retrieved_items: list[RetrievedItem]
    ) -> AnswerRecord:
        del retrieved_items  # unused -- see docstring above
        if arm not in ARM_QUERY_FNS:
            raise ValueError(
                f"unknown arm {arm!r}; expected one of {sorted(ARM_QUERY_FNS)}"
            )
        text = query_text_by_id[query_id]
        with _instrumented(arm, ledger):
            result = ARM_QUERY_FNS[arm](text)
        return AnswerRecord(
            query_id=query_id,
            arm=arm,
            answer_text=result["answer"],
            cited_source_ids=parse_cited_source_ids(result["answer"]),
        )

    return answer_fn
