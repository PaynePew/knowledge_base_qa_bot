"""Deep module per Ousterhout. Public surface: ``query``, ``stream_query``, ``warm_llm_client``.

Retrieval layer — query() and stream_query() for the grounded /chat endpoint.

query() flow:
  1. tokenize query
  2. search top-3 Sections via BM25
  3. pre-LLM Cannot Confirm gate (ADR-0001)
  4. build_prompt with Citation markers
  5. call LLM (with error mapping for OpenAI exceptions)
  6. post-LLM Grounding Check via grounding.verify() (ADR-0004 layer 3)
  7. write chat log entry
  8. return {answer, sources, grounding_outcome}

stream_query() (Phase 9, ADR-0009) wraps query() to yield a partial result
dict immediately after retrieval (so the gateway can emit the sources SSE
event before the LLM calls run), then yields the full result after
draft + verify complete.

The retrieve / draft split is implemented via two private helpers:
  _retrieve_and_gate() — BM25 search + pre-LLM Cannot Confirm gates
  _draft_and_verify()  — build_prompt + LLM draft + Grounding Check
Public query() composes them; contract is unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import openai
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import grounding as grounding_module
from . import indexer
from .errors import LLMError
from .grounding import GroundingOutcome
from .logger import log_event
from .prompt_builder import SYSTEM_PROMPT, build_prompt
from .schemas import qa_schema_lint_code

# Score threshold below which retrieval is treated as "no match" and the bot
# returns Cannot Confirm.
#
# The default is empirically justified by the #253 calibration sweep
# (eval/negative_case/calibration_report.md): every threshold in the [0.25, 1.25]
# plateau rejects all clearly-out-of-scope queries (~0 BM25 score) at 0%
# over-refusal of in-scope queries (min in-scope score ~1.41). 0.5 sits inside
# that plateau with margin on both sides, so it is kept (a working default is not
# churned to a different equally-good value). The residual adjacent-absent leaks
# score inside the in-scope range and are unreachable by ANY threshold without
# over-refusing real queries — they need semantic reranking (Phase 13 / FM2), not
# threshold tuning.
#
# Override with KB_SCORE_THRESHOLD env var. Read at import time: a server restart
# picks up a new value; runtime changes do not (tests monkeypatch _SCORE_THRESHOLD
# directly).
_KB_SCORE_THRESHOLD_DEFAULT = 0.5
_SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", str(_KB_SCORE_THRESHOLD_DEFAULT)))

# Per-language Chinese threshold (#261). Chinese bigram BM25 scores sit in a higher
# band than English (the same query emits more tokens, inflating the summed score —
# ADR-0014), so the English-calibrated 0.5 leaks Chinese adjacent-absent queries.
# The #256/#261 re-sweep over the enlarged 10-topic Chinese corpus
# (eval/negative_case/calibration_report_zh.md) found the catchable adjacent-absent
# leaks at ~1.9-2.8 sitting below the min in-scope score ~4.9, with 4.0 the
# Youden-J-optimal separator (correct-refusal 95%, over-refusal 0%): it lifts
# Chinese correct-refusal from 76% (at the global 0.5) to 95%.
#
# INTERIM — superseded by the Phase 13 reranker (roadmap Phase 13 / ADR-0014). A
# per-language threshold cannot catch the residual adjacent-absent leak that scores
# *inside* the in-scope range (a query whose surface tokens match a real Section but
# whose specific ask is absent); that semantic-overlap case is Phase 13 territory in
# both languages. When Phase 13 lands, this knob and its routing should be retired.
#
# Override with KB_SCORE_THRESHOLD_ZH env var (import-time, like KB_SCORE_THRESHOLD).
_KB_SCORE_THRESHOLD_ZH_DEFAULT = 4.0
_SCORE_THRESHOLD_ZH = float(os.getenv("KB_SCORE_THRESHOLD_ZH", str(_KB_SCORE_THRESHOLD_ZH_DEFAULT)))

# Dominant-script gate routing (#582). A code-switched query used to be gated by
# TWO independently maintained classifiers that could disagree: _is_cjk_query
# (any CJK char present -> the zh threshold) decided the threshold, while
# indexer.detect_lang's ratio (>= 0.20 -> "zh") decided the corpus slice. A
# Latin-dominant query naming a single CJK product could get the zh threshold
# applied to an en-only corpus slice's score -- a false Cannot Confirm (pinned
# by #601's characterization fixtures). _dominant_script() below replaces both
# with ONE classifier so the threshold and the corpus slice always agree.
#
# The band is deliberately wider than indexer._ZH_RATIO_THRESHOLD (0.20): that
# ratio is tuned for index-time Section TAGGING (a whole page's dominant
# language), not for a short, possibly genuinely-balanced QUERY. Inside the
# band neither script is a confident majority, so _gate_route() falls back to
# union (both corpus slices searched, gate passes if either clears its own
# threshold) instead of committing to a single, possibly-wrong route.
_DOMINANT_SCRIPT_LOW = 0.40
_DOMINANT_SCRIPT_HIGH = 0.60

# Sentinel strings the system returns to /chat clients. Tests import these
# constants so a typo in production is caught instead of silently passing
# against a hardcoded test literal.
CANNOT_CONFIRM_PHRASE = "I cannot confirm from the knowledge base."
NOT_INDEXED_MESSAGE = "The knowledge base has not been indexed yet. Call POST /ingest to populate the wiki, then POST /index."

# Fixed seed for the drafter + retry-drafter LLMs (ADR-0038's C5-judge pattern,
# extended to the serving chain by ADR-0042 / issue #572). Per-module private
# constant. Best-effort only: OpenAI's seed is not a hard guarantee, but paired
# with temperature=0 it cuts run-to-run draft rewording that would otherwise
# hand the verifier a legitimately different claim set on re-ask.
_ANSWER_CHAIN_LLM_SEED = 7

_llm = None
# Separate singleton for temperature=0 grounding retries.
# Tests monkeypatch both _llm and _retry_llm (or get_llm / get_retry_llm).
_retry_llm = None


def get_llm():
    global _llm
    if _llm is None:
        # temperature=0: the draft must be deterministic. At the langchain
        # default temperature the model intermittently self-refuses on a
        # question it can answer, so the same query flip-flops between a
        # grounded answer and a false Cannot Confirm across calls.
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            seed=_ANSWER_CHAIN_LLM_SEED,
            timeout=20,
            max_retries=1,
        )
    return _llm


def get_retry_llm():
    """Return the temperature=0 LLM used for grounding retries."""
    global _retry_llm
    if _retry_llm is None:
        _retry_llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            seed=_ANSWER_CHAIN_LLM_SEED,
            timeout=20,
            max_retries=1,
        )
    return _retry_llm


def warm_llm_client() -> None:
    """Fire one tiny ping at the answer LLM client to prime its connection (issue #439).

    Cold-start fix: the first REAL ``/chat`` request otherwise pays
    ``ChatOpenAI`` client construction + TLS handshake + first-connection
    latency on top of the actual answer call. Gateway startup (opt-in behind
    ``KB_WARMUP_PING`` — see ``gateway/app/warmup.py``) calls this once per
    process so that one-time cost lands at boot instead of on a user's first
    question. ``max_tokens=1`` keeps the completion itself a few tokens; the
    reply is discarded.

    Best-effort: any failure (auth, quota, network) is caught and logged, never
    raised — a failed ping degrades to the pre-issue-#439 behaviour (the client
    still lazily constructs + connects on the next real call) and never blocks
    Gateway startup.
    """
    try:
        get_llm().invoke("Hi", max_tokens=1)
        log_event("startup_warmup", "client=wiki_llm status=ok")
    except Exception as exc:
        log_event("startup_warmup", f"client=wiki_llm status=failed exc={type(exc).__name__}")


def query(question: str, exclude_qa: bool = False) -> dict:
    """Answer a question against the indexed corpus.

    Returns a dict with keys:
        answer            — grounded text (may be "I cannot confirm from the knowledge base.")
        sources           — list of {source, heading, score, content} dicts
        grounding_outcome — GroundingOutcome instance (always present, never None)

    Pre-LLM gates (ADR-0001 — never hand weak context to the LLM):
    1. If sections is empty → not indexed yet.
    2. If BM25 yields no results → Cannot Confirm (retrieval_empty).
    3. If top score < threshold → Cannot Confirm (below_threshold).

    Post-LLM gate (ADR-0004 layer 3):
    4. grounding.verify() validates every claim against cited Sections.
       Any unsupported claim → Cannot Confirm (claim_unsupported).
       Verifier failure after retry → Cannot Confirm (verifier_unavailable).

    ``exclude_qa`` (tier-B S4, issue #380, ADR-0026 decision 1) drops every
    ``wiki/qa/`` Section from retrieval for this call — the C9 Re-file
    remediation's internal re-synthesis step, so a stale Filed Answer being
    re-derived can never retrieve (and re-cite) itself. Default ``False``
    preserves every existing caller (``/chat``, ``stream_query``, the CLI/
    MCP/hybrid callers) unchanged.

    Phase 9: composes _retrieve_and_gate() + _draft_and_verify().
    Public contract is unchanged; the split is behaviour-preserving.
    """
    gate = _retrieve_and_gate(question, exclude_qa=exclude_qa)
    if gate["early_exit"]:
        # Pre-LLM gate fired — return early without calling the LLM.
        return {
            "answer": gate["answer"],
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }

    # LLM phase — draft + Grounding Check.
    result = _draft_and_verify(question, gate["ranked"], gate["sources"])
    return result


def stream_query(question: str, original_question: str | None = None) -> Iterator[dict]:
    """Generator that yields two dicts for use by the SSE streaming endpoint.

    Phase 9 — ADR-0009 (verify-then-stream / sources-first).

    Issue #579 (multi-turn rewrite drift): ``question`` is always the query
    actually used for retrieval — the Gateway's rewritten, self-contained
    follow-up on turn 2+, or the raw query on turn 1 passthrough. It drives
    BM25 search, the LLM prompt, and the chat log exactly as before.
    ``original_question`` is the caller's literal ask, forwarded ONLY for
    filing (``dispatch_filing``'s ``query`` arg — slug + the ``question``
    frontmatter field); it defaults to ``question`` when omitted (every
    caller except the Gateway's turn-2+ wiki dispatch, where the two
    diverge). ``question`` itself is passed through unchanged as
    ``retrieval_query`` so the filed page's audit metadata always reflects
    what actually retrieved it.

    Yields:
        1. A *partial* result dict immediately after retrieval (before any
           LLM call), so the gateway can emit the ``sources`` SSE event
           right away (~instant for BM25, ~1 embedding round-trip for RAG).
           Shape: ``{sources, grounding_outcome, _phase: "sources_ready"}``.
           ``grounding_outcome`` at this stage is provisional — it is the
           pre-LLM gate outcome (may be a Cannot Confirm if below threshold).
        2. A *full* result dict after draft + Grounding Check complete.
           Shape: ``{answer, sources, grounding_outcome}`` — identical to
           what ``query()`` returns.

    The gateway endpoint must:
      a. Emit ``sources`` SSE event from yield 1.
      b. Emit ``token`` + ``done`` SSE events from yield 2.

    ADR-0009: only verified text is ever emitted as tokens.  The generator
    never yields an unverified draft.
    """
    gate = _retrieve_and_gate(question)

    # Yield the sources-ready partial result so the gateway can emit the
    # sources event before making any LLM call.
    yield {
        "_phase": "sources_ready",
        "sources": gate["sources"],
        "grounding_outcome": gate["grounding_outcome"],
        "early_exit": gate["early_exit"],
        "answer": gate.get("answer", ""),
        "ranked": gate.get("ranked", []),
    }

    if gate["early_exit"]:
        # Pre-LLM gate fired — the partial result IS the full result.
        # Yield the final form so the caller always gets a full result on
        # the second yield.
        #
        # SSE uniformity (ADR-0009 / Phase 9 Slice 2): all five CC reasons must
        # stream CANNOT_CONFIRM_PHRASE in the token events so the UI token
        # stream is identical regardless of which gate fired.  The ``index_missing``
        # path in _retrieve_and_gate returns NOT_INDEXED_MESSAGE (the verbose
        # curator-targeted string); normalise to CANNOT_CONFIRM_PHRASE here so
        # stream callers never need to branch on the reason.  query() is
        # unaffected (it does not call this path via stream_query).
        # The specific reason is still preserved in grounding_outcome.reason for
        # machine consumers (done.reason in the SSE done event).
        yield {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }
        return

    # LLM phase — draft + Grounding Check.
    result = _draft_and_verify(question, gate["ranked"], gate["sources"])

    # Phase 9 Slice 4: file on grounding-pass — same behaviour as /chat.
    # Filing happens server-side at the post-verify point, independent of
    # client delivery (a disconnected client never causes partial/unfiled
    # state, and a partial answer is never filed). The existing SSE serializer
    # already reads result.get("filed") and populates done.filed.
    # Import is deferred to avoid a circular import with qa.py.
    from . import qa as _qa_module  # noqa: PLC0415

    # Issue #579: file under the caller's literal ask (original_question),
    # never the rewrite; ``question`` — the string actually used for
    # retrieval above — becomes the frontmatter audit field instead.
    # original_question defaults to question when the caller has no rewrite
    # to distinguish (turn 1 / non-Gateway callers).
    filed = _qa_module.dispatch_filing(
        original_question if original_question is not None else question,
        result,
        retrieval_query=question,
    )
    # Attach filing outcome to result so events_for_result() picks it up.
    result = {**result, "filed": filed}

    yield result


# ---------------------------------------------------------------------------
# Phase 9 private helpers — retrieve+gate and draft+verify
# ---------------------------------------------------------------------------


def _retrieve_and_gate(question: str, exclude_qa: bool = False) -> dict:
    """BM25 retrieval + all pre-LLM Cannot Confirm gates (ADR-0001).

    Returns a dict with:
        sources          — list of {source, heading, score, content, derived_from}
        grounding_outcome — provisional outcome (pre-LLM gate result)
        early_exit       — True when a pre-LLM gate fired (no LLM needed)
        answer           — set to the Cannot Confirm phrase on early_exit paths
        ranked           — raw (Section, score) pairs for _draft_and_verify;
                           only meaningful when early_exit is False

    Callers (query() and stream_query()) use the early_exit flag to decide
    whether to call _draft_and_verify().  The sources list is always
    populated even on early_exit paths — the gateway emits a sources SSE
    event before checking early_exit.

    ``exclude_qa`` is forwarded to ``indexer.search`` (ADR-0026 decision 1 —
    see ``query()``'s docstring).
    """
    if not indexer.sections:
        # Lazy-load the persisted .kb/index.json from disk so a fresh Gateway
        # process can serve stack=wiki without requiring a POST /wiki/index call
        # in the same process (mirrors the RAG lazy-load fix, issue #133 / #148).
        # load_index_json() returns (0, 0) and leaves sections=[] when no
        # persisted index exists on disk.
        indexer.load_index_json()

    if not indexer.sections:
        log_event(
            "chat_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason=not_indexed',
        )
        return {
            "sources": [],
            "grounding_outcome": GroundingOutcome(passed=False, reason="index_missing"),
            "early_exit": True,
            "answer": NOT_INDEXED_MESSAGE,
            "ranked": [],
        }

    # Dominant-script gate routing (#582) — supersedes the #261 any-char
    # _is_cjk_query threshold pick. One classifier decides both which
    # threshold applies and which corpus slice is searched, so they can never
    # disagree; a near-50/50 query falls back to union (both slices searched,
    # gate passes if either clears its own threshold). See _gate_route.
    ranked, threshold = _gate_route(question, exclude_qa)
    top_score = ranked[0][1] if ranked else 0.0

    # Phase 6 Slice 6-3: each entry carries an optional ``derived_from`` chain
    # populated from the parent wiki page's ``frontmatter.sources``. This is
    # response-only audit data — the chain must NEVER appear in the LLM CONTEXT
    # or verifier prompt (ADR-0006 / PRD #78 Q4 W1 invariant).
    sources = [
        {
            "source": sec.id,
            "heading": " > ".join(sec.heading_path),
            "score": round(score, 3),
            "content": sec.content[:240],
            "derived_from": _derived_from_for_section(sec),
            "path": _wiki_page_path_for_section(sec),
            # C10 coordinate for the reader's teaching tag: a schema-invalid Filed
            # Answer surfacing as a source (e.g. a planted count:0 fixture). Pure —
            # from the Section's own frontmatter; None for valid or non-qa Sections.
            "lint": qa_schema_lint_code(sec.metadata),
        }
        for sec, score in ranked
    ]

    if top_score < threshold:
        truncated = question[:60].replace('"', "'")
        gate_reason = "retrieval_empty" if not ranked else "below_threshold"
        if ranked:
            top_sec = ranked[0][0]
            log_event(
                "chat_fallback",
                f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}'
                f" top_section={top_sec.id}",
            )
        else:
            log_event(
                "chat_fallback",
                f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}',
            )
        return {
            "sources": sources,
            "grounding_outcome": GroundingOutcome(passed=False, reason=gate_reason),
            "early_exit": True,
            "answer": CANNOT_CONFIRM_PHRASE,
            "ranked": [],
        }

    return {
        "sources": sources,
        "grounding_outcome": GroundingOutcome(passed=True, reason="claim_supported"),
        "early_exit": False,
        "answer": "",
        "ranked": ranked,
    }


def _draft_and_verify(
    question: str,
    ranked: list,
    sources: list[dict],
) -> dict:
    """LLM draft + post-LLM Grounding Check (ADR-0004 layer 3).

    Called only when the pre-LLM gates passed (early_exit is False).

    Args:
        question: The original user query.
        ranked:   (Section, score) pairs from BM25.
        sources:  Already-built sources list (passed through unchanged).

    Returns:
        Full result dict: {answer, sources, grounding_outcome}.
        Never returns unverified text — on grounding failure, answer is
        CANNOT_CONFIRM_PHRASE and grounding_outcome.passed is False.
    """
    # B3 page expansion (Slice 4-4): expand BM25 hits to full parent pages so
    # the LLM receives page-coherent context. The expanded list is used for
    # prompt construction and grounding verification; sources[] stays BM25 top-K.
    ranked_sections = [sec for sec, _score in ranked]
    expanded_sections = indexer.expand_to_pages(ranked_sections)
    prompt_text = build_prompt(question, expanded_sections)

    draft = _call_llm_with_error_handling(question, prompt_text)

    # LLM self-refusal short-circuit (green-light-on-non-answer fix).
    # SYSTEM_PROMPT Rules 3/6 instruct the model to emit CANNOT_CONFIRM_PHRASE
    # verbatim when the CONTEXT is insufficient. This fires on "adjacent-absent"
    # queries that clear the BM25 threshold but whose specific answer is not in the
    # retrieved Sections (the FM2 leak noted at _SCORE_THRESHOLD). The refusal carries
    # no factual claim, so grounding.verify() has nothing to refute and would return
    # passed=True — surfacing a green "Grounded" badge on a non-answer. Treat the
    # model's own refusal as Cannot Confirm directly and skip the verifier entirely
    # (nothing to verify, and the spurious pass is exactly the bug). reason reuses
    # claim_unsupported — the draft is not a grounded answer — so the outcome stays
    # within the existing GroundingInfo / lint value set with no schema change, and
    # Phase 5 /lint C1 still aggregates it into the coverage backlog.
    if draft.strip() == CANNOT_CONFIRM_PHRASE:
        cited_ids = ",".join(sec.id for sec in expanded_sections)
        log_event(
            "chat_grounding_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason=claim_unsupported'
            f" cited={cited_ids}",
        )
        _write_chat_log(question, ranked)
        return {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": sources,
            "grounding_outcome": GroundingOutcome(passed=False, reason="claim_unsupported"),
        }

    # Post-LLM Grounding Check (ADR-0004 layer 3).
    # verify() never raises — all verifier failures map to
    # grounding_outcome.reason = "verifier_unavailable".
    outcome = grounding_module.verify(draft, expanded_sections)

    if outcome.passed:
        answer = draft
    else:
        answer = CANNOT_CONFIRM_PHRASE
        cited_ids = ",".join(sec.id for sec in expanded_sections)
        log_event(
            "chat_grounding_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason={outcome.reason}'
            f" cited={cited_ids}",
        )

    _write_chat_log(question, ranked)

    return {
        "answer": answer,
        "sources": sources,
        "grounding_outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_cjk_query(question: str) -> bool:
    """True when the query contains any CJK character (#261).

    Reuses ``indexer._is_cjk`` (the same predicate the CJK-bigram tokeniser uses,
    ADR-0014) so query-language detection and tokenisation never drift apart.

    No longer used for gate threshold routing — ``_dominant_script`` (#582)
    replaced this any-char predicate there because it could disagree with
    ``indexer.detect_lang``'s ratio-based corpus-slice pick for a code-switched
    query (#601). Kept as a plain, still-tested predicate; not dead code to
    remove, just no longer wired into ``_retrieve_and_gate``.
    """
    return any(indexer._is_cjk(ch) for ch in question)


def _script_ratio(question: str) -> float:
    """CJK-vs-Latin codepoint ratio of ``question`` for #582 dominant-script routing.

    Counts CJK ideographs (``indexer._is_cjk``) against Latin/alphabetic
    letters; digits, whitespace, and punctuation are neutral and excluded from
    both the numerator and the denominator — mirrors the "letter characters
    only" exclusion in ``indexer.detect_lang`` so the ratio is neither diluted
    nor inflated by non-language-bearing characters.

    A query with no letters at all (empty, digits/symbols only) has no script
    signal and reads as ``0.0`` — the low end of the range, resolving to the
    "en" fail-closed default via ``_dominant_script`` (mirrors
    ``indexer._DEFAULT_LANG``).
    """
    cjk = 0
    latin = 0
    for ch in question:
        if indexer._is_cjk(ch):
            cjk += 1
        elif ch.isalpha():
            latin += 1
    total = cjk + latin
    return cjk / total if total else 0.0


def _dominant_script(question: str) -> str:
    """Classify ``question``'s dominant script for gate routing (#582).

    Returns ``"zh"`` when the CJK ratio (``_script_ratio``) is at or above
    ``_DOMINANT_SCRIPT_HIGH``, ``"en"`` when at or below
    ``_DOMINANT_SCRIPT_LOW``, else ``"mixed"`` — the near-50/50 boundary that
    ``_gate_route`` resolves via union rather than committing to a single,
    possibly-wrong route.
    """
    ratio = _script_ratio(question)
    if ratio >= _DOMINANT_SCRIPT_HIGH:
        return "zh"
    if ratio <= _DOMINANT_SCRIPT_LOW:
        return "en"
    return "mixed"


def _search_for_lang(
    question: str, lang: str, exclude_qa: bool
) -> list[tuple[indexer.Section, float]]:
    """Search the ``lang`` corpus slice, forcing an override only when it matters.

    ``indexer.search`` already resolves its own corpus slice via
    ``indexer.detect_lang(question)`` (ratio >= 0.20) when no ``lang=`` is
    given. That bar is lower than the #582 dominant-script band
    (``_DOMINANT_SCRIPT_LOW``/``_HIGH``), so the two classifiers can disagree
    for a query whose CJK ratio sits in [0.20, ``_DOMINANT_SCRIPT_LOW``) —
    exactly the #601 mismatch this routing closes. An explicit ``lang=``
    override is passed only when it would change ``indexer.search``'s own
    pick, so the call shape stays ``indexer.search(question, k=3[,
    exclude_qa=True])`` — byte-identical to every pre-#582 caller — whenever
    no override is needed. Several test doubles across the suite (e.g.
    ``kb_cli``'s CLI tests) replace ``indexer.search`` with a fixed
    ``lambda query, k=3: ...`` and would TypeError on an unconditional new
    kwarg — the same reason the pre-#582 ``exclude_qa`` argument is forwarded
    conditionally in ``_retrieve_and_gate``.
    """
    override = lang if indexer.detect_lang(question) != lang else None
    if exclude_qa:
        if override:
            return indexer.search(question, k=3, exclude_qa=True, lang=override)
        return indexer.search(question, k=3, exclude_qa=True)
    if override:
        return indexer.search(question, k=3, lang=override)
    return indexer.search(question, k=3)


def _gate_route(
    question: str, exclude_qa: bool
) -> tuple[list[tuple[indexer.Section, float]], float]:
    """Return ``(ranked, threshold)`` — the #582 dominant-script routing decision.

    A clear script majority (outside the near-50/50 band, ``_dominant_script``)
    routes to a single corpus slice AND that language's threshold together —
    the same classifier decides both, so they can never disagree (closes the
    #601 mismatch: the old ``_is_cjk_query`` any-char threshold predicate and
    ``indexer.detect_lang``'s ratio-based corpus predicate could pick
    different languages for the same query).

    A near-50/50 query (``"mixed"``) falls back to union: BOTH corpus slices
    are searched and the gate passes if EITHER clears its own threshold. The
    caller only ever receives one ``(ranked, threshold)`` pair — never a
    cross-language merge, so downstream context building always stays
    single-language (PRD #284) — chosen as follows:
      * exactly one route passes -> that route (the one that actually clears);
      * both pass, or both fail -> the route matching the query's own script
        lean (``_script_ratio`` >= 0.5 -> zh), for a deterministic, explicable
        choice when the routes already agree on pass/fail.
    """
    script = _dominant_script(question)
    if script != "mixed":
        threshold = _SCORE_THRESHOLD_ZH if script == "zh" else _SCORE_THRESHOLD
        return _search_for_lang(question, script, exclude_qa), threshold

    zh_ranked = _search_for_lang(question, "zh", exclude_qa)
    en_ranked = _search_for_lang(question, "en", exclude_qa)
    zh_top = zh_ranked[0][1] if zh_ranked else 0.0
    en_top = en_ranked[0][1] if en_ranked else 0.0
    zh_pass = zh_top >= _SCORE_THRESHOLD_ZH
    en_pass = en_top >= _SCORE_THRESHOLD

    if zh_pass and not en_pass:
        return zh_ranked, _SCORE_THRESHOLD_ZH
    if en_pass and not zh_pass:
        return en_ranked, _SCORE_THRESHOLD
    if _script_ratio(question) >= 0.5:
        return zh_ranked, _SCORE_THRESHOLD_ZH
    return en_ranked, _SCORE_THRESHOLD


def _wiki_page_path_for_section(sec: indexer.Section) -> str | None:
    """Return the repo-relative path to the wiki page a retrieved Section came from.

    Issue #266: the reader UI renders a clickable citation that opens this file
    in-page via ``GET /read/file`` (which whitelists ``wiki/``). The path is
    decided here (server-side, CODING_STANDARD §12.5) because the client cannot
    reconstruct the wiki ``type`` subdir from the bare ``slug#heading`` Section id.

    The Section was indexed FROM ``wiki/<subdir>/<slug>.md`` (same layout
    ``_derived_from_for_section`` resolves), so the path resolves by construction.
    Returns ``None`` when the page ``type`` is absent/unknown (e.g. a docs-style
    test corpus), so the UI degrades to plain, non-clickable citation text.
    """
    page_type = (sec.metadata or {}).get("type")
    subdir = {"entity": "entities", "concept": "concepts", "qa": "qa"}.get(page_type)
    if subdir is None:
        return None
    # Forward slashes: this is a /read/file relpath (URL-style), not an OS path.
    return f"wiki/{subdir}/{sec.file}.md"


def _derived_from_for_section(sec: indexer.Section) -> list[dict] | None:
    """Return the docs/ citation chain for a retrieved wiki Section.

    Phase 6 Slice 6-3 — closes ADR-0006 §"PROMPT.md citation contract evolution"
    deferred to Phase 6. Reads the parent wiki page's ``frontmatter.sources``
    and converts each ``"<docs-filename>#<heading-slug>"`` entry into a
    ``{source, heading}`` dict (matches ``schemas.CitationRef`` shape so Pydantic
    can validate at the route boundary without an explicit model conversion).

    Return contract:
        * Populated list — parent ``frontmatter.sources`` had entries.
        * ``None`` — parent had ``sources: []`` (or absent / non-list), the
          parent file is missing, or parsing parent frontmatter raised.
          ``None`` is semantically distinct from ``[]`` per PRD #78 Q4.

    Fail-soft contract: never raises. Parse failures degrade to ``None`` and
    emit a ``parse_warning`` log entry (reuses the existing kind; no new
    log_kinds entry per the slice constraints).

    Parent-page resolution: the wiki layout is ``<wiki>/<type-subdir>/<slug>.md``
    where ``<type-subdir>`` comes from ``Section.metadata["type"]`` (one of
    ``"entity"``, ``"concept"``, ``"qa"``). If ``type`` is absent or unknown
    we cannot construct the path -> degrade to ``None`` without a log entry
    (this is a structural issue surfaced elsewhere by C10 lint / parse_warning
    at index time).
    """
    metadata = sec.metadata or {}
    page_type = metadata.get("type")
    if page_type not in ("entity", "concept", "qa"):
        return None

    subdir_name = {"entity": "entities", "concept": "concepts", "qa": "qa"}[page_type]
    # Read indexer.WIKI_DIR via the module so tests monkeypatching
    # ``indexer.WIKI_DIR`` (the autouse conftest fixture does this) are
    # honoured at query time.
    parent_path: Path = indexer.WIKI_DIR / subdir_name / f"{sec.file}.md"

    if not parent_path.exists():
        log_event(
            "parse_warning",
            f"derived_from: parent wiki page missing for section={sec.id} path={parent_path.name}",
        )
        return None

    try:
        raw = parent_path.read_text(encoding="utf-8")
    except OSError as exc:
        log_event(
            "parse_warning",
            f"derived_from: parent read failed for section={sec.id} exc={type(exc).__name__}",
        )
        return None

    # Skip optional sentinel HTML comment preceding the frontmatter block.
    lines = raw.splitlines()
    dash_indices = [i for i, ln in enumerate(lines) if ln.strip() == "---"]
    if len(dash_indices) < 2:
        log_event(
            "parse_warning",
            f"derived_from: parent frontmatter fences missing for section={sec.id} "
            f"path={parent_path.name}",
        )
        return None

    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:
        log_event(
            "parse_warning",
            f"derived_from: parent frontmatter YAML invalid for section={sec.id} "
            f"path={parent_path.name} exc={type(exc).__name__}",
        )
        return None

    if not isinstance(parsed, dict):
        log_event(
            "parse_warning",
            f"derived_from: parent frontmatter not a mapping for section={sec.id} "
            f"path={parent_path.name}",
        )
        return None

    sources_field = parsed.get("sources")
    # Empty / absent / non-list -> semantic None per PRD #78 Q4.
    if not isinstance(sources_field, list) or not sources_field:
        return None

    # Convert each "docs-filename#heading-slug" entry to a CitationRef-shaped
    # dict. Entries that don't contain "#" or are not strings are skipped
    # silently — surfacing them belongs to C2 red-link lint, not retrieval.
    refs: list[dict] = []
    for entry in sources_field:
        if not isinstance(entry, str) or "#" not in entry:
            continue
        source, _, heading = entry.partition("#")
        refs.append({"source": source, "heading": heading})

    return refs or None


def _call_llm_with_error_handling(question: str, prompt_text: str) -> str:
    """Invoke the LLM and map OpenAI exceptions to a transport-agnostic LLMError.

    Error mapping (ADR-0015 — status table moves to the HTTP route):
      - APITimeoutError, RateLimitError → LLMError(retryable=True)
      - AuthenticationError            → LLMError(retryable=False)
      - Any other APIError             → LLMError(retryable=False)

    Each error is logged to wiki/log.md via log_event with the appropriate
    kind tag (openai_transient | openai_auth | openai_api) BEFORE raising,
    so the log always captures the failure regardless of the caller's
    exception-handling path.

    Callers that render via HTTP map retryable → 503, non-retryable → 500.
    Other adapters (SSE, MCP, CLI) render the LLMError in their own way.
    """
    truncated = question[:60].replace('"', "'")
    try:
        response = get_llm().invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ]
        )
        return response.content
    except (openai.APITimeoutError, openai.RateLimitError) as exc:
        log_event(
            "chat_error",
            f'"{truncated}" kind=openai_transient exc={type(exc).__name__}',
        )
        raise LLMError(
            retryable=True,
            message="LLM service temporarily unavailable, please retry.",
        ) from exc
    except openai.AuthenticationError as exc:
        log_event(
            "chat_error",
            f'"{truncated}" kind=openai_auth exc={type(exc).__name__}',
        )
        raise LLMError(
            retryable=False,
            message="LLM service auth failed (check OPENAI_API_KEY).",
        ) from exc
    except openai.APIError as exc:
        log_event(
            "chat_error",
            f'"{truncated}" kind=openai_api exc={type(exc).__name__}',
        )
        raise LLMError(
            retryable=False,
            message=f"LLM service error: {exc!s}",
        ) from exc


def _write_chat_log(
    question: str,
    ranked: list[tuple[indexer.Section, float]],
) -> None:
    """Append a chat log entry to wiki/log.md.

    Format:
        ## [<ts>] chat | "<truncated_query>" top=<section_id>:<score>
    """
    truncated = question[:60].replace('"', "'")
    if ranked:
        top_sec, top_score = ranked[0]
        summary = f'"{truncated}" top={top_sec.id}:{round(top_score, 3)}'
    else:
        summary = f'"{truncated}" top=none'
    log_event("chat", summary)
