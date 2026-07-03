"""FastMCP server exposing the knowledge base over stdio.

Phase 12 (ADR-0016 / ADR-0017).  Wraps ``markdown_kb`` and ``vector_rag`` deep
modules directly (NOT the Gateway).  Exposes:
  - ``kb_ask_v1``            — grounded-answer tool (LLM draft + Grounding Check)
  - ``kb_search_v1``         — raw-evidence search with no LLM call
  - ``kb_read_hot_v1``       — read working-memory hot cache
  - ``kb_save_hot_v1``       — persist working-memory hot cache
  - ``kb_capture_v1``        — author a Markdown Source from conversation to docs/ (Slice 4, #230)
  - ``kb_ingest_v1``         — single-Source ingest; auto-routes large Sources to async job (Fix 1b)
  - ``kb_ingest_start_v1``   — submit large Source to background job; returns job_id immediately (Fix 1b)
  - ``kb_ingest_status_v1``  — poll a background ingest job for completion (Fix 1b)
  - ``kb_index_v1``          — rebuild the Section Index via build_index (Slice 5, #231)
  - ``kb_lint_v1``           — run the Lint Pass via run_lint (Slice 5, #231)
  - ``kb_import_v1``         — import a local file into docs/ via the Import deep module (Slice 6, #233)

Launch via ``python -m kb_mcp`` (stdio transport, Claude Desktop compatible).

Server ``instructions`` (~200 tokens) guide the MCP host on:
  - tool-choice (kb_ask for grounded answers; kb_search for raw evidence)
  - ``stack`` default (always start with ``wiki``; only switch to ``rag`` on
    explicit user instruction)
  - Cannot-Confirm guidance (surface ``grounding.reason`` to the user; do NOT
    retry to force an answer)
  - LLMError guidance (isError=True means the LLM service is unavailable;
    code='LLM_UNAVAILABLE' is retryable, 'LLM_ERROR' is not)
  - Large-Source async flow (kb_ingest_v1 auto-routes; or use kb_ingest_start_v1
    + kb_ingest_status_v1 directly for very large Sources)
"""

from __future__ import annotations

import contextlib
import json
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from .freshness import reload_if_stale
from .hot_cache import read_hot, save_hot
from .normalizer import normalize_hybrid_results, normalize_rag_results, normalize_wiki_results

# ---------------------------------------------------------------------------
# Server-level instructions (~200 tokens)
# ---------------------------------------------------------------------------
_INSTRUCTIONS = (
    "You are connected to a grounded knowledge-base assistant.\n\n"
    "Available tools:\n"
    "- kb_ask_v1: Ask a question and receive a grounded answer (LLM synthesis "
    "with citation and grounding check).  Use this as the default for "
    "user questions.  Returns {stack, answer, citations, grounding}.\n"
    "- kb_search_v1: Retrieve raw Sections or Chunks from the KB index with no "
    "LLM synthesis.  Use this when the user wants to see the raw evidence or "
    "when you want to reason over sources yourself before composing an answer.\n"
    "- kb_read_hot_v1: Read working-memory hot cache (wiki/hot.md).  "
    "Call this at session start to recover where the previous session left off.  "
    "Returns empty string on the first session — that is normal, not an error.\n"
    "- kb_save_hot_v1: Persist a working-memory summary to the hot cache.  "
    "Call this at session end (or at a natural checkpoint) with a ~500-word "
    "summary composed by you.  The server only persists the bytes; you compose "
    "the summary.\n"
    "- kb_capture_v1: Author a Markdown Source from this conversation and "
    "persist it to docs/.  Use this to turn session reasoning into a permanent "
    "KB Source.  Capture skips Import — content is already canonical Markdown.  "
    "Provenance frontmatter is stamped automatically.  After capturing, run "
    "kb_ingest_v1 / kb_index_v1 to make it retrievable.  "
    "Returns {ok: true, path: str}.  Unsafe filenames return isError with "
    "code='CAPTURE_REJECTED'.\n"
    "- kb_import_v1: Import a local file from the filesystem into docs/ by "
    "reading it and converting it via the Import deep module.  Supported: "
    ".html, .txt, .md.  Returns {ok: true, source: str, status: str}.  "
    "Bad paths / unsafe basenames return isError with code='IMPORT_REJECTED'.\n\n"
    "Stack guidance:\n"
    "- Always start with stack='wiki' (curated BM25 index).  "
    "Only switch to stack='rag' when the user explicitly asks to use the "
    "Vector RAG arm for comparison.  "
    "Use stack='hybrid' when the user wants fused BM25 + dense retrieval "
    "(RRF over the curated wiki, grounded answer with citations).\n\n"
    "Cannot-Confirm guidance:\n"
    "- When kb_ask returns grounding.passed=false, the KB cannot support the "
    "answer.  Surface grounding.reason to the user and do NOT retry.  "
    '"Cannot Confirm" is a valid, expected KB boundary — not a failure.\n\n'
    "LLM error guidance:\n"
    "- If kb_ask or kb_lint_v1 returns isError=true, the LLM service failed.  "
    "code='LLM_UNAVAILABLE' is transient — retry after a short wait.  "
    "code='LLM_ERROR' is non-recoverable — report the message to the user.\n\n"
    "Large-Source async ingest:\n"
    "- kb_ingest_v1 auto-routes Sources that exceed the soft token cap to a "
    "background async job and returns immediately with "
    "{status: 'routed_async', job_id: ..., note: ...}.\n"
    "- For Sources you know are large, use kb_ingest_start_v1 directly; it "
    "returns {job_id, status} immediately without waiting for the pipeline.\n"
    "- Poll kb_ingest_status_v1({job_id}) until status is 'completed' or "
    "'failed'.  On completed, result carries pages_created etc.  "
    "On failed, error carries {code, message}.  "
    "Unknown job_id returns {status: 'unknown'} — not an error."
)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(name="kb_mcp", instructions=_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Tool: kb_ask_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_ask_v1",
    description=(
        "Ask a question and receive a grounded answer synthesised by the LLM "
        "with a post-LLM Grounding Check (ADR-0004).  Use this as the default "
        "tool for user questions.\n\n"
        "Parameters:\n"
        "  query  — the user's question (required)\n"
        "  stack  — 'wiki' (curated BM25, default), 'rag' (Vector RAG), or\n"
        "           'hybrid' (BM25 + dense RRF over the curated wiki — grounded\n"
        "           answer with citations, same safety gates as wiki/rag)\n\n"
        "Returns on success: {stack, answer, citations, grounding}\n"
        "  answer      — grounded text, or the Cannot-Confirm phrase when the\n"
        "                KB cannot support the answer\n"
        "  citations   — list of {source, heading, score, content} dicts\n"
        "                (score is null for hybrid — the RRF fused score is not\n"
        "                a calibrated relevance magnitude, ADR-0018)\n"
        "  grounding   — {passed: bool, reason: str}\n"
        "                passed=false means Cannot Confirm (a success result,\n"
        "                NOT isError — it is a valid KB boundary per ADR-0016)\n\n"
        "Returns isError=true on LLM failure:\n"
        "  {code, message} where code is 'LLM_UNAVAILABLE' (retryable) or\n"
        "  'LLM_ERROR' (non-retryable, report message to user)."
    ),
)
def kb_ask_v1(
    query: Annotated[str, Field(description="The question to answer.")],
    stack: Annotated[
        Literal["wiki", "rag", "hybrid"],
        Field(
            description=(
                "Retrieval stack: 'wiki' (curated BM25, default), "
                "'rag' (Vector RAG), or 'hybrid' (BM25 + dense RRF over the "
                "curated wiki — grounded answer with citations, ADR-0018)."
            )
        ),
    ] = "wiki",
) -> Any:
    """Answer a question with LLM synthesis and a post-LLM Grounding Check.

    Routes to ``markdown_kb.app.retrieval.query()`` (wiki stack),
    ``vector_rag.app.retrieval.query()`` (rag stack), or
    ``hybrid_kb.app.query.query()`` (hybrid stack — BM25 + dense RRF fusion,
    S3 #313).  Returns a normalised dict on success.  On ``LLMError``
    (ADR-0015), returns a ``CallToolResult`` with ``isError=True`` so the MCP
    host receives a structured error payload instead of a raw exception.

    Cannot Confirm (``grounding.passed=False``) is ALWAYS a success result
    (not ``isError``).  The host must treat it as a KB boundary, not a failure.

    Defaults are enforced server-side — the MCP host MUST NOT rely on the model
    supplying default values (ADR-0016 strict schema).
    """
    from markdown_kb.app.errors import LLMError

    # Enforce defaults server-side regardless of what the host sends.
    if stack is None:
        stack = "wiki"

    try:
        if stack == "wiki":
            reload_if_stale()
            from markdown_kb.app.retrieval import query as wiki_query

            result = wiki_query(query)
        elif stack == "rag":
            from vector_rag.app.retrieval import query as rag_query  # type: ignore[import-untyped]

            result = rag_query(query)
        else:  # stack == "hybrid"
            from hybrid_kb.app.query import query as hybrid_query  # type: ignore[import-untyped]

            result = hybrid_query(query)

    except LLMError as exc:
        # ADR-0015 / ADR-0016: LLMError → structured MCP isError payload.
        # retryable=True  → code='LLM_UNAVAILABLE'
        # retryable=False → code='LLM_ERROR'
        code = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        payload = json.dumps({"code": code, "message": exc.message})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    # Map retrieval result to the MCP neutral shape.
    # result keys: answer, sources, grounding_outcome
    # sources: list of {source, heading, content, [score], [derived_from], ...}
    # grounding_outcome: GroundingOutcome(passed, reason, ...)
    # Note: score may be absent for rag and hybrid stacks (no calibrated score
    # exposed); use .get() so all three stacks share this normalization.
    grounding_outcome = result["grounding_outcome"]
    citations = [
        {
            "source": src["source"],
            "heading": src["heading"],
            "score": src.get("score"),
            "content": src["content"],
        }
        for src in result.get("sources", [])
    ]
    return {
        "stack": stack,
        "answer": result["answer"],
        "citations": citations,
        "grounding": {
            "passed": grounding_outcome.passed,
            "reason": grounding_outcome.reason,
        },
    }


# ---------------------------------------------------------------------------
# Tool: kb_search_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_search_v1",
    description=(
        "Search the knowledge base and return raw Sections or Chunks with no "
        "LLM synthesis.  Use this tool to retrieve evidence you will reason "
        "over yourself, or when the user asks to see the raw KB content.\n\n"
        "Parameters:\n"
        "  query  — the search string (required)\n"
        "  stack  — 'wiki' (curated BM25, default), 'rag' (Vector RAG), or\n"
        "           'hybrid' (BM25 + dense RRF over the curated wiki — returns\n"
        "           fused wiki Sections with no LLM synthesis, ADR-0018)\n"
        "  k      — number of results to return (1–10, default 3)\n\n"
        "Returns: {stack, results: [{id, content, score|null}]}\n"
        "  score is a BM25 float for wiki; null for rag and hybrid (no score\n"
        "  exposed — the RRF fused score is not a calibrated magnitude)."
    ),
)
def kb_search_v1(
    query: Annotated[str, Field(description="The search query string.")],
    stack: Annotated[
        Literal["wiki", "rag", "hybrid"],
        Field(
            description=(
                "Retrieval stack: 'wiki' (curated BM25, default), "
                "'rag' (Vector RAG), or 'hybrid' (BM25 + dense RRF over the "
                "curated wiki — fused Sections, no LLM synthesis, ADR-0018)."
            )
        ),
    ] = "wiki",
    k: Annotated[
        int,
        Field(
            default=3,
            ge=1,
            le=10,
            description="Number of results to return (1–10).",
        ),
    ] = 3,
) -> dict:
    """Search the knowledge base index, returning normalized results.

    Routes to ``markdown_kb.app.indexer.search`` (wiki stack),
    ``vector_rag.app.indexer.search`` (rag stack), or
    ``hybrid_kb.app.retrieval.retrieve_and_gate`` (hybrid stack — fused wiki
    Sections via RRF, S2 #312).  No LLM call on any path.

    Defaults are enforced server-side — the MCP host MUST NOT rely on the model
    supplying default values (ADR-0016 strict schema).
    """
    # Enforce defaults server-side regardless of what the host sends.
    if stack is None:
        stack = "wiki"
    if k is None:
        k = 3
    k = max(1, min(10, k))

    if stack == "wiki":
        reload_if_stale()
        from markdown_kb.app.indexer import search as wiki_search

        hits = wiki_search(query, k=k)
        results = normalize_wiki_results(hits)
    elif stack == "rag":
        from vector_rag.app.indexer import search as rag_search

        chunks = rag_search(query, k=k)
        results = normalize_rag_results(chunks)
    else:  # stack == "hybrid"
        # Lazy-load both arms' indexes before calling the retrieval core via
        # the public warmup seam (CODING_STANDARD §2.4 — no cross-package
        # private import). ``ensure_indexes_loaded`` checks each arm
        # independently and is a no-op when both are already warm (common path).
        from hybrid_kb.app.query import ensure_indexes_loaded  # type: ignore[import-untyped]
        from hybrid_kb.app.retrieval import retrieve_and_gate  # type: ignore[import-untyped]

        ensure_indexes_loaded()
        gate = retrieve_and_gate(query, top_k=k)
        results = normalize_hybrid_results(gate["sections"])

    return {"stack": stack, "results": results}


# ---------------------------------------------------------------------------
# Tool: kb_read_hot_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_read_hot_v1",
    description=(
        "Read the working-memory hot cache (wiki/hot.md).  "
        "Call this at session start to recover where the previous session left off.\n\n"
        "Returns: {content: str}\n"
        "  content is the full text of wiki/hot.md, or '' on the first session "
        "(file absent is a normal state, not an error).\n\n"
        "This is a TOOL (agent-initiated), not a resource — the agent decides "
        "when to call it (L0 of the read-depth budget per ADR-0016)."
    ),
)
def kb_read_hot_v1() -> dict:
    """Return the hot-cache contents, or empty string when absent.

    Uses the module-level ``HOT_PATH`` which tests monkeypatch to a tmp dir.
    """
    content = read_hot()
    return {"content": content}


# ---------------------------------------------------------------------------
# Tool: kb_save_hot_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_save_hot_v1",
    description=(
        "Persist a working-memory summary to the hot cache (wiki/hot.md).\n\n"
        "Parameters:\n"
        "  summary — the ~500-word working-memory summary (required).  "
        "The host composes the summary; the server only persists the bytes.\n\n"
        "Returns: {ok: true} on success.\n\n"
        "Writes atomically (tmp-file + os.replace) so a crash mid-write never "
        "leaves a partial file.  Overwrites any previous hot.md."
    ),
)
def kb_save_hot_v1(
    summary: Annotated[str, Field(description="The working-memory summary to persist.")],
) -> dict:
    """Atomically persist ``summary`` to the hot-cache file.

    Uses the module-level ``HOT_PATH`` (from hot_cache) which tests monkeypatch.
    """
    save_hot(summary)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Tool: kb_index_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_index_v1",
    description=(
        "Rebuild the Section Index from the curated wiki (ADR-0003).  "
        "Call this after curating wiki pages (adding, editing, or removing pages) "
        "so the BM25 index reflects the latest content.\n\n"
        "Takes no parameters.\n\n"
        "Returns: {files_indexed, sections_indexed}\n"
        "  files_indexed   — number of source files indexed\n"
        "  sections_indexed — total Sections in the rebuilt index"
    ),
)
def kb_index_v1() -> dict:
    """Rebuild the Section Index by calling build_index() from the deep module.

    Thin wrapper: no defaults to enforce (zero-arg tool).  The deep module
    ``markdown_kb.app.indexer.build_index`` scans SOURCE_DIRS, persists
    ``.kb/index.json`` atomically, and returns ``(files_indexed, sections_indexed)``.

    No LLMError can arise here — build_index is a pure local operation.
    """
    from markdown_kb.app.indexer import build_index

    files_indexed, sections_indexed = build_index()
    return {"files_indexed": files_indexed, "sections_indexed": sections_indexed}


def _routed_fill_hint(route: str | None) -> str:
    """Render a Routed check's navigation hint as plain text (ADR-0027).

    Mirrors ``kb_cli.main._routed_fill_hint`` (same shared taxonomy, same
    text) — kept as a small local duplicate rather than a new shared module,
    since the two surfaces already render tier/route information
    independently (Console's copy differs in wording too).
    """
    if route == "import":
        return "kb import <file> && kb ingest [source]"
    return f"route: {route}"  # pragma: no cover — no other route exists yet


# ---------------------------------------------------------------------------
# Tool: kb_lint_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_lint_v1",
    description=(
        "Run the Lint Pass over the wiki and return structured findings "
        "(ADR-0003, Phase 5).  Use this to retrieve lint findings so you "
        "can reason over contradictions, stale pages, coverage gaps, and "
        "propose curator actions.\n\n"
        "Parameters:\n"
        "  include_c5 — whether to run the LLM-backed C5 page-pair "
        "contradiction check (default true).  Pass false to skip C5 and "
        "receive only the fast local checks.\n\n"
        "Returns on success: "
        "{report_path, findings, summary, check_errors, axis_groups}\n"
        "  findings   — structured per-check finding lists\n"
        "  summary    — {total_findings, findings_by_check, llm_calls, "
        "cost_usd, c5_pairs_capped, generated_at}\n"
        "  check_errors — dict of check_id → error string for any check that "
        "raised (other checks still ran — continue-on-error semantics)\n"
        "  axis_groups — the same findings reshaped by Lint Axis (Freshness, "
        "Coherence, Coverage, Lifecycle, in that order): a list of "
        "{axis, checks: [{code, label, count}]} so you can reason over the "
        "wiki's health by category without re-deriving the check→axis "
        "mapping yourself.\n"
        "  findings.promotion_candidates[] (C8) and "
        "findings.stale_filed_answers[] (C9) each additionally carry a "
        "'path' (wiki/qa/<slug>.md); C9 entries also carry 'question' "
        "(read from the page; null if unreadable) so you can report what "
        "needs curating.  There is no MCP tool to promote or discard a "
        "Filed Answer — gates resolve on human surfaces only, Console or "
        "CLI `kb qa promote|discard` (ADR-0026); report findings to your "
        "human instead of acting on them.\n\n"
        "  findings.coverage_gaps[] (C1) and findings.red_links[] (C2) each "
        "additionally carry a 'fill_via' text hint (e.g. 'kb import <file> "
        "&& kb ingest [source]') — these are Routed findings (ADR-0027): the "
        "system cannot fix them itself (a coverage gap or red link has no "
        "Source to ground a draft against), so there is no tool call that "
        "resolves them either.  Report the finding and the hint to your "
        "human instead of attempting to synthesise a fix.\n\n"
        "Returns isError=true when the C5 LLM call fails catastrophically:\n"
        "  {code, message} where code is 'LLM_UNAVAILABLE' (retryable) or\n"
        "  'LLM_ERROR' (non-retryable, report message to user).\n"
        "  Individual per-pair LLM errors within C5 are NOT isError — they "
        "are recorded in check_errors['c5'] and the check returns partial "
        "results (continue-on-error per the deep module contract)."
    ),
)
def kb_lint_v1(
    include_c5: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether to run C5 (LLM-backed page-pair contradiction check). "
                "Pass false to skip C5 and receive only fast local checks."
            ),
        ),
    ] = True,
) -> Any:
    """Run the Lint Pass via run_lint() from the deep module.

    Thin wrapper.  Defaults are enforced server-side (ADR-0016): ``include_c5``
    defaults to True so the full lint suite runs by default.

    On ``LLMError`` (ADR-0015), returns a ``CallToolResult`` with ``isError=True``
    so the MCP host receives a structured error payload.  This covers the case
    where the *entire* C5 check fails (e.g. LLM unreachable before any pair is
    judged).  Per-pair C5 failures are handled inside the deep module via
    continue-on-error semantics and appear in ``check_errors['c5']`` of the
    success payload — they do NOT trigger isError.

    ``axis_groups`` is added on top of ``run_lint()``'s ``LintResponse`` by
    reshaping ``response.findings`` through the shared
    ``group_findings_by_axis`` helper and its ``LINT_CHECK_TAXONOMY`` (issue
    #361 S1) — the same taxonomy the CLI's ``kb lint`` and the
    ``lint-report.md`` renderer consume, so the three surfaces never disagree
    on check→axis mapping (ADR-0017 interface parity).

    C8/C9 visibility (issue #377 / ADR-0026 decision 3): ``qa_view`` (a
    read-only helper, not a mutation) adds a ``path`` key to every
    ``promotion_candidates`` (C8) and ``stale_filed_answers`` (C9) entry, and
    a ``question`` key to C9 entries (``QaStalenessFinding`` carries no
    question field; C8's already does via ``PromotionCandidateFinding``).
    This is added here rather than on the Pydantic finding models themselves
    so the change stays inside kb_mcp/kb_cli — markdown_kb's lint/schema
    modules are untouched, keeping this slice parallel-safe with sibling
    tier-B slices editing ``qa.py`` / ``lint.py`` concurrently. No tool
    resolves a Gated remediation from these findings — MCP sees everything
    and approves nothing (ADR-0026).

    C1/C2 Routed navigation hint (tier-B S7, issue #383, ADR-0027): every
    ``coverage_gaps`` (C1) and ``red_links`` (C2) entry gains a ``fill_via``
    key — plain text sourced from the SAME shared ``remediation_for(...).
    route`` the Console's "Fill via Import" control and ``kb lint`` render,
    since a Routed check has no draft and no gate for a tool to resolve
    (there is nothing here for an agent to act on directly — report the
    finding and this hint to the human instead).
    """
    from markdown_kb.app.errors import LLMError
    from markdown_kb.app.lint import group_findings_by_axis, remediation_for, run_lint

    from . import qa_view

    # Enforce default server-side
    if include_c5 is None:
        include_c5 = True

    try:
        response = run_lint(include_c5=include_c5)
    except LLMError as exc:
        # ADR-0015 / ADR-0016: LLMError → structured MCP isError payload.
        code = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        payload = json.dumps({"code": code, "message": exc.message})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    # Serialise the Pydantic LintResponse to a plain dict for MCP transport.
    # model_dump() converts nested Pydantic models to plain dicts/lists;
    # mode="json" ensures non-JSON-native types (e.g. datetime) are strings.
    payload = response.model_dump(mode="json")
    payload["axis_groups"] = [
        {
            "axis": axis_group.axis,
            "checks": [
                {"code": meta.code, "label": meta.label, "count": len(finding_list)}
                for meta, finding_list in axis_group.checks
            ],
        }
        for axis_group in group_findings_by_axis(response.findings)
    ]

    for candidate in payload["findings"]["promotion_candidates"]:
        candidate["path"] = qa_view.display_path(candidate["slug"])
    for stale in payload["findings"]["stale_filed_answers"]:
        stale["path"] = qa_view.display_path(stale["page_slug"])
        page = qa_view.read_qa_page(stale["page_slug"])
        stale["question"] = page.question if page is not None else None

    # Routed (tier-B S7, issue #383, ADR-0027): C1/C2 have no gate-resolving
    # tool because there is nothing to gate — route the same taxonomy value
    # `kb lint` renders as text, per finding, so an agent can report it
    # without re-deriving the hint.
    coverage_gap_hint = _routed_fill_hint(remediation_for("C1").route)
    for gap in payload["findings"]["coverage_gaps"]:
        gap["fill_via"] = coverage_gap_hint
    red_link_hint = _routed_fill_hint(remediation_for("C2").route)
    for red_link in payload["findings"]["red_links"]:
        red_link["fill_via"] = red_link_hint

    return payload


# ---------------------------------------------------------------------------
# Tool: kb_ingest_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_ingest_v1",
    description=(
        "Ingest a single named Source from docs/ synchronously, synthesising "
        "wiki pages and running the Grounding Check.  Use this to curate one "
        "Source at a time and see the result before moving on.\n\n"
        "Parameters:\n"
        "  source — bare filename of the Source to ingest (required);\n"
        "           e.g. 'refund_policy.md'\n\n"
        "Returns on success: "
        "{source, pages_created, pages_overwritten, grounding_failed_pages, "
        "failed, status}\n"
        "  pages_created         — list of wiki page paths written for the first time\n"
        "  pages_overwritten     — list of paths that already existed and were "
        "overwritten (cross-call slug collision is visible here, not silent — "
        "#54 / CODING_STANDARD §12.8)\n"
        "  grounding_failed_pages — list of page slugs that were written but failed "
        "the Grounding Check (status=failed_grounding); a non-empty list is a "
        "SUCCESS result (not isError) — it means the KB accepted the page with "
        "a failed-grounding marker, which is the ADR-0004 fail-soft outcome\n"
        "  failed                — True when the source was not found, could not be "
        "parsed, or was rejected by the size guard (non-LLM failure)\n"
        "  reason                — present only on failure: a human-readable cause "
        "(e.g. the Source exceeds the ingest size limit). Report it to the user "
        "instead of retrying\n"
        "  status                — 'created', 'updated', 'skipped', or 'failed'\n\n"
        "Returns isError=true on LLM failure:\n"
        "  {code, message} where code is 'LLM_UNAVAILABLE' (retryable) or\n"
        "  'LLM_ERROR' (non-retryable, report message to user).\n\n"
        "Progress notifications are emitted during the run so the host does not "
        "time out on a slow Source.  No batch parameter is exposed — loop over "
        "Sources one at a time."
    ),
)
async def kb_ingest_v1(
    source: Annotated[
        str, Field(description="Bare filename of the Source to ingest (e.g. 'refund_policy.md').")
    ],
    ctx: Context,
) -> Any:
    """Ingest a single Source synchronously and return a neutral result dict.

    Emits MCP progress notifications before and after the pipeline step so the
    Claude Desktop host does not time out on slow Sources.

    Cannot-Confirm / grounding-failed outcome: reported as a SUCCESS result
    (not isError) — ``grounding_failed_pages`` will be non-empty.

    LLMError: caught here and returned as isError with code ∈
    {LLM_UNAVAILABLE, LLM_ERROR} per ADR-0015.

    Cross-call slug collision visibility (#54 / CODING_STANDARD §12.8):
    ``pages_overwritten`` is populated when a page already existed on disk and
    was overwritten.  The caller can detect a cross-call slug collision by
    checking whether pages_overwritten is non-empty.
    """
    from markdown_kb.app.errors import LLMError
    from markdown_kb.app.indexer import split_frontmatter
    from markdown_kb.app.ingest import _should_route_async, aingest_sources

    async def _progress(n: float, total: float, message: str) -> None:
        """Emit a progress notification, silently no-op when no request context.

        The in-process test harness and Claude Desktop calls without a progress
        token both lack a request context; swallowing the ValueError keeps
        the tool functional in both environments.
        """
        with contextlib.suppress(Exception):
            await ctx.report_progress(n, total, message=message)

    await _progress(0, 3, message=f"Starting ingest for {source!r}")

    # Fix 1b: large-source routing.
    # Read the Source text and check whether it exceeds the per-Source SOFT token
    # cap (KB_INGEST_MAX_TOKENS).  If so, do NOT run the pipeline inline — that
    # would risk the MCP host tool-call timeout (-32001) on a huge Source.
    # Instead, auto-submit a background job and return immediately so the host can
    # poll kb_ingest_status_v1 for the result.
    #
    # Small Sources (<= cap) continue through the original inline await path below.
    # Source not-found / unreadable falls through to the inline path (aingest_sources
    # will record the failure in batch.failed_sources, as before).
    try:
        import markdown_kb.app._paths as _paths_mod

        source_path = _paths_mod.DOCS_DIR / source
        if source_path.exists():
            raw_text = source_path.read_text(encoding="utf-8")
            _, source_content = split_frontmatter(raw_text)
            if _should_route_async(source_content):
                # Over soft cap → submit background job and return immediately.
                from . import ingest_jobs

                job = ingest_jobs.submit(source)
                return {
                    "status": "routed_async",
                    "job_id": job.job_id,
                    "note": (
                        f"Source {source!r} exceeds the soft token cap and has been "
                        "submitted as a background ingest job.  "
                        "Poll kb_ingest_status_v1 with the job_id to track progress "
                        "and retrieve the result."
                    ),
                }
    except Exception:  # noqa: BLE001
        # If the routing check itself fails (e.g. unexpected IO error), fall
        # through to the regular inline path so behaviour is unchanged.
        pass

    try:
        await _progress(1, 3, message=f"Running synthesis pipeline for {source!r}")
        # Fix 1a: await aingest_sources so the stdio event loop is not blocked
        # during multi-minute ingest runs (avoids -32001 timeout from Claude Desktop).
        batch = await aingest_sources([source])
    except LLMError as exc:
        # ADR-0015 / ADR-0016: LLMError → structured MCP isError payload.
        code = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        payload = json.dumps({"code": code, "message": exc.message})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    await _progress(2, 3, message=f"Ingest pipeline complete for {source!r}")

    # Map IngestBatchResult → neutral MCP result dict.
    # batch.results[0] is the outcome for our single Source (when successful).
    # batch.failed_sources contains the source name on non-LLM failure.
    # batch.pages_with_failed_grounding lists slugs that failed the Grounding Check.
    if batch.failed_sources and source in batch.failed_sources:
        result_payload = {
            "source": source,
            "pages_created": [],
            "pages_overwritten": [],
            "grounding_failed_pages": [],
            "failed": True,
            "status": "failed",
        }
        # Surface a per-source failure reason when the deep module recorded one
        # (e.g. the size guard) so the host sees *why*, not just failed=True.
        reason = batch.failed_reasons.get(source)
        if reason:
            result_payload["reason"] = reason
    elif batch.results:
        src_result = batch.results[0]
        result_payload = {
            "source": source,
            "pages_created": src_result.pages_created,
            "pages_overwritten": src_result.pages_updated,  # updated = overwritten
            "grounding_failed_pages": batch.pages_with_failed_grounding,
            "failed": False,
            "status": src_result.status,
        }
    else:
        # Skipped (hash-match no-op)
        skipped = batch.skipped_sources[0] if batch.skipped_sources else None
        result_payload = {
            "source": source,
            "pages_created": [],
            "pages_overwritten": [],
            "grounding_failed_pages": [],
            "failed": False,
            "status": skipped.status if skipped else "skipped",
        }

    await _progress(3, 3, message=f"Done: {source!r}")
    return result_payload


# ---------------------------------------------------------------------------
# Tool: kb_ingest_start_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_ingest_start_v1",
    description=(
        "Submit a Source for background ingest and return a job_id immediately.\n\n"
        "Use this tool instead of kb_ingest_v1 when the Source is large (e.g. > 64 000 "
        "tokens) and you want to avoid blocking the host tool-call timeout (-32001).  "
        "The ingest pipeline runs as a background task; poll kb_ingest_status_v1 for "
        "progress and the final result.\n\n"
        "Parameters:\n"
        "  source — bare filename of the Source to ingest (required);\n"
        "           e.g. 'large_manual.md'\n\n"
        "Returns immediately: {job_id, status}\n"
        "  job_id — opaque string; pass to kb_ingest_status_v1 to poll\n"
        "  status — 'submitted' or 'working' (the pipeline is already running)\n\n"
        "Note: kb_ingest_v1 automatically calls this for Sources above the soft "
        "token cap (KB_INGEST_MAX_TOKENS, default 64 000 tokens)."
    ),
)
async def kb_ingest_start_v1(
    source: Annotated[
        str,
        Field(description="Bare filename of the Source to ingest (e.g. 'large_manual.md')."),
    ],
) -> dict:
    """Submit a Source for background ingest; return job_id immediately.

    Delegates to ``ingest_jobs.submit(source)`` which schedules an asyncio.Task
    and returns a Job without awaiting it.  The caller must use
    ``kb_ingest_status_v1`` to poll for completion.

    MUST be called from an async context (inside the MCP server's event loop)
    so that ``asyncio.create_task`` in ``ingest_jobs.submit`` is valid.
    """
    from . import ingest_jobs

    job = ingest_jobs.submit(source)
    return {"job_id": job.job_id, "status": job.status}


# ---------------------------------------------------------------------------
# Tool: kb_ingest_status_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_ingest_status_v1",
    description=(
        "Poll the status of a background ingest job started by kb_ingest_start_v1.\n\n"
        "Parameters:\n"
        "  job_id — the opaque identifier returned by kb_ingest_start_v1 (required)\n\n"
        "Returns: {job_id, status, progress, result?|error?}\n"
        "  status   — 'submitted' | 'working' | 'completed' | 'failed' | 'unknown'\n"
        "  progress — [done, total] integers (e.g. [0, 1] during run, [1, 1] on done)\n"
        "  result   — present when status='completed'; same shape as kb_ingest_v1 result\n"
        "  error    — present when status='failed'; {code, message}\n"
        "             code is 'LLM_UNAVAILABLE' (retryable), 'LLM_ERROR' (non-retryable),\n"
        "             or 'INGEST_ERROR' (unexpected pipeline failure)\n\n"
        "An unknown job_id returns {status: 'unknown'} — not an error (isError=false).\n"
        "Poll every 1–5 seconds until status is 'completed' or 'failed'."
    ),
)
def kb_ingest_status_v1(
    job_id: Annotated[
        str,
        Field(description="Job identifier returned by kb_ingest_start_v1."),
    ],
) -> dict:
    """Look up a background ingest job; return its current state.

    Delegates to ``ingest_jobs.status(job_id)``.  When the job_id is unknown,
    returns ``{job_id, status: 'unknown'}`` — NEVER raises, NEVER isError.
    """
    from . import ingest_jobs

    job = ingest_jobs.status(job_id)
    if job is None:
        return {"job_id": job_id, "status": "unknown"}

    payload: dict = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": list(job.progress),
    }
    if job.result is not None:
        payload["result"] = job.result
    if job.error is not None:
        payload["error"] = job.error
    return payload


# ---------------------------------------------------------------------------
# Tool: kb_capture_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_capture_v1",
    description=(
        "Author a Markdown Source directly from this conversation and persist it "
        "to docs/.  Use this when you want to turn session reasoning into a "
        "permanent KB Source without leaving the conversation.\n\n"
        "Parameters:\n"
        "  filename — plain basename for the new Source (required, e.g. 'my_note.md').\n"
        "             Must not contain path separators, '..', or control characters.\n"
        "  content  — the Markdown body of the Source (required).\n\n"
        "Returns on success: {ok: true, path: str}\n"
        "  path — absolute path of the written file in docs/.\n\n"
        "Returns isError=true on rejection:\n"
        "  {code: 'CAPTURE_REJECTED', message: str}\n"
        "  Filename validation failures (traversal, separators) produce this error.\n\n"
        "Capture skips Import — content is assumed to be canonical Markdown already.\n"
        "Mandatory provenance frontmatter (origin/created_at/authored_by) is stamped\n"
        "automatically by the server; the caller must NOT include it in content.\n\n"
        "The captured Source flows into the normal Ingest → Index lifecycle via "
        "kb_ingest_v1 / kb_index_v1 — Capture only writes the Source to disk."
    ),
)
def kb_capture_v1(
    filename: Annotated[
        str, Field(description="Plain basename for the new Source (e.g. 'note.md').")
    ],
    content: Annotated[str, Field(description="The Markdown body of the Source.")],
) -> Any:
    """Write a Markdown Source to docs/ with mandatory provenance frontmatter.

    Delegates to ``markdown_kb.app.capture.capture_source``.  On
    ``ValueError`` (unsafe filename), returns a ``CallToolResult`` with
    ``isError=True`` so the MCP host receives a structured error payload
    instead of a raw exception.
    """
    from markdown_kb.app.capture import capture_source

    try:
        target = capture_source(filename, content)
    except ValueError as exc:
        payload = json.dumps({"code": "CAPTURE_REJECTED", "message": str(exc)})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    return {"ok": True, "path": str(target)}


# ---------------------------------------------------------------------------
# Tool: kb_import_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_import_v1",
    description=(
        "Import a local file into the knowledge base by reading it from the "
        "filesystem and converting it to a docs/ Source.  Reuses the Import "
        "deep module (``import_path``) so format conversion is always "
        "programmatic — the same path the CLI ``kb import`` uses.\n\n"
        "Parameters:\n"
        "  path — absolute path to the local file to import (required).\n"
        "         Supported formats: .html, .txt, .md.\n"
        "         The basename must be traversal-safe (no '#', ':', separators,\n"
        "         control characters, or bidi override codepoints).\n\n"
        "Returns on success: {ok: true, source: str, status: str}\n"
        "  source — basename of the written docs/ Source (e.g. 'note.md').\n"
        "  status — 'created' (fresh write), 'updated' (hash-drift overwrite),\n"
        "           or 'skipped' (hash-match no-op).\n\n"
        "Returns isError=true on rejection:\n"
        "  {code: 'IMPORT_REJECTED', message: str}\n"
        "  Covers: file not found, traversal-unsafe basename, unsupported\n"
        "  extension, empty/oversized file, and other ImportPathError cases.\n\n"
        "Reading an arbitrary local path is safe under the single-operator\n"
        "posture (ADR-0017).  After importing, run kb_ingest_v1 / kb_index_v1\n"
        "to make the new Source retrievable."
    ),
)
def kb_import_v1(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to the local file to import "
                "(e.g. 'C:\\\\docs\\\\note.txt' or '/home/user/note.html')."
            )
        ),
    ],
) -> Any:
    """Read a local file and convert it to a docs/ Source via the Import deep module.

    Delegates to ``markdown_kb.app.importer.import_path``.  On
    ``ImportPathError``, returns a ``CallToolResult`` with ``isError=True``
    so the MCP host receives a structured error payload instead of a raw exception.

    Reading an arbitrary local path is safe under the single-operator posture
    (ADR-0017).  Defaults are enforced server-side (ADR-0016 strict schema).
    """
    from pathlib import Path as _Path

    from markdown_kb.app.importer import ImportPathError, import_path

    try:
        result = import_path(_Path(path))
    except ImportPathError as exc:
        payload = json.dumps({"code": "IMPORT_REJECTED", "message": exc.message})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    # Neutral dict shape — no raw importer types cross the MCP boundary.
    source_basename = _Path(result.docs_path).name
    return {"ok": True, "source": source_basename, "status": result.status}


# ---------------------------------------------------------------------------
# Patch schema to add additionalProperties:false (ADR-0016 strict schema)
# ---------------------------------------------------------------------------
# FastMCP generates the tool parameters schema from the function signature but
# does not set additionalProperties:false by default.  We patch it here so MCP
# hosts (e.g. Claude Desktop) receive the strict schema and know not to send
# extra fields.  This does not affect FastMCP's internal argument validation.
def _add_strict_schema() -> None:
    """Patch tool parameter schemas to include additionalProperties:false."""
    for tool_name in (
        "kb_ask_v1",
        "kb_search_v1",
        "kb_read_hot_v1",
        "kb_save_hot_v1",
        "kb_index_v1",
        "kb_lint_v1",
        "kb_ingest_v1",
        "kb_ingest_start_v1",
        "kb_ingest_status_v1",
        "kb_capture_v1",
        "kb_import_v1",
    ):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters["additionalProperties"] = False


_add_strict_schema()
