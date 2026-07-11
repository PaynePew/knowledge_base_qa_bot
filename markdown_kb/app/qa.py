"""Deep module per Ousterhout. Public surface: ``compute_slug``, ``normalize_question``, ``maybe_file_answer``, ``dispatch_filing``, ``promote``, ``promote_batch``, ``delete``, ``edit``, ``refile``, ``demote``, ``QaPageNotFound``, ``QaPageCorrupt``, ``QaPageLive``, ``QaEditRejected``, ``QaRefileRejected``, ``RefiledAnswer``.

Phase 6 Slice 6-2 — Answer Filing for ``POST /chat``.
Phase 6 Slice 6-4 — curator-driven ``promote(slug)`` flips ``status: draft -> live``.
Phase 9 Slice 4 — ``dispatch_filing()`` shared helper extracts the gating logic
from the ``/chat`` route so the Wiki ``stream_query`` can reuse it without
duplicating the ``outcome.passed`` check or the ``_SectionRef`` adapter.

When ``/chat`` produces a Grounded Answer (``GroundingOutcome.passed == True``),
the route layer dispatches one line to ``maybe_file_answer(...)`` which
synchronously creates or touches a ``wiki/qa/<slug>.md`` page with
``frontmatter.status: draft``. This closes Karpathy's Two-output rule on the
query side: queries now compound into curator-promotable artifacts.

Design constraints (PRD #78):

- **Q3 S5 slug strategy** — ``slugify(question)[:40] + "-" + sha1(normalized)[:6]``.
  Whitespace / punctuation / case differences collapse to the same slug
  (idempotency); 24 bits of entropy keep collisions sub-1% in demo scale.
- **Q7 sync, fail-soft, threading.Lock** — filing runs in-handler (latency
  overhead ~6ms on a ~3s LLM-dominated baseline); IOErrors fail-soft by
  emitting ``qa_filing_error`` + returning ``None`` so ``/chat`` still ships
  the answer; a module-level ``_filing_lock`` covers the whole "read existing
  state → decide create/touch → write → emit reflect" critical section so two
  concurrent uvicorn threads cannot race to create duplicate files.
  Multi-worker production deployment triggers a future ``filelock`` upgrade
  isolated to this module.
- **Q8a sentinel** — qa-specific HTML comment that explains hand-edits are
  safe (B2 touch semantics never overwrite the body). Distinct from the
  entity/concept sentinel which says "manual edits will be overwritten".
- **Q8d orphan-visibility defence layer 2** — touch attempt against a page
  whose existing ``status`` is anything other than ``{"draft", "live"}``
  refuses to mutate; emits ``qa_filing_error reason=orphan_status``; returns
  ``None``. Mirrors the indexer-layer defence in ``_passes_index_filter``.

Atomic write uses tmp file + ``os.replace`` per CODING_STANDARD §2.6 — same
convention as ``wiki_writer.py``.

tier-B S3 (issue #379, ADR-0026 decision 2) — ``edit(slug, question, body)``
completes the Curation Queue gate's verb set (approve / edit-then-approve /
discard). Draft-only: refuses a ``status: live`` page (live hand-edits keep
the documented file-level path). Re-runs the Grounding Check against the
page's cited Sections (``frontmatter.sources``, resolved back to their wiki
content by ``_resolve_cited_sections``) on the submitted body — a failing
check writes nothing (``QaEditRejected``); only a pass rewrites the page.

tier-B S4 (issue #380, ADR-0026 decision 1) — ``refile(slug)`` is the C9
stale-Filed-Answer remediation: a single chained operation in a fixed
internal order — (1) re-synthesize the page's recorded ``question`` through
the chat pipeline (``retrieval.query``) with ``wiki/qa/`` excluded from
retrieval, so the re-derivation can never retrieve (and re-cite) the stale
page itself; (2) grounding-check the fresh answer BEFORE any write — a
failing check raises ``QaRefileRejected`` and writes nothing (old live page
keeps serving, the C9 finding stays); (3) only on pass, overwrite the SAME
slug in place with the fresh answer + its fresh cited Sections,
``status: draft``. The corpus gap between re-file and promote is deliberate
(ADR-0026) — the curator reviews the re-filed draft in the existing
Promote/Edit/Discard Curation Queue loop.

tier-B S6 (issue #382, ADR-0023 Consequences) — ``promote_batch(slugs)`` is
the one pre-authorized Direct-tier batch endpoint ("a batch-promote
endpoint, deferred" — this closes that gap). ``slugs`` is the explicit list
the operator saw rendered in the Curation Queue, never "all drafts" resolved
server-side, so a draft filed after the operator looked is never approved
sight-unseen. Per-slug validation is independent (a bad slug never aborts
the batch — non-transactional, ADR-0023): missing file, unparseable
frontmatter, an invalid ``status``, or ``status == "live"`` are each skipped
with a reason; only a ``status == "draft"`` slug is flipped to ``live``.
Reindexing is the route layer's job, once, after the whole batch — mirrors
``promote``'s own route/domain split.

issue #535 (ADR-0037) — ``demote(slug)`` is the C10 remediation for a
schema-invalid ``status: live`` Filed Answer: ``qa.delete`` refuses any live
page (ADR-0012), which previously dead-ended a live-but-defective page
(neither fixable nor discardable). ``demote`` flips ``status: live -> draft``
in place, preserving question/body/count/created, so the page leaves the
BM25 corpus and re-enters the normal Promote/Edit/Discard Curation Queue
loop. Generalises the demote-to-draft primitive ADR-0035 built inline inside
``refile``'s retire step into a standalone, first-class Lifecycle action —
a distinct primitive from that inline step, which keeps its own
reason-parametrized ``qa_reflect op=retired`` logging for the C9 audit trail.

issue #573 — ``dispatch_filing``'s novelty gate (ADR-0020) skipped filing
only when *every* cited source was a ``wiki/qa/`` page. That missed the
case where the answer's only meaningfully-used citation was a live QA page
but the BM25 top-k (``result["sources"]``) also carried one or two
lower-ranked, unused non-qa candidates — the stricter "all" check saw a
mix and filed a near-duplicate draft anyway. ``_top_citation_is_qa``
broadens the gate to "the top-ranked cited source is a ``wiki/qa/`` page"
(``sources`` is already rank-ordered best-first), which subsumes the
original all-qa case and additionally catches this one; it emits the new
``qa_filing_skip`` kind so the skip is observable. Deliberately
conservative: a QA page cited only *below* position 0 does not gate.
"""

from __future__ import annotations

import datetime
import hashlib
import re
import threading
from pathlib import Path
from typing import Any

import yaml

from .atomic import write_text_atomic
from .grounding import CONTENT_FAILURE_REASONS, CitableContent, GroundingOutcome
from .indexer import parse_markdown, slugify
from .logger import log_event
from .schemas import (
    FiledStatus,
    GroundingClaim,
    GroundingInfo,
    QaPromoteBatchResponse,
    SkippedSlug,
    WikiPageFrontmatter,
)
from .slugs import is_bare_slug

# ---------------------------------------------------------------------------
# Sentinel HTML comment (PRD #78 Q8a — verbatim)
# ---------------------------------------------------------------------------

# Qa-specific sentinel. Distinct from wiki_writer's entity/concept sentinel
# because B2 touch semantics explicitly preserve the body across re-asks —
# hand-edits are safe and authoritative once ``status: live``. Writing this
# comment on every created page makes the lifecycle contract visible to any
# human who opens the file in an editor.
SENTINEL_COMMENT = (
    "<!-- Auto-filed by POST /chat. Body persists across re-asks "
    "(re-asks only bump count + updated). Hand-edits are safe and "
    "authoritative once status=live. Delete the file to re-file fresh "
    "on next /chat. -->"
)

# ---------------------------------------------------------------------------
# Concurrency lock (PRD #78 Q7 L1)
# ---------------------------------------------------------------------------

# Single module-level lock covers the ENTIRE filing decision:
#   1. read existing frontmatter (if any)
#   2. decide create vs touch
#   3. atomic write
#   4. emit qa_reflect
# Holding the lock through reflect emission guarantees that mutation and log
# entry are inseparable — there is no window in which a write happened but the
# log entry has not yet appeared.
#
# Demo / single-worker uvicorn scope only. Multi-worker production requires
# upgrading to a cross-process file lock (e.g. ``filelock`` library) — the
# change is fully contained in this module per PRD #78 §"Filing trigger
# placement" (Q7).
_filing_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pure helpers (Q3)
# ---------------------------------------------------------------------------


_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    The output is the input to ``sha1`` in ``compute_slug`` — two questions
    differing only by trailing punctuation or extra whitespace must produce
    the same normalised string so they hash to the same suffix and resolve
    to the same Filed Answer.

    Examples:
        normalize_question("How do I cancel?") == "how do i cancel"
        normalize_question("  HOW do I CANCEL?? ") == "how do i cancel"
        normalize_question("How  do\tI\ncancel") == "how do i cancel"
    """
    lowered = question.lower()
    no_punct = _PUNCTUATION_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", no_punct).strip()


def compute_slug(question: str) -> str:
    """Compute the ``wiki/qa/<slug>.md`` slug for a question (PRD #78 Q3 S5).

    Strategy: ``slugify(question)[:40] + "-" + sha1(normalized_question)[:6]``

    The 6-hex-char SHA-1 suffix gives 24 bits of entropy — enough that the
    expected number of collisions across a demo-scale pool (~10k questions)
    is sub-1%. The slugified prefix keeps slugs grep-able by the curator.

    Phase 16: ``slugify`` now preserves CJK (and other Unicode) characters
    verbatim, so a CJK question like "如何取消订单？" produces a readable
    prefix "如何取消订单" rather than the degenerate "section". The ``"section"``
    → ``"qa"`` special-case is retained as a safety net for fully-punctuation
    inputs that produce no slug-able characters at all.
    """
    prefix = slugify(question)[:40]
    if prefix == "section":
        # ``slugify`` returns ``"section"`` for inputs with no ASCII
        # alphanumerics. PRD #78 Q3 envisions ``qa-<hash>`` as the CJK-safe
        # degenerate form; this matches that.
        prefix = "qa"
    digest = hashlib.sha1(normalize_question(question).encode("utf-8")).hexdigest()[:6]
    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------
# Filing entry point
# ---------------------------------------------------------------------------


def _qa_dir() -> Path:
    """Resolve ``wiki/qa/`` lazily so tests' monkeypatched WIKI_DIR is honoured.

    The indexer module is imported at call time (not at module load) because
    test fixtures monkeypatch ``indexer.WIKI_DIR`` to ``tmp_path / "wiki"``;
    binding the path at import would lock in the production location before
    tests get a chance to swap it.
    """
    from . import indexer

    return indexer.WIKI_DIR / "qa"


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp matching the existing wiki_writer / templates format."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(text: str, limit: int = 60) -> str:
    """Truncate text for inclusion in log summaries.

    Mirrors the pattern used in ``retrieval.py`` (60-char cap, double-quotes
    flipped to singles so the surrounding ``"..."`` log wrapper stays unambiguous).
    """
    return text[:limit].replace('"', "'")


def _render_frontmatter_yaml(fm: WikiPageFrontmatter) -> str:
    """Serialise a WikiPageFrontmatter to its YAML block (no fence lines).

    Mirrors ``wiki_writer._render_frontmatter`` but includes the qa-only
    ``question`` and ``count`` fields. We keep the rendering local to this
    module rather than reaching into wiki_writer because:

    1. wiki_writer's renderer takes a ``WikiPageDraft`` (which carries a
       Section heading, body, etc.) — qa pages have no Section heading and
       the answer is the body verbatim, so the Draft type would be a
       contortion.
    2. The qa frontmatter shape is stable in this slice; if a later phase
       needs a shared renderer, factoring it out is a refactor with a clear
       trigger (third use site).
    """
    data: dict[str, Any] = {
        "id": fm.id,
        "type": fm.type,
        "created": fm.created,
        "updated": fm.updated,
        "sources": fm.sources,
        "status": fm.status,
        "open_questions": fm.open_questions,
        "question": fm.question,
        "count": fm.count,
    }
    return yaml.dump(data, default_flow_style=False, allow_unicode=True).rstrip()


def _render_qa_page(fm: WikiPageFrontmatter, body: str) -> str:
    """Render the complete ``wiki/qa/<slug>.md`` file content.

    Layout:

        <sentinel HTML comment>

        ---
        <YAML frontmatter>
        ---

        <answer body>

    Body-layout decision (PRD #78 Q2): answer-only, no Q-then-A dialogue.
    Question lives in frontmatter so BM25 retrieval on the qa corpus does
    not self-amplify against the original query terms.
    """
    fm_yaml = _render_frontmatter_yaml(fm)
    parts = [
        SENTINEL_COMMENT,
        "",
        "---",
        fm_yaml,
        "---",
        "",
        body.rstrip(),
        "",
    ]
    return "\n".join(parts)


def _atomic_write(path: Path, content: str) -> None:
    """Delegate to the shared ``write_text_atomic`` helper (CODING_STANDARD §2.6).

    Thin wrapper preserved as a monkeypatch seam: tests that inject failures
    via ``monkeypatch.setattr(app.atomic.os, "replace", ...)`` still exercise
    the fail-soft ``OSError`` handling in ``maybe_file_answer``.
    """
    write_text_atomic(path, content)


def _read_existing_frontmatter(path: Path) -> dict | None:
    """Read YAML frontmatter from a wiki/qa/<slug>.md page if present.

    Skips the optional sentinel HTML comment that prefixes every page (so a
    page with the sentinel parses identically to one without). Returns
    ``None`` when the file is missing OR the frontmatter cannot be parsed —
    the caller treats ``None`` as "no usable existing state" and falls into
    the create path.
    """
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(dash_indices) < 2:
        return None
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_existing_body(path: Path) -> str:
    """Read the body (everything after the second ``---``) from a qa page.

    Returns the body text including trailing newline. Used by the touch path
    so the existing body is preserved verbatim while frontmatter is rewritten
    with the bumped count + updated timestamp.
    """
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(dash_indices) < 2:
        # Defensive: caller has already validated frontmatter; if structure
        # broke between calls, return empty body so the rewrite still produces
        # a valid file shape.
        return ""
    return "".join(lines[dash_indices[1] + 1 :]).lstrip("\n")


def _citation_id(section: CitableContent) -> str:
    """Extract the citation id from a CitableContent (Section)."""
    return section.id


def _compute_cited_delta(new_cited: list[str], existing_cited: list[str]) -> str:
    """Return ``added:<list>,dropped:<list>`` or ``none``.

    PRD #78 Q5 — drift signal core for ``qa_reflect op=touched``. A slug
    repeatedly touched with non-``none`` deltas means the wiki layer is
    shifting beneath the Filed Answer (curator-actionable drift).

    Order-preserving on the new list so the curator can spot the most-recent
    citation first when grepping the log.
    """
    new_set = set(new_cited)
    existing_set = set(existing_cited)
    added = [c for c in new_cited if c not in existing_set]
    dropped = [c for c in existing_cited if c not in new_set]
    if not added and not dropped:
        return "none"
    return f"added:{','.join(added)},dropped:{','.join(dropped)}"


def maybe_file_answer(
    query: str,
    answer: str,
    cited_sections: list[CitableContent],
) -> FiledStatus | None:
    """File or touch ``wiki/qa/<slug>.md`` for a grounding-passing chat answer.

    Phase 6 Slice 6-2 core: synchronous, thread-safe-within-uvicorn, fail-soft.

    Flow inside ``_filing_lock``:

    1. Compute slug from ``query``.
    2. Read existing ``wiki/qa/<slug>.md`` frontmatter (if any).
    3. If existing frontmatter has ``status`` not in ``{"draft", "live"}``:
       emit ``qa_filing_error reason=orphan_status``, return ``None`` —
       the filing-layer member of the three-layer orphan-visibility defence
       (PRD #78 §"Orphan-visibility three-layer defence").
    4. If no existing page: write a fresh one with ``status: draft``,
       ``count: 1``, sentinel comment, full frontmatter. Emit
       ``qa_reflect op=created``.
    5. If existing page with ``status`` in ``{draft, live}``: increment
       ``count``, refresh ``updated``, preserve body verbatim (B2 touch).
       Emit ``qa_reflect op=touched`` with ``cited_delta``.
    6. On any ``OSError`` during write: emit ``qa_filing_error reason=io_error``,
       return ``None`` (F3 fail-soft — caller's ``/chat`` response still ships
       the answer with ``filed: None``).

    Returns:
        FiledStatus describing what happened, or None on any fail-soft path.
        ``op`` is always ``"created"`` or ``"touched"``; ``op="promoted"`` is
        Slice 6-3 territory and surfaces through ``POST /qa/{slug}/promote``,
        not through this entry point.
    """
    slug = compute_slug(query)
    qa_dir = _qa_dir()
    path = qa_dir / f"{slug}.md"
    cited_ids = [_citation_id(s) for s in cited_sections]
    now = _utc_now_iso()

    with _filing_lock:
        existing = _read_existing_frontmatter(path)

        # ---- orphan-status defence (touch path against invalid status) ----
        if existing is not None:
            existing_status = existing.get("status")
            if existing_status not in {"draft", "live"}:
                log_event(
                    "qa_filing_error",
                    f"slug={slug} reason=orphan_status "
                    f"exc=ValueError: invalid_existing_status={existing_status!r}",
                )
                return None

        # ---- create path ----
        if existing is None:
            fm = WikiPageFrontmatter(
                id=slug,
                type="qa",
                created=now,
                updated=now,
                sources=cited_ids,
                status="draft",
                open_questions=[],
                question=query,
                count=1,
            )
            content = _render_qa_page(fm, answer)
            try:
                _atomic_write(path, content)
            except OSError as exc:
                log_event(
                    "qa_filing_error",
                    f"slug={slug} reason=io_error exc={type(exc).__name__}: {exc}",
                )
                return None

            log_event(
                "qa_reflect",
                f'slug={slug} op=created question="{_truncate(query)}" '
                f"cited={','.join(cited_ids)} count=1",
            )
            return FiledStatus(slug=slug, status="draft", op="created", count=1)

        # ---- touch path ----
        existing_status = existing.get("status")
        # Re-read existing data; preserve created + body
        existing_count_raw = existing.get("count", 1)
        try:
            existing_count = int(existing_count_raw)
        except (TypeError, ValueError):
            existing_count = 1
        new_count = existing_count + 1
        existing_sources = [str(s) for s in existing.get("sources", [])]
        existing_question = existing.get("question") or query
        existing_created = existing.get("created", now)
        existing_open_questions = existing.get("open_questions") or []
        # Reuse existing status verbatim — touch must not flip draft→live or vice versa.
        try:
            fm = WikiPageFrontmatter(
                id=existing.get("id", slug),
                type="qa",
                created=existing_created,
                updated=now,
                sources=existing_sources,
                status=existing_status,  # validated above to be draft|live
                open_questions=existing_open_questions,
                question=existing_question,
                count=new_count,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            # Existing file's frontmatter dict didn't validate against our schema
            # (e.g. a curator hand-edit broke a field). Surface as a filing
            # error rather than a 500; the orphan-status defence already
            # covered ``status``, so this is for everything else (type wrong,
            # created malformed, etc.).
            log_event(
                "qa_filing_error",
                f"slug={slug} reason=frontmatter_read_error exc={type(exc).__name__}: {exc}",
            )
            return None

        body = _read_existing_body(path)
        content = _render_qa_page(fm, body)
        try:
            _atomic_write(path, content)
        except OSError as exc:
            log_event(
                "qa_filing_error",
                f"slug={slug} reason=io_error exc={type(exc).__name__}: {exc}",
            )
            return None

        cited_delta = _compute_cited_delta(cited_ids, existing_sources)
        log_event(
            "qa_reflect",
            f"slug={slug} op=touched cited_delta={cited_delta} count={new_count}",
        )
        return FiledStatus(
            slug=slug,
            status=existing_status,
            op="touched",
            count=new_count,
        )


# ---------------------------------------------------------------------------
# Shared filing dispatcher (Phase 9 Slice 4)
# ---------------------------------------------------------------------------


class _SectionRef:
    """Minimal CitableContent adapter for ``maybe_file_answer``.

    The qa module's ``maybe_file_answer`` accepts any ``CitableContent``
    (Protocol — requires ``id``, ``heading_path``, ``content``). Route and
    retrieval layers hold sources as plain dicts; reconstructing a full
    ``indexer.Section`` would couple callers to the indexer dataclass.  This
    shim is the smallest viable adapter — only ``id`` is set because the
    filing path reads no other field.

    Defined here (not in routes.py) so that both the ``/chat`` route AND
    ``stream_query`` can call ``dispatch_filing`` without each defining their
    own adapter shim.
    """

    __slots__ = ("content", "heading_path", "id")

    def __init__(self, id: str) -> None:
        self.id = id
        self.heading_path = [id]
        self.content = ""


def _top_citation_is_qa(sources: list[dict]) -> bool:
    """Return True when the highest-ranked cited source is a ``wiki/qa/`` page.

    Issue #573 novelty gate — broadens the original ADR-0020 "every cited
    source is qa" check to "the *top* cited source is qa": when the answer's
    best-ranked citation is itself a live QA page, there is nothing new to
    capture even if the retrieval top-k also carried lower-ranked, unused
    non-qa candidates. Filing anyway would produce a near-duplicate draft
    that the Curation Queue surfaces immediately as a promotion candidate,
    making Promote appear to be a no-op — the ADR-0020 problem statement,
    just missed by the stricter "all" predicate when the top-k mixes qa and
    non-qa sources.

    ``sources`` (``result["sources"]``) is already rank-ordered best-first —
    ``indexer.search`` sorts descending by BM25 score before ``retrieval.py``
    builds this list — so ``sources[0]`` is the top citation. Deliberately
    conservative per the issue's scope decision: a QA page cited only *below*
    position 0 does not gate; that combination still files (a Source section
    on top means genuinely new coverage even if a QA page is cited lower).

    Detection: a source is a qa page when its ``path`` key starts with
    ``"wiki/qa/"`` (set by ``retrieval._wiki_page_path_for_section``; issue
    #266). No separate ``status == "live"`` lookup is needed — ``indexer.
    _passes_index_filter`` admits only ``status: live`` pages under
    ``wiki/qa/`` into the BM25 corpus, so any ``wiki/qa/`` citation reaching
    this function is already guaranteed live (a draft or retired page can
    never be retrieved, let alone cited).

    Empty sources list: returns ``False`` — nothing to gate on (should not
    happen after grounding passes, but be conservative).
    """
    if not sources:
        return False
    return (sources[0].get("path") or "").startswith("wiki/qa/")


def dispatch_filing(query: str, result: dict) -> FiledStatus | None:
    """Gate-and-dispatch filing for a single query result.

    Shared helper used by both ``POST /chat`` (routes.py) and the Wiki
    ``stream_query`` path (retrieval.py) to avoid duplicated dispatch logic
    (Phase 9 Slice 4 / issue #121 AC1).

    Gating rules:
    - ``result["grounding_outcome"].passed`` must be True; early-exit Cannot
      Confirm paths (``passed == False``) return ``None`` without calling
      ``maybe_file_answer``.
    - ``result["answer"]`` must not be the Cannot-Confirm sentinel. An LLM can
      emit that sentence itself; it then trivially passes grounding
      (``passed == True``, no unsupported claims) yet is a non-answer that must
      never be filed as a curatable draft.
    - The top-ranked cited source is a ``wiki/qa/`` page (ADR-0020 / issue
      #573 novelty gate): the answer's best evidence is already-curated Q&A;
      nothing new to capture. Emits ``qa_filing_skip`` when this fires. A
      Source section on top with a QA page cited only *lower* still files
      (the gate looks at position 0 only — see ``_top_citation_is_qa``).
    - RAG paths never call this function (caller contract; enforcement is
      the caller's responsibility — this function does not inspect the stack).

    Args:
        query:  The original user question (used for slug computation).
        result: A result dict as returned by ``query()`` / the final yield
                of ``stream_query()``.  Must contain ``sources`` (list of
                dicts each with a ``source`` key carrying the section id)
                and ``grounding_outcome`` (a ``GroundingOutcome`` instance).

    Returns:
        ``FiledStatus`` describing what happened, or ``None`` on any
        fail-soft path (Cannot Confirm, IOError, orphan-status guard,
        novelty gate).
    """
    outcome = result["grounding_outcome"]
    if not outcome.passed:
        return None
    # An LLM can emit the Cannot-Confirm sentence as its own answer. That text
    # carries no factual claims, so the grounding check trivially passes
    # (outcome.passed == True) — but it is a non-answer, not a curatable Q&A.
    # Filing it plants an "I cannot confirm…" draft in wiki/qa/ that the
    # Curation Queue then surfaces as a promotion candidate. Gate it out here so
    # the LLM-emitted CC case is skipped too (early-exit CC already returned via
    # passed == False above). Lazy import keeps qa <-> retrieval one-directional.
    from .retrieval import CANNOT_CONFIRM_PHRASE

    if result.get("answer", "").strip() == CANNOT_CONFIRM_PHRASE:
        return None

    # ADR-0020 / issue #573 novelty gate: skip filing when the top-ranked
    # cited source is already a live wiki/qa/ page. A re-ask whose rewritten/
    # rephrased query misses the promoted qa page and is answered from raw
    # Sources still files — accepted as a legitimate coverage signal.
    sources = result.get("sources", [])
    if _top_citation_is_qa(sources):
        top_source = sources[0].get("source", "")
        log_event(
            "qa_filing_skip",
            f'"{_truncate(query)}" reason=top_citation_is_qa top={top_source}',
        )
        return None

    cited_refs = [_SectionRef(id=s["source"]) for s in result["sources"]]
    return maybe_file_answer(query, result["answer"], cited_refs)


# ---------------------------------------------------------------------------
# Promote (Slice 6-4)
# ---------------------------------------------------------------------------


class QaPageNotFound(Exception):
    """Raised by ``promote`` when no ``wiki/qa/<slug>.md`` exists for the slug.

    Route layer maps this to ``HTTP 404`` per issue #83 AC. Not a subclass of
    ``OSError`` / ``FileNotFoundError`` because the contract is semantic ("the
    curator referenced a slug that has never been filed"), not a low-level
    I/O signal — making it a distinct sentinel lets the route handler dispatch
    cleanly without overloading exception meaning.
    """


class QaPageCorrupt(Exception):
    """Raised by ``promote`` when the page exists but its frontmatter is invalid.

    Triggers:
      - Existing ``frontmatter.status`` is not in ``{"draft", "live"}`` — covers
        curator-typo ``Live`` (capital L), forward-compat reserved values like
        ``stale`` / ``superseded``, missing ``status`` key, etc.
      - Frontmatter cannot be parsed at all (corrupt YAML, missing fences).

    Route layer maps this to ``HTTP 500`` per issue #83 AC. The orphan-visibility
    three-layer defence (PRD #78 §"Orphan-visibility three-layer defence") says
    broken state must remain visible — promote must NOT silently rewrite the
    page to a recognised status. Raising surfaces the curator typo loudly via
    the HTTP response.
    """


def _flip_draft_to_live(slug: str, path: Path, existing: dict) -> int:
    """Rewrite an already-validated draft page to ``status: live`` in place.

    Caller must hold ``_filing_lock`` and must have already validated
    ``existing["status"] == "draft"`` — this helper does no validation of
    its own. Shared by ``promote`` (one slug) and ``promote_batch`` (N
    slugs, one call each per valid slug) so the draft->live rewrite stays
    in exactly one place. Rebuilds the full frontmatter (avoids surgical
    mid-line edits that would break round-tripping); body is preserved
    verbatim. Emits the same ``qa_reflect op=promoted by=curator`` log line
    either caller would emit on its own.

    Returns the preserved ``count`` so each caller can build its own return
    shape (``FiledStatus`` for ``promote``; an accumulator list for
    ``promote_batch``).
    """
    try:
        existing_count = int(existing.get("count", 1))
    except (TypeError, ValueError):
        existing_count = 1
    now = _utc_now_iso()
    fm = WikiPageFrontmatter(
        id=existing.get("id", slug),
        type="qa",
        created=existing.get("created", now),
        updated=now,
        sources=[str(s) for s in existing.get("sources", [])],
        status="live",
        open_questions=existing.get("open_questions") or [],
        question=existing.get("question"),
        count=existing_count,
    )
    body = _read_existing_body(path)
    content = _render_qa_page(fm, body)
    _atomic_write(path, content)

    log_event("qa_reflect", f"slug={slug} op=promoted by=curator")
    return existing_count


def promote(slug: str) -> FiledStatus:
    """Curator-driven promotion: flip ``wiki/qa/<slug>.md`` ``status: draft -> live``.

    Acceptance criteria (issue #83):

    - Acquires ``_filing_lock`` so promote and filing cannot interleave on
      the same slug (same lock as ``maybe_file_answer``).
    - Reads existing frontmatter; raises ``QaPageNotFound`` if absent.
    - Raises ``QaPageCorrupt`` if the existing ``status`` is anything other
      than ``{"draft", "live"}`` — the orphan-visibility defence keeps the
      broken state visible (PRD #78 §"Orphan-visibility three-layer defence").
    - **Idempotent**: if ``status`` is already ``live``, returns the existing
      FiledStatus without rewriting the file and without emitting a second
      reflect log entry.
    - Otherwise rewrites the page atomically (tmp + ``os.replace``) with
      ``status: "live"``, preserving body / question / count / created /
      sources / open_questions verbatim. Refreshes ``updated`` to "now".
    - Emits ``qa_reflect op=promoted by=curator`` log entry per PRD #78 Q5c.

    Returns:
        ``FiledStatus`` with ``status="live"``, ``op="touched"`` (promotion
        is structurally a touch from the FiledStatus enum perspective —
        see issue #83 AC), and the preserved ``count``.

    Raises:
        QaPageNotFound: no ``wiki/qa/<slug>.md`` on disk, OR ``slug`` is not
            a bare filename component (issue #397 — a FastAPI path segment
            cannot contain ``/`` but CAN contain ``\\`` / ``:``, which act as
            separators once joined on Windows; ``slugs.is_bare_slug``
            rejects these before any filesystem access, same 404 a garbage
            slug produces on Linux).
        QaPageCorrupt: existing frontmatter has invalid / unparseable
            ``status`` (orphan zombie). The file is left untouched so the
            curator can inspect.
    """
    if not is_bare_slug(slug):
        raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

    qa_dir = _qa_dir()
    path = qa_dir / f"{slug}.md"

    with _filing_lock:
        if not path.exists():
            raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

        existing = _read_existing_frontmatter(path)
        if existing is None:
            # File exists but frontmatter could not be parsed at all.
            # Orphan-visibility defence: surface as corrupt rather than
            # silently rewriting to a recognised shape.
            raise QaPageCorrupt(f"wiki/qa/{slug}.md exists but its frontmatter could not be parsed")

        existing_status = existing.get("status")
        if existing_status not in {"draft", "live"}:
            raise QaPageCorrupt(
                f"wiki/qa/{slug}.md has invalid status={existing_status!r}; "
                "expected 'draft' or 'live'. The page is left untouched so the "
                "curator can inspect and repair."
            )

        # Preserve count for the FiledStatus return value (and for the live-
        # already idempotent path).
        try:
            existing_count = int(existing.get("count", 1))
        except (TypeError, ValueError):
            existing_count = 1

        # Idempotent: re-promote of a live page is a no-op. No rewrite, no log.
        if existing_status == "live":
            return FiledStatus(
                slug=slug,
                status="live",
                op="touched",
                count=existing_count,
            )

        # Draft -> live transition, shared with promote_batch's per-slug write.
        existing_count = _flip_draft_to_live(slug, path, existing)
        return FiledStatus(slug=slug, status="live", op="touched", count=existing_count)


# ---------------------------------------------------------------------------
# Promote batch (tier-B S6, issue #382, ADR-0023 Consequences)
# ---------------------------------------------------------------------------


def promote_batch(slugs: list[str]) -> QaPromoteBatchResponse:
    """Curator-driven batch promotion: flip every valid slug in ``slugs`` draft -> live.

    tier-B S6 (issue #382) — the one pre-authorized Direct-tier batch
    endpoint ADR-0023 Consequences deferred ("a batch-promote endpoint,
    deferred" — this closes that gap). ``slugs`` must be the explicit list
    the operator actually saw rendered in the Curation Queue at click time —
    never resolved as "all drafts" server-side — so a draft filed after the
    operator looked is never approved sight-unseen.

    Per-slug validation, each independent of the others (a bad slug never
    aborts the batch — non-transactional, ADR-0023):

    - slug not a bare filename component      -> skipped, reason="invalid_slug"
      (separators / parent refs / NUL — see ``slugs.is_bare_slug``; the
      batch's slugs arrive in the JSON body, so they never got the no-"/"
      guarantee a FastAPI path segment gives the single-item endpoints —
      those endpoints run the same guard directly, issue #397)
    - missing file                            -> skipped, reason="not_found"
    - frontmatter unparseable                 -> skipped, reason="corrupt_frontmatter"
    - ``status`` not in ``{"draft", "live"}``  -> skipped, reason="invalid_status:<value>"
    - ``status == "live"``                     -> skipped, reason="already_live"
      (the operator's queue only ever renders drafts, so a submitted slug
      that is already live means someone else promoted it since the queue
      was rendered; silently re-promoting it would hide that from the
      curator instead of surfacing it)
    - ``status == "draft"``                    -> promoted

    Each valid slug is flipped under ``_filing_lock`` (the same lock
    ``promote`` uses) one slug at a time — the batch does not hold a single
    lock for its whole duration, so a large batch never blocks concurrent
    filing/promote on unrelated slugs for longer than one page's write.

    Reindexing is deliberately NOT this function's job (mirrors ``promote``'s
    own route/domain split): the caller (route layer) calls
    ``build_index()`` exactly once after this returns, regardless of how
    many slugs were promoted (issue #382 AC: "exactly one reindex regardless
    of N").

    Each successful promotion emits the same ``qa_reflect op=promoted
    by=curator`` log line a single ``promote()`` call would — a batch is
    just N of the same curator action, so a log reader grepping
    ``op=promoted`` sees every promoted page the same way regardless of the
    caller. Skipped slugs emit nothing, mirroring ``promote``'s own
    not-found/corrupt refusal paths.

    Returns:
        ``QaPromoteBatchResponse`` with ``promoted`` (slugs flipped to live,
        submission order) and ``skipped`` (slug + reason, submission order).
    """
    promoted: list[str] = []
    skipped: list[SkippedSlug] = []
    qa_dir = _qa_dir()

    for slug in slugs:
        if not is_bare_slug(slug):
            skipped.append(SkippedSlug(slug=slug, reason="invalid_slug"))
            continue

        path = qa_dir / f"{slug}.md"
        with _filing_lock:
            if not path.exists():
                skipped.append(SkippedSlug(slug=slug, reason="not_found"))
                continue

            existing = _read_existing_frontmatter(path)
            if existing is None:
                skipped.append(SkippedSlug(slug=slug, reason="corrupt_frontmatter"))
                continue

            existing_status = existing.get("status")
            if existing_status not in {"draft", "live"}:
                skipped.append(SkippedSlug(slug=slug, reason=f"invalid_status:{existing_status}"))
                continue
            if existing_status == "live":
                skipped.append(SkippedSlug(slug=slug, reason="already_live"))
                continue

            # status == "draft" — the only promotable case. Shared with
            # promote's single-slug draft->live write.
            _flip_draft_to_live(slug, path, existing)
            promoted.append(slug)

    return QaPromoteBatchResponse(promoted=promoted, skipped=skipped)


# ---------------------------------------------------------------------------
# Edit (tier-B S3, issue #379, ADR-0026 decision 2)
# ---------------------------------------------------------------------------


class QaEditRejected(Exception):
    """Raised by ``edit`` when the submitted body fails the LLM-free
    grounding re-check against the page's cited Sections (ADR-0026
    decision 2: "the re-check is LLM-free and instant"). Carries the
    caller-facing failure list so the route can render every problem
    honestly. Nothing is written when this is raised."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = failures
        super().__init__("qa edit content failed grounding re-check")


def _resolve_cited_sections(source_ids: list[str], wiki_dir: Path) -> list[CitableContent]:
    """Resolve a qa page's ``frontmatter.sources`` ids back into Sections.

    Each id is either ``<wiki-page-slug>#<heading-slug>`` or a BARE
    ``<wiki-page-slug>`` — the slug-based addressing ``indexer.parse_markdown``
    assigns to wiki-derived pages: rule 7 gives a page with zero headings a
    single Section whose id is the bare slug, which is exactly what a
    promoted qa page looks like (issue #495 — skipping bare ids made every
    edit of a draft citing one permanently rejected). The page's type subdir
    is not encoded in the id, so ``entities``/``concepts``/``qa`` are each
    tried in turn; a filed qa answer can itself cite an already-promoted qa
    page. Re-parses the cited pages directly from disk rather than depending
    on the in-memory BM25 index being loaded, so the check works the same
    whether or not ``build_index()`` has run yet.

    A citation whose page no longer resolves, or whose heading no longer
    exists on that page, is skipped — best-effort, mirrors
    ``reconcile._collect_union_sections``'s tolerance of a missing Source (a
    dangling citation is then reported by ``_check_edit_grounding`` as an
    unresolvable-citation failure).
    """
    sections: list[CitableContent] = []
    seen_ids: set[str] = set()
    parsed_by_page: dict[str, list] = {}

    for source_id in source_ids:
        # A bare id (no '#') is a whole heading-less page (rule 7): the page
        # slug IS the id, and its single Section's id matches it verbatim.
        page_slug = source_id.partition("#")[0]
        if not page_slug:
            continue
        if page_slug not in parsed_by_page:
            page_path = None
            for subdir_name in ("entities", "concepts", "qa"):
                candidate = wiki_dir / subdir_name / f"{page_slug}.md"
                if candidate.exists():
                    page_path = candidate
                    break
            if page_path is None:
                parsed_by_page[page_slug] = []
            else:
                try:
                    parsed_by_page[page_slug] = parse_markdown(page_path, source_id=page_slug)
                except Exception:  # noqa: BLE001 — a malformed page degrades context, not a hard error
                    parsed_by_page[page_slug] = []

        for sec in parsed_by_page[page_slug]:
            if sec.id == source_id and sec.id not in seen_ids:
                seen_ids.add(sec.id)
                sections.append(sec)

    return sections


_INLINE_CITATION_RE = re.compile(r"\[Source:\s*([^\]]+?)\s*\]")


def _check_edit_grounding(body: str, sources: list[str], wiki_dir: Path) -> list[str]:
    """LLM-free grounding re-check for a draft edit (ADR-0026 decision 2,
    relaxed 2026-07 — see ADR-0026 § Amendment).

    Deterministic and instant — no LLM call (the ADR's rejected-alternatives
    note: "The re-check is LLM-free and instant; there is no cost argument
    for skipping it"). The human curator is the semantic judge at this gate;
    the server enforces only that the edited text does not FABRICATE grounding:

      1. every inline ``[Source: <id>]`` the body DOES carry is among
         ``frontmatter.sources`` (an edit never widens sources — Re-file is
         the path to fresh Sources), and
      2. every such cited id still resolves to a Section on disk.

    A body with NO inline citation is accepted as long as the page still has
    a grounding record in ``frontmatter.sources``. That list — not the inline
    ``[Source: ...]`` markers — IS the Filed Answer's citation record: filing
    (``maybe_file_answer``) writes the raw LLM answer verbatim, and the model
    frequently omits inline markers even though ``SYSTEM_PROMPT`` rule 2 asks
    for them (grounding still passes via the claim-level verifier, which never
    reads the body's markers). The old "body must carry ≥1 citation" rule
    punished the curator for the model's omission and made every such draft
    impossible to edit-then-approve — a body that was legal to *file* was
    illegal to *edit* without changing a single claim. Deleting a marker is
    therefore harmless: the grounding record in ``sources`` is preserved (an
    edit never touches it), so the answer stays grounded; only fabricating a
    citation to a Source the page does not cite is rejected (rule 1).

    Returns the failure list; empty means the edit passes.
    """
    cited = [c.strip() for c in _INLINE_CITATION_RE.findall(body)]
    if not cited:
        # No inline citation: allowed as long as the page has a grounding
        # record. A Filed Answer always does (filing sets sources from the
        # grounded chat answer's cited_ids); this guard only bites the
        # degenerate case of an edit to a page with no cited Sources at all.
        if not sources:
            return [
                "this Filed Answer has no cited Sources (its frontmatter.sources "
                "is empty) — add a [Source: ...] citation drawn from the page's "
                "Sources so the answer stays grounded"
            ]
        return []

    failures: list[str] = []
    allowed = set(sources)
    resolved_ids = {s.id for s in _resolve_cited_sections(sorted(set(cited)), wiki_dir)}
    for citation in dict.fromkeys(cited):
        if citation not in allowed:
            failures.append(
                f"citation '{citation}' is not among this page's sources — an edit "
                "cannot widen sources (use Re-file to re-derive from fresh Sources)"
            )
        elif citation not in resolved_ids:
            failures.append(f"citation '{citation}' no longer resolves to a wiki Section")
    return failures


def edit(slug: str, question: str, body: str) -> FiledStatus:
    """Curator-driven edit: rewrite a draft ``wiki/qa/<slug>.md``'s question/body in place.

    ADR-0026 decision 2 — completes the gate's verb set: approve (``promote``)
    / edit-then-approve (``edit`` then ``promote``) / discard (``delete``).

    - Acquires ``_filing_lock`` (same as ``maybe_file_answer``/``promote``/
      ``delete``) so an edit cannot interleave with a concurrent filing touch
      or promote on the same slug.
    - Not found → ``QaPageNotFound``. Corrupt/invalid-status frontmatter →
      ``QaPageCorrupt`` (orphan-visibility — surface broken state rather than
      silently rewriting it, mirrors ``promote``).
    - ``status == "live"`` → ``QaPageLive``. Draft-only: live hand-edits keep
      the documented file-level path (ADR-0026).
    - Re-runs the LLM-free grounding check (``_check_edit_grounding``)
      against the page's *existing* cited Sections (``frontmatter.sources``
      is not widened by the edit) on the submitted ``body``. A failing check
      raises ``QaEditRejected`` with the failure list and writes nothing.
    - On pass, rewrites the page: ``question``/``body`` become the submitted
      values, ``updated`` bumps to now, ``status`` stays ``"draft"`` (an edit
      never promotes) — ``id``/``created``/``sources``/``count``/
      ``open_questions`` are preserved verbatim.

    Returns:
        ``FiledStatus`` with ``status="draft"``, ``op="touched"`` (edit is
        structurally a touch from the FiledStatus enum perspective, same
        reuse ``promote`` makes of the enum), and the preserved ``count``.

    Raises:
        QaPageNotFound: no ``wiki/qa/<slug>.md`` on disk, OR ``slug`` is not
            a bare filename component (issue #397 — see ``promote``'s
            ``Raises`` entry for why; ``slugs.is_bare_slug`` rejects it
            before any filesystem access).
        QaPageCorrupt: existing frontmatter has invalid / unparseable
            ``status`` (orphan zombie).
        QaPageLive: existing ``status`` is ``"live"`` — edit refused.
        QaEditRejected: the submitted body failed the grounding re-check
            against its cited Sections; nothing is written.
    """
    if not is_bare_slug(slug):
        raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

    qa_dir = _qa_dir()
    path = qa_dir / f"{slug}.md"

    with _filing_lock:
        if not path.exists():
            raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

        existing = _read_existing_frontmatter(path)
        if existing is None:
            raise QaPageCorrupt(f"wiki/qa/{slug}.md exists but its frontmatter could not be parsed")

        existing_status = existing.get("status")
        if existing_status not in {"draft", "live"}:
            raise QaPageCorrupt(
                f"wiki/qa/{slug}.md has invalid status={existing_status!r}; "
                "expected 'draft' or 'live'. The page is left untouched so the "
                "curator can inspect and repair."
            )
        if existing_status == "live":
            raise QaPageLive(
                f"wiki/qa/{slug}.md has status=live; edit refused (draft-only, ADR-0026). "
                "Live hand-edits keep the documented file-level path."
            )

        existing_sources = [str(s) for s in existing.get("sources", [])]
        failures = _check_edit_grounding(body, existing_sources, qa_dir.parent)
        if failures:
            log_event("qa_edit_rejected", f"slug={slug} failures={len(failures)}")
            raise QaEditRejected(failures)

        try:
            existing_count = int(existing.get("count", 1))
        except (TypeError, ValueError):
            existing_count = 1
        existing_created = existing.get("created", _utc_now_iso())
        existing_open_questions = existing.get("open_questions") or []

        fm = WikiPageFrontmatter(
            id=existing.get("id", slug),
            type="qa",
            created=existing_created,
            updated=_utc_now_iso(),
            sources=existing_sources,
            status="draft",
            open_questions=existing_open_questions,
            question=question,
            count=existing_count,
        )
        content = _render_qa_page(fm, body)
        _atomic_write(path, content)

        log_event("qa_reflect", f"slug={slug} op=edited count={existing_count}")
        return FiledStatus(slug=slug, status="draft", op="touched", count=existing_count)


# ---------------------------------------------------------------------------
# Delete (Phase 15 Slice 6 / ADR-0012)
# ---------------------------------------------------------------------------


class QaPageLive(Exception):
    """Raised by ``delete`` and ``edit`` when the page's ``status`` is ``live``.

    Live pages are the only pages that enter the BM25 corpus and are the
    "precious" state that must not be removed via a one-click console action
    (``delete``, ADR-0012) or bypass the file-level hand-edit path
    (``edit``, ADR-0026 — draft-only). The Console UI gate and the route
    layer both map this to ``HTTP 409`` (Conflict).

    Not a subclass of ``ValueError`` or ``OSError`` — a distinct sentinel
    keeps the route handler dispatch clean without overloading exception
    meaning (same reasoning as ``QaPageNotFound`` / ``QaPageCorrupt``).
    """


class DeletedQaPage:
    """Lightweight result returned by ``delete``.

    Carries the slug and the status the page had before deletion so the
    caller (route + tests) can assert on what was removed without reaching
    back into the filesystem.

    Not a Pydantic model: this is purely a route-internal result; it never
    crosses an HTTP boundary as a response body (the route returns HTTP 204
    No Content). Using a plain dataclass keeps the qa module free of Pydantic
    concerns for a value that is never serialised.
    """

    __slots__ = ("slug", "prev_status")

    def __init__(self, slug: str, prev_status: str) -> None:
        self.slug = slug
        self.prev_status = prev_status


def delete(slug: str) -> DeletedQaPage:
    """Curator-driven delete: remove an inert ``wiki/qa/<slug>.md`` page.

    Acceptance criteria (issue #174 / ADR-0012):

    - Acquires ``_filing_lock`` so delete and filing/promote cannot
      interleave on the same slug (same lock used by all qa mutators).
    - Not found → raise ``QaPageNotFound`` (route → 404).
    - Reads frontmatter; if ``status == "live"`` → raise ``QaPageLive``
      (route → 409 Conflict).  Live is the precious corpus state — refuse
      to delete it via a console button (ADR-0012).
    - All other cases are **inert** and safe to delete:
        - ``status == "draft"`` — explicitly deletable.
        - Frontmatter unparseable (``_read_existing_frontmatter`` returns
          ``None``) — the page was never in the corpus; curator typo or
          corrupt file.
        - ``status`` not in ``{draft, live}`` (e.g. ``stale``, ``Live``
          typo, forward-compat reserved value) — also inert; delete is
          the safe remediation surfaced by the C10 lint check.
    - Deletes the file via ``path.unlink()``.
    - Emits a ``qa_deleted`` log event inside the same lock critical
      section (atomically with the delete).

    Returns:
        ``DeletedQaPage`` with the slug and the ``prev_status`` the page
        had before deletion (``"<unparseable>"`` when frontmatter could not
        be parsed).

    Raises:
        QaPageNotFound: no ``wiki/qa/<slug>.md`` on disk (route → 404), OR
            ``slug`` is not a bare filename component (issue #397 — see
            ``promote``'s ``Raises`` entry for why; ``slugs.is_bare_slug``
            rejects it before any filesystem access — also route → 404).
        QaPageLive: existing ``status`` is ``"live"`` — delete refused
            (route → 409).
    """
    if not is_bare_slug(slug):
        raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

    qa_dir = _qa_dir()
    path = qa_dir / f"{slug}.md"

    with _filing_lock:
        if not path.exists():
            raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

        existing = _read_existing_frontmatter(path)
        # Determine the pre-deletion status for the log event.
        if existing is None:
            # Frontmatter could not be parsed at all (corrupt YAML, missing
            # fences, etc.).  The page is inert — it was never in the corpus
            # because the indexer filter rejects unparseable frontmatter.
            # Delete it: this is the C10 "fix-or-discard" use case.
            prev_status = "<unparseable>"
        else:
            prev_status = str(existing.get("status", "<missing>"))
            # The ONLY refusal: live pages are precious corpus content.
            # Everything else (draft, invalid status value, missing status)
            # is inert and may be deleted.
            if prev_status == "live":
                raise QaPageLive(
                    f"wiki/qa/{slug}.md has status=live and cannot be deleted "
                    "via the Console. Use re-ingest to refresh a stale live page."
                )

        path.unlink()
        log_event(
            "qa_deleted",
            f"slug={slug} prev_status={prev_status}",
        )
        return DeletedQaPage(slug=slug, prev_status=prev_status)


# ---------------------------------------------------------------------------
# Demote (issue #535, ADR-0037)
# ---------------------------------------------------------------------------


def demote(slug: str) -> FiledStatus:
    """Curator-driven demote: flip ``wiki/qa/<slug>.md`` ``status: live -> draft`` in place.

    ADR-0037 — the C10 remediation for a schema-invalid ``status: live``
    page: ``qa.delete`` refuses any live page (ADR-0012), so a live-but-
    defective Filed Answer could previously neither be discarded nor fixed.
    Demote is the reversible inverse of ``promote`` — a lifecycle bit flip,
    no LLM, no synthesis — so the page leaves the BM25 corpus and re-enters
    the Promote/Edit/Discard Curation Queue loop, where the curator either
    fixes the schema and re-promotes, or discards it (draft delete is
    already allowed). A standalone primitive, distinct from ``refile``'s own
    inline retire step (ADR-0035), which stays parametrized by a specific
    re-ground failure reason and keeps its own ``qa_reflect op=retired``
    logging for the C9 audit trail.

    - Acquires ``_filing_lock`` (same as every other qa mutator).
    - Not found -> ``QaPageNotFound``, OR ``slug`` is not a bare filename
      component (issue #397 — see ``promote``'s ``Raises`` entry for why;
      ``slugs.is_bare_slug`` rejects it before any filesystem access).
    - Corrupt/invalid-status frontmatter -> ``QaPageCorrupt`` (orphan-
      visibility — surface broken state rather than silently rewriting it,
      mirrors ``promote``).
    - **Idempotent**: if ``status`` is already ``draft``, returns the
      existing ``FiledStatus`` without rewriting the file or emitting a log
      entry — mirrors ``promote``'s already-live idempotence.
    - Otherwise rewrites the page atomically (tmp + ``os.replace``) with
      ``status: "draft"``, preserving ``question``/body/``count``/
      ``created``/``sources``/``open_questions`` verbatim. Refreshes
      ``updated`` to "now".
    - Emits ``qa_demoted slug=<slug> prev_status=live`` — a distinct log
      kind from ``qa_reflect`` (this is a curator lifecycle action, not a
      filing reflect entry).

    Returns:
        ``FiledStatus`` with ``status="draft"``, ``op="touched"`` (demote is
        structurally a touch from the FiledStatus enum perspective, same
        reuse ``promote``/``edit`` make of it), and the preserved ``count``.

    Raises:
        QaPageNotFound: no ``wiki/qa/<slug>.md`` on disk, OR ``slug`` is not
            a bare filename component.
        QaPageCorrupt: existing frontmatter has invalid/unparseable
            ``status``. The page is left untouched so the curator can
            inspect and repair.
    """
    if not is_bare_slug(slug):
        raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

    qa_dir = _qa_dir()
    path = qa_dir / f"{slug}.md"

    with _filing_lock:
        if not path.exists():
            raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

        existing = _read_existing_frontmatter(path)
        if existing is None:
            raise QaPageCorrupt(f"wiki/qa/{slug}.md exists but its frontmatter could not be parsed")

        existing_status = existing.get("status")
        if existing_status not in {"draft", "live"}:
            raise QaPageCorrupt(
                f"wiki/qa/{slug}.md has invalid status={existing_status!r}; "
                "expected 'draft' or 'live'. The page is left untouched so the "
                "curator can inspect and repair."
            )

        try:
            existing_count = int(existing.get("count", 1))
        except (TypeError, ValueError):
            existing_count = 1

        # Idempotent: demoting an already-draft page is a no-op. No rewrite, no log.
        if existing_status == "draft":
            return FiledStatus(slug=slug, status="draft", op="touched", count=existing_count)

        now = _utc_now_iso()
        fm = WikiPageFrontmatter(
            id=existing.get("id", slug),
            type="qa",
            created=existing.get("created", now),
            updated=now,
            sources=[str(s) for s in existing.get("sources", [])],
            status="draft",
            open_questions=existing.get("open_questions") or [],
            question=existing.get("question"),
            count=existing_count,
        )
        body = _read_existing_body(path)
        content = _render_qa_page(fm, body)
        _atomic_write(path, content)

        log_event("qa_demoted", f"slug={slug} prev_status=live")
        return FiledStatus(slug=slug, status="draft", op="touched", count=existing_count)


# ---------------------------------------------------------------------------
# Re-file (tier-B S4, issue #380, ADR-0026 decision 1)
# ---------------------------------------------------------------------------


class QaRefileRejected(Exception):
    """Raised by ``refile`` when the fresh re-synthesis fails the Grounding
    Check (ADR-0026 decision 1, step 2 — "grounding-check before any write").

    Nothing is written when this is raised: the old live page keeps serving
    and the C9 finding stays (ADR-0026 § Consequences Invariant: "a failed
    re-ground during re-file writes nothing: no demote, no draft, no
    reindex"). Carries the caller-facing ``GroundingInfo`` so the route can
    report why — mirrors ``reconcile.ReconcileGroundingFailed``."""

    def __init__(self, grounding: GroundingInfo) -> None:
        self.grounding = grounding
        super().__init__(f"qa refile failed to re-ground: reason={grounding.reason}")


# The re-synthesis failure reasons that trigger the fail-closed RETIRE path
# (demote-to-draft) on a LIVE page — ADR-0035 — because continuing to serve a
# stale answer the KB can no longer back is worse, for a fail-closed KB, than
# returning Cannot Confirm. Aliased to the shared
# ``grounding.CONTENT_FAILURE_REASONS`` (the single source of truth for the
# content-vs-transient split) so a new content-failure reason added to the
# reason enum can never be missed here — drift would silently reclassify an
# un-groundable answer as transient and keep serving it forever, the exact bug
# this gate exists to prevent. ``verifier_unavailable`` / ``index_missing`` are
# operational/transient and stay write-nothing (keep serving, retry later); an
# unrecognised reason is likewise treated as transient (not in the set).
_RETIRE_REFILE_REASONS = CONTENT_FAILURE_REASONS


class RefiledAnswer:
    """Result of a ``refile`` that changed the page.

    Two shapes, discriminated by ``retired``:

    - ``retired == False`` (the happy path) — a fresh answer re-grounded and
      overwrote the page. ``filed`` is ``status="draft"`` with the FRESH
      content; ``grounding`` is the passing ``GroundingInfo``.
    - ``retired == True`` (ADR-0035 fail-closed path) — the re-synthesis could
      NOT be grounded (a content failure in ``_RETIRE_REFILE_REASONS``) and the
      page was LIVE, so the stale answer was demoted to draft IN PLACE (old
      content preserved) rather than left serving. ``filed`` is ``status="draft"``
      with the OLD content; ``grounding`` is the FAILING ``GroundingInfo`` that
      justified the retire. Either way the page leaves the live corpus and the
      curator reviews it via the existing Promote/Edit/Discard Curation Queue
      loop; a transient failure instead raises ``QaRefileRejected`` (nothing
      written, keep serving).

    Not a Pydantic model: mirrors ``DeletedQaPage``'s route-internal-only
    convention — the route layer wraps this into its own response schema
    (adding ``sections_indexed`` from its own ``build_index()`` call, the
    same reindex-is-a-route-concern split ``promote``/``reconcile_apply``
    already use)."""

    __slots__ = ("filed", "grounding", "retired")

    def __init__(self, filed: FiledStatus, grounding: GroundingInfo, retired: bool = False) -> None:
        self.filed = filed
        self.grounding = grounding
        self.retired = retired


def _grounding_info_from_outcome(outcome: GroundingOutcome) -> GroundingInfo:
    """Map a single ``GroundingOutcome`` to the caller-facing ``GroundingInfo``.

    Refile re-synthesizes from exactly one call, so there is only ever one
    outcome to map — no combination step like ``reconcile._combine_grounding``
    (two pages) needs. Mirrors the ``/chat`` route's inline mapping
    (``routes.chat``) so the three surfaces that expose grounding detail
    (``/chat``, reconcile/collision, refile) stay shape-consistent.
    """
    claims = None
    if outcome.result is not None and outcome.result.claims:
        claims = [
            GroundingClaim(
                text=c.text, supported=c.supported, citing_section_ids=c.citing_section_ids
            )
            for c in outcome.result.claims
        ]
    unsupported_claims = None
    if outcome.reason == "claim_unsupported" and outcome.result is not None:
        unsupported_claims = outcome.result.unsupported_claims or None
    return GroundingInfo(
        passed=outcome.passed,
        reason=outcome.reason,
        claims=claims,
        unsupported_claims=unsupported_claims,
    )


def refile(slug: str) -> RefiledAnswer:
    """Curator-driven C9 remediation: chained re-file (ADR-0026 decision 1).

    Fixed internal order — the order is the design:

    1. Read the page's recorded ``question`` (lock held only for this read;
       the re-synthesis call below is an LLM round-trip and must not hold
       ``_filing_lock`` for its duration — that would block every other qa
       mutator on any slug for the ~seconds a chat call takes).
    2. Re-synthesize: run the question through the chat pipeline
       (``retrieval.query``) with ``wiki/qa/`` excluded from retrieval
       (ADR-0026 decision 1 step 1) — the fresh answer must re-derive from
       entities/concepts, never from the stale qa page being re-filed.
    3. Grounding-check BEFORE any write. A failing outcome splits by reason
       (ADR-0035): a CONTENT failure (``_RETIRE_REFILE_REASONS`` — the KB can
       no longer ground a fresh answer) on a LIVE page RETIRES the stale
       answer, demoting it to draft in place (old content preserved) so it
       leaves the corpus and lands in the Curation Queue
       (``RefiledAnswer.retired = True``); a TRANSIENT failure (verifier/index
       unavailable), or a non-live page, raises ``QaRefileRejected`` and
       writes nothing (keep serving, retry later).
    4. Only on a passing re-ground: re-acquire the lock and overwrite the SAME slug in
       place with the fresh answer and its fresh cited Sections,
       ``status: draft``, bumped ``updated`` — preserving ``id``/
       ``created``/``count``/``open_questions``. The stale answer leaves the
       BM25 corpus once the caller reindexes (route layer, mirrors
       ``promote``'s auto-reindex convention — reindex is not this module's
       concern, per ``POST /qa/{slug}/promote``).

    Args:
        slug: the ``wiki/qa/<slug>.md`` slug to re-file.

    Returns:
        ``RefiledAnswer`` (``filed`` status=draft; the passing
        ``GroundingInfo``).

    Raises:
        QaPageNotFound: no ``wiki/qa/<slug>.md`` on disk (checked both at
            the initial read and again before the write — a page deleted by
            a concurrent operation during the re-synthesis round-trip is
            reported the same way), OR ``slug`` is not a bare filename
            component (issue #397 — see ``promote``'s ``Raises`` entry for
            why; ``slugs.is_bare_slug`` rejects it up front, before the
            initial read).
        QaPageCorrupt: existing frontmatter has invalid/unparseable
            ``status``, or no recorded ``question`` (orphan-visibility —
            surface broken state rather than guessing).
        QaRefileRejected: the re-synthesis failed for a TRANSIENT reason
            (verifier/index unavailable), or on a non-live page; nothing is
            written. A CONTENT failure on a LIVE page RETIRES instead (returns
            ``RefiledAnswer.retired = True``, not raised — see step 3 / ADR-0035).
    """
    if not is_bare_slug(slug):
        raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

    qa_dir = _qa_dir()
    path = qa_dir / f"{slug}.md"

    with _filing_lock:
        if not path.exists():
            raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

        existing = _read_existing_frontmatter(path)
        if existing is None:
            raise QaPageCorrupt(f"wiki/qa/{slug}.md exists but its frontmatter could not be parsed")

        existing_status = existing.get("status")
        if existing_status not in {"draft", "live"}:
            raise QaPageCorrupt(
                f"wiki/qa/{slug}.md has invalid status={existing_status!r}; "
                "expected 'draft' or 'live'. The page is left untouched so the "
                "curator can inspect and repair."
            )

        question = existing.get("question")
        if not question:
            raise QaPageCorrupt(f"wiki/qa/{slug}.md has no recorded question; cannot re-file")

        existing_id = existing.get("id", slug)
        try:
            existing_count = int(existing.get("count", 1))
        except (TypeError, ValueError):
            existing_count = 1
        existing_created = existing.get("created", _utc_now_iso())
        existing_open_questions = existing.get("open_questions") or []

    # ---- re-synthesize + grounding-check OUTSIDE the lock (LLM round-trip,
    # ADR-0026 decision 1 steps 1-2). Lazy import: retrieval.py lazy-imports
    # this module too (stream_query's filing dispatch), so a module-level
    # import here would create an import cycle.
    from .retrieval import query as _retrieval_query

    result = _retrieval_query(question, exclude_qa=True)
    outcome = result["grounding_outcome"]
    grounding = _grounding_info_from_outcome(outcome)

    if not outcome.passed:
        # ADR-0035 — split the failure by what it says about the KB, superseding
        # ADR-0026's blanket "a failed re-ground writes nothing":
        #   * CONTENT failure (_RETIRE_REFILE_REASONS) on a LIVE page → RETIRE:
        #     the KB genuinely can no longer ground a fresh answer, so the stale
        #     live answer is demoted to draft IN PLACE (old content preserved).
        #     It leaves the corpus (/chat now fails closed with Cannot Confirm),
        #     C9 stops firing (no longer live), and the draft lands in the
        #     Curation Queue for the curator to salvage (edit) or discard. This
        #     is the escape hatch for the otherwise-permanently-stuck state
        #     (re-file fails, delete refuses a live page).
        #   * TRANSIENT failure (verifier_unavailable / index_missing), or a page
        #     that is not live → write nothing, keep serving, retry later (the
        #     original ADR-0026 behavior — the failure is operational, not a
        #     verdict on the KB's content).
        if existing_status == "live" and outcome.reason in _RETIRE_REFILE_REASONS:
            with _filing_lock:
                if not path.exists():
                    raise QaPageNotFound(f"wiki/qa/{slug}.md not found")
                current = _read_existing_frontmatter(path)
                if current is None:
                    raise QaPageCorrupt(
                        f"wiki/qa/{slug}.md exists but its frontmatter could not be parsed"
                    )
                if current.get("status") != "live":
                    # A concurrent operation already moved it out of the live
                    # corpus during the re-synthesis round-trip — nothing to
                    # retire; report the re-ground failure as a transient one.
                    log_event("qa_refile_rejected", f"slug={slug} reason={outcome.reason}")
                    raise QaRefileRejected(grounding)
                try:
                    retired_count = int(current.get("count", 1))
                except (TypeError, ValueError):
                    retired_count = 1
                fm = WikiPageFrontmatter(
                    id=current.get("id", slug),
                    type="qa",
                    created=current.get("created", _utc_now_iso()),
                    updated=_utc_now_iso(),
                    sources=[str(s) for s in current.get("sources", [])],
                    status="draft",
                    open_questions=current.get("open_questions") or [],
                    question=current.get("question") or question,
                    count=retired_count,
                )
                body = _read_existing_body(path)
                content = _render_qa_page(fm, body)
                _atomic_write(path, content)
                log_event("qa_reflect", f"slug={slug} op=retired reason={outcome.reason}")
                return RefiledAnswer(
                    filed=FiledStatus(slug=slug, status="draft", op="touched", count=retired_count),
                    grounding=grounding,
                    retired=True,
                )
        log_event("qa_refile_rejected", f"slug={slug} reason={outcome.reason}")
        raise QaRefileRejected(grounding)

    cited_ids = [s["source"] for s in result["sources"]]

    # ---- write, only on pass (ADR-0026 decision 1 step 3) ----
    with _filing_lock:
        if not path.exists():
            raise QaPageNotFound(f"wiki/qa/{slug}.md not found")

        fm = WikiPageFrontmatter(
            id=existing_id,
            type="qa",
            created=existing_created,
            updated=_utc_now_iso(),
            sources=cited_ids,
            status="draft",
            open_questions=existing_open_questions,
            question=question,
            count=existing_count,
        )
        content = _render_qa_page(fm, result["answer"])
        _atomic_write(path, content)

        log_event("qa_reflect", f"slug={slug} op=refiled count={existing_count}")
        return RefiledAnswer(
            filed=FiledStatus(slug=slug, status="draft", op="touched", count=existing_count),
            grounding=grounding,
        )
