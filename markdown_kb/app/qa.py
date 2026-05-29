"""Deep module per Ousterhout. Public surface: ``compute_slug``, ``normalize_question``, ``maybe_file_answer``, ``dispatch_filing``, ``promote``, ``QaPageNotFound``, ``QaPageCorrupt``.

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
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

from .grounding import CitableContent
from .indexer import slugify
from .logger import log_event
from .schemas import FiledStatus, WikiPageFrontmatter

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
    """Tmp file + ``os.replace`` write per CODING_STANDARD §2.6.

    Mirrors ``wiki_writer._write_pages_for_source``'s pattern: write to a
    sibling tmp file in the same directory, then ``os.replace`` to swap.
    Same-directory tmp file is required for atomicity on POSIX (cross-device
    rename is non-atomic) — works on Windows too because Python's
    ``os.replace`` is a renameat-style call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
        prefix=f"{path.stem}_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path_str, path)
    except Exception:
        # Best-effort tmp cleanup if write or replace failed.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path_str)
        raise


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


def dispatch_filing(query: str, result: dict) -> FiledStatus | None:
    """Gate-and-dispatch filing for a single query result.

    Shared helper used by both ``POST /chat`` (routes.py) and the Wiki
    ``stream_query`` path (retrieval.py) to avoid duplicated dispatch logic
    (Phase 9 Slice 4 / issue #121 AC1).

    Gating rules (behaviour-preserving — identical to the inline block that
    previously lived in routes.chat):
    - ``result["grounding_outcome"].passed`` must be True; Cannot Confirm
      paths return ``None`` without calling ``maybe_file_answer``.
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
        fail-soft path (Cannot Confirm, IOError, orphan-status guard).
    """
    outcome = result["grounding_outcome"]
    if not outcome.passed:
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
        QaPageNotFound: no ``wiki/qa/<slug>.md`` on disk.
        QaPageCorrupt: existing frontmatter has invalid / unparseable
            ``status`` (orphan zombie). The file is left untouched so the
            curator can inspect.
    """
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

        # Draft -> live transition. Rebuild the full frontmatter so the YAML
        # block stays canonical (avoids surgical mid-line edits that would
        # break round-tripping). Body is preserved verbatim.
        now = _utc_now_iso()
        existing_sources = [str(s) for s in existing.get("sources", [])]
        existing_question = existing.get("question")
        existing_created = existing.get("created", now)
        existing_open_questions = existing.get("open_questions") or []

        fm = WikiPageFrontmatter(
            id=existing.get("id", slug),
            type="qa",
            created=existing_created,
            updated=now,
            sources=existing_sources,
            status="live",
            open_questions=existing_open_questions,
            question=existing_question,
            count=existing_count,
        )
        body = _read_existing_body(path)
        content = _render_qa_page(fm, body)
        _atomic_write(path, content)

        log_event("qa_reflect", f"slug={slug} op=promoted by=curator")
        return FiledStatus(slug=slug, status="live", op="touched", count=existing_count)
