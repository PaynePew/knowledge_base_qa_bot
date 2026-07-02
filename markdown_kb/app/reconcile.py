"""Deep module per Ousterhout. Public surface: ``generate_reconcile``, ``apply_reconcile``, ``ReconcileApplyResult``, ``PageNotFound``, ``PageCorrupt``, ``ReconcileInvalidPair``, ``ReconcileHashMismatch``, ``ReconcileGroundingFailed``.

Coherence Remediation (C5) — stateless two-phase Reconcile flow (tier-B S1,
issue #376, ADR-0028).

Two routes call through here (``markdown_kb/app/routes.py``):

    POST /pages/reconcile        -> generate_reconcile()  (writes nothing to disk)
    POST /pages/reconcile/apply  -> apply_reconcile()      (writes both pages, once, on pass)

Design (ADR-0028):

- **Stateless, server-revalidated.** ``generate_reconcile`` drafts from the
  union of both pages' Sources via ``lint.generate_reconcile_draft`` (the
  LLM call site stays inside ``lint.py``, the ADR-0005-blessed module for
  contradiction-related calls), grounding-checks the draft, and returns it
  with each page's content hash. Nothing is written to disk — no page write,
  no reindex.
- **Content preserved except the post-frontmatter blob.** A page's ``id`` /
  ``type`` / ``created`` / ``sources`` / ``source_hashes`` frontmatter is
  copied through UNCHANGED on apply — only ``updated``/``status`` are bumped
  and the content after the frontmatter fence (heading + prose + citation
  line, one opaque blob — the same unit ``lint._load_wiki_pages`` already
  judges C5 pairs by) is replaced. The LLM drafts *from* the union of both
  pages' Sources as grounding context, but a page's own declared ``sources``
  field is intentionally left narrower than what informed the rewrite — the
  apply-time grounding re-check re-verifies the exact submitted content
  against that same union every time, so this is safe, and it avoids
  inventing a multi-file citation-line convention outside tracer-bullet scope.
- **Hash-based optimistic concurrency.** ``hash_a``/``hash_b`` are SHA-256 of
  each page's FULL on-disk file text (frontmatter + content) at generate
  time. ``apply_reconcile`` recomputes both hashes from the CURRENT on-disk
  files and refuses (``ReconcileHashMismatch``) if either changed — the
  draft was computed against page state that no longer exists.
- **Apply re-verifies grounding on the exact submitted content** (possibly
  human-edited from the generated draft) against the same union of Sources,
  and refuses (``ReconcileGroundingFailed``) on failure. Only a pass reaches
  the write.

Reindex is deliberately NOT triggered from this module — ``routes.py`` calls
``indexer.build_index()`` exactly once after ``apply_reconcile`` returns,
mirroring the existing ``POST /qa/{slug}/promote`` convention (reindex lives
at the route layer, not the domain layer).
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from ._paths import DOCS_DIR
from .atomic import write_text_atomic
from .grounding import GroundingOutcome, verify
from .indexer import parse_markdown
from .lint import generate_reconcile_draft
from .logger import log_event
from .schemas import GroundingClaim, GroundingInfo, ReconcileApplyRequest, ReconcileGenerateResponse

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PageNotFound(Exception):
    """Raised when a reconcile target slug does not resolve to an existing
    ``wiki/entities/`` or ``wiki/concepts/`` page."""


class PageCorrupt(Exception):
    """Raised when a reconcile target page exists on disk but its
    frontmatter cannot be parsed (mirrors ``qa.QaPageCorrupt`` — orphan-
    visibility: surface broken state rather than silently rewriting it)."""


class ReconcileInvalidPair(Exception):
    """Raised when ``page_a`` and ``page_b`` name the same slug."""


class ReconcileHashMismatch(Exception):
    """Raised when either page's current on-disk content hash no longer
    matches the generate-time hash (ADR-0028 Invariant — apply refuses on
    hash mismatch)."""


class ReconcileGroundingFailed(Exception):
    """Raised when the apply-time grounding re-check fails for either page's
    submitted content (ADR-0028 Invariant). Carries the combined
    ``GroundingInfo`` so the route can render the failure honestly."""

    def __init__(self, grounding: GroundingInfo) -> None:
        self.grounding = grounding
        super().__init__("reconcile content failed grounding re-check")


# ---------------------------------------------------------------------------
# Path / page-reading helpers
# ---------------------------------------------------------------------------


def _resolve_wiki_dir(wiki_dir: Path | None) -> Path:
    """Resolve the wiki root, importing ``indexer`` at call time so a
    test's ``monkeypatch.setattr(indexer, "WIKI_DIR", ...)`` is honoured."""
    if wiki_dir is not None:
        return wiki_dir
    from . import indexer

    return indexer.WIKI_DIR


def _find_page_path(slug: str, wiki_dir: Path) -> Path:
    """Return the on-disk path for ``slug`` under entities/ or concepts/.

    C5 excludes ``type: qa`` pages from candidate pair generation (Slice 6-5
    modifier), so a reconcile target is always an entity or concept page.
    """
    for subdir_name in ("entities", "concepts"):
        candidate = wiki_dir / subdir_name / f"{slug}.md"
        if candidate.exists():
            return candidate
    raise PageNotFound(slug)


def _parse_frontmatter_text(raw_text: str) -> dict | None:
    """Parse the YAML frontmatter block from a wiki page's full file text.

    Scans for ``---`` fence LINES (not ``str.startswith``) so a page that
    opens with a sentinel HTML comment before the fence still parses —
    mirrors ``lint._parse_frontmatter`` / ``wiki_writer.read_existing_frontmatter``.
    """
    lines = raw_text.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(dash_indices) < 2:
        return None
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_content_after_frontmatter(raw_text: str) -> str:
    """Return everything after the frontmatter's closing ``---`` fence.

    This is the reconcile unit of content — heading + prose + trailing
    citation line as one opaque blob, the same shape
    ``lint._load_wiki_pages`` already uses to judge C5 pairs.
    """
    lines = raw_text.splitlines(keepends=True)
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(dash_indices) < 2:
        return raw_text
    return "".join(lines[dash_indices[1] + 1 :]).lstrip("\n")


def _content_hash(raw_text: str) -> str:
    """SHA-256 hex digest of a page's full on-disk file text (OCC token)."""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def _read_page(slug: str, wiki_dir: Path) -> tuple[Path, dict, str, str]:
    """Return ``(path, frontmatter, raw_text, content_after_frontmatter)`` for ``slug``.

    Raises ``PageNotFound`` when the slug does not resolve to a page at all,
    or ``PageCorrupt`` when the file exists but its frontmatter cannot be
    parsed (a page reconcile cannot safely round-trip through corrupt
    frontmatter).
    """
    path = _find_page_path(slug, wiki_dir)
    raw_text = path.read_text(encoding="utf-8")
    fm = _parse_frontmatter_text(raw_text)
    if fm is None:
        raise PageCorrupt(slug)
    content = _extract_content_after_frontmatter(raw_text)
    return path, fm, raw_text, content


# ---------------------------------------------------------------------------
# Union-of-Sources collection (ADR-0028: the LLM drafts from BOTH pages'
# Sources; the grounding re-check verifies against the same union)
# ---------------------------------------------------------------------------


def _union_source_filenames(fm_a: dict, fm_b: dict) -> list[str]:
    """Return the deduplicated, order-preserving list of Source filenames
    cited by either page's ``frontmatter.sources`` (anchor stripped)."""
    seen: set[str] = set()
    out: list[str] = []
    for fm in (fm_a, fm_b):
        for citation in fm.get("sources", []) or []:
            file_part = str(citation).split("#")[0].strip()
            if file_part and file_part not in seen:
                seen.add(file_part)
                out.append(file_part)
    return out


def _collect_union_sections(fm_a: dict, fm_b: dict, docs_dir: Path) -> list:
    """Parse every Source file cited by either page and return the
    deduplicated union of their Sections (CitableContent-satisfying).

    A Source file that no longer exists is skipped (best-effort — matches
    C11's orphan-tolerant scan style; a missing Source degrades the
    grounding context rather than raising).
    """
    sections: list = []
    seen_ids: set[str] = set()
    for filename in _union_source_filenames(fm_a, fm_b):
        source_path = docs_dir / filename
        if not source_path.exists():
            matches = list(docs_dir.glob(f"**/{filename}"))
            if not matches:
                continue
            source_path = matches[0]
        try:
            parsed = parse_markdown(source_path)
        except Exception:  # noqa: BLE001 — a malformed Source degrades context, not a hard error
            continue
        for section in parsed:
            if section.id not in seen_ids:
                seen_ids.add(section.id)
                sections.append(section)
    return sections


# ---------------------------------------------------------------------------
# Grounding combination (two pages, one report)
# ---------------------------------------------------------------------------


def _claims_from_outcome(outcome: GroundingOutcome) -> list[GroundingClaim]:
    if outcome.result is not None and outcome.result.claims:
        return [
            GroundingClaim(
                text=c.text, supported=c.supported, citing_section_ids=c.citing_section_ids
            )
            for c in outcome.result.claims
        ]
    return []


def _combine_grounding(outcome_a: GroundingOutcome, outcome_b: GroundingOutcome) -> GroundingInfo:
    """Merge two per-page ``GroundingOutcome``s into one caller-facing report.

    ``passed`` is the AND of both. ``reason`` prefers the failing side's
    reason (page A first, deterministic) so a curator sees why it failed;
    ``claim_supported`` when both pass. Claims/unsupported claims from both
    pages are concatenated so nothing is silently dropped.
    """
    passed = outcome_a.passed and outcome_b.passed
    if passed:
        reason = "claim_supported"
    elif not outcome_a.passed:
        reason = outcome_a.reason
    else:
        reason = outcome_b.reason

    claims = _claims_from_outcome(outcome_a) + _claims_from_outcome(outcome_b)

    unsupported: list[str] = []
    for outcome in (outcome_a, outcome_b):
        if outcome.reason == "claim_unsupported" and outcome.result is not None:
            unsupported.extend(outcome.result.unsupported_claims or [])

    return GroundingInfo(
        passed=passed,
        reason=reason,  # type: ignore[arg-type]
        claims=claims or None,
        unsupported_claims=unsupported or None,
    )


# ---------------------------------------------------------------------------
# Write-back (apply only)
# ---------------------------------------------------------------------------

_SENTINEL_TEMPLATE = (
    "<!-- Reconciled by POST /pages/reconcile/apply on {ts} (paired with '{other_slug}').\n"
    "     Grounded in the union of both pages' Sources, re-verified at apply time.\n"
    "     Manual edits are safe until the next reconcile or ingest of the\n"
    "     underlying Source(s) — edit the Source for a permanent change. -->"
)


def _write_reconciled_page(
    path: Path, fm: dict, content: str, now_iso: str, other_slug: str
) -> None:
    """Rewrite one page in place: bumped ``updated``/``status``, new content.

    All other frontmatter fields (``id``, ``type``, ``created``, ``sources``,
    ``source_hashes``, ...) are copied through unchanged — see the module
    docstring for why the ``sources`` field is intentionally NOT widened to
    the union used for grounding.
    """
    updated_fm = dict(fm)
    updated_fm["updated"] = now_iso
    updated_fm["status"] = "live"
    updated_fm.pop("grounding_failure", None)
    fm_yaml = yaml.dump(updated_fm, default_flow_style=False, allow_unicode=True).rstrip()
    sentinel = _SENTINEL_TEMPLATE.format(ts=now_iso, other_slug=other_slug)
    text = "\n".join([sentinel, "", "---", fm_yaml, "---", "", content.rstrip(), ""])
    write_text_atomic(path, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_reconcile(
    page_a: str,
    page_b: str,
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> ReconcileGenerateResponse:
    """Draft a reconciled version of two contradicting pages. Writes nothing to disk.

    Raises ``PageNotFound`` when either slug does not resolve, or
    ``ReconcileInvalidPair`` when ``page_a == page_b``.
    """
    if page_a == page_b:
        raise ReconcileInvalidPair(page_a)

    resolved_wiki = _resolve_wiki_dir(wiki_dir)
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR

    _path_a, fm_a, raw_a, old_content_a = _read_page(page_a, resolved_wiki)
    _path_b, fm_b, raw_b, old_content_b = _read_page(page_b, resolved_wiki)

    hash_a = _content_hash(raw_a)
    hash_b = _content_hash(raw_b)

    union_sections = _collect_union_sections(fm_a, fm_b, resolved_docs)

    draft = generate_reconcile_draft(page_a, old_content_a, page_b, old_content_b, union_sections)

    outcome_a = verify(draft.content_a, union_sections)
    outcome_b = verify(draft.content_b, union_sections)
    grounding = _combine_grounding(outcome_a, outcome_b)

    log_event(
        "reconcile_generate",
        f"page_a={page_a} page_b={page_b} passed={grounding.passed} reason={grounding.reason}",
    )

    return ReconcileGenerateResponse(
        page_a=page_a,
        page_b=page_b,
        old_content_a=old_content_a,
        old_content_b=old_content_b,
        content_a=draft.content_a,
        content_b=draft.content_b,
        grounding=grounding,
        hash_a=hash_a,
        hash_b=hash_b,
    )


@dataclass
class ReconcileApplyResult:
    """Outcome of a successful ``apply_reconcile`` call.

    ``routes.py`` wraps this into a ``ReconcileApplyResponse`` after calling
    ``build_index()`` exactly once (reindex stays a route-layer concern —
    matches ``POST /qa/{slug}/promote``).
    """

    page_a: str
    page_b: str
    grounding: GroundingInfo


def apply_reconcile(
    req: ReconcileApplyRequest,
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> ReconcileApplyResult:
    """Re-verify and write back the final (possibly human-edited) reconcile content.

    Raises:
        PageNotFound: either slug does not resolve to an existing page.
        ReconcileInvalidPair: ``page_a == page_b``.
        ReconcileHashMismatch: either page's current on-disk hash no longer
            matches ``req.hash_a`` / ``req.hash_b`` (409 — a page changed
            since generate; the finding may no longer hold).
        ReconcileGroundingFailed: the apply-time grounding re-check failed
            for either page's submitted content (422).
    """
    if req.page_a == req.page_b:
        raise ReconcileInvalidPair(req.page_a)

    resolved_wiki = _resolve_wiki_dir(wiki_dir)
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR

    path_a, fm_a, raw_a, _old_content_a = _read_page(req.page_a, resolved_wiki)
    path_b, fm_b, raw_b, _old_content_b = _read_page(req.page_b, resolved_wiki)

    current_hash_a = _content_hash(raw_a)
    current_hash_b = _content_hash(raw_b)
    if current_hash_a != req.hash_a or current_hash_b != req.hash_b:
        raise ReconcileHashMismatch(f"page_a={req.page_a} page_b={req.page_b}")

    union_sections = _collect_union_sections(fm_a, fm_b, resolved_docs)

    outcome_a = verify(req.content_a, union_sections)
    outcome_b = verify(req.content_b, union_sections)
    grounding = _combine_grounding(outcome_a, outcome_b)
    if not grounding.passed:
        log_event(
            "reconcile_apply_refused",
            f"page_a={req.page_a} page_b={req.page_b} reason={grounding.reason}",
        )
        raise ReconcileGroundingFailed(grounding)

    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_reconciled_page(path_a, fm_a, req.content_a, now_iso, req.page_b)
    _write_reconciled_page(path_b, fm_b, req.content_b, now_iso, req.page_a)

    log_event("reconcile_applied", f"page_a={req.page_a} page_b={req.page_b}")

    return ReconcileApplyResult(page_a=req.page_a, page_b=req.page_b, grounding=grounding)
