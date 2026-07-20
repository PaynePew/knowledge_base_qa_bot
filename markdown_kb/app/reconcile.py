"""Deep module per Ousterhout. Public surface: ``generate_reconcile``, ``apply_reconcile``, ``ReconcileApplyResult``, ``PageNotFound``, ``PageCorrupt``, ``ReconcileInvalidPair``, ``ReconcileHashMismatch``, ``ReconcileGroundingFailed``, ``generate_collision_merge``, ``apply_collision_merge``, ``CollisionMergeApplyResult``, ``generate_collision_differentiate``, ``apply_collision_differentiate``, ``CollisionDifferentiateApplyResult``, ``CollisionInvalidGroup``, ``CollisionHashMismatch``, ``CollisionGroundingFailed``, ``CollisionReferenceGuardFailed``.

Coherence Remediation (C5) — stateless two-phase Reconcile flow (tier-B S1,
issue #376, ADR-0028). Coherence Remediation (C4) — dual resolution for
slug-collision groups on top of the same machinery (tier-B S2, issue #378,
ADR-0028).

Six routes call through here (``markdown_kb/app/routes.py``):

    POST /pages/reconcile        -> generate_reconcile()  (writes nothing to disk)
    POST /pages/reconcile/apply  -> apply_reconcile()      (writes both pages, once, on pass)
    POST /pages/collision/merge                -> generate_collision_merge()  (writes nothing to disk)
    POST /pages/collision/merge/apply           -> apply_collision_merge()    (rewrites base, deletes reference-free variants, once, on pass)
    POST /pages/collision/differentiate         -> generate_collision_differentiate()  (writes nothing to disk)
    POST /pages/collision/differentiate/apply   -> apply_collision_differentiate()     (rewrites every group member, once, on pass)

C4's two resolutions (ADR-0028):

- **Merge into base** — merged content lands on the group's unsuffixed
  ``base_slug``; the suffixed variants are deleted *inside apply*, behind
  an **inbound-reference guard**: the server refuses (``CollisionReferenceGuardFailed``)
  when any variant has inbound ``[[links]]`` or qa citations, listing the
  referrers. This is a distinct deletion path from ADR-0025's full-orphan
  ``DELETE /pages/{slug}`` — different predicate, different operation, no
  endpoint sharing.
- **Differentiate** — every page in the group is rewritten in place to be
  complementary and more specific; nobody is deleted, so no reference guard
  is needed.

Both paths reuse S1's hash-based optimistic-concurrency + apply-time
grounding re-verification, generalised from two pages to N.

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

Concurrency: ``apply_reconcile`` writes TWO pages, so — mirroring
``ingest_sources``'s "under ``_index_lock``: delete orphans, then write
pages" convention (both are multi-file wiki writes) — both writes happen
inside one ``indexer._index_lock`` acquisition. Without it, a concurrent
``run_lint()`` (which holds the same lock for its full read sweep) could
observe page_a already rewritten and page_b not yet, i.e. a still-
contradicting snapshot.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from ._paths import DOCS_DIR
from .atomic import write_text_atomic
from .grounding import GroundingOutcome, verify
from .indexer import Section, _index_lock, parse_markdown
from .lint import (
    DIFFERENTIATE_SENTINEL_TEMPLATE,
    _judge_page_pair,
    find_inbound_references,
    generate_collision_differentiate_draft,
    generate_collision_merge_draft,
    generate_reconcile_draft,
)
from .logger import log_event
from .schemas import (
    CitedSourceSection,
    CollisionDifferentiateApplyRequest,
    CollisionDifferentiateGenerateResponse,
    CollisionMergeApplyRequest,
    CollisionMergeGenerateResponse,
    GroundingClaim,
    GroundingInfo,
    InboundReference,
    ReconcileApplyRequest,
    ReconcileGenerateResponse,
)

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


class ReconcileNotConverged(Exception):
    """Raised when the apply-time convergence re-judge finds the submitted
    drafts still contradict each other (ADR-0038): both are grounded, yet
    they give incompatible answers — a source-rooted contradiction the wiki
    layer cannot fix. 422 — fix a Source, not the pages. Grounding cannot
    signal this (it is an existence check against the self-contradicting
    Source union); the C5 contradiction oracle re-run on the drafts can."""

    def __init__(self, page_a: str, page_b: str, summary: str | None = None) -> None:
        self.page_a = page_a
        self.page_b = page_b
        self.summary = summary
        super().__init__(f"reconcile drafts still contradict: page_a={page_a} page_b={page_b}")


class CollisionInvalidGroup(Exception):
    """Raised when a C4 collision group request is malformed: ``base_slug``
    also named in ``variant_slugs``, an empty ``variant_slugs``/``slugs``
    list, duplicate slugs, or (apply-time) a ``content``/``hashes`` mapping
    whose keys do not match the group's slugs exactly."""


class CollisionHashMismatch(Exception):
    """Raised when any group member's current on-disk content hash no
    longer matches the generate-time hash (ADR-0028 Invariant — apply
    refuses on hash mismatch, mirroring ``ReconcileHashMismatch``)."""


class CollisionGroundingFailed(Exception):
    """Raised when the apply-time grounding re-check fails for any group
    member's submitted content (ADR-0028 Invariant). Carries the combined
    ``GroundingInfo`` so the route can render the failure honestly."""

    def __init__(self, grounding: GroundingInfo) -> None:
        self.grounding = grounding
        super().__init__("collision content failed grounding re-check")


class CollisionReferenceGuardFailed(Exception):
    """Raised when a C4 merge-apply finds inbound ``[[links]]`` or qa
    citations to a variant slated for deletion (ADR-0028 Invariant — the
    server refuses, listing referrers, rather than silently orphaning the
    reference)."""

    def __init__(self, referrers: list[InboundReference]) -> None:
        self.referrers = referrers
        super().__init__("collision merge refused: referenced variant(s)")


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


def _read_pages(slugs: list[str], wiki_dir: Path) -> dict[str, tuple[Path, dict, str, str]]:
    """``_read_page`` for every slug in ``slugs``, keyed by slug.

    Generalises the C5 two-page read to a C4 group of N — same
    ``PageNotFound``/``PageCorrupt`` semantics per member, first failure wins
    (deterministic — ``slugs`` order).
    """
    return {slug: _read_page(slug, wiki_dir) for slug in slugs}


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
# Per-page cited Sections (C5 Source comparison payload — issue #534,
# ADR-0036 decision 3). Deliberately SEPARATE from the union collected
# above: the grounding union stays whole-file (decision 7, unchanged), while
# this is presentation data for ONE page's OWN narrower citations.
# ---------------------------------------------------------------------------


def _resolve_cited_source_path(
    source_ref: str, docs_dir: Path
) -> tuple[Path | None, str | None, Literal["resolved", "missing", "ambiguous"]]:
    """Resolve a bare Source citation to its actual on-disk Path, for the C5
    Source comparison view's ``/read/file`` links (ADR-0036 decision 3).

    Mirrors ``lint._resolve_c3_source_path``'s basename-glob-with-ambiguity
    contract (issue #445) — kept as a small local duplicate rather than a
    cross-module private import, the same precedent
    ``ingest._resolve_single_source_pairs`` already set: strip any
    ``#anchor``, take the basename, match it against every file under
    ``docs_dir``. A basename matching 2+ files is never silently guessed.

    Returns ``(None, None, "missing")`` when ``source_ref`` has no basename
    or no file matches, ``(None, None, "ambiguous")`` when 2+ files match, or
    ``(actual_path, "docs/<relative>", "resolved")`` on exactly one match —
    the repo-relative label mirrors ``FailedGroundingFinding.source_path``'s
    display convention.
    """
    file_part = source_ref.split("#")[0].strip()
    filename = Path(file_part).name if file_part else ""
    if not filename:
        return None, None, "missing"

    matches = sorted(docs_dir.glob(f"**/{filename}"))
    if not matches:
        return None, None, "missing"
    if len(matches) > 1:
        return None, None, "ambiguous"

    path = matches[0]
    label = f"docs/{path.relative_to(docs_dir).as_posix()}"
    return path, label, "resolved"


def _cited_sections_for_page(fm: dict, docs_dir: Path) -> list[CitedSourceSection]:
    """Resolve one page's OWN cited Source sections (ADR-0036 decision 3).

    A citation with no ``#anchor`` names a whole Source file rather than one
    section, so every Section parsed from it is included. A citation that
    cannot be resolved to content — an ambiguous/missing Source file, or a
    stale anchor that no longer matches any parsed heading — still produces
    an entry (``heading``/``content`` left ``None``) rather than being
    silently dropped, so the curator sees which citation could not be shown.
    """
    out: list[CitedSourceSection] = []
    seen_ids: set[str] = set()
    parsed_cache: dict[Path, list[Section]] = {}

    def _append_once(entry: CitedSourceSection) -> None:
        if entry.id in seen_ids:
            return
        seen_ids.add(entry.id)
        out.append(entry)

    for raw_citation in fm.get("sources", []) or []:
        citation = str(raw_citation)
        file_part, sep, anchor = citation.partition("#")
        filename = Path(file_part.strip()).name if file_part.strip() else ""
        if not filename:
            continue

        path, source_path, resolution = _resolve_cited_source_path(citation, docs_dir)
        target_id = f"{filename}#{anchor}" if sep else None
        fallback_id = target_id or filename

        if path is None:
            _append_once(
                CitedSourceSection(
                    id=fallback_id, source_path=source_path, source_resolution=resolution
                )
            )
            continue

        if path not in parsed_cache:
            try:
                parsed_cache[path] = parse_markdown(path)
            except Exception:  # noqa: BLE001 — a malformed Source degrades context, not a hard error
                parsed_cache[path] = []
        sections = parsed_cache[path]
        matches = [s for s in sections if s.id == target_id] if target_id else sections

        if not matches:
            # A stale anchor (renamed heading) or an empty parsed file — the
            # Source resolved, but this specific citation has no content.
            _append_once(
                CitedSourceSection(
                    id=fallback_id, source_path=source_path, source_resolution=resolution
                )
            )
            continue

        for section in matches:
            _append_once(
                CitedSourceSection(
                    id=section.id,
                    heading=section.heading,
                    content=section.content,
                    source_path=source_path,
                    source_resolution=resolution,
                )
            )

    # issue #635 (ADR-0044): disclose every remaining sibling section of each
    # resolved cited file, flagged cited=False, AFTER the cited entries (so
    # the cited-first ordering issue #534's consumers rely on is untouched).
    # The grounding/convergence evidence is whole-file (ADR-0036 decision 7);
    # hiding siblings made the report cite claims the curator could not see.
    # parsed_cache preserves first-parse (citation) order; _append_once
    # already dedupes against the cited ids.
    for path, sections in parsed_cache.items():
        # Only resolved files ever enter parsed_cache, so the label is
        # derivable directly — no second basename glob.
        sibling_source_path = f"docs/{path.relative_to(docs_dir).as_posix()}"
        for section in sections:
            _append_once(
                CitedSourceSection(
                    id=section.id,
                    heading=section.heading,
                    content=section.content,
                    source_path=sibling_source_path,
                    source_resolution="resolved",
                    cited=False,
                )
            )
    return out


def _union_source_filenames_n(frontmatters: list[dict]) -> list[str]:
    """N-way generalisation of ``_union_source_filenames``: the deduplicated,
    order-preserving list of Source filenames cited by ANY frontmatter in
    ``frontmatters``."""
    seen: set[str] = set()
    out: list[str] = []
    for fm in frontmatters:
        for citation in fm.get("sources", []) or []:
            file_part = str(citation).split("#")[0].strip()
            if file_part and file_part not in seen:
                seen.add(file_part)
                out.append(file_part)
    return out


def _collect_union_sections_n(frontmatters: list[dict], docs_dir: Path) -> list:
    """N-way generalisation of ``_collect_union_sections`` — the deduplicated
    union of Sections cited by ANY of a C4 collision group's members.

    Same best-effort missing-Source tolerance as the two-page version.
    """
    sections: list = []
    seen_ids: set[str] = set()
    for filename in _union_source_filenames_n(frontmatters):
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
        # mypy cannot narrow GroundingOutcome.reason (the full 6-variant Literal
        # grounding.py declares) to GroundingInfo's post-LLM-only subset from the
        # runtime guard above — same shape as ingest._verify_draft's ignore.
        reason=reason,  # type: ignore[arg-type]
        claims=claims or None,
        unsupported_claims=unsupported or None,
    )


def _combine_grounding_n(outcomes: list[GroundingOutcome]) -> GroundingInfo:
    """N-way generalisation of ``_combine_grounding`` for a C4 collision
    group. ``passed`` is the AND of every outcome; ``reason`` prefers the
    first failing outcome's reason (deterministic — input order) so a
    curator sees why it failed. Claims/unsupported claims from every outcome
    are concatenated so nothing is silently dropped.
    """
    passed = all(outcome.passed for outcome in outcomes)
    reason = "claim_supported" if passed else next(o.reason for o in outcomes if not o.passed)

    claims: list[GroundingClaim] = []
    unsupported: list[str] = []
    for outcome in outcomes:
        claims.extend(_claims_from_outcome(outcome))
        if outcome.reason == "claim_unsupported" and outcome.result is not None:
            unsupported.extend(outcome.result.unsupported_claims or [])

    return GroundingInfo(
        passed=passed,
        reason=reason,  # type: ignore[arg-type] — see _combine_grounding's identical ignore
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


def _rewrite_page(path: Path, fm: dict, content: str, now_iso: str, sentinel: str) -> None:
    """Rewrite one page in place: bumped ``updated``/``status``, new content,
    the caller-built sentinel comment prepended.

    All other frontmatter fields (``id``, ``type``, ``created``, ``sources``,
    ``source_hashes``, ...) are copied through unchanged — see the module
    docstring for why the ``sources`` field is intentionally NOT widened to
    the union used for grounding. Shared by every apply write path (S1's
    two-page reconcile, S2's merge base rewrite, S2's per-member
    differentiate rewrite) — only the sentinel text differs per caller.
    """
    updated_fm = dict(fm)
    updated_fm["updated"] = now_iso
    updated_fm["status"] = "live"
    updated_fm.pop("grounding_failure", None)
    fm_yaml = yaml.dump(updated_fm, default_flow_style=False, allow_unicode=True).rstrip()
    text = "\n".join([sentinel, "", "---", fm_yaml, "---", "", content.rstrip(), ""])
    write_text_atomic(path, text)


def _write_reconciled_page(
    path: Path, fm: dict, content: str, now_iso: str, other_slug: str
) -> None:
    """Rewrite one C5 reconcile page in place — see ``_rewrite_page``."""
    sentinel = _SENTINEL_TEMPLATE.format(ts=now_iso, other_slug=other_slug)
    _rewrite_page(path, fm, content, now_iso, sentinel)


_MERGE_SENTINEL_TEMPLATE = (
    "<!-- Merged by POST /pages/collision/merge/apply on {ts} (merged in: '{variants}').\n"
    "     Grounded in the union of every group member's Sources, re-verified at apply time.\n"
    "     Manual edits are safe until the next reconcile/collision resolution or ingest of the\n"
    "     underlying Source(s) — edit the Source for a permanent change. -->"
)


def _write_merged_base_page(
    path: Path, fm: dict, content: str, now_iso: str, variant_slugs: list[str]
) -> None:
    """Rewrite the group's base page in place with the merged content —
    see ``_rewrite_page``. ``variant_slugs`` names the (now-deleted)
    variants folded into this page, for the sentinel comment."""
    sentinel = _MERGE_SENTINEL_TEMPLATE.format(ts=now_iso, variants=", ".join(variant_slugs))
    _rewrite_page(path, fm, content, now_iso, sentinel)


def _write_differentiated_page(
    path: Path, fm: dict, content: str, now_iso: str, group_slugs: list[str]
) -> None:
    """Rewrite one group member in place with its differentiated content —
    see ``_rewrite_page``. ``group_slugs`` names the FULL collision group
    (including this page's own slug), for the sentinel comment lint's C4-a
    differentiate exemption parses (the template lives in ``lint`` so writer
    and parser cannot drift)."""
    sentinel = DIFFERENTIATE_SENTINEL_TEMPLATE.format(ts=now_iso, group=", ".join(group_slugs))
    _rewrite_page(path, fm, content, now_iso, sentinel)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _rejudge_convergence(
    page_a: str, content_a: str, page_b: str, content_b: str
) -> tuple[bool, str | None]:
    """Re-judge two reconcile drafts for cross-page CONVERGENCE (ADR-0038).

    Grounding is an existence check against the whole-file Source union, so a
    source-rooted pair (each draft faithful to its own Source, the Sources
    themselves disagreeing) passes grounding while the drafts still contradict
    — grounding cannot signal source-rooted. Re-run the C5 contradiction oracle
    (``_judge_page_pair``) on the DRAFTS instead: ``severity == "none"`` means
    the pages now agree (converged → wiki-rooted, Apply-able); any other verdict
    means they still disagree (not converged → source-rooted). Returns
    ``(converged, summary)`` where ``summary`` is the oracle's prose for the
    Source-view note when not converged, else ``None``.

    Fails safe to NOT converged on any judge error — Apply is never enabled
    under uncertainty (cost asymmetry: a false source-rooted only makes the
    curator toggle views; a false converged writes a fresh contradiction into
    the corpus).
    """
    try:
        finding = _judge_page_pair(page_a, content_a, page_b, content_b)
    except Exception as exc:  # noqa: BLE001 — fail closed on any judge error
        log_event(
            "reconcile_rejudge_error",
            f"page_a={page_a} page_b={page_b} err={type(exc).__name__}: {exc}",
        )
        return False, None
    converged = finding.severity == "none"
    return converged, (None if converged else finding.summary)


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

    # Convergence re-judge (ADR-0038): grounding only proves each draft is
    # faithful to the Sources, never that the two drafts AGREE — a source-rooted
    # pair passes grounding while still contradicting. Re-run the C5 oracle on
    # the drafts to get the real source-rooted signal. Drives the two-view
    # modal's default view AND the Apply gate; grounding.passed no longer does.
    converged, convergence_summary = _rejudge_convergence(
        page_a, draft.content_a, page_b, draft.content_b
    )

    # C5 Source-comparison payload (issue #534, ADR-0036 decision 3) — each
    # page's OWN cited sections, narrower than (and independent of) the
    # whole-file union above. Presentation data only: does not affect
    # drafting or the grounding re-check.
    cited_sections_a = _cited_sections_for_page(fm_a, resolved_docs)
    cited_sections_b = _cited_sections_for_page(fm_b, resolved_docs)

    log_event(
        "reconcile_generate",
        f"page_a={page_a} page_b={page_b} passed={grounding.passed} "
        f"reason={grounding.reason} converged={converged}",
    )

    return ReconcileGenerateResponse(
        page_a=page_a,
        page_b=page_b,
        old_content_a=old_content_a,
        old_content_b=old_content_b,
        content_a=draft.content_a,
        content_b=draft.content_b,
        grounding=grounding,
        converged=converged,
        convergence_summary=convergence_summary,
        hash_a=hash_a,
        hash_b=hash_b,
        cited_sections_a=cited_sections_a,
        cited_sections_b=cited_sections_b,
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

    # Convergence gate (ADR-0038 Invariant): both drafts are grounded, but they
    # may still contradict each other (source-rooted, or a human edit that left
    # a conflict). Grounding cannot catch that; the C5 oracle re-run on the
    # submitted content can. Refuse rather than write a fresh contradiction.
    converged, convergence_summary = _rejudge_convergence(
        req.page_a, req.content_a, req.page_b, req.content_b
    )
    if not converged:
        log_event(
            "reconcile_apply_refused",
            f"page_a={req.page_a} page_b={req.page_b} reason=not_converged",
        )
        raise ReconcileNotConverged(req.page_a, req.page_b, convergence_summary)

    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Both pages write under one lock acquisition (see module docstring —
    # mirrors ingest_sources's multi-file wiki write convention) so a
    # concurrent run_lint() never observes one page rewritten and the other
    # still contradicting.
    with _index_lock:
        _write_reconciled_page(path_a, fm_a, req.content_a, now_iso, req.page_b)
        _write_reconciled_page(path_b, fm_b, req.content_b, now_iso, req.page_a)

    log_event("reconcile_applied", f"page_a={req.page_a} page_b={req.page_b}")

    return ReconcileApplyResult(page_a=req.page_a, page_b=req.page_b, grounding=grounding)


# ---------------------------------------------------------------------------
# Public API — C4 collision dual resolution (tier-B S2, issue #378, ADR-0028)
# ---------------------------------------------------------------------------


def _validate_group(base_slug: str, variant_slugs: list[str]) -> None:
    """Shared C4 merge-group shape validation: non-empty, no duplicates,
    ``base_slug`` not among ``variant_slugs``."""
    if not variant_slugs:
        raise CollisionInvalidGroup("variant_slugs must be non-empty")
    if len(set(variant_slugs)) != len(variant_slugs):
        raise CollisionInvalidGroup(f"variant_slugs contains duplicates: {variant_slugs}")
    if base_slug in variant_slugs:
        raise CollisionInvalidGroup(f"base_slug '{base_slug}' must not appear in variant_slugs")


def _validate_slugs(slugs: list[str]) -> None:
    """Shared C4 differentiate-group shape validation: at least 2 members,
    no duplicates."""
    if len(slugs) < 2:
        raise CollisionInvalidGroup(f"slugs must contain at least 2 members, got {slugs}")
    if len(set(slugs)) != len(slugs):
        raise CollisionInvalidGroup(f"slugs contains duplicates: {slugs}")


def generate_collision_merge(
    base_slug: str,
    variant_slugs: list[str],
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> CollisionMergeGenerateResponse:
    """Draft a merged version of a C4 collision group's base page. Writes
    nothing to disk.

    Raises ``CollisionInvalidGroup`` when ``variant_slugs`` is empty,
    contains ``base_slug``, or has duplicates. Raises ``PageNotFound``/
    ``PageCorrupt`` when ``base_slug`` or any variant slug does not resolve
    to an existing page, or its frontmatter is corrupt.
    """
    _validate_group(base_slug, variant_slugs)

    resolved_wiki = _resolve_wiki_dir(wiki_dir)
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR

    _base_path, base_fm, base_raw, old_content_base = _read_page(base_slug, resolved_wiki)
    variant_pages = _read_pages(variant_slugs, resolved_wiki)

    hash_base = _content_hash(base_raw)
    hash_variants = {slug: _content_hash(raw) for slug, (_p, _fm, raw, _c) in variant_pages.items()}
    variant_contents = {slug: content for slug, (_p, _fm, _raw, content) in variant_pages.items()}

    all_fms = [base_fm] + [fm for _p, fm, _raw, _c in variant_pages.values()]
    union_sections = _collect_union_sections_n(all_fms, resolved_docs)

    draft = generate_collision_merge_draft(
        base_slug, old_content_base, variant_contents, union_sections
    )

    outcome = verify(draft.content_base, union_sections)
    grounding = _combine_grounding_n([outcome])

    log_event(
        "collision_merge_generate",
        f"base_slug={base_slug} variants={','.join(variant_slugs)} "
        f"passed={grounding.passed} reason={grounding.reason}",
    )

    return CollisionMergeGenerateResponse(
        base_slug=base_slug,
        variant_slugs=variant_slugs,
        old_content_base=old_content_base,
        content_base=draft.content_base,
        grounding=grounding,
        hash_base=hash_base,
        hash_variants=hash_variants,
    )


@dataclass
class CollisionMergeApplyResult:
    """Outcome of a successful ``apply_collision_merge`` call. ``routes.py``
    wraps this into a ``CollisionMergeApplyResponse`` after calling
    ``build_index()`` exactly once (reindex stays a route-layer concern —
    matches ``apply_reconcile``)."""

    base_slug: str
    deleted_variants: list[str]
    grounding: GroundingInfo


def apply_collision_merge(
    req: CollisionMergeApplyRequest,
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> CollisionMergeApplyResult:
    """Re-verify and commit the final (possibly human-edited) merge content,
    then delete the reference-free variants.

    Check order (cheapest / most likely to short-circuit first, so a doomed
    call never reaches the LLM grounding round-trip):
    1. Group shape validation (``CollisionInvalidGroup``).
    2. Hash-based optimistic concurrency for the base + every variant
       (``CollisionHashMismatch`` — 409).
    3. Inbound-reference guard for every variant slated for deletion
       (``CollisionReferenceGuardFailed`` — ADR-0028 Invariant).
    4. Apply-time grounding re-check on the submitted base content
       (``CollisionGroundingFailed`` — 422).
    Only a full pass reaches the write.

    Raises:
        CollisionInvalidGroup: malformed group shape, or ``hash_variants``
            keys do not match ``variant_slugs`` exactly.
        PageNotFound / PageCorrupt: ``base_slug`` or any variant does not
            resolve / is corrupt.
        CollisionHashMismatch: the base or any variant's on-disk content
            changed since generate.
        CollisionReferenceGuardFailed: any variant slated for deletion has
            an inbound ``[[link]]`` or qa citation.
        CollisionGroundingFailed: the apply-time grounding re-check failed
            for the submitted base content.
    """
    _validate_group(req.base_slug, req.variant_slugs)
    if set(req.hash_variants) != set(req.variant_slugs):
        raise CollisionInvalidGroup(
            f"hash_variants keys {sorted(req.hash_variants)} must match "
            f"variant_slugs {sorted(req.variant_slugs)}"
        )

    resolved_wiki = _resolve_wiki_dir(wiki_dir)
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR

    base_path, base_fm, base_raw, _old_content_base = _read_page(req.base_slug, resolved_wiki)
    variant_pages = _read_pages(req.variant_slugs, resolved_wiki)

    if _content_hash(base_raw) != req.hash_base:
        raise CollisionHashMismatch(f"base_slug={req.base_slug}")
    for slug, (_path, _fm, raw, _content) in variant_pages.items():
        if _content_hash(raw) != req.hash_variants[slug]:
            raise CollisionHashMismatch(f"variant={slug}")

    referrers: list[InboundReference] = []
    for slug in req.variant_slugs:
        wiki_refs, qa_refs = find_inbound_references(slug, resolved_wiki)
        if wiki_refs or qa_refs:
            referrers.append(
                InboundReference(variant_slug=slug, wiki_referrers=wiki_refs, qa_referrers=qa_refs)
            )
    if referrers:
        log_event(
            "collision_merge_guard_refused",
            f"base_slug={req.base_slug} "
            f"referenced_variants={','.join(r.variant_slug for r in referrers)}",
        )
        raise CollisionReferenceGuardFailed(referrers)

    all_fms = [base_fm] + [fm for _p, fm, _raw, _c in variant_pages.values()]
    union_sections = _collect_union_sections_n(all_fms, resolved_docs)

    outcome = verify(req.content_base, union_sections)
    grounding = _combine_grounding_n([outcome])
    if not grounding.passed:
        log_event(
            "collision_merge_apply_refused",
            f"base_slug={req.base_slug} reason={grounding.reason}",
        )
        raise CollisionGroundingFailed(grounding)

    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Base rewrite + variant deletions under one lock acquisition (mirrors
    # apply_reconcile's multi-file write convention) so a concurrent
    # run_lint() never observes a half-applied merge.
    with _index_lock:
        _write_merged_base_page(base_path, base_fm, req.content_base, now_iso, req.variant_slugs)
        for _slug, (path, _fm, _raw, _content) in variant_pages.items():
            path.unlink()

    log_event(
        "collision_merge_applied",
        f"base_slug={req.base_slug} deleted_variants={','.join(req.variant_slugs)}",
    )

    return CollisionMergeApplyResult(
        base_slug=req.base_slug, deleted_variants=list(req.variant_slugs), grounding=grounding
    )


def generate_collision_differentiate(
    slugs: list[str],
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> CollisionDifferentiateGenerateResponse:
    """Draft complementary content for every page in a C4 collision group.
    Writes nothing to disk.

    Raises ``CollisionInvalidGroup`` when ``slugs`` has fewer than 2 members
    or duplicates. Raises ``PageNotFound``/``PageCorrupt`` when any slug does
    not resolve / is corrupt.
    """
    _validate_slugs(slugs)

    resolved_wiki = _resolve_wiki_dir(wiki_dir)
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR

    pages = _read_pages(slugs, resolved_wiki)
    old_content = {slug: content for slug, (_p, _fm, _raw, content) in pages.items()}
    hashes = {slug: _content_hash(raw) for slug, (_p, _fm, raw, _c) in pages.items()}
    all_fms = [fm for _p, fm, _raw, _c in pages.values()]
    union_sections = _collect_union_sections_n(all_fms, resolved_docs)

    draft = generate_collision_differentiate_draft(old_content, union_sections)
    draft_by_slug = {p.slug: p.content for p in draft.pages}
    # Best-effort: a slug missing from the LLM's structured output falls back
    # to its old content rather than crashing — the apply-time grounding
    # re-check still runs on whatever is ultimately submitted, so an
    # incomplete draft cannot silently write ungrounded content.
    content = {slug: draft_by_slug.get(slug, old_content[slug]) for slug in slugs}

    outcomes = [verify(content[slug], union_sections) for slug in slugs]
    grounding = _combine_grounding_n(outcomes)

    log_event(
        "collision_differentiate_generate",
        f"slugs={','.join(slugs)} passed={grounding.passed} reason={grounding.reason}",
    )

    return CollisionDifferentiateGenerateResponse(
        slugs=slugs,
        old_content=old_content,
        content=content,
        grounding=grounding,
        hashes=hashes,
    )


@dataclass
class CollisionDifferentiateApplyResult:
    """Outcome of a successful ``apply_collision_differentiate`` call.
    ``routes.py`` wraps this into a ``CollisionDifferentiateApplyResponse``
    after calling ``build_index()`` exactly once."""

    slugs: list[str]
    grounding: GroundingInfo


def apply_collision_differentiate(
    req: CollisionDifferentiateApplyRequest,
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> CollisionDifferentiateApplyResult:
    """Re-verify and commit the final (possibly human-edited) differentiate
    content for every group member. No deletion — nobody dies — so no
    reference guard is needed (ADR-0028: "Differentiate... nobody is
    deleted").

    Raises:
        CollisionInvalidGroup: malformed group shape, or ``content``/
            ``hashes`` keys do not match ``slugs`` exactly.
        PageNotFound / PageCorrupt: any slug does not resolve / is corrupt.
        CollisionHashMismatch: any page's on-disk content changed since
            generate.
        CollisionGroundingFailed: the apply-time grounding re-check failed
            for any page's submitted content.
    """
    _validate_slugs(req.slugs)
    if set(req.content) != set(req.slugs) or set(req.hashes) != set(req.slugs):
        raise CollisionInvalidGroup(
            f"content/hashes keys must match slugs exactly; got "
            f"content={sorted(req.content)} hashes={sorted(req.hashes)} slugs={sorted(req.slugs)}"
        )

    resolved_wiki = _resolve_wiki_dir(wiki_dir)
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR

    pages = _read_pages(req.slugs, resolved_wiki)
    for slug, (_path, _fm, raw, _content) in pages.items():
        if _content_hash(raw) != req.hashes[slug]:
            raise CollisionHashMismatch(f"slug={slug}")

    all_fms = [fm for _p, fm, _raw, _c in pages.values()]
    union_sections = _collect_union_sections_n(all_fms, resolved_docs)

    outcomes = [verify(req.content[slug], union_sections) for slug in req.slugs]
    grounding = _combine_grounding_n(outcomes)
    if not grounding.passed:
        log_event(
            "collision_differentiate_apply_refused",
            f"slugs={','.join(req.slugs)} reason={grounding.reason}",
        )
        raise CollisionGroundingFailed(grounding)

    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Every member writes under one lock acquisition (mirrors
    # apply_reconcile's multi-file write convention) so a concurrent
    # run_lint() never observes a partially-differentiated group.
    with _index_lock:
        for slug, (path, fm, _raw, _content) in pages.items():
            _write_differentiated_page(path, fm, req.content[slug], now_iso, req.slugs)

    log_event("collision_differentiate_applied", f"slugs={','.join(req.slugs)}")

    return CollisionDifferentiateApplyResult(slugs=list(req.slugs), grounding=grounding)
