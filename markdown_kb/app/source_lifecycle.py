"""Deep module per Ousterhout. Public surface: ``retire``, ``restore``, ``list_trash``,
``compute_impact``, ``RetireResult``, ``TrashEntry``, ``ImpactPreview``,
``InvalidRelpath``, ``SourceNotFound``, ``TrashEntryNotFound``,
``RestoreTargetOccupied``, ``RestoreBasenameCollision``.

Source lifecycle — governed retire/restore of Sources into a timestamped
Source Trash (ADR-0041, issue #604 — **S1 of 3**: this slice ships the
trash-move core: retire, restore, the shared validator, and the audit log.
S2 adds rename; S3 adds the Console/CLI/MCP surfaces.

**Retire** (Confirmed) is one atomic whole-file move ``docs/<relpath>`` ->
``<TRASH_DIR>/<UTC-timestamp>/docs/<relpath>`` (``atomic.replace_atomic`` —
``os.replace`` under the hood, so Source bytes are never read or rewritten,
only moved). ``TRASH_DIR`` sits OUTSIDE ``docs/`` (see ``_paths.TRASH_DIR``),
so every Source scanner (upload origin resolution, ingest pairing, lint
citation resolution) is structurally blind to a retired file — no exclusion
list to maintain, no re-ingest resurrection path. The confirmation dialog is
the server-computed impact preview (``compute_impact`` / ``GET
/sources/{relpath}/impact``): which derived wiki pages would become full
orphans (deletable via the existing C11 -> Confirmed delete flow,
``pages.delete_full_orphan``) vs. partial (some citations still resolve).
Retire deliberately does NOT cascade onto wiki pages (ADR-0041 decision 3):
the vacuum window between retire and the eventual C11 delete is
curator-held, exactly like every other Confirmed Remediation in this
codebase never auto-chaining into a second mutation (ADR-0024).

**Restore** (Direct) is the atomic inverse move, keyed by a trash entry's
``(timestamp, relpath)``. Two refusal guards share ADR-0036 §6's *refuse,
never fall back* posture — restore never overwrites and never auto-suffixes
a name:

- ``RestoreTargetOccupied`` — the original ``relpath`` is now occupied (a
  same-name Source was uploaded or restored since the retire).
- ``RestoreBasenameCollision`` — the basename already exists ELSEWHERE under
  ``docs/``; restoring would mint the ``ambiguous_source`` state ADR-0036 §6
  exists to prevent.

Restore does zero page bookkeeping (ADR-0041 decision 4): a still-live
derived page's orphan status clears on the next lint recompute (a stale C11
delete click re-verifies the full-orphan predicate at delete time and 409s —
``lint.check_full_orphan``, ADR-0025), and an already-deleted page
re-synthesizes on the next ingest because its ``source_hashes`` skip-state
died with it (ADR-0029 decision 4).

**Shared validator** (ADR-0041 decision 6): ``_is_safe_relpath`` /
``_is_safe_component`` reuse ``upload._is_safe_basename``'s character rules
(CVE-2021-42574 bidi controls, ASCII control chars, the Section.id ``'#'``
delimiter) applied per path segment — duplicated rather than imported.
Module-private helpers stay module-private absent a named ADR blessing cross-
module reuse (CODING_STANDARD §2.4); this mirrors ``upload.
_resolve_overwrite_target``'s own documented precedent for the identical
situation (that function duplicates ``lint._resolve_c3_source_path``'s
basename-glob rule rather than importing it). ``_would_be_full_orphan``
similarly duplicates ``lint._orphan_predicate``'s full-orphan test rather
than importing it — needed here (not the public ``lint.check_full_orphan``)
because the impact preview must simulate a Source's removal BEFORE it
actually happens; ``check_full_orphan`` always reads the real, current
``docs_dir``. Page frontmatter reads DO reuse the public
``wiki_writer.read_existing_frontmatter`` (handles the sentinel HTML comment
real ingest-produced pages carry before the ``---`` fence) — no duplication
needed there, since it is already the shared public reader ``pages.py``
itself uses.

**Audit**: one bounded ``log_event`` line per act (``source_retired`` /
``source_restored`` — act, relpath, timestamp, and impact counts for
retire), from this module's own channel (``wiki/log.md``, same as every
other ``markdown_kb`` module — CODING_STANDARD §5.1). No manifest file
(ADR-0041 decision 8): the timestamped trash tree IS the physical audit
trail; ``list_trash`` reads it directly, no separate authoritative state.

**No reindex** anywhere in this module: retire/restore never touch
``wiki/`` (ADR-0041 Invariant — corpus exit stays solely the existing C11
Confirmed delete), so there is nothing for a reindex to pick up.

**Concurrency**: every mutating sequence (impact-compute + move for retire;
occupancy/collision-check + move for restore) runs under
``indexer._index_lock`` — mirrors ``pages.delete_full_orphan`` /
``reconcile.py``'s apply-time convention, so a concurrent ``run_lint()``
sweep (which holds the same lock for its full read pass) never observes a
half-moved Source, and a concurrent retire/restore of the same relpath
never races.

**Endpoints live in the Gateway, not here** (mirrors ``upload.py`` /
``read.py``, both deep modules in ``markdown_kb/app/`` wired from
``gateway/app/routes.py`` — Upload/Console-adjacent system boundaries are a
Gateway concern per ADR-0010): ``GET /sources/{relpath}/impact``, ``POST
/sources/retire``, ``POST /sources/restore``, ``GET /sources/trash``. This
module has no dependency on FastAPI or HTTP status codes (CODING_STANDARD
§4.4 — validation at the route boundary, domain exceptions here).

See ADR-0041 and GitHub issue #604 (S1) for design rationale.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ._paths import DOCS_DIR, TRASH_DIR, WIKI_DIR
from .atomic import replace_atomic
from .indexer import _index_lock
from .logger import log_event
from .wiki_writer import read_existing_frontmatter

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InvalidRelpath(Exception):
    """A ``relpath`` (or trash ``timestamp``) fails the traversal/character
    safety check (route -> 422). Nothing is read or written."""


class SourceNotFound(Exception):
    """No file exists at ``docs_dir / relpath`` (route -> 404)."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        super().__init__(f"no Source found at docs/{relpath}")


class TrashEntryNotFound(Exception):
    """No trash entry exists at ``(timestamp, relpath)`` (route -> 404) — the
    entry was already restored, or never existed."""

    def __init__(self, timestamp: str, relpath: str) -> None:
        self.timestamp = timestamp
        self.relpath = relpath
        super().__init__(f"no trash entry at {timestamp}/docs/{relpath}")


class RestoreTargetOccupied(Exception):
    """``docs_dir / relpath`` already exists — restore refused, never
    overwrites (route -> 409, ADR-0041 decision 4)."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        super().__init__(f"docs/{relpath} is already occupied; restore refused")


class RestoreBasenameCollision(Exception):
    """The basename exists elsewhere under ``docs_dir`` — restoring would
    mint an ``ambiguous_source`` state; refused, never auto-suffixed (route
    -> 409, ADR-0041 decision 4 / ADR-0036 §6)."""

    def __init__(self, relpath: str, collisions: list[str]) -> None:
        self.relpath = relpath
        self.collisions = collisions
        super().__init__(f"basename of {relpath} collides with {collisions}; restore refused")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ImpactPreview:
    """Server-computed retire impact preview (ADR-0041 decision 2).

    ``full_orphans`` / ``partial_orphans`` are wiki page slugs — sorted,
    de-duplicated by construction (each page contributes at most once).
    """

    relpath: str
    full_orphans: list[str] = field(default_factory=list)
    partial_orphans: list[str] = field(default_factory=list)


@dataclass
class RetireResult:
    """Outcome of a successful ``retire()`` call — the response payload that
    routes the curator to the resulting C11 findings (ADR-0041 decision 2)."""

    relpath: str
    timestamp: str
    impact: ImpactPreview


@dataclass
class TrashEntry:
    """One entry in the Source Trash (CONTEXT.md "Source Trash")."""

    timestamp: str
    relpath: str


# ---------------------------------------------------------------------------
# Shared validator (ADR-0041 decision 6)
# ---------------------------------------------------------------------------

# Same rule set as upload._is_safe_basename (CVE-2021-42574 Trojan Source
# bidi controls + ASCII control chars + the Section.id "{filename}#{slug}"
# delimiter) — duplicated per module-private-stays-private, see module
# docstring.
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")
_CONTROL_RE = re.compile(r"[\x00-\x1f]")


def _is_safe_component(name: str) -> tuple[bool, str]:
    """Validate a single path component — a relpath segment, or a trash
    ``timestamp`` — is traversal- and injection-resistant.

    Returns ``(True, '')`` if safe, ``(False, reason)`` otherwise.
    """
    if not name or not name.strip():
        return False, "must not be empty"
    if "/" in name or "\\" in name:
        return False, f"must not contain a path separator: {name!r}"
    if name in (".", ".."):
        return False, f"must not be '.' or '..': {name!r}"
    if _CONTROL_RE.search(name):
        return False, f"contains a control character: {name!r}"
    for bidi in _BIDI_CONTROLS:
        if bidi in name:
            return False, f"contains bidi control character U+{ord(bidi):04X}: {name!r}"
    if "#" in name:
        return False, f"must not contain '#': {name!r}"
    return True, ""


def _is_safe_relpath(relpath: str) -> tuple[bool, str]:
    """Validate a Source ``relpath`` (forward-slash separated, possibly
    nested under a subdirectory of ``docs/``) is traversal- and
    injection-resistant. Applies ``_is_safe_component`` to every segment.

    Returns ``(True, '')`` if safe, ``(False, reason)`` otherwise.
    """
    if not relpath or not relpath.strip():
        return False, "relpath must not be empty."
    if "\\" in relpath:
        return False, f"relpath must use forward slashes: {relpath!r}"
    p = PurePosixPath(relpath)
    if p.is_absolute():
        return False, f"relpath must not be an absolute path: {relpath!r}"
    if not p.parts:
        return False, "relpath must not be empty."
    for part in p.parts:
        ok, reason = _is_safe_component(part)
        if not ok:
            return False, f"relpath {reason} (in {relpath!r})"
    return True, ""


# ---------------------------------------------------------------------------
# Wiki page iteration + orphan-predicate simulation (impact preview only)
# ---------------------------------------------------------------------------


def _iter_lifecycle_wiki_pages(wiki_dir: Path):
    """Yield ``(slug, page_path)`` for every entities/concepts page.

    Mirrors ``lint._iter_wiki_pages``'s glob shape — duplicated small
    filesystem walk (§2.4; no public iterator exists to reuse, and this is
    six lines of ``glob``, not logic worth centralising on its own).
    """
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            yield page_path.stem, page_path


def _page_sources(page_path: Path) -> list[str]:
    """Read a wiki page's frontmatter ``sources`` list.

    Reuses the PUBLIC ``wiki_writer.read_existing_frontmatter`` (the same
    reader ``pages.py`` uses) — correctly handles the sentinel HTML comment
    real ingest-produced pages carry before the ``---`` fence.
    """
    fm = read_existing_frontmatter(page_path)
    if fm is None:
        return []
    sources = fm.get("sources", []) or []
    if not isinstance(sources, list):
        return []
    return [str(s) for s in sources if s]


def _cites_basename(sources: list[str], basename: str) -> bool:
    """True when any of ``sources`` cites a file whose basename is ``basename``."""
    for citation in sources:
        file_part = citation.split("#")[0].strip()
        if file_part and Path(file_part).name == basename:
            return True
    return False


def _would_be_full_orphan(sources: list[str], docs_filenames_after: set[str]) -> bool:
    """Mirror ``lint._orphan_predicate``'s full-orphan test (ADR-0025)
    against a caller-supplied post-removal docs/ filename snapshot.

    Duplicated rather than imported (§2.4): the impact preview must simulate
    a Source's removal BEFORE it actually happens, and ``lint.
    check_full_orphan`` always reads the real, current ``docs_dir`` — it has
    no way to accept a hypothetical snapshot.
    """
    valid_citations = 0
    missing = 0
    for citation in sources:
        file_part = citation.split("#")[0].strip()
        if not file_part:
            continue
        valid_citations += 1
        if Path(file_part).name not in docs_filenames_after:
            missing += 1
    return valid_citations > 0 and missing == valid_citations


# ---------------------------------------------------------------------------
# Public API — impact preview
# ---------------------------------------------------------------------------


def _compute_impact_unlocked(relpath: str, docs_dir: Path, wiki_dir: Path) -> ImpactPreview:
    """Core impact computation, called under an already-held ``_index_lock``
    (by ``retire``) or by ``compute_impact`` (which acquires the lock itself).
    """
    ok, reason = _is_safe_relpath(relpath)
    if not ok:
        raise InvalidRelpath(reason)

    target = docs_dir / relpath
    if not target.is_file():
        raise SourceNotFound(relpath)

    basename = PurePosixPath(relpath).name
    target_resolved = target.resolve()
    # Simulate the post-retire docs/ state: every *.md basename under
    # docs_dir except THIS exact file. A basename shared with another file
    # elsewhere under docs/ (an anomalous, pre-existing state — normally
    # excluded by ingest's own ambiguous-origin refusal) correctly keeps the
    # basename "present" via the survivor.
    docs_filenames_after = {
        p.name for p in docs_dir.glob("**/*.md") if p.resolve() != target_resolved
    }

    full_orphans: list[str] = []
    partial_orphans: list[str] = []
    for slug, page_path in _iter_lifecycle_wiki_pages(wiki_dir):
        sources = _page_sources(page_path)
        if not _cites_basename(sources, basename):
            continue
        if _would_be_full_orphan(sources, docs_filenames_after):
            full_orphans.append(slug)
        else:
            partial_orphans.append(slug)

    return ImpactPreview(
        relpath=relpath,
        full_orphans=sorted(full_orphans),
        partial_orphans=sorted(partial_orphans),
    )


def compute_impact(
    relpath: str,
    *,
    docs_dir: Path | None = None,
    wiki_dir: Path | None = None,
) -> ImpactPreview:
    """Server-computed impact preview for retiring ``docs/<relpath>``
    (ADR-0041 decision 2) — the confirmation dialog's data source.

    Read-only: does not move or write anything.

    Args:
        relpath:  Path of the Source relative to ``docs_dir`` (e.g.
                  ``'policy.md'`` or ``'demo-zh/policy.md'``), forward
                  slashes, no leading ``'docs/'``.
        docs_dir: Override ``DOCS_DIR`` (tests only).
        wiki_dir: Override ``WIKI_DIR`` (tests only).

    Returns:
        ``ImpactPreview`` with the page slugs that would become full orphans
        (deletable via the existing C11 flow) vs. partial (some citations
        still resolve).

    Raises:
        InvalidRelpath: ``relpath`` fails the traversal/character safety check.
        SourceNotFound: no file exists at ``docs_dir / relpath``.
    """
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR
    resolved_wiki = wiki_dir if wiki_dir is not None else WIKI_DIR
    with _index_lock:
        return _compute_impact_unlocked(relpath, resolved_docs, resolved_wiki)


# ---------------------------------------------------------------------------
# Public API — retire
# ---------------------------------------------------------------------------


def _trash_timestamp() -> str:
    """Filesystem-safe UTC timestamp for a trash act-folder name.

    Mirrors ``logger.log_event``'s ISO-8601 UTC timestamp precision
    (microsecond) but strips ``':'`` — illegal in a Windows directory name —
    so the same act-folder name is both a sortable audit key and a real path
    component on every supported OS. Microsecond resolution makes repeated
    retires of the same relpath collision-free in practice (ADR-0041
    decision 2).
    """
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def retire(
    relpath: str,
    *,
    docs_dir: Path | None = None,
    wiki_dir: Path | None = None,
    trash_dir: Path | None = None,
) -> RetireResult:
    """Retire the Source at ``docs_dir / relpath`` (ADR-0041 decision 2).

    One atomic whole-file move to ``trash_dir/<UTC-timestamp>/docs/<relpath>``
    (``atomic.replace_atomic`` — ``os.replace`` under the hood). Source bytes
    are never read or rewritten, only moved.

    No cascade onto wiki pages (ADR-0041 decision 3) and no reindex (retire
    never touches ``wiki/``) — derived pages surface as C11 orphan findings
    on the next lint; corpus exit stays the existing C11 Confirmed delete.

    Args:
        relpath:  Path of the Source relative to ``docs_dir``, e.g.
                  ``'policy.md'`` or ``'demo-zh/policy.md'``.
        docs_dir: Override ``DOCS_DIR`` (tests only).
        wiki_dir: Override ``WIKI_DIR`` (tests only).
        trash_dir: Override ``TRASH_DIR`` (tests only).

    Returns:
        ``RetireResult`` carrying the trash ``timestamp`` and the impact
        preview computed just before the move.

    Raises:
        InvalidRelpath: ``relpath`` fails the traversal/character safety check.
        SourceNotFound: no file exists at ``docs_dir / relpath``.
    """
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR
    resolved_wiki = wiki_dir if wiki_dir is not None else WIKI_DIR
    resolved_trash = trash_dir if trash_dir is not None else TRASH_DIR

    with _index_lock:
        impact = _compute_impact_unlocked(relpath, resolved_docs, resolved_wiki)

        source_path = resolved_docs / relpath
        timestamp = _trash_timestamp()
        target = resolved_trash / timestamp / "docs" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        replace_atomic(source_path, target)

        log_event(
            "source_retired",
            f"relpath={relpath} timestamp={timestamp} "
            f"full_orphans={len(impact.full_orphans)} partial_orphans={len(impact.partial_orphans)}",
        )

    return RetireResult(relpath=relpath, timestamp=timestamp, impact=impact)


# ---------------------------------------------------------------------------
# Public API — restore
# ---------------------------------------------------------------------------


def restore(
    timestamp: str,
    relpath: str,
    *,
    docs_dir: Path | None = None,
    trash_dir: Path | None = None,
) -> None:
    """Restore a trash entry keyed by ``(timestamp, relpath)`` (ADR-0041
    decision 4) — the atomic inverse of ``retire``.

    Two refusal guards, sharing ADR-0036 §6's *refuse, never fall back*
    posture — restore never overwrites and never auto-suffixes a name.

    No page bookkeeping, no reindex: a still-live derived page's orphan
    status clears on the next lint recompute; an already-deleted page
    re-synthesizes on the next ingest.

    Args:
        timestamp: The trash act-folder name (from ``list_trash`` /
                   ``RetireResult.timestamp``).
        relpath:   The original Source relpath, e.g. ``'policy.md'``.
        docs_dir:  Override ``DOCS_DIR`` (tests only).
        trash_dir: Override ``TRASH_DIR`` (tests only).

    Raises:
        InvalidRelpath: ``relpath`` or ``timestamp`` fails the safety check.
        TrashEntryNotFound: no trash entry exists at ``(timestamp, relpath)``.
        RestoreTargetOccupied: ``docs_dir / relpath`` already exists.
        RestoreBasenameCollision: the basename exists elsewhere under
            ``docs_dir`` — restoring would mint an ``ambiguous_source`` state.
    """
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR
    resolved_trash = trash_dir if trash_dir is not None else TRASH_DIR

    ok, reason = _is_safe_relpath(relpath)
    if not ok:
        raise InvalidRelpath(reason)
    ok_ts, ts_reason = _is_safe_component(timestamp)
    if not ok_ts:
        raise InvalidRelpath(f"timestamp {ts_reason}")

    trash_path = resolved_trash / timestamp / "docs" / relpath

    with _index_lock:
        if not trash_path.is_file():
            raise TrashEntryNotFound(timestamp, relpath)

        target = resolved_docs / relpath
        if target.exists():
            raise RestoreTargetOccupied(relpath)

        basename = PurePosixPath(relpath).name
        collisions = sorted(
            p.relative_to(resolved_docs).as_posix() for p in resolved_docs.glob(f"**/{basename}")
        )
        if collisions:
            raise RestoreBasenameCollision(relpath, collisions)

        target.parent.mkdir(parents=True, exist_ok=True)
        replace_atomic(trash_path, target)

        log_event("source_restored", f"relpath={relpath} timestamp={timestamp}")


# ---------------------------------------------------------------------------
# Public API — trash listing
# ---------------------------------------------------------------------------


def list_trash(*, trash_dir: Path | None = None) -> list[TrashEntry]:
    """List every entry in the Source Trash (ADR-0041 decision 8 — the trash
    tree itself IS the audit trail; no separate manifest).

    Read-only, no lock: mirrors ``pages.get_resolution_map``'s reasoning — a
    moment-stale read under a concurrent retire/restore is acceptable (each
    entry's existence is an independent filesystem fact), never a torn one.

    Args:
        trash_dir: Override ``TRASH_DIR`` (tests only).

    Returns:
        ``list[TrashEntry]`` sorted by timestamp then relpath. Empty when
        ``trash_dir`` does not exist yet (no retire has ever run).
    """
    resolved_trash = trash_dir if trash_dir is not None else TRASH_DIR
    if not resolved_trash.is_dir():
        return []

    entries: list[TrashEntry] = []
    for ts_dir in sorted(p for p in resolved_trash.iterdir() if p.is_dir()):
        docs_sub = ts_dir / "docs"
        if not docs_sub.is_dir():
            continue
        for file_path in sorted(p for p in docs_sub.glob("**/*") if p.is_file()):
            relpath = file_path.relative_to(docs_sub).as_posix()
            entries.append(TrashEntry(timestamp=ts_dir.name, relpath=relpath))
    return entries
