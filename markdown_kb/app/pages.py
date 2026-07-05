"""Deep module per Ousterhout. Public surface: ``delete_full_orphan``, ``add_alias``, ``remove_alias``, ``get_resolution_map``, ``PageNotFound``, ``PageCorrupt``, ``PageNotFullOrphan``, ``InvalidAlias``, ``AliasCollision``, ``AliasNotFound``.

C11 Confirmed Remediation (tier-B S5, issue #381, ADR-0024/0025) — the first
delete of a corpus-resident wiki page. ``DELETE /pages/{slug}`` (routes.py)
wraps ``delete_full_orphan`` (CODING_STANDARD §2.3: all business logic lives
here, the route stays a shallow HTTP<->exception mapper).

Issue #409 (ADR-0030 decision 3) adds ``add_alias`` — the Direct-class,
human-surfaces-only assign-alias operation. ``POST /pages/{slug}/aliases``
(routes.py) wraps it the same way. No LLM, never batches (ADR-0030
Invariant), and its collision check reuses the SAME shared resolver
(``slugs.build_alias_resolution_map``) C2 and C12 already consult (ADR-0030
Invariant: "every consumer of wikilink resolution uses the shared
resolver").

Issue #491 (ADR-0030 extension) adds ``remove_alias`` — the mirror-image
operation: the executable fix the C12 alias-collision Remediation names.
``DELETE /pages/{slug}/aliases/{alias}`` (routes.py) wraps it the same way.
Direct-class, no LLM, never batches — one call clears one page's claim on
one alias. Unlike ``add_alias`` there is no collision check (nothing to
collide with when removing an entry); a request naming an alias the page
never declared is refused (``AliasNotFound``, 404) rather than reported as a
fake-success no-op.

Predicate (ADR-0025, the only condition under which a live page may be
deleted): the page is a **full orphan** — its ``sources`` frontmatter is
non-empty and every citation's file is missing under ``docs/**``. This is
recomputed server-side at delete time via ``lint.check_full_orphan`` — the
SAME predicate C11's bulk sweep uses (``lint._orphan_predicate``), so the two
can never disagree. The client's lint finding is never trusted as-is: a
Source may have been restored or re-imported since the report rendered.

Not a general page delete: only ``entities/`` and ``concepts/`` pages
qualify, and only when the full-orphan predicate holds; everything else is
refused (``PageNotFound`` / ``PageNotFullOrphan``). ADR-0012's qa-scoped
"delete inert only, refuse live" rule is left untouched — this is a
different page family (entities/concepts vs qa) with a different lifecycle
(corpus residency vs draft/live status).

Governance: Confirmed Remediation (ADR-0024) — a human confirms the named
irreversible operation; no LLM is involved anywhere in this module; this
module exposes no batch entry point (Confirmed never batches, ADR-0024
Invariant).

Concurrency: the read-predicate-check-delete sequence runs under
``indexer._index_lock`` (mirrors ``reconcile.py``'s apply-time convention),
so a concurrent ``run_lint()`` sweep — which holds the same lock for its
full read pass — never observes a half-deleted page, and the predicate is
re-verified against a state a concurrent ``/ingest`` cannot be mutating.

Issue #410 (ADR-0030 decision 5) adds ``get_resolution_map`` — the read-only
composition behind ``GET /pages/resolution-map``, consumed by every linkify
client (Console viewer, reader chat, chat-side citation viewer). No lock: a
pure read of the SAME shared resolver (``slugs.build_alias_resolution_map``)
every other consumer uses, plus a page-location lookup
(``slugs.build_slug_paths``) so a client never constructs a wiki path from a
bare slug itself.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ._paths import DOCS_DIR
from .atomic import write_text_atomic
from .indexer import _index_lock
from .lint import check_full_orphan
from .logger import log_event
from .schemas import ResolutionMapResponse
from .slugs import build_alias_resolution_map, build_slug_paths, is_bare_slug
from .wiki_writer import read_existing_frontmatter, read_page_parts

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PageNotFound(Exception):
    """Raised when ``slug`` does not resolve to an existing ``wiki/entities/``
    or ``wiki/concepts/`` page (route -> 404)."""


class PageCorrupt(Exception):
    """Raised when the page exists but its frontmatter cannot be parsed
    (route -> 500 — orphan-visibility: surface broken state rather than
    silently acting on it, mirrors ``qa.QaPageCorrupt`` / ``reconcile.
    PageCorrupt``)."""


class PageNotFullOrphan(Exception):
    """Raised when the full-orphan predicate does not hold at delete time
    (route -> 409 Conflict) — a stale lint report, a restored/re-imported
    Source, a partial orphan (some citations still resolve), or a page with
    an empty ``sources`` list (ADR-0025: "sources non-empty and every
    citation missing")."""


class InvalidAlias(Exception):
    """Raised when the requested alias is blank/whitespace-only (route ->
    422) — issue #409. A blank alias would be silently dropped by
    ``slugs.build_alias_resolution_map`` (it skips blank entries), so
    accepting one here would report success for an assignment that resolves
    nothing."""


class AliasCollision(Exception):
    """Raised when ``alias`` already resolves (via the shared resolver) to a
    DIFFERENT page than the one that just requested it — either a real page
    slug, or another page's own alias (route -> 409, issue #409, ADR-0030
    decision 3: "409 with the conflicting owner named — consistent with C12
    semantics"). Nothing is written."""

    def __init__(self, alias: str, owner: str) -> None:
        self.alias = alias
        self.owner = owner
        super().__init__(f"alias {alias!r} already resolves to page {owner!r}")


class AliasNotFound(Exception):
    """Raised when ``alias`` is not currently in ``slug``'s frontmatter
    ``aliases`` list (route -> 404, issue #491). Removing an alias a page
    never declared is refused honestly rather than reported as a
    fake-success no-op — e.g. a stale Console row where the alias was
    already removed by a concurrent request."""

    def __init__(self, alias: str, slug: str) -> None:
        self.alias = alias
        self.slug = slug
        super().__init__(f"alias {alias!r} is not assigned to page {slug!r}")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_wiki_dir(wiki_dir: Path | None) -> Path:
    """Resolve the wiki root, importing ``indexer`` at call time so a
    test's ``monkeypatch.setattr(indexer, "WIKI_DIR", ...)`` is honoured
    (mirrors ``reconcile._resolve_wiki_dir``)."""
    if wiki_dir is not None:
        return wiki_dir
    from . import indexer

    return indexer.WIKI_DIR


def _find_page_path(slug: str, wiki_dir: Path) -> Path:
    """Return the on-disk path for ``slug`` under ``entities/`` or
    ``concepts/`` — C11 scans only these two subdirs, so a delete target is
    always an entity or concept page. Slugs are corpus-unique
    (``wiki_writer.resolve_slug_collision``), so the server resolves the
    subdir itself (ADR-0025: "Slug resolved server-side").

    Rejects a path-shaped ``slug`` (separators, ``..``, a Windows drive
    prefix, NUL) up front via ``slugs.is_bare_slug``, before either
    candidate join is even built — issue #397: a FastAPI ``{slug}`` path
    segment cannot contain ``/`` but CAN contain ``\\`` / ``:``, which join
    OUTSIDE ``wiki_dir`` on Windows. Raises the same ``PageNotFound`` a
    missing-but-well-shaped slug would, so the route's existing 404 mapping
    covers this with no route-layer change.
    """
    if not is_bare_slug(slug):
        raise PageNotFound(slug)
    for subdir_name in ("entities", "concepts"):
        candidate = wiki_dir / subdir_name / f"{slug}.md"
        if candidate.exists():
            return candidate
    raise PageNotFound(slug)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delete_full_orphan(
    slug: str,
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> None:
    """Delete ``slug`` only if the ADR-0025 full-orphan predicate holds NOW.

    Raises:
        PageNotFound: ``slug`` does not resolve to an entities/concepts page,
            OR ``slug`` is not a bare filename component (issue #397 — see
            ``_find_page_path``; rejected before any filesystem access).
        PageCorrupt: the page exists but its frontmatter cannot be parsed.
        PageNotFullOrphan: the predicate does not hold — refused (409).

    No reindex here: mirrors ``reconcile.apply_reconcile`` — the caller
    (``routes.py``) triggers exactly one ``build_index()`` after this
    returns successfully (reindex is a route-layer concern, not a
    domain-layer one).
    """
    resolved_wiki = _resolve_wiki_dir(wiki_dir)
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR

    with _index_lock:
        path = _find_page_path(slug, resolved_wiki)

        fm = read_existing_frontmatter(path)
        if fm is None:
            raise PageCorrupt(slug)

        sources = fm.get("sources", []) or []
        if not isinstance(sources, list):
            sources = []

        if not check_full_orphan(sources, resolved_docs):
            raise PageNotFullOrphan(slug)

        path.unlink()
        log_event("orphan_page_deleted", f"slug={slug}")


# ---------------------------------------------------------------------------
# Assign-alias (issue #409, ADR-0030 decision 3)
# ---------------------------------------------------------------------------


def add_alias(
    slug: str,
    alias: str,
    *,
    wiki_dir: Path | None = None,
) -> None:
    """Assign ``alias`` to the entities/concepts page ``slug``.

    Direct-class curator mutation (ADR-0030 decision 3): no LLM involved,
    never batches (ADR-0030 Invariant), reversible via a frontmatter
    hand-edit. Surgical frontmatter-only rewrite — ``wiki_writer.
    read_page_parts`` splits the page into prefix/frontmatter/suffix so the
    sentinel comment, heading, body, and citation line are preserved
    byte-for-byte; only the parsed frontmatter dict's ``aliases`` list is
    mutated and the whole dict is re-serialised (mirrors ``qa.
    _flip_draft_to_live``'s "rebuild the whole frontmatter, never
    surgically string-splice" convention, applied here to the raw parsed
    dict rather than a typed model — reconstructing a ``WikiPageFrontmatter``
    would force restating every optional field this endpoint never touches).

    The collision check reuses the SAME shared resolver C2 and C12 already
    consult (``slugs.build_alias_resolution_map``, ADR-0030 Invariant:
    "every consumer of wikilink resolution uses the shared resolver"),
    computed BEFORE the write so it reflects the corpus's current state,
    not the file about to change.

    Idempotent: re-assigning an alias the page already owns is a no-op
    (success, no rewrite) — it is not "another page's alias", so it is not
    a collision.

    Args:
        slug:     Target entities/concepts page slug.
        alias:    The alias to assign.
        wiki_dir: Root wiki directory. Defaults to ``indexer.WIKI_DIR``.

    Raises:
        PageNotFound: ``slug`` does not resolve to an entities/concepts
            page, OR ``slug`` is not a bare filename component (issue #397
            — see ``_find_page_path``; rejected before any filesystem
            access).
        PageCorrupt: the page exists but its frontmatter cannot be parsed.
        InvalidAlias: ``alias`` is blank / whitespace-only.
        AliasCollision: ``alias`` already resolves (via the shared
            resolver) to a DIFFERENT page — a real page slug or another
            page's alias. Nothing is written.
    """
    alias = alias.strip()
    if not alias:
        raise InvalidAlias("alias must not be blank")

    resolved_wiki = _resolve_wiki_dir(wiki_dir)

    with _index_lock:
        path = _find_page_path(slug, resolved_wiki)

        parts = read_page_parts(path)
        if parts is None:
            raise PageCorrupt(slug)
        prefix, fm, suffix = parts

        existing_aliases = fm.get("aliases") or []
        if not isinstance(existing_aliases, list):
            existing_aliases = []
        if alias in existing_aliases:
            return  # already assigned to this page — idempotent no-op

        resolution = build_alias_resolution_map(resolved_wiki)
        owner = resolution.get(alias)
        if owner is not None:
            raise AliasCollision(alias, owner)

        fm["aliases"] = [*existing_aliases, alias]
        fm_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True)
        if not fm_text.endswith("\n"):
            fm_text += "\n"
        write_text_atomic(path, prefix + fm_text + suffix)
        log_event("alias_assigned", f"slug={slug} alias={alias}")


# ---------------------------------------------------------------------------
# Remove-alias (issue #491, ADR-0030 extension)
# ---------------------------------------------------------------------------


def remove_alias(
    slug: str,
    alias: str,
    *,
    wiki_dir: Path | None = None,
) -> None:
    """Remove ``alias`` from the entities/concepts page ``slug``.

    Direct-class curator mutation (issue #491) — the mirror-image of
    ``add_alias``: no LLM involved, never batches, reversible via a
    frontmatter hand-edit. This is the executable fix the C12
    alias-collision Remediation names (``_REMEDIATION_TAXONOMY["C12"]`` in
    ``lint.py``): the add-only assign-alias endpoint can never resolve a
    collision (re-assigning to the same page no-ops, assigning to any other
    page 409s), so the only real fix is removing the alias from whichever
    page(s) should not keep it.

    Same surgical frontmatter-only rewrite as ``add_alias`` —
    ``wiki_writer.read_page_parts`` splits the page into
    prefix/frontmatter/suffix so the sentinel comment, heading, body, and
    citation line are preserved byte-for-byte; only the parsed frontmatter
    dict's ``aliases`` list is mutated and the whole dict is re-serialised.

    Unlike ``add_alias``, this performs no collision check — there is
    nothing to collide with when removing an entry.

    Args:
        slug:     Target entities/concepts page slug.
        alias:    The alias to remove.
        wiki_dir: Root wiki directory. Defaults to ``indexer.WIKI_DIR``.

    Raises:
        PageNotFound: ``slug`` does not resolve to an entities/concepts
            page, OR ``slug`` is not a bare filename component (issue #397
            — see ``_find_page_path``; rejected before any filesystem
            access).
        PageCorrupt: the page exists but its frontmatter cannot be parsed.
        AliasNotFound: ``alias`` is not currently in this page's ``aliases``
            list — refused (404) rather than a fake-success no-op.
    """
    alias = alias.strip()
    resolved_wiki = _resolve_wiki_dir(wiki_dir)

    with _index_lock:
        path = _find_page_path(slug, resolved_wiki)

        parts = read_page_parts(path)
        if parts is None:
            raise PageCorrupt(slug)
        prefix, fm, suffix = parts

        existing_aliases = fm.get("aliases") or []
        if not isinstance(existing_aliases, list):
            existing_aliases = []
        if alias not in existing_aliases:
            raise AliasNotFound(alias, slug)

        fm["aliases"] = [a for a in existing_aliases if a != alias]
        fm_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True)
        if not fm_text.endswith("\n"):
            fm_text += "\n"
        write_text_atomic(path, prefix + fm_text + suffix)
        log_event("alias_removed", f"slug={slug} alias={alias}")


# ---------------------------------------------------------------------------
# Resolution map (issue #410, ADR-0030 decision 5)
# ---------------------------------------------------------------------------


def get_resolution_map(*, wiki_dir: Path | None = None) -> ResolutionMapResponse:
    """Build the ADR-0030 linkify resolution-map view for ``GET
    /pages/resolution-map``.

    Read-only, no lock: this is presentation data derived from two pure
    reads — ``slugs.build_alias_resolution_map`` (the SAME shared resolver
    C2 and ``add_alias`` already consult; ADR-0030 Invariant: one resolver,
    every consumer) and ``slugs.build_slug_paths`` (a page-location lookup,
    not a resolution decision). A client sees at worst a moment-stale map
    under a concurrent write, never a torn one — each underlying page read
    is independent.

    Args:
        wiki_dir: Root wiki directory. Defaults to ``indexer.WIKI_DIR``.

    Returns:
        ``ResolutionMapResponse`` with ``slugs`` (real slug -> wiki-relative
        path) and ``aliases`` (alias -> canonical slug, a key into ``slugs``).
    """
    resolved_wiki = _resolve_wiki_dir(wiki_dir)

    resolution = build_alias_resolution_map(resolved_wiki)
    slug_paths = build_slug_paths(resolved_wiki)
    # Every non-self entry in the shared resolver's flat map IS an alias by
    # construction (a real slug always maps to itself — see
    # build_alias_resolution_map's docstring) — this partitions its output,
    # it does not make an independent resolvability judgement.
    aliases = {alias: canonical for alias, canonical in resolution.items() if alias != canonical}

    return ResolutionMapResponse(slugs=slug_paths, aliases=aliases)
