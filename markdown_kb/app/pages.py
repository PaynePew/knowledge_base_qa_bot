"""Deep module per Ousterhout. Public surface: ``delete_full_orphan``, ``PageNotFound``, ``PageCorrupt``, ``PageNotFullOrphan``.

C11 Confirmed Remediation (tier-B S5, issue #381, ADR-0024/0025) — the first
delete of a corpus-resident wiki page. ``DELETE /pages/{slug}`` (routes.py)
wraps ``delete_full_orphan`` (CODING_STANDARD §2.3: all business logic lives
here, the route stays a shallow HTTP<->exception mapper).

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
"""

from __future__ import annotations

from pathlib import Path

from ._paths import DOCS_DIR
from .indexer import _index_lock
from .lint import check_full_orphan
from .logger import log_event
from .slugs import is_bare_slug
from .wiki_writer import read_existing_frontmatter

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
