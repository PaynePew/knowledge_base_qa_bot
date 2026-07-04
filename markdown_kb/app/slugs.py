"""Shared slug-safety and link-resolution helpers (CODING_STANDARD §2.4).

Public surface: ``is_bare_slug``, ``build_alias_resolution_map``,
``build_slug_paths``.

This is the canonical home for the path-shape guard originally written as
``qa._is_bare_slug`` for ``qa.promote_batch``'s body-supplied slugs (issue
#382). Promoted to its own module once ``pages.py`` needed the same
predicate for ``delete_full_orphan`` (issue #397) — CODING_STANDARD §2.4's
escalation rule: "the moment a second package needs the same `_private`
symbol, promote it to the owner's public API instead of importing it
privately again." Mirrors ``atomic.py``'s pattern for a small, non-domain
technical helper shared across modules (no Ousterhout depth label — this
module wires no larger subsystem together, it just centralises one
predicate two callers need identically).

Issue #406 (ADR-0030) adds ``build_alias_resolution_map`` — the single
shared link-layer resolver every wikilink-resolution consumer (``lint.py``'s
C2 red-link check and ``find_inbound_references``; a future linkify surface)
must use, per ADR-0030's Invariant "every consumer of wikilink resolution
uses the shared resolver; no surface builds its own slug set." Its home is
this module (not ``lint.py``) because the resolver is not itself a lint
check — it is a link-layer primitive several unrelated call sites share,
matching this module's existing "small, non-domain technical helper" role.
"""

from __future__ import annotations

from pathlib import Path

from .wiki_writer import read_existing_frontmatter

# Wiki subdirectories that hold alias-eligible (entity/concept) pages.
# ``wiki/qa/`` Filed Answers are deliberately excluded — ADR-0030's "target
# page" is always an entity/concept synthesis page, never a Filed Answer.
# TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS
# string-name companion — see the same TODO already left in lint.py.
_ALIAS_ELIGIBLE_SUBDIRS: tuple[str, ...] = ("entities", "concepts")


def is_bare_slug(slug: str) -> bool:
    """True iff ``slug`` is a bare single filename component, so that
    ``some_dir / f"{slug}.md"`` cannot resolve outside ``some_dir``.

    A FastAPI path segment for ``{slug}`` cannot contain ``/`` — that
    property is free — but it CAN contain ``\\`` or ``:``, which the route
    matcher never rejects yet which act as path separators once joined on
    Windows (``%5C`` decodes to a backslash; ``D:x`` is a drive-relative
    path). Callers of this guard — ``qa.py``'s single-item mutators
    (``promote`` / ``delete`` / ``edit`` / ``refile``) and its body-supplied
    ``promote_batch`` list, plus ``pages.delete_full_orphan`` — treat this
    as a path-shape guard, never a charset allowlist: real corpus slugs
    include CJK (``compute_slug`` preserves Unicode verbatim) and stay
    valid.
    """
    if not slug or slug in {".", ".."}:
        return False
    return not any(ch in slug for ch in ("/", "\\", ":", "\x00"))


def build_alias_resolution_map(wiki_dir: Path) -> dict[str, str]:
    """Build the ADR-0030 link-layer resolution map: existing slugs UNION
    aliases -> canonical slug.

    Scans every ``entities/`` and ``concepts/`` page's frontmatter. Every
    real page slug maps to itself; every non-blank ``aliases:`` entry maps
    to that page's slug, EXCEPT when the resolution is ambiguous:

    - **A real page slug always wins over a colliding alias.** If some page
      declares an alias equal to another page's actual slug, the alias is
      simply dropped — the real page's self-mapping is authoritative and is
      never overridden.
    - **An alias-vs-alias tie** (two or more pages independently claim the
      SAME alias, and no real page owns that slug) **breaks
      lexicographically by canonical slug** — the lowest-sorted slug wins,
      so resolution is deterministic regardless of filesystem iteration
      order (ADR-0030 decision 4).

    This is the ONE shared resolver every wikilink-resolution consumer must
    call (ADR-0030 Invariant) — ``lint._check_c2_red_links`` (a link is red
    iff unresolvable here) and ``lint.find_inbound_references`` (an
    alias-mediated ``[[link]]`` is a real inbound reference) both use it
    instead of building their own slug set.

    Args:
        wiki_dir: Root wiki directory (``entities/`` and ``concepts/`` are
            read from beneath it; missing subdirectories are skipped).

    Returns:
        Dict mapping every resolvable key (real slug or alias) to its
        canonical (real) target slug.
    """
    real_slugs: set[str] = set()
    for subdir_name in _ALIAS_ELIGIBLE_SUBDIRS:
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in subdir.glob("*.md"):
            real_slugs.add(page_path.stem)

    # alias -> best canonical slug claimed so far (lexicographically lowest).
    alias_claims: dict[str, str] = {}
    for subdir_name in _ALIAS_ELIGIBLE_SUBDIRS:
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            canonical = page_path.stem
            fm = read_existing_frontmatter(page_path)
            if fm is None:
                continue
            raw_aliases = fm.get("aliases", [])
            if not isinstance(raw_aliases, list):
                continue
            for raw_alias in raw_aliases:
                alias = str(raw_alias).strip()
                if not alias or alias in real_slugs:
                    # Blank entry, or a real page slug always wins (decision 4).
                    continue
                existing = alias_claims.get(alias)
                if existing is None or canonical < existing:
                    alias_claims[alias] = canonical

    resolution: dict[str, str] = {slug: slug for slug in real_slugs}
    for alias, canonical in alias_claims.items():
        # `alias not in real_slugs` already guaranteed above, so this never
        # overwrites a real page's self-mapping.
        resolution[alias] = canonical
    return resolution


def build_slug_paths(wiki_dir: Path) -> dict[str, str]:
    """Map every real entities/concepts page slug to its wiki-relative path.

    A LOCATION lookup, not a resolution decision — ``build_alias_
    resolution_map`` (above) remains the sole authority on "does this string
    resolve" (ADR-0030 Invariant: one shared resolver). This exists so a
    linkify client (issue #410, ADR-0030 decision 5) can navigate to a
    resolved canonical page WITHOUT ever constructing a wiki path from a bare
    slug itself — CODING_STANDARD §12.5 / §12.4 already require this for the
    reader UI's citation links (``source.path`` is server-supplied); the
    resolution-map endpoint (``pages.get_resolution_map``) extends the same
    convention to wikilink navigation.

    Args:
        wiki_dir: Root wiki directory (``entities/`` and ``concepts/`` are
            read from beneath it; missing subdirectories are skipped).

    Returns:
        Dict mapping every real page slug to ``"wiki/<subdir>/<slug>.md"``.
    """
    paths: dict[str, str] = {}
    for subdir_name in _ALIAS_ELIGIBLE_SUBDIRS:
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in subdir.glob("*.md"):
            paths[page_path.stem] = f"wiki/{subdir_name}/{page_path.name}"
    return paths
