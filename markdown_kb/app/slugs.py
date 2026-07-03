"""Shared slug-safety helpers (CODING_STANDARD §2.4).

Public surface: ``is_bare_slug``.

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
"""

from __future__ import annotations


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
