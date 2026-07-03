"""Shallow module per Ousterhout. Public surface: ``DOCS_DIR``, ``WIKI_DIR``, ``INDEX_PATH``, ``is_bare_slug``.

Canonical filesystem locations for the markdown_kb app.

Centralises the three path constants previously declared in
:mod:`markdown_kb.app.indexer`. Both ``indexer.py`` (BM25 indexing /
``/index`` route) and ``ingest.py`` (Source -> wiki/ synthesis,
``/ingest`` route) reference the same canonical Sources directory, so
having ``ingest`` import the constant from ``indexer`` created a
one-way coupling that the Slice #29 implementer flagged.

The constants are intentionally module-level Path objects rather than
helper functions so that:

  * Existing test ``monkeypatch.setattr(<module>, "DOCS_DIR", ...)`` calls
    continue to work — ``from ._paths import DOCS_DIR`` rebinds the name
    inside the importing module's namespace, and the monkeypatch patches
    that local binding (the same machinery as before).
  * The ``is``-based default-sentinel check in ``indexer.build_index``
    (``docs_dir is not DOCS_DIR``) keeps its identity semantics — both
    callers see the same Path instance.

``is_bare_slug`` (issue #397) is the shared path-shape guard originally
written for ``qa._is_bare_slug`` / ``qa.promote_batch`` (issue #382). It
moved here — the shared canonical-locations module — once ``pages.py``
needed the same predicate for ``delete_full_orphan``, per CODING_STANDARD
§2.4's escalation rule ("the moment a second package needs the same
``_private`` symbol, promote it to the owner's public API instead of
importing it privately again"). Both ``qa.py`` and ``pages.py`` call it at
the top of every slug-taking mutator, before any filesystem access, so a
path-shaped slug (separators, ``..``, a Windows drive prefix, NUL) is
rejected the same way on every platform instead of relying on a FastAPI
path segment's incidental "cannot contain ``/``" property.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DOCS_DIR = _REPO_ROOT / "docs"
WIKI_DIR = _REPO_ROOT / "wiki"
INDEX_PATH = _REPO_ROOT / ".kb" / "index.json"


def is_bare_slug(slug: str) -> bool:
    """True iff ``slug`` is a bare single filename component, so that
    ``some_dir / f"{slug}.md"`` cannot resolve outside ``some_dir``.

    A FastAPI path segment for ``{slug}`` cannot contain ``/`` — that
    property is free — but it CAN contain ``\\`` or ``:``, which the route
    matcher never rejects yet which act as path separators once joined on
    Windows (``%5C`` decodes to a backslash; ``D:x`` is a drive-relative
    path). Both callers of this guard — ``qa.py``'s single-item mutators
    (``promote`` / ``delete`` / ``edit`` / ``refile``) and its body-supplied
    ``promote_batch`` list, plus ``pages.delete_full_orphan`` — treat this
    as a path-shape guard, never a charset allowlist: real corpus slugs
    include CJK (``compute_slug`` preserves Unicode verbatim) and stay
    valid.
    """
    if not slug or slug in {".", ".."}:
        return False
    return not any(ch in slug for ch in ("/", "\\", ":", "\x00"))
