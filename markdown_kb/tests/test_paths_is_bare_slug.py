"""Tests for ``markdown_kb.app._paths.is_bare_slug`` — the shared path-shape guard.

Issue #397: a FastAPI ``{slug}`` path segment cannot contain ``/`` (route
matching rejects that for free) but CAN contain ``\\`` or a bare ``:`` —
neither affects route matching, so the request reaches the handler with the
raw character inside the slug. Both act as path separators once joined into
a filesystem path on Windows (``%5C`` decodes to a backslash; ``D:x`` is a
drive-relative path), so ``is_bare_slug`` rejects them the same way
regardless of whether the slug arrived via a path param or a JSON body.

This predicate originated as ``qa._is_bare_slug`` (issue #382,
``promote_batch``) and moved to ``_paths.py`` once ``pages.py`` needed the
same guard (CODING_STANDARD §2.4 escalation: "the moment a second package
needs the same ``_private`` symbol, promote it to the owner's public API").
Every call site (``qa.promote`` / ``qa.delete`` / ``qa.edit`` /
``qa.refile`` / ``qa.promote_batch`` / ``pages.delete_full_orphan``) is
covered by its own endpoint-level tests; this file tests the ONE shared
implementation directly and exhaustively.
"""

from __future__ import annotations

import pytest

from app._paths import is_bare_slug

# ---------------------------------------------------------------------------
# Rejected shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "",
        ".",
        "..",
        "a/b",
        "../escape",
        "a\\b",
        "..\\escape",
        "D:drive-relative",
        "nul\x00byte",
    ],
)
def test_is_bare_slug_rejects_path_shaped_input(slug):
    assert is_bare_slug(slug) is False


# ---------------------------------------------------------------------------
# Accepted shapes — a path-shape guard, not a charset allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "refund-policy",
        "how-do-i-cancel-my-order-abc123",
        "你們接受哪些付款方式-fb0f2e",
        "退款政策是什麼-abc123",
        "a.b",  # a single embedded dot is not a parent-ref, only bare "." / ".." are
        "...",  # three dots is not the literal ".." sentinel either
    ],
)
def test_is_bare_slug_accepts_real_corpus_shapes(slug):
    assert is_bare_slug(slug) is True
