"""Read-only view helper for ``wiki/qa/<slug>.md`` pages — human-surface visibility.

Issue #377 / ADR-0026 decision 3 ("gates resolve on human surfaces only"): the
MCP server sees everything and approves nothing, so this module never writes
to ``wiki/qa/``. It gives both human surfaces that need to *display* a Filed
Answer — the CLI ``kb qa`` group and the MCP ``kb_lint_v1`` C8/C9 output — one
shared way to resolve a qa page's path and read its question/status/sources/
body, instead of duplicating markdown_kb's private frontmatter parsing.
``qa.py`` / ``lint.py`` keep their own internal readers for the write paths
this module never touches (CODING_STANDARD §2.4 — no private-symbol reach-in;
this module only calls markdown_kb's PUBLIC surface: ``indexer.WIKI_DIR`` and
``indexer.split_frontmatter``).

``kb_cli`` already depends on ``kb_mcp`` (``kb_cli.main`` imports
``kb_mcp.freshness``), so this module lives here and both packages share it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _wiki_dir() -> Path:
    """Resolve ``wiki/`` lazily so a monkeypatched ``indexer.WIKI_DIR`` is honoured."""
    import markdown_kb.app.indexer as indexer_mod

    return indexer_mod.WIKI_DIR


def qa_page_path(slug: str) -> Path:
    """Absolute on-disk path for ``wiki/qa/<slug>.md``. Does not imply existence."""
    return _wiki_dir() / "qa" / f"{slug}.md"


def display_path(slug: str) -> str:
    """Repo-relative display path for a qa page slug, e.g. ``wiki/qa/<slug>.md``.

    Purely string formatting — derived from the qa dir convention documented
    in ``markdown_kb.app.qa`` (``wiki/qa/<slug>.md``), not from a file read.
    Safe to call for a slug whose page does not exist (used to show *where a
    finding lives* even when the caller has not read the file).
    """
    return f"wiki/qa/{slug}.md"


@dataclass(frozen=True, slots=True)
class QaPageView:
    """A read-only snapshot of a Filed Answer page, shaped for display.

    ``slug``     — the qa page slug (filename stem under ``wiki/qa/``).
    ``status``   — ``frontmatter.status`` verbatim (``draft`` / ``live`` / etc.),
                   or ``None`` when absent.
    ``question`` — ``frontmatter.question`` verbatim (untruncated — unlike the
                   C8 lint finding's report-column truncation), or ``None``.
    ``count``    — ``frontmatter.count`` coerced to ``int``, or ``None`` when
                   absent or not coercible.
    ``sources``  — ``frontmatter.sources`` as a list of strings (empty list
                   when absent or not a list).
    ``body``     — the answer body text (frontmatter + sentinel comment
                   stripped, whitespace-trimmed).
    ``path``     — the repo-relative display path (``display_path(slug)``).
    """

    slug: str
    status: str | None
    question: str | None
    count: int | None
    sources: list[str] = field(default_factory=list)
    body: str = ""
    path: str = ""


def read_qa_page(slug: str) -> QaPageView | None:
    """Read ``wiki/qa/<slug>.md`` and return a display view, or ``None`` if unusable.

    Uses ``markdown_kb.app.indexer.split_frontmatter`` (public deep-module
    surface) to strip the leading sentinel HTML comment (``qa.SENTINEL_COMMENT``)
    and the YAML frontmatter block, the same way the write path renders it.

    Never raises. Returns ``None`` uniformly for: missing file, unreadable file
    (``OSError``), and unparseable/absent frontmatter — callers (CLI ``show``,
    MCP lint visibility) treat "no usable page" as one case rather than
    threading multiple failure branches through display code.
    """
    path = qa_page_path(slug)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    from markdown_kb.app.indexer import split_frontmatter

    metadata, body = split_frontmatter(text)
    if not metadata:
        return None

    raw_sources = metadata.get("sources", [])
    sources = [str(s) for s in raw_sources] if isinstance(raw_sources, list) else []

    raw_count = metadata.get("count")
    try:
        count = int(raw_count) if raw_count is not None else None
    except (TypeError, ValueError):
        count = None

    return QaPageView(
        slug=slug,
        status=metadata.get("status"),
        question=metadata.get("question"),
        count=count,
        sources=sources,
        body=body.strip(),
        path=display_path(slug),
    )
