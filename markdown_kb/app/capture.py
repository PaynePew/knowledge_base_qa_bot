"""Deep module per Ousterhout. Public surface: ``capture_source``.

Capture — MCP agent-authored Markdown → docs/.

Provides ``capture_source(filename, content)`` which validates the filename
(traversal-safe basename, reused logic from ``upload._is_safe_basename``),
stamps mandatory provenance frontmatter, and writes the Source atomically
to ``docs/``.  Capture **skips Import** — the content is already canonical
Markdown; no format conversion is performed.

Provenance frontmatter (mandatory per ADR-0017 / issue #226):
    origin: mcp-conversation
    created_at: <ISO-8601 UTC>
    authored_by: agent

The captured Source flows into the normal Ingest → Index lifecycle via the
other tools — that part is out of scope for this module.

Public surface:
    ``capture_source(filename, content, *, docs_dir)``
        Accepts a plain filename (no path components) and Markdown content
        string, stamps provenance frontmatter, and writes atomically to
        ``docs_dir`` (defaults to ``DOCS_DIR``).

        Raises ``ValueError`` if the filename is unsafe (traversal, separators,
        control characters, empty string).

See ADR-0017 and GitHub issue #230 for design rationale.
"""

from __future__ import annotations

import datetime

from pathlib import Path

from ._paths import DOCS_DIR
from .atomic import write_text_atomic
from .logger import log_event
from .upload import _is_safe_basename

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def capture_source(
    filename: str,
    content: str,
    *,
    docs_dir: Path | None = None,
) -> Path:
    """Write a Markdown Source to docs/ with mandatory provenance frontmatter.

    Validates the filename with the shared traversal-safe basename check
    (reused from ``upload._is_safe_basename``), then prepends the three
    mandatory provenance keys as a YAML front-matter block and writes the
    result atomically to ``docs_dir / filename``.

    Import is deliberately **not** called — the content is already canonical
    Markdown authored by the MCP agent.

    Args:
        filename:  Plain basename for the new Source (e.g. ``"my_note.md"``).
                   Must not contain path separators, ``..``, control chars,
                   or bidi control characters.
        content:   Markdown body of the Source (UTF-8 string).
        docs_dir:  Override the target directory.  Production callers omit
                   this; tests pass ``tmp_path / "docs"`` to stay hermetic.

    Returns:
        The ``Path`` of the written file.

    Raises:
        ValueError:  If the filename fails the safe-basename check.
    """
    effective_docs_dir = docs_dir if docs_dir is not None else DOCS_DIR

    # Validate filename — reuse the same guard as the Upload module so the
    # traversal-safety contract is consistent across all write surfaces.
    is_safe, reason = _is_safe_basename(filename)
    if not is_safe:
        raise ValueError(reason)

    # Stamp mandatory provenance frontmatter.  The block is prepended to the
    # caller's content; the caller's own YAML front-matter (if any) follows
    # the ``---`` separator naturally as valid YAML multi-document — but in
    # practice the MCP agent is expected to pass a bare Markdown body without
    # its own front-matter block.
    created_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    frontmatter = (
        "---\n"
        "origin: mcp-conversation\n"
        f"created_at: {created_at}\n"
        "authored_by: agent\n"
        "---\n"
    )
    full_content = frontmatter + content

    # Atomic write — tmp + os.replace so a crash mid-write never leaves a
    # partial file in docs/ (CODING_STANDARD §2.6).
    target = effective_docs_dir / filename
    write_text_atomic(target, full_content)

    log_event("capture_written", f"filename={filename!r}")

    return target
