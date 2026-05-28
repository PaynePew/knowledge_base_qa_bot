"""Shallow module per Ousterhout. Public surface: ``log_event``, ``LOG_PATH``.

Vector RAG (Stack B) Wiki Log writer — the app's single observability channel
(CODING_STANDARD §5.1). Stack B stays decoupled from ``markdown_kb`` (PRD #100 /
issue #103), so it owns its own append-only log at ``vector_rag/log.md`` rather
than writing into markdown_kb's ``wiki/log.md``.

Each line uses the repo-wide format (CODING_STANDARD §5.2):

    ## [<ISO-8601 UTC>] <kind> | <summary>

The log is append-only; entries are never deleted or modified. Every
``kind`` emitted here has a row under the "Vector RAG (Stack B)" section of
``project-docs/log-kinds.md``.
"""

from __future__ import annotations

import datetime
from pathlib import Path

# Default log path — tests monkeypatch this to a tmp file. Lives under
# vector_rag/ (NOT markdown_kb's wiki/) so the two apps stay decoupled.
LOG_PATH = Path(__file__).resolve().parents[1] / "log.md"


def log_event(kind: str, summary: str, log_path: Path | None = None) -> None:
    """Append one log line to vector_rag/log.md (or the given log_path).

    Creates the parent directory if it does not exist.

    Line format:
        ## [2026-05-28T12:34:56.789012Z] index_built | files=3 chunks=9
    """
    target = log_path if log_path is not None else LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    line = f"## [{ts}] {kind} | {summary}\n"

    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
