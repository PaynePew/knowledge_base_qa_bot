"""Shallow module per Ousterhout. Public surface: ``log_event``, ``LOG_PATH``.

Gateway Log writer — appends structured event lines to gateway/log.md.

The Gateway is the composition layer (ADR-0010) and owns its own log channel,
separate from the per-app channels of ``markdown_kb`` (``wiki/log.md``) and
``vector_rag`` (``vector_rag/log.md``). Per CODING_STANDARD §5.1: each package
owns ONE log channel (``<package>/log.md`` via its own ``log_event``); writing
into another package's log is the violation — not having a per-package log.

Each line uses the repo-wide format (CODING_STANDARD §5.2):

    ## [<ISO-8601 UTC>] <kind> | <summary>

The log is append-only; entries are never deleted or modified. Every
``kind`` emitted here has a row under the "Gateway" section of
``project-docs/log-kinds.md`` (Phase 11, issue #161).
"""

from __future__ import annotations

import datetime
from pathlib import Path

# Default log path — tests monkeypatch this to a tmp file.
# Lives under gateway/ (NOT markdown_kb's wiki/ or vector_rag/) so all three
# app packages stay decoupled (CODING_STANDARD §5.1 per-app-log-channel rule).
LOG_PATH = Path(__file__).resolve().parents[1] / "log.md"


def log_event(kind: str, summary: str, log_path: Path | None = None) -> None:
    """Append one log line to gateway/log.md (or the given log_path).

    Creates the parent directory if it does not exist.

    Line format:
        ## [2026-05-29T12:34:56.789012Z] chat_rewrite | session=X raw="..." rewritten="..."
    """
    target = log_path if log_path is not None else LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    line = f"## [{ts}] {kind} | {summary}\n"

    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
