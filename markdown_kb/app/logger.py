"""Shallow module per Ousterhout. Public surface: ``log_event``, ``LOG_PATH``.

Wiki Log writer — appends structured event lines to wiki/log.md.

Each line has the format:
    ## [<ISO-8601 UTC>] <kind> | <summary>

The log is append-only; entries are never deleted or modified.
"""

from __future__ import annotations

import datetime
from pathlib import Path

# Default log path — tests monkeypatch this to a tmp file.
LOG_PATH = Path(__file__).resolve().parents[2] / "wiki" / "log.md"


def log_event(kind: str, summary: str, log_path: Path | None = None) -> None:
    """Append one log line to wiki/log.md (or the given log_path).

    Creates the parent directory if it does not exist.

    Line format:
        ## [2026-05-25T12:34:56.789012Z] index_built | files=3 sections=9
    """
    target = log_path if log_path is not None else LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    line = f"## [{ts}] {kind} | {summary}\n"

    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
