"""Shallow module per Ousterhout. Public surface: ``log_event``, ``LOG_PATH``.

Hybrid Retrieval (Stack C) Wiki Log writer — the app's single observability
channel (CODING_STANDARD §5.1). Stack C is additive and stays decoupled from
``markdown_kb`` (Stack A) and ``vector_rag`` (Stack B) per ADR-0018, so it owns
its own append-only log at ``hybrid_kb/log.md`` rather than writing into either
existing channel.

Each line uses the repo-wide format (CODING_STANDARD §5.2):

    ## [<ISO-8601 UTC>] <kind> | <summary>

The log is append-only; entries are never deleted or modified. Every ``kind``
emitted here has a row under the "Hybrid Retrieval (Stack C)" section of
``project-docs/log-kinds.md``.
"""

from __future__ import annotations

import datetime
from pathlib import Path

# Default log path — tests monkeypatch this to a tmp file. Lives under
# hybrid_kb/ (NOT markdown_kb's wiki/ or vector_rag/) so the three stacks stay
# decoupled (ADR-0018 additive invariant).
LOG_PATH = Path(__file__).resolve().parents[1] / "log.md"


def log_event(kind: str, summary: str, log_path: Path | None = None) -> None:
    """Append one log line to hybrid_kb/log.md (or the given log_path).

    Creates the parent directory if it does not exist.

    Line format:
        ## [2026-06-28T12:34:56.789012Z] dense_index_built | sections=161
    """
    target = log_path if log_path is not None else LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    line = f"## [{ts}] {kind} | {summary}\n"

    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
