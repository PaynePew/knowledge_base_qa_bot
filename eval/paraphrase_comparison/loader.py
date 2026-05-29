"""Shallow module per Ousterhout. Public surface: ``load_paraphrases``, ``load_metadata``, ``write_text_atomic``, ``replace_atomic``, ``QUERIES_PATH``.

Read/write helpers for the Phase 8 Paraphrase set and report (PRD #100).

``load_paraphrases`` parses ``queries.yaml`` into ``Paraphrase`` objects.
``write_text_atomic`` is the tmp-file + ``os.replace`` writer used for both
``queries.yaml`` and ``report.md`` so a crash mid-write never leaves a
half-written file (CODING_STANDARD §2.6 atomic write).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import yaml

from .models import Paraphrase

_PKG_ROOT = Path(__file__).resolve().parent
QUERIES_PATH = _PKG_ROOT / "queries.yaml"


def replace_atomic(
    src: str | Path, dst: Path, *, attempts: int = 5, backoff: float = 0.1
) -> None:
    """``os.replace`` with a bounded retry for transient Windows file locks.

    On Windows an antivirus or the Search indexer can briefly open a just-written
    file to scan it, making ``os.replace`` raise a transient ``PermissionError``
    (WinError 5 / 32). A short retry clears the race; a persistent failure still
    raises (CODING_STANDARD §2.6). Zero overhead on the normal first-try success;
    on POSIX the race does not occur so the loop runs exactly once.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff)


def load_paraphrases(path: Path = QUERIES_PATH) -> list[Paraphrase]:
    """Parse ``queries.yaml`` into ``Paraphrase`` objects.

    Raises on malformed YAML or a missing required field rather than silently
    dropping a Paraphrase — a corrupt query set is a fail-fast condition for the
    comparison (mirrors CODING_STANDARD §4.1 fail-fast on data corruption).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paraphrases: list[Paraphrase] = []
    for entry in data.get("paraphrases", []):
        paraphrases.append(
            Paraphrase(
                paraphrase_id=entry["paraphrase_id"],
                paraphrase_type=entry["paraphrase_type"],
                text=entry["text"],
                gold_docs_section_id=entry["gold_docs_section_id"],
                key_tokens_docs=list(entry["key_tokens_docs"]),
                key_tokens_wiki=list(entry["key_tokens_wiki"]),
                generation_notes=entry.get("generation_notes", ""),
            )
        )
    return paraphrases


def load_metadata(path: Path = QUERIES_PATH) -> dict:
    """Return the ``queries.yaml`` ``metadata`` block (generator, seed, cost, …).

    Returns an empty dict if the block is absent. The report's cost log reads
    ``cost_usd`` from here verbatim so an offline ``n/a (offline)`` value is
    surfaced honestly rather than fabricated (issue #104 cost-honesty note).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data.get("metadata", {}))


def write_text_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (tmp + os.replace; §2.6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=f"{path.stem}_"
    )
    try:
        # newline="\n": force LF on every OS so committed artifacts honour the
        # repo's `* eol=lf` .gitattributes (CODING_STANDARD §1.1) — Windows text
        # mode would otherwise translate "\n" to CRLF and dirty the working tree.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        replace_atomic(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
