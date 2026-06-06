"""Shallow module per Ousterhout. Public surface: ``load_paraphrases``, ``load_metadata``, ``write_text_atomic``, ``replace_atomic``, ``QUERIES_PATH``.

Read/write helpers for the Phase 8 Paraphrase set and report (PRD #100).

``load_paraphrases`` parses ``queries.yaml`` into ``Paraphrase`` objects.
``write_text_atomic`` and ``replace_atomic`` are re-exported from
``markdown_kb.app.atomic``, which is the canonical home per CODING_STANDARD §2.6
(issue #211).  Callers that import these names from this module are unchanged.

``os`` and ``time`` are kept as module-level imports so that existing tests that
patch ``loader.os.replace`` / ``loader.time.sleep`` continue to work — those
attributes are the same Python singletons used by the implementation in
``markdown_kb.app.atomic``, so the monkeypatch seam remains valid.
"""

from __future__ import annotations

import os  # noqa: F401 — kept for test seam: monkeypatch.setattr(loader.os, "replace", …)
import time  # noqa: F401 — kept for test seam: monkeypatch.setattr(loader.time, "sleep", …)
from pathlib import Path

import yaml

from markdown_kb.app.atomic import replace_atomic, write_text_atomic  # noqa: F401

from .models import Paraphrase

_PKG_ROOT = Path(__file__).resolve().parent
QUERIES_PATH = _PKG_ROOT / "queries.yaml"


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
