"""Deep module per Ousterhout. Public surface: ``FeedbackStoreFull``,
``FEEDBACK_PATH``, ``MAX_COMMENT_CHARS``, ``MAX_STORE_BYTES``,
``append_feedback``, ``list_feedback``.

Reader Feedback store (issue #558, 2026-07-11 mini-grill). See CONTEXT.md
"Reader Feedback vocabulary" for the domain contract this module implements:
a self-contained opinion record — one Reaction (``up``/``down``) plus an
optional Comment — bound to an answer card by a client-minted ``answer_id``.
Reader Feedback is opinion data ABOUT the corpus, never part of it: this
module never touches the Section Index, the BM25/dense arms, or ``wiki/`` —
it only owns its own append-only store.

Persistence: ``.kb/feedback.jsonl``, one JSON object per line, append-only.
Already covered by the repo-wide ``.kb/`` gitignore rule (never force-added,
unlike the committed BM25/dense seeds) — container-ephemeral by design, same
posture as the Conversation Store. Duplicate submissions for the same
``answer_id`` are resolved at READ time by folding on ``answer_id``, last
write wins (``list_feedback``) — the append path never rewrites history.

Concurrency: a single module-level ``threading.Lock`` guards the
store-size check AND the append as one critical section, mirroring
``gateway.app.budget.DailyBudget.reserve_pages``'s atomic check-then-charge
pattern — two concurrent callers can never both pass the size check before
either write lands. No ``fsync`` (accepted trade-off per the issue: a crash
may lose the last line).
"""

from __future__ import annotations

import datetime
import json
import threading
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default store path — tests monkeypatch this to a tmp file (mirrors
#: ``gateway.app.logger.LOG_PATH`` / ``markdown_kb.app._paths.INDEX_PATH``).
FEEDBACK_PATH = _REPO_ROOT / ".kb" / "feedback.jsonl"

#: Comment length cap (issue #558 AC2) — enforced at the Pydantic request
#: schema boundary in ``gateway.app.routes`` (CODING_STANDARD §4.4); exported
#: here so the schema imports the single source of truth instead of a second
#: literal ``500``.
MAX_COMMENT_CHARS = 500

#: Store-size cap in bytes — once ``FEEDBACK_PATH`` is already at or beyond
#: this size, ``append_feedback`` refuses further writes (issue #558 AC2).
MAX_STORE_BYTES = 1 * 1024 * 1024  # 1 MB

#: Cap on the number of FOLDED records ``list_feedback`` returns (issue #558).
MAX_FOLDED_RECORDS = 200

_lock = threading.Lock()


class FeedbackStoreFull(Exception):
    """Raised by ``append_feedback`` when ``FEEDBACK_PATH`` is already >= ``MAX_STORE_BYTES``."""


def append_feedback(record: dict) -> dict:
    """Append one Reader Feedback record, adding the server-owned fields.

    Adds ``id`` (server-minted UUID, distinct from the client-minted
    ``answer_id`` already in ``record``), ``ts`` (ISO-8601 UTC), and ``v``
    (schema version, always ``1``) — mirrors the envelope convention the
    issue specifies. Returns the full stored record (including the added
    fields) so the route layer can build its response and log line from it
    without re-deriving them.

    Args:
        record: The validated request body as a plain dict (client-supplied
            fields only — ``answer_id``, ``reaction``, ``query``, etc.).

    Raises:
        FeedbackStoreFull: ``FEEDBACK_PATH`` is already at or beyond
            ``MAX_STORE_BYTES``. The size check and the write happen inside
            the SAME lock hold, so two concurrent callers can never both pass
            the check before either append lands.
    """
    stored = dict(record)
    stored["id"] = str(uuid.uuid4())
    stored["ts"] = datetime.datetime.now(datetime.UTC).isoformat()
    stored["v"] = 1
    line = json.dumps(stored, ensure_ascii=False) + "\n"

    with _lock:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if FEEDBACK_PATH.exists() and FEEDBACK_PATH.stat().st_size >= MAX_STORE_BYTES:
            raise FeedbackStoreFull("feedback store full")
        with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line)

    return stored


def list_feedback() -> dict:
    """Return the read-time folded view: ``{records, counts}``.

    Folds every raw line by ``answer_id`` — last write wins, so a comment
    appended after an earlier reaction-only submission for the same
    ``answer_id`` fully supersedes it (CONTEXT.md "Reader Feedback": "read
    time fold makes it supersede"). ``records`` is the folded set, newest
    first by each folded record's own ``ts``, capped at
    ``MAX_FOLDED_RECORDS``. ``counts`` reports ``up``/``down`` over the FULL
    folded (distinct-answer) set — not the capped slice — and ``total_raw``
    over every line on disk, so an operator can see distinct-answer
    sentiment separately from resubmission volume.

    A missing ``FEEDBACK_PATH`` (no feedback submitted yet) returns the empty
    shape rather than raising — the store is created lazily by the first
    ``append_feedback`` call, so "not yet created" is a normal, not a
    corrupt, state (distinct from CODING_STANDARD §4.1 fail-fast, which
    applies to a store that exists but is unreadable).

    "Newest first" orders by APPEND (file) position, not by parsing and
    comparing the ``ts`` string: two records minted in rapid succession (a
    reaction immediately followed by its own comment append, or two answers
    submitted back-to-back) can land within the same wall-clock tick on a
    coarse-resolution clock, and a ``ts``-string sort has no way to break
    that tie correctly. File order has no such tie — each append is
    strictly after the previous one — so it is the more reliable "recency"
    signal here.
    """
    if not FEEDBACK_PATH.exists():
        return {"records": [], "counts": {"up": 0, "down": 0, "total_raw": 0}}

    # folded: answer_id -> (line_index, record). line_index is the append
    # position (monotonically increasing), used as the sort tiebreaker/key
    # instead of the record's own `ts` string (see "Newest first" above).
    folded: dict[str, tuple[int, dict]] = {}
    total_raw = 0
    with FEEDBACK_PATH.open("r", encoding="utf-8") as fh:
        for line_index, raw_line in enumerate(fh):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total_raw += 1
            rec = json.loads(raw_line)
            folded[rec["answer_id"]] = (line_index, rec)  # last write wins

    all_folded = [rec for _, rec in sorted(folded.values(), key=lambda item: item[0], reverse=True)]
    up = sum(1 for r in all_folded if r["reaction"] == "up")
    down = sum(1 for r in all_folded if r["reaction"] == "down")

    return {
        "records": all_folded[:MAX_FOLDED_RECORDS],
        "counts": {"up": up, "down": down, "total_raw": total_raw},
    }
