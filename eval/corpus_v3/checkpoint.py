"""Deep module per Ousterhout. Public surface: ``CheckpointRow``,
``load_checkpoint``, ``append_checkpoint_row``, ``replay_ledger``,
``ledger_call_dicts``.

Append-only JSONL checkpointing for the corpus v3 live verdict run
(``live_runner.py``, issue #679 AC 2 -- the #672 lesson: a generation run was
lost TWICE to write-at-end when a network blip outlasted an hour of paid
work). Every answered ``(query_id, arm)`` pair is appended to disk the
instant it is produced (:func:`append_checkpoint_row`), carrying its own
ledger deltas (:func:`ledger_call_dicts`); a restart reloads every row
(:func:`load_checkpoint`) so the caller can skip already-answered pairs, and
reconstructs the FULL cost ledger from the checkpoint alone
(:func:`replay_ledger`) -- not just the current process's own spend -- so the
mid-run spend abort and the final actual-spend report in
``live_runner.run_live_verdict`` stay correct across any number of crashes
and resumes.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata

from .content_axes import AnswerRecord


@dataclass(frozen=True)
class CheckpointRow:
    """One durably-recorded answer: enough to (a) skip re-answering this
    ``(query_id, arm)`` pair on resume, (b) rebuild its ``AnswerRecord`` for
    axis scoring, and (c) replay its ledger calls into a fresh ``CostLedger``
    (:func:`replay_ledger`) so a resumed run's spend total covers every prior
    process, not just the current one."""

    query_id: str
    arm: str
    answer_text: str
    cited_source_ids: frozenset[str] = field(default_factory=frozenset)
    retrieved_source_ids: frozenset[str] = field(default_factory=frozenset)
    ledger_calls: tuple[dict, ...] = field(default_factory=tuple)

    def to_answer_record(self) -> AnswerRecord:
        return AnswerRecord(
            query_id=self.query_id,
            arm=self.arm,
            answer_text=self.answer_text,
            cited_source_ids=self.cited_source_ids,
            retrieved_source_ids=self.retrieved_source_ids,
        )

    def to_json_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "arm": self.arm,
            "answer_text": self.answer_text,
            "cited_source_ids": sorted(self.cited_source_ids),
            "retrieved_source_ids": sorted(self.retrieved_source_ids),
            "ledger_calls": list(self.ledger_calls),
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> CheckpointRow:
        return cls(
            query_id=d["query_id"],
            arm=d["arm"],
            answer_text=d["answer_text"],
            cited_source_ids=frozenset(d.get("cited_source_ids", [])),
            retrieved_source_ids=frozenset(d.get("retrieved_source_ids", [])),
            ledger_calls=tuple(d.get("ledger_calls", [])),
        )


def load_checkpoint(path: Path) -> list[CheckpointRow]:
    """Load every row of an existing checkpoint file, or ``[]`` if ``path``
    does not exist yet (a fresh run, not an error).

    A torn FINAL line (``append_checkpoint_row``'s ``fsync`` happened, but
    the process was killed before the trailing newline landed -- issue #683
    item 1) is skipped with a warning rather than raised: every row ahead of
    it already durably answered its ``(query_id, arm)`` pair, and a resume
    must not lose them over one incomplete tail write. A torn line that is
    NOT last means the append-only write already completed later rows around
    it -- real corruption, not a crash-mid-write tail -- so that still raises
    (CODING_STANDARD §4.1 fail-fast)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    last_index = len(lines) - 1
    rows = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(CheckpointRow.from_json_dict(json.loads(line)))
        except json.JSONDecodeError:
            if i != last_index:
                raise
            print(
                f"checkpoint {path}: skipping a torn final line "
                "(crash mid-write) -- resuming from the last complete row",
                file=sys.stderr,
            )
    return rows


def _truncate_torn_tail(path: Path) -> None:
    """If ``path``'s last byte is not a newline -- a previous
    :func:`append_checkpoint_row` was interrupted mid-write, before the
    trailing newline landed -- truncate the file back to the end of its last
    complete line (issue #683 finding 1, adversarial-verified HIGH).

    Without this, the next append's ``"a"`` mode would concatenate the new
    row's JSON directly onto the torn partial one, producing a single
    invalid line that is no longer LAST. :func:`load_checkpoint`'s torn-tail
    tolerance only forgives a torn line that IS last (its own docstring); a
    torn line buried mid-file is real corruption and raises forever --
    turning one already-lost row into every row appended after it also
    becoming unrecoverable.

    This is a pure trailing-byte check, not a JSON-validity check: it always
    truncates when the last byte isn't ``b"\\n"`` or ``b"\\r"``, even if the
    dangling content happens to parse as complete JSON. ``append_checkpoint_row``
    always terminates its own writes with ``"\\n"``, so a missing trailing
    terminator can only mean an interrupted write -- there is no legitimate
    case where a complete row is deliberately left unterminated. The
    discarded partial row is not newly lost data: :func:`load_checkpoint`
    already treats a torn final line as unrecoverable and skips it the same
    way.

    ``b"\\r"`` counts as a complete terminator too, alongside ``b"\\n"``,
    for a Windows-specific reason: ``append_checkpoint_row`` opens in TEXT
    mode, so on Windows every ``"\\n"`` it writes lands on disk as ``"\\r\\n"``.
    A crash torn exactly between those two bytes leaves the file ending in a
    lone ``b"\\r"`` with the row's own JSON content already fully written --
    only the newline SEQUENCE is torn, not the row. :func:`load_checkpoint`
    reads with universal newlines, so it already treats that lone ``\\r`` as
    a line terminator and durably counts the row as done; if this function
    disagreed and truncated it away, a durably-committed, already-answered
    ``(query_id, arm)`` pair would be silently dropped from disk and never
    re-answered (the caller's in-memory ``done`` set still has it marked
    answered from the earlier load). The truncation-point search below
    mirrors this: it looks for the last ``b"\\n"`` OR ``b"\\r"`` -- not just
    ``b"\\n"`` -- so a genuinely torn LATER row never wipes out an earlier
    row whose only terminator is a lone ``\\r``. Neither byte can appear
    inside a row's own JSON content (``json.dumps`` always escapes control
    characters), so both are unambiguous row-boundary markers.
    """
    data = path.read_bytes()
    if not data or data.endswith((b"\n", b"\r")):
        return
    # 0 when neither terminator exists anywhere in the file.
    truncate_at = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1
    with path.open("r+b") as fh:
        fh.truncate(truncate_at)


def append_checkpoint_row(path: Path, row: CheckpointRow) -> None:
    """Append one row as a single JSON line, flushing immediately -- the
    append-only durability property issue #679 AC 2 requires (never
    write-at-end; a crash after this call has already banked the answer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _truncate_torn_tail(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_json_dict()) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def replay_ledger(rows: Iterable[CheckpointRow]) -> CostLedger:
    """Rebuild a ``CostLedger`` from checkpointed rows' recorded calls -- the
    resume-time reconstruction that keeps the mid-run spend abort and the
    final actual-spend report accurate across restarts."""
    ledger = CostLedger()
    for row in rows:
        for call in row.ledger_calls:
            ledger.record(
                stack=call["stack"],
                phase=call["phase"],
                model=call["model"],
                usage=UsageMetadata.from_raw(call.get("usage")),
            )
    return ledger


def ledger_call_dicts(calls: Sequence) -> tuple[dict, ...]:
    """Serialise a slice of ``CostLedger.calls`` into ``CheckpointRow
    .ledger_calls``'s exact shape -- the caller's job (``live_runner
    .run_live_answering``) is to pass only the calls a single answer just
    added (a ledger-length before/after slice), not the whole ledger."""
    return tuple(
        {
            "stack": c.stack,
            "phase": c.phase,
            "model": c.model,
            "usage": {
                "input_tokens": c.usage.input_tokens,
                "output_tokens": c.usage.output_tokens,
                "total_tokens": c.usage.total_tokens,
            },
        }
        for c in calls
    )
