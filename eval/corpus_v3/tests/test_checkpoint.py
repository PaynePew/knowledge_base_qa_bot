"""checkpoint.py tests — external behaviour only (CODING_STANDARD §0.2).

Covers issue #679 AC 2's checkpointing requirement in isolation from the
live-answering orchestration (``test_live_runner.py`` exercises resume/abort
behaviour end to end; this file exercises the JSONL persistence format and
ledger-replay reconstruction directly).
"""

from __future__ import annotations

import json

import pytest

from eval.corpus_v3 import checkpoint


def test_checkpoint_row_round_trips_through_json_dict():
    row = checkpoint.CheckpointRow(
        query_id="q1",
        arm="wiki",
        answer_text="Hours are 9-6. [Source: store-hours]",
        cited_source_ids=frozenset({"store-hours"}),
        retrieved_source_ids=frozenset({"store-hours", "other-page"}),
        ledger_calls=(
            {
                "stack": "wiki",
                "phase": "query",
                "model": "gpt-4o-mini",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            },
        ),
    )

    restored = checkpoint.CheckpointRow.from_json_dict(row.to_json_dict())

    assert restored == row


def test_checkpoint_row_to_answer_record_carries_the_scored_fields():
    row = checkpoint.CheckpointRow(
        query_id="q1",
        arm="rag",
        answer_text="Hours are 9-6. [Source: a.md#h]",
        cited_source_ids=frozenset({"a.md#h"}),
        retrieved_source_ids=frozenset({"a.md#h", "b.md#h"}),
    )
    record = row.to_answer_record()
    assert record.query_id == "q1"
    assert record.arm == "rag"
    assert record.cited_source_ids == frozenset({"a.md#h"})
    assert record.retrieved_source_ids == frozenset({"a.md#h", "b.md#h"})


def test_load_checkpoint_returns_empty_list_when_file_absent(tmp_path):
    assert checkpoint.load_checkpoint(tmp_path / "missing.jsonl") == []


def test_append_then_load_checkpoint_round_trips_in_order(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    row1 = checkpoint.CheckpointRow(query_id="q1", arm="wiki", answer_text="a1")
    row2 = checkpoint.CheckpointRow(query_id="q1", arm="rag", answer_text="a2")

    checkpoint.append_checkpoint_row(path, row1)
    checkpoint.append_checkpoint_row(path, row2)
    loaded = checkpoint.load_checkpoint(path)

    assert loaded == [row1, row2]


def test_replay_ledger_reconstructs_totals_from_checkpoint_rows():
    rows = [
        checkpoint.CheckpointRow(
            query_id="q1",
            arm="wiki",
            answer_text="a",
            ledger_calls=(
                {
                    "stack": "wiki",
                    "phase": "query",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                    },
                },
                {
                    "stack": "wiki",
                    "phase": "query",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "input_tokens": 300,
                        "output_tokens": 50,
                        "total_tokens": 350,
                    },
                },
            ),
        ),
        checkpoint.CheckpointRow(query_id="q2", arm="wiki", answer_text="refused"),
    ]

    ledger = checkpoint.replay_ledger(rows)

    totals = ledger.totals(stack="wiki", phase="query")
    assert totals.calls == 2
    assert totals.total_tokens == 120 + 350


def test_load_checkpoint_skips_a_torn_final_line_with_a_warning(tmp_path, capsys):
    """Issue #683 item 1: a crash mid-``fh.write`` can leave the last JSONL
    line truncated (e.g. process killed after the OS buffered a partial
    write but before the next newline). That is a normal resume scenario,
    not corruption -- the two complete rows ahead of it must still load."""
    path = tmp_path / "checkpoint.jsonl"
    row1 = checkpoint.CheckpointRow(query_id="q1", arm="wiki", answer_text="a1")
    row2 = checkpoint.CheckpointRow(query_id="q1", arm="rag", answer_text="a2")
    checkpoint.append_checkpoint_row(path, row1)
    checkpoint.append_checkpoint_row(path, row2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"query_id": "q2", "arm": "wiki", "answer_te')  # torn, no newline

    loaded = checkpoint.load_checkpoint(path)

    assert loaded == [row1, row2]
    assert "torn" in capsys.readouterr().err


def test_load_checkpoint_raises_on_a_torn_line_that_is_not_last(tmp_path):
    """A torn line BEFORE the final one is real corruption (the append-only
    write already completed later rows), not a crash-mid-write tail -- fail
    fast per CODING_STANDARD §4.1 rather than silently dropping data."""
    path = tmp_path / "checkpoint.jsonl"
    row2 = checkpoint.CheckpointRow(query_id="q1", arm="rag", answer_text="a2")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"query_id": "q1", "arm": "wiki", "answer_te\n')  # torn, not last
        fh.write(json.dumps(row2.to_json_dict()) + "\n")

    with pytest.raises(json.JSONDecodeError):
        checkpoint.load_checkpoint(path)


def test_append_checkpoint_row_truncates_a_torn_tail_before_appending(tmp_path):
    """Issue #683 finding 1 (adversarial-verified HIGH): append_checkpoint_row
    used to open in "a" mode unconditionally, so a torn tail left by a crash
    mid-write got the NEXT row's JSON concatenated onto it -- producing one
    invalid line that is no longer LAST, which load_checkpoint's torn-tail
    tolerance (only forgives a torn line that IS last) then raises on
    forever, destroying every row appended after the crash too. The verifier's
    exact repro: append two good rows, simulate a crash writing a partial
    third row with no trailing newline, then append three MORE rows -- every
    one of those appends must start on its own fresh line, and the final
    load must return all five complete rows without ever raising."""
    path = tmp_path / "checkpoint.jsonl"
    row1 = checkpoint.CheckpointRow(query_id="q1", arm="wiki", answer_text="a1")
    row2 = checkpoint.CheckpointRow(query_id="q1", arm="rag", answer_text="a2")
    checkpoint.append_checkpoint_row(path, row1)
    checkpoint.append_checkpoint_row(path, row2)
    # Simulate a crash mid-write: a partial row with no trailing newline.
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"query_id": "q3", "arm": "wiki", "answer_te')

    row3 = checkpoint.CheckpointRow(query_id="q3", arm="wiki", answer_text="a3")
    row4 = checkpoint.CheckpointRow(query_id="q4", arm="wiki", answer_text="a4")
    row5 = checkpoint.CheckpointRow(query_id="q5", arm="wiki", answer_text="a5")
    checkpoint.append_checkpoint_row(path, row3)
    checkpoint.append_checkpoint_row(path, row4)
    checkpoint.append_checkpoint_row(path, row5)

    loaded = checkpoint.load_checkpoint(path)

    assert loaded == [row1, row2, row3, row4, row5]


def test_append_checkpoint_row_truncates_a_file_that_is_entirely_a_torn_tail(tmp_path):
    """A checkpoint file that crashed on its very first write (no complete
    row at all yet, just a torn partial line with no newline) must truncate
    to empty before the next append, not concatenate onto it."""
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"query_id": "q1", "arm": "wiki", "answer_te', encoding="utf-8")

    row = checkpoint.CheckpointRow(query_id="q1", arm="wiki", answer_text="a1")
    checkpoint.append_checkpoint_row(path, row)

    loaded = checkpoint.load_checkpoint(path)
    assert loaded == [row]


def test_append_checkpoint_row_truncates_when_file_has_no_newline_at_all(tmp_path):
    """Even a technically well-formed JSON row missing only its trailing
    newline is treated as an unterminated tail and discarded before the next
    append. The torn-tail check is a pure trailing-byte check (mirrors how
    append_checkpoint_row always writes its OWN trailing newline), not a
    JSON-validity check -- it can never leave an ambiguous partial write on
    disk for the next append to build on."""
    path = tmp_path / "checkpoint.jsonl"
    row0 = checkpoint.CheckpointRow(query_id="q0", arm="wiki", answer_text="a0")
    path.write_text(json.dumps(row0.to_json_dict()), encoding="utf-8")  # no trailing \n

    row1 = checkpoint.CheckpointRow(query_id="q1", arm="wiki", answer_text="a1")
    checkpoint.append_checkpoint_row(path, row1)

    loaded = checkpoint.load_checkpoint(path)
    assert loaded == [row1]


def test_append_checkpoint_row_does_not_discard_a_row_torn_only_on_the_trailing_lf(
    tmp_path,
):
    """Windows regression (verdict follow-up on issue #683 finding 1):
    append_checkpoint_row opens in text mode, so on Windows every ``"\\n"``
    it writes becomes ``"\\r\\n"`` on disk. A crash torn exactly between the
    ``\\r`` and the ``\\n`` of that trailing sequence leaves the file ending
    in a lone ``b"\\r"`` -- the row's own JSON content is fully written, only
    the newline SEQUENCE itself is torn. load_checkpoint reads with
    universal newlines, so it already treats that lone ``\\r`` as a line
    terminator and successfully loads the row, durably counting its
    ``(query_id, arm)`` pair as done. ``_truncate_torn_tail`` must agree: if
    it instead discards this row (because its raw bytes don't end in
    ``b"\\n"``), a durably-committed, already-"done" paid answer is silently
    dropped out from under the next append -- and it is never re-answered,
    since the caller's in-memory ``done`` set still has the pair marked
    answered from the earlier load."""
    path = tmp_path / "checkpoint.jsonl"
    row1 = checkpoint.CheckpointRow(query_id="q1", arm="wiki", answer_text="a1")
    with path.open("wb") as fh:
        fh.write(json.dumps(row1.to_json_dict()).encode("utf-8") + b"\r")

    # load_checkpoint (universal newlines) already treats this as complete.
    assert checkpoint.load_checkpoint(path) == [row1]

    row2 = checkpoint.CheckpointRow(query_id="q2", arm="wiki", answer_text="a2")
    checkpoint.append_checkpoint_row(path, row2)

    assert checkpoint.load_checkpoint(path) == [row1, row2]


def test_append_checkpoint_row_truncation_search_respects_a_lone_cr_boundary(tmp_path):
    """The truncation-point search (used when the tail genuinely IS torn)
    must also recognize a lone ``\\r`` as a valid row boundary, not just
    ``\\n`` -- otherwise, with no ``\\n`` anywhere in the file, it would
    wrongly truncate all the way back to empty on a later row's real tear,
    discarding an earlier row that was already durably complete (terminated
    only by ``\\r``, the same Windows-torn-CRLF case as above)."""
    path = tmp_path / "checkpoint.jsonl"
    row1 = checkpoint.CheckpointRow(query_id="q1", arm="wiki", answer_text="a1")
    with path.open("wb") as fh:
        fh.write(json.dumps(row1.to_json_dict()).encode("utf-8") + b"\r")
        fh.write(b'{"query_id": "q2", "arm": "wiki", "answer_te')  # genuinely torn

    row2 = checkpoint.CheckpointRow(query_id="q2", arm="wiki", answer_text="a2")
    checkpoint.append_checkpoint_row(path, row2)

    assert checkpoint.load_checkpoint(path) == [row1, row2]


def test_ledger_call_dicts_matches_checkpoint_row_ledger_calls_shape(tmp_path):
    """The exact shape ``run_live_answering`` feeds into ``CheckpointRow
    .ledger_calls`` -- round-trips through ``replay_ledger`` unchanged."""
    from eval.cost_ledger.ledger import CostLedger
    from eval.cost_ledger.models import UsageMetadata

    ledger = CostLedger()
    ledger.record(
        stack="wiki",
        phase="query",
        model="gpt-4o-mini",
        usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
    )

    call_dicts = checkpoint.ledger_call_dicts(ledger.calls)
    row = checkpoint.CheckpointRow(
        query_id="q1", arm="wiki", answer_text="a", ledger_calls=call_dicts
    )
    replayed = checkpoint.replay_ledger([row])

    assert replayed.totals(stack="wiki", phase="query").total_tokens == 15
