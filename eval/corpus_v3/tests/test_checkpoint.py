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
