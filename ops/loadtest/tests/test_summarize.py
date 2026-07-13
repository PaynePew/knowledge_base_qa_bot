"""Fast hermetic unit tests for ops/loadtest/summarize.py (issue #600).

Pure dict-in, str/float-out math against hand-built result dicts shaped like
``scenarios.run_scenario``'s return value — no server, no real result files.
"""

from __future__ import annotations

from ops.loadtest.summarize import peak_rss_mb, request_error_rate, to_markdown_table


def _result(**overrides):
    base = {
        "scenario_id": "S1_chat_c6",
        "description": "test scenario",
        "wall_clock_sec": 12.3,
        "memory": {
            "peak_rss_polled_mb": 150.0,
            "peak_wset_os_mb": 160.0,
            "sample_count": 40,
        },
        "chat_load": {
            "requests_sent": 60,
            "requests_ok": 60,
            "requests_error": 0,
            "wall_clock_sec": 10.0,
        },
        "import_load": None,
    }
    base.update(overrides)
    return base


def test_peak_rss_mb_prefers_os_peak_wset_when_present():
    assert peak_rss_mb(_result()) == 160.0


def test_peak_rss_mb_falls_back_to_polled_when_no_peak_wset():
    r = _result(
        memory={
            "peak_rss_polled_mb": 150.0,
            "peak_wset_os_mb": None,
            "sample_count": 40,
        }
    )
    assert peak_rss_mb(r) == 150.0


def test_peak_rss_mb_none_when_no_memory_block():
    assert peak_rss_mb(_result(memory=None)) is None


def test_request_error_rate_zero_errors():
    assert request_error_rate(_result()) == 0.0


def test_request_error_rate_computes_fraction():
    r = _result(
        chat_load={
            "requests_sent": 20,
            "requests_ok": 15,
            "requests_error": 5,
            "wall_clock_sec": 5.0,
        }
    )
    assert request_error_rate(r) == 0.25


def test_request_error_rate_none_when_no_chat_load():
    assert request_error_rate(_result(chat_load=None)) is None


def test_request_error_rate_none_when_zero_requests_sent():
    r = _result(
        chat_load={
            "requests_sent": 0,
            "requests_ok": 0,
            "requests_error": 0,
            "wall_clock_sec": 0.0,
        }
    )
    assert request_error_rate(r) is None


def test_to_markdown_table_header_and_row_count():
    table = to_markdown_table([_result(), _result(scenario_id="S2_import")])
    lines = table.strip().splitlines()
    assert lines[0].startswith("| Scenario |")
    assert lines[1].startswith("|---")
    assert len(lines) == 4  # header + separator + 2 data rows


def test_to_markdown_table_includes_peak_and_scenario_id():
    table = to_markdown_table([_result()])
    assert "S1_chat_c6" in table
    assert "160.0" in table


def test_to_markdown_table_handles_missing_memory_and_import():
    table = to_markdown_table([_result(memory=None, import_load=None)])
    assert "| - |" in table


def test_to_markdown_table_empty_list():
    table = to_markdown_table([])
    assert table.startswith("| Scenario |")
