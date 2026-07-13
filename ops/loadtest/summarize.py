"""Pure merge/format logic for turning committed scenario JSONs into the
report's Markdown tables. No I/O beyond simple file read/write in the CLI
wrapper (``harness.py``'s ``summarize`` command) — kept as pure functions
here so the math is unit-testable without a server.
"""

from __future__ import annotations

from typing import Any


def peak_rss_mb(result: dict[str, Any]) -> float | None:
    """Best available peak-RSS estimate: the OS-tracked Windows peak_wset when
    present (more accurate than the polled figure), else the polled peak."""
    memory = result.get("memory")
    if not memory:
        return None
    return memory.get("peak_wset_os_mb") or memory.get("peak_rss_polled_mb")


def request_error_rate(result: dict[str, Any]) -> float | None:
    """Fraction of chat-load requests that errored (0.0-1.0), or None if the
    scenario had no chat load."""
    chat = result.get("chat_load")
    if not chat or not chat.get("requests_sent"):
        return None
    return round(chat["requests_error"] / chat["requests_sent"], 4)


def to_markdown_table(results: list[dict[str, Any]]) -> str:
    """Render the per-scenario peak-RSS summary table (methodology's headline
    table). Rows are emitted in the order *results* is given — callers sort
    first if a specific order matters."""
    header = (
        "| Scenario | Description | Peak RSS (MB) | Source | Wall clock (s) | "
        "Chat sent/ok/err | Import status |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        memory = r.get("memory") or {}
        peak = peak_rss_mb(r)
        source = "peak_wset (OS)" if memory.get("peak_wset_os_mb") else "polled RSS"
        chat = r.get("chat_load")
        chat_cell = (
            f"{chat['requests_sent']}/{chat['requests_ok']}/{chat['requests_error']}"
            if chat
            else "-"
        )
        imp = r.get("import_load")
        import_cell = (
            f"{imp['status']} ({imp['files_done']}/{imp['files_total']})"
            if imp
            else "-"
        )
        label = r.get("result_id") or r["scenario_id"]
        rows.append(
            f"| {label} | {r['description']} | "
            f"{peak if peak is not None else '-'} | {source} | {r['wall_clock_sec']} | "
            f"{chat_cell} | {import_cell} |"
        )
    return header + "\n".join(rows) + "\n"
