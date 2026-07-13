"""CLI entry point for the memory-envelope load-test harness (issue #600).

Usage::

    uv run python -m ops.loadtest.harness run S1_chat_c6
    uv run python -m ops.loadtest.harness run S3_headline --env KB_MAX_INFLIGHT=2 --out-name S4_maxinflight2
    uv run python -m ops.loadtest.harness list
    uv run python -m ops.loadtest.harness summarize

Each ``run`` invocation is ONE synchronous command: it spawns the server,
drives load, samples memory, tears down, and writes exactly one
``<out-name>.json`` under ``ops/loadtest/results/`` before returning. No
agent-side background waits — see ``scenarios.py``'s module docstring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .scenarios import SCENARIOS, run_scenario
from .summarize import to_markdown_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _cmd_list(_args: argparse.Namespace) -> int:
    for scenario_id, spec in SCENARIOS.items():
        print(f"{scenario_id}: {spec.description}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = SCENARIOS.get(args.scenario_id)
    if spec is None:
        print(
            f"Unknown scenario {args.scenario_id!r}. Known: {', '.join(SCENARIOS)}",
            file=sys.stderr,
        )
        return 2

    try:
        extra_env = config.parse_env_overrides(args.env)
    except ValueError as exc:
        print(f"Bad --env argument: {exc}", file=sys.stderr)
        return 2

    out_name = args.out_name or args.scenario_id
    print(f"Running {args.scenario_id} (env overrides: {extra_env or '(none)'}) ...")
    try:
        result = run_scenario(spec, extra_env=extra_env)
    except Exception as exc:  # noqa: BLE001 — CLI boundary, report and exit non-zero
        print(f"Scenario {args.scenario_id} FAILED: {exc}", file=sys.stderr)
        return 1

    # Distinct from scenario_id (the spec's fixed identity) when --out-name
    # renames a knob-sensitivity rerun (e.g. S4_maxinflight2 for a S3_headline
    # rerun) — summarize's table uses this to label the two rows differently.
    result["result_id"] = out_name

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{out_name}.json"
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        print(f"No result JSONs found under {results_dir}", file=sys.stderr)
        return 1
    results = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    table = to_markdown_table(results)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(table)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ops.loadtest.harness")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List known scenario ids.")
    list_parser.set_defaults(func=_cmd_list)

    run_parser = sub.add_parser("run", help="Run one scenario synchronously.")
    run_parser.add_argument("scenario_id", choices=sorted(SCENARIOS))
    run_parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Knob override for this run (repeatable), e.g. --env KB_MAX_INFLIGHT=2.",
    )
    run_parser.add_argument(
        "--out-name",
        default=None,
        help="Result filename stem (default: the scenario id). Use a distinct "
        "name for a knob-sensitivity rerun so it doesn't overwrite the baseline.",
    )
    run_parser.set_defaults(func=_cmd_run)

    summarize_parser = sub.add_parser(
        "summarize", help="Merge result JSONs into a Markdown table."
    )
    summarize_parser.add_argument("--results-dir", default=None)
    summarize_parser.add_argument(
        "--out", default=None, help="Write the table here instead of stdout."
    )
    summarize_parser.set_defaults(func=_cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
