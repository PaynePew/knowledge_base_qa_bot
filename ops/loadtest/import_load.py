"""``POST /wiki/import/jobs`` (raw -> docs) load driver — issues zero LLM-shaped
requests (see ``scenarios.py``'s module docstring for why the Gateway process
itself still needs *some* key to boot at all).

``markdown_kb/app/importer.py``'s ``RAW_DIR`` / (target) ``docs/`` are fixed
repo-root paths (no env override), so exercising the real batch-import route
against a real server subprocess means writing real files under ``raw/`` and
letting import write real files under ``docs/`` — exactly the working-tree
mutation issue #600's technical brief warns about. This module is the one
place that touches those directories, and it ALWAYS cleans up in a
``finally`` (both the planted ``raw/`` inputs and the ``docs/`` outputs
import produced from them), keyed off a run-unique filename prefix so
cleanup can never touch a real committed fixture.

Fixture sizes are kept under ``KB_LONGFORM_MIN_CHARS`` (2000, see
``markdown_kb/app/structure_enrichment.py``) so the Structure Enrichment gate
never fires — the mechanical ``.txt`` passthrough path this scenario
exercises never calls an LLM, regardless of the fake upstream being reachable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "raw"
DOCS_DIR = REPO_ROOT / "docs"

_FIXTURE_CHARS = 1200  # well under the 2000-char longform floor


@dataclass
class ImportLoadResult:
    status: str
    files_total: int
    files_done: int
    wall_clock_sec: float
    error: str | None = None


def _fixture_body(index: int, run_id: str) -> str:
    line = f"Load-test fixture line {index} for run {run_id}. "
    repeated = (line * (_FIXTURE_CHARS // len(line) + 1))[:_FIXTURE_CHARS]
    return f"# loadtest fixture {index}\n\n{repeated.strip()}\n"


def run_import_load(
    base_url: str, num_files: int = 6, poll_timeout: float = 60.0
) -> ImportLoadResult:
    """Plant *num_files* small ``.txt`` files under ``raw/``, submit a batch
    Import job, poll to a terminal state, then remove every file this run
    planted or produced — always, even on failure."""
    run_id = uuid.uuid4().hex[:8]
    prefix = f"_loadtest_{run_id}_"
    planted_raw: list[Path] = []
    produced_docs: list[Path] = []
    start = time.monotonic()

    try:
        RAW_DIR.mkdir(exist_ok=True)
        for i in range(num_files):
            raw_path = RAW_DIR / f"{prefix}{i}.txt"
            raw_path.write_text(_fixture_body(i, run_id), encoding="utf-8")
            planted_raw.append(raw_path)
            produced_docs.append(DOCS_DIR / f"{prefix}{i}.md")

        with httpx.Client(timeout=10.0) as client:
            submit_resp = client.post(f"{base_url}/wiki/import/jobs")
            submit_resp.raise_for_status()
            job_id = submit_resp.json()["job_id"]

            deadline = time.monotonic() + poll_timeout
            status = "submitted"
            files_total = 0
            files_done = 0
            while time.monotonic() < deadline:
                poll_resp = client.get(f"{base_url}/wiki/import/jobs/{job_id}")
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data["status"]
                files_total = data.get("files_total", 0)
                files_done = data.get("files_done", 0)
                if status in ("completed", "failed"):
                    break
                time.sleep(0.3)

        return ImportLoadResult(
            status=status,
            files_total=files_total,
            files_done=files_done,
            wall_clock_sec=round(time.monotonic() - start, 3),
        )
    except httpx.HTTPError as exc:
        return ImportLoadResult(
            status="error",
            files_total=0,
            files_done=0,
            wall_clock_sec=round(time.monotonic() - start, 3),
            error=str(exc),
        )
    finally:
        for path in (*planted_raw, *produced_docs):
            path.unlink(missing_ok=True)
