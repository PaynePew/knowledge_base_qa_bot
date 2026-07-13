"""Scenario registry + single-scenario orchestration.

Each scenario is a spec (chat load params, import load on/off) plus a shared
``run_scenario`` driver: start the fake upstream, spawn the Gateway
subprocess, wait for health, sample memory across the load window, run the
load, tear everything down, and return a JSON-serializable result dict. One
call = one scenario = one synchronous command (issue #600 harness design
constraint — no background processes spanning turns).

The fake upstream + a dummy ``OPENAI_API_KEY`` run for EVERY scenario, not
just the chat ones — verified during implementation that ``vector_rag``'s
own sub-app lifespan (``vector_rag/app/main.py``, unconditionally mounted at
``/rag``) calls ``load_vector_index()`` -> ``get_embeddings()``, which raises
``RuntimeError`` at boot when no key is present at all. This contradicts the
issue's technical brief, which expected import-only scenarios to run fully
key-free; in practice the Gateway process cannot boot without *some* key.
The distinction that survives is at the LOAD level, not the process level:
``S2_import`` never issues an LLM-shaped request (the fake upstream sees zero
traffic during that scenario), so it is still the right scenario for
isolating import's own footprint from the chat code path's.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .chat_load import ChatLoadResult, run_chat_load
from .fake_upstream import run_fake_upstream
from .import_load import ImportLoadResult, run_import_load
from .process import spawn_gateway, terminate_gateway, wait_for_health
from .sampler import MemorySampler
from .transcribe_load import TranscribeLoadResult, run_transcribe_load


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    description: str
    chat_concurrency: int = 0
    chat_requests_per_worker: int = 0
    run_import: bool = False
    import_files: int = 6
    run_transcribe: bool = False
    transcribe_pages: int = 16
    settle_sec: float = 2.0
    # Per-scenario env layer (e.g. KB_TRANSCRIBE_ENABLED for S5) — merged in
    # UNDER the CLI's own --env overrides (see run_scenario), so a knob-
    # sensitivity rerun can still override a scenario default the same way
    # S4 overrode S3's KB_MAX_INFLIGHT.
    scenario_env: dict[str, str] = field(default_factory=dict)


SCENARIOS: dict[str, ScenarioSpec] = {
    "S0_idle": ScenarioSpec(
        scenario_id="S0_idle",
        description="Idle-after-warmup baseline (dummy key + fake upstream, zero requests).",
        settle_sec=5.0,
    ),
    "S1_chat_c1": ScenarioSpec(
        scenario_id="S1_chat_c1",
        description="Chat-only, stack=wiki, concurrency=1.",
        chat_concurrency=1,
        chat_requests_per_worker=20,
    ),
    "S1_chat_c6": ScenarioSpec(
        scenario_id="S1_chat_c6",
        description="Chat-only, stack=wiki, concurrency=6 (== KB_MAX_INFLIGHT default).",
        chat_concurrency=6,
        chat_requests_per_worker=10,
    ),
    "S1_chat_c12": ScenarioSpec(
        scenario_id="S1_chat_c12",
        description="Chat-only, stack=wiki, concurrency=12 (2x KB_MAX_INFLIGHT default; expect 503s past cap).",
        chat_concurrency=12,
        chat_requests_per_worker=8,
    ),
    "S2_import": ScenarioSpec(
        scenario_id="S2_import",
        description="Import-batch-only (zero LLM-shaped requests), POST /wiki/import/jobs submit+poll.",
        run_import=True,
        import_files=6,
    ),
    "S3_headline": ScenarioSpec(
        scenario_id="S3_headline",
        description="Headline: chat load at concurrency=6 running concurrently with an import batch.",
        chat_concurrency=6,
        chat_requests_per_worker=10,
        run_import=True,
        import_files=6,
    ),
    "S5_transcribe_c16": ScenarioSpec(
        scenario_id="S5_transcribe_c16",
        description=(
            "Transcribe-only: one synthetic 16-page scanned PDF (single source, "
            "so page-level concurrency saturates within it), KB_TRANSCRIBE_CONCURRENCY "
            "default (16). Rerun with --env KB_TRANSCRIBE_CONCURRENCY=3 --out-name "
            "S5_transcribe_c3 for the prod-cap comparison (issue #627)."
        ),
        run_transcribe=True,
        transcribe_pages=16,
        scenario_env={"KB_TRANSCRIBE_ENABLED": "true"},
    ),
    "S5_transcribe_headline": ScenarioSpec(
        scenario_id="S5_transcribe_headline",
        description=(
            "Chat load at concurrency=6 running concurrently with the same 16-page "
            "Transcribe batch as S5_transcribe_c16 (mirrors S3_headline's shape, "
            "issue #627)."
        ),
        chat_concurrency=6,
        chat_requests_per_worker=10,
        run_transcribe=True,
        transcribe_pages=16,
        scenario_env={"KB_TRANSCRIBE_ENABLED": "true"},
    ),
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rmtree_best_effort(path: str, attempts: int = 5, delay_sec: float = 0.3) -> None:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt == attempts - 1:
                return
            time.sleep(delay_sec)


def run_scenario(
    spec: ScenarioSpec,
    extra_env: dict[str, str] | None = None,
    gateway_port: int | None = None,
) -> dict[str, Any]:
    """Run one scenario end to end; returns the JSON-serializable result dict."""
    extra_env = extra_env or {}
    port = gateway_port or _free_port()
    base_gateway_url = f"http://127.0.0.1:{port}"

    with run_fake_upstream(_free_port()) as fake_base_url:
        env_layers: list[dict[str, str]] = [
            dict(os.environ),
            config.HARNESS_BASE_ENV,
            {"OPENAI_API_KEY": "dummy-loadtest-key", "OPENAI_API_BASE": fake_base_url},
            spec.scenario_env,
            extra_env,
        ]
        env = config.resolve_env(*env_layers)

        tmp_dir = tempfile.mkdtemp(prefix="kbqabot-loadtest-")
        try:
            log_path = Path(tmp_dir) / f"{spec.scenario_id}.server.log"
            proc = spawn_gateway(env, port, log_path)
            sample_result = None
            chat_result: ChatLoadResult | None = None
            import_result: ImportLoadResult | None = None
            transcribe_result: TranscribeLoadResult | None = None
            wall_start = time.monotonic()
            try:
                wait_for_health(
                    base_gateway_url, timeout=30.0, proc=proc, log_path=log_path
                )
                time.sleep(spec.settle_sec)

                sampler = MemorySampler(proc.pid)
                sampler.start()
                try:
                    if spec.chat_concurrency > 0 and spec.run_import:
                        chat_result, import_result = _run_concurrently(
                            base_gateway_url, spec
                        )
                    elif spec.chat_concurrency > 0 and spec.run_transcribe:
                        chat_result, transcribe_result = _run_chat_and_transcribe(
                            base_gateway_url, spec
                        )
                    elif spec.chat_concurrency > 0:
                        chat_result = run_chat_load(
                            base_gateway_url,
                            spec.chat_concurrency,
                            spec.chat_requests_per_worker,
                        )
                    elif spec.run_import:
                        import_result = run_import_load(
                            base_gateway_url, spec.import_files
                        )
                    elif spec.run_transcribe:
                        transcribe_result = run_transcribe_load(
                            base_gateway_url, spec.transcribe_pages
                        )
                    else:
                        time.sleep(spec.settle_sec)
                finally:
                    sample_result = sampler.stop()
            finally:
                terminate_gateway(proc)
        finally:
            # Best-effort: on Windows a just-terminated child's file handle can
            # stay locked for a few hundred ms after proc.wait() returns (OS
            # handle-table teardown lags process-exit signaling). Retry briefly,
            # then give up silently — this is a throwaway server log, not a
            # result artifact, and a stray temp dir is not worth failing the run.
            _rmtree_best_effort(tmp_dir)

    wall_clock_sec = round(time.monotonic() - wall_start, 3)
    return {
        "scenario_id": spec.scenario_id,
        "description": spec.description,
        "env_overrides": extra_env,
        "wall_clock_sec": wall_clock_sec,
        "platform": platform.system(),
        "memory": asdict(sample_result) if sample_result else None,
        "chat_load": asdict(chat_result) if chat_result else None,
        "import_load": asdict(import_result) if import_result else None,
        "transcribe_load": asdict(transcribe_result) if transcribe_result else None,
    }


def _run_concurrently(
    base_url: str, spec: ScenarioSpec
) -> tuple[ChatLoadResult, ImportLoadResult]:
    """Run chat load and an import batch on two threads, both blocking until done."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        chat_future = pool.submit(
            run_chat_load,
            base_url,
            spec.chat_concurrency,
            spec.chat_requests_per_worker,
        )
        import_future = pool.submit(run_import_load, base_url, spec.import_files)
        return chat_future.result(), import_future.result()


def _run_chat_and_transcribe(
    base_url: str, spec: ScenarioSpec
) -> tuple[ChatLoadResult, TranscribeLoadResult]:
    """Run chat load and a Transcribe batch on two threads, both blocking until
    done (issue #627 S5 — mirrors ``_run_concurrently``'s chat+import shape)."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        chat_future = pool.submit(
            run_chat_load,
            base_url,
            spec.chat_concurrency,
            spec.chat_requests_per_worker,
        )
        transcribe_future = pool.submit(
            run_transcribe_load, base_url, spec.transcribe_pages
        )
        return chat_future.result(), transcribe_future.result()
