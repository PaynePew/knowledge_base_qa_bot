"""Spawn / health-wait / teardown for the Gateway server subprocess under test.

Invokes the venv's ``uvicorn`` directly (matching ``Dockerfile``'s prod CMD:
single worker, no reload) rather than ``uv run uvicorn`` — the latter adds an
extra wrapper-process layer between the harness and the process we actually
want to measure.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]


def spawn_gateway(
    env: dict[str, str], port: int, log_path: Path, host: str = "127.0.0.1"
) -> subprocess.Popen:
    """Start ``uvicorn gateway.app.main:app`` as a child process.

    ``env`` is the FULL environment for the child (callers pass
    ``config.resolve_env(dict(os.environ), ...)`` so PATH etc. survive) —
    this function does not merge with the harness's own environment.

    stdout/stderr are redirected to *log_path* (a real file, not
    ``subprocess.PIPE``): under sustained request load uvicorn's access log
    can exceed the OS pipe buffer, and with nothing draining a PIPE while the
    load driver is busy, the child blocks on its own stdout write — a subtle
    hang. A file has no such backpressure.
    """
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "gateway.app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        # The child inherits its own handle/fd to the file during Popen();
        # closing our copy on context-exit (standard idiom) doesn't affect it.
        return subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def wait_for_health(
    base_url: str, timeout: float, proc: subprocess.Popen, log_path: Path
) -> None:
    """Poll ``GET /healthz`` until 200 or *timeout* seconds elapse.

    Raises:
        RuntimeError: the process exited before becoming healthy, or the
            deadline passed while still starting — either way, the message
            includes the tail of *log_path* for diagnosis.
    """
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"gateway process exited early (code={proc.returncode}):\n{_tail(log_path)}"
            )
        try:
            resp = httpx.get(f"{base_url}/healthz", timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_exc = exc
        time.sleep(0.2)
    raise RuntimeError(
        f"gateway did not become healthy within {timeout}s (last error: {last_exc}):\n{_tail(log_path)}"
    )


def terminate_gateway(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Terminate the child process, escalating to kill on timeout.

    On Windows ``Popen.terminate()`` is already a hard stop (no SIGTERM
    equivalent) — a load-test harness does not need graceful in-flight-request
    draining.
    """
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def _tail(log_path: Path, max_chars: int = 4000) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no server log available)"
    return text[-max_chars:]
