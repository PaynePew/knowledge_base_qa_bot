"""Background RSS sampler for the Gateway process tree.

Polls ``psutil`` at a fixed interval and tracks the peak of the SUMMED RSS
across the target process + all live children. The tree walk is NOT
defensive padding — verified during issue #600's implementation pass that on
this Windows dev box, ``sys.executable -m uvicorn ... --workers 1`` (no
``--reload``) runs the actual server in ONE child process, while the
``subprocess.Popen``-returned PID stays alive as a thin, near-zero-RSS
launcher for the process's whole lifetime. Reading only the launcher PID
silently reports ~5MB regardless of real server memory — summing the tree is
required for a correct number, not optional. On Windows, also captures
``memory_info().peak_wset`` per live process in the tree (an OS-maintained
historical high-water-mark, more accurate than a ~200ms poll could
reconstruct) — summed the same way, at ``stop()`` while the tree is still
alive (a process that has already exited can no longer be queried).
"""

from __future__ import annotations

import platform
import threading
from dataclasses import dataclass, field

import psutil

_BYTES_PER_MB = 1024 * 1024


@dataclass
class SampleResult:
    peak_rss_polled_mb: float
    peak_wset_os_mb: float | None  # Windows only; None elsewhere
    sample_count: int
    platform: str = field(default_factory=platform.system)


class MemorySampler:
    """Poll-based peak-RSS tracker for one process (+ children), started/stopped
    around a single scenario's load-driving window."""

    def __init__(self, pid: int, interval_sec: float = 0.2) -> None:
        self._pid = pid
        self._interval_sec = interval_sec
        self._peak_rss_bytes = 0
        self._sample_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="loadtest-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> SampleResult:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        peak_wset = self._read_peak_wset()
        return SampleResult(
            peak_rss_polled_mb=round(self._peak_rss_bytes / _BYTES_PER_MB, 2),
            peak_wset_os_mb=round(peak_wset / _BYTES_PER_MB, 2)
            if peak_wset is not None
            else None,
            sample_count=self._sample_count,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self._interval_sec)
        # One final sample so a very short scenario still has a data point.
        self._sample_once()

    def _tree(self) -> list[psutil.Process]:
        try:
            proc = psutil.Process(self._pid)
        except psutil.NoSuchProcess:
            return []
        return [proc, *proc.children(recursive=True)]

    def _sample_once(self) -> None:
        total = 0
        alive = False
        for p in self._tree():
            try:
                total += p.memory_info().rss
                alive = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if alive:
            self._peak_rss_bytes = max(self._peak_rss_bytes, total)
            self._sample_count += 1

    def _read_peak_wset(self) -> int | None:
        if platform.system() != "Windows":
            return None
        total = 0
        found = False
        for p in self._tree():
            try:
                value = getattr(p.memory_info(), "peak_wset", None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if value is not None:
                total += value
                found = True
        return total if found else None
