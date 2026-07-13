"""Knob registry + env-dict resolution for the load-test harness.

Documents the concurrency knobs named in issue #600's technical brief (file:line
refs verified against ``main@7debcc3``) and provides the pure merge/parse logic
the harness's ``run`` command needs to turn CLI ``--env KEY=VALUE`` overrides
into a subprocess environment. No I/O here — kept import-light and unit-testable
in isolation from the server subprocess / fake upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Knob registry — name -> (default, source) for the report's methodology table.
# Defaults mirror the app's own read-once-at-import fallback (see each knob's
# cited module); this dict does not enforce or change those defaults, it only
# documents them so the harness + report can cite one source of truth.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Knob:
    name: str
    default: int
    note: str


KNOWN_KNOBS: dict[str, Knob] = {
    "KB_MAX_INFLIGHT": Knob(
        "KB_MAX_INFLIGHT", 6, "general read semaphore (gateway/app/middleware.py)"
    ),
    "KB_MAX_ADMIN": Knob(
        "KB_MAX_ADMIN", 2, "admin/mutate semaphore (gateway/app/middleware.py)"
    ),
    "KB_SSE_MAX_CONCURRENT": Knob(
        "KB_SSE_MAX_CONCURRENT", 6, "SSE-specific cap (gateway/app/sse_capacity.py)"
    ),
    "KB_TRANSCRIBE_CONCURRENCY": Knob(
        "KB_TRANSCRIBE_CONCURRENCY",
        16,
        "page-worker pool (markdown_kb/app/transcriber.py)",
    ),
    "KB_TRANSCRIBE_PAGE_COUNT_CONCURRENCY": Knob(
        "KB_TRANSCRIBE_PAGE_COUNT_CONCURRENCY",
        4,
        "page-count CapacityLimiter (transcriber.py, #482)",
    ),
    "KB_TRANSCRIBE_MAX_CONCURRENT_JOBS": Knob(
        "KB_TRANSCRIBE_MAX_CONCURRENT_JOBS",
        2,
        "transcribe job registry (transcribe_jobs.py)",
    ),
    "KB_IMPORT_MAX_CONCURRENT_JOBS": Knob(
        "KB_IMPORT_MAX_CONCURRENT_JOBS", 2, "import job registry (import_jobs.py)"
    ),
    "KB_RATE_LIMIT_PER_IP": Knob(
        "KB_RATE_LIMIT_PER_IP",
        30,
        "per-IP fixed window; 0 disables (gateway/app/ratelimit.py)",
    ),
}

# Harness-side defaults layered under every scenario so the concurrency knobs
# under test are the only thing bounding a run — NOT an incidental rate limit
# or a daily-budget-exhaustion flip into degraded (no-LLM) serving, which
# would silently understate later requests' memory cost mid-scenario.
HARNESS_BASE_ENV: dict[str, str] = {
    "KB_RATE_LIMIT_PER_IP": "0",
    "KB_DAILY_USD_CAP": "1000",
    # Default-off already (unset); pinned explicitly so a stray dev .env
    # can't flip it on and add real OpenAI pings to a "key-free" scenario.
    "KB_WARMUP_PING": "",
}


def parse_env_overrides(pairs: list[str]) -> dict[str, str]:
    """Parse ``["KEY=VALUE", ...]`` CLI args into a dict.

    Raises:
        ValueError: an entry has no ``=`` or an empty key.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--env expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--env entry has an empty key: {pair!r}")
        out[key] = value
    return out


def resolve_env(*layers: dict[str, str]) -> dict[str, str]:
    """Merge env-dict layers left-to-right; later layers win on key conflict."""
    merged: dict[str, str] = {}
    for layer in layers:
        merged.update(layer)
    return merged
