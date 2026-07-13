"""Concurrent ``POST /chat/stream`` load driver.

Fires overlapping SSE requests against a running Gateway process using a
thread pool (one blocking ``httpx`` client per worker — matches how real
browser tabs generate concurrent load, and keeps this module dependency-free
beyond ``httpx``, already a repo dev dependency).

Queries are a fixed rotation of questions verified (issue #600 implementation
pass, against the committed ``markdown_kb`` BM25 index) to retrieve non-empty,
above-threshold sections — so every request reaches the draft+verify LLM path
instead of the pre-LLM Cannot-Confirm short-circuit. Fidelity of the *answer*
text doesn't matter here (the fake upstream returns a canned stub); reaching
the same code path real traffic reaches is what matters for a memory
characterization.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx

# Verified against the committed corpus to retrieve >0 sections (ranked_n=3,
# early_exit=False) as of issue #600's implementation pass — see
# project-docs/memory-envelope-600.md methodology for how these were checked.
GROUNDED_QUERIES: tuple[str, ...] = (
    "How long do refunds take?",
    "Can I change my email address?",
    "What is your return policy?",
    "How do I reset my password?",
    "How do I request a refund?",
)


@dataclass
class ChatLoadResult:
    requests_sent: int = 0
    requests_ok: int = 0
    requests_error: int = 0
    wall_clock_sec: float = 0.0
    errors: list[str] = field(default_factory=list)


def _one_request(
    base_url: str, query: str, stack: str, timeout: float
) -> tuple[bool, str | None]:
    """POST one chat/stream request, consume the SSE body, report ok/error.

    ``ok`` means HTTP 200 and a terminal ``done`` event was observed before
    the stream closed — a 503 (cap saturation), a terminal SSE ``error``
    event, or a connection failure is NOT ok, and is exactly the signal S1's
    concurrency sweep is looking for at the higher concurrency levels.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                f"{base_url}/chat/stream",
                params={"stack": stack},
                json={"query": query},
            ) as resp:
                if resp.status_code != 200:
                    return False, f"http_{resp.status_code}"
                saw_done = False
                for line in resp.iter_lines():
                    if line.startswith("event:") and "done" in line:
                        saw_done = True
                return (
                    (True, None) if saw_done else (False, "stream_closed_without_done")
                )
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}"


def run_chat_load(
    base_url: str,
    concurrency: int,
    requests_per_worker: int,
    stack: str = "wiki",
    timeout: float = 30.0,
) -> ChatLoadResult:
    """Drive ``concurrency`` workers, each firing ``requests_per_worker`` sequential
    requests (so total requests = concurrency * requests_per_worker), and block
    until all complete."""
    result = ChatLoadResult()
    start = time.monotonic()

    def worker(worker_index: int) -> list[tuple[bool, str | None]]:
        outcomes = []
        for i in range(requests_per_worker):
            query = GROUNDED_QUERIES[(worker_index + i) % len(GROUNDED_QUERIES)]
            outcomes.append(_one_request(base_url, query, stack, timeout))
        return outcomes

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(concurrency)]
        for future in as_completed(futures):
            for ok, err in future.result():
                result.requests_sent += 1
                if ok:
                    result.requests_ok += 1
                else:
                    result.requests_error += 1
                    if err is not None:
                        result.errors.append(err)

    result.wall_clock_sec = round(time.monotonic() - start, 3)
    return result
