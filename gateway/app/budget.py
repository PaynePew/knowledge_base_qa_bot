"""Shallow module per Ousterhout. Public surface: ``DailyBudget``, ``budget``,
``estimate_cost``, ``DAILY_USD_CAP``, ``TRANSCRIBE_PAGE_USD``.

Per-UTC-day cost accumulator for the demo deploy's hard $/day ceiling (issue
#269, AC4).  Each *heavy* (OpenAI-calling) request is charged a **conservative
per-endpoint cost estimate** the moment it is admitted; once the running UTC-day
total reaches ``KB_DAILY_USD_CAP`` (default $3.00), the middleware short-circuits
further heavy requests to a 503 ``{"detail":"daily demo budget reached"}``.  The
accumulator is keyed by UTC calendar day (``YYYY-MM-DD``) so it resets at UTC
midnight without any background timer — a new day simply has no prior total.

Why estimates, not metering:  this is a *ceiling* guard, and for a ceiling the
safe error is to **overestimate** — an overestimate trips the brake early
(cheap), an underestimate lets cost overrun the hard limit (the failure we are
buying insurance against).  Precise token metering is the follow-up (AC4 note).

Per-endpoint estimate table (USD per request, deliberately generous):

    | Mounted path     | Estimate | Rationale (conservative ceiling)                |
    |------------------|----------|-------------------------------------------------|
    | /chat/stream     | 0.02     | one grounded answer: draft + verify LLM round   |
    | /wiki/chat       | 0.02     | same answer round on the Wiki stack             |
    | /rag/chat        | 0.02     | same answer round on the RAG stack              |
    | /wiki/lint       | 0.15     | C5 contradiction audit fans out to ~30 LLM pairs|
    | /wiki/ingest     | 0.10     | classify-on-outline + per-section LLM passes    |
    | /wiki/import     | 0.10     | mechanical conversion flat floor (Transcribe's per-page cost is charged on top, see below) |
    | /wiki/import/jobs | 0.10    | same Import surface as /wiki/import, just async (issue #497) — same flat floor, same per-page hook on top |
    | /wiki/transcribe | 0.00     | true cost is charged per-page (see below), not a flat estimate |
    | /wiki/transcribe/batch | 0.00 | same force-transcribe surface as /wiki/transcribe, just async (issue #447); true cost is the SAME per-page hook, not a second flat estimate |
    | /wiki/index      | 0.05     | Wiki re-index touches the corpus                |
    | /rag/index       | 0.50     | re-embeds the WHOLE corpus (many embed calls)   |
    | /hybrid/index    | 0.50     | re-embeds the WHOLE wiki Section corpus         |
    | /upload          | 0.01     | staging bytes; tiny, but heavy-gated for safety |
    | /wiki/pages/reconcile       | 0.05 | C5 draft over two pages' Source union + grounding check |
    | /wiki/pages/reconcile/apply | 0.02 | grounding re-check on the submitted final content       |
    | /wiki/pages/collision/merge                | 0.05 | C4 merge draft over a group's Source union + grounding check |
    | /wiki/pages/collision/merge/apply           | 0.02 | grounding re-check on the submitted base content              |
    | /wiki/pages/collision/differentiate         | 0.08 | C4 differentiate draft over N pages' Source union + N grounding checks |
    | /wiki/pages/collision/differentiate/apply   | 0.03 | grounding re-check on N submitted pages                        |
    | /wiki/qa/{slug}/refile | 0.05 | C9 re-file: full grounded answer round (draft + verify) server-side |
    | /wiki/pages/{slug} (DELETE) | 0.00 | C11 Confirmed orphan-delete: no LLM call at all (ADR-0024 Invariant) — hard delete + one BM25 reindex |
    | /wiki/qa/promote-batch | 0.00 | Direct-tier batch promote: no LLM call — per-slug status flips + one BM25 reindex (ADR-0023, issue #382) |
    | /wiki/pages/{slug}/aliases (POST) | 0.00 | Direct-tier assign-alias: no LLM call — one frontmatter field write, no reindex (ADR-0030, issue #409) |
    | /wiki/pages/{slug}/aliases/{alias} (DELETE) | 0.00 | Direct-tier remove-alias: no LLM call — one frontmatter field write, no reindex (ADR-0030 extension, issue #491) |
    | (default heavy)  | 0.10     | unknown heavy path → assume a mid-range cost    |

The ``/wiki/qa/{slug}/refile``, ``/wiki/pages/{slug}``,
``/wiki/pages/{slug}/aliases``, and ``/wiki/pages/{slug}/aliases/{alias}``
keys are the middleware's ``QA_REFILE_TEMPLATE`` / ``PAGES_DELETE_TEMPLATE``
/ ``ALIAS_ASSIGN_TEMPLATE`` / ``ALIAS_REMOVE_TEMPLATE`` — concrete per-slug
(and per-alias) paths are canonicalised to them before ``charge()`` is
called, so the table stays exact-match (ADR-0026 decision 1, issue #380;
ADR-0025, issue #381; ADR-0030 decision 3, issue #409; ADR-0030 extension,
issue #491). ``/wiki/qa/promote-batch``
is not parameterized (no slug in the path) so it needs no canonicalisation —
it is charged directly, mirroring the delete path's explicit $0.00 entry
(not left to the default-heavy fallback) so its zero LLM cost is documented
rather than silently indistinguishable from an un-tabulated path.

The numbers are intentionally above plausible real cost on a small demo corpus
(GPT-class answer rounds are sub-cent; ``text-embedding-3-small`` re-embeds are
fractions of a cent per thousand tokens).  Overestimating keeps the demo safely
under the $3/day in-app ceiling and the $15/mo provider hard limit.

Per-page Transcribe charging (issue #460): Transcribe (ADR-0032) makes one
vision-model call per PDF page, so a flat per-request estimate radically
undercounts a multi-page batch — ``/wiki/import`` in single-request batch
mode loops every staged raw source under ONE admission-time charge, so a
handful of large scans could transcribe hundreds of pages for $0.10 total.
``DailyBudget.charge_pages`` charges ``page_count * TRANSCRIBE_PAGE_USD``
into the SAME per-UTC-day ledger ``charge()`` uses. It is invoked via a hook
the Gateway registers with ``markdown_kb.app.transcriber`` at startup (see
``gateway/app/main.py``) — called with a PDF's page count BEFORE any vision
call for that file (reserve-before-spend), for both the ``/wiki/import``
auto-route and the forced ``/wiki/transcribe``. The hook checks
``over_cap()`` first and raises ``TranscribeBudgetExceeded`` before charging
if already at/over the ceiling, so a multi-file batch trips the cap partway
through — every file after the ceiling is crossed is rejected before its
pages are billed — rather than only discovering the overrun once the whole
batch (all vision calls) has already run.

Single-worker model (CODING_STANDARD §2.6/§2.7): one in-process ``dict`` is
the whole store; multi-worker would need a shared store. Within that single
process, ``charge()``/``charge_pages()`` are called from more than one thread
(issue #472) — the event loop's ``ProdMiddleware.charge()`` plus the
Transcribe page-budget hook running on worker threads (``asyncio.to_thread``
batch runs, anyio threadpool sync handlers) — so a read-modify-write on the
GIL alone is NOT safe (the GIL only makes each individual bytecode atomic,
not the multi-step "read total, add estimate, write total" sequence). Every
mutation and the check-then-charge admission (``reserve_pages``) takes the
same ``threading.Lock`` so concurrent callers see exact sums and no two
callers can both pass an over-cap check before either charge lands.
"""

from __future__ import annotations

import datetime
import os
import threading

# Conservative per-endpoint estimates (USD). See the module docstring table.
_COST_ESTIMATES: dict[str, float] = {
    "/chat/stream": 0.02,
    "/wiki/chat": 0.02,
    "/rag/chat": 0.02,
    "/wiki/lint": 0.15,
    "/wiki/ingest": 0.10,
    "/wiki/import": 0.10,
    "/wiki/import/jobs": 0.10,
    "/wiki/transcribe": 0.00,
    "/wiki/transcribe/batch": 0.00,
    "/wiki/index": 0.05,
    "/rag/index": 0.50,
    "/hybrid/index": 0.50,
    "/upload": 0.01,
    "/wiki/pages/reconcile": 0.05,
    "/wiki/pages/reconcile/apply": 0.02,
    "/wiki/pages/collision/merge": 0.05,
    "/wiki/pages/collision/merge/apply": 0.02,
    "/wiki/pages/collision/differentiate": 0.08,
    "/wiki/pages/collision/differentiate/apply": 0.03,
    "/wiki/qa/{slug}/refile": 0.05,
    "/wiki/pages/{slug}": 0.00,
    "/wiki/qa/promote-batch": 0.00,
    "/wiki/pages/{slug}/aliases": 0.00,
    "/wiki/pages/{slug}/aliases/{alias}": 0.00,
}

# Fallback for any heavy path missing from the table — assume a mid-range cost
# rather than zero, so an un-tabulated heavy endpoint still counts against the
# ceiling (overestimate-is-safe, AC4).
_DEFAULT_HEAVY_ESTIMATE = 0.10


def _read_cap() -> float:
    """Read ``KB_DAILY_USD_CAP`` at construction time (default 3.0)."""
    raw = os.getenv("KB_DAILY_USD_CAP", "3.0")
    try:
        return float(raw)
    except ValueError:
        return 3.0


# Module-level default, read once at import (restart to apply a new value).
DAILY_USD_CAP = _read_cap()


def _read_transcribe_page_cost() -> float:
    """Read ``KB_TRANSCRIBE_PAGE_USD`` at construction time (default $0.01/page).

    ADR-0032 estimates real Transcribe cost (``gpt-5-mini``) at ~$2 per 1,000
    pages (~$0.002/page); $0.01/page keeps the same generous overestimate
    margin (~5x) as the rest of this module's flat per-endpoint estimates.
    """
    raw = os.getenv("KB_TRANSCRIBE_PAGE_USD", "0.01")
    try:
        return float(raw)
    except ValueError:
        return 0.01


# Module-level default, read once at import (restart to apply a new value).
TRANSCRIBE_PAGE_USD = _read_transcribe_page_cost()


def estimate_cost(path: str) -> float:
    """Return the conservative USD estimate for a heavy ``path``.

    ``path`` is the full mounted path with any trailing slash already stripped
    (the caller — the middleware — normalises it).  Unknown heavy paths return
    ``_DEFAULT_HEAVY_ESTIMATE`` so they still count against the ceiling.
    """
    return _COST_ESTIMATES.get(path, _DEFAULT_HEAVY_ESTIMATE)


def _utc_today() -> str:
    """Return the current UTC calendar day as ``YYYY-MM-DD``."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")


class DailyBudget:
    """Accumulates heavy-request cost estimates per UTC day against a cap.

    Public surface:
      - ``charge(path, day=None)``       — add the path's estimate to ``day``'s total.
      - ``charge_pages(page_count, day=None)`` — add ``page_count * TRANSCRIBE_PAGE_USD``
        (issue #460 — Transcribe's per-page metered charge).
      - ``reserve_pages(page_count, day=None)`` — atomic over_cap-then-charge_pages
        (issue #472 — the Transcribe page-budget hook's admission check).
      - ``over_cap(day=None)``      — True once ``day``'s total reaches the cap.
      - ``day_total(day=None)``     — current accumulated USD for ``day``.

    The store is a ``{day: total}`` dict.  Old days are never pruned (one float
    per calendar day is negligible for a demo lifetime); a lookup for a day with
    no entry returns 0.0, which is the UTC-midnight reset semantics.

    Thread safety (issue #472): every read-modify-write and the admission
    check in ``reserve_pages`` hold ``self._lock`` for their full duration, so
    concurrent callers (event-loop coroutine + worker threads) never lose an
    update and never both pass an over-cap check before either charge lands.
    """

    def __init__(self, *, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self._totals: dict[str, float] = {}
        self._lock = threading.Lock()

    def charge(self, path: str, *, day: str | None = None) -> float:
        """Add ``path``'s estimate to ``day``'s total; return the new total."""
        key = day if day is not None else _utc_today()
        with self._lock:
            new_total = self._totals.get(key, 0.0) + estimate_cost(path)
            self._totals[key] = new_total
            return new_total

    def charge_pages(self, page_count: int, *, day: str | None = None) -> float:
        """Add ``page_count * TRANSCRIBE_PAGE_USD`` to ``day``'s total (issue #460).

        Charges the SAME ledger ``charge()`` uses, so Transcribe's per-page
        cost competes for the same daily ceiling as every other heavy path.
        """
        key = day if day is not None else _utc_today()
        with self._lock:
            new_total = self._totals.get(key, 0.0) + (page_count * TRANSCRIBE_PAGE_USD)
            self._totals[key] = new_total
            return new_total

    def reserve_pages(self, page_count: int, *, day: str | None = None) -> bool:
        """Atomically check ``over_cap`` then ``charge_pages`` under one lock hold.

        Returns ``True`` when the day was under cap and the charge was applied,
        ``False`` when the day was already at/over cap (no charge is applied —
        reserve-before-spend). Doing the check and the charge under a single
        lock acquisition (issue #472) closes the race where two concurrent
        callers both observe "under cap" before either has charged, which
        would let both admissions through and overshoot the ceiling.
        """
        key = day if day is not None else _utc_today()
        with self._lock:
            current = self._totals.get(key, 0.0)
            if current >= self.cap_usd:
                return False
            self._totals[key] = current + (page_count * TRANSCRIBE_PAGE_USD)
            return True

    def day_total(self, *, day: str | None = None) -> float:
        """Return the accumulated USD for ``day`` (0.0 if the day is unseen)."""
        key = day if day is not None else _utc_today()
        with self._lock:
            return self._totals.get(key, 0.0)

    def over_cap(self, *, day: str | None = None) -> bool:
        """True once ``day``'s accumulated total has reached the cap.

        This read alone is NOT the admission check for concurrent
        charge_pages callers — use ``reserve_pages`` for an atomic
        check-then-charge. ``over_cap`` remains correct standalone use (e.g.
        the event-loop middleware's single-threaded check before ``charge``).
        """
        return self.day_total(day=day) >= self.cap_usd


# Module-level singleton — the one accumulator the middleware shares across
# requests (single-worker model; CODING_STANDARD §2.7).
budget = DailyBudget(cap_usd=DAILY_USD_CAP)
