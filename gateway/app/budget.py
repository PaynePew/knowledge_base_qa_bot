"""Shallow module per Ousterhout. Public surface: ``DailyBudget``, ``budget``,
``estimate_cost``, ``DAILY_USD_CAP``.

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
    | /wiki/import     | 0.10     | conversion + classification LLM passes          |
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
    | (default heavy)  | 0.10     | unknown heavy path → assume a mid-range cost    |

The ``/wiki/qa/{slug}/refile`` and ``/wiki/pages/{slug}`` keys are the
middleware's ``QA_REFILE_TEMPLATE`` / ``PAGES_DELETE_TEMPLATE`` — concrete
per-slug paths are canonicalised to them before ``charge()`` is called, so
the table stays exact-match (ADR-0026 decision 1, issue #380; ADR-0025,
issue #381). The delete path's $0.00 entry is explicit (not left to the
default-heavy fallback) so its zero LLM cost is documented rather than
silently indistinguishable from an un-tabulated path.

The numbers are intentionally above plausible real cost on a small demo corpus
(GPT-class answer rounds are sub-cent; ``text-embedding-3-small`` re-embeds are
fractions of a cent per thousand tokens).  Overestimating keeps the demo safely
under the $3/day in-app ceiling and the $15/mo provider hard limit.

Single-worker model (CODING_STANDARD §2.6/§2.7): a plain in-process ``dict``
guarded by the GIL is sufficient; multi-worker would need a shared store.
"""

from __future__ import annotations

import datetime
import os

# Conservative per-endpoint estimates (USD). See the module docstring table.
_COST_ESTIMATES: dict[str, float] = {
    "/chat/stream": 0.02,
    "/wiki/chat": 0.02,
    "/rag/chat": 0.02,
    "/wiki/lint": 0.15,
    "/wiki/ingest": 0.10,
    "/wiki/import": 0.10,
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
      - ``charge(path, day=None)``  — add the path's estimate to ``day``'s total.
      - ``over_cap(day=None)``      — True once ``day``'s total reaches the cap.
      - ``day_total(day=None)``     — current accumulated USD for ``day``.

    The store is a ``{day: total}`` dict.  Old days are never pruned (one float
    per calendar day is negligible for a demo lifetime); a lookup for a day with
    no entry returns 0.0, which is the UTC-midnight reset semantics.
    """

    def __init__(self, *, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self._totals: dict[str, float] = {}

    def charge(self, path: str, *, day: str | None = None) -> float:
        """Add ``path``'s estimate to ``day``'s total; return the new total."""
        key = day if day is not None else _utc_today()
        new_total = self._totals.get(key, 0.0) + estimate_cost(path)
        self._totals[key] = new_total
        return new_total

    def day_total(self, *, day: str | None = None) -> float:
        """Return the accumulated USD for ``day`` (0.0 if the day is unseen)."""
        key = day if day is not None else _utc_today()
        return self._totals.get(key, 0.0)

    def over_cap(self, *, day: str | None = None) -> bool:
        """True once ``day``'s accumulated total has reached the cap."""
        return self.day_total(day=day) >= self.cap_usd


# Module-level singleton — the one accumulator the middleware shares across
# requests (single-worker model; CODING_STANDARD §2.7).
budget = DailyBudget(cap_usd=DAILY_USD_CAP)
