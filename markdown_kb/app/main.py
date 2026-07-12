"""Shallow module per Ousterhout. Public surface: ``app`` (the FastAPI instance).

FastAPI app entrypoint. Loads .env, wires the router, rehydrates the
Section Index on startup via lifespan."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

# Env must be loaded BEFORE importing app modules — `retrieval` reads
# KB_SCORE_THRESHOLD at import time. E402 below is intentional.
load_dotenv(find_dotenv(usecwd=True))

from .indexer import load_index_json  # noqa: E402
from .qa import flush_pending_counts  # noqa: E402
from .routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Rehydrate the Section Index from .kb/index.json before serving requests.

    issue #581 — after ``yield`` (app shutdown), force-flush any wiki/qa/*.md
    count deltas still buffered in memory (KB_QA_COUNT_FLUSH_SEC batching) so
    a clean shutdown never silently drops a re-ask's count bump.
    """
    load_index_json()
    yield
    flush_pending_counts(force=True)


app = FastAPI(title="Markdown Knowledge Base Q&A Bot", lifespan=lifespan)
app.include_router(router)
