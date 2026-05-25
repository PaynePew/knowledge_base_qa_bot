from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

# Env must be loaded BEFORE importing app modules — `retrieval` reads
# KB_SCORE_THRESHOLD at import time. E402 below is intentional.
load_dotenv(find_dotenv(usecwd=True))

from .indexer import load_index_json  # noqa: E402
from .routes import router  # noqa: E402

app = FastAPI(title="Markdown Knowledge Base Q&A Bot")
app.include_router(router)


@app.on_event("startup")
def load_persisted_index():
    load_index_json()
