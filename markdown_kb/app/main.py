from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

load_dotenv(find_dotenv(usecwd=True))

from .indexer import load_index_json
from .routes import router

app = FastAPI(title="Markdown Knowledge Base Q&A Bot")
app.include_router(router)


@app.on_event("startup")
def load_persisted_index():
    load_index_json()
