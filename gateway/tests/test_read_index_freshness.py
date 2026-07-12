"""Gateway endpoint test for GET /read/index-freshness (issue #559 A2).

Thin wiring test — the staleness *logic* is covered exhaustively by
``markdown_kb/tests/test_indexer_index_stale.py``; this file only proves the
endpoint composes ``markdown_kb.app.indexer.index_stale`` correctly and
returns the documented ``{"stale": bool}`` shape.

Hermetic pattern mirrors ``test_read_endpoints.py``'s ``counts_env`` fixture
(monkeypatches ``markdown_kb.app.indexer.WIKI_DIR`` / ``INDEX_PATH`` to
tmp dirs — this endpoint never touches ``markdown_kb.app.read``'s roots).
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def freshness_env(tmp_path, monkeypatch):
    """Wire indexer.py's WIKI_DIR/INDEX_PATH to tmp dirs and return the client."""
    import markdown_kb.app.indexer as indexer_module

    wiki = tmp_path / "wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    index_path = tmp_path / ".kb" / "index.json"

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki)
    monkeypatch.setattr(indexer_module, "INDEX_PATH", index_path)

    from gateway.app.main import app

    client = TestClient(app, raise_server_exceptions=True)

    return {"client": client, "wiki": wiki, "index_path": index_path}


def test_index_freshness_response_shape_is_stale_bool_only(freshness_env):
    """A fresh (empty) pipeline instance -> {"stale": False}, no extra keys."""
    client = freshness_env["client"]
    resp = client.get("/read/index-freshness")
    assert resp.status_code == 200
    assert resp.json() == {"stale": False}


def test_index_freshness_true_when_wiki_has_content_and_no_index(freshness_env):
    client = freshness_env["client"]
    wiki = freshness_env["wiki"]
    (wiki / "entities").mkdir()
    (wiki / "entities" / "acme.md").write_text("# Acme\n", encoding="utf-8")

    resp = client.get("/read/index-freshness")
    assert resp.status_code == 200
    assert resp.json() == {"stale": True}


def test_index_freshness_false_when_index_rebuilt_after_wiki_edit(freshness_env):
    client = freshness_env["client"]
    wiki = freshness_env["wiki"]
    index_path = freshness_env["index_path"]

    (wiki / "entities").mkdir()
    page = wiki / "entities" / "acme.md"
    page.write_text("# Acme\n", encoding="utf-8")
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(page, (now - 10, now - 10))
    os.utime(index_path, (now, now))  # index rebuilt after the wiki edit

    resp = client.get("/read/index-freshness")
    assert resp.status_code == 200
    assert resp.json() == {"stale": False}
