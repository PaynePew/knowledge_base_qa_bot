"""Gateway endpoint tests for the Source lifecycle S1 surface (issue #604,
ADR-0041): ``GET /sources/{relpath}/impact``, ``POST /sources/retire``,
``POST /sources/restore``, ``GET /sources/trash``.

Tests use Starlette TestClient (in-process) with ``markdown_kb.app.
source_lifecycle``'s module-level ``DOCS_DIR`` / ``WIKI_DIR`` / ``TRASH_DIR``
redirected to tmp dirs, so no real docs/ / wiki/ / .trash/ is touched and no
OPENAI_API_KEY is needed. Hermetic pattern mirrors ``test_read_endpoints.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _page_text(slug: str, sources: list[str]) -> str:
    sources_block = "".join(f"  - {s}\n" for s in sources) if sources else "  []\n"
    return (
        "---\n"
        f"id: {slug}\n"
        "type: concept\n"
        "created: '2026-07-12T00:00:00Z'\n"
        "updated: '2026-07-12T00:00:00Z'\n"
        f"sources:\n{sources_block}"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes: {}\n"
        "---\n\n"
        f"# {slug}\n\nSome content.\n"
    )


@pytest.fixture()
def sources_env(tmp_path: Path, monkeypatch):
    """Wire source_lifecycle's roots to tmp dirs and return the TestClient."""
    import markdown_kb.app.logger as logger_module
    import markdown_kb.app.read as read_module
    import markdown_kb.app.source_lifecycle as sl_module

    docs = tmp_path / "docs"
    raw = tmp_path / "raw"
    wiki = tmp_path / "wiki"
    trash = tmp_path / ".trash"
    docs.mkdir(parents=True)
    raw.mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)

    (docs / "policy.md").write_text("# Policy\n\nRefunds take 7 days.\n", encoding="utf-8")
    (wiki / "concepts" / "full-orphan-candidate.md").write_text(
        _page_text("full-orphan-candidate", ["policy.md#refunds"]), encoding="utf-8"
    )

    monkeypatch.setattr(sl_module, "DOCS_DIR", docs)
    monkeypatch.setattr(sl_module, "WIKI_DIR", wiki)
    monkeypatch.setattr(sl_module, "TRASH_DIR", trash)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "log.md")
    # GET /read/file's whitelist is a SEPARATE module-level dict (read.py) —
    # redirect it too so the "read a trashed file" test below is hermetic.
    monkeypatch.setattr(
        read_module,
        "_WHITELIST_ROOTS",
        {"docs": docs, "raw": raw, "wiki": wiki, ".trash": trash},
    )

    from gateway.app.main import app

    client = TestClient(app, raise_server_exceptions=True)

    return {"client": client, "docs": docs, "wiki": wiki, "trash": trash}


# ---------------------------------------------------------------------------
# GET /sources/{relpath}/impact
# ---------------------------------------------------------------------------


def test_impact_reports_full_orphan(sources_env):
    resp = sources_env["client"].get("/sources/policy.md/impact")
    assert resp.status_code == 200
    data = resp.json()
    assert data["relpath"] == "policy.md"
    assert data["full_orphans"] == ["full-orphan-candidate"]
    assert data["partial_orphans"] == []


def test_impact_missing_source_404(sources_env):
    resp = sources_env["client"].get("/sources/ghost.md/impact")
    assert resp.status_code == 404


def test_impact_traversal_422(sources_env):
    resp = sources_env["client"].get("/sources/../../etc/passwd/impact")
    # FastAPI's :path converter normalises "../" segments away before this
    # ever reaches the handler on some ASGI stacks — assert it either never
    # resolves an existing file (404) or is refused outright (422), but never
    # succeeds (200).
    assert resp.status_code in (404, 422)


def test_impact_nested_relpath(sources_env):
    (sources_env["docs"] / "demo-zh").mkdir()
    (sources_env["docs"] / "demo-zh" / "policy.md").write_text("# 政策\n", encoding="utf-8")

    resp = sources_env["client"].get("/sources/demo-zh/policy.md/impact")
    assert resp.status_code == 200
    assert resp.json()["relpath"] == "demo-zh/policy.md"


# ---------------------------------------------------------------------------
# POST /sources/retire
# ---------------------------------------------------------------------------


def test_retire_moves_file_and_returns_impact(sources_env):
    resp = sources_env["client"].post("/sources/retire", json={"relpath": "policy.md"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["relpath"] == "policy.md"
    assert data["impact"]["full_orphans"] == ["full-orphan-candidate"]
    assert "timestamp" in data and data["timestamp"]

    assert not (sources_env["docs"] / "policy.md").exists()
    trashed = sources_env["trash"] / data["timestamp"] / "docs" / "policy.md"
    assert trashed.exists()


def test_retire_missing_source_404(sources_env):
    resp = sources_env["client"].post("/sources/retire", json={"relpath": "ghost.md"})
    assert resp.status_code == 404


def test_retire_traversal_422(sources_env):
    resp = sources_env["client"].post("/sources/retire", json={"relpath": "../escape.md"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /sources/restore
# ---------------------------------------------------------------------------


def test_restore_moves_file_back(sources_env):
    retire_resp = sources_env["client"].post("/sources/retire", json={"relpath": "policy.md"})
    timestamp = retire_resp.json()["timestamp"]

    resp = sources_env["client"].post(
        "/sources/restore", json={"timestamp": timestamp, "relpath": "policy.md"}
    )
    assert resp.status_code == 204, resp.text
    assert (sources_env["docs"] / "policy.md").exists()


def test_restore_missing_entry_404(sources_env):
    resp = sources_env["client"].post(
        "/sources/restore",
        json={"timestamp": "20260101T000000000000Z", "relpath": "policy.md"},
    )
    assert resp.status_code == 404


def test_restore_refuses_when_target_occupied_409(sources_env):
    retire_resp = sources_env["client"].post("/sources/retire", json={"relpath": "policy.md"})
    timestamp = retire_resp.json()["timestamp"]
    (sources_env["docs"] / "policy.md").write_text("# New\n", encoding="utf-8")

    resp = sources_env["client"].post(
        "/sources/restore", json={"timestamp": timestamp, "relpath": "policy.md"}
    )
    assert resp.status_code == 409


def test_restore_refuses_on_basename_collision_409(sources_env):
    retire_resp = sources_env["client"].post("/sources/retire", json={"relpath": "policy.md"})
    timestamp = retire_resp.json()["timestamp"]
    (sources_env["docs"] / "elsewhere").mkdir()
    (sources_env["docs"] / "elsewhere" / "policy.md").write_text("# Elsewhere\n", encoding="utf-8")

    resp = sources_env["client"].post(
        "/sources/restore", json={"timestamp": timestamp, "relpath": "policy.md"}
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /sources/trash
# ---------------------------------------------------------------------------


def test_trash_lists_retired_sources(sources_env):
    empty = sources_env["client"].get("/sources/trash")
    assert empty.status_code == 200
    assert empty.json()["entries"] == []

    retire_resp = sources_env["client"].post("/sources/retire", json={"relpath": "policy.md"})
    timestamp = retire_resp.json()["timestamp"]

    resp = sources_env["client"].get("/sources/trash")
    assert resp.status_code == 200
    assert resp.json()["entries"] == [{"timestamp": timestamp, "relpath": "policy.md"}]


def test_trash_readable_via_read_file(sources_env):
    """The retired Source is inspectable through the read whitelist before
    the curator decides to restore it (ADR-0041)."""
    retire_resp = sources_env["client"].post("/sources/retire", json={"relpath": "policy.md"})
    timestamp = retire_resp.json()["timestamp"]

    resp = sources_env["client"].get(
        "/read/file", params={"path": f".trash/{timestamp}/docs/policy.md"}
    )
    assert resp.status_code == 200
    assert resp.json()["content"].startswith("# Policy")
