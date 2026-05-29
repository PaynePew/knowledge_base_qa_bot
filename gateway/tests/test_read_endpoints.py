"""Gateway endpoint tests for GET /read/tree and GET /read/file (Phase 15 S5, issue #171).

Tests use Starlette TestClient (in-process) with monkeypatched roots so that:
  - No real docs/ / raw/ / wiki/ directories are required.
  - No OPENAI_API_KEY is needed.
  - Path-traversal rejection is exercised via the endpoint, not just the module.

Hermetic pattern mirrors test_chat_stream_filing.py (tmp_path + monkeypatch).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def read_env(tmp_path, monkeypatch):
    """Wire the read module's roots to tmp dirs and return the TestClient."""
    import markdown_kb.app.read as read_module

    docs = tmp_path / "docs"
    raw = tmp_path / "raw"
    wiki = tmp_path / "wiki"
    for d in (docs, raw, wiki):
        d.mkdir(parents=True, exist_ok=True)

    fake_roots = {"docs": docs, "raw": raw, "wiki": wiki}
    monkeypatch.setattr(read_module, "_WHITELIST_ROOTS", fake_roots)

    # Import the gateway app AFTER patching the roots so the router picks them up.
    from gateway.app.main import app

    client = TestClient(app, raise_server_exceptions=True)

    return {"client": client, "docs": docs, "raw": raw, "wiki": wiki, "roots": fake_roots}


# ---------------------------------------------------------------------------
# GET /read/tree — happy path
# ---------------------------------------------------------------------------


def test_tree_root_lists_three_roots(read_env):
    """GET /read/tree?path= returns docs, raw, wiki as directory entries."""
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": ""})
    assert resp.status_code == 200

    data = resp.json()
    names = {e["name"] for e in data["entries"]}
    assert names == {"docs", "raw", "wiki"}
    for entry in data["entries"]:
        assert entry["is_dir"] is True


def test_tree_lists_files_in_docs(read_env):
    """GET /read/tree?path=docs lists .md files present in docs/."""
    client = read_env["client"]
    (read_env["docs"] / "policy.md").write_text("# Policy\n", encoding="utf-8")
    (read_env["docs"] / "guide.md").write_text("# Guide\n", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "docs"})
    assert resp.status_code == 200

    names = {e["name"] for e in resp.json()["entries"]}
    assert "policy.md" in names
    assert "guide.md" in names


def test_tree_dirs_before_files(read_env):
    """Directories come before files in tree response."""
    client = read_env["client"]
    docs = read_env["docs"]
    (docs / "subdir").mkdir()
    (docs / "afile.md").write_text("x", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "docs"})
    entries = resp.json()["entries"]
    dir_idx = next(i for i, e in enumerate(entries) if e["is_dir"])
    file_idx = next(i for i, e in enumerate(entries) if not e["is_dir"])
    assert dir_idx < file_idx


def test_tree_entry_relpath_navigable(read_env):
    """Entry relpath from tree response is usable in subsequent read/file call."""
    client = read_env["client"]
    (read_env["docs"] / "readme.md").write_text("# Readme\n\nHello.", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "docs"})
    assert resp.status_code == 200

    entries = resp.json()["entries"]
    file_entry = next(e for e in entries if not e["is_dir"])
    relpath = file_entry["relpath"]

    # Use that relpath to read the file
    file_resp = client.get("/read/file", params={"path": relpath})
    assert file_resp.status_code == 200
    assert "Readme" in file_resp.json()["content"]


# ---------------------------------------------------------------------------
# GET /read/file — happy path
# ---------------------------------------------------------------------------


def test_file_returns_content(read_env):
    """GET /read/file?path=docs/policy.md returns raw Markdown text."""
    client = read_env["client"]
    content = "# Policy\n\nRefunds take 7 days.\n"
    (read_env["docs"] / "policy.md").write_text(content, encoding="utf-8")

    resp = client.get("/read/file", params={"path": "docs/policy.md"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["relpath"] == "docs/policy.md"
    assert data["content"] == content


def test_file_reads_wiki_log(read_env):
    """GET /read/file?path=wiki/log.md returns the log when present."""
    client = read_env["client"]
    log_content = "## [2026-05-29T00:00:00Z] index_built | files=3\n"
    (read_env["wiki"] / "log.md").write_text(log_content, encoding="utf-8")

    resp = client.get("/read/file", params={"path": "wiki/log.md"})
    assert resp.status_code == 200
    assert resp.json()["content"] == log_content


def test_file_reads_lint_report(read_env):
    """GET /read/file?path=wiki/lint-report.md returns lint report when present."""
    client = read_env["client"]
    report = "# Lint Report\n\nC1: OK\n"
    (read_env["wiki"] / "lint-report.md").write_text(report, encoding="utf-8")

    resp = client.get("/read/file", params={"path": "wiki/lint-report.md"})
    assert resp.status_code == 200
    assert resp.json()["content"] == report


# ---------------------------------------------------------------------------
# Security: path-traversal rejection at the endpoint level
# ---------------------------------------------------------------------------


def test_tree_dotdot_returns_400(read_env):
    """GET /read/tree?path=docs/../wiki returns HTTP 400."""
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": "docs/../wiki"})
    assert resp.status_code == 400


def test_file_dotdot_returns_400(read_env):
    """GET /read/file?path=docs/../../etc/passwd returns HTTP 400."""
    client = read_env["client"]
    resp = client.get("/read/file", params={"path": "docs/../../etc/passwd"})
    assert resp.status_code == 400


def test_tree_absolute_path_returns_400(read_env):
    """GET /read/tree with an absolute path returns HTTP 400."""
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": "/etc/passwd"})
    assert resp.status_code == 400


def test_file_absolute_path_returns_400(read_env):
    """GET /read/file with an absolute path returns HTTP 400."""
    client = read_env["client"]
    resp = client.get("/read/file", params={"path": "/etc/shadow"})
    assert resp.status_code == 400


def test_tree_kb_root_returns_400(read_env):
    """GET /read/tree?path=.kb returns HTTP 400 (not in whitelist)."""
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": ".kb"})
    assert resp.status_code == 400


def test_file_kb_path_returns_400(read_env):
    """GET /read/file?path=docs/../.kb/index.json returns HTTP 400."""
    client = read_env["client"]
    resp = client.get("/read/file", params={"path": "docs/../.kb/index.json"})
    assert resp.status_code == 400


def test_file_unknown_root_returns_400(read_env):
    """GET /read/file?path=secrets/config returns HTTP 400."""
    client = read_env["client"]
    resp = client.get("/read/file", params={"path": "secrets/config"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_file_nonexistent_returns_404(read_env):
    """GET /read/file for a path that does not exist returns HTTP 404."""
    client = read_env["client"]
    resp = client.get("/read/file", params={"path": "docs/ghost.md"})
    assert resp.status_code == 404


def test_tree_nonexistent_subdir_returns_404(read_env):
    """GET /read/tree for a path that does not exist returns HTTP 404."""
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": "docs/no-such-subdir"})
    assert resp.status_code == 404


def test_file_on_directory_returns_400(read_env):
    """GET /read/file on a directory returns HTTP 400."""
    client = read_env["client"]
    (read_env["docs"] / "subdir").mkdir()
    resp = client.get("/read/file", params={"path": "docs/subdir"})
    assert resp.status_code == 400
