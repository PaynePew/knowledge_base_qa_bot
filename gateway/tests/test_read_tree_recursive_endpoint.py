"""Gateway endpoint tests for GET /read/tree?recursive=true (issue #644).

Companion to ``test_read_endpoints.py`` (kept untouched — a new file per
this slice's one-file-per-new-surface convention). Covers:
  - ``recursive=true`` returns a flat cross-tree listing with `truncated`.
  - The default (``recursive`` absent) is byte-identical to the pre-#644
    response shape — no ``truncated`` key leaks in.
  - Traversal/whitelist guards hold in recursive mode.
  - The hard cap + truncated:true flag over HTTP.

Uses Starlette TestClient (in-process) with monkeypatched roots, same
hermetic pattern as test_read_endpoints.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


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

    from gateway.app.main import app

    client = TestClient(app, raise_server_exceptions=True)

    return {"client": client, "docs": docs, "raw": raw, "wiki": wiki, "roots": fake_roots}


# ---------------------------------------------------------------------------
# recursive=true — happy path
# ---------------------------------------------------------------------------


def test_recursive_true_flattens_nested_entries(read_env):
    client = read_env["client"]
    sub = read_env["docs"] / "a" / "b"
    sub.mkdir(parents=True)
    (sub / "deep.md").write_text("# Deep\n", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "", "recursive": "true"})
    assert resp.status_code == 200
    data = resp.json()

    relpaths = {e["relpath"] for e in data["entries"]}
    assert "docs/a/b/deep.md" in relpaths
    assert data["truncated"] is False


def test_recursive_true_entry_shape_matches_non_recursive(read_env):
    """Each entry still has name/relpath/is_dir/size — same TreeEntrySchema
    shape as the non-recursive listing, just flattened."""
    client = read_env["client"]
    (read_env["docs"] / "policy.md").write_text("# Policy\n", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "docs", "recursive": "true"})
    entries = resp.json()["entries"]
    file_entry = next(e for e in entries if not e["is_dir"])
    assert set(file_entry.keys()) == {"name", "relpath", "is_dir", "size"}


def test_recursive_true_relpath_is_navigable_to_read_file(read_env):
    client = read_env["client"]
    sub = read_env["docs"] / "sub"
    sub.mkdir()
    (sub / "note.md").write_text("Hello.", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "", "recursive": "true"})
    entries = resp.json()["entries"]
    file_entry = next(e for e in entries if e["relpath"] == "docs/sub/note.md")

    file_resp = client.get("/read/file", params={"path": file_entry["relpath"]})
    assert file_resp.status_code == 200
    assert "Hello" in file_resp.json()["content"]


# ---------------------------------------------------------------------------
# recursive absent/false — byte-identical to pre-#644 shape
# ---------------------------------------------------------------------------


def test_recursive_absent_response_has_no_truncated_key(read_env):
    """The default response_model=TreeListSchema path must not leak the new
    `truncated` field — issue #644 AC: 'existing one-level behavior
    byte-identical when absent'."""
    client = read_env["client"]
    (read_env["docs"] / "policy.md").write_text("# Policy\n", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "docs"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"entries"}
    assert "truncated" not in data


def test_recursive_false_explicit_matches_recursive_absent(read_env):
    client = read_env["client"]
    (read_env["docs"] / "policy.md").write_text("# Policy\n", encoding="utf-8")

    resp_absent = client.get("/read/tree", params={"path": "docs"})
    resp_false = client.get("/read/tree", params={"path": "docs", "recursive": "false"})
    assert resp_absent.json() == resp_false.json()


def test_recursive_false_does_not_flatten_nested_dirs(read_env):
    client = read_env["client"]
    sub = read_env["docs"] / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("x", encoding="utf-8")

    resp = client.get("/read/tree", params={"path": "docs", "recursive": "false"})
    names = {e["name"] for e in resp.json()["entries"]}
    assert names == {"sub"}


# ---------------------------------------------------------------------------
# SECURITY — same guards, recursive mode (issue #644 AC)
# ---------------------------------------------------------------------------


def test_recursive_true_dotdot_returns_400(read_env):
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": "docs/../wiki", "recursive": "true"})
    assert resp.status_code == 400


def test_recursive_true_kb_root_returns_400(read_env):
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": ".kb", "recursive": "true"})
    assert resp.status_code == 400


def test_recursive_true_kb_never_listed_even_if_present_on_disk(tmp_path, monkeypatch):
    import markdown_kb.app.read as read_module

    docs = tmp_path / "docs"
    raw = tmp_path / "raw"
    wiki = tmp_path / "wiki"
    for d in (docs, raw, wiki):
        d.mkdir(parents=True, exist_ok=True)
    kb = tmp_path / ".kb"
    kb.mkdir()
    (kb / "index.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(read_module, "_WHITELIST_ROOTS", {"docs": docs, "raw": raw, "wiki": wiki})

    from gateway.app.main import app

    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/read/tree", params={"path": "", "recursive": "true"})
    assert resp.status_code == 200
    relpaths = {e["relpath"] for e in resp.json()["entries"]}
    assert not any(".kb" in r for r in relpaths)


def test_recursive_true_nonexistent_subdir_returns_404(read_env):
    client = read_env["client"]
    resp = client.get("/read/tree", params={"path": "docs/no-such-subdir", "recursive": "true"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Hard cap + truncated:true over HTTP (issue #644 AC)
# ---------------------------------------------------------------------------


def test_recursive_true_truncated_flag_when_cap_hit(tmp_path, monkeypatch):
    import markdown_kb.app.read as read_module

    import gateway.app.routes as routes_module

    docs = tmp_path / "docs"
    raw = tmp_path / "raw"
    wiki = tmp_path / "wiki"
    for d in (docs, raw, wiki):
        d.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (docs / f"f{i}.md").write_text("x", encoding="utf-8")

    monkeypatch.setattr(read_module, "_WHITELIST_ROOTS", {"docs": docs, "raw": raw, "wiki": wiki})

    def _capped_list_tree_recursive(path, **kwargs):
        return read_module.list_tree_recursive(path, roots=read_module._WHITELIST_ROOTS, limit=3)

    monkeypatch.setattr(routes_module, "_list_tree_recursive", _capped_list_tree_recursive)

    from gateway.app.main import app

    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/read/tree", params={"path": "docs", "recursive": "true"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entries"]) == 3
    assert data["truncated"] is True
