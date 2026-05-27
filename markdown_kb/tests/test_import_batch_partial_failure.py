"""Tests for batch partial failure semantics — Slice 7-2.

AC coverage (issue #91 — Slice 7-2):
  - 5 raw files with 2 failures: HTTP 200, 3 in imported_sources, 2 in failed_sources
  - Failure of N files in a batch does NOT abort the batch
  - Successful files appear in imported_sources; failed files appear in failed_sources
  - Wiki Log emits import_error event per failure with correct payload
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"

LOG_LINE_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] (\S+) \| (.+)$")


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log path into importer.py for isolation."""
    import app.importer as importer_module
    import app.logger as logger_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    log_path = tmp_path / "wiki" / "log.md"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    from app.main import app

    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    return {
        "client": client,
        "raw_dir": raw_dir,
        "docs_dir": docs_dir,
        "log_path": log_path,
    }


def _parse_log(log_path: Path) -> list[tuple[str, str]]:
    """Return list of (kind, summary) tuples from the log file."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    result = []
    for line in lines:
        m = LOG_LINE_RE.match(line)
        if m:
            result.append((m.group(2), m.group(3)))
    return result


def test_batch_partial_failure_5_files_2_failures(import_env):
    """5 raw files with 2 failures returns HTTP 200, 3 in imported_sources, 2 in failed_sources."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    # 3 good files
    (raw_dir / "good1.html").write_text("<h1>Good 1</h1><p>Content.</p>", encoding="utf-8")
    (raw_dir / "good2.html").write_text("<h1>Good 2</h1><p>Content.</p>", encoding="utf-8")
    (raw_dir / "good3.txt").write_text("Good 3 content.", encoding="utf-8")

    # 2 bad files — empty sources
    (raw_dir / "empty1.html").write_bytes(b"")
    (raw_dir / "empty2.txt").write_bytes(b"")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 3, (
        f"Expected 3 imported sources, got: {len(data['imported_sources'])}"
    )
    assert len(data["failed_sources"]) == 2, (
        f"Expected 2 failed sources, got: {len(data['failed_sources'])}"
    )

    # Verify failure types
    for failure in data["failed_sources"]:
        assert failure["error_type"] == "EmptySource"
        assert len(failure["error_message"]) <= 200

    # Verify successful docs files exist
    assert (docs_dir / "good1.md").exists()
    assert (docs_dir / "good2.md").exists()
    assert (docs_dir / "good3.md").exists()


def test_batch_partial_failure_continues_after_error(import_env):
    """Failure of one file does not abort the batch — subsequent files are processed."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    # Good file first
    (raw_dir / "before.html").write_text("<h1>Before</h1>", encoding="utf-8")
    # Bad file in the middle
    (raw_dir / "empty.html").write_bytes(b"")
    # Good file after
    (raw_dir / "after.html").write_text("<h1>After</h1>", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 2, (
        "Files before and after the failure must still be imported"
    )
    assert len(data["failed_sources"]) == 1


def test_batch_all_failures_still_http_200(import_env):
    """All files failing still returns HTTP 200 (not 4xx/5xx)."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    # All empty files
    (raw_dir / "a.html").write_bytes(b"")
    (raw_dir / "b.html").write_bytes(b"")
    (raw_dir / "c.txt").write_bytes(b"")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert data["imported_sources"] == []
    assert len(data["failed_sources"]) == 3


# ---------------------------------------------------------------------------
# import_error Wiki Log per failure
# ---------------------------------------------------------------------------


def test_wiki_log_import_error_per_failure(import_env):
    """import_error is emitted once per failed file with correct payload."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    (raw_dir / "good.html").write_text("<h1>Good</h1><p>Content.</p>", encoding="utf-8")
    (raw_dir / "empty.html").write_bytes(b"")

    resp = client.post("/import")
    assert resp.status_code == 200

    events = _parse_log(log_path)
    error_events = [(k, s) for k, s in events if k == "import_error"]

    assert len(error_events) == 1, f"Expected 1 import_error event, got: {error_events}"
    kind, summary = error_events[0]
    # Payload must include raw=, error_type=, error_message=
    assert "raw=" in summary, f"import_error summary must include raw=: {summary}"
    assert "error_type=" in summary, f"import_error summary must include error_type=: {summary}"
    assert "error_message=" in summary, (
        f"import_error summary must include error_message=: {summary}"
    )
    assert "EmptySource" in summary


def test_wiki_log_import_error_multiple_failures(import_env):
    """One import_error event per failed file (2 failures → 2 import_error events)."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    (raw_dir / "fail1.html").write_bytes(b"")
    (raw_dir / "fail2.html").write_bytes(b"")

    resp = client.post("/import")
    assert resp.status_code == 200

    events = _parse_log(log_path)
    error_events = [(k, s) for k, s in events if k == "import_error"]

    assert len(error_events) == 2, (
        f"Expected 2 import_error events (one per failure), got: {error_events}"
    )


def test_wiki_log_no_import_error_on_success(import_env):
    """No import_error event when all files succeed."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    (raw_dir / "good.html").write_text("<h1>Good</h1><p>Content.</p>", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    events = _parse_log(log_path)
    error_events = [(k, s) for k, s in events if k == "import_error"]

    assert error_events == [], (
        f"Expected no import_error events on full success, got: {error_events}"
    )
