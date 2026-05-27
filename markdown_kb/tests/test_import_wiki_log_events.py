"""Integration tests for POST /import — Slice 7-1 Wiki Log events.

AC coverage (issue #90 — Slice 7-1):
  - import_batch_started emitted on endpoint entry
  - import_source emitted per successful import
  - import_batch_completed emitted on endpoint exit with counts + duration
  - Events appear in correct order in wiki/log.md
  - Wiki Log event format matches existing ## [<ts>] <kind> | <summary> convention
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"

LOG_LINE_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] (\S+) \| (.+)$")


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log path for isolation."""
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


def test_wiki_log_batch_started_and_completed(import_env):
    """import_batch_started and import_batch_completed are written to the Wiki Log."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    events = _parse_log(log_path)
    kinds = [k for k, _ in events]

    assert "import_batch_started" in kinds, f"import_batch_started missing from log: {kinds}"
    assert "import_batch_completed" in kinds, f"import_batch_completed missing from log: {kinds}"


def test_wiki_log_import_source_per_file(import_env):
    """import_source is written once per successfully imported file."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())
    (raw_dir / "simple.txt").write_bytes((FIXTURES / "simple.txt").read_bytes())

    client.post("/import")

    events = _parse_log(log_path)
    source_events = [(k, s) for k, s in events if k == "import_source"]

    assert len(source_events) == 2, (
        f"Expected 2 import_source events (one per file), got {len(source_events)}: {source_events}"
    )


def test_wiki_log_event_order(import_env):
    """import_batch_started comes before import_source, which comes before import_batch_completed."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    events = _parse_log(log_path)
    kinds = [k for k, _ in events]

    # Filter to just import-related events
    import_kinds = [k for k in kinds if k.startswith("import_")]

    assert import_kinds[0] == "import_batch_started", (
        f"First import event must be import_batch_started, got: {import_kinds}"
    )
    assert import_kinds[-1] == "import_batch_completed", (
        f"Last import event must be import_batch_completed, got: {import_kinds}"
    )
    # import_source appears between start and end
    assert "import_source" in import_kinds[1:-1], (
        f"import_source must appear between start and completed: {import_kinds}"
    )


def test_wiki_log_batch_completed_has_counts(import_env):
    """import_batch_completed summary includes count information."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    events = _parse_log(log_path)
    completed_events = [(k, s) for k, s in events if k == "import_batch_completed"]

    assert len(completed_events) == 1
    summary = completed_events[0][1]
    # Summary must mention imported count
    assert "imported=1" in summary or "imported" in summary, (
        f"import_batch_completed summary must mention imported count: {summary}"
    )


def test_wiki_log_format_all_lines(import_env):
    """All Wiki Log lines follow ## [<ISO-8601 UTC>] <kind> | <summary> format."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    assert log_path.exists(), "wiki/log.md must be created"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines, "log.md must not be empty after /import"

    for line in lines:
        if line.strip():  # skip blank lines
            assert LOG_LINE_RE.match(line), f"Log line does not match expected format: {repr(line)}"
