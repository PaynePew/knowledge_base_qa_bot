"""Component tests for logger.py — log_event format and append behavior."""
import re
from pathlib import Path

import pytest

import app.logger as logger_module
from app.logger import log_event

LOG_LINE_RE = re.compile(
    r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] \S+ \| .+\n$"
)


# ---------------------------------------------------------------------------
# Acceptance criterion: test_logger_format_and_append
# ---------------------------------------------------------------------------


def test_logger_format_and_append(tmp_path, monkeypatch):
    """log_event writes ISO-8601 UTC line and successive calls preserve order."""
    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"

    # Patch LOG_PATH inside the logger module to a temp path
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    # First call — directory does not yet exist, must be created
    log_event("chat", "hello")
    assert log_path.exists(), "log.md must be created if missing"

    content = log_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    assert len(lines) == 1, f"Expected 1 line, got {len(lines)}: {lines}"

    line = lines[0]
    # The exact format: ## [<ISO-8601 UTC>] chat | hello\n
    assert re.match(
        r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] chat \| hello\n$",
        line,
    ), f"Line format mismatch: {repr(line)}"

    # Second call — must append, not overwrite
    log_event("index_built", "files=3 sections=9")
    content2 = log_path.read_text(encoding="utf-8")
    lines2 = content2.splitlines(keepends=True)
    assert len(lines2) == 2, f"Expected 2 lines after second log_event, got {len(lines2)}"

    # Insertion order preserved
    assert "chat" in lines2[0]
    assert "index_built" in lines2[1]


def test_logger_creates_wiki_dir(tmp_path, monkeypatch):
    """log_event creates the wiki/ directory if it doesn't exist."""
    wiki_dir = tmp_path / "wiki"
    log_path = wiki_dir / "log.md"

    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    assert not wiki_dir.exists(), "Pre-condition: wiki/ must not exist"
    log_event("test", "creating dir")
    assert wiki_dir.exists(), "wiki/ directory should be created by log_event"
    assert log_path.exists(), "log.md should exist after log_event"
