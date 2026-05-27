"""Integration tests for POST /import — Slice 7-3 hash skip (no-op re-import).

AC coverage (issue #92 — Slice 7-3):
  - Re-import of unchanged raw file: skipped_sources entry, status="skipped"
  - No disk write on hash-match (mtime of docs file unchanged)
  - No markdownify invocation on hash-match (conversion path not entered)
  - ImportResponse.skipped_sources populated with hash-match entries
  - ImportSourceResult.status='skipped' in skipped_sources
  - Wiki Log emits import_skipped event with raw_path, docs_path, content_sha256
  - Hash compare happens BEFORE markdownify (verified via spy)
  - content_sha256 in skipped_sources entry matches original
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

LOG_LINE_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] (\S+) \| (.+)$")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log_path for isolation."""
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


# ---------------------------------------------------------------------------
# Hash skip — skipped_sources populated
# ---------------------------------------------------------------------------


def test_reimport_unchanged_file_goes_to_skipped_sources(import_env):
    """Re-import of unchanged raw file produces a skipped_sources entry, not imported_sources."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    raw_file = raw_dir / "article.html"
    raw_file.write_text("<h1>Hello</h1><p>Some content.</p>", encoding="utf-8")

    # First import: creates the docs file
    resp1 = client.post("/import")
    assert resp1.status_code == 200
    assert len(resp1.json()["imported_sources"]) == 1
    assert resp1.json()["skipped_sources"] == []

    # Second import: same raw bytes -> should be skipped
    resp2 = client.post("/import")
    assert resp2.status_code == 200
    data = resp2.json()

    assert data["imported_sources"] == [], (
        "Re-import of unchanged file must not appear in imported_sources"
    )
    assert len(data["skipped_sources"]) == 1, (
        "Re-import of unchanged file must produce exactly one skipped_sources entry"
    )

    skipped = data["skipped_sources"][0]
    assert skipped["status"] == "skipped", (
        f"Skipped entry must have status='skipped', got: {skipped['status']}"
    )
    assert skipped["raw_path"].endswith("article.html")
    assert skipped["docs_path"].endswith("article.md")
    assert skipped["original_format"] == "html"


def test_reimport_unchanged_file_has_content_sha256_in_skipped(import_env):
    """Skipped entry carries the content_sha256 that caused the skip."""
    import hashlib

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    raw_bytes = b"<h1>Hello</h1><p>Content here.</p>"
    raw_file = raw_dir / "article.html"
    raw_file.write_bytes(raw_bytes)

    client.post("/import")  # first import

    resp = client.post("/import")  # second import — expect skip
    data = resp.json()
    skipped = data["skipped_sources"][0]

    expected_sha = hashlib.sha256(raw_bytes).hexdigest()
    assert skipped["content_sha256"] == expected_sha, (
        f"content_sha256 in skipped entry must match SHA-256 of raw bytes: "
        f"expected {expected_sha}, got {skipped['content_sha256']}"
    )


# ---------------------------------------------------------------------------
# Hash skip — no disk write
# ---------------------------------------------------------------------------


def test_reimport_unchanged_file_does_not_overwrite_docs(import_env):
    """Re-import of unchanged file: docs/<basename>.md mtime is unchanged."""
    import time

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    raw_file = raw_dir / "article.html"
    raw_file.write_text("<h1>Hello</h1><p>Content.</p>", encoding="utf-8")

    client.post("/import")  # first import
    docs_file = docs_dir / "article.md"
    mtime_after_first = docs_file.stat().st_mtime

    # Small pause to ensure mtime would differ if file was rewritten
    time.sleep(0.05)

    client.post("/import")  # second import — must not write
    mtime_after_second = docs_file.stat().st_mtime

    assert mtime_after_first == mtime_after_second, (
        "Docs file mtime must be unchanged after hash-skip re-import — no disk write expected"
    )


# ---------------------------------------------------------------------------
# Hash skip — Wiki Log import_skipped event
# ---------------------------------------------------------------------------


def test_reimport_unchanged_file_emits_import_skipped_log_event(import_env):
    """Re-import no-op emits import_skipped Wiki Log event."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    raw_file = raw_dir / "article.html"
    raw_file.write_text("<h1>Hello</h1><p>Content.</p>", encoding="utf-8")

    client.post("/import")  # first import
    client.post("/import")  # second import — expect import_skipped event

    events = _parse_log(log_path)
    kinds = [k for k, _ in events]
    assert "import_skipped" in kinds, (
        f"import_skipped event must be in Wiki Log after hash-skip re-import, got: {kinds}"
    )


def test_import_skipped_log_event_has_expected_payload(import_env):
    """import_skipped summary includes raw=, docs=, content_sha256= fields."""
    import hashlib

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    raw_bytes = b"<h1>Hello</h1><p>Specific content.</p>"
    raw_file = raw_dir / "article.html"
    raw_file.write_bytes(raw_bytes)

    client.post("/import")  # first import
    client.post("/import")  # second import — expect import_skipped

    events = _parse_log(log_path)
    skipped_events = [(k, s) for k, s in events if k == "import_skipped"]

    assert len(skipped_events) == 1, f"Expected one import_skipped event, got: {skipped_events}"
    summary = skipped_events[0][1]

    expected_sha = hashlib.sha256(raw_bytes).hexdigest()
    assert "raw=" in summary, f"import_skipped summary must contain raw=, got: {summary}"
    assert "docs=" in summary, f"import_skipped summary must contain docs=, got: {summary}"
    assert expected_sha in summary, (
        f"import_skipped summary must contain content_sha256 hex, got: {summary}"
    )


# ---------------------------------------------------------------------------
# Hash computed over raw bytes (not text)
# ---------------------------------------------------------------------------


def test_content_sha256_is_computed_from_raw_bytes(import_env):
    """content_sha256 in docs frontmatter matches SHA-256 of raw bytes (not text)."""
    import hashlib

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    raw_bytes = b"<h1>Hello</h1><p>Byte-level content here.</p>"
    raw_file = raw_dir / "bytecontent.html"
    raw_file.write_bytes(raw_bytes)

    client.post("/import")

    docs_file = docs_dir / "bytecontent.md"
    content = docs_file.read_text(encoding="utf-8")
    end = content.index("---\n", 4)
    fm = yaml.safe_load(content[4:end])

    expected_sha = hashlib.sha256(raw_bytes).hexdigest()
    assert "content_sha256" in fm, "content_sha256 must appear in docs frontmatter after 7-3"
    assert fm["content_sha256"] == expected_sha, (
        f"content_sha256 must be SHA-256 of raw bytes: expected {expected_sha}, got {fm['content_sha256']}"
    )


# ---------------------------------------------------------------------------
# First import: status='created' and content_sha256 in frontmatter
# ---------------------------------------------------------------------------


def test_first_import_status_created_and_sha256_in_frontmatter(import_env):
    """First import of new file: status='created' and content_sha256 written to frontmatter."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    raw_file = raw_dir / "newfile.html"
    raw_file.write_text("<h1>New</h1><p>Brand new content.</p>", encoding="utf-8")

    resp = client.post("/import")
    data = resp.json()

    assert len(data["imported_sources"]) == 1
    result = data["imported_sources"][0]
    assert result["status"] == "created"
    assert result["content_sha256"] != "", "content_sha256 must not be empty in ImportSourceResult"

    # Also verify it's in the written frontmatter
    docs_file = docs_dir / "newfile.md"
    content = docs_file.read_text(encoding="utf-8")
    end = content.index("---\n", 4)
    fm = yaml.safe_load(content[4:end])
    assert "content_sha256" in fm
    assert fm["content_sha256"] == result["content_sha256"]


# ---------------------------------------------------------------------------
# Hash skip — no conversion side-effects
# ---------------------------------------------------------------------------


def test_reimport_unchanged_file_does_not_invoke_markdownify(import_env, monkeypatch):
    """Hash-match skip must happen BEFORE markdownify is called (no conversion side-effects)."""
    import app.importer as importer_module

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    raw_file = raw_dir / "article.html"
    raw_file.write_text("<h1>Hello</h1><p>Content.</p>", encoding="utf-8")

    client.post("/import")  # first import

    # Track whether _convert_to_markdown is called on second import
    conversion_calls: list[str] = []
    original_convert = importer_module._convert_to_markdown

    def spy_convert(raw_text: str, fmt: str) -> str:
        conversion_calls.append(fmt)
        return original_convert(raw_text, fmt)

    monkeypatch.setattr(importer_module, "_convert_to_markdown", spy_convert)

    client.post("/import")  # second import — hash-match, no conversion expected

    assert conversion_calls == [], (
        f"_convert_to_markdown must NOT be called on hash-match skip; "
        f"got calls for formats: {conversion_calls}"
    )


# ---------------------------------------------------------------------------
# Batch with mixed: one unchanged, one new
# ---------------------------------------------------------------------------


def test_batch_mixed_skip_and_create(import_env):
    """Batch import with one unchanged (skip) and one new (create) produces correct partitioning."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    # Place two files
    (raw_dir / "existing.html").write_text("<h1>Existing</h1><p>Old content.</p>", encoding="utf-8")
    (raw_dir / "brandnew.html").write_text(
        "<h1>Brand New</h1><p>Fresh content.</p>", encoding="utf-8"
    )

    # First import: both should be created
    resp1 = client.post("/import")
    assert len(resp1.json()["imported_sources"]) == 2
    assert resp1.json()["skipped_sources"] == []

    # Remove brandnew to simulate only existing.html remains, re-add both but existing unchanged
    (raw_dir / "brandnew2.html").write_text(
        "<h1>Brand New 2</h1><p>Another new file.</p>", encoding="utf-8"
    )

    # Second import: existing.html unchanged (skip), brandnew2.html new (create)
    resp2 = client.post("/import")
    data2 = resp2.json()

    # existing.html and brandnew.html should be skipped; brandnew2.html should be created
    skipped_paths = [r["raw_path"] for r in data2["skipped_sources"]]
    imported_paths = [r["raw_path"] for r in data2["imported_sources"]]

    assert any("existing.html" in p for p in skipped_paths), (
        f"existing.html must be in skipped_sources; skipped: {skipped_paths}"
    )
    assert any("brandnew2.html" in p for p in imported_paths), (
        f"brandnew2.html must be in imported_sources; imported: {imported_paths}"
    )
