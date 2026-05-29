"""Regression tests: the committed raw/README.md inbox marker must not import.

Reported bug: every batch ``POST /import`` reported a spurious
``HandAuthoredCollision`` because ``raw/README.md`` (the committed inbox-marker
doc — the sole gitignore exception in ``raw/``) was globbed as a source and
mapped onto the hand-authored ``docs/README.md``. The console rendered that
failure under the docs path, so it looked like an uploaded ``.md`` had "become"
README.md.

Covers:
  - raw/README.md is excluded from batch collection (no spurious failure)
  - real sources alongside it still import
  - docs/README.md is left untouched
  - a *non-README* hand-authored collision reports the raw source path (not the
    docs path) in ImportFailure.raw_path, so the console points at the
    actionable file.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log path into importer.py for isolation."""
    import app.importer as importer_module
    import app.logger as logger_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    from app.main import app

    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    return {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir}


def test_raw_readme_marker_not_imported(import_env):
    """raw/README.md is the inbox marker — batch import must skip it entirely."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    # The committed inbox-marker doc + the hand-authored docs structure doc.
    (raw_dir / "README.md").write_text("# raw/ — import inbox\n", encoding="utf-8")
    docs_readme = docs_dir / "README.md"
    docs_readme.write_text("# docs/\n\nHand-authored structure doc.\n", encoding="utf-8")

    # A real source dropped alongside the marker.
    (raw_dir / "tetet.txt").write_text("hello", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    # No spurious README failure — the only outcome is the real source importing.
    assert data["failed_sources"] == [], (
        f"raw/README.md must not produce a failure, got: {data['failed_sources']}"
    )
    imported_docs = {Path(r["docs_path"]).name for r in data["imported_sources"]}
    assert imported_docs == {"tetet.md"}

    # The hand-authored docs/README.md is untouched.
    assert docs_readme.read_text(encoding="utf-8") == "# docs/\n\nHand-authored structure doc.\n"


def test_hand_authored_collision_reports_raw_source_path(import_env):
    """A genuine collision names the raw source in raw_path, not the docs target."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    (raw_dir / "account_help.txt").write_text("New content.", encoding="utf-8")
    # Hand-authored docs target (no imported_from frontmatter).
    (docs_dir / "account_help.md").write_text("# Account help\n\nCurated.\n", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["failed_sources"]) == 1
    failure = data["failed_sources"][0]
    assert failure["error_type"] == "HandAuthoredCollision"
    # raw_path points at the source the curator dropped, not the protected docs file.
    assert Path(failure["raw_path"]).name == "account_help.txt"
    assert failure["raw_path"].replace("\\", "/").endswith("raw/account_help.txt")
