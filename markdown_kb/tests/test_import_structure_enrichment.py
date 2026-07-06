"""Integration tests: POST /import wired to Structure Enrichment (ADR-0033
decision 2, issue #512).

AC coverage:
  - Well-headed Sources bypass enrichment byte-identically (no LLM call, no
    frontmatter change).
  - `structure: enriched` frontmatter present when enrichment materializes
    headings.
  - Re-import of an unchanged raw file does not re-enrich or re-bill
    (existing hash-skip check runs BEFORE Structure Enrichment).
  - Enrichment LLM failure fails soft to the un-enriched transcript; import
    still succeeds and the degradation is reported via the Wiki Log.

The enrichment LLM is mocked at the lazy-singleton getter
(``structure_enrichment.get_enrichment_llm``), per CODING_STANDARD §6.3.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

FILLER = "Lorem ipsum filler text about nothing in particular. "


def _long_zero_heading_body(paragraphs: int = 8) -> str:
    body = "\n\n".join(f"Paragraph {i} opens here. " + (FILLER * 6) for i in range(paragraphs))
    assert len(body.strip()) >= 2000
    return body


def _well_headed_body() -> str:
    return "\n\n".join(
        f"## Chapter {i}\n\n" + f"Chapter {i} content. " + (FILLER * 5) for i in range(4)
    )


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir/docs_dir into importer.py for isolation (mirrors sibling fixtures)."""
    import app.importer as importer_module
    import app.logger as logger_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    return {
        "client": client,
        "raw_dir": raw_dir,
        "docs_dir": docs_dir,
        "log_path": logger_module.LOG_PATH,
    }


def _fake_llm_with_chapters(chapters: list[SimpleNamespace]) -> MagicMock:
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = SimpleNamespace(chapters=chapters)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


def _log_kinds(log_path) -> list[str]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [line.split("|")[0].split("] ")[-1].strip() for line in lines if line.startswith("##")]


# ---------------------------------------------------------------------------
# Bypass: well-headed Source, byte-identical, no LLM call
# ---------------------------------------------------------------------------


def test_well_headed_source_bypasses_enrichment(import_env, monkeypatch):
    import app.structure_enrichment as se

    def _boom():
        raise AssertionError("enrichment LLM must not be called for a well-headed Source")

    monkeypatch.setattr(se, "get_enrichment_llm", _boom)

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    body = _well_headed_body()
    (raw_dir / "handbook.md").write_text(body, encoding="utf-8")

    resp = client.post("/import", json={"source": "handbook.md"})
    assert resp.status_code == 200

    content = (docs_dir / "handbook.md").read_text(encoding="utf-8")
    assert "structure: enriched" not in content
    assert body in content


# ---------------------------------------------------------------------------
# Enrichment fires: frontmatter gains structure: enriched
# ---------------------------------------------------------------------------


def test_longform_source_gains_structure_enriched_frontmatter(import_env, monkeypatch):
    import app.structure_enrichment as se

    body = _long_zero_heading_body()
    chapters = [
        SimpleNamespace(title="Part One", boundary_anchor="Paragraph 0 opens here."),
        SimpleNamespace(title="Part Two", boundary_anchor="Paragraph 4 opens here."),
    ]
    fake_llm = _fake_llm_with_chapters(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    (raw_dir / "report.txt").write_text(body, encoding="utf-8")

    resp = client.post("/import", json={"source": "report.txt"})
    assert resp.status_code == 200

    content = (docs_dir / "report.md").read_text(encoding="utf-8")
    assert "structure: enriched" in content
    assert "## Part One" in content
    assert "## Part Two" in content
    assert content.index("## Part One") < content.index("## Part Two")


# ---------------------------------------------------------------------------
# Idempotency: unchanged raw file never re-enriches / re-bills on re-import
# ---------------------------------------------------------------------------


def test_reimport_unchanged_source_does_not_reenrich(import_env, monkeypatch):
    import app.structure_enrichment as se

    body = _long_zero_heading_body()
    chapters = [SimpleNamespace(title="Only Part", boundary_anchor="Paragraph 0 opens here.")]
    fake_llm = _fake_llm_with_chapters(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    (raw_dir / "report.txt").write_text(body, encoding="utf-8")

    resp1 = client.post("/import", json={"source": "report.txt"})
    assert resp1.json()["imported_sources"][0]["status"] == "created"
    assert fake_llm.with_structured_output.return_value.invoke.call_count == 1

    docs_file = docs_dir / "report.md"
    mtime_after_first = docs_file.stat().st_mtime

    resp2 = client.post("/import", json={"source": "report.txt"})
    data2 = resp2.json()
    assert data2["imported_sources"] == []
    assert data2["skipped_sources"][0]["status"] == "skipped"
    assert docs_file.stat().st_mtime == mtime_after_first
    # No second enrichment LLM call on the hash-skip path.
    assert fake_llm.with_structured_output.return_value.invoke.call_count == 1


# ---------------------------------------------------------------------------
# Fail-soft: enrichment LLM error degrades to the un-enriched transcript
# ---------------------------------------------------------------------------


def test_enrichment_failure_fails_soft_and_reports_degradation(import_env, monkeypatch):
    import app.structure_enrichment as se

    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("simulated model failure")
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    body = _long_zero_heading_body()

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    (raw_dir / "report.txt").write_text(body, encoding="utf-8")

    resp = client.post("/import", json={"source": "report.txt"})
    assert resp.status_code == 200
    assert resp.json()["imported_sources"][0]["status"] == "created"

    content = (docs_dir / "report.md").read_text(encoding="utf-8")
    assert "structure: enriched" not in content
    assert body in content, "Un-enriched transcript must still be written on enrichment failure"

    assert "structure_enrichment_failed" in _log_kinds(import_env["log_path"])
