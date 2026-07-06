"""Live integration smoke test for Structure Enrichment (ADR-0033 decision 2, issue #512).

Makes ONE real OpenAI call to confirm the chapter-outline proposal +
heading-materialization pipeline end-to-end via ``POST /import`` on a
synthetic, zero-heading, longform-shaped document. Opt-in only: skipped by
default; run with:

    pytest -m live

Requirements:
    OPENAI_API_KEY must be set in the environment; the test fails with a
    clear message if it is absent rather than silently passing or skipping.

This is the ONE authorised @pytest.mark.live test for the Structure
Enrichment surface (ADR-0005 §"LLM-facing surface enumeration" /
CODING_STANDARD §6.4) — do not add a second.
"""

from __future__ import annotations

import os

import pytest

_PARAGRAPH = (
    "This report examines quarterly performance across every region without "
    "using a single heading anywhere in its text, which is exactly the "
    "degenerate structural shape Structure Enrichment exists to repair."
)


def _long_zero_heading_body(paragraphs: int = 12) -> str:
    body = "\n\n".join(f"{_PARAGRAPH} Paragraph index {i}." for i in range(paragraphs))
    assert len(body.strip()) >= 2000, "fixture must clear the KB_LONGFORM_MIN_CHARS floor"
    return body


@pytest.mark.live
def test_structure_enrichment_live_via_import(tmp_path, monkeypatch):
    """POST /import on a long, zero-heading .txt Source triggers real enrichment.

    Assertions are SHAPE-only (CODING_STANDARD §6.2/§6.4): HTTP 200,
    `structure: enriched` frontmatter present, and at least two materialized
    `## ` chapter headings — never specific chapter titles or wording, since
    models update and tests must outlive them.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail(
            "OPENAI_API_KEY is not set. "
            "Export your key before running live tests: "
            "export OPENAI_API_KEY=sk-..."
        )

    import app.importer as importer_module
    import app.logger as logger_module
    import app.structure_enrichment as enrichment_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")
    # Reset the cached LLM singleton so we get a fresh real ChatOpenAI instance
    # with the current OPENAI_API_KEY from the environment.
    monkeypatch.setattr(enrichment_module, "_enrichment_llm", None)

    (raw_dir / "quarterly_report.txt").write_text(_long_zero_heading_body(), encoding="utf-8")

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post("/import", json={"source": "quarterly_report.txt"})

    assert resp.status_code == 200, f"Expected HTTP 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert len(data["imported_sources"]) == 1

    content = (docs_dir / "quarterly_report.md").read_text(encoding="utf-8")
    assert "structure: enriched" in content, (
        f"Expected 'structure: enriched' frontmatter after live enrichment, got:\n{content}"
    )
    heading_count = content.count("\n## ")
    assert heading_count >= 2, f"Expected >=2 materialized chapter headings, got {heading_count}"
