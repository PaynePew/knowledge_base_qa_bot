"""Offline unit tests for the Acme-Shop corpus generator (AC4, issue #143).

Tests cover the deterministic seams of the generator:
- DOC_SPECS plan integrity (basename uniqueness, section counts, entity flags)
- Gold Section eligibility count (≥50 from the plan and from the committed corpus)
- Scaffold generation (deterministic output, valid Markdown structure)
- derive_gold_sections integration on scaffold output
- Live generation path is not invoked here (opt-in: @pytest.mark.live)

All tests are offline and deterministic — no OPENAI_API_KEY or network I/O.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eval.paraphrase_comparison.generation.corpus_generator import (
    CORPUS_ENTITY_SOURCES,
    DOC_SPECS,
    FAKE_DOCS_DIR,
    DocSpec,
    SectionSpec,
    generate_scaffold,
)
from eval.paraphrase_comparison.generation.sampling import (
    derive_gold_sections,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PLAN_BASENAMES: set[str] = {d.basename for d in DOC_SPECS}
_PLAN_CONCEPT_DOCS: tuple[DocSpec, ...] = tuple(d for d in DOC_SPECS if not d.is_entity)


@pytest.fixture()
def scaffold_dir(tmp_path: Path) -> Path:
    """A directory populated with scaffold stubs for all DOC_SPECS."""
    for doc in DOC_SPECS:
        generate_scaffold(doc, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# AC1 — generator script exists and is re-runnable (import test)
# ---------------------------------------------------------------------------


def test_corpus_generator_module_imports() -> None:
    """The corpus_generator module imports without errors (no LLM calls needed)."""
    # If import failed, the test-collection phase would already fail; this
    # assertion is a belt-and-suspenders signal that's easy to grep.
    assert DOC_SPECS is not None
    assert len(DOC_SPECS) > 0


def test_generate_scaffold_is_callable() -> None:
    """generate_scaffold is a callable that accepts a DocSpec and a Path."""
    import inspect

    sig = inspect.signature(generate_scaffold)
    params = list(sig.parameters)
    assert "doc" in params
    assert "output_dir" in params


# ---------------------------------------------------------------------------
# AC2 — ~50 Gold Sections from the plan and from the committed corpus
# ---------------------------------------------------------------------------


def test_doc_specs_yield_at_least_50_gold_sections() -> None:
    """DOC_SPECS defines ≥50 Gold-eligible sections (concept docs only)."""
    total = sum(d.gold_section_count for d in DOC_SPECS)
    assert total >= 50, (
        f"DOC_SPECS must define ≥50 Gold-eligible sections, got {total}. "
        "Add more concept docs or sections to reach the ~50 target."
    )


def test_committed_corpus_yields_at_least_50_gold_sections() -> None:
    """derive_gold_sections on docs/fake-docs/ returns ≥50 Gold Sections.

    This uses the sampling module's default entity_sources (``warranty.md`` only),
    which is the perspective the eval pipeline uses when deriving the gold inventory.
    """
    sections = derive_gold_sections(FAKE_DOCS_DIR)
    assert len(sections) >= 50, (
        f"docs/fake-docs/ must yield ≥50 Gold Sections for the comparison "
        f"(sampling.CORPUS_ENTITY_SOURCES excluded), got {len(sections)}."
    )


def test_scaffold_dir_yields_at_least_50_gold_sections(scaffold_dir: Path) -> None:
    """scaffold output: derive_gold_sections picks up ≥50 sections (structural test)."""
    # Use the generator's CORPUS_ENTITY_SOURCES so entity docs are excluded.
    sections = derive_gold_sections(scaffold_dir, entity_sources=CORPUS_ENTITY_SOURCES)
    total = sum(d.gold_section_count for d in DOC_SPECS)
    assert len(sections) >= 50, (
        f"Scaffold output must yield ≥50 Gold Sections, got {len(sections)}. "
        f"DOC_SPECS plans {total} concept sections."
    )


# ---------------------------------------------------------------------------
# AC3 — globally-unique basenames
# ---------------------------------------------------------------------------


def test_doc_spec_basenames_are_globally_unique() -> None:
    """Every basename in DOC_SPECS is unique — no two docs share a basename."""
    basenames = [d.basename for d in DOC_SPECS]
    seen: set[str] = set()
    duplicates: list[str] = []
    for b in basenames:
        if b in seen:
            duplicates.append(b)
        seen.add(b)
    assert not duplicates, f"Duplicate basenames in DOC_SPECS: {duplicates}"


def test_doc_spec_basenames_end_with_md() -> None:
    """Every basename has a .md extension (enforced by the importer contract)."""
    bad = [d.basename for d in DOC_SPECS if not d.basename.endswith(".md")]
    assert not bad, f"Non-.md basenames: {bad}"


def test_doc_spec_basenames_have_no_forbidden_chars() -> None:
    """Basenames must not contain '/', '#', or ':' (Section-id contract).

    The importer rejects filenames with these characters; the generator must
    not produce them.
    """
    bad = [
        d.basename for d in DOC_SPECS if any(c in d.basename for c in ("/", "#", ":"))
    ]
    assert not bad, f"Basenames with forbidden chars (/, #, :): {bad}"


def test_doc_specs_cover_committed_corpus() -> None:
    """Every committed file in docs/fake-docs/ is named in DOC_SPECS.

    This catches the case where a new file was added to the corpus without
    updating the generator plan.
    """
    committed = {f.name for f in FAKE_DOCS_DIR.glob("*.md")}
    not_in_plan = committed - _PLAN_BASENAMES
    assert not not_in_plan, (
        f"Files in docs/fake-docs/ not in DOC_SPECS: {sorted(not_in_plan)}. "
        "Add them to DOC_SPECS in corpus_generator.py."
    )


def test_doc_specs_do_not_introduce_unknown_basenames() -> None:
    """Every basename in DOC_SPECS has a corresponding committed file.

    This is the complementary check: the plan must not list files that don't
    exist in the committed corpus (they must be committed or removed from the plan).
    """
    committed = {f.name for f in FAKE_DOCS_DIR.glob("*.md")}
    not_committed = _PLAN_BASENAMES - committed
    assert not not_committed, (
        f"Basenames in DOC_SPECS but not committed in docs/fake-docs/: "
        f"{sorted(not_committed)}. Either commit the file or remove it from DOC_SPECS."
    )


# ---------------------------------------------------------------------------
# AC4 — deterministic scaffolding is unit-tested offline
# ---------------------------------------------------------------------------


def test_generate_scaffold_produces_valid_markdown(tmp_path: Path) -> None:
    """generate_scaffold writes a syntactically valid Markdown file with H1 + H2s."""
    doc = DocSpec(
        basename="test_fixture.md",
        title="Test Fixture",
        sections=(
            SectionSpec("Section Alpha", "Alpha content hint."),
            SectionSpec("Section Beta", "Beta content hint."),
        ),
    )
    dest = generate_scaffold(doc, tmp_path)
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")

    # H1 must match the title.
    assert "# Test Fixture" in content
    # Each H2 must appear.
    assert "## Section Alpha" in content
    assert "## Section Beta" in content
    # No H1 duplication.
    h1_lines = [ln for ln in content.splitlines() if re.match(r"^# ", ln)]
    assert len(h1_lines) == 1, f"Expected exactly 1 H1, got: {h1_lines}"


def test_generate_scaffold_is_deterministic(tmp_path: Path) -> None:
    """Calling generate_scaffold twice with the same DocSpec produces identical output."""
    doc = DOC_SPECS[2]  # First concept doc (account_management)
    first = generate_scaffold(doc, tmp_path / "run1")
    second = generate_scaffold(doc, tmp_path / "run2")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_generate_scaffold_is_idempotent(tmp_path: Path) -> None:
    """Calling generate_scaffold twice in the same dir overwrites cleanly."""
    doc = DOC_SPECS[2]
    generate_scaffold(doc, tmp_path)
    generate_scaffold(doc, tmp_path)
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1, f"Expected 1 file, got {files}"


def test_generate_scaffold_creates_output_dir(tmp_path: Path) -> None:
    """generate_scaffold creates the output directory if it does not exist."""
    new_dir = tmp_path / "nested" / "output"
    assert not new_dir.exists()
    doc = DOC_SPECS[2]
    generate_scaffold(doc, new_dir)
    assert new_dir.is_dir()
    assert (new_dir / doc.basename).exists()


def test_generate_scaffold_body_derived_from_hint(tmp_path: Path) -> None:
    """Scaffold body contains text derived from the prompt_hint (deterministic)."""
    hint = "Unique hint text for testing purposes XYZ."
    doc = DocSpec(
        basename="hint_test.md",
        title="Hint Test",
        sections=(SectionSpec("Alpha Section", hint),),
    )
    dest = generate_scaffold(doc, tmp_path)
    content = dest.read_text(encoding="utf-8")
    assert "Unique hint text for testing purposes XYZ" in content


def test_generate_scaffold_each_section_appears(tmp_path: Path) -> None:
    """All sections in a DocSpec appear as ## headings in the scaffold output."""
    doc = DOC_SPECS[3]  # account_security: 3 sections
    dest = generate_scaffold(doc, tmp_path)
    content = dest.read_text(encoding="utf-8")
    for section in doc.sections:
        assert f"## {section.heading}" in content, (
            f"Section heading '## {section.heading}' missing from scaffold."
        )


def test_scaffold_all_docs_no_basename_collision(scaffold_dir: Path) -> None:
    """Generating all DOC_SPECS into the same dir produces exactly len(DOC_SPECS) files."""
    files = {f.name for f in scaffold_dir.glob("*.md")}
    assert files == _PLAN_BASENAMES, (
        f"Expected {len(_PLAN_BASENAMES)} distinct files, got {len(files)}."
    )


def test_scaffold_sections_parseable_by_derive_gold_sections(
    scaffold_dir: Path,
) -> None:
    """derive_gold_sections can parse every scaffolded concept doc without error."""
    sections = derive_gold_sections(scaffold_dir, entity_sources=CORPUS_ENTITY_SOURCES)
    section_ids = {s.section_id for s in sections}
    for doc in _PLAN_CONCEPT_DOCS:
        for spec in doc.sections:
            slug = re.sub(r"[^a-z0-9]+", "-", spec.heading.lower()).strip("-")
            expected_id = f"{doc.basename}#{slug}"
            assert expected_id in section_ids, (
                f"derive_gold_sections did not find section id {expected_id!r}. "
                f"Check heading slugification in corpus_generator._slug."
            )


def test_doc_spec_concept_docs_have_at_least_2_sections() -> None:
    """Every concept DocSpec has ≥2 sections (minimum for meaningful Gold coverage)."""
    short = [d.basename for d in _PLAN_CONCEPT_DOCS if len(d.sections) < 2]
    assert not short, f"Concept docs with <2 sections: {short}"


def test_entity_sources_constant_matches_corpus_generator() -> None:
    """CORPUS_ENTITY_SOURCES in corpus_generator includes the known entity basenames."""
    assert "warranty.md" in CORPUS_ENTITY_SOURCES
    assert "acme_shop_about.md" in CORPUS_ENTITY_SOURCES


def test_doc_spec_entity_flag_matches_corpus_entity_sources() -> None:
    """Every DocSpec with is_entity=True has its basename in CORPUS_ENTITY_SOURCES."""
    entity_docs = {d.basename for d in DOC_SPECS if d.is_entity}
    assert entity_docs == CORPUS_ENTITY_SOURCES, (
        f"Mismatch: is_entity basenames={entity_docs}, "
        f"CORPUS_ENTITY_SOURCES={CORPUS_ENTITY_SOURCES}. "
        "Keep them in sync."
    )
