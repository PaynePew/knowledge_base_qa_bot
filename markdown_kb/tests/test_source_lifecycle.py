"""Hermetic unit tests for the Source lifecycle deep module
(markdown_kb.app.source_lifecycle) — issue #604 (S1), ADR-0041.

AC coverage:
  - compute_impact: full-orphan / partial-orphan / no-citation classification;
    InvalidRelpath / SourceNotFound refusals.
  - retire: one atomic move docs/<relpath> -> trash/<ts>/docs/<relpath>;
    Source bytes untouched; impact returned; source_retired logged.
  - restore: the atomic inverse; both refusal guards (RestoreTargetOccupied,
    RestoreBasenameCollision); source_restored logged.
  - list_trash: entries reflect the physical trash tree.
  - Resurrection regression (ADR-0041 code fact 1): a retired Source is
    invisible to ingest pairing, lint C3 resolution, and upload origin
    resolution — the trash lives OUTSIDE docs/, so every docs_dir.glob("**/…")
    scanner is structurally blind to it.
  - End-to-end: retire -> next lint reports a C11 full orphan; restore ->
    a stale C11 delete click 409s (ADR-0025 at-delete recompute).
  - read.py whitelist: a retired Source is readable via the .trash root.

No OPENAI_API_KEY required — this module is pure filesystem I/O; the
grounding verifier is mocked by the suite's autouse
``_mock_ingest_verifier_supported`` fixture wherever ingest/lint run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.ingest as ingest_module
import app.lint as lint_module
import app.pages as pages_module
import app.read as read_module
import app.source_lifecycle as sl
import app.upload as upload_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _page_text(slug: str, sources: list[str]) -> str:
    sources_block = "".join(f"  - {s}\n" for s in sources) if sources else "  []\n"
    return (
        "---\n"
        f"id: {slug}\n"
        "type: concept\n"
        "created: '2026-07-12T00:00:00Z'\n"
        "updated: '2026-07-12T00:00:00Z'\n"
        f"sources:\n{sources_block}"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes: {}\n"
        "---\n\n"
        f"# {slug}\n\nSome content.\n"
    )


@pytest.fixture()
def lifecycle_env(tmp_path: Path):
    """docs/, wiki/concepts/, .trash/ under one tmp_path, plus one Source and
    one citing full-orphan-shaped wiki page."""
    docs = tmp_path / "docs"
    wiki = tmp_path / "wiki"
    trash = tmp_path / ".trash"
    (wiki / "concepts").mkdir(parents=True)
    docs.mkdir(parents=True)

    (docs / "policy.md").write_text("# Policy\n\nRefunds take 7 days.\n", encoding="utf-8")
    (docs / "other.md").write_text("# Other\n", encoding="utf-8")

    (wiki / "concepts" / "full-orphan-candidate.md").write_text(
        _page_text("full-orphan-candidate", ["policy.md#refunds"]),
        encoding="utf-8",
    )
    (wiki / "concepts" / "partial-orphan-candidate.md").write_text(
        _page_text("partial-orphan-candidate", ["policy.md#refunds", "other.md#s"]),
        encoding="utf-8",
    )
    (wiki / "concepts" / "unrelated.md").write_text(
        _page_text("unrelated", ["other.md#s"]),
        encoding="utf-8",
    )

    return {"docs": docs, "wiki": wiki, "trash": trash, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# compute_impact
# ---------------------------------------------------------------------------


def test_compute_impact_classifies_full_and_partial_orphans(lifecycle_env):
    preview = sl.compute_impact(
        "policy.md", docs_dir=lifecycle_env["docs"], wiki_dir=lifecycle_env["wiki"]
    )
    assert preview.relpath == "policy.md"
    assert preview.full_orphans == ["full-orphan-candidate"]
    assert preview.partial_orphans == ["partial-orphan-candidate"]
    assert "unrelated" not in preview.full_orphans
    assert "unrelated" not in preview.partial_orphans


def test_compute_impact_missing_source_raises(lifecycle_env):
    with pytest.raises(sl.SourceNotFound):
        sl.compute_impact(
            "ghost.md", docs_dir=lifecycle_env["docs"], wiki_dir=lifecycle_env["wiki"]
        )


@pytest.mark.parametrize(
    "bad_relpath",
    [
        "../escape.md",
        "/etc/passwd",
        "a/../../b.md",
        "",
        "D:evil.md",  # Windows drive-relative escape (issue #397 landmine)
        "D:sub/evil.md",
    ],
)
def test_compute_impact_unsafe_relpath_raises(lifecycle_env, bad_relpath):
    with pytest.raises(sl.InvalidRelpath):
        sl.compute_impact(
            bad_relpath, docs_dir=lifecycle_env["docs"], wiki_dir=lifecycle_env["wiki"]
        )


def test_retire_rejects_windows_drive_relative_relpath_before_touching_disk(lifecycle_env):
    """issue #397 landmine: ``docs_dir / "D:evil.md"`` resolves to
    ``D:evil.md`` on Windows — a colon anywhere in relpath must be refused
    BEFORE any filesystem access, mirroring ``slugs.is_bare_slug``'s own
    unconditional ':' ban."""
    with pytest.raises(sl.InvalidRelpath):
        sl.retire(
            "D:evil.md",
            docs_dir=lifecycle_env["docs"],
            wiki_dir=lifecycle_env["wiki"],
            trash_dir=lifecycle_env["trash"],
        )
    assert not lifecycle_env["trash"].exists()


def test_compute_impact_nested_relpath_resolves(tmp_path):
    docs = tmp_path / "docs"
    wiki = tmp_path / "wiki" / "concepts"
    (docs / "sub").mkdir(parents=True)
    wiki.mkdir(parents=True)
    (docs / "sub" / "nested.md").write_text("# Nested\n", encoding="utf-8")

    preview = sl.compute_impact("sub/nested.md", docs_dir=docs, wiki_dir=tmp_path / "wiki")
    assert preview.relpath == "sub/nested.md"
    assert preview.full_orphans == []
    assert preview.partial_orphans == []


# ---------------------------------------------------------------------------
# retire
# ---------------------------------------------------------------------------


def test_retire_moves_file_and_preserves_bytes(lifecycle_env):
    original_bytes = (lifecycle_env["docs"] / "policy.md").read_bytes()

    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    assert not (lifecycle_env["docs"] / "policy.md").exists()
    trashed = lifecycle_env["trash"] / result.timestamp / "docs" / "policy.md"
    assert trashed.exists()
    assert trashed.read_bytes() == original_bytes
    assert result.relpath == "policy.md"
    assert result.impact.full_orphans == ["full-orphan-candidate"]


def test_retire_logs_source_retired(lifecycle_env, monkeypatch):
    log_path = lifecycle_env["tmp_path"] / "log.md"
    import app.logger as logger_module

    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "source_retired" in log_text
    assert "relpath=policy.md" in log_text
    assert f"timestamp={result.timestamp}" in log_text
    assert "full_orphans=1" in log_text
    # "partial-orphan-candidate" also cites policy.md (plus other.md), so
    # retiring policy.md makes it a real partial orphan too.
    assert "partial_orphans=1" in log_text


def test_retire_missing_source_raises_and_writes_nothing(lifecycle_env):
    with pytest.raises(sl.SourceNotFound):
        sl.retire(
            "ghost.md",
            docs_dir=lifecycle_env["docs"],
            wiki_dir=lifecycle_env["wiki"],
            trash_dir=lifecycle_env["trash"],
        )
    assert not lifecycle_env["trash"].exists()


def test_retire_unsafe_relpath_raises_and_writes_nothing(lifecycle_env):
    with pytest.raises(sl.InvalidRelpath):
        sl.retire(
            "../escape.md",
            docs_dir=lifecycle_env["docs"],
            wiki_dir=lifecycle_env["wiki"],
            trash_dir=lifecycle_env["trash"],
        )
    assert not lifecycle_env["trash"].exists()


def test_retire_nested_relpath_preserves_subdirectory_shape(tmp_path):
    docs = tmp_path / "docs"
    wiki = tmp_path / "wiki" / "concepts"
    trash = tmp_path / ".trash"
    (docs / "demo-zh").mkdir(parents=True)
    wiki.mkdir(parents=True)
    (docs / "demo-zh" / "policy.md").write_text("# 政策\n", encoding="utf-8")

    result = sl.retire(
        "demo-zh/policy.md", docs_dir=docs, wiki_dir=tmp_path / "wiki", trash_dir=trash
    )

    assert (trash / result.timestamp / "docs" / "demo-zh" / "policy.md").exists()
    assert not (docs / "demo-zh" / "policy.md").exists()


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def test_restore_moves_file_back(lifecycle_env):
    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    sl.restore(
        result.timestamp,
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        trash_dir=lifecycle_env["trash"],
    )

    assert (lifecycle_env["docs"] / "policy.md").exists()
    assert (lifecycle_env["docs"] / "policy.md").read_text(encoding="utf-8").startswith("# Policy")
    assert not (lifecycle_env["trash"] / result.timestamp / "docs" / "policy.md").exists()


def test_restore_logs_source_restored(lifecycle_env, monkeypatch):
    log_path = lifecycle_env["tmp_path"] / "log.md"
    import app.logger as logger_module

    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )
    sl.restore(
        result.timestamp,
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        trash_dir=lifecycle_env["trash"],
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "source_restored" in log_text
    assert "relpath=policy.md" in log_text


def test_restore_missing_entry_raises(lifecycle_env):
    with pytest.raises(sl.TrashEntryNotFound):
        sl.restore(
            "20260101T000000000000Z",
            "policy.md",
            docs_dir=lifecycle_env["docs"],
            trash_dir=lifecycle_env["trash"],
        )


def test_restore_rejects_windows_drive_relative_timestamp(lifecycle_env):
    """The same issue #397 colon guard applies to the ``timestamp`` component."""
    with pytest.raises(sl.InvalidRelpath):
        sl.restore(
            "D:evil",
            "policy.md",
            docs_dir=lifecycle_env["docs"],
            trash_dir=lifecycle_env["trash"],
        )


def test_restore_refuses_when_target_occupied(lifecycle_env):
    """Guard 1 (ADR-0041 decision 4): a same-name Source was uploaded since retire."""
    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )
    (lifecycle_env["docs"] / "policy.md").write_text("# New Policy\n", encoding="utf-8")

    with pytest.raises(sl.RestoreTargetOccupied):
        sl.restore(
            result.timestamp,
            "policy.md",
            docs_dir=lifecycle_env["docs"],
            trash_dir=lifecycle_env["trash"],
        )
    # nothing moved
    assert (lifecycle_env["docs"] / "policy.md").read_text(encoding="utf-8") == "# New Policy\n"
    assert (lifecycle_env["trash"] / result.timestamp / "docs" / "policy.md").exists()


def test_restore_refuses_on_basename_collision_elsewhere(lifecycle_env):
    """Guard 2 (ADR-0041 decision 4): the basename now exists elsewhere under
    docs/ — restoring would mint an ambiguous_source state."""
    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )
    (lifecycle_env["docs"] / "elsewhere").mkdir()
    (lifecycle_env["docs"] / "elsewhere" / "policy.md").write_text(
        "# Elsewhere\n", encoding="utf-8"
    )

    with pytest.raises(sl.RestoreBasenameCollision):
        sl.restore(
            result.timestamp,
            "policy.md",
            docs_dir=lifecycle_env["docs"],
            trash_dir=lifecycle_env["trash"],
        )
    assert not (lifecycle_env["docs"] / "policy.md").exists()
    assert (lifecycle_env["trash"] / result.timestamp / "docs" / "policy.md").exists()


# ---------------------------------------------------------------------------
# list_trash
# ---------------------------------------------------------------------------


def test_list_trash_empty_when_never_retired(lifecycle_env):
    assert sl.list_trash(trash_dir=lifecycle_env["trash"]) == []


def test_list_trash_reflects_physical_tree(lifecycle_env):
    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    entries = sl.list_trash(trash_dir=lifecycle_env["trash"])
    assert entries == [sl.TrashEntry(timestamp=result.timestamp, relpath="policy.md")]

    sl.restore(
        result.timestamp,
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        trash_dir=lifecycle_env["trash"],
    )
    assert sl.list_trash(trash_dir=lifecycle_env["trash"]) == []


# ---------------------------------------------------------------------------
# Resurrection regression (ADR-0041 code fact 1) — a retired Source must be
# invisible to every docs_dir.glob("**/…") scanner.
# ---------------------------------------------------------------------------


def test_retired_source_invisible_to_ingest_pairing(lifecycle_env):
    sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    pairs, ambiguous = ingest_module._resolve_single_source_pairs(
        ["policy.md"], lifecycle_env["docs"]
    )
    assert ambiguous == []
    # Zero matches under docs/: falls through to the flat not-found path
    # (ingest_sources's own "source_not_found" rung), never the trashed copy.
    assert pairs == [("policy.md", lifecycle_env["docs"] / "policy.md")]
    assert not pairs[0][1].exists()


def test_retired_source_invisible_to_lint_c3_resolution(lifecycle_env):
    sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    path, resolution = lint_module._resolve_c3_source_path(
        "policy.md#refunds", lifecycle_env["docs"]
    )
    assert resolution == "missing"
    assert path is None


def test_retired_source_invisible_to_upload_origin_resolution(lifecycle_env):
    """A same-basename re-upload resolves to exactly ONE match (itself) — the
    trashed copy must not make this ambiguous."""
    sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    batch = upload_module.upload_files(
        [("policy.md", b"# Replacement Policy\n")],
        raw_dir=lifecycle_env["tmp_path"] / "raw",
        docs_dir=lifecycle_env["docs"],
    )
    assert batch.results[0].status == "written"

    target, refusal = upload_module._resolve_overwrite_target(
        "policy.md", ".md", lifecycle_env["docs"], "docs/policy.md"
    )
    assert refusal == ""
    assert target == lifecycle_env["docs"] / "policy.md"


# ---------------------------------------------------------------------------
# End-to-end with lint / pages (ADR-0025 at-delete recompute)
# ---------------------------------------------------------------------------


def test_retire_then_lint_reports_c11_full_orphan(lifecycle_env, monkeypatch):
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", lifecycle_env["wiki"])
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [
            lifecycle_env["wiki"] / "entities",
            lifecycle_env["wiki"] / "concepts",
            lifecycle_env["wiki"] / "qa",
        ],
    )

    sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    report = lint_module.run_lint(
        wiki_dir=lifecycle_env["wiki"],
        docs_dir=lifecycle_env["docs"],
        log_path=lifecycle_env["tmp_path"] / "log.md",
    )

    full_orphan_slugs = {f.page_slug for f in report.findings.orphans if f.full}
    assert "full-orphan-candidate" in full_orphan_slugs
    assert "partial-orphan-candidate" not in full_orphan_slugs


def test_restore_makes_a_stale_delete_click_409(lifecycle_env, monkeypatch):
    """ADR-0025 at-delete recompute: a curator sees the C11 full-orphan
    finding, retires happens, curator restores the Source before clicking
    delete — the server re-verifies NOW and refuses (409), never trusting
    the (now stale) finding."""
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", lifecycle_env["wiki"])
    monkeypatch.setattr(pages_module, "DOCS_DIR", lifecycle_env["docs"])

    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )
    # The curator restores the Source before acting on the (now stale) finding.
    sl.restore(
        result.timestamp,
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        trash_dir=lifecycle_env["trash"],
    )

    with pytest.raises(pages_module.PageNotFullOrphan):
        pages_module.delete_full_orphan("full-orphan-candidate")


# ---------------------------------------------------------------------------
# read.py whitelist — pre-restore inspection of a retired Source (ADR-0041)
# ---------------------------------------------------------------------------


def test_trashed_file_readable_via_whitelist(lifecycle_env):
    result = sl.retire(
        "policy.md",
        docs_dir=lifecycle_env["docs"],
        wiki_dir=lifecycle_env["wiki"],
        trash_dir=lifecycle_env["trash"],
    )

    roots = {
        "docs": lifecycle_env["docs"],
        "raw": lifecycle_env["tmp_path"] / "raw",
        "wiki": lifecycle_env["wiki"],
        ".trash": lifecycle_env["trash"],
    }
    relpath = f".trash/{result.timestamp}/docs/policy.md"
    content = read_module.read_file(relpath, roots=roots)
    assert content.startswith("# Policy")


def test_kb_dir_still_unreachable_alongside_trash_root(lifecycle_env):
    """Adding .trash to the whitelist must not loosen the existing .kb/ exclusion."""
    roots = {
        "docs": lifecycle_env["docs"],
        "raw": lifecycle_env["tmp_path"] / "raw",
        "wiki": lifecycle_env["wiki"],
        ".trash": lifecycle_env["trash"],
    }
    with pytest.raises(read_module.PathRejected):
        read_module.read_file(".kb/index.json", roots=roots)
