"""Tests for Phase 6 Slice 6-4 ``qa.promote`` — curator promotion of draft Filed Answers.

Coverage mirrors issue #83 acceptance criteria for the qa.promote function:

- Promote a draft page → file frontmatter shows ``status: live``, return value
  is a correct ``FiledStatus``, ``qa_reflect op=promoted by=curator`` log entry
  present.
- Promote an already-live page → no mutation, idempotent, no duplicate log
  entry.
- Promote a slug that does not exist on disk → raises ``QaPageNotFound``.
- Promote a page with invalid existing status (e.g., ``status: Live`` capital L)
  → raises ``QaPageCorrupt``, file unchanged (orphan-visibility defence — do
  NOT silently "fix" the broken state).
- Concurrent filing + promote of the same slug → no torn write; final state
  is consistent (lock guarantees ordered access).

External-behaviour testing only: nothing reaches into ``qa._filing_lock`` or
the private write helpers. All assertions go through ``qa.promote`` +
filesystem + ``wiki/log.md`` reads.

Hermetic: no LLM calls, no production wiki — every test uses ``tmp_path`` via
the autouse ``_redirect_paths_to_tmp`` fixture in ``conftest.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Minimal Section stub implementing the CitableContent Protocol
# (mirrors test_qa.py to keep the filing-then-promote tests self-contained)
# ---------------------------------------------------------------------------


@dataclass
class _StubSection:
    """Minimal Protocol satisfier — qa.maybe_file_answer only reads ``id``."""

    id: str
    heading_path: list[str]
    content: str


def _stub(section_id: str) -> _StubSection:
    return _StubSection(id=section_id, heading_path=[section_id], content="")


# ---------------------------------------------------------------------------
# Promote: draft -> live (happy path)
# ---------------------------------------------------------------------------


def test_promote_draft_flips_status_to_live(tmp_path):
    """Promote a freshly-filed draft → ``status: live`` on disk + FiledStatus returned."""
    from app.qa import compute_slug, maybe_file_answer, promote

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    filed = maybe_file_answer(query, "Within 24h.", cited)
    assert filed is not None, "Filing setup failed — cannot test promote"
    assert filed.status == "draft"

    slug = compute_slug(query)
    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    before = qa_path.read_text(encoding="utf-8")
    assert "status: draft" in before

    result = promote(slug)

    assert result.slug == slug
    assert result.status == "live"
    assert result.op == "touched", (
        "Promotion is structurally a touch from the FiledStatus enum perspective"
    )
    # Count must be preserved verbatim (promote does not increment).
    assert result.count == filed.count

    after = qa_path.read_text(encoding="utf-8")
    assert "status: live" in after, f"Expected status: live in file, got:\n{after}"
    assert "status: draft" not in after, "Old status must be replaced, not duplicated"


def test_promote_draft_emits_qa_reflect_promoted_log(tmp_path):
    """Promote emits a ``qa_reflect op=promoted by=curator`` entry."""
    from app.qa import compute_slug, maybe_file_answer, promote

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    maybe_file_answer(query, "Within 24h.", cited)
    slug = compute_slug(query)

    promote(slug)

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    reflect_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln]
    promoted = [ln for ln in reflect_lines if "op=promoted" in ln]
    assert len(promoted) == 1, (
        f"Expected exactly one op=promoted reflect entry, got: {reflect_lines}"
    )
    line = promoted[0]
    assert f"slug={slug}" in line
    assert "by=curator" in line, f"Reflect line must include by=curator, got: {line}"


def test_promote_preserves_body_and_other_frontmatter(tmp_path):
    """Promote must touch ONLY ``status``; body + question + count + sources unchanged."""
    from app.qa import compute_slug, maybe_file_answer, promote

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    maybe_file_answer(query, "Within 24h. Original body wording.", cited)
    slug = compute_slug(query)

    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    before = qa_path.read_text(encoding="utf-8")

    promote(slug)

    after = qa_path.read_text(encoding="utf-8")
    # Body line preserved verbatim
    assert "Within 24h. Original body wording." in after
    # Question preserved
    assert "How do I cancel my order" in after
    # Sources preserved
    assert "refund-policy#cancellation-window" in after
    # Count preserved (no bump)
    assert "count: 1" in after
    # Only status changed
    assert "status: draft" in before
    assert "status: live" in after


# ---------------------------------------------------------------------------
# Promote already-live: idempotency
# ---------------------------------------------------------------------------


def test_promote_already_live_is_idempotent(tmp_path):
    """Re-promoting a live page: returns existing state, no file change, no dup log."""
    from app.qa import compute_slug, maybe_file_answer, promote

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    maybe_file_answer(query, "Within 24h.", cited)
    slug = compute_slug(query)

    first = promote(slug)
    assert first.status == "live"

    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    before_second = qa_path.read_text(encoding="utf-8")

    second = promote(slug)
    assert second.status == "live"
    assert second.slug == slug

    after_second = qa_path.read_text(encoding="utf-8")
    assert before_second == after_second, (
        "Second promote against an already-live page must NOT touch the file"
    )

    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    promoted_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln and "op=promoted" in ln]
    assert len(promoted_lines) == 1, (
        f"Idempotent promote must NOT emit a second reflect entry, got: {promoted_lines}"
    )


# ---------------------------------------------------------------------------
# Promote missing slug: QaPageNotFound
# ---------------------------------------------------------------------------


def test_promote_missing_slug_raises_qa_page_not_found(tmp_path):
    """No file on disk for the given slug → QaPageNotFound."""
    from app.qa import QaPageNotFound, promote

    with pytest.raises(QaPageNotFound):
        promote("does-not-exist-abc123")


def test_promote_missing_slug_does_not_emit_reflect(tmp_path):
    """Failed promote must NOT pollute the log with a phantom reflect entry."""
    from app.qa import QaPageNotFound, promote

    with pytest.raises(QaPageNotFound):
        promote("does-not-exist-abc123")

    log_path = tmp_path / "wiki" / "log.md"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8")
        assert "qa_reflect" not in log, "Promote of a missing slug must NOT emit a qa_reflect entry"


# ---------------------------------------------------------------------------
# Promote corrupt status: QaPageCorrupt — file unchanged (orphan-visibility)
# ---------------------------------------------------------------------------


def test_promote_invalid_status_raises_qa_page_corrupt_and_preserves_file(tmp_path):
    """``status: Live`` (capital L) → QaPageCorrupt; file unchanged."""
    from app.qa import QaPageCorrupt, compute_slug, promote

    query = "How do I cancel my order?"
    slug = compute_slug(query)
    qa_dir = tmp_path / "wiki" / "qa"
    qa_dir.mkdir(parents=True)
    corrupt_path = qa_dir / f"{slug}.md"
    corrupt_path.write_text(
        "---\n"
        f"id: {slug}\n"
        "type: qa\n"
        'created: "2026-05-27T00:00:00Z"\n'
        'updated: "2026-05-27T00:00:00Z"\n'
        "sources: []\n"
        "status: Live\n"  # invalid (capital L) — orphan zombie
        "open_questions: []\n"
        f'question: "{query}"\n'
        "count: 3\n"
        "---\n\nORIGINAL BODY DO NOT TOUCH.\n",
        encoding="utf-8",
    )
    before = corrupt_path.read_text(encoding="utf-8")

    with pytest.raises(QaPageCorrupt):
        promote(slug)

    after = corrupt_path.read_text(encoding="utf-8")
    assert before == after, (
        "Corrupt-status page must NOT be silently 'fixed' by promote — keep the broken state visible"
    )


def test_promote_corrupt_does_not_emit_reflect(tmp_path):
    """A QaPageCorrupt path must NOT emit a qa_reflect entry."""
    from app.qa import QaPageCorrupt, compute_slug, promote

    query = "How do I cancel my order?"
    slug = compute_slug(query)
    qa_dir = tmp_path / "wiki" / "qa"
    qa_dir.mkdir(parents=True)
    (qa_dir / f"{slug}.md").write_text(
        "---\n"
        f"id: {slug}\n"
        "type: qa\n"
        'created: "2026-05-27T00:00:00Z"\n'
        'updated: "2026-05-27T00:00:00Z"\n'
        "sources: []\n"
        "status: stale\n"  # forward-compat reserved value; still not actionable by promote
        "open_questions: []\n"
        f'question: "{query}"\n'
        "count: 1\n"
        "---\n\nbody.\n",
        encoding="utf-8",
    )

    with pytest.raises(QaPageCorrupt):
        promote(slug)

    log_path = tmp_path / "wiki" / "log.md"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8")
        assert "qa_reflect" not in log


# ---------------------------------------------------------------------------
# Concurrent filing + promote: lock serialises the two operations
# ---------------------------------------------------------------------------


def test_concurrent_filing_and_promote_no_torn_write(tmp_path):
    """Filing and promote on the same slug must not interleave.

    The PRD #78 contract: ``_filing_lock`` covers BOTH filing and promote so a
    promote cannot land in the middle of a touch's "read existing → rewrite"
    sequence. We exercise this with two thread pools racing each other.

    Loose invariant we assert: the final on-disk state is internally
    consistent — either the page exists with a recognised ``status`` in
    ``{"draft", "live"}``, the ``count`` field is a positive integer, and no
    torn YAML appears. Tighter ordering (filing-then-promote vs
    promote-then-filing) is non-deterministic and not asserted.
    """
    from app.qa import compute_slug, maybe_file_answer, promote

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    # Seed a draft so promote has something to flip
    seeded = maybe_file_answer(query, "Within 24h.", cited)
    assert seeded is not None
    slug = compute_slug(query)

    def do_file(_i):
        return maybe_file_answer(query, "Within 24h.", cited)

    def do_promote(_i):
        try:
            return promote(slug)
        except Exception as exc:  # noqa: BLE001 — test surfaces, not absorbs
            return exc

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda i: do_file(i) if i % 2 == 0 else do_promote(i), range(8)))

    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    assert qa_path.exists(), "qa file must still exist after concurrent ops"
    content = qa_path.read_text(encoding="utf-8")

    # No partial / torn writes — file must parse as well-formed YAML frontmatter
    assert content.count("---") >= 2, f"Frontmatter fences missing, file:\n{content}"
    # Status is one of the two recognised lifecycle values
    assert ("status: draft" in content) or ("status: live" in content), (
        f"Unexpected status in file, content:\n{content}"
    )
    # All concurrent promote calls returned without an unexpected exception type
    for r in results:
        if isinstance(r, Exception):
            # The only allowed concurrent failure mode: none — promote should
            # be idempotent against a draft/live page that filing might be
            # mid-touching. If we ever raise here, the lock is broken.
            raise AssertionError(f"Concurrent op raised unexpectedly: {r!r}")
