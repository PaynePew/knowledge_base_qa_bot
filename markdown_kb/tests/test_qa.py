"""Tests for Phase 6 Slice 6-2 ``app.qa`` module.

Coverage mirrors issue #80 acceptance criteria for the qa.py deep module:

- Slug determinism + entropy (Q3 S5)
- Create / touch lifecycle + B2 body-preservation semantics
- F3 fail-soft on IOError (returns None, emits qa_filing_error, no exception escapes)
- L1 concurrency (8 threads filing the same query → exactly one create)
- Reflect emission 1:1 with successful mutation
- Orphan-status touch refusal (PRD #78 Q8d defence layer 2)

External-behaviour testing only: nothing reaches into ``qa._filing_lock``
or the private write helpers. All assertions go through ``maybe_file_answer``
+ filesystem + ``wiki/log.md`` reads.

Hermetic: no LLM calls, no production wiki — every test uses ``tmp_path``
via the autouse ``_redirect_paths_to_tmp`` fixture in ``conftest.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Minimal Section stub implementing the CitableContent Protocol
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
# Slug helpers (compute_slug + normalize_question)
# ---------------------------------------------------------------------------


def test_compute_slug_is_whitespace_case_punctuation_invariant():
    """PRD #78 Q3 S5: slug must be stable across cosmetic question variants."""
    from app.qa import compute_slug

    base = compute_slug("How do I cancel my order?")
    assert compute_slug("how do i cancel my order") == base, "case must not affect slug"
    assert compute_slug("  How  do I cancel my order?  ") == base, "whitespace must not affect slug"
    assert compute_slug("How do I cancel my order!!!") == base, (
        "trailing punctuation must not affect slug"
    )
    assert compute_slug("How do I CANCEL my order.") == base, (
        "internal case + punct must not affect slug"
    )


def test_compute_slug_distinct_questions_produce_distinct_slugs():
    """Entropy probe: a small batch of unrelated questions must all hash apart."""
    from app.qa import compute_slug

    questions = [
        "How do I cancel my order?",
        "What is your refund policy?",
        "Can I change my email address?",
        "How long does shipping take?",
        "Do you offer student discounts?",
        "Where is my package?",
        "Is COD available in my area?",
        "How do I reset my password?",
        "Can I exchange a product?",
        "What payment methods do you accept?",
    ]
    slugs = {compute_slug(q) for q in questions}
    assert len(slugs) == len(questions), (
        f"Expected {len(questions)} unique slugs across distinct questions, got {len(slugs)}: {slugs}"
    )


def test_compute_slug_cjk_question_produces_readable_slug():
    """Phase 16: CJK questions produce a readable CJK-prefix slug.

    Pre-Phase-16 behaviour: slugify("如何取消订单？") returned "section" (CJK was
    dropped) → compute_slug mapped that to "qa-<hash>".

    Post-Phase-16 behaviour: slugify preserves CJK characters verbatim, so
    a CJK question now gets a grep-able CJK prefix (e.g. "如何取消订单-<hash>"),
    which is strictly more useful than the opaque "qa-<hash>" form.
    The hash suffix still ensures uniqueness.
    """
    from app.qa import compute_slug

    cjk_slug = compute_slug("如何取消订单？")
    # Phase 16: CJK chars are preserved → prefix contains CJK characters
    # (the question-mark ？ is stripped as punctuation, so the slug starts
    # with the CJK run directly)
    cjk_chars = [ch for ch in cjk_slug if "一" <= ch <= "鿿"]
    assert cjk_chars, f"Phase 16: expected CJK characters in slug prefix, got {cjk_slug!r}"
    # Hash suffix (6 hex chars) is always present
    parts = cjk_slug.rsplit("-", 1)
    assert (
        len(parts) == 2 and len(parts[1]) == 6 and all(c in "0123456789abcdef" for c in parts[1])
    ), f"Expected '<prefix>-<6hex>' format, got {cjk_slug!r}"


def test_normalize_question_strips_punct_collapses_whitespace():
    """Direct contract on the pure helper — input to sha1 must be canonical."""
    from app.qa import normalize_question

    assert normalize_question("How do I cancel?") == "how do i cancel"
    assert normalize_question("  HOW do I CANCEL?? ") == "how do i cancel"
    assert normalize_question("How  do\tI\ncancel") == "how do i cancel"


# ---------------------------------------------------------------------------
# Create path: first filing creates wiki/qa/<slug>.md with full schema
# ---------------------------------------------------------------------------


def test_create_writes_full_frontmatter_and_sentinel(tmp_path):
    """First filing produces a wiki/qa page with status:draft, count:1, sentinel."""
    from app.qa import SENTINEL_COMMENT, compute_slug, maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    result = maybe_file_answer(query, "Within 24h.", cited)

    assert result is not None, "first filing must return a FiledStatus"
    assert result.op == "created"
    assert result.status == "draft"
    assert result.count == 1
    assert result.slug == compute_slug(query)

    qa_path = tmp_path / "wiki" / "qa" / f"{result.slug}.md"
    assert qa_path.exists(), f"Expected qa file at {qa_path}"

    content = qa_path.read_text(encoding="utf-8")
    assert SENTINEL_COMMENT in content, "Sentinel HTML comment must be present"
    assert "status: draft" in content
    assert "count: 1" in content
    assert "type: qa" in content
    assert (
        f"question: {query}" in content
        or f'question: "{query}"' in content
        or (f"question: '{query}'" in content)
    ), f"Expected question in frontmatter, got:\n{content}"
    assert "Within 24h." in content


def test_create_emits_qa_reflect_created_log(tmp_path):
    """qa_reflect with op=created and the verbatim cited list must be logged."""
    from app.qa import maybe_file_answer

    cited_ids = ["refund-policy#cancellation-window", "refund-policy#refund-timeline"]
    cited = [_stub(c) for c in cited_ids]
    result = maybe_file_answer("How do I cancel my order?", "Within 24h.", cited)

    assert result is not None

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    assert "qa_reflect" in log
    assert f"slug={result.slug}" in log
    assert "op=created" in log
    # cited= must list both ids comma-separated
    assert ",".join(cited_ids) in log, f"Expected cited list in log, got:\n{log}"
    assert "count=1" in log


# ---------------------------------------------------------------------------
# Touch path: B2 semantics — body preserved, count bumped, updated refreshed
# ---------------------------------------------------------------------------


def test_touch_preserves_body_and_bumps_count(tmp_path):
    """B2 semantics: re-asking the same Q keeps the body verbatim, bumps count."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    first_answer = "Within 24h."
    cited = [_stub("refund-policy#cancellation-window")]
    first = maybe_file_answer(query, first_answer, cited)
    assert first is not None

    qa_path = tmp_path / "wiki" / "qa" / f"{first.slug}.md"
    first_content = qa_path.read_text(encoding="utf-8")
    assert first_answer in first_content

    # Different "answer" passed on the touch — body must NOT change
    second = maybe_file_answer(query, "DIFFERENT WORDING", cited)
    assert second is not None
    assert second.op == "touched"
    assert second.count == 2
    assert second.slug == first.slug

    second_content = qa_path.read_text(encoding="utf-8")
    assert first_answer in second_content, "Body must persist verbatim across re-asks"
    assert "DIFFERENT WORDING" not in second_content, "Touch must NOT overwrite body"
    assert "count: 2" in second_content


def test_touch_emits_reflect_with_cited_delta(tmp_path):
    """Touch reflect line must compute cited_delta against existing sources."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    first_cited = [_stub("refund-policy#cancellation-window")]
    maybe_file_answer(query, "Within 24h.", first_cited)

    # Second ask cites a different section → cited_delta should show added+dropped
    second_cited = [_stub("refund-policy#refund-timeline")]
    second = maybe_file_answer(query, "Within 24h.", second_cited)
    assert second is not None and second.op == "touched"

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    # There must be both a created entry and a touched entry
    reflect_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln]
    assert len(reflect_lines) == 2, f"Expected 2 qa_reflect entries, got: {reflect_lines}"
    touched_line = reflect_lines[-1]
    assert "op=touched" in touched_line
    assert "added:refund-policy#refund-timeline" in touched_line, (
        f"added: missing the new citation, got: {touched_line}"
    )
    assert "dropped:refund-policy#cancellation-window" in touched_line, (
        f"dropped: missing the old citation, got: {touched_line}"
    )
    assert "count=2" in touched_line


def test_touch_with_same_cited_emits_none_delta(tmp_path):
    """Identical cited list across re-asks → cited_delta=none."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    maybe_file_answer(query, "Within 24h.", cited)
    second = maybe_file_answer(query, "Within 24h.", cited)
    assert second is not None and second.op == "touched"

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    reflect_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln]
    touched_line = reflect_lines[-1]
    assert "cited_delta=none" in touched_line, (
        f"Identical cited list should produce cited_delta=none, got: {touched_line}"
    )


# ---------------------------------------------------------------------------
# Orphan-status touch refusal (Q8d defence layer 2)
# ---------------------------------------------------------------------------


def test_touch_against_invalid_status_returns_none(tmp_path):
    """Pre-existing wiki/qa page with status:Live (capital L) → refuse, log error."""
    from app.qa import compute_slug, maybe_file_answer

    query = "How do I cancel my order?"
    slug = compute_slug(query)
    qa_dir = tmp_path / "wiki" / "qa"
    qa_dir.mkdir(parents=True)
    orphan_path = qa_dir / f"{slug}.md"
    # Hand-plant a curator-typo orphan zombie
    orphan_path.write_text(
        "---\n"
        f"id: {slug}\n"
        "type: qa\n"
        'created: "2026-05-27T00:00:00Z"\n'
        'updated: "2026-05-27T00:00:00Z"\n'
        "sources: []\n"
        "status: Live\n"  # invalid (capital L)
        "open_questions: []\n"
        f'question: "{query}"\n'
        "count: 1\n"
        "---\n\nORIGINAL BODY DO NOT TOUCH.\n",
        encoding="utf-8",
    )
    before = orphan_path.read_text(encoding="utf-8")

    result = maybe_file_answer(query, "Within 24h.", [_stub("refund-policy#cancellation-window")])
    assert result is None, "touch against invalid status must return None"

    # File must be unchanged
    after = orphan_path.read_text(encoding="utf-8")
    assert before == after, "Orphan file must NOT be mutated"

    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "qa_filing_error" in log
    assert f"slug={slug}" in log
    assert "reason=orphan_status" in log


# ---------------------------------------------------------------------------
# F3 fail-soft on IOError
# ---------------------------------------------------------------------------


def test_filing_failsoft_on_oserror_returns_none(tmp_path, monkeypatch):
    """Monkeypatch os.replace → IOError; filing must not raise, must log, must return None."""
    import app.atomic as atomic_module
    from app.qa import maybe_file_answer

    def boom(src, dst):
        raise OSError("simulated disk full")

    # _atomic_write delegates to write_text_atomic in app.atomic, so the seam
    # is app.atomic.os.replace (not qa_module.os.replace any more).
    monkeypatch.setattr(atomic_module.os, "replace", boom)
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    result = maybe_file_answer(
        "How do I cancel my order?",
        "Within 24h.",
        [_stub("refund-policy#cancellation-window")],
    )
    assert result is None, "F3 fail-soft must return None on IOError"

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    assert "qa_filing_error" in log
    assert "reason=io_error" in log
    assert "OSError" in log


# ---------------------------------------------------------------------------
# L1 concurrency: 8 threads filing the same query
# ---------------------------------------------------------------------------


def test_concurrent_filings_produce_one_file_and_correct_count(tmp_path):
    """8 threads filing the same query → exactly 1 created + correct final count.

    PRD #78 Q7 L1: ``_filing_lock`` covers the whole filing decision so two
    threads cannot both create the file; the final ``count`` on disk must
    match the number of submitted requests.
    """
    from app.qa import compute_slug, maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]

    def file_once():
        return maybe_file_answer(query, "Within 24h.", cited)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: file_once(), range(8)))

    assert all(r is not None for r in results), f"All filings must succeed, got: {results}"
    ops = [r.op for r in results]
    assert ops.count("created") == 1, f"Exactly one filing must be the create; got ops: {ops}"
    assert ops.count("touched") == 7

    slug = compute_slug(query)
    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    assert qa_path.exists()

    # Final count on disk must be 8
    content = qa_path.read_text(encoding="utf-8")
    assert "count: 8" in content, (
        f"Final count on disk must equal request count (8), got file content:\n{content}"
    )
    # Final FiledStatus returned with op=touched and count must equal max observed
    max_count = max(r.count for r in results)
    assert max_count == 8, f"Highest observed FiledStatus.count must be 8, got {max_count}"


# ---------------------------------------------------------------------------
# 1:1 reflect emission for create / touch
# ---------------------------------------------------------------------------


def test_reflect_emission_one_to_one_with_mutations(tmp_path):
    """Three successful filings (1 create + 2 touch) produce exactly 3 reflect entries."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    for _ in range(3):
        maybe_file_answer(query, "Within 24h.", cited)

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    reflect_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln]
    assert len(reflect_lines) == 3, (
        f"Expected 3 qa_reflect entries, got {len(reflect_lines)}: {reflect_lines}"
    )
    assert sum(1 for ln in reflect_lines if "op=created" in ln) == 1
    assert sum(1 for ln in reflect_lines if "op=touched" in ln) == 2


def test_failsoft_path_emits_no_reflect(tmp_path, monkeypatch):
    """An IOError must NOT also emit a qa_reflect entry — only qa_filing_error."""
    import app.atomic as atomic_module
    from app.qa import maybe_file_answer

    # _atomic_write delegates to write_text_atomic in app.atomic; seam moves there.
    monkeypatch.setattr(
        atomic_module.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("nope"))
    )
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    result = maybe_file_answer(
        "How do I cancel my order?",
        "Within 24h.",
        [_stub("refund-policy#cancellation-window")],
    )
    assert result is None

    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "qa_reflect" not in log, (
        "Reflect emission must be atomic with the write; an aborted write must NOT log a reflect.\n"
        f"Log:\n{log}"
    )
    assert "qa_filing_error" in log


# ---------------------------------------------------------------------------
# Path independence sanity (no machine-specific paths leak into produced files)
# ---------------------------------------------------------------------------


def test_qa_path_resolves_under_monkeypatched_wiki_dir(tmp_path):
    """The qa file must land under the tmp wiki dir, not the real project wiki."""
    from app.qa import maybe_file_answer

    result = maybe_file_answer(
        "How do I cancel my order?",
        "Within 24h.",
        [_stub("refund-policy#cancellation-window")],
    )
    assert result is not None
    qa_path = tmp_path / "wiki" / "qa" / f"{result.slug}.md"
    # The file lives strictly under tmp_path — no real project wiki contamination.
    assert qa_path.exists()
    assert Path("wiki") not in (qa_path.parents)  # tmp_path is the only real parent
