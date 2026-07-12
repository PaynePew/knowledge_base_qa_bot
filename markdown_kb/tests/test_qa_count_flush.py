"""Tests for issue #581 — qa-page count write batching.

Scope decision (2026-07-12): a re-ask that hits an existing ``wiki/qa/``
page accumulates its ``count`` bump in memory and only persists it once the
page's ``KB_QA_COUNT_FLUSH_SEC`` batching window (default 300s) has
elapsed, or on ``flush_pending_counts(force=True)`` (the app-shutdown
path). Draft TTL / dedup are out of scope for this slice.

External-behaviour testing only, per ``test_qa.py``'s own stated
discipline: nothing here reaches into ``qa._filing_lock``,
``qa._pending_counts``, or any other private state — every assertion goes
through the public ``maybe_file_answer`` / ``flush_pending_counts`` surface,
the filesystem, and ``wiki/log.md``.

Hermetic: no LLM calls, no production wiki — every test uses ``tmp_path``
via the autouse ``_redirect_paths_to_tmp`` fixture in ``conftest.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class _StubSection:
    """Minimal Protocol satisfier — qa.maybe_file_answer only reads ``id``."""

    id: str
    heading_path: list[str]
    content: str


def _stub(section_id: str) -> _StubSection:
    return _StubSection(id=section_id, heading_path=[section_id], content="")


# ---------------------------------------------------------------------------
# Default batching window: a touch buffers in memory, disk stays stale
# ---------------------------------------------------------------------------


def test_touch_defers_disk_write_within_default_window(tmp_path):
    """Default KB_QA_COUNT_FLUSH_SEC (300s, unset in the test env) means a
    touch does NOT rewrite the page — the count bump is only buffered."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    first = maybe_file_answer(query, "Within 24h.", cited)
    assert first is not None and first.op == "created"

    qa_path = tmp_path / "wiki" / "qa" / f"{first.slug}.md"
    second = maybe_file_answer(query, "Within 24h.", cited)

    assert second is not None
    assert second.op == "touched"
    assert second.count == 2, "the RETURNED count is always the true, up-to-date value"

    on_disk = qa_path.read_text(encoding="utf-8")
    assert "count: 1" in on_disk, (
        f"disk write must be deferred inside the default batching window, got:\n{on_disk}"
    )
    assert "count: 2" not in on_disk


def test_multiple_touches_accumulate_a_single_pending_delta(tmp_path):
    """Three re-asks inside the window all buffer; the returned count keeps
    climbing even though nothing new has hit disk since the create."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    first = maybe_file_answer(query, "Within 24h.", cited)
    assert first is not None

    qa_path = tmp_path / "wiki" / "qa" / f"{first.slug}.md"

    results = [maybe_file_answer(query, "Within 24h.", cited) for _ in range(3)]
    assert [r.count for r in results] == [2, 3, 4]

    on_disk = qa_path.read_text(encoding="utf-8")
    assert "count: 1" in on_disk, "still un-flushed after 3 buffered touches"


# ---------------------------------------------------------------------------
# flush_pending_counts(force=True) catches the buffered delta up
# ---------------------------------------------------------------------------


def test_force_flush_persists_the_buffered_count(tmp_path):
    from app.qa import flush_pending_counts, maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    first = maybe_file_answer(query, "Within 24h.", cited)
    assert first is not None
    qa_path = tmp_path / "wiki" / "qa" / f"{first.slug}.md"

    for _ in range(3):
        maybe_file_answer(query, "Within 24h.", cited)
    assert "count: 1" in qa_path.read_text(encoding="utf-8")

    flushed = flush_pending_counts(force=True)
    assert flushed >= 1, (
        "must flush at least this test's own pending slug (module-level "
        "_pending_counts is process-wide, so other tests' still-buffered "
        "slugs may also be swept up here — that's fine, this assertion "
        "only needs a lower bound)"
    )

    on_disk = qa_path.read_text(encoding="utf-8")
    assert "count: 4" in on_disk, f"flush must persist the true accumulated count, got:\n{on_disk}"


def test_flush_preserves_body_sources_and_status(tmp_path):
    """A flush touches only count/updated — body, sources, status, question
    survive verbatim, same B2 guarantee a synchronous touch write gives."""
    from app.qa import flush_pending_counts, maybe_file_answer

    query = "How do I cancel my order?"
    answer = "Within 24h. See the policy for exceptions."
    cited = [_stub("refund-policy#cancellation-window")]
    first = maybe_file_answer(query, answer, cited)
    assert first is not None
    qa_path = tmp_path / "wiki" / "qa" / f"{first.slug}.md"

    maybe_file_answer(query, "A DIFFERENT WORDING that must not overwrite the body", cited)
    flush_pending_counts(force=True)

    on_disk = qa_path.read_text(encoding="utf-8")
    assert answer in on_disk
    assert "A DIFFERENT WORDING" not in on_disk
    assert "status: draft" in on_disk
    assert "refund-policy#cancellation-window" in on_disk


def test_flush_leaves_a_pristine_page_unaffected(tmp_path):
    """A page with zero pending touches is untouched by a flush call —
    it only visits slugs with a nonzero pending delta. Deliberately does
    NOT assert a global zero count: ``_pending_counts`` is process-wide, so
    another test's still-buffered slug may coexist regardless of run order.
    """
    from app.qa import flush_pending_counts, maybe_file_answer

    result = maybe_file_answer(
        "How do I cancel my order?", "Within 24h.", [_stub("refund-policy#cancellation-window")]
    )
    assert result is not None
    qa_path = tmp_path / "wiki" / "qa" / f"{result.slug}.md"
    before = qa_path.read_text(encoding="utf-8")

    flush_pending_counts(force=True)  # create wrote synchronously; nothing pending for this slug

    after = qa_path.read_text(encoding="utf-8")
    assert before == after, "flush must not rewrite a page it owes no pending delta to"


# ---------------------------------------------------------------------------
# qa_reflect keeps firing in real time even though the write is deferred
# ---------------------------------------------------------------------------


def test_reflect_log_still_emits_synchronously_while_buffered(tmp_path):
    """The audit trail (wiki/log.md) is unaffected by batching — only the
    qa-page file write is deferred."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    maybe_file_answer(query, "Within 24h.", cited)
    maybe_file_answer(query, "Within 24h.", cited)

    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    reflect_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln]
    assert len(reflect_lines) == 2, (
        f"one reflect line per call regardless of flush state: {reflect_lines}"
    )
    assert "op=touched" in reflect_lines[-1]
    assert "count=2" in reflect_lines[-1], (
        "log line carries the true count, not the stale disk value"
    )


# ---------------------------------------------------------------------------
# KB_QA_COUNT_FLUSH_SEC<=0 opts back into the pre-#581 synchronous behaviour
# ---------------------------------------------------------------------------


def test_flush_sec_zero_writes_every_touch_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_QA_COUNT_FLUSH_SEC", "0")
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    first = maybe_file_answer(query, "Within 24h.", cited)
    assert first is not None
    qa_path = tmp_path / "wiki" / "qa" / f"{first.slug}.md"

    second = maybe_file_answer(query, "Within 24h.", cited)
    assert second is not None and second.count == 2

    on_disk = qa_path.read_text(encoding="utf-8")
    assert "count: 2" in on_disk, (
        f"KB_QA_COUNT_FLUSH_SEC<=0 must disable batching (flush every touch), got:\n{on_disk}"
    )


# ---------------------------------------------------------------------------
# Create is never batched
# ---------------------------------------------------------------------------


def test_create_always_writes_immediately_regardless_of_flush_window(tmp_path):
    from app.qa import maybe_file_answer

    result = maybe_file_answer(
        "How do I cancel my order?", "Within 24h.", [_stub("refund-policy#cancellation-window")]
    )
    assert result is not None and result.op == "created"

    qa_path = tmp_path / "wiki" / "qa" / f"{result.slug}.md"
    assert qa_path.exists(), "create must never be deferred, even under batching"
    assert "count: 1" in qa_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# App-shutdown wiring: main.py's lifespan force-flushes on exit
# ---------------------------------------------------------------------------


def test_lifespan_shutdown_force_flushes_pending_counts(monkeypatch):
    """issue #581 scope: "...或程序關閉時 flush 一次到檔" — main.py's lifespan
    calls flush_pending_counts(force=True) after ``yield`` (app shutdown)."""
    import app.main as main_module

    calls: list = []
    monkeypatch.setattr(main_module, "load_index_json", lambda: None)
    monkeypatch.setattr(
        main_module, "flush_pending_counts", lambda force=False: calls.append(force)
    )

    async def _run() -> None:
        async with main_module.lifespan(main_module.app):
            assert calls == [], "must not flush before shutdown"

    asyncio.run(_run())
    assert calls == [True], "shutdown must force-flush exactly once"
