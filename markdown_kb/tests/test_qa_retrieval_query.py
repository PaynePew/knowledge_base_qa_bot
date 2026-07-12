"""Tests for issue #579 — filing under the original question, ``retrieval_query``
audit field.

Coverage:

- ``maybe_file_answer`` create path: ``question`` frontmatter stays the literal
  ``query`` argument; ``retrieval_query`` carries the rewrite when given, and
  defaults to ``query`` (passthrough) when omitted.
- ``maybe_file_answer`` touch path: ``question`` stays pinned to whatever was
  first filed (existing B2 semantics, untouched by this issue);
  ``retrieval_query`` refreshes to the latest call's value.
- ``dispatch_filing`` forwards ``retrieval_query`` through to
  ``maybe_file_answer`` unchanged, and still uses ``query`` (not
  ``retrieval_query``) for slug computation.
- ``promote`` / ``demote`` preserve an existing ``retrieval_query`` verbatim
  (same "preserve everything untouched" convention as ``question``/``sources``).

External-behaviour testing only — same discipline as ``test_qa.py``: no
reaching into ``qa._filing_lock`` or private write helpers. Hermetic —
``tmp_path`` via the autouse ``_redirect_paths_to_tmp`` fixture in
``conftest.py``.
"""

from __future__ import annotations

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
# maybe_file_answer — create path
# ---------------------------------------------------------------------------


def test_create_sets_question_to_original_and_retrieval_query_to_rewrite(tmp_path):
    """A create call with a distinct rewrite: question=original, retrieval_query=rewrite."""
    from app.qa import compute_slug, maybe_file_answer

    original = "and exchanges?"
    rewrite = "how long do exchanges take?"
    cited = [_stub("refund-policy#cancellation-window")]

    result = maybe_file_answer(original, "Within 24h.", cited, retrieval_query=rewrite)
    assert result is not None
    # Slug must be computed from the ORIGINAL question, never the rewrite.
    assert result.slug == compute_slug(original)

    qa_path = tmp_path / "wiki" / "qa" / f"{result.slug}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert f"question: {original}" in content, f"question must be the literal ask, got:\n{content}"
    assert f"retrieval_query: {rewrite}" in content, (
        f"retrieval_query must carry the rewrite, got:\n{content}"
    )


def test_create_defaults_retrieval_query_to_query_when_omitted(tmp_path):
    """No retrieval_query given (single-turn /chat, turn 1 passthrough) → both fields equal."""
    from app.qa import maybe_file_answer

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]

    result = maybe_file_answer(query, "Within 24h.", cited)
    assert result is not None

    qa_path = tmp_path / "wiki" / "qa" / f"{result.slug}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert f"question: {query}" in content
    assert f"retrieval_query: {query}" in content, (
        f"retrieval_query must default to query when not given, got:\n{content}"
    )


# ---------------------------------------------------------------------------
# maybe_file_answer — touch path
# ---------------------------------------------------------------------------


def test_touch_refreshes_retrieval_query_while_question_stays_pinned(tmp_path, monkeypatch):
    """A touch with a NEW retrieval_query updates it; question is unaffected (B2)."""
    from app.qa import maybe_file_answer

    # Write-through mode (#581 scope) — this test pins the persistence
    # contract, same convention as test_qa.py's touch tests.
    monkeypatch.setenv("KB_QA_COUNT_FLUSH_SEC", "0")
    original = "and exchanges?"
    cited = [_stub("refund-policy#cancellation-window")]

    first = maybe_file_answer(
        original, "Within 24h.", cited, retrieval_query="how long do exchanges take?"
    )
    assert first is not None and first.op == "created"

    second = maybe_file_answer(
        original, "Within 24h.", cited, retrieval_query="what about exchange timing?"
    )
    assert second is not None and second.op == "touched"
    assert second.slug == first.slug

    qa_path = tmp_path / "wiki" / "qa" / f"{first.slug}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert f"question: {original}" in content, "question must stay pinned across touches (B2)"
    assert "retrieval_query: what about exchange timing?" in content, (
        f"retrieval_query must refresh to the latest call's value, got:\n{content}"
    )
    assert "retrieval_query: how long do exchanges take?" not in content


# ---------------------------------------------------------------------------
# dispatch_filing — forwards retrieval_query, slugs on query
# ---------------------------------------------------------------------------


def test_dispatch_filing_forwards_retrieval_query_and_slugs_on_query(tmp_path):
    """dispatch_filing(query, result, retrieval_query=...) files under query,
    with retrieval_query landing in the frontmatter audit field."""
    from app.grounding import GroundingOutcome
    from app.qa import compute_slug, dispatch_filing

    original = "and exchanges?"
    rewrite = "how long do exchanges take?"
    result = {
        "answer": "Exchanges take 5-7 business days. [Source: refund-policy#cancellation-window]",
        "sources": [
            {"source": "refund-policy#cancellation-window", "heading": "h", "content": "c"}
        ],
        "grounding_outcome": GroundingOutcome(passed=True, reason="claim_supported"),
    }

    filed = dispatch_filing(original, result, retrieval_query=rewrite)
    assert filed is not None
    assert filed.slug == compute_slug(original)

    qa_path = tmp_path / "wiki" / "qa" / f"{filed.slug}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert f"question: {original}" in content
    assert f"retrieval_query: {rewrite}" in content


def test_dispatch_filing_without_retrieval_query_defaults_to_query(tmp_path):
    """Existing 2-arg call shape (POST /chat, no rewrite) keeps working unmodified."""
    from app.grounding import GroundingOutcome
    from app.qa import dispatch_filing

    query = "How long do refunds take?"
    result = {
        "answer": "5-7 business days. [Source: refund-policy#cancellation-window]",
        "sources": [
            {"source": "refund-policy#cancellation-window", "heading": "h", "content": "c"}
        ],
        "grounding_outcome": GroundingOutcome(passed=True, reason="claim_supported"),
    }

    filed = dispatch_filing(query, result)
    assert filed is not None

    qa_path = tmp_path / "wiki" / "qa" / f"{filed.slug}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert f"retrieval_query: {query}" in content


# ---------------------------------------------------------------------------
# promote / demote — preserve retrieval_query verbatim
# ---------------------------------------------------------------------------


def test_promote_preserves_retrieval_query(tmp_path):
    """Promoting a draft must not disturb its retrieval_query audit field."""
    from app.qa import maybe_file_answer, promote

    original = "and exchanges?"
    rewrite = "how long do exchanges take?"
    cited = [_stub("refund-policy#cancellation-window")]
    filed = maybe_file_answer(original, "Within 24h.", cited, retrieval_query=rewrite)
    assert filed is not None

    promote(filed.slug)

    qa_path = tmp_path / "wiki" / "qa" / f"{filed.slug}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert "status: live" in content
    assert f"retrieval_query: {rewrite}" in content, (
        f"promote must preserve retrieval_query verbatim, got:\n{content}"
    )


def test_demote_preserves_retrieval_query(tmp_path):
    """Demoting a live page must not disturb its retrieval_query audit field."""
    from app.qa import demote, maybe_file_answer, promote

    original = "and exchanges?"
    rewrite = "how long do exchanges take?"
    cited = [_stub("refund-policy#cancellation-window")]
    filed = maybe_file_answer(original, "Within 24h.", cited, retrieval_query=rewrite)
    assert filed is not None
    promote(filed.slug)

    demote(filed.slug)

    qa_path = tmp_path / "wiki" / "qa" / f"{filed.slug}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert "status: draft" in content
    assert f"retrieval_query: {rewrite}" in content, (
        f"demote must preserve retrieval_query verbatim, got:\n{content}"
    )
