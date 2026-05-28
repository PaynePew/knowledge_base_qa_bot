"""CitableContent protocol conformance for vector_rag's Chunk (issue #103).

vector_rag is the first real second consumer of markdown_kb's ``CitableContent``
Protocol (ADR-0004 Q9) — the validation the protocol was designed for. These
tests prove a ``Chunk`` satisfies the protocol structurally AND that
``grounding.verify()`` consumes a list of Chunks with NO changes to
``grounding.py`` (we exercise its internal user-message builder, which is the
only place the protocol fields are read).
"""

from __future__ import annotations

from markdown_kb.app.grounding import CitableContent, _build_user_message

from vector_rag.app.indexer import Chunk


def _make_chunk() -> Chunk:
    return Chunk(
        id="refund_policy.md#refund-timeline",
        source="refund_policy.md#refund-timeline",
        heading_path=["Refund Policy", "Refund Timeline"],
        content="Approved refunds are processed within 5-7 business days.",
    )


def test_chunk_satisfies_citable_content_protocol():
    """A Chunk is a structural CitableContent (runtime_checkable Protocol)."""
    chunk = _make_chunk()
    assert isinstance(chunk, CitableContent), (
        "Chunk must satisfy CitableContent (id / heading_path / content)"
    )


def test_chunk_exposes_protocol_fields_with_correct_types():
    """The three protocol fields exist with the contracted types."""
    chunk = _make_chunk()
    assert isinstance(chunk.id, str)
    assert isinstance(chunk.heading_path, list)
    assert all(isinstance(part, str) for part in chunk.heading_path)
    assert isinstance(chunk.content, str)


def test_grounding_consumes_chunks_unchanged():
    """grounding.py's protocol consumer formats Chunks without any changes.

    ``_build_user_message`` is the function inside the UNCHANGED grounding
    module that reads the CitableContent fields. Feeding it Chunks proves the
    adoption works through the protocol seam — no Section-specific code path.
    """
    chunks = [_make_chunk()]
    message = _build_user_message("Refunds take 5-7 business days.", chunks)

    assert "[Source: refund_policy.md#refund-timeline]" in message
    assert "Heading: Refund Policy > Refund Timeline" in message
    assert "Approved refunds are processed within 5-7 business days." in message
    assert "DRAFT_ANSWER:" in message
