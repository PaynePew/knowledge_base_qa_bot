"""Integration tests for kb_cli — kb ask, kb index, REPL dispatch.

Tests use typer's CliRunner (``typer.testing.CliRunner``) to invoke the CLI
in-process without spawning a subprocess.

Mocking follows the project pattern (CODING_STANDARD §11 / implement.md §3.1):
  - LLM mocked at ``markdown_kb.app.retrieval.get_llm`` / ``_llm`` (not a deep
    entry point)
  - ``reload_if_stale`` mocked at ``kb_mcp.freshness.reload_if_stale``
  - Index search mocked at ``markdown_kb.app.indexer.search``
  - ``build_index`` mocked where needed for the ``kb index`` subcommand

The ``_isolate_module_state`` autouse fixture in conftest.py provides module-state
isolation; path-redirect logic is not duplicated here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubSection:
    """Minimal Section stub."""

    id: str
    content: str
    heading_path: list[str] = field(default_factory=lambda: ["Test Heading"])
    metadata: dict = field(default_factory=dict)
    file: str = "stub"


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, answer: str = "Refunds take 5-7 days.") -> None:
        self._answer = answer

    def invoke(self, messages: list) -> _FakeLLMResponse:
        return _FakeLLMResponse(content=self._answer)


# ---------------------------------------------------------------------------
# Shared patch helper
# ---------------------------------------------------------------------------


def _patch_retrieval(monkeypatch, fake_llm=None):
    """Patch the wiki retrieval stack to avoid real I/O."""
    import kb_mcp.freshness as freshness_mod
    import markdown_kb.app.indexer as wiki_indexer
    import markdown_kb.app.retrieval as retrieval_mod
    from markdown_kb.app.grounding import GroundingOutcome

    if fake_llm is None:
        fake_llm = _FakeLLM()

    stub_section = _StubSection(id="stub#heading", content="Refunds take 5-7 days.")

    monkeypatch.setattr(wiki_indexer, "search", lambda q, k=3: [(stub_section, 1.5)])
    monkeypatch.setattr(wiki_indexer, "expand_to_pages", lambda secs: secs)
    monkeypatch.setattr(freshness_mod, "reload_if_stale", lambda *_a, **_kw: False)
    monkeypatch.setattr(retrieval_mod, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_mod, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        retrieval_mod.grounding_module,
        "verify",
        lambda draft, secs: GroundingOutcome(passed=True, reason="claim_supported"),
    )

    # Populate sections in-place (NOT rebind — mirrors kb_mcp conftest pattern)
    wiki_indexer.sections.append(stub_section)
    return stub_section


# ---------------------------------------------------------------------------
# AC-1: kb ask "Q" prints a grounded answer + citations
# ---------------------------------------------------------------------------


def test_kb_ask_exits_zero(monkeypatch):
    """``kb ask "Q"`` exits with code 0 on a grounded answer."""
    from kb_cli.main import app

    _patch_retrieval(monkeypatch)
    result = runner.invoke(app, ["ask", "How long do refunds take?"])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"


def test_kb_ask_prints_answer(monkeypatch):
    """``kb ask "Q"`` prints the answer text to stdout."""
    from kb_cli.main import app

    _patch_retrieval(monkeypatch)
    result = runner.invoke(app, ["ask", "How long do refunds take?"])
    assert "Refunds take 5-7 days" in result.output, (
        f"Expected answer in output, got:\n{result.output}"
    )


def test_kb_ask_prints_citation(monkeypatch):
    """``kb ask "Q"`` prints at least one citation."""
    from kb_cli.main import app

    _patch_retrieval(monkeypatch)
    result = runner.invoke(app, ["ask", "How long do refunds take?"])
    # Citation must appear — section id "stub#heading" is a citation
    assert "stub" in result.output.lower() or "citation" in result.output.lower(), (
        f"Expected citation in output, got:\n{result.output}"
    )


def test_kb_ask_stack_rag_flag(monkeypatch):
    """``kb ask "Q" --stack rag`` accepts the rag stack flag without error."""
    import vector_rag.app.retrieval as rag_retrieval_mod
    from markdown_kb.app.grounding import GroundingOutcome

    from kb_cli.main import app

    # Patch the rag retrieval path
    monkeypatch.setattr(
        rag_retrieval_mod,
        "query",
        lambda q: {
            "answer": "rag answer",
            "sources": [
                {
                    "source": "rag-doc",
                    "heading": "rag heading",
                    "score": None,
                    "content": "rag content",
                    "derived_from": None,
                }
            ],
            "grounding_outcome": GroundingOutcome(passed=True, reason="claim_supported"),
        },
    )
    result = runner.invoke(app, ["ask", "What is X?", "--stack", "rag"])
    assert result.exit_code == 0, f"Expected exit 0 for rag stack, got:\n{result.output}"


# ---------------------------------------------------------------------------
# AC-1: kb index (re)builds the Section Index
# ---------------------------------------------------------------------------


def test_kb_index_exits_zero(monkeypatch, tmp_path):
    """``kb index`` exits with code 0 after building the index."""
    import kb_mcp.freshness as freshness_mod
    import markdown_kb.app.indexer as wiki_indexer

    from kb_cli.main import app

    # Patch build_index to avoid real file I/O
    monkeypatch.setattr(wiki_indexer, "build_index", lambda: (3, 12))
    monkeypatch.setattr(freshness_mod, "_last_mtime", None)

    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"


def test_kb_index_prints_summary(monkeypatch, tmp_path):
    """``kb index`` prints a summary with files / sections counts."""
    import markdown_kb.app.indexer as wiki_indexer

    from kb_cli.main import app

    monkeypatch.setattr(wiki_indexer, "build_index", lambda: (3, 12))

    result = runner.invoke(app, ["index"])
    output = result.output
    # Must report counts somehow
    assert "3" in output or "12" in output or "indexed" in output.lower(), (
        f"Expected index summary in output, got:\n{output}"
    )


# ---------------------------------------------------------------------------
# AC-2: LLMError renders as non-zero exit + stderr message
# ---------------------------------------------------------------------------


def test_kb_ask_llm_error_nonzero_exit(monkeypatch):
    """LLMError from the wiki stack renders as non-zero exit code."""
    import kb_mcp.freshness as freshness_mod
    import markdown_kb.app.indexer as wiki_indexer
    import markdown_kb.app.retrieval as retrieval_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    stub = _StubSection(id="stub#heading", content="test")
    monkeypatch.setattr(wiki_indexer, "search", lambda q, k=3: [(stub, 1.5)])
    monkeypatch.setattr(wiki_indexer, "expand_to_pages", lambda s: s)
    monkeypatch.setattr(freshness_mod, "reload_if_stale", lambda *_a, **_kw: False)
    wiki_indexer.sections.append(stub)

    def _raise_llm_error(q, prompt):
        raise LLMError(retryable=True, message="LLM service temporarily unavailable.")

    monkeypatch.setattr(retrieval_mod, "_call_llm_with_error_handling", _raise_llm_error)

    result = runner.invoke(app, ["ask", "will fail"])
    assert result.exit_code != 0, f"Expected non-zero exit for LLMError, got {result.exit_code}"


def test_kb_ask_llm_error_message_in_stderr(monkeypatch):
    """LLMError message appears in output (stderr mixed into CliRunner output)."""
    import kb_mcp.freshness as freshness_mod
    import markdown_kb.app.indexer as wiki_indexer
    import markdown_kb.app.retrieval as retrieval_mod
    from markdown_kb.app.errors import LLMError

    from kb_cli.main import app

    stub = _StubSection(id="stub#heading", content="test")
    monkeypatch.setattr(wiki_indexer, "search", lambda q, k=3: [(stub, 1.5)])
    monkeypatch.setattr(wiki_indexer, "expand_to_pages", lambda s: s)
    monkeypatch.setattr(freshness_mod, "reload_if_stale", lambda *_a, **_kw: False)
    wiki_indexer.sections.append(stub)

    def _raise_llm_error(q, prompt):
        raise LLMError(retryable=False, message="LLM auth failed (check OPENAI_API_KEY).")

    monkeypatch.setattr(retrieval_mod, "_call_llm_with_error_handling", _raise_llm_error)

    # mix_stderr=True (default) merges stderr into output — works with typer.echo(..., err=True)
    result = runner.invoke(app, ["ask", "will fail"])
    # Message should appear in the combined output
    combined = result.output or ""
    assert "LLM" in combined or "Error" in combined, (
        f"Expected LLM error message in output, got:\n{combined}"
    )


# ---------------------------------------------------------------------------
# AC-3: Bare ``kb`` enters a REPL, dispatches a query against the warm index
# ---------------------------------------------------------------------------


def test_repl_dispatches_query_and_quits(monkeypatch):
    """REPL dispatches a user query, prints a response, then quits on 'quit'."""
    from kb_cli.main import app

    _patch_retrieval(monkeypatch)

    # Simulate user typing "How long do refunds take?" then "quit"
    result = runner.invoke(
        app,
        [],  # bare kb → REPL
        input="How long do refunds take?\nquit\n",
    )
    assert result.exit_code == 0, f"REPL unexpected exit code: {result.exit_code}\n{result.output}"
    assert "kb>" in result.output or ">" in result.output, (
        f"Expected REPL prompt in output, got:\n{result.output}"
    )


def test_repl_shows_answer(monkeypatch):
    """REPL shows the LLM answer after a query."""
    from kb_cli.main import app

    _patch_retrieval(monkeypatch)

    result = runner.invoke(app, [], input="How long do refunds take?\nquit\n")
    assert "Refunds take 5-7 days" in result.output, (
        f"Expected answer in REPL output, got:\n{result.output}"
    )


def test_repl_stack_toggle(monkeypatch):
    """REPL :stack rag and :stack wiki toggle the stack without error."""
    from kb_cli.main import app

    _patch_retrieval(monkeypatch)

    result = runner.invoke(
        app,
        [],
        input=":stack rag\n:stack wiki\nquit\n",
    )
    assert result.exit_code == 0, f"REPL stack toggle failed: {result.exit_code}\n{result.output}"


def test_repl_quit_exits_cleanly(monkeypatch):
    """REPL 'quit' exits with code 0."""
    from kb_cli.main import app

    _patch_retrieval(monkeypatch)

    result = runner.invoke(app, [], input="quit\n")
    assert result.exit_code == 0, (
        f"REPL quit should exit 0, got: {result.exit_code}\n{result.output}"
    )
