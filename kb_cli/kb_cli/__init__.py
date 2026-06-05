"""kb_cli — CLI for the knowledge base.

Phase 12 Slice 3 (ADR-0016).  One-shot subcommands (``kb ask``, ``kb index``)
and an interactive REPL (bare ``kb``).  Wraps ``markdown_kb`` deep modules
directly (NOT the Gateway), sharing the ``kb_mcp.freshness.reload_if_stale``
mtime-reload mechanism.
"""
