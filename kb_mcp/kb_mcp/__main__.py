"""Entry point for ``python -m kb_mcp``.

Starts the FastMCP server over stdio (Claude Desktop transport).
"""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

# Claude Desktop launches this as ``python -m kb_mcp`` with no OPENAI_API_KEY in
# the environment, so ``kb_ask_v1`` would fail.  Load ``.env`` from the cwd here —
# parity with ``markdown_kb.app.main`` / ``gateway.app.main``.  Done before
# importing ``.server`` so any future import-time env read (e.g. KB_SCORE_THRESHOLD)
# also sees the loaded values; E402 below is intentional.
load_dotenv(find_dotenv(usecwd=True))

from .server import mcp  # noqa: E402

if __name__ == "__main__":
    mcp.run(transport="stdio")
