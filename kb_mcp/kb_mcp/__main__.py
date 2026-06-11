"""Entry point for ``python -m kb_mcp``.

Starts the FastMCP server over stdio (Claude Desktop transport).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# markdown_kb / vector_rag are PEP 420 namespace packages and `package = false`
# workspace members (root pyproject) — never installed, importable only with the
# repo root on sys.path. `python -m kb_mcp` supplies it via cwd, but that's fragile:
# `.server` -> `.hot_cache` does a module-level `import markdown_kb.app.atomic`, so a
# launch from a non-repo cwd dies before the server starts. Insert the repo root
# (this file is <repo>/kb_mcp/kb_mcp/__main__.py → parents[2]) before importing
# `.server`. Mirrors kb_cli/main.py (#221).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Claude Desktop launches this as ``python -m kb_mcp`` with no OPENAI_API_KEY in
# the environment, so ``kb_ask_v1`` would fail.  Load ``.env`` from the cwd here —
# parity with ``markdown_kb.app.main`` / ``gateway.app.main``.  Done before
# importing ``.server`` so any future import-time env read (e.g. KB_SCORE_THRESHOLD)
# also sees the loaded values; E402 below is intentional.
load_dotenv(find_dotenv(usecwd=True))

from .server import mcp  # noqa: E402

if __name__ == "__main__":
    mcp.run(transport="stdio")
