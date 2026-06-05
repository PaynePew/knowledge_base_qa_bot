"""Entry point for ``python -m kb_mcp``.

Starts the FastMCP server over stdio (Claude Desktop transport).
"""

from __future__ import annotations

from .server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
