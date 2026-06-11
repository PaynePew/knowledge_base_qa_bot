"""Regression: ``python -m kb_mcp`` must resolve markdown_kb from any cwd.

``markdown_kb`` / ``vector_rag`` are PEP 420 namespace packages and
``package = false`` workspace members (root ``pyproject.toml``) — never installed,
importable only when the repo root is on ``sys.path``.  ``python -m kb_mcp`` happens
to supply it via the cwd, but that is fragile: ``kb_mcp.hot_cache`` does a
**module-level** ``import markdown_kb.app.atomic``, so importing the server from a
non-repo cwd raises ``ModuleNotFoundError`` before the server even starts.  This is
the latent twin of the ``kb_cli`` bug fixed in #221.

``kb_mcp.__main__`` fixes it by inserting the repo root onto ``sys.path`` at import,
before ``from .server import mcp``.  This test pins that: a fresh subprocess, run
from a NON-repo cwd with no ``PYTHONPATH``, imports ``kb_mcp.__main__`` (which pulls
in the server → hot_cache → markdown_kb) and must succeed.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_entry_makes_namespace_packages_importable_from_any_cwd(tmp_path):
    """Importing ``kb_mcp.__main__`` works from a non-repo cwd (server import chain).

    ``kb_mcp`` resolves via its editable install, so the subprocess can import it
    with no ``PYTHONPATH``; the ``markdown_kb`` pulled in by ``.server`` →
    ``.hot_cache`` resolves ONLY if ``__main__`` put the repo root on ``sys.path``.
    Importing ``kb_mcp.__main__`` does not run the server: ``__name__`` is
    ``"kb_mcp.__main__"``, so the ``if __name__ == "__main__"`` guard skips
    ``mcp.run()``.
    """
    script = (
        "import kb_mcp.__main__  # noqa: F401  -- bootstraps sys.path, then imports .server\n"
        "import markdown_kb.app.errors  # noqa: F401  -- namespace pkg via the bootstrap\n"
        "print('OK')\n"
    )

    # cwd = tmp_path (NOT the repo root) and PYTHONPATH stripped, so the repo root
    # reaches sys.path ONLY through __main__'s bootstrap — mirroring how a Claude
    # Desktop launch of `python -m kb_mcp` runs when cwd is not the repo.
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env,
    )

    assert result.returncode == 0 and result.stdout.strip() == "OK", (
        "kb_mcp.__main__ did not make markdown_kb importable from a non-repo cwd. "
        "The server import chain (.server -> .hot_cache -> import markdown_kb.app.atomic) "
        "fails without repo-root-on-path. Check that __main__.py inserts "
        "Path(__file__).resolve().parents[2] onto sys.path before `from .server import mcp`. "
        f"returncode={result.returncode}\nstderr:\n{result.stderr}"
    )
