"""Regression: the installed ``kb`` entry point must resolve the namespace
packages it lazily imports, from any working directory.

``markdown_kb`` and ``vector_rag`` are PEP 420 namespace packages and
``package = false`` workspace members (root ``pyproject.toml``) — they are never
installed into the environment, so they import only when the **repo root is on
``sys.path``**.  pytest provides that via ``pythonpath`` and ``python -m`` via the
cwd, but the installed ``kb`` console script has neither: ``sys.path[0]`` is the
launcher dir, not the repo root.  So ``uv run kb ask`` died with
``ModuleNotFoundError: No module named 'markdown_kb'`` at the lazy import inside
``_ask_one`` — a gap the test suite never caught because pytest's ``pythonpath``
masked it.

``kb_cli.main`` fixes this by inserting the repo root onto ``sys.path`` at import.
This test pins that: a fresh subprocess, run from a NON-repo cwd with no
``PYTHONPATH``, imports ``kb_cli.main`` and must then be able to import
``markdown_kb`` — exactly the path the ``kb`` script takes.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_entry_makes_namespace_packages_importable_from_any_cwd(tmp_path):
    """Importing ``kb_cli.main`` makes ``markdown_kb`` importable without cwd help.

    ``kb_cli`` itself resolves via its editable install (a site-packages finder),
    so the subprocess can import it with no ``PYTHONPATH``; ``markdown_kb`` resolves
    ONLY if ``kb_cli.main`` put the repo root on ``sys.path``.
    """
    script = (
        "import kb_cli.main  # noqa: F401  -- inserts the repo root onto sys.path\n"
        "import markdown_kb.app.errors  # noqa: F401  -- the import that crashed `kb ask`\n"
        "print('OK')\n"
    )

    # cwd = tmp_path (NOT the repo root) and PYTHONPATH stripped, so the repo root
    # reaches sys.path ONLY through kb_cli.main's bootstrap — mirroring how the
    # installed `kb` launcher runs. The repo root carries no .pth (markdown_kb is a
    # package=false namespace member), so nothing else leaks it in.
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env,
    )

    assert result.returncode == 0 and result.stdout.strip() == "OK", (
        "kb_cli.main did not make markdown_kb importable from a non-repo cwd. "
        "The installed `kb` console script has no repo-root-on-path, so the lazy "
        "`import markdown_kb...` in the command bodies fails. Check that main.py "
        "inserts Path(__file__).resolve().parents[2] onto sys.path at import. "
        f"returncode={result.returncode}\nstderr:\n{result.stderr}"
    )
