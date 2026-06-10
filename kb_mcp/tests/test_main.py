"""Tests for the ``python -m kb_mcp`` entry point (``kb_mcp/__main__.py``).

The kb_mcp server is launched by Claude Desktop as ``python -m kb_mcp``.  Unlike
``markdown_kb.app.main`` / ``gateway.app.main`` — which call
``load_dotenv(find_dotenv(usecwd=True))`` at import — the entry point historically
did NOT load ``.env``, so ``OPENAI_API_KEY`` was absent and ``kb_ask_v1`` failed
when the host had not injected the key via the Claude Desktop config ``env`` block.

This test pins the parity fix: importing ``kb_mcp.__main__`` must load the ``.env``
sitting in the current working directory (the repo root, when Claude Desktop sets
``cwd`` to the project).  It is run in a fresh subprocess so it observes the real
import-time side effect without the in-process autouse isolation fixture, and it is
fully hermetic — the ``.env`` lives in ``tmp_path`` (under the OS temp dir, never
inside the repo), so ``find_dotenv(usecwd=True)`` cannot reach the repo's own ``.env``.
"""

from __future__ import annotations

import os
import subprocess
import sys

# A test-only probe key that no real process or .env defines, so its presence in
# the subprocess environment can ONLY come from the tmp .env loaded on import.
_PROBE_KEY = "KB_MCP_DOTENV_PROBE"


def test_main_loads_dotenv_from_cwd(tmp_path):
    """Importing ``kb_mcp.__main__`` loads ``.env`` from the cwd (parity fix).

    Without the ``load_dotenv(find_dotenv(usecwd=True))`` call in ``__main__.py``
    the probe key is absent and the subprocess prints ``<absent>``.
    """
    # A .env in the subprocess cwd carries the probe key.
    (tmp_path / ".env").write_text(f"{_PROBE_KEY}=present\n", encoding="utf-8")

    # Import (not run) the entry point: __name__ is "kb_mcp.__main__", so the
    # `if __name__ == "__main__"` guard is False and mcp.run() never blocks.
    script = (
        "import os\n"
        "import kb_mcp.__main__  # noqa: F401  -- import-time load_dotenv side effect\n"
        f"print(os.environ.get({_PROBE_KEY!r}, '<absent>'))\n"
    )

    # Strip the probe key from the inherited env so the only way it can appear is
    # via the tmp .env that __main__ loads.  Propagate the parent's sys.path via
    # PYTHONPATH so the subprocess can import kb_mcp + markdown_kb from cwd=tmp_path
    # (markdown_kb's package root is the repo root, normally provided by cwd).
    # find_dotenv(usecwd=True) walks the filesystem from cwd, NOT sys.path, so this
    # does not let it reach the repo's own .env — the test stays hermetic.
    child_env = {k: v for k, v in os.environ.items() if k != _PROBE_KEY}
    child_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env,
    )

    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "present", (
        "kb_mcp.__main__ did not load .env from cwd on import — "
        "expected the probe key to be present in os.environ. "
        "Check that __main__.py calls load_dotenv(find_dotenv(usecwd=True)) "
        "before mcp.run(), mirroring markdown_kb.app.main / gateway.app.main. "
        f"Got: {result.stdout.strip()!r}"
    )
