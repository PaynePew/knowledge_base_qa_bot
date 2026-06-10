"""Tests for ``.env`` loading at the ``kb`` CLI entry point (``kb_cli/main.py``).

The ``kb`` console script (``kb_cli.main:app``) is the runtime entry point for
``uv run kb ask`` / the REPL.  Unlike ``markdown_kb.app.main`` /
``gateway.app.main`` / ``kb_mcp.__main__`` — which call
``load_dotenv(find_dotenv(usecwd=True))`` at import — the CLI historically did NOT
load ``.env``, and ``uv`` has no env-file config, so ``OPENAI_API_KEY`` was absent
and the grounded-answer path failed with an LLM auth error unless the user passed
``uv run --env-file .env`` or had already exported the key.  (The ``kb index`` and
search-only paths need no key, so the gap was silent until a grounded ask.)

This test pins the parity fix: importing ``kb_cli.main`` must load the ``.env``
sitting in the current working directory.  Run in a fresh subprocess so it observes
the real import-time side effect without the in-process autouse isolation fixture,
and fully hermetic — the ``.env`` lives in ``tmp_path`` (under the OS temp dir, never
inside the repo), so ``find_dotenv(usecwd=True)`` cannot reach the repo's own ``.env``.

Sibling of ``kb_mcp/tests/test_main.py`` (the kb_mcp parity fix, PR #219).
"""

from __future__ import annotations

import os
import subprocess
import sys

# A test-only probe key that no real process or .env defines, so its presence in
# the subprocess environment can ONLY come from the tmp .env loaded on import.
_PROBE_KEY = "KB_CLI_DOTENV_PROBE"


def test_cli_entry_loads_dotenv_from_cwd(tmp_path):
    """Importing ``kb_cli.main`` loads ``.env`` from the cwd (parity fix).

    Without the ``load_dotenv(find_dotenv(usecwd=True))`` call in ``main.py``
    the probe key is absent and the subprocess prints ``<absent>``.
    """
    # A .env in the subprocess cwd carries the probe key.
    (tmp_path / ".env").write_text(f"{_PROBE_KEY}=present\n", encoding="utf-8")

    # Import (not invoke) the entry module: building the Typer ``app`` object has
    # no side effects and nothing runs the CLI, so the import does not block.
    script = (
        "import os\n"
        "import kb_cli.main  # noqa: F401  -- import-time load_dotenv side effect\n"
        f"print(os.environ.get({_PROBE_KEY!r}, '<absent>'))\n"
    )

    # Strip the probe key from the inherited env so the only way it can appear is
    # via the tmp .env that the entry module loads.  Propagate the parent's
    # sys.path via PYTHONPATH so the subprocess can import kb_cli from
    # cwd=tmp_path.  find_dotenv(usecwd=True) walks the filesystem from cwd, NOT
    # sys.path, so this does not let it reach the repo's own .env — stays hermetic.
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
        "kb_cli.main did not load .env from cwd on import — "
        "expected the probe key to be present in os.environ. "
        "Check that main.py calls load_dotenv(find_dotenv(usecwd=True)) at import, "
        "mirroring markdown_kb.app.main / gateway.app.main / kb_mcp.__main__. "
        f"Got: {result.stdout.strip()!r}"
    )
