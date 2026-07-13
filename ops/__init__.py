"""Ops tooling namespace — not a distributed package, never imported by the app.

Dev/ops scripts live under subpackages here (see ``ops/loadtest/``). Nothing
under ``ops/`` is a workspace member (no ``pyproject.toml``); it rides the
root venv, same as ``scripts/``.
"""
