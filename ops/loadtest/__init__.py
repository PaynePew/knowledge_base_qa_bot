"""Memory-envelope load-test harness (issue #600).

Characterizes 512MB-box memory behavior under concurrent chat + ingest
without changing any app code (env-only integration). See
``project-docs/memory-envelope-600.md`` for methodology + measured results
and this package's module docstrings for the harness internals.

Entry point: ``uv run python -m ops.loadtest.harness --help``.
"""
