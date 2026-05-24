"""Shared fixtures for the markdown_kb test suite."""
import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.live tests unless they were explicitly selected with -m live."""
    # If the user passed -m live (or any expression selecting live), leave items alone.
    # Otherwise deselect every item that has the live marker so it shows up as skipped.
    marker_expr = config.option.markexpr if hasattr(config.option, "markexpr") else ""
    if "live" in marker_expr:
        # Opt-in: let the selected tests run as-is.
        return
    skip_live = pytest.mark.skip(reason="live test — run with: pytest -m live")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture()
def tmp_docs(tmp_path):
    """Create a minimal docs/ directory for testing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    return docs


@pytest.fixture()
def tmp_kb(tmp_path):
    """Provide a tmp .kb directory path."""
    return tmp_path / ".kb"


@pytest.fixture()
def tmp_wiki(tmp_path):
    """Provide a tmp wiki directory path."""
    return tmp_path / "wiki"
