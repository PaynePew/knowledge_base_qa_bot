"""Shared fixtures for the markdown_kb test suite."""
import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient


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
