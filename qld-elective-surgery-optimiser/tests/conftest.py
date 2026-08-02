"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def temporary_project_root(tmp_path: Path) -> Path:
    """Return an isolated temporary project directory."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    return project_root