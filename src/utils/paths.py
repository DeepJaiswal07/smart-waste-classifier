"""Path resolution utilities ensuring platform-agnostic relative paths."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return absolute Path to the project root directory."""
    # This file is located in src/utils/paths.py, so root is two levels up from src (parents[2])
    root = Path(__file__).resolve().parents[2]
    return root


def resolve_path(relative_or_absolute: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return get_project_root() / path
