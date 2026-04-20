"""Runtime path resolution for RLM.

Resolution order for all path functions:
  1. Explicit argument passed by caller
  2. Environment variable ``RLM_DATA_ROOT`` (for get_data_root)
  3. Fall back to ``Path.cwd() / "data"``

Never crawls upward from ``__file__``.  Safe for installed packages,
editable installs, and arbitrary working directories.

Usage::

    from rlm.data.paths import get_data_root, get_raw_data_dir, get_processed_data_dir

    root = get_data_root()                  # uses RLM_DATA_ROOT or cwd/data
    root = get_data_root("/mnt/lake")       # explicit override
"""

from __future__ import annotations

import os
from pathlib import Path


def get_data_root(explicit: str | Path | None = None) -> Path:
    """Return the absolute data root path.

    Resolution order:
    1. *explicit* argument
    2. ``RLM_DATA_ROOT`` environment variable
    3. ``Path.cwd() / "data"``
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    env = os.environ.get("RLM_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    return (Path.cwd() / "data").resolve()


def get_raw_data_dir(explicit: str | Path | None = None) -> Path:
    """Return the ``raw/`` sub-directory of the data root."""
    return get_data_root(explicit) / "raw"


def get_processed_data_dir(explicit: str | Path | None = None) -> Path:
    """Return the ``processed/`` sub-directory and ensure it exists."""
    path = get_data_root(explicit) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir(explicit: str | Path | None = None) -> Path:
    """Return the ``models/`` sub-directory and ensure it exists."""
    path = get_data_root(explicit) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_artifacts_dir(explicit: str | Path | None = None) -> Path:
    """Return the ``artifacts/`` sub-directory and ensure it exists."""
    path = get_data_root(explicit) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path
