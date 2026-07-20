"""Version information for ``soothe_deepagents``.

Canonical version lives in ``pyproject.toml`` ``[project].version``.
This module re-exports the installed package metadata for runtime use.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("soothe-deepagents")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
