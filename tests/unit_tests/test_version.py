"""Test that package version is consistent across configuration files."""

import tomllib
from pathlib import Path

import soothe_deepagents
from soothe_deepagents.backends import protocol
from soothe_deepagents.backends.protocol import ReadResult


def test_version_matches_pyproject() -> None:
    """Verify that ``__version__`` matches ``pyproject.toml`` metadata."""
    init_version = soothe_deepagents.__version__

    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    project = pyproject_data["project"]
    assert "dynamic" not in project or "version" not in project.get("dynamic", [])
    assert project["name"] == "soothe-deepagents"
    assert project["version"] == init_version
    assert init_version.count(".") == 2


def test_soothe_deepagents_namespace_imports_submodules() -> None:
    """Canonical ``soothe_deepagents`` imports should resolve consistently."""
    assert ReadResult is protocol.ReadResult
