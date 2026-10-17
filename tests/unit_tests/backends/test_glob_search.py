"""Tests for the `fd` + scandir glob helpers consumed by soothe-nano.

`FilesystemBackend.glob` follows the upstream grep-include contract and does
not use these helpers, but they remain part of the package's public surface
(`soothe_nano.filesystem.{workspace,local}` and `soothe_nano.diagnose.tool_deps`
import them directly), so they are pinned here against silent breakage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from soothe_deepagents.backends import glob_search as glob_search_module
from soothe_deepagents.backends.glob_search import (
    get_fd_bin,
    is_fd_available,
    list_via_scandir,
    max_depth_for_pattern,
    parse_glob_pattern,
    reset_fd_bin_cache,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_parse_glob_pattern_trailing_slash() -> None:
    assert parse_glob_pattern("packages/*/") == ("packages/*", True)
    assert parse_glob_pattern("**/*.py") == ("**/*.py", False)


def test_max_depth_for_pattern() -> None:
    assert max_depth_for_pattern("*.py") is None  # recursive basename
    assert max_depth_for_pattern("**/*.py") is None
    assert max_depth_for_pattern("packages/*") == 2


def test_list_via_scandir_files_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub" / "b.py").write_text("y")

    files, truncated = list_via_scandir(tmp_path, workspace=tmp_path)
    assert not truncated
    assert ("a.py", False) in files
    assert ("sub", True) in files


def test_list_via_scandir_dirs_only(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "readme.txt").write_text("x")

    entries, _truncated = list_via_scandir(tmp_path, workspace=tmp_path, dirs_only=True)
    assert {name for name, is_dir in entries} == {"alpha", "beta"}
    assert all(is_dir for _name, is_dir in entries)


def test_list_via_scandir_respects_max_depth(tmp_path: Path) -> None:
    deep = tmp_path / "packages" / "alpha" / "src"
    deep.mkdir(parents=True)
    (deep / "c.py").write_text("z")

    shallow, _truncated = list_via_scandir(tmp_path, workspace=tmp_path, max_depth=2)
    assert "packages/alpha" in dict.fromkeys(name for name, _is_dir in shallow)
    assert "packages/alpha/src/c.py" not in shallow


def test_fd_env_override_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOOTHE_FD_PATH", "/opt/fd/bin/fd")
    monkeypatch.setattr(glob_search_module.shutil, "which", lambda _name: "/opt/fd/bin/fd")
    reset_fd_bin_cache()

    assert get_fd_bin() == "/opt/fd/bin/fd"
    assert is_fd_available() is True

    monkeypatch.delenv("SOOTHE_FD_PATH")
    monkeypatch.setattr(glob_search_module.shutil, "which", lambda _name: None)
    reset_fd_bin_cache()

    assert get_fd_bin() is None
    assert is_fd_available() is False
    reset_fd_bin_cache()
