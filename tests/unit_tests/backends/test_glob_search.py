"""Tests for fd + scandir glob (directory support and Python fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from soothe_deepagents.backends import filesystem as fs_module, glob_search as glob_search_module
from soothe_deepagents.backends.filesystem import FilesystemBackend
from soothe_deepagents.backends.glob_search import (
    max_depth_for_pattern,
    parse_glob_pattern,
)

if TYPE_CHECKING:
    import pytest


def test_parse_and_depth_helpers() -> None:
    assert parse_glob_pattern("packages/*/") == ("packages/*", True)
    assert max_depth_for_pattern("*.py") is None  # recursive basename
    assert max_depth_for_pattern("packages/*") == 2


def test_glob_dirs_only_trailing_slash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs_module, "list_via_fd", lambda *_a, **_k: None)
    (tmp_path / "packages" / "alpha").mkdir(parents=True)
    (tmp_path / "packages" / "beta").mkdir(parents=True)
    (tmp_path / "packages" / "alpha" / "src").mkdir()
    (tmp_path / "readme.txt").write_text("x")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = be.glob("packages/*/")
    assert result.error is None
    matches = result.matches or []
    assert matches
    assert all(m.get("is_dir") for m in matches)
    names = {Path(m["path"]).name for m in matches}
    assert names == {"alpha", "beta"}


def test_glob_default_excludes_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs_module, "list_via_fd", lambda *_a, **_k: None)
    (tmp_path / "subdir").mkdir()
    (tmp_path / "a.py").write_text("x")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = be.glob("*", path="/")
    paths = {m["path"] for m in (result.matches or [])}
    assert "/a.py" in paths
    assert "/subdir" not in paths


def test_glob_include_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs_module, "list_via_fd", lambda *_a, **_k: None)
    (tmp_path / "subdir").mkdir()
    (tmp_path / "a.py").write_text("x")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = be.glob("*", path="/", include_dirs=True)
    by_name = {Path(m["path"]).name: m.get("is_dir") for m in (result.matches or [])}
    assert by_name.get("a.py") is False
    assert by_name.get("subdir") is True


def test_glob_works_without_fd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glob_search_module, "get_fd_bin", lambda: None)
    glob_search_module.reset_fd_bin_cache()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "x.py").write_text("print(1)\n")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = be.glob("*.py", path="/")
    paths = {m["path"] for m in (result.matches or [])}
    assert "/nested/x.py" in paths
