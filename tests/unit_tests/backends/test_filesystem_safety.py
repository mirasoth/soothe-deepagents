"""Tests for FilesystemBackend atomic write, backup, locks, and batched edits."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import TYPE_CHECKING

import pytest

from soothe_deepagents.backends.edit_locks import FileEditLockRegistry
from soothe_deepagents.backends.filesystem import FilesystemBackend
from soothe_deepagents.backends.fs_safety import compute_version_stamp, write_atomic
from soothe_deepagents.backends.protocol import BatchedEditOperation
from soothe_deepagents.backends.state import StateBackend

if TYPE_CHECKING:
    from pathlib import Path


class TestFsSafetyHelpers:
    def test_write_atomic_creates_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        write_atomic(target, "hello\n")
        assert target.read_text(encoding="utf-8") == "hello\n"
        assert not list(tmp_path.glob(".*.tmp"))

    def test_write_atomic_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("old", encoding="utf-8")
        write_atomic(target, "new")
        assert target.read_text(encoding="utf-8") == "new"

    def test_version_stamp_changes_on_write(self, tmp_path: Path) -> None:
        target = tmp_path / "stamp.txt"
        target.write_text("a", encoding="utf-8")
        before = compute_version_stamp(target)
        time.sleep(0.01)
        target.write_text("ab", encoding="utf-8")
        after = compute_version_stamp(target)
        assert before != after


class TestFilesystemBackendBackup:
    def test_write_with_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("original", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        res = be.write(str(path), "updated", backup=True)
        assert res.error is None
        assert res.backup_path is not None
        assert path.read_text(encoding="utf-8") == "updated"
        backups = list((tmp_path / ".backups").glob("f.txt.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "original"

    def test_edit_with_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "e.txt"
        path.write_text("hello world", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        res = be.edit(str(path), "world", "there", backup=True)
        assert res.error is None
        assert res.occurrences == 1
        assert res.backup_path is not None
        assert path.read_text(encoding="utf-8") == "hello there"

    def test_delete_with_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "d.txt"
        path.write_text("gone", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        res = be.delete(str(path), backup=True)
        assert res.error is None
        assert res.backup_path is not None
        assert not path.exists()
        backups = list((tmp_path / ".backups").glob("d.txt.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "gone"

    def test_write_without_backup_leaves_no_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "n.txt"
        path.write_text("x", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        res = be.write(str(path), "y", backup=False)
        assert res.error is None
        assert res.backup_path is None
        assert not (tmp_path / ".backups").exists()


class TestFilesystemBackendAtomicEdit:
    def test_edit_atomic_and_locked(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_text("one\ntwo\n", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        res = be.edit(str(path), "two", "TWO")
        assert res.error is None
        assert path.read_text(encoding="utf-8") == "one\nTWO\n"

    @pytest.mark.asyncio
    async def test_aedit_batched_replace(self, tmp_path: Path) -> None:
        path = tmp_path / "b.txt"
        path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        result = await be.aedit_batched(
            str(path),
            [
                BatchedEditOperation(
                    operation_type="replace",
                    start_line=2,
                    end_line=2,
                    content="changed",
                    original_call_id="c1",
                )
            ],
            backup=False,
        )
        assert result.error is None
        assert result.operations_applied == 1
        assert path.read_text(encoding="utf-8") == "line1\nchanged\nline3\n"

    @pytest.mark.asyncio
    async def test_aedit_batched_overlap_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "o.txt"
        path.write_text("a\nb\nc\n", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        result = await be.aedit_batched(
            str(path),
            [
                BatchedEditOperation(
                    operation_type="replace",
                    start_line=1,
                    end_line=2,
                    content="x",
                    original_call_id="a",
                ),
                BatchedEditOperation(
                    operation_type="replace",
                    start_line=2,
                    end_line=3,
                    content="y",
                    original_call_id="b",
                ),
            ],
        )
        assert result.error is not None
        assert "Overlapping" in result.error
        assert path.read_text(encoding="utf-8") == "a\nb\nc\n"

    @pytest.mark.asyncio
    async def test_aedit_batched_with_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "bak.txt"
        path.write_text("keep\nme\n", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        result = await be.aedit_batched(
            str(path),
            [
                BatchedEditOperation(
                    operation_type="delete",
                    start_line=2,
                    end_line=2,
                    content="",
                )
            ],
            backup=True,
        )
        assert result.error is None
        assert result.backup_path is not None
        assert path.read_text(encoding="utf-8") == "keep\n"


class TestEditLockRegistry:
    @pytest.mark.asyncio
    async def test_same_path_serializes(self) -> None:
        reg = FileEditLockRegistry()
        order: list[int] = []

        async def hold(n: int) -> None:
            async with reg.acquire("lock_serial.txt"):
                order.append(n)
                await asyncio.sleep(0.05)
                order.append(n + 10)

        await asyncio.gather(hold(1), hold(2))
        # Fully nested serialization: 1, 11, 2, 12 or 2, 12, 1, 11
        assert order in ([1, 11, 2, 12], [2, 12, 1, 11])

    @pytest.mark.asyncio
    async def test_different_paths_parallel(self) -> None:
        reg = FileEditLockRegistry()
        t0 = time.monotonic()

        async def hold(path: str) -> None:
            async with reg.acquire(path):
                await asyncio.sleep(0.1)

        await asyncio.gather(hold("parallel_a.txt"), hold("parallel_b.txt"))
        assert time.monotonic() - t0 < 0.18


class TestStateBackendBackupNoop:
    def test_accepts_backup_kwarg(self) -> None:
        """StateBackend signatures accept backup= (no-op in-memory)."""
        for name in ("write", "edit", "delete"):
            params = inspect.signature(getattr(StateBackend, name)).parameters
            assert "backup" in params
            assert params["backup"].default is False


class TestPublicSearchApi:
    def test_python_search_public(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("needle here\n", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        results, truncated, err = be.python_search("needle", tmp_path, "*.py")
        assert err is None
        assert truncated is False
        assert any("needle" in line for items in results.values() for _, line in items)

    def test_ripgrep_search_public_or_none(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("findme\n", encoding="utf-8")
        be = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        results, _truncated = be.ripgrep_search("findme", tmp_path, None)
        # None means ripgrep unavailable — still a valid public API result.
        if results is not None:
            assert any("findme" in line for items in results.values() for _, line in items)
