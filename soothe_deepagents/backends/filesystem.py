"""`FilesystemBackend`: Read and write files directly from the filesystem."""

import base64
import errno
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from soothe_deepagents.backends.edit_locks import FileEditLockRegistry
from soothe_deepagents.backends.fs_safety import (
    compute_version_stamp,
    create_backup,
    write_atomic,
)
from soothe_deepagents.backends.glob_search import (
    list_via_fd,
    list_via_scandir,
    max_depth_for_pattern,
    parse_glob_pattern,
)
from soothe_deepagents.backends.grep_search import (
    python_search as _python_search,
    ripgrep_search as _ripgrep_search,
)
from soothe_deepagents.backends.protocol import (
    DEFAULT_GREP_TIMEOUT,
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    BackendProtocol,
    BatchedEditOperation,
    BatchedEditResult,
    DeleteResult,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileOperationError,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from soothe_deepagents.backends.utils import (
    MAX_VIDEO_INPUT_BYTES,
    _get_backend_read_file_type,
    check_empty_content,
    compile_recursive_glob,
    perform_string_replacement,
)

logger = logging.getLogger(__name__)

_DEFAULT_GLOB_TIMEOUT = 5
"""Wall-clock budget in seconds for a single local `glob` walk.

A fixed bound keeps `glob` from hanging on huge or slow trees; when it elapses
the walk returns whatever it found so far with `GlobResult.truncated=True`
rather than erroring. Kept below the middleware's `GLOB_TIMEOUT`
(`soothe_deepagents.middleware.filesystem.GLOB_TIMEOUT`, currently 10s) so the backend
returns partial results before that outer net abandons the call; the ordering is
guarded by `test_glob_backend_budget_below_middleware_deadline`.
"""


class FilesystemBackend(BackendProtocol):
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory. Content is read/written
    as plain text, and metadata (timestamps) are derived from filesystem stats.

    !!! warning "Security Warning"

        This backend grants agents direct filesystem read/write access. Use with
        caution and only in appropriate environments.

        **Appropriate use cases:**

        - Local development CLIs (coding assistants, development tools)
        - CI/CD pipelines (see security considerations below)

        **Inappropriate use cases:**

        - Web servers or HTTP APIs - use `StateBackend`, `StoreBackend`, or
            `SandboxBackend` instead

        **Security risks:**

        - Agents can read any accessible file, including secrets (API keys,
            credentials, `.env` files)
        - Combined with network tools, secrets may be exfiltrated via SSRF attacks
        - File modifications are permanent and irreversible

        **Recommended safeguards:**

        1. Enable Human-in-the-Loop (HITL) middleware to review sensitive operations
        2. Exclude secrets from accessible filesystem paths (especially in CI/CD)
        3. For production environments, prefer `StateBackend`, `StoreBackend` or `SandboxBackend`

        In general, we expect this backend to be used with Human-in-the-Loop (HITL)
        middleware, or within a properly sandboxed environment if you need to run
        untrusted workloads.

        !!! note

            `virtual_mode=True` is primarily for virtual path semantics (for example with
            `CompositeBackend`). It can also provide path-based guardrails by blocking
            traversal (`..`, `~`) and absolute paths outside `root_dir`, but it does not
            provide sandboxing or process isolation. The default (`virtual_mode=False`)
            provides no security even with `root_dir` set.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool = True,  # noqa: FBT001, FBT002
        max_file_size_mb: int = 10,
        *,
        backup_dir: str | Path = ".backups",
    ) -> None:
        """Initialize filesystem backend.

        Args:
            root_dir: Optional root directory for file operations.

                Defaults to the current working directory.

                - When `virtual_mode=False` (default): Only affects relative path resolution.
                - When `virtual_mode=True`: Acts as a virtual root for filesystem operations.

            virtual_mode: Enable virtual path mode.

                **Primary use case:** stable, backend-independent path semantics when
                used with `CompositeBackend`, which strips route prefixes and forwards
                normalized paths to the routed backend.

                When `True`, all paths are treated as virtual paths anchored to
                `root_dir`. Path traversal (`..`, `~`) is blocked and all resolved paths
                are verified to remain within `root_dir`.

                When `False` (default), absolute paths are used as-is and relative paths
                are resolved under `root_dir`. This provides no security against an agent
                choosing paths outside `root_dir`.

                - Absolute paths (e.g., `/etc/passwd`) bypass `root_dir` entirely
                - Relative paths with `..` can escape `root_dir`
                - Agents have unrestricted filesystem access

            max_file_size_mb: Maximum file size in megabytes for operations like
                grep's Python fallback search.

                Files exceeding this limit are skipped during search. Defaults to 10 MB.
            backup_dir: Relative (to `root_dir`/`cwd`) or absolute directory used
                when `backup=True` on write/edit/delete.
        """
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._backup_dir = Path(backup_dir)
        self._edit_locks = FileEditLockRegistry()

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path with security checks.

        When `virtual_mode=True`, treat incoming paths as virtual absolute paths under
        `self.cwd`, disallow traversal (`..`, `~`) and ensure resolved path stays within
        root.

        When `virtual_mode=False`, preserve legacy behavior: absolute paths are allowed
        as-is; relative paths resolve under cwd.

        Args:
            key: File path (absolute, relative, or virtual when `virtual_mode=True`).

        Returns:
            Resolved absolute `Path` object.

        Raises:
            ValueError: If path traversal is attempted in `virtual_mode` or if the
                resolved path escapes the root directory.
            OSError: If the path is a symlink loop (`ELOOP`).
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                msg = "Path traversal not allowed"
                raise ValueError(msg)
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                msg = f"Path:{full} outside root directory: {self.cwd}"
                raise ValueError(msg) from None
            _raise_if_symlink_loop(full)
            return full

        path = Path(key)
        if path.is_absolute():
            _raise_if_symlink_loop(path)
            return path
        resolved = (self.cwd / path).resolve()
        _raise_if_symlink_loop(resolved)
        return resolved

    def _to_virtual_path(self, path: Path) -> str:
        """Convert a filesystem path to a virtual path relative to cwd.

        Args:
            path: Filesystem path to convert.

        Returns:
            Forward-slash relative path string prefixed with `/`.

        Raises:
            ValueError: If path is outside cwd.
            OSError: If `Path.resolve()` raises during resolution (e.g.,
                permission denied, or `ELOOP` on Python 3.13+).
            RuntimeError: If `Path.resolve()` detects a symlink loop on
                Python <=3.12 (wraps the underlying `OSError(ELOOP)`).
        """
        return "/" + path.resolve().relative_to(self.cwd).as_posix()

    def _display_path(self, path: Path) -> str:
        """Render a path for agent-visible messages without leaking the real root.

        In `virtual_mode`, surfacing the resolved on-disk path would defeat the
        virtual-path abstraction (and leak `root_dir`), so convert to the virtual
        form; fall back to the bare name (or `/` for a root path with no name
        component) if that conversion fails (e.g., the path escaped the root or
        could not be resolved). In non-virtual mode the real path is already the
        caller's own, so return it unchanged.

        Args:
            path: Filesystem path to render.

        Returns:
            A virtual path string in `virtual_mode`, otherwise the real path.
        """
        if not self.virtual_mode:
            return str(path)
        try:
            return self._to_virtual_path(path)
        except (ValueError, OSError, RuntimeError):
            return path.name or "/"

    def ls(self, path: str) -> LsResult:  # noqa: C901, PLR0912, PLR0915  # Complex virtual_mode logic
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute directory path to list files from.

        Returns:
            `LsResult` with `entries` listing files and directories directly in the
                directory on success.

                Directories have a trailing `/` in their path and `is_dir=True`.

                Missing paths set `error` to `Path '<path>': path_not_found`
                with `entries=None`.

                File paths set `error` to `Path '<path>': not_a_directory`
                with `entries=None`.

                Empty directories return `error=None` and `entries=[]`.
        """
        try:
            dir_path = self._resolve_path(path)
            if not dir_path.exists():
                return LsResult(error=f"Path '{path}': path_not_found", entries=None)
            if not dir_path.is_dir():
                return LsResult(error=f"Path '{path}': not_a_directory", entries=None)
        except (OSError, RuntimeError) as e:
            msg = f"Cannot list '{path}': {e}"
            logger.warning("%s", msg)
            return LsResult(error=msg, entries=None)

        results: list[FileInfo] = []
        errors: list[str] = []

        # Convert cwd to string for comparison
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # List only direct children (non-recursive)
        try:
            for child_path in dir_path.iterdir():
                try:
                    is_file = child_path.is_file()
                    is_dir = child_path.is_dir()
                except (OSError, RuntimeError) as e:
                    msg = f"child error: cannot stat '{child_path}': {e}"
                    logger.warning("%s", msg)
                    errors.append(msg)
                    continue

                abs_path = str(child_path)
                if not is_file and not is_dir:
                    # `is_symlink()` itself can raise OSError on stale handles or
                    # mid-walk permission flips; keep it inside the guard.
                    try:
                        if child_path.is_symlink():
                            child_path.resolve()
                            _raise_if_symlink_loop(child_path)
                    except (OSError, RuntimeError) as e:
                        msg = f"child error: cannot resolve '{child_path}': {e}"
                        logger.warning("%s", msg)
                        errors.append(msg)
                    continue

                if not self.virtual_mode:
                    # Non-virtual mode: use absolute paths
                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path + "/", "is_dir": True})
                else:
                    # Virtual mode: strip cwd prefix using Path for cross-platform support
                    try:
                        virt_path = self._to_virtual_path(child_path)
                    except ValueError:
                        logger.debug("Skipping path outside root: %s", child_path)
                        continue
                    except (OSError, RuntimeError) as e:
                        msg = f"child error: cannot resolve '{child_path}': {e}"
                        logger.warning("%s", msg)
                        errors.append(msg)
                        continue

                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # Local filesystem timestamps don't need timezone
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path + "/", "is_dir": True})
        except (OSError, RuntimeError) as e:
            # iterdir() itself can raise mid-iteration (NFS drops, FUSE failures,
            # permission flips). Surface as a top-level abort so partial results
            # are not labeled as authoritative.
            msg = f"Listing of '{path}' aborted: {e}"
            logger.warning("%s", msg)
            errors.append(msg)

        # Keep deterministic order by path
        results.sort(key=lambda x: x.get("path", ""))
        # Sort errors for deterministic output across filesystems (iterdir()
        # ordering varies); newline-join keeps them readable when any individual
        # message contains punctuation.
        error = "\n".join(sorted(errors)) if errors else None
        return LsResult(error=error, entries=results)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content for the requested line range.

        Args:
            file_path: Absolute or relative file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            `ReadResult` with raw (unformatted) content for the requested window.

                Line-number formatting is applied by the middleware.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")

        try:
            if not resolved_path.exists() or not resolved_path.is_file():
                return ReadResult(error=f"File '{file_path}' not found")

            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                file_type = _get_backend_read_file_type(file_path)
                if file_type != "text":
                    if file_type == "video" and os.fstat(fd).st_size > MAX_VIDEO_INPUT_BYTES:
                        return ReadResult(error=f"Video file exceeds maximum input size of {MAX_VIDEO_INPUT_BYTES} bytes")
                    with os.fdopen(fd, "rb") as f:
                        fd = -1
                        raw = f.read()
                    encoded = base64.standard_b64encode(raw).decode("ascii")
                    file_data = FileData(content=encoded, encoding="base64")
                else:
                    with os.fdopen(fd, "r", encoding="utf-8") as f:
                        fd = -1
                        content = f.read()
            finally:
                if fd >= 0:
                    os.close(fd)

            if file_type == "text":
                empty_msg = check_empty_content(content)
                if empty_msg:
                    file_data = FileData(content=empty_msg, encoding="utf-8")
                else:
                    # `splitlines(keepends=True)` preserves whether the final line
                    # has a terminator; joining with `""` round-trips the file's
                    # trailing-newline state. Required so `edit()` can detect
                    # EOF-newline mismatches in the model's `old_string`.
                    lines = content.splitlines(keepends=True)
                    start_idx = offset
                    end_idx = min(start_idx + limit, len(lines))

                    if start_idx >= len(lines):
                        return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")

                    file_data = FileData(content="".join(lines[start_idx:end_idx]), encoding="utf-8")

            return ReadResult(file_data=file_data)
        except (OSError, UnicodeDecodeError) as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")

    @property
    def edit_locks(self) -> FileEditLockRegistry:
        """Per-path edit lock registry shared with product façades (e.g. nano line tools)."""
        return self._edit_locks

    def _resolved_backup_dir(self) -> Path:
        """Return the absolute backup directory for this backend."""
        if self._backup_dir.is_absolute():
            return self._backup_dir
        return self.cwd / self._backup_dir

    def _backup_result_path(self, backup: Path | None) -> str | None:
        """Format a backup path for result objects (prefer cwd-relative)."""
        if backup is None:
            return None
        try:
            return str(backup.resolve().relative_to(self.cwd))
        except ValueError:
            return str(backup)

    def write(
        self,
        file_path: str,
        content: str,
        *,
        backup: bool = False,
    ) -> WriteResult:
        """Write content to a file via atomic temp-file + ``os.replace``.

        Args:
            file_path: Path where the file will be written.
            content: Text content to write to the file.
            backup: When True and the target already exists, copy it into
                ``backup_dir`` before overwriting.

        Returns:
            `WriteResult` with path on success, or error message on write failure.

        Note:
            Atomic rename requires the temp file and target to share a
            filesystem volume (temp is created beside the target).
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

        try:
            with self._edit_locks.acquire_sync(resolved_path):
                for attempt in range(2):
                    stamp_before = compute_version_stamp(resolved_path)
                    backup_path: Path | None = None
                    if backup and resolved_path.exists() and resolved_path.is_file():
                        backup_path = create_backup(resolved_path, backup_dir=self._resolved_backup_dir())
                    stamp_after = compute_version_stamp(resolved_path)
                    if stamp_before != stamp_after and attempt == 0:
                        continue
                    write_atomic(resolved_path, content)
                    return WriteResult(
                        path=file_path,
                        backup_path=self._backup_result_path(backup_path),
                    )
            return WriteResult(error=f"Error writing file '{file_path}': concurrent modification")
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
        *,
        backup: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Uses a per-path lock, optimistic version stamps (mtime+size), and
        atomic commit. Retries once on concurrent external modification.

        Args:
            file_path: Path to the file to edit.
            old_string: The text to search for and replace.
            new_string: The replacement text.
            replace_all: If `True`, replace all occurrences. If `False` (default),
                replace only if exactly one occurrence exists.
            backup: When True, copy the file into ``backup_dir`` before editing.

        Returns:
            `EditResult` with path and occurrence count on success, or error
                message if file not found or replacement fails.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

        try:
            if not resolved_path.exists() or not resolved_path.is_file():
                return EditResult(error=f"Error: File '{file_path}' not found")

            # Normalize line endings in old_string/new_string to match the
            # text-mode read below. Python universal newlines (the default
            # when newline=None) converts \r\n and bare \r to \n on read.
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")

            with self._edit_locks.acquire_sync(resolved_path):
                for attempt in range(2):
                    stamp_before = compute_version_stamp(resolved_path)

                    fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "r", encoding="utf-8") as f:
                        content = f.read()

                    result = perform_string_replacement(content, old_string, new_string, replace_all)
                    if isinstance(result, str):
                        return EditResult(error=result)

                    new_content, occurrences = result

                    backup_path: Path | None = None
                    if backup:
                        backup_path = create_backup(resolved_path, backup_dir=self._resolved_backup_dir())

                    stamp_after = compute_version_stamp(resolved_path)
                    if stamp_before != stamp_after and attempt == 0:
                        continue

                    write_atomic(resolved_path, new_content)
                    return EditResult(
                        path=file_path,
                        occurrences=int(occurrences),
                        backup_path=self._backup_result_path(backup_path),
                    )
            return EditResult(error=f"Error editing file '{file_path}': concurrent modification")
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def delete(self, file_path: str, *, backup: bool = False) -> DeleteResult:
        """Delete a file or directory from the filesystem.

        Files are unlinked. Directories are removed recursively along with all
        of their contents. Symlinks are removed as links and never followed into
        their target (so deleting a symlink to a directory removes only the link).

        Args:
            file_path: Path to the file or directory to delete.
            backup: When True and the path is a regular file, copy it into
                ``backup_dir`` before deletion.

        Returns:
            `DeleteResult` with the deleted path on success, or an error if the
                path does not exist or removal fails. A recursive directory
                removal may delete some entries before failing partway (for
                example when a nested entry is not writable).
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return DeleteResult(error=f"Error deleting '{file_path}': {e}")

        try:
            if not resolved_path.exists() and not resolved_path.is_symlink():
                return DeleteResult(error=f"Error: '{file_path}' not found")
            backup_path: Path | None = None
            if backup and resolved_path.is_file() and not resolved_path.is_symlink():
                backup_path = create_backup(resolved_path, backup_dir=self._resolved_backup_dir())
            if resolved_path.is_symlink():
                resolved_path.unlink()
            elif resolved_path.is_dir():
                shutil.rmtree(resolved_path)
            else:
                resolved_path.unlink()
            return DeleteResult(
                path=file_path,
                backup_path=self._backup_result_path(backup_path),
            )
        except (OSError, RuntimeError) as e:
            return DeleteResult(error=f"Error deleting '{file_path}': {e}")

    async def aedit_batched(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        file_path: str,
        operations: list[BatchedEditOperation],
        *,
        backup: bool = False,
    ) -> BatchedEditResult:
        """Apply multiple line-oriented edits in one atomic read/modify/write.

        Operations are applied in-memory then committed via atomic rename.
        Overlapping replace ranges fail fast. Concurrent external writers
        trigger one retry via version stamps.

        Args:
            file_path: Path to the file to edit.
            operations: Line edits to apply (delete → insert → replace order).
            backup: When True, create a backup before writing.

        Returns:
            `BatchedEditResult` with apply counts or an error.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as e:
            return BatchedEditResult(path=file_path, error=f"Error editing '{file_path}': {e}")

        if not resolved_path.exists() or not resolved_path.is_file():
            return BatchedEditResult(path=file_path, error=f"Error: File '{file_path}' not found")

        deletions = [op for op in operations if op.operation_type == "delete"]
        insertions = [op for op in operations if op.operation_type == "insert"]
        replacements = [op for op in operations if op.operation_type == "replace"]

        for i, op_a in enumerate(replacements):
            for op_b in replacements[i + 1 :]:
                if op_a.start_line <= op_b.end_line and op_b.start_line <= op_a.end_line:
                    return BatchedEditResult(
                        path=file_path,
                        error=(f"Overlapping edits: lines {op_a.start_line}-{op_a.end_line} and {op_b.start_line}-{op_b.end_line}"),
                        failed_operations=[
                            op_a.original_call_id or "",
                            op_b.original_call_id or "",
                        ],
                    )

        async with self._edit_locks.acquire(resolved_path):
            for attempt in range(2):
                stamp_before = compute_version_stamp(resolved_path)
                try:
                    fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "r", encoding="utf-8") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError) as e:
                    return BatchedEditResult(path=file_path, error=f"Error reading '{file_path}': {e}")

                lines = content.splitlines(keepends=True)
                backup_path: Path | None = None
                if backup:
                    backup_path = create_backup(resolved_path, backup_dir=self._resolved_backup_dir())

                stamp_after = compute_version_stamp(resolved_path)
                if stamp_before != stamp_after and attempt == 0:
                    continue

                total_lines_changed = 0
                operations_applied = 0
                failed_ops: list[str] = []

                for op in sorted(deletions, key=lambda o: o.start_line, reverse=True):
                    if op.start_line < 1 or op.end_line > len(lines) or op.start_line > op.end_line:
                        failed_ops.append(op.original_call_id or "")
                        continue
                    lines = lines[: op.start_line - 1] + lines[op.end_line :]
                    total_lines_changed += op.end_line - op.start_line + 1
                    operations_applied += 1

                for op in sorted(insertions, key=lambda o: o.start_line):
                    if op.start_line < 1 or op.start_line > len(lines) + 1:
                        failed_ops.append(op.original_call_id or "")
                        continue
                    new_lines = op.content.split("\n")
                    if new_lines and new_lines[-1] == "":
                        new_lines = new_lines[:-1]
                    formatted = [line + "\n" for line in new_lines]
                    lines = lines[: op.start_line - 1] + formatted + lines[op.start_line - 1 :]
                    total_lines_changed += len(formatted)
                    operations_applied += 1

                for op in sorted(replacements, key=lambda o: o.start_line, reverse=True):
                    if op.start_line < 1 or op.end_line > len(lines) or op.start_line > op.end_line:
                        failed_ops.append(op.original_call_id or "")
                        continue
                    new_lines = op.content.split("\n")
                    if new_lines and new_lines[-1] == "":
                        new_lines = new_lines[:-1]
                    formatted = [line + "\n" for line in new_lines]
                    lines = lines[: op.start_line - 1] + formatted + lines[op.end_line :]
                    total_lines_changed += max(op.end_line - op.start_line + 1, len(formatted))
                    operations_applied += 1

                try:
                    write_atomic(resolved_path, "".join(lines))
                except (OSError, UnicodeEncodeError) as e:
                    return BatchedEditResult(path=file_path, error=f"Error writing '{file_path}': {e}")

                return BatchedEditResult(
                    path=file_path,
                    operations_applied=operations_applied,
                    failed_operations=failed_ops or None,
                    backup_path=self._backup_result_path(backup_path),
                    total_lines_changed=total_lines_changed,
                )

        return BatchedEditResult(
            path=file_path,
            error=f"Error editing '{file_path}': concurrent modification",
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal text pattern in files.

        Uses ripgrep if available, falling back to Python search.

        Args:
            pattern: Literal string to search for (NOT regex).
            path: Directory or file path to search in. Defaults to current directory.
            glob: Optional glob pattern to filter which files to search.

        Returns:
            `GrepResult` with matches or error.
        """
        # Resolve base path
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return GrepResult(matches=[])
        except (OSError, RuntimeError) as e:
            search_path = path or "."
            return GrepResult(error=f"Error searching path '{search_path}': {e}", matches=[])

        try:
            if not base_full.exists():
                return GrepResult(matches=[])
        except OSError as e:
            search_path = path or "."
            return GrepResult(error=f"Error searching path '{search_path}': {e}", matches=[])

        # Try ripgrep first (with -F flag for literal search)
        results, truncated = self.ripgrep_search(pattern, base_full, glob)
        partial_error: str | None = None
        if results is None:
            # Python fallback does literal substring matching on the raw pattern.
            results, truncated, partial_error = self.python_search(pattern, base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return GrepResult(error=partial_error, matches=matches, truncated=truncated)

    def ripgrep_search(
        self,
        pattern: str,
        base_full: Path,
        include_glob: str | None,
    ) -> tuple[dict[str, list[tuple[int, str]]] | None, bool]:
        """Search using ripgrep with fixed-string (literal) mode.

        Public entry point for consumers that need ripgrep without going
        through `grep`. Implementation lives in ``backends.grep_search``.
        """
        return _ripgrep_search(
            pattern,
            base_full,
            include_glob,
            path_formatter=self._grep_path_formatter,
        )

    def python_search(
        self,
        pattern: str,
        base_full: Path,
        include_glob: str | None,
        *,
        timeout: int = DEFAULT_GREP_TIMEOUT,
    ) -> tuple[dict[str, list[tuple[int, str]]], bool, str | None]:
        """Fallback search using Python when ripgrep is unavailable.

        Public entry point for consumers that need the Python substring
        search without going through `grep`. Implementation lives in
        ``backends.grep_search``.
        """
        return _python_search(
            pattern,
            base_full,
            include_glob,
            path_formatter=self._grep_path_formatter,
            display_path=self._display_path(base_full),
            virtual_mode=self.virtual_mode,
            max_file_size_bytes=self.max_file_size_bytes,
            timeout=timeout,
        )

    def _grep_path_formatter(self, path: Path) -> str | None:
        """Format a grep hit path for result keys (virtual or host-absolute)."""
        if self.virtual_mode:
            try:
                return self._to_virtual_path(path)
            except ValueError:
                logger.debug("Skipping grep result outside root: %s", path)
                return None
            except (OSError, RuntimeError):
                logger.warning("Could not resolve grep result path: %s", path, exc_info=True)
                return None
        return str(path)

    def glob(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        pattern: str,
        path: str | None = None,
        *,
        include_dirs: bool = False,
        excludes: list[str] | None = None,
    ) -> GlobResult:
        """Find files (and optionally directories) matching a glob pattern.

        Uses ``fd`` when available (like ``rg`` for grep), otherwise
        ``os.scandir``. Trailing ``/`` on the pattern means directories only.
        When ``include_dirs`` is False (default), non-trailing patterns return
        files only for backward compatibility.

        Args:
            pattern: Glob pattern (e.g., ``'*.py'``, ``'**/*.txt'``, ``'packages/*/'``).
            path: Base directory to search from. Defaults to ``root_dir`` / ``cwd``.
            include_dirs: When True, include matching directories as well as files
                (``is_dir`` set). Ignored when the pattern is directories-only.
            excludes: Optional ``fd``/walk exclude names (e.g. ``node_modules``).

        Returns:
            ``GlobResult`` with matches. ``truncated`` is True when the walk
            exceeded its wall-clock budget.
        """
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        if self.virtual_mode and ".." in Path(pattern).parts:
            msg = "Path traversal not allowed in glob pattern"
            raise ValueError(msg)

        try:
            search_path = self.cwd if path is None or path == "/" else self._resolve_path(path)
            if not search_path.exists() or not search_path.is_dir():
                return GlobResult(matches=[])
        except (OSError, RuntimeError) as e:
            display_path = path if path is not None else "<default>"
            return GlobResult(error=f"Error globbing path '{display_path}': {e}", matches=[])

        match_pattern, dirs_only = parse_glob_pattern(pattern)
        if dirs_only and not match_pattern:
            return GlobResult(
                matches=[],
                error=f"Invalid directories-only glob pattern {pattern!r}",
            )

        want_dirs = dirs_only or include_dirs
        depth = max_depth_for_pattern(match_pattern)
        deadline = time.monotonic() + _DEFAULT_GLOB_TIMEOUT
        truncated = False
        results: list[FileInfo] = []

        try:
            matches_pattern = compile_recursive_glob(match_pattern)
        except (OSError, RuntimeError, ValueError) as e:
            display_path = path if path is not None else "<default>"
            return GlobResult(
                error=f"Glob of '{display_path}' aborted partway: {e}",
                matches=[],
            )

        # Prefer fd unless the pattern uses brace expansion (fd brace support varies).
        candidates: list[tuple[Path, bool]] = []
        used_fd = False
        if "{" not in match_pattern:
            fd_hits = list_via_fd(
                search_path,
                match_pattern or "*",
                dirs_only=dirs_only,
                excludes=excludes,
                max_depth=depth,
                timeout_s=max(_DEFAULT_GLOB_TIMEOUT, 1.0),
                include_ignored=False,
                include_hidden=True,
            )
            if fd_hits is not None:
                used_fd = True
                for abs_str, is_dir in fd_hits:
                    candidates.append((Path(abs_str), is_dir))

        if not used_fd:
            try:
                entries, truncated = list_via_scandir(
                    search_path,
                    workspace=search_path,
                    is_ignored=None,
                    max_depth=depth,
                    dirs_only=dirs_only,
                    include_ignored=True,
                    deadline=deadline,
                )
            except (OSError, RuntimeError, ValueError) as e:
                display_path = path if path is not None else "<default>"
                msg = f"Glob of '{display_path}' aborted partway: {e}"
                logger.warning("%s", msg, exc_info=True)
                return GlobResult(error=msg, matches=results)

            if truncated:
                logger.warning(
                    "Glob of '%s' timed out after %ss with partial results; returning gathered matches",
                    path if path is not None else "<default>",
                    _DEFAULT_GLOB_TIMEOUT,
                )
            for rel_posix, is_dir in entries:
                candidates.append((search_path / rel_posix, is_dir))

        try:
            for matched_path, is_dir in candidates:
                if time.monotonic() > deadline:
                    truncated = True
                    break
                try:
                    rel_path = matched_path.relative_to(search_path).as_posix()
                except ValueError:
                    continue
                if not matches_pattern(rel_path):
                    continue
                if dirs_only and not is_dir:
                    continue
                if not want_dirs and is_dir:
                    continue
                if self.virtual_mode:
                    try:
                        matched_path.resolve().relative_to(self.cwd)
                    except (OSError, RuntimeError, ValueError):
                        continue

                out_path: str
                if self.virtual_mode:
                    try:
                        out_path = self._to_virtual_path(matched_path)
                    except ValueError:
                        logger.debug("Skipping glob result outside root: %s", matched_path)
                        continue
                    except (OSError, RuntimeError):
                        logger.warning("Could not resolve glob result path: %s", matched_path, exc_info=True)
                        continue
                else:
                    out_path = str(matched_path)

                info: FileInfo = {"path": out_path, "is_dir": is_dir}
                if not is_dir:
                    try:
                        st = matched_path.stat()
                        info["size"] = int(st.st_size)
                        info["modified_at"] = datetime.fromtimestamp(st.st_mtime).isoformat()  # noqa: DTZ006
                    except OSError:
                        pass
                results.append(info)
        except (OSError, RuntimeError, ValueError) as e:
            display_path = path if path is not None else "<default>"
            msg = f"Glob of '{display_path}' aborted partway: {e}"
            logger.warning("%s", msg, exc_info=True)
            results.sort(key=lambda x: x.get("path", ""))
            return GlobResult(error=msg, matches=results)

        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the filesystem.

        Args:
            files: List of `(path, content)` tuples where content is bytes.

        Returns:
            List of `FileUploadResponse` objects, one per input file.

                Response order matches input order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved_path = self._resolve_path(path)

                # Create parent directories if needed
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved_path, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)

                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileUploadResponse(path=path, error=error))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the filesystem.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                if resolved_path.is_dir():
                    responses.append(FileDownloadResponse(path=path, content=None, error=IS_DIRECTORY))
                    continue
                # Use flags to optionally prevent symlink following if
                # supported by the OS
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses


def _map_exception_to_standard_error(exc: Exception) -> FileOperationError | None:
    """Map a caught exception to a standardized `FileOperationError` code.

    Classification is based on exception type only (stdlib hierarchy).
    Returns `None` for any exception that cannot be classified by type,
    letting callers decide whether to re-raise or fall back to `str(exc)`.

    Args:
        exc: The exception to classify.

    Returns:
        A `FileOperationError` literal, or `None` if unrecognized.
    """
    error: FileOperationError | None = None
    if isinstance(exc, FileNotFoundError):
        error = FILE_NOT_FOUND
    elif _is_symlink_loop_error(exc):
        error = INVALID_PATH
    elif isinstance(exc, PermissionError):
        error = PERMISSION_DENIED
    elif isinstance(exc, IsADirectoryError):
        error = IS_DIRECTORY
    elif isinstance(exc, (NotADirectoryError, FileExistsError, ValueError)):
        error = INVALID_PATH
    return error


# Win32 `ERROR_CANT_RESOLVE_FILENAME`, surfaced by NTFS for reparse-point
# cycles. Python's mapping to `errno.ELOOP` is unreliable on this code path,
# so we match the raw winerror when classifying symlink-loop failures.
_WIN32_ERROR_CANT_RESOLVE_FILENAME = 1921


def _is_eloop_oserror(exc: BaseException | None) -> bool:
    """Return `True` if `exc` is an `OSError` reporting a symlink loop on any platform."""
    return isinstance(exc, OSError) and (exc.errno == errno.ELOOP or getattr(exc, "winerror", None) == _WIN32_ERROR_CANT_RESOLVE_FILENAME)


def _is_symlink_loop_error(exc: Exception) -> bool:
    """Return `True` when an exception came from an `ELOOP` filesystem error."""
    if _is_eloop_oserror(exc):
        return True

    # Python <=3.12 wraps `OSError(errno.ELOOP, ...)` from `Path.resolve()` in
    # `RuntimeError`. The stable signal is the exception context, not the
    # human-readable RuntimeError message.
    return isinstance(exc, RuntimeError) and any(_is_eloop_oserror(chained) for chained in (exc.__cause__, exc.__context__))


def _raise_if_symlink_loop(path: Path) -> None:
    """Raise `OSError(ELOOP)` if `path` is an unresolvable symlink loop.

    Python 3.13+ changed `Path.resolve(strict=False)` to silently return the
    unresolved path for symlink loops instead of raising. This restores the
    pre-3.13 contract by probing with a `stat()` that follows symlinks and
    re-raising loop errors. Other errors (broken target, permission denied)
    are left for downstream existence checks to surface.

    Windows surfaces NTFS reparse-point cycles as `OSError` with
    `winerror=1921` (`ERROR_CANT_RESOLVE_FILENAME`); Python's mapping to
    `errno.ELOOP` is unreliable on this path, so we match the Win32 code
    explicitly via `_is_eloop_oserror`.
    """
    if not path.is_symlink():
        return
    try:
        path.stat()
    except OSError as exc:
        if _is_eloop_oserror(exc):
            raise
