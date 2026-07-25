"""High-performance glob discovery: ``fd`` fast path + ``os.scandir`` fallback.

Mirrors the ripgrep + Python fallback pattern used by
`FilesystemBackend.grep`. ``fd`` is optional; the scandir path always works.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_FD_ENV_VARS = ("DEEPAGENTS_FD_PATH", "SOOTHE_FD_PATH")
_DEFAULT_FD_TIMEOUT_S = 10.0
_SCANDIR_SCAN_CAP = 50_000
_FD_ERROR_EXIT_CODE = 2
_fd_missing_logged = False


@lru_cache(maxsize=1)
def get_fd_bin() -> str | None:
    """Resolve ``fd`` / ``fdfind`` binary (env override or ``PATH``)."""
    global _fd_missing_logged  # noqa: PLW0603
    for env_var in _FD_ENV_VARS:
        override = os.environ.get(env_var, "").strip()
        if override:
            if Path(override).is_file() or shutil.which(override):
                return override
            return None
    path = shutil.which("fd") or shutil.which("fdfind")
    if path is None and not _fd_missing_logged:
        _fd_missing_logged = True
        logger.info(
            "fd ('fd'/'fdfind') not found on PATH; using Python scandir glob fallback. "
            "Install fd for faster searches and automatic .gitignore handling."
        )
    return path


def reset_fd_bin_cache() -> None:
    """Clear cached fd resolution (tests)."""
    global _fd_missing_logged  # noqa: PLW0603
    _fd_missing_logged = False
    clear = getattr(get_fd_bin, "cache_clear", None)
    if callable(clear):
        clear()


def is_fd_available() -> bool:
    """Return True when an ``fd`` binary is resolvable."""
    return get_fd_bin() is not None


def parse_glob_pattern(pattern: str) -> tuple[str, bool]:
    """Normalize a glob pattern and detect directories-only (trailing slash).

    Args:
        pattern: Raw glob pattern (leading slashes should already be stripped).

    Returns:
        Tuple of ``(match_pattern, dirs_only)``.
    """
    dirs_only = pattern.endswith("/")
    match_pattern = pattern.rstrip("/") if dirs_only else pattern
    return match_pattern, dirs_only


def max_depth_for_pattern(pattern: str) -> int | None:
    """Return max scandir/fd depth for a pattern, or None when unbounded.

    Basename-only patterns (no ``/``) are unbounded so they keep recursive
    ``*.py`` / rglob semantics. Patterns with ``**`` are unbounded. Otherwise
    depth equals the number of path segments (e.g. ``packages/*`` → 2).
    """
    if "**" in pattern or "/" not in pattern:
        return None
    parts = [p for p in pattern.split("/") if p]
    return len(parts) if parts else 1


def list_via_fd(  # noqa: C901, PLR0912
    search_path: Path,
    pattern: str,
    *,
    dirs_only: bool = False,
    excludes: list[str] | None = None,
    max_results: int = 10_000,
    max_depth: int | None = None,
    timeout_s: float = _DEFAULT_FD_TIMEOUT_S,
    include_ignored: bool = False,
    include_hidden: bool = True,
) -> list[tuple[str, bool]] | None:
    """List matching entries via ``fd``. Returns None when fd is unavailable/fails.

    Paths in the result are absolute. ``is_dir`` is set from ``-t`` when
    dirs-only; otherwise classified with ``Path.is_dir()``.
    """
    fd_bin = get_fd_bin()
    if fd_bin is None:
        return None

    # Full-path match when the pattern has directory separators. fd matches
    # basename by default; prefix ``**/`` so ``packages/*`` hits ``packages/foo``.
    fd_pattern = pattern or "*"
    use_full_path = "/" in fd_pattern
    if use_full_path and not fd_pattern.startswith("**/"):
        fd_pattern = f"**/{fd_pattern}"

    cmd: list[str] = [
        fd_bin,
        "--glob",
        "-0",
        "--max-results",
        str(max(1, max_results)),
        "--no-require-git",
    ]
    if include_hidden:
        cmd.append("--hidden")
    if dirs_only:
        cmd.extend(["--type", "directory"])
    if include_ignored:
        cmd.append("--no-ignore")
    if max_depth is not None:
        cmd.extend(["--max-depth", str(max_depth)])
    for excl in excludes or ():
        cmd.extend(["--exclude", excl])
    if use_full_path:
        cmd.append("--full-path")
    cmd.extend(["--", fd_pattern, str(search_path)])

    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # fd: 0 = matches, 1 = no matches, >=2 = error
    if proc.returncode >= _FD_ERROR_EXIT_CODE:
        return None

    entries: list[tuple[str, bool]] = []
    search_resolved = search_path.resolve()
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        abs_path = Path(os.fsdecode(raw))
        try:
            resolved = abs_path.resolve()
        except OSError:
            continue
        try:
            resolved.relative_to(search_resolved)
        except ValueError:
            continue
        if dirs_only:
            is_dir = True
        else:
            try:
                is_dir = abs_path.is_dir()
            except OSError:
                continue
            if not is_dir and not abs_path.is_file():
                continue
        entries.append((str(resolved), is_dir))
        if len(entries) >= max_results:
            break
    return entries


def list_via_scandir(  # noqa: C901, PLR0912, PLR0915
    search_path: Path,
    *,
    workspace: Path,
    is_ignored: Callable[[str], bool] | None = None,
    max_depth: int | None = None,
    dirs_only: bool = False,
    include_ignored: bool = False,
    scan_cap: int = _SCANDIR_SCAN_CAP,
    deadline: float | None = None,
) -> tuple[list[tuple[str, bool]], bool]:
    """Enumerate files/dirs under ``search_path`` with ``os.scandir``.

    Returns ``(entries, truncated)`` where entries are workspace-relative
    ``(rel_posix, is_dir)`` pairs. When ``deadline`` (monotonic) is set and
    elapsed, returns whatever was gathered with ``truncated=True``.
    """
    results: list[tuple[str, bool]] = []
    truncated = False
    search_resolved = search_path.resolve()
    workspace_resolved = workspace.resolve()

    def _rel_to_workspace(abs_path: Path) -> str | None:
        try:
            rel = abs_path.resolve().relative_to(workspace_resolved)
        except ValueError:
            return None
        return "." if rel == Path() else rel.as_posix()

    stack: list[tuple[Path, int]] = [(search_resolved, 0)]

    while stack and len(results) < scan_cap:
        if deadline is not None and time.monotonic() > deadline:
            truncated = True
            break
        current, depth = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError:
            continue

        for entry in entries:
            if deadline is not None and time.monotonic() > deadline:
                truncated = True
                break
            if len(results) >= scan_cap:
                truncated = True
                break
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue

            child = Path(entry.path)
            rel_ws = _rel_to_workspace(child)
            if rel_ws is None or rel_ws == ".":
                continue

            if not include_ignored and is_ignored is not None and is_ignored(rel_ws):
                continue

            child_depth = depth + 1
            if max_depth is not None and child_depth > max_depth:
                continue

            if is_dir:
                results.append((rel_ws, True))
                if max_depth is None or child_depth < max_depth:
                    stack.append((child, child_depth))
            elif not dirs_only:
                try:
                    if entry.is_file(follow_symlinks=False):
                        results.append((rel_ws, False))
                except OSError:
                    continue
        if truncated:
            break

    return results, truncated


__all__ = [
    "get_fd_bin",
    "is_fd_available",
    "list_via_fd",
    "list_via_scandir",
    "max_depth_for_pattern",
    "parse_glob_pattern",
    "reset_fd_bin_cache",
]
