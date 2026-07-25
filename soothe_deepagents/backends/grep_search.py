"""Grep discovery: ``rg`` fast path + Python substring fallback.

Mirrors ``glob_search`` (``fd`` + scandir). Ripgrep is optional; the Python
path always works. Path formatting (virtual vs host) is injected by the caller
so this module stays free of ``FilesystemBackend`` state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from soothe_deepagents.backends.protocol import DEFAULT_GREP_TIMEOUT
from soothe_deepagents.backends.utils import compile_grep_include_glob

logger = logging.getLogger(__name__)

_RG_ENV_VARS = ("DEEPAGENTS_RG_PATH", "SOOTHE_RG_PATH")
_rg_missing_logged = False

PathFormatter = Callable[[Path], str | None]
"""Map a filesystem path to a result key, or ``None`` to skip the entry."""


@lru_cache(maxsize=1)
def get_rg_bin() -> str | None:
    """Resolve the ``rg`` binary (env override or ``PATH``)."""
    global _rg_missing_logged  # noqa: PLW0603
    for env_var in _RG_ENV_VARS:
        override = os.environ.get(env_var, "").strip()
        if not override:
            continue
        resolved = override if Path(override).is_file() or shutil.which(override) else None
        if resolved:
            return resolved
        # Invalid override — ignore and fall through to PATH.
    path = shutil.which("rg")
    if path is None and not _rg_missing_logged:
        _rg_missing_logged = True
        logger.info(
            "ripgrep ('rg') not found on PATH; using Python grep fallback. Install ripgrep for faster searches and automatic .gitignore handling."
        )
    return path


def is_rg_available() -> bool:
    """Return True when ripgrep is resolvable (fast path available)."""
    return get_rg_bin() is not None


def reset_rg_bin_cache() -> None:
    """Clear cached ripgrep resolution (tests)."""
    global _rg_missing_logged  # noqa: PLW0603
    _rg_missing_logged = False
    clear = getattr(get_rg_bin, "cache_clear", None)
    if callable(clear):
        clear()


def ripgrep_search(  # noqa: C901, PLR0912, PLR0915
    pattern: str,
    base_full: Path,
    include_glob: str | None,
    *,
    path_formatter: PathFormatter,
    timeout: int = DEFAULT_GREP_TIMEOUT,
) -> tuple[dict[str, list[tuple[int, str]]] | None, bool]:
    """Search using ripgrep with fixed-string (literal) mode.

    Args:
        pattern: Literal string to search for (unescaped).
        base_full: Resolved base path to search in.
        include_glob: Optional glob pattern to filter files.
        path_formatter: Maps each match path to a result key (or ``None`` to skip).
        timeout: Subprocess timeout seconds.

    Returns:
        ``(results, truncated)``. ``results`` is ``None`` when ripgrep is
        unavailable or hard-failed (caller should use Python fallback).
    """
    rg_path = get_rg_bin()
    if rg_path is None:
        return None, False

    cmd = [rg_path, "--json", "-F"]  # -F enables fixed-string (literal) mode
    if include_glob:
        cmd.extend(["--glob", include_glob])
    # When rg is given an absolute search path, directory-component
    # globs (e.g. "docs/*.md") silently match nothing if the process cwd
    # != search root (#2732). For a directory, set `cwd=base_full` and
    # use `.` as the search path so `--glob` resolves correctly.
    rg_cwd: str | None = None
    if base_full.is_dir():
        cmd.extend(["--", pattern, "."])
        rg_cwd = str(base_full)
    else:
        cmd.extend(["--", pattern, str(base_full)])

    truncated = False
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=rg_cwd,
        )
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if not stdout:
            logger.warning(
                "ripgrep timed out after %ds with no output; using Python grep fallback",
                timeout,
            )
            return None, False
        logger.warning("ripgrep timed out after %ds; returning partial results", timeout)
        truncated = True
    except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
        logger.warning(
            "ripgrep subprocess failed (%s: %s); using Python grep fallback",
            type(e).__name__,
            e,
        )
        reset_rg_bin_cache()
        return None, False

    if not truncated and proc.returncode not in (0, 1):
        stderr = proc.stderr.strip()[:500] if proc.stderr else ""
        logger.warning(
            "ripgrep exited %d (stderr=%r); using Python grep fallback",
            proc.returncode,
            stderr,
        )
        return None, False

    results: dict[str, list[tuple[int, str]]] = {}
    base_resolved = base_full.resolve()
    for line in stdout.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        data_type = data.get("type")
        if data_type == "error":
            logger.debug("ripgrep per-file error frame: %s", data.get("data"))
            continue
        if data_type != "match":
            continue
        pdata = data.get("data", {})
        ftext = pdata.get("path", {}).get("text")
        if not ftext:
            continue
        raw = Path(ftext)
        p = raw if raw.is_absolute() else (base_full / raw)
        try:
            p.resolve().relative_to(base_resolved)
        except (ValueError, OSError):
            logger.warning(
                "Skipping ripgrep result outside search root: path=%s root=%s",
                p,
                base_full,
            )
            continue
        virt = path_formatter(p)
        if virt is None:
            continue
        ln = pdata.get("line_number")
        lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
        if ln is None:
            continue
        results.setdefault(virt, []).append((int(ln), lt))

    return results, truncated


def python_search(  # noqa: C901, PLR0912, PLR0915
    pattern: str,
    base_full: Path,
    include_glob: str | None,
    *,
    path_formatter: PathFormatter,
    display_path: str,
    virtual_mode: bool = True,
    max_file_size_bytes: int = 10 * 1024 * 1024,
    timeout: int = DEFAULT_GREP_TIMEOUT,
) -> tuple[dict[str, list[tuple[int, str]]], bool, str | None]:
    """Fallback search using Python when ripgrep is unavailable.

    Args:
        pattern: Literal string to search for (substring match, not regex).
        base_full: Resolved base path to search in.
        include_glob: Optional glob pattern to filter files by name.
        path_formatter: Maps each file path to a result key (or ``None`` to skip).
        display_path: Agent-safe path string for timeout/abort messages.
        virtual_mode: When True, sanitize exception details in error messages.
        max_file_size_bytes: Skip files larger than this size.
        timeout: Maximum wall-clock seconds before the search is aborted.

    Returns:
        ``(results, truncated, error)``.
    """
    deadline = time.monotonic() + timeout
    glob_matcher = compile_grep_include_glob(include_glob) if include_glob else None

    results: dict[str, list[tuple[int, str]]] = {}
    file_errors: list[str] = []
    root = base_full if base_full.is_dir() else base_full.parent

    def _log_timeout() -> None:
        logger.warning(
            "Grep of '%s' timed out after %ss with %d matching file(s); returning partial results",
            display_path,
            timeout,
            len(results),
        )

    def _file_errors_msg() -> str | None:
        if not file_errors:
            return None
        return "One or more files could not be fully searched:\n" + "\n".join(file_errors)

    def _safe_detail(exc: Exception) -> str:
        if isinstance(exc, OSError):
            detail = exc.strerror
        else:
            detail = getattr(exc, "reason", None)
            if detail is None and not virtual_mode:
                detail = str(exc)
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__

    try:
        for fp in root.rglob("*"):
            if time.monotonic() > deadline:
                _log_timeout()
                return results, True, None
            try:
                if not fp.is_file():
                    continue
            except (PermissionError, OSError, RuntimeError):
                continue
            if glob_matcher is not None:
                rel_path = fp.relative_to(root).as_posix()
                if not glob_matcher(rel_path):
                    continue
            try:
                if fp.stat().st_size > max_file_size_bytes:
                    continue
            except (OSError, RuntimeError):
                continue
            scanned_lines = 0
            virt_path = ""
            try:
                formatted = path_formatter(fp)
                if formatted is None:
                    continue
                virt_path = formatted
                with fp.open(encoding="utf-8", errors="strict") as handle:
                    for line_num, raw_line in enumerate(handle, 1):
                        scanned_lines = line_num
                        if line_num % 2048 == 0 and time.monotonic() > deadline:
                            _log_timeout()
                            return results, True, None
                        if pattern not in raw_line:
                            continue
                        line = raw_line.rstrip("\n")
                        results.setdefault(virt_path, []).append((line_num, line))
            except UnicodeDecodeError as e:
                if scanned_lines > 0 or virt_path in results:
                    file_errors.append(f"- {virt_path}: {_safe_detail(e)}")
                else:
                    logger.debug("Skipping undecodable file in grep fallback: %s", fp)
                continue
            except (OSError, RuntimeError) as e:
                file_errors.append(f"- {virt_path}: {_safe_detail(e)}")
                logger.debug("Could not fully read %s in grep fallback", fp, exc_info=True)
                continue
    except (OSError, RuntimeError) as e:
        msg = f"Grep of '{display_path}' aborted after {len(results)} matching file(s): {_safe_detail(e)}"
        logger.warning("%s", msg, exc_info=True)
        return results, False, msg

    return results, False, _file_errors_msg()


__all__ = [
    "get_rg_bin",
    "is_rg_available",
    "python_search",
    "reset_rg_bin_cache",
    "ripgrep_search",
]
