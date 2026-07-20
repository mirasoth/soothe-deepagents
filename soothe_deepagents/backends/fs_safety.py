"""Safe local-file helpers for `FilesystemBackend` (atomic write, backup, stamps)."""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def compute_version_stamp(resolved: Path) -> str | None:
    """Return ``mtime_ns:size`` stamp, or ``None`` when the path is absent."""
    try:
        stat = resolved.stat()
    except OSError:
        return None
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def create_backup(resolved: Path, *, backup_dir: Path) -> Path | None:
    """Copy ``resolved`` into ``backup_dir`` as a timestamped ``.bak`` file.

    Returns:
        Backup path, or ``None`` when the source does not exist.
    """
    if not resolved.exists() or not resolved.is_file():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{resolved.name}.{stamp}.bak"
    shutil.copy2(resolved, backup_path)
    return backup_path


def write_atomic(resolved: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` via temp file in the same directory + atomic rename.

    The temp file lives beside the target so the rename stays on one filesystem.
    """
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{resolved.name}.{uuid.uuid4().hex}.tmp"
    tmp_path = resolved.parent / tmp_name
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        # Temp must not follow a symlink; create exclusive when possible.
        if hasattr(os, "O_EXCL"):
            flags |= os.O_EXCL
        fd = os.open(tmp_path, flags, 0o644)
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(content)
        tmp_path.replace(resolved)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
