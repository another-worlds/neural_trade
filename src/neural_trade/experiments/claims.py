"""Cell claims: a lock file per cell so two processes never train the same cell (NT-030, from the
NT-026 QA note).

The runner picks its pending cells from the index and trains them one at a time; two runners on one
scenario (a sweep's parallel batch, or a second terminal) would both pick the same pending cell.
A claim is a file ``<scenario dir>/claims/<cell key>.lock`` created with ``open(path, "x")`` (atomic on
one machine), holding the owner's pid. A claim whose process is gone is stale and is taken over; a
claim whose process is alive is respected. The owner releases its claim when the cell ends, however
it ends. Nothing here deletes a run directory or a result.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CLAIMS_DIR = "claims"


def pid_alive(pid: int) -> bool:
    """True when a process with this pid is running (Windows and POSIX)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class CellClaims:
    """Claims under one directory (``<scenario dir>/claims``)."""

    def __init__(self, directory):
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.lock"

    def holder(self, key: str) -> Optional[int]:
        """The pid of a live claim on ``key``, else None."""
        try:
            pid = int(json.loads(self._path(key).read_text(encoding="utf-8")).get("pid", 0))
        except (OSError, ValueError, TypeError):
            return None
        return pid if pid_alive(pid) else None

    def claim(self, key: str) -> bool:
        """Take the claim on ``key``; False when a live process holds it."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        for _ in range(2):
            try:
                with open(path, "x", encoding="utf-8") as fh:
                    json.dump({"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat()}, fh)
                return True
            except FileExistsError:
                if self.holder(key) is not None:
                    return False
                logger.warning("claim on %s is stale (its process is gone): taking it over", key)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        return False

    def release(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass


__all__ = ["CLAIMS_DIR", "CellClaims", "pid_alive"]
