"""Cell claims: a lock file per cell so two processes never train the same cell (NT-030, from the
NT-026 QA note).

The runner picks its pending cells from the index and trains them one at a time; two runners on one
scenario (a sweep's parallel batch, or a second terminal) would both pick the same pending cell.
A claim is a file ``<scenario dir>/claims/<cell key>.lock`` created with ``open(path, "x")`` (atomic on
one machine), holding the owner's pid and its process start time (so a reused pid is not mistaken for
the owner). A claim whose process is gone is stale and is taken over; a claim whose process is alive
is respected. A claim file that cannot be read yet (the owner is mid-write, or Windows refuses access
while it is being created or removed) counts as HELD unless it is older than ``MIDWRITE_GRACE_S``. A
stale claim is taken over atomically: it is first renamed away (only one process can rename it), then
the new claim is created exclusively. The owner releases its claim when the cell ends, however it
ends. Nothing here deletes a run directory or a result.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CLAIMS_DIR = "claims"
MIDWRITE_GRACE_S = 10.0          # an unreadable claim younger than this is a claim being written
START_TOLERANCE_S = 2.0          # process start times of one process differ by less than this between reads


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


def process_start_time(pid: int) -> Optional[float]:
    """Seconds since the epoch at which ``pid`` started (None when unknown): tells a reused pid from the owner."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            try:
                created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
                if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                                ctypes.byref(kernel), ctypes.byref(user)):
                    return None
                ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime        # 100 ns since 1601
                return ticks / 1e7 - 11644473600.0
            finally:
                kernel32.CloseHandle(handle)
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        after = stat[stat.rindex(")") + 2:].split()
        boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime"))
        return boot + float(after[19]) / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 - unknown is a valid answer
        return None


class CellClaims:
    """Claims under one directory (``<scenario dir>/claims``)."""

    def __init__(self, directory):
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.lock"

    def _state(self, key: str) -> str:
        """"held" (a live owner, or a claim that cannot be judged yet), "stale" (its owner is gone or is a
        different process that reuses the pid), or "free" (no file)."""
        path = self._path(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "free"
        except OSError:                                    # Windows: access refused while being created / removed
            return "held"
        try:
            doc = json.loads(raw)
            pid = int(doc["pid"])
        except (ValueError, KeyError, TypeError):          # empty or half-written
            try:
                young = time.time() - path.stat().st_mtime < MIDWRITE_GRACE_S
            except OSError:
                young = True
            return "held" if young else "stale"
        if not pid_alive(pid):
            return "stale"
        recorded, now = doc.get("start"), process_start_time(pid)
        if recorded is not None and now is not None and abs(float(recorded) - now) > START_TOLERANCE_S:
            return "stale"                                  # the pid now belongs to another process
        return "held"

    def holder(self, key: str) -> Optional[int]:
        """The pid of a live claim on ``key``, else None."""
        if self._state(key) != "held":
            return None
        try:
            return int(json.loads(self._path(key).read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return -1

    def claim(self, key: str) -> bool:
        """Take the claim on ``key``; False when a live process holds it."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        for attempt in range(20):
            if attempt:
                time.sleep(0.02)
            try:
                with open(path, "x", encoding="utf-8") as fh:
                    json.dump({"pid": os.getpid(), "start": process_start_time(os.getpid()),
                               "utc": datetime.now(timezone.utc).isoformat()}, fh)
                return True
            except (FileExistsError, PermissionError):
                state = self._state(key)
                if state == "held":
                    return False
                if state == "stale":
                    logger.warning("claim on %s is stale (its process is gone): taking it over", key)
                    away = path.with_name(f"{path.name}.stale-{os.getpid()}-{time.time_ns()}")
                    try:
                        os.rename(path, away)              # only one process can move it away
                    except OSError:
                        continue                           # someone else did, or it is busy: look again
                    try:
                        away.unlink()
                    except OSError:
                        pass
        return False

    def release(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except (FileNotFoundError, PermissionError):
            pass


__all__ = ["CLAIMS_DIR", "CellClaims", "pid_alive", "process_start_time"]
