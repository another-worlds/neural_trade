"""Cell claims: a lock file per cell so two processes never train the same cell (NT-030, from the
NT-026 QA note).

The runner picks its pending cells from the index and trains them one at a time; two runners on one
scenario (a sweep's parallel batch, or a second terminal) would both pick the same pending cell.
A claim is a file ``<scenario dir>/claims/<cell key>.lock`` created with ``open(path, "x")`` (atomic on
one machine), holding the owner's pid and its process start time (so a reused pid is not mistaken for
the owner). A claim whose process is gone is stale and is taken over; a claim whose process is alive
is respected. A claim file that cannot be read yet (the owner is mid-write, or Windows refuses access
while it is being created or removed) counts as HELD unless it is older than ``MIDWRITE_GRACE_S``. A stale claim is taken over under a sentinel (NT-183):

1. the taker creates ``<cell>.lock.takeover`` with ``open(path, "x")``; exactly one taker wins, the
   others back off and treat the claim as held (the winner is replacing it right now);
2. the winner RE-READS the claim under the sentinel and goes on only if it is still stale; a claim
   that was taken over or created since the taker's first read is now held, so the taker backs off;
3. the winner writes its claim to a temporary file and ``os.replace``s it over the stale file (the claim
   path is never absent, so no plain ``"x"`` creator slips in between), then removes the sentinel in a
   ``finally``.

Proof sketch. The claim file changes only by (i) an exclusive create, which needs it absent, (ii) a
takeover replace, which needs the sentinel, and (iii) the owner's release (a dead owner never
releases). Two takers cannot hold the sentinel together, and the one that does decides on a read made
after it got the sentinel; every earlier taker's replace has finished by then (it removes the sentinel
only after its replace), so the re-read sees its fresh live claim and the taker backs off. A stale
verdict made before the sentinel is therefore never acted on. No timing is assumed. A sentinel left by
a crashed taker is reaped when its pid is dead (or reused) and it is older than ``SENTINEL_STALE_S``;
the reaper moves it away by an atomic rename, so one reaper wins. The only residue is two reapers
meeting a sentinel that is both dead and over ``SENTINEL_STALE_S`` old at once, with one of them
delayed past a third taker's fresh sentinel: it needs a crashed taker, a 30 s pause and that
interleaving together, and it costs one duplicated cell (the run store rejects a duplicate run id,
NT-182). The owner releases its claim when the cell ends, however it
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
SENTINEL_STALE_S = 30.0          # a takeover sentinel of a dead taker older than this is reaped
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
                    self._after_stale_decision(key)
                    outcome = self._take_over(key)
                    if outcome is not None:
                        return outcome                    # ours, or someone else is taking it / took it
        return False

    def _sentinel(self, key: str) -> Path:
        return self.dir / f"{key}.lock.takeover"

    def _sentinel_dead(self, sentinel: Path) -> bool:
        """True for a sentinel of a crashed taker: old, and its pid gone or reused."""
        try:
            if time.time() - sentinel.stat().st_mtime < SENTINEL_STALE_S:
                return False
            doc = json.loads(sentinel.read_text(encoding="utf-8"))
            pid = int(doc["pid"])
        except FileNotFoundError:
            return False
        except (OSError, ValueError, KeyError, TypeError):      # unreadable and old
            return True
        if not pid_alive(pid):
            return True
        recorded, now = doc.get("start"), process_start_time(pid)
        return recorded is not None and now is not None and abs(float(recorded) - now) > START_TOLERANCE_S

    def _take_over(self, key: str) -> Optional[bool]:
        """Replace a stale claim under the takeover sentinel. True: ours now. False: held by someone else.
        None: try again (a dead sentinel was reaped)."""
        path, sentinel = self._path(key), self._sentinel(key)
        try:
            with open(sentinel, "x", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "start": process_start_time(os.getpid())}, fh)
        except (FileExistsError, PermissionError):
            if self._sentinel_dead(sentinel):
                away = sentinel.with_name(f"{sentinel.name}.dead-{os.getpid()}-{time.time_ns()}")
                try:
                    os.rename(sentinel, away)
                    away.unlink()
                except OSError:
                    pass
                return None
            return False
        try:
            if self._state(key) != "stale":                     # re-read under the sentinel
                return False
            logger.warning("claim on %s is stale (its process is gone): taking it over", key)
            tmp = path.with_name(f"{path.name}.new-{os.getpid()}-{time.time_ns()}")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "start": process_start_time(os.getpid()),
                           "utc": datetime.now(timezone.utc).isoformat()}, fh)
            for _ in range(50):
                try:
                    os.replace(tmp, path)
                    return True
                except PermissionError:                         # Windows: a reader has it open for a moment
                    time.sleep(0.02)
            tmp.unlink()
            return False
        finally:
            try:
                sentinel.unlink()
            except OSError:
                pass

    def _after_stale_decision(self, key: str) -> None:
        """Seam for tests: called after a stale verdict, before the takeover."""

    def release(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except (FileNotFoundError, PermissionError):
            pass


__all__ = ["CLAIMS_DIR", "CellClaims", "pid_alive", "process_start_time"]
