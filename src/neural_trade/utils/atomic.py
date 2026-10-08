"""Atomic file replacement that survives a reader on Windows (NT-185).

On Windows ``os.replace`` fails with PermissionError / WinError 5 or 32 while another process holds the
target open (an open control-panel notebook polling a sweep's summary). The replace is retried with a short
exponential backoff; after about two seconds a clear error names the path.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, "os.PathLike[str]"]

REPLACE_ATTEMPTS = 20
REPLACE_FIRST_DELAY_S = 0.010
REPLACE_MAX_DELAY_S = 0.200

_counter = itertools.count()


class AtomicReplaceError(OSError):
    """``os.replace`` kept failing for every retry; the message names the path."""


class ReadOnlyTargetError(PermissionError):
    """The replace target carries the read-only attribute: a deterministic denial (WinError 5 as well), so it is
    raised at once instead of being retried, and it carries no ``winerror`` so a caller can tell it from a lock."""


def _is_busy(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32)


def _is_read_only(dst: PathLike) -> bool:
    """True when ``dst`` exists and is not writable (the read-only attribute on Windows). An ACL denial is not
    detected: it looks like a lock and is retried (docs/RUNBOOK.md "Not a verdict")."""
    try:
        return os.path.isfile(dst) and not os.access(dst, os.W_OK)
    except OSError:
        return False


def replace_with_retry(src: PathLike, dst: PathLike) -> None:
    """``os.replace(src, dst)``, retried while the target is busy (PermissionError, WinError 5 / 32)."""
    delay = REPLACE_FIRST_DELAY_S
    last: OSError | None = None
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            if not _is_busy(exc):
                raise
            if _is_read_only(dst):          # NT-199: no retry will change a read-only target
                raise ReadOnlyTargetError(13, f"the target is read-only: {dst}") from exc
            last = exc
        if attempt < REPLACE_ATTEMPTS - 1:
            time.sleep(delay)
            delay = min(delay * 2, REPLACE_MAX_DELAY_S)
    raise AtomicReplaceError(f"could not replace {dst} after {REPLACE_ATTEMPTS} attempts "
                             f"(another process holds it open?): {last}") from last


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8", newline: str | None = "\n") -> None:
    """Write ``text`` to a temp file next to ``path`` (unique per pid and call), then replace ``path`` with it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{next(_counter)}.tmp")
    try:
        tmp.write_text(text, encoding=encoding, newline=newline)
        replace_with_retry(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: PathLike, obj: Any, *, indent: int | None = 2) -> None:
    """``atomic_write_text`` of ``json.dumps(obj, indent=indent, default=str)``."""
    atomic_write_text(path, json.dumps(obj, indent=indent, default=str))


__all__ = ["AtomicReplaceError", "ReadOnlyTargetError", "atomic_write_json", "atomic_write_text", "replace_with_retry"]
