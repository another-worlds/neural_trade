"""NT-185: an atomic write must survive a reader that holds the file open (Windows replace)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from neural_trade.utils import atomic
from neural_trade.utils.atomic import AtomicReplaceError, atomic_write_json, atomic_write_text, replace_with_retry

READER = textwrap.dedent('''
    import sys, time
    path, stop = sys.argv[1], sys.argv[2]
    import os
    n = 0
    while not os.path.exists(stop):
        try:
            with open(path, encoding="utf-8") as fh:
                fh.read()
            n += 1
        except OSError:
            pass
    print(n)
''')

WRITER = textwrap.dedent('''
    import json, sys
    path, n, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if mode == "bare":
        import os
        def write(p, obj):
            tmp = p + "." + str(os.getpid()) + ".tmp"
            open(tmp, "w", encoding="utf-8").write(json.dumps(obj))
            os.replace(tmp, p)
    else:
        from neural_trade.utils.atomic import atomic_write_json as write
    fails = 0
    for i in range(n):
        try:
            write(path, {"i": i, "pad": "x" * 2000})
        except OSError:
            fails += 1
    print(fails)
''')


def _hammer(tmp_path, mode, n=1000):
    target = tmp_path / "sweep.json"
    stop = tmp_path / "stop"
    atomic_write_text(target, "{}")
    reader = subprocess.Popen([sys.executable, "-c", READER, str(target), str(stop)], stdout=subprocess.PIPE, text=True)
    try:
        w = subprocess.run([sys.executable, "-c", WRITER, str(target), str(n), mode], capture_output=True, text=True,
                           timeout=300)
    finally:
        stop.write_text("x")
        reader.communicate(timeout=60)
    assert w.returncode == 0, w.stderr
    return int(w.stdout.strip())


@pytest.mark.skipif(os.name != "nt", reason="the Windows replace-while-open failure")
def test_a_reader_loop_cannot_make_the_writer_fail(tmp_path):
    assert _hammer(tmp_path, "atomic", n=150) == 0


@pytest.mark.slow
@pytest.mark.skipif(os.name != "nt", reason="the Windows replace-while-open failure")
def test_a_reader_loop_cannot_make_the_writer_fail_in_1000_writes(tmp_path):
    assert _hammer(tmp_path, "atomic", n=1000) == 0


@pytest.mark.skipif(os.name != "nt", reason="the Windows replace-while-open failure")
def test_the_bare_replace_does_fail_under_the_same_reader(tmp_path):
    """The reproduction: without the retry the same loop fails (so the test above is a real check)."""
    assert _hammer(tmp_path, "bare", n=300) > 0


def _flaky(monkeypatch, n_failures, exc=PermissionError(13, "busy")):
    calls = []
    real = os.replace

    def fake(src, dst):
        calls.append(1)
        if len(calls) <= n_failures:
            raise exc
        real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", fake)
    monkeypatch.setattr(atomic.time, "sleep", lambda s: None)
    return calls


def test_the_replace_is_retried_until_it_succeeds(tmp_path, monkeypatch):
    calls = _flaky(monkeypatch, 5)
    atomic_write_json(tmp_path / "a.json", {"k": 1})
    assert len(calls) == 6 and json.loads((tmp_path / "a.json").read_text()) == {"k": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_winerror_32_oserror_is_retried_too(tmp_path, monkeypatch):
    exc = OSError(32, "in use")
    exc.winerror = 32
    calls = _flaky(monkeypatch, 2, exc)
    atomic_write_text(tmp_path / "a.txt", "hi")
    assert len(calls) == 3


def test_a_replace_that_always_fails_raises_a_clear_error_and_leaves_no_temp(tmp_path, monkeypatch):
    calls = _flaky(monkeypatch, 10 ** 6)
    with pytest.raises(AtomicReplaceError, match="a.json"):
        atomic_write_json(tmp_path / "a.json", {"k": 1})
    assert len(calls) == atomic.REPLACE_ATTEMPTS
    assert list(tmp_path.glob("*.tmp")) == [] and not (tmp_path / "a.json").exists()


def test_another_oserror_is_not_retried(tmp_path, monkeypatch):
    calls = _flaky(monkeypatch, 10, FileNotFoundError(2, "gone"))
    with pytest.raises(FileNotFoundError):
        replace_with_retry(tmp_path / "x", tmp_path / "y")
    assert len(calls) == 1


def test_the_json_bytes_are_those_of_the_old_writer(tmp_path):
    obj = {"b": [1, 2.5, None], "a": {"x": "é"}, "p": tmp_path}
    atomic_write_json(tmp_path / "o.json", obj)
    assert (tmp_path / "o.json").read_bytes() == json.dumps(obj, indent=2, default=str).encode("utf-8")


def test_list_sweeps_tolerates_a_half_written_or_vanishing_summary(tmp_path):
    from neural_trade.notebook import panel_data as PD
    good = tmp_path / "sweeps" / "good"
    half = tmp_path / "sweeps" / "half"
    empty = tmp_path / "sweeps" / "empty"
    none = tmp_path / "sweeps" / "none"
    for d in (good, half, empty, none):
        d.mkdir(parents=True)
    atomic_write_json(good / "sweep.json", {"sweep_id": "good", "scenario": "s", "mode": "quick", "trials": []})
    (half / "sweep.json").write_text('{"sweep_id": "half", "trials": [', encoding="utf-8")
    (empty / "sweep.json").write_text("", encoding="utf-8")
    assert [s.sweep_id for s in PD.list_sweeps(tmp_path)] == ["good"]
    assert PD.list_sweeps(tmp_path / "missing") == []
