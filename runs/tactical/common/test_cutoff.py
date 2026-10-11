"""H66 recut guard: no recomputed script reads a bar after common/cutoff.py's CUTOFF.
Static: every recomputed script points at C.CSV (the cut copy) and never at the raw file.
Dynamic: each data loader the recomputed scripts use returns bars whose last timestamp is <= CUTOFF (run in a subprocess:
several scripts chdir into the checkout on import).
usage: python -m pytest runs/tactical/common/test_cutoff.py -q"""
import os, re, subprocess, sys
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE); sys.path.insert(0, HERE)
import cutoff as C  # noqa: E402

# every script whose result is recomputed in the recut (H66)
RECOMPUTED = ["hourly/hourly.py", "regimes/part2.py", "geom2/meta.py", "geom2/meta_nextopen.py",
              "lab/decay.py", "lab/decay_lib.py", "lab/decay_nextopen.py", "lab/decay_vol.py", "lab/decay_asym.py",
              "recent/feats.py", "nextopen/nolib.py", "volnet/volnet_lib.py", "calendar/callib.py", "calendar/run_calendar.py",
              "flow/flowlib.py", "fwd/fwdlib.py"]
RAW = re.compile(r"neural_trade/Bitcoin_BTCUSDT\.csv|neural_trade\\Bitcoin_BTCUSDT\.csv")
PY = sys.executable


def test_cut_csv_stops_at_the_cutoff():
    with open(C.CSV, "rb") as f:
        f.seek(-200, 2); last = f.read().decode().strip().splitlines()[-1]
    assert last[:16] == C.CUTOFF, last
    ts = pd.read_csv(C.CSV, usecols=["timestamp"])["timestamp"]
    assert ts.iloc[-1] <= C.CUTOFF + ":59" and (ts.iloc[-5:] < "2025-07-21").all()
    assert ts.is_monotonic_increasing and ts.iloc[0].startswith("2017-01-01")


def test_assert_cut_raises_for_a_later_bar():
    C.assert_cut(pd.Series(["2025-07-20 23:59:00"]))
    with pytest.raises(AssertionError):
        C.assert_cut(pd.Series(["2025-07-20 23:59:00", "2025-07-21 00:00:00"]))
    with pytest.raises(AssertionError):
        C.assert_cut(pd.DataFrame({"timestamp": ["2025-09-29 23:57:00"]}))


def test_read_csv_refuses_any_other_file_and_is_bounded():
    with pytest.raises(ValueError):
        C.read_csv("D:/somewhere/else.csv")
    d = C.read_csv(usecols=["timestamp", "close"])
    assert pd.to_datetime(d["timestamp"]).max() <= pd.Timestamp(C.CUTOFF)
    mx = max(pd.to_datetime(ch["timestamp"]).max() for ch in C.read_csv(usecols=["timestamp", "close"], chunksize=500_000))
    assert mx <= pd.Timestamp(C.CUTOFF)


@pytest.mark.parametrize("rel", RECOMPUTED)
def test_script_reads_only_the_cut_file(rel):
    src = open(os.path.join(TAC, rel), encoding="utf-8").read()
    assert not RAW.search(src), f"{rel} still names the raw CSV"
    if "read_csv" in src or "C.CSV" in src:
        assert "import cutoff as C" in src and "C.CSV" in src or "C.read_csv" in src or "callib" in rel, rel


def _last(code):
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=TAC, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8"))
    assert r.returncode == 0, r.stderr[-800:]
    return pd.Timestamp(r.stdout.strip().splitlines()[-1])


LOADERS = {
    "decay_lib": "import sys; sys.path.insert(0,'lab'); import decay_lib as D; print(D.load_bars('1m').index.max())",
    "decay_lib_1h": "import sys; sys.path.insert(0,'lab'); import decay_lib as D; print(D.load_bars('1h').index.max())",
    "nolib": "import sys; sys.path.insert(0,'nextopen'); import nolib as N; print(N.load_bars('1m').index.max())",
    "volnet_lib": "import sys; sys.path.insert(0,'volnet'); import volnet_lib as V; print(V.load_bars().index.max())",
    "flowlib": "import sys; sys.path.insert(0,'flow'); import flowlib as F; print(F.load_bars('1m').index.max())",
    "recent_feats": "import sys; sys.path.insert(0,'recent'); import feats as F; print(F.load_bars('1m').index.max())",
    "callib": "import sys; sys.path.insert(0,'calendar'); import callib as K, cutoff; print(K.hourly_returns(cutoff.CSV).index.max())",
    "geom2_meta": "import sys; sys.path.insert(0,'geom2'); import meta as M; M.SETUP['1min_full'] = dict(M.SETUP['1min'], end='2100-01-01'); print(M.bars('1min_full').index.max())",
    "geom2_meta_nextopen": "import sys; sys.path.insert(0,'geom2'); import meta_nextopen as M; M.SETUP['1min_full'] = dict(M.SETUP['1min'], end='2100-01-01'); print(M.bars('1min_full').index.max())",
    "hourly": "import sys; sys.path.insert(0,'hourly'); import hourly as H; print(H.bars().index.max())",
}


@pytest.mark.parametrize("name", list(LOADERS))
def test_loader_returns_no_bar_after_the_cutoff(name):
    pre = "import sys; sys.path.insert(0, r'" + HERE.replace("\\", "/") + "'); "
    last = _last(pre + LOADERS[name])
    assert last <= pd.Timestamp(C.CUTOFF), (name, last)
    assert last >= pd.Timestamp("2025-07-20 00:00"), (name, "loader stops too early", last)
