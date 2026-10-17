"""T14 harness, part 1: data, the hourly UTC anchor grid and `bars_before`.

One anchor = a whole UTC hour `ta` (the minute the last bar closes = the start of the forecast hour). Every candidate view is computed on exactly
these anchors, from bars strictly before `ta`. The grid is the intersection of T11's V3 rows (T01 build with stride 1, thinned to the hour, its
`ok`) and T10's T = 60 rows (H3). T10 already drops anchors within 1,500 minutes of a file seam (cut file) or with a gap in the 1,441 bars
before / 60 after; we add `contig240` (no missing bar from 1,441 bars before to 240 after, which T01's 4-hour filter reads). `ok` is the
conjunction of both targets' own filters, contig240 and no seam; rows with ok False stay in the table (so a view can be aligned) but are never
trained on or scored.

  anchors_{source}.parquet   index `ta` (UTC, naive), columns pos, ok, okA, okB, contig240
  base_{source}.npz          the arrays of the unchanged T01/T11/T10 builds on those rows (loaded by evaluate.py)
  bars_{source}.parquet      the guarded 1-minute bars (file time), a cache of the CSV read
  anchors_{source}.json      counts (rows, ok, dropped by reason)

Sources: 'cut' (Bitstamp file cut at 2025-07-20 23:59; file time, converted to UTC by timefix) and 'fwd' (Binance forward file, UTC, to 2026-09-30
23:59). Never read: 2025-07-21..2025-09-30, anything after 2026-09-30 23:59, the original Bitcoin_BTCUSDT.csv (`check_range` refuses).
"""
import json
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _libs import T01, T10, T11, TF  # noqa: E402

CACHE_DIR = os.environ.get("T14_CACHE", "D:/nt/nt_t14_cache")
CUT_CSV, FWD_CSV = T01.TRAIN_CSV, T01.FWD_CSV
CUTOFF = pd.Timestamp("2025-07-20 23:59")
FWD_FIRST, FWD_LAST = pd.Timestamp("2025-10-01 00:00"), pd.Timestamp("2026-09-30 23:59")
NEVER_LO, NEVER_HI = pd.Timestamp("2025-07-21 00:00"), pd.Timestamp("2025-09-30 23:59:59")
CUT_LOAD_FROM = pd.Timestamp("2023-06-20 00:00")       # 365 days before Look 1 (2024-07) plus the 1,620-bar warm-up
SOURCES = {"cut": {"csv": CUT_CSV, "lo": CUT_LOAD_FROM, "hi": CUTOFF, "tf": "bitstamp"},
           "fwd": {"csv": FWD_CSV, "lo": FWD_FIRST, "hi": FWD_LAST, "tf": "binance"}}
LOOKS = {"look1": {"source": "cut", "first": pd.Period("2024-07", "M"), "last": pd.Period("2025-07", "M"), "train_from": pd.Timestamp("2023-07-01")},
         "look2": {"source": "fwd", "first": pd.Period("2025-11", "M"), "last": pd.Period("2026-09", "M"), "train_from": pd.Timestamp("2025-10-02")}}
H_A, H_B, X_B = 60, 60, 50                              # target A / B horizon (minutes) and B's barrier in bps
EPS_RV = 1e-12


class ForbiddenSpanError(ValueError):
    pass


def look_months(look):
    c = LOOKS[look]
    return list(pd.period_range(c["first"], c["last"], freq="M"))


def _src(source):
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}; use 'cut' or 'fwd'")
    return SOURCES[source]


def check_range(source, lo=None, hi=None):
    """Refuse any request that reaches into the never-read span or outside the source's allowed span (bars lo..hi, file time)."""
    s = _src(source); lo = s["lo"] if lo is None else pd.Timestamp(lo); hi = s["hi"] if hi is None else pd.Timestamp(hi)
    if lo <= NEVER_HI and hi >= NEVER_LO:
        raise ForbiddenSpanError(f"{source}: {lo}..{hi} touches the never-read span {NEVER_LO}..{NEVER_HI}")
    if hi > s["hi"]:
        raise ForbiddenSpanError(f"{source}: {hi} is after the last allowed bar {s['hi']}")
    if source == "fwd" and lo < FWD_FIRST:
        raise ForbiddenSpanError(f"fwd: {lo} is before {FWD_FIRST}")
    return lo, hi


def check_ta(source, ta):
    """An anchor close time `ta` (UTC) is allowed when the bars before it lie inside the allowed span: cut: ta up to the cut file's last close
    (2025-07-21 00:00 UTC-equivalent, file time + the 4-5 h offset still before the splice-free end), fwd: 2025-10-01 < ta <= 2026-10-01 00:00."""
    ta = pd.Timestamp(ta)
    if source == "cut":
        if ta > CUTOFF + pd.Timedelta(minutes=1):
            raise ForbiddenSpanError(f"cut: anchor {ta} is after the cut file's last bar {CUTOFF}")
    else:
        if ta <= FWD_FIRST or ta > FWD_LAST + pd.Timedelta(minutes=1):
            raise ForbiddenSpanError(f"fwd: anchor {ta} outside {FWD_FIRST}..{FWD_LAST}")
    return ta


# ------------------------------------------------------------------ bars
def _cache_path(name):
    return os.path.join(CACHE_DIR, name)


def load_bars(source, force=False):
    """The guarded 1-minute OHLCV of a source, indexed by bar open time in FILE time (cut: Bitstamp file time; fwd: UTC). Cached as parquet."""
    s = _src(source); check_range(source)
    p = _cache_path(f"bars_{source}.parquet")
    if os.path.exists(p) and not force:
        b = pd.read_parquet(p)
    else:
        b = T01.load_bars(s["csv"], s["lo"], s["hi"])
        os.makedirs(CACHE_DIR, exist_ok=True); b.to_parquet(p)
    b.index = pd.DatetimeIndex(b.index).astype("datetime64[ns]")
    assert b.index.min() >= s["lo"] and b.index.max() <= s["hi"]
    assert not ((b.index >= NEVER_LO) & (b.index <= NEVER_HI)).any()
    return b


class BarStore:
    """Bars of one source in UTC (duplicate UTC stamps from the cut file's daylight-saving seams are dropped, first kept) with
    `bars_before` (strictly before an anchor close) and the 5-minute resample / log-RV series."""

    def __init__(self, bars_file_time, source):
        self.source = source; tfn = _src(source)["tf"]
        df = bars_file_time[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.DatetimeIndex(TF.to_utc(df.index, tfn)).astype("datetime64[ns]")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        self.m1 = df; self._m5 = None; self._rv5 = None

    def m5(self):
        """5-minute OHLCV (UTC, bin open time) of complete bins only (5 one-minute bars)."""
        if self._m5 is None:
            g = self.m1.resample("5min")
            m5 = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            cnt = g["close"].count(); self._m5 = m5[(cnt == 5).values].astype(float)
        return self._m5

    def rv5(self):
        """5-minute realised variance series (UTC, bin open time): rv5 = sum of squared 1-minute log returns in the bin (a return needs the
        previous bar exactly one minute earlier), logrv5 = ln(rv5 + 1e-12), logrv_1h = ln(sum of the last 12 bins' rv5 + 1e-12) (the hour that
        ends with the bin; NaN unless the 12 bins are contiguous and complete)."""
        if self._rv5 is None:
            c = self.m1["close"]; idx = self.m1.index
            lr = np.log(c).diff(); lr[(idx.to_series().diff() != pd.Timedelta(minutes=1)).values] = np.nan
            g = (lr ** 2).resample("5min"); s = g.sum(min_count=5); s[(g.count() < 5).values] = np.nan
            out = pd.DataFrame({"rv5": s}); out["logrv5"] = np.log(out["rv5"] + EPS_RV)
            full = out["rv5"].reindex(pd.date_range(out.index.min(), out.index.max(), freq="5min"))
            r12 = full.rolling(12, min_periods=12).sum()
            out["logrv_1h"] = np.log(r12.reindex(out.index) + EPS_RV)
            self._rv5 = out
        return self._rv5

    def bars_before(self, ta, n_bars, freq="1min"):
        """The last `n_bars` OHLCV bars (UTC index) that closed at or before the anchor close `ta` (so no bar open time >= ta; for '5min' only
        bins whose last minute is before ta). Fewer rows than asked at the start of the source; gaps are not filled."""
        ta = pd.Timestamp(ta); check_ta(self.source, ta)
        if freq == "1min":
            df = self.m1; end = int(df.index.searchsorted(ta, side="left"))
        elif freq == "5min":
            df = self.m5(); end = int(df.index.searchsorted(ta - pd.Timedelta(minutes=5), side="right"))
        else:
            raise ValueError("freq must be '1min' or '5min'")
        out = df.iloc[max(0, end - n_bars):end].copy()
        assert len(out) == 0 or out.index[-1] < ta
        return out

    def rv5_before(self, ta, n_bars):
        """The last `n_bars` rows of `rv5()` whose bin closed at or before `ta`."""
        ta = pd.Timestamp(ta); check_ta(self.source, ta); df = self.rv5()
        end = int(df.index.searchsorted(ta - pd.Timedelta(minutes=5), side="right"))
        return df.iloc[max(0, end - n_bars):end].copy()


_STORES = {}


def get_store(source):
    if source not in _STORES:
        _STORES[source] = BarStore(load_bars(source), source)
    return _STORES[source]


def bars_before(source, ta, n_bars, freq="1min"):
    """OHLCV strictly before the anchor close `ta` (UTC) of `source`: the last `n_bars` 1-minute (or complete 5-minute) bars, UTC index."""
    return get_store(source).bars_before(ta, n_bars, freq)


def rv5_before(source, ta, n_bars):
    return get_store(source).rv5_before(ta, n_bars)


def rv5_series(source, force=False):
    """The cached 5-minute log-RV series (columns rv5, logrv5, logrv_1h; UTC bin open time): D:/nt/nt_t14_cache/rv5_{source}.parquet."""
    p = _cache_path(f"rv5_{source}.parquet")
    if os.path.exists(p) and not force:
        return pd.read_parquet(p)
    r = get_store(source).rv5(); os.makedirs(CACHE_DIR, exist_ok=True); r.to_parquet(p); return r


# ------------------------------------------------------------------ the grid and the unchanged baselines' arrays
def build_hourly(bars, source):
    """T11's V3 inputs (T01 build, stride 1 thinned to the UTC hour) and T10's T = 60 build on one bar frame (file time), joined on the hourly
    positions. Returns (ta_utc, arrays, info); `arrays` is a flat dict of numpy arrays (see `base_from_arrays`)."""
    s = _src(source); tfn = s["tf"]
    old = T01.STRIDE; T01.STRIDE = 1                    # T01.build reads the module global; the code itself is unchanged
    try:
        BA = T11.build(bars, T01.WARM, tfn)
    finally:
        T01.STRIDE = old
    D = BA["D"]; posA = D.anch; ta_utc_all = pd.DatetimeIndex(TF.to_utc(D.ta, tfn))
    hourly = np.where(np.asarray(ta_utc_all.minute == 0))[0]
    P = T10.prep(bars, source); BB = T10.build(P, H_B); posB = BB["a"]
    common, ja, jb = np.intersect1d(posA[hourly], posB, return_indices=True)
    ia = hourly[ja]
    ta = pd.DatetimeIndex(ta_utc_all[ia]).astype("datetime64[ns]")
    assert (np.asarray(pd.DatetimeIndex(BB["D"].ta[jb]).astype("datetime64[ns]")) == np.asarray(ta)).all(), "T10 and T11 disagree on the anchor time"
    a = common
    contig = (P.tmin[a + 240 - 1] - P.tmin[a - T10.LOOKBACK - 1]) == (240 + T10.LOOKBACK)
    okA = BA["ok"][ia]; okB = BB["ok"][jb]
    seam = P.seam[a - 1]
    ok = okA & okB & contig & ~seam
    cal = BA["cal"]; F = BA["F"]["S3"]
    arr = {"pos": a.astype(np.int64), "ok": ok, "okA": okA, "okB": okB, "contig240": contig,
           "A_S3": F[ia], "A_dA": BA["T"]["A"][ia], "A_AR": BA["AR"]["A"][ia], "A_LV0": BA["LV0"][ia], "A_har": D.har[ia],
           "A_hr_sc": cal["hr_sc"][ia], "A_ny": cal["ny"][ia], "A_slot": cal["slot"][ia], "A_r15": BA["r15"][ia], "A_v15": BA["v15"][ia],
           "B_har": BB["D"].har[jb], "B_ex10": BB["ex10"][jb], "B_harm": BB["harm"][jb], "B_trail": BB["trail"][H_B][jb], "B_ew": BB["ew"][jb],
           "B_r": BB["r"][H_B][jb], "B_move": BB["move"][jb], "B_close0": BB["close0"][jb],
           **{f"B_y{X}": BB["y"][X][jb] for X in T10.XS}}
    info = {"source": source, "n_bars": int(len(bars)), "first_bar": str(bars.index.min()), "last_bar": str(bars.index.max()),
            "n_hourly_A": int(len(hourly)), "n_T10_rows": int(len(posB)), "n_rows": int(len(a)), "n_okA": int(okA.sum()), "n_okB": int(okB.sum()),
            "n_noncontig240": int((~contig).sum()), "n_seam": int(seam.sum()), "n_ok": int(ok.sum()),
            "t11_drops": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in BA["drops"].items()}}
    return ta, arr, info


def base_from_arrays(ta, arr):
    """Rebuild the dict shapes the unchanged T11 / T10 functions read (B['D'].ta is the UTC anchor time in both)."""
    ta = pd.DatetimeIndex(ta)
    A = {"D": types.SimpleNamespace(ta=ta, har=arr["A_har"]), "F": {"S3": arr["A_S3"]}, "T": {"A": arr["A_dA"]}, "AR": {"A": arr["A_AR"]},
         "LV0": arr["A_LV0"], "ok": arr["okA"], "cal": {"hr_sc": arr["A_hr_sc"], "ny": arr["A_ny"], "slot": arr["A_slot"]},
         "r15": arr["A_r15"], "v15": arr["A_v15"], "drops": {}}
    B = {"D": types.SimpleNamespace(har=arr["B_har"], ta=ta), "T": H_B, "a": arr["pos"], "ok": arr["okB"], "trail": {H_B: arr["B_trail"]},
         "ew": arr["B_ew"], "ex10": arr["B_ex10"], "harm": arr["B_harm"], "r": {H_B: arr["B_r"]}, "move": arr["B_move"],
         "y": {X: arr[f"B_y{X}"] for X in T10.XS}, "close0": arr["B_close0"], "tF": ta}
    return A, B


def anchors_frame(ta, arr):
    return pd.DataFrame({"pos": arr["pos"], "ok": arr["ok"], "okA": arr["okA"], "okB": arr["okB"], "contig240": arr["contig240"]},
                        index=pd.DatetimeIndex(ta, name="ta"))


def build_all(source, force=False):
    """Build and save anchors_{source}.parquet, base_{source}.npz and anchors_{source}.json. Deterministic."""
    bars = load_bars(source, force=force); ta, arr, info = build_hourly(bars, source)
    os.makedirs(CACHE_DIR, exist_ok=True)
    anchors_frame(ta, arr).to_parquet(_cache_path(f"anchors_{source}.parquet"))
    np.savez_compressed(_cache_path(f"base_{source}.npz"), ta_ns=np.asarray(ta).astype("datetime64[ns]").astype(np.int64), **arr)
    json.dump(info, open(_cache_path(f"anchors_{source}.json"), "w"), indent=1)
    return ta, arr, info


class Base:
    """The saved baseline arrays of one source: .ta (UTC anchors), .ok, .A / .B (dicts for T11 / T10 functions), .arr (flat arrays)."""

    def __init__(self, ta, arr):
        self.ta = pd.DatetimeIndex(ta); self.arr = arr; self.ok = arr["ok"].copy(); self.A, self.B = base_from_arrays(self.ta, arr)

    def __len__(self):
        return len(self.ta)


def load_anchors(source):
    """The anchor table (index `ta`, UTC). Built with `build_all(source)` on first use."""
    p = _cache_path(f"anchors_{source}.parquet")
    if not os.path.exists(p):
        build_all(source)
    df = pd.read_parquet(p); df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]"); df.index.name = "ta"; return df


def load_base(source):
    p = _cache_path(f"base_{source}.npz")
    if not os.path.exists(p):
        build_all(source)
    z = np.load(p); arr = {k: z[k] for k in z.files if k != "ta_ns"}
    return Base(pd.DatetimeIndex(z["ta_ns"].astype("datetime64[ns]")), arr)


if __name__ == "__main__":
    for src in [a for a in sys.argv[1:] if not a.startswith("--")] or ["cut", "fwd"]:
        _, _, i = build_all(src, force="--force" in sys.argv)
        print(src, json.dumps(i))
