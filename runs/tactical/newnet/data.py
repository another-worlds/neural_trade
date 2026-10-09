"""New network, data layer (PLAN 10). Everything is computed once per BAR from the continuous minute file, causally (a bar's
features read only bars up to and including it), and windows are built on the fly by indexing: no 60-bar window array is ever
stored for a long span.

  per-bar arrays (cache/series_vec.npy, series_seq.npy, float16, built once for the whole file, memory-mapped on use):
    vec [T, 60]  the last-bar vector: lab `rich` (43, lab.fs_rich on the 60-bar window ending at the bar: identical definitions)
                 + 17 context features beyond the window (returns over 60/240/1440/10080 bars over volatility, log realised
                 volatility at those scales, 3 scale ratios, position in the 1-day and 1-week range, time of day and day of week sin/cos)
    seq [T, 8]   the sequence channels: 1-bar change, close-EMA5, close-EMA20, range, body (all over a trailing 60-bar sd of the
                 1-bar change), log1p(volume / trailing 60-bar EMA volume), RSI14, Stoch14
  window i (start bar i) = bars i .. i+59, its last bar is i+59, labels r[i, h] = close[i+59+h] / close[i+59] - 1, h in 10, 15, 20.

  per (slice, span) (cache/<slice>_<span>.npz, small): the split and the standardisation statistics of the TRAINING span only.

The val block of a slice is the lab cache's (located in the series and asserted equal, windows and labels); the training span ends
81 bars before the val block's first window (the cache's purge gap; longtrain.py uses the same), grows backwards over `span` windows
(clamped at the data start + a 1,440-bar warm-up; the actual count is recorded).

Deviation from PLAN 10 "cached per (slice, span)": the per-bar features do not depend on the slice (they are causal), so they are
cached ONCE for the whole file (about 0.6 GB in float16) instead of once per slice and span (24 x up to 0.2 GB); only the
standardisation statistics are per (slice, span)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
LAB_CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache")
FINAL_CACHE = os.path.join(HERE, "..", "lab", "cache_final")      # newnet2: the held-out FINAL slices' val blocks (lab/cache2.py --final)

WIN = 60
HZ = (10, 15, 20)
GAPB = 81                      # last training window starts 81 bars before the val block's first window (as longtrain.py)
INNER, INNER_GAP = 0.15, 80    # inner early-stopping split: the last 15% of the train span, 80 windows apart
WARM = 1440                    # bars of history before the first training window (one day: the 1-week context features are expanding-window over the first week of the file)
SPANS = {"11.5d": 16560, "90d": 129600, "1y": 525600, "3y": 1576800}   # windows, 1-minute stride
CLIMB6 = ["2017-05-02T12", "2018-11-16T02", "2020-03-30T04", "2021-10-13T18", "2023-04-29T08", "2024-11-11T23"]
CTX_SCALES = (60, 240, 1440, 10080)
N_RICH, N_CTX, N_SEQ = 43, 17, 8
N_VEC = N_RICH + N_CTX
IDX_LOGVOL60 = N_RICH + 4      # the context column holding log realised volatility over the last 60 bars (the trailing-vol baseline)
CLIP = 8.0                     # standardised features are winsorised at +-CLIP
F16 = 6e4


def slice_names(n):
    """n: 6 (the climb screen), 24 (all) or 'final' (the held-out slices; their blocks live in FINAL_CACHE and Split gets it)."""
    n = str(n)
    if n == "final":
        return sorted(f[:-4] for f in os.listdir(FINAL_CACHE) if f.endswith(".npz"))
    allp = sorted(f[:-4] for f in os.listdir(LAB_CACHE) if f.endswith(".npz"))
    return [s for s in allp if s in CLIMB6] if n == "6" else allp


# ---------------------------------------------------------------- the per-bar features
def load_series(csv=CSV):
    """OHLCV [T, 5] float64 and the minute-of-day / day-of-week of each bar (a continuous 1-minute file)."""
    import pandas as pd
    d = pd.read_csv(csv, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    t = pd.to_datetime(d["timestamp"])
    return (d[["open", "high", "low", "close", "volume"]].to_numpy(np.float64),
            (t.dt.hour * 60 + t.dt.minute).to_numpy(np.float64), t.dt.dayofweek.to_numpy(np.float64))


def _ema(x, p):
    from scipy.signal import lfilter
    a = 2.0 / (p + 1.0)
    return lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[0]])[0]


def _roll(x, k, fn):
    import pandas as pd
    return getattr(pd.Series(x).rolling(k, min_periods=1), fn)().to_numpy()


def _trailing(x, k):
    """Causal trailing sums of x over the last k bars (fewer at the start): (sum, count)."""
    cs = np.concatenate([[0.0], np.cumsum(x)]); t = np.arange(1, len(x) + 1)
    lo = np.maximum(t - k, 0)
    return cs[t] - cs[lo], (t - lo).astype(np.float64)


def context_features(A, tod, dow):
    """[T, 17] float64, causal: 4 volatility-scaled returns, 4 log realised vols, 3 scale ratios, 2 range positions, 4 time."""
    c = A[:, 3]; lc = np.log(np.maximum(c, 1e-9)); lr = np.concatenate([[0.0], np.diff(lc)])
    t = np.arange(len(c)); rets, logvol = [], []
    for k in CTX_SCALES:
        _, n = _trailing(lr, k); s2, _ = _trailing(lr * lr, k)
        vol = np.sqrt(np.maximum(s2 / n, 0.0)) + 1e-7
        ret = lc - lc[np.maximum(t - k, 0)]
        rets.append(np.clip(ret / (vol * np.sqrt(np.maximum(np.minimum(t, k), 1))), -20, 20)); logvol.append(np.log(vol))
    out = rets + logvol + [logvol[i] - logvol[i + 1] for i in range(3)]
    for k in (1440, 10080):
        hi, lo = _roll(A[:, 1], k, "max"), _roll(A[:, 2], k, "min"); out.append((c - lo) / (hi - lo + 1e-9) - 0.5)
    ph = 2 * np.pi * tod / 1440.0; pw = 2 * np.pi * (dow + tod / 1440.0) / 7.0
    out += [np.sin(ph), np.cos(ph), np.sin(pw), np.cos(pw)]
    return np.stack(out, 1)


def seq_features(A):
    """[T, 8] float64, causal."""
    o, h, l, c, v = (A[:, i] for i in range(5)); d = np.concatenate([[0.0], np.diff(c)])
    s1, n = _trailing(d, WIN); s2, _ = _trailing(d * d, WIN)
    sd = np.maximum(np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0)), 1e-5 * c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    rsi = (100.0 - 100.0 / (1.0 + _ema(up, 14) / (_ema(dn, 14) + 1e-12)) - 50.0) / 50.0
    lo14, hi14 = _roll(l, 14, "min"), _roll(h, 14, "max")
    return np.stack([d / sd, (c - _ema(c, 5)) / sd, (c - _ema(c, 20)) / sd, (h - l) / sd, (c - o) / sd,
                     np.log1p(v / (_ema(v, WIN) + 1e-9)), rsi, (c - lo14) / (hi14 - lo14 + 1e-12) - 0.5], 1)


def _f16(x):
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=F16, neginf=-F16), -F16, F16).astype(np.float16)


def rich_features(A, lab, chunk=20000, log=None):
    """[T, 43] float16: lab.fs_rich on the 60-bar window ending at each bar (rows before bar 59 are 0)."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.zeros((len(A), N_RICH), np.float16)
    for a in range(WIN - 1, len(A), chunk):
        b = min(a + chunk, len(A)); W = sliding_window_view(A[a - WIN + 1:b], WIN, axis=0).transpose(0, 2, 1)
        out[a:b] = _f16(lab.fs_rich(np.ascontiguousarray(W).astype(np.float32)))
        if log and (a // chunk) % 40 == 0:
            log(f"rich {a}/{len(A)}")
    return out


def bar_arrays(A, tod, dow, lab, log=None):
    """The per-bar arrays for a series: vec [T, 60] and seq [T, 8], both float16, causal."""
    vec = np.concatenate([rich_features(A, lab, log=log), _f16(context_features(A, tod, dow))], 1)
    return vec, _f16(seq_features(A))


def build_series_cache(lab, log=print):
    """Compute the per-bar arrays for the whole file once and store them (float16 .npy)."""
    os.makedirs(CACHE, exist_ok=True)
    A, tod, dow = load_series()
    vec, seq = bar_arrays(A, tod, dow, lab, log)
    np.save(f"{CACHE}/series_vec.npy", vec); np.save(f"{CACHE}/series_seq.npy", seq)
    np.save(f"{CACHE}/series_close.npy", A[:, 3])
    log("series cache written")


class Series:
    """The memory-mapped per-bar arrays. `from_arrays` builds the same object from in-memory arrays (tests)."""
    def __init__(self, vec=None, seq=None, close=None):
        if vec is None:
            if not os.path.exists(f"{CACHE}/series_close.npy"):
                raise RuntimeError("series cache missing: run `python data.py --build`")
            vec, seq, close = (np.load(f"{CACHE}/series_{k}.npy", mmap_mode="r") for k in ("vec", "seq", "close"))
        self.vec, self.seq, self.close = vec, seq, close
        self._ar = np.arange(WIN)

    def labels(self, starts):
        last = np.asarray(starts) + WIN - 1
        return np.stack([self.close[last + h] / self.close[last] - 1.0 for h in HZ], 1).astype(np.float32)

    def raw_vec(self, starts):
        return np.asarray(self.vec[np.asarray(starts) + WIN - 1], np.float32)

    def raw_seq(self, starts):
        return np.asarray(self.seq[np.asarray(starts)[:, None] + self._ar], np.float32)


# ---------------------------------------------------------------- slices, splits, standardisation
def locate_val(close, Wva):
    """Series bar index where the lab cache's first val window starts (closes of the first and last val window must match)."""
    n = len(Wva)
    for i in np.where(np.abs(np.asarray(close) - float(Wva[0, 0, 3])) < 1e-2)[0]:
        if i + n + WIN <= len(close) and np.allclose(close[i:i + WIN], Wva[0, :, 3], rtol=1e-5, atol=1e-6) \
                and np.allclose(close[i + n - 1:i + n - 1 + WIN], Wva[-1, :, 3], rtol=1e-5, atol=1e-6):
            return int(i)
    raise RuntimeError("the cache's val block was not found in the series")


class Split:
    """One (slice, span): training window starts s0 .. s0+n-1 (n recorded), val windows vs .. vs+nva-1 (the lab cache's block),
    the inner early-stopping split (itr / iva are positions in the span) and the training-span standardisation."""
    def __init__(self, name, span, series, lab_cache=None):
        lab_cache = lab_cache or (FINAL_CACHE if os.path.exists(f"{FINAL_CACHE}/{name}.npz") else LAB_CACHE)
        self.name, self.span, self.series = name, span, series
        D = np.load(f"{lab_cache}/{name}.npz")
        self.deadband = float(D["deadband"]); self.rva = D["rva"]; self.nva = len(self.rva)
        self.vs = locate_val(series.close, D["Wva"])
        avail = self.vs - GAPB - (WARM - WIN + 1) + 1                  # window starts WARM-59 .. vs-81 (the context warm-up)
        self.n = int(min(SPANS[span], avail)); self.s0 = self.vs - GAPB - (self.n - 1)
        self.full = self.n == SPANS[span]
        self.cut = int(round((1 - INNER) * self.n))
        self.itr = np.arange(0, self.cut - INNER_GAP); self.iva = np.arange(self.cut, self.n)
        self.va_starts = self.vs + np.arange(self.nva); self.tr_starts = self.s0 + np.arange(self.n)
        self._stats(series)

    def check_val_block(self, A, lab_cache=None):
        """Assert the val windows (OHLCV from the file A [T, 5]) and labels equal the lab cache's."""
        from numpy.lib.stride_tricks import sliding_window_view
        D = np.load(f"{lab_cache or LAB_CACHE}/{self.name}.npz")
        W = sliding_window_view(A[self.vs:self.vs + self.nva + WIN - 1], WIN, axis=0).transpose(0, 2, 1)
        assert np.allclose(W.astype(np.float32), D["Wva"], rtol=1e-5, atol=1e-6), "Wva differs from the series"
        assert np.allclose(self.series.labels(self.va_starts), self.rva, rtol=1e-4, atol=2e-6), "rva differs from the series"
        return True

    def _stats(self, series):
        p = f"{CACHE}/{self.name}_{self.span}.npz"
        if os.path.exists(p):
            z = np.load(p)
            if int(z["s0"]) == self.s0 and int(z["n"]) == self.n:
                self.mu_v, self.sd_v, self.mu_s, self.sd_s, self.mu_vol = z["mu_v"], z["sd_v"], z["mu_s"], z["sd_s"], z["mu_vol"]
                return
        k = max(1, -(-self.n // 300000)); st = self.tr_starts[::k]
        V = series.raw_vec(st); self.mu_v, self.sd_v = V.mean(0), V.std(0) + 1e-6
        Sq = np.asarray(series.seq[self.s0:self.s0 + self.n + WIN - 1:k], np.float32)
        self.mu_s, self.sd_s = Sq.mean(0), Sq.std(0) + 1e-6
        self.mu_vol = np.log(np.abs(series.labels(st)) + 1e-5).mean(0)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(p, mu_v=self.mu_v, sd_v=self.sd_v, mu_s=self.mu_s, sd_s=self.sd_s, mu_vol=self.mu_vol,
                 s0=self.s0, n=self.n, vs=self.vs, nva=self.nva)

    # batches, from window starts
    def vec(self, starts):
        return np.clip((self.series.raw_vec(starts) - self.mu_v) / self.sd_v, -CLIP, CLIP).astype(np.float32)

    def seq(self, starts):
        return np.clip((self.series.raw_seq(starts) - self.mu_s) / self.sd_s, -CLIP, CLIP).astype(np.float32)

    def labels(self, starts):
        """(returns [B,3], up [B,3] float, mask [B,3] float: |return| above the deadband)."""
        r = self.series.labels(starts)
        return r, (r > 0).astype(np.float32), (np.abs(r) > self.deadband).astype(np.float32)


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        sys.path.insert(0, os.path.join(HERE, "..", "lab"))
        import lab
        build_series_cache(lab)
