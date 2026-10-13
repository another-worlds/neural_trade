"""T02 (SPEC.md): volatility-timed long-only exposure to BTC. Library: loaders with the never-read span, hourly signals, strategies, metrics.

Conventions (reading of the SPEC, see REPORT.md):
- Hour k starts at T_k (minute 0). The exposure for hour k is decided with bars < T_k only and filled at the open of bar T_k
  (the next open after the last bar it saw). The hour return is open[T_{k+1}] / open[T_k] - 1.
- Cost: cost_bps * |w_k - w_{k-1}| deducted from hour k's return (turnover of the target weight; drift inside an hour ignored).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TAC, "common"))
import cutoff as C  # noqa: E402

CUT = pd.Timestamp(C.CUTOFF)                       # 2025-07-20 23:59
FWD_CSV = "D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv"
FWD_START = pd.Timestamp("2025-10-01 00:00")
FWD_END = pd.Timestamp("2026-10-01 00:00")         # last open used (the hour 2026-09-30 23:00 exits here)
NEVER_LO, NEVER_HI = pd.Timestamp("2025-07-21 00:00"), pd.Timestamp("2025-09-30 23:59")
TRAIN_DAYS = 365
RV_LOOK = (60, 240, 1440, 10080)                   # 1 h, 4 h, 24 h, 7 d
LABEL_H = 1440                                     # next-24-hour realised volatility
STRIDE_C = 7                                       # S1 vol-change model: training anchor stride (gaps SPEC study 3)
HOURS_PER_YEAR = 8760.0


# ------------------------------------------------------------------ data
def _grid(df):
    """Full 1-minute grid; a missing minute carries the previous close (zero return), open = previous close, volume 0."""
    full = pd.date_range(df.index[0], df.index[-1], freq="min")
    n_missing = len(full) - len(df)
    if n_missing:
        df = df.reindex(full)
        df["close"] = df["close"].ffill()
        df["open"] = df["open"].fillna(df["close"])
    return df, n_missing


def load_main():
    df = C.read_csv(usecols=["timestamp", "open", "close"])
    df.index = pd.to_datetime(df.pop("timestamp"))
    df = df.sort_index()
    C.assert_cut(df.index, "main")
    return _grid(df)


def load_fwd(path=FWD_CSV):
    df = pd.read_csv(path, usecols=["timestamp", "open", "close"])
    df.index = pd.to_datetime(df.pop("timestamp"))
    df = df.sort_index()
    return _grid(check_fwd(df))


def check_fwd(df):
    """Hard bound for the forward file: nothing in the never-read span; truncated at FWD_END."""
    if ((df.index >= NEVER_LO) & (df.index <= NEVER_HI)).any():
        raise AssertionError("forward file holds bars from the never-read span 2025-07-21..2025-09-30")
    df = df[(df.index >= FWD_START) & (df.index <= FWD_END)]
    if len(df) and df.index.min() < FWD_START:
        raise AssertionError("forward bar before 2025-10-01")
    return df


class Bars:
    """Arrays over a gap-free minute grid. cs2 = cumulative sum of squared 1-minute log returns (cs2[k] = sum of r[:k])."""

    def __init__(self, df):
        self.t = df.index.values.astype("datetime64[m]")
        self.index = df.index
        self.o = df["open"].values.astype(float)
        c = df["close"].values.astype(float)
        self.c = c
        lr = np.concatenate([[0.0], np.diff(np.log(c))])
        self.cs2 = np.concatenate([[0.0], np.cumsum(lr ** 2)])
        self.n = len(c)

    def rv(self, a, lo, hi):
        """sum of squared returns over bars a+lo .. a+hi-1 (relative offsets)."""
        return self.cs2[a + hi] - self.cs2[a + lo]


def hour_anchors(B):
    """Positions of full-hour bars with 7 days of history and the next hour's open present."""
    mins = B.index.minute.values
    p = np.where(mins == 0)[0]
    return p[(p >= RV_LOOK[-1]) & (p + 60 < B.n)]


# ------------------------------------------------------------------ features (bars < a only)
def har_feats(B, a):
    """log per-minute RV over 1h/4h/24h/7d plus hour-of-day sin/cos at anchors a (bars a-k .. a-1)."""
    cols = [np.log(B.rv(a, -k, 0) / k + 1e-12) for k in RV_LOOK]
    hr = B.index[a - 1].hour.values
    return np.column_stack(cols + [np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])


def s1_feats(B, a):
    """gaps study 3 S1: log RV15/60/240/1440 (per minute) and hour sin/cos."""
    cols = [np.log(B.rv(a, -k, 0) / k + 1e-12) for k in (15, 60, 240, 1440)]
    hr = B.index[a - 1].hour.values
    return np.column_stack(cols + [np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])


def _ridge(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    m = Ridge(alpha=1.0).fit((X - mu) / sd, y)
    return lambda Z: m.predict((Z - mu) / sd)


class Models:
    """The two models fitted on one training span (ta >= span_lo, label end <= span_hi) and the training-span distributions."""

    def __init__(self, B, span_lo, span_hi):
        """span_lo / span_hi: np.datetime64[m]; an anchor a is available at t[a-1]+1min = t[a]. Label ends at t[a] + label length."""
        self.span = (span_lo, span_hi)
        ta_all = B.t
        # (1) HAR 24 h model on hourly anchors
        ah = hour_anchors(B)
        ah = ah[(ah + LABEL_H < B.n)]
        okh = (ta_all[ah] >= span_lo) & (ta_all[ah] + np.timedelta64(LABEL_H, "m") <= span_hi)
        ah = ah[okh]
        assert len(ah) > 2000, len(ah)
        X = har_feats(B, ah)
        y = np.log(B.rv(ah, 0, LABEL_H) / LABEL_H + 1e-12)
        good = np.isfinite(y)
        self.har = _ridge(X[good], y[good])
        self.train_sig = np.exp(0.5 * self.har(X))                    # vol units (per-minute sigma)
        # (2) S1 vol-change model, stride 7 anchors, label = ln RV(a,a+60) - ln RV(a-60,a)
        ac = np.arange(1441, B.n - 60, STRIDE_C)
        okc = (ta_all[ac] >= span_lo) & (ta_all[ac] + np.timedelta64(60, "m") <= span_hi)
        ac = ac[okc]
        r0, r1 = B.rv(ac, -60, 0), B.rv(ac, 0, 60)
        pos = (r0 > 0) & (r1 > 0)
        ac = ac[pos]
        Xc = s1_feats(B, ac)
        yc = np.log(B.rv(ac, 0, 60) + 1e-12) - np.log(B.rv(ac, -60, 0) + 1e-12)
        self.s1 = _ridge(Xc, yc)
        self.n_c = len(ac)
        self.train_da = self.s1(s1_feats(B, ah))                      # in-sample predictions at the hourly training anchors

    def thresholds(self):
        q = np.quantile
        return {"sig_med": float(np.median(self.train_sig)),
                "sig_q70": float(q(self.train_sig, 0.7)), "sig_q80": float(q(self.train_sig, 0.8)), "sig_q90": float(q(self.train_sig, 0.9)),
                "da_q80": float(q(self.train_da, 0.8)), "da_q90": float(q(self.train_da, 0.9))}


def signals_for(B, models, a):
    """Per hour anchor a: sig_hat (HAR forecast, vol units) and da_hat (S1 predicted change), plus the models' thresholds repeated."""
    th = models.thresholds()
    out = {"sig_hat": np.exp(0.5 * models.har(har_feats(B, a))), "da_hat": models.s1(s1_feats(B, a))}
    for k, v in th.items():
        out[k] = np.full(len(a), v)
    return out


def monthly_signals(B, m_lo, m_hi, last_open_cut=None):
    """Refit each month from the 365 days before it; return a frame of hourly signals for months m_lo..m_hi (Period 'YYYY-MM')."""
    rows = []
    ah_all = hour_anchors(B)
    if last_open_cut is not None:
        ah_all = ah_all[B.t[ah_all] <= last_open_cut]
    for m in pd.period_range(m_lo, m_hi, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        s_lo = np.datetime64(m0 - pd.Timedelta(days=TRAIN_DAYS), "m")
        s_hi = np.datetime64(m0, "m")
        M = Models(B, s_lo, s_hi)
        sel = ah_all[(B.t[ah_all] >= np.datetime64(m0, "m")) & (B.t[ah_all] < np.datetime64(m1, "m"))]
        if len(sel) == 0:
            continue
        sg = signals_for(B, M, sel)
        sg["pos"] = sel
        sg["month"] = np.full(len(sel), str(m))
        rows.append(pd.DataFrame(sg))
    return pd.concat(rows, ignore_index=True)


def hour_frame(B, sig):
    """Attach the hour time, open and the next hour's open (return) to a signal frame (column pos)."""
    p = sig["pos"].values
    f = sig.copy()
    f["time"] = B.index[p]
    f["ret"] = B.o[p + 60] / B.o[p] - 1.0
    return f


# ------------------------------------------------------------------ strategies
CONFIGS = ([("a", k, None) for k in (0.5, 0.75, 1.0, 1.5)] + [("b", q, None) for q in (0.7, 0.8, 0.9)]
           + [("c", q, None) for q in (0.8, 0.9)] + [("ab", k, 0.2) for k in (0.75, 1.0, 1.5)])


def cname(cfg):
    r, p, band = cfg
    return {"a": f"a_k{p}", "b": f"b_q{p}", "c": f"c_q{p}", "ab": f"a_band_k{p}"}[r]


def exposure(f, cfg):
    """Target weights per hour from a signal frame (one row per hour, in time order)."""
    r, p, band = cfg
    if r in ("a", "ab"):
        w = np.minimum(1.0, p * f["sig_med"].values / f["sig_hat"].values)
    elif r == "b":
        w = np.where(f["sig_hat"].values > f[f"sig_q{int(round(p * 100))}"].values, 0.0, 1.0)
    else:
        w = np.where(f["da_hat"].values > f[f"da_q{int(round(p * 100))}"].values, 0.0, 1.0)
    if band is not None:
        out = np.empty_like(w)
        cur = w[0]
        for i, x in enumerate(w):
            if abs(x - cur) > band:
                cur = x
            out[i] = cur
        w = out
    return w


def strat_returns(w, ret, cost_bps):
    prev = np.concatenate([[w[0]], w[:-1]])           # the initial position is taken at no turnover cost
    turn = np.abs(w - prev)
    return w * ret - cost_bps * 1e-4 * turn, turn


def equity_stats(r):
    eq = np.cumprod(1.0 + r)
    n = len(r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = float(np.max(1.0 - eq / peak))
    total = float(eq[-1] - 1.0)
    ann = float(eq[-1] ** (HOURS_PER_YEAR / n) - 1.0) if eq[-1] > 0 else -1.0
    calmar = ann / dd if dd > 0 else float("nan")
    return {"total_return": total, "ann_return": ann, "max_dd": dd, "calmar": calmar}


def compare(f, cfg, cost_bps):
    """strategy, its constant null (c = mean w, no turnover) and buy-and-hold on one hourly frame."""
    w = exposure(f, cfg)
    r, turn = strat_returns(w, f["ret"].values, cost_bps)
    c = float(w.mean())
    rn = c * f["ret"].values
    rb = f["ret"].values
    return {"strategy": equity_stats(r), "null": equity_stats(rn), "bh": equity_stats(rb), "mean_w": c,
            "turnover_per_year": float(turn.sum() * HOURS_PER_YEAR / len(w))}, (w, r, rn, rb)


def calmar_of(r):
    return equity_stats(r)["calmar"]


def boot_calmar_diff(r_s, r_n, block=168, nboot=2000, seed=0):
    """Paired moving-block bootstrap (1-week blocks) of Calmar(strategy) - Calmar(null)."""
    rng = np.random.default_rng(seed)
    T = len(r_s)
    nb = int(np.ceil(T / block))
    d = []
    for _ in range(nboot):
        st = rng.integers(0, T - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:T]
        a, b = calmar_of(r_s[idx]), calmar_of(r_n[idx])
        if np.isfinite(a) and np.isfinite(b):
            d.append(a - b)
    d = np.array(d)
    return {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "mean": float(d.mean()), "n": int(len(d))}


def exposure_for_dd(r, cap=0.05):
    """Largest constant exposure c in [0, 1] (grid 0.005) with max DD of c * r below cap (buy-and-hold path)."""
    best = 0.0
    for c in np.arange(0.0, 1.0001, 0.005):
        if equity_stats(c * r)["max_dd"] < cap:
            best = float(c)
        else:
            break
    return best


def scale_for_dd(w, ret, cost_bps, cap=0.05):
    """Largest s in [0, 1] (grid 0.005) such that the strategy with weights s*w has max DD below cap."""
    best = 0.0
    for s in np.arange(0.0, 1.0001, 0.005):
        r, _ = strat_returns(s * w, ret, cost_bps)
        if equity_stats(r)["max_dd"] < cap:
            best = float(s)
        else:
            break
    return best
