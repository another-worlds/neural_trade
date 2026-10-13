"""T07 (SPEC.md): an ensemble TRADING strategy. Library: guarded loaders, the minute grid, member position builders (pure functions of
predictions or prices), combinations, cost accounting on turnover, metrics, the random-side null.

Conventions (reading of the SPEC, see REPORT in the hand-back):
- Everything is accounted on a 1-minute grid. A member is a signed position per minute (int8 / float). The minute return is
  r[j] = open[j+1] / open[j] - 1 (the last minute of a data span uses its own close). Sums of simple minute returns carry no bias against
  the open-to-open hour return (the cross terms are zero mean).
- A signal is made from bars that closed before the entry minute and is filled at the OPEN of the next bar (next-open fills; no stops).
- A cost of c bps per ROUND TRIP is charged as c/2 bps per unit of turnover |w[j] - w[j-1]| (entry 0 -> 1 costs c/2, exit costs c/2).
- Hourly net return = sum over the hour's minutes of w * r - c/2 * 1e-4 * |dw|. Equity = cumprod(1 + hourly net).
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TAC, "common"))
import cutoff as C  # noqa: E402

MAIN_CSV = C.CSV
FWD_CSV = "D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv"
SEL_FIRST, SEL_LAST = "2019-01", "2024-06"
SEL_END = pd.Timestamp("2024-07-01")                  # exclusive: no bar at or after it is read by the selection stage
LOOK_FIRST, LOOK_END = "2024-07", pd.Timestamp(C.CUTOFF)
FWD_START, FWD_LAST = pd.Timestamp("2025-10-01 00:00"), pd.Timestamp("2026-09-30 23:59")
NEVER_LO, NEVER_HI = pd.Timestamp("2025-07-21 00:00"), pd.Timestamp("2025-09-30 23:59")
FWD_CAL, FWD_SCORED = "2025-10", [str(m) for m in pd.period_range("2025-11", "2026-09", freq="M")]
COSTS = (0, 1, 2, 5)
VERDICT_COST = 2
HOURS_PER_YEAR = 8760.0
Q = 0.9                                                # top-10% confidence quantile
NZ = 0.02                                              # |w| above this counts as a non-zero position (trades, runs, vote)
MEMBERS = ("M1", "M2", "M3", "M4", "M5", "M6")
COMBOS = ("C1", "C2", "C3", "C4", "C5", "C6")
CACHE = os.environ.get("T07_CACHE", "D:/nt/nt_t07_cache")


# ------------------------------------------------------------------ guarded loaders
def read_main(end_excl=None):
    """Raw 1-minute rows of the cut main file, strictly before end_excl (default: the whole cut file)."""
    d = C.read_csv(usecols=["timestamp", "open", "high", "low", "close", "volume"])
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if end_excl is not None:
        d = d[d["timestamp"] < pd.Timestamp(end_excl)]
    assert d["timestamp"].max() <= pd.Timestamp(C.CUTOFF)
    return d.set_index("timestamp").sort_index()


def check_fwd(df):
    """The forward file: refuse any bar of the never-read span; keep 2025-10-01 .. 2026-09-30 23:59 only."""
    if ((df.index >= NEVER_LO) & (df.index <= NEVER_HI)).any():
        raise AssertionError("forward file holds bars from the never-read span 2025-07-21..2025-09-30")
    df = df[(df.index >= FWD_START) & (df.index <= FWD_LAST)]
    assert len(df) == 0 or (df.index.min() >= FWD_START and df.index.max() <= FWD_LAST)
    return df


def read_fwd(path=FWD_CSV):
    d = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return check_fwd(d.set_index("timestamp").sort_index())


# ------------------------------------------------------------------ the minute grid
class Grid:
    """Gap-free 1-minute arrays from first full hour to the last full hour of a raw frame (missing minute: open = close = previous close)."""

    def __init__(self, raw):
        t0 = raw.index[0].ceil("h")
        t1 = raw.index[-1].floor("h") + pd.Timedelta(minutes=59)
        full = pd.date_range(t0, t1, freq="min")
        df = raw.reindex(full)
        self.n_missing = int(df["close"].isna().sum())
        c = df["close"].ffill().bfill()
        o = df["open"].where(df["open"].notna(), c.shift(1).fillna(c))
        self.index = full
        self.t = full.values.astype("datetime64[m]")
        self.o, self.c = o.values.astype(float), c.values.astype(float)
        nxt = np.concatenate([self.o[1:], self.c[-1:]])
        self.r = nxt / self.o - 1.0
        self.n = len(full)
        assert self.n % 60 == 0
        self.nh = self.n // 60
        self.hour_t = full[::60]
        self.hc = self.c[59::60]                               # hourly close (minute 59)
        self.ho = self.o[::60]

    def hour_of(self, ts):
        """Index of the hour starting at (or containing) timestamp ts."""
        return int(self.minute_of(np.datetime64(pd.Timestamp(ts), "m")) // 60)

    def view(self, h0, h1):
        """The hours [h0, h1) as a light Grid-like object (r, n, nh, hour_t) for the evaluation functions."""
        v = type("View", (), {})()
        v.r = self.r[h0 * 60:h1 * 60]; v.n = len(v.r); v.nh = h1 - h0; v.hour_t = self.hour_t[h0:h1]; v.index = self.index[h0 * 60:h1 * 60]
        return v

    def minute_of(self, ts):
        """Grid positions of timestamps (np.datetime64 / DatetimeIndex)."""
        return ((np.asarray(ts, dtype="datetime64[m]") - self.t[0]) // np.timedelta64(1, "m")).astype(np.int64)


def hsum(x):
    return x.reshape(-1, 60).sum(1)


def hmean(x):
    return x.reshape(-1, 60).mean(1)


# ------------------------------------------------------------------ position builders
def positions_from_trades(grid, entry, exit_, side):
    """Minute position (int8) from trades [entry, exit) (np.datetime64 minutes); trades must not overlap."""
    d = np.zeros(grid.n + 1, np.int64)
    e = grid.minute_of(entry); x = grid.minute_of(exit_)
    ok = (e >= 0) & (e < grid.n)
    x = np.minimum(x, grid.n)
    np.add.at(d, e[ok], np.asarray(side)[ok]); np.add.at(d, x[ok], -np.asarray(side)[ok])
    pos = np.cumsum(d)[:-1]
    assert np.abs(pos).max(initial=0) <= 1, "overlapping trades"
    return pos.astype(np.int8)


def non_overlapping(entry, exit_):
    """Keep mask over signals sorted by entry: skip a signal while a position is open (entry < previous exit)."""
    keep = np.zeros(len(entry), bool); last = None
    for i in range(len(entry)):
        if last is None or entry[i] >= last:
            keep[i] = True; last = exit_[i]
    return keep


def month_thresholds(conf_by_month, months, q=Q):
    """Threshold of month m = the q-quantile of |p - 0.5| over the PREVIOUS month's predictions (V_prev, H79). Months in order."""
    thr = {}
    for prev, m in zip(months[:-1], months[1:]):
        if prev in conf_by_month and len(conf_by_month[prev]):
            thr[m] = float(np.quantile(conf_by_month[prev], q))
    return thr


def signal_trades(t_entry, p, month, thr, hold):
    """From predictions p of anchors (entry time t_entry as datetime64[m], month label per anchor): the non-overlapping trades.
    A signal needs |p - 0.5| >= thr[month]; long if p > 0.5. Returns entry, exit (datetime64[m]) and side."""
    t_entry = np.asarray(t_entry, dtype="datetime64[m]"); p = np.asarray(p); month = np.asarray(month)
    th = np.array([thr.get(m, np.inf) for m in month])
    k = np.abs(p - 0.5) >= th
    ent = t_entry[k]; side = np.where(p[k] > 0.5, 1, -1)
    o = np.argsort(ent, kind="stable"); ent, side = ent[o], side[o]
    ex = ent + np.timedelta64(hold, "m")
    keep = non_overlapping(ent, ex)
    return ent[keep], ex[keep], side[keep]


def ema_arr(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def pos_trend(hc):
    """M4: hourly EMA 20/50 cross on hourly closes, always in the market. Position of hour k from closes of hours < k."""
    sig = np.where(ema_arr(hc, 20) > ema_arr(hc, 50), 1, -1).astype(np.int8)
    p = np.concatenate([[0], sig[:-1]]).astype(np.int8)
    p[:60] = 0                                             # warm-up
    return p


def rsi_arr(hc, n=14):
    d = np.diff(hc, prepend=hc[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1.0 / n, adjust=False).mean().values; ad = pd.Series(dn).ewm(alpha=1.0 / n, adjust=False).mean().values
    return np.where(ad == 0, 100.0, 100.0 - 100.0 / (1.0 + au / np.where(ad == 0, 1, ad)))


def pos_meanrev(hc):
    """M5: hourly RSI(14): long below 30, short above 70, exit at 50. Position of hour k from closes of hours < k."""
    rsi = rsi_arr(hc); st = 0; out = np.zeros(len(hc), np.int8)
    for k in range(len(hc)):
        out[k] = st                                        # state held during hour k was decided at the close of k-1
        if k < 30:
            continue
        v = rsi[k]
        if st == 1 and v >= 50: st = 0
        elif st == -1 and v <= 50: st = 0
        if st == 0:
            st = 1 if v < 30 else (-1 if v > 70 else 0)
    return out


def pos_breakout(hc):
    """M6: Bollinger(20, 2) on hourly closes: a close outside the band enters in the direction of the break, exit at the middle band."""
    s = pd.Series(hc); m = s.rolling(20).mean().values; sd = s.rolling(20).std(ddof=0).values
    up, lo = m + 2 * sd, m - 2 * sd; st = 0; out = np.zeros(len(hc), np.int8)
    for k in range(len(hc)):
        out[k] = st
        if k < 25 or not np.isfinite(m[k]):
            continue
        v = hc[k]
        if st == 1 and v < m[k]: st = 0
        elif st == -1 and v > m[k]: st = 0
        if st == 0:
            st = 1 if v > up[k] else (-1 if v < lo[k] else 0)
    return out


# ------------------------------------------------------------------ member books
def book(pos, r):
    """Hourly gross return g, hourly one-way turnover u (unit notional) and hourly mean position p of a minute position series."""
    pos = np.asarray(pos, np.float64)
    dw = np.abs(np.diff(pos, prepend=0.0))
    return hsum(pos * r), hsum(dw), hmean(pos)


def net(g, u, cost_bps):
    return g - cost_bps * 0.5e-4 * u


# ------------------------------------------------------------------ combinations
def trailing_std(x, win, min_n):
    """Std of x[k-win:k] (strictly before k); NaN until min_n values exist."""
    s = pd.Series(x).shift(1)
    return s.rolling(win, min_periods=min_n).std().values


def trailing_sharpe(x, win, min_n):
    s = pd.Series(x).shift(1); r = s.rolling(win, min_periods=min_n)
    sd = r.std().values
    return np.where(sd > 0, r.mean().values / np.where(sd > 0, sd, 1) * np.sqrt(HOURS_PER_YEAR), np.nan)


SCALE_CAP = 5.0                                          # a member's equal-risk scale is capped (a member with a tiny sigma would otherwise dominate)
VOL_WIN, VOL_MIN = 720, 336                              # 30 days, at least 14 days
SR_WIN, SR_MIN = 2160, 336                               # 90 days, at least 14 days


def equal_risk_weights(nets):
    """nets [M, H] own-book hourly net returns. a_i[k] = sigma_ref[k] / sigma_i[k] / M, sigma_i the trailing 30-day std of own returns,
    sigma_ref the median over members with sigma > 0. A member with sigma 0 or NaN gets 0; before 14 days of history: 1 / M."""
    M, H = nets.shape
    sig = np.stack([trailing_std(nets[i], VOL_WIN, VOL_MIN) for i in range(M)])
    ok = np.isfinite(sig) & (sig > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(np.where(ok, sig, np.nan), axis=0)
    a = np.where(ok, np.minimum(SCALE_CAP, ref[None, :] / np.where(ok, sig, 1.0)), 0.0) / M
    young = ~np.isfinite(ref)                             # no history at all yet
    a[:, young] = 1.0 / M
    return a


def inverse_loss_weights(nets):
    """C4: equal-risk scale times the member's trailing 90-day Sharpe clipped at 0, normalised to sum 1 over members.
    Before 14 days of history: the equal-risk weights."""
    base = equal_risk_weights(nets)
    sr = np.stack([trailing_sharpe(nets[i], SR_WIN, SR_MIN) for i in range(nets.shape[0])])
    have = np.isfinite(sr).any(0)
    om = np.clip(np.nan_to_num(sr, nan=0.0), 0.0, None)
    tot = om.sum(0)
    om = np.where(tot > 0, om / np.where(tot > 0, tot, 1.0), 0.0)
    scale = base * nets.shape[0]                          # the sigma_ref / sigma_i scale (without 1/M)
    a = scale * om
    a[:, ~have] = base[:, ~have]
    return a


def combine_linear(pos, a):
    """Minute position = sum_i a_i[hour] * pos_i[minute]; clipped to the 1x notional cap. pos [M, n] ints, a [M, nh]."""
    w = np.zeros(pos.shape[1], np.float64)
    for i in range(pos.shape[0]):
        w += np.repeat(a[i], 60) * pos[i]
    return np.clip(w, -1.0, 1.0)


def vote_hourly(p_hour, need=3):
    """C3: sign of the sum of the members' hourly mean positions, active only when at least `need` members are non-zero and agree with it."""
    s = np.sign(p_hour.sum(0))
    agree = (np.sign(p_hour) == s[None, :]) & (np.abs(p_hour) > 0) & (s[None, :] != 0)
    return np.where(agree.sum(0) >= need, s, 0.0)


def har_scale(sig_hat, sig_med):
    """Exposure multiplier min(1, sigma* / sigma_HAR); NaN forecasts (no history yet) scale by 1."""
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.minimum(1.0, sig_med / sig_hat)
    return np.where(np.isfinite(s), s, 1.0)


def build_combos(pos, p_hour, nets2, har):
    """All six minute-position paths. pos [M, n]; p_hour [M, nh] mean positions; nets2 [M, nh] own-book net returns at 2 bps;
    har: hourly multiplier [nh]."""
    a1 = equal_risk_weights(nets2)
    out = {}
    out["C1"] = combine_linear(pos, a1)
    out["C2"] = combine_linear(pos, np.full(a1.shape, 1.0 / pos.shape[0]))
    v = vote_hourly(p_hour); out["C3"] = np.repeat(v, 60)
    out["C4"] = combine_linear(pos, inverse_loss_weights(nets2))
    h = np.repeat(har, 60)
    out["C5"] = out["C1"] * h
    out["C6"] = out["C3"] * h
    return out


# ------------------------------------------------------------------ metrics
def equity_stats(r_hour):
    r_hour = np.asarray(r_hour, float)
    eq = np.cumprod(1.0 + r_hour); n = len(r_hour)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = float(np.max(1.0 - eq / peak)) if n else 0.0
    total = float(eq[-1] - 1.0) if n else 0.0
    ann = float(eq[-1] ** (HOURS_PER_YEAR / n) - 1.0) if n and eq[-1] > 0 else (-1.0 if n else 0.0)
    return {"total_return": total, "ann_return": ann, "max_dd": dd, "calmar": (ann / dd if dd > 0 else float("nan"))}


def runs_of(w):
    """Run structure of a minute position: sign (|w| > NZ else 0), run id (starts at each change into a non-zero sign), mask of the first minute."""
    s = np.where(np.abs(w) > NZ, np.sign(w), 0).astype(np.int8)
    prev = np.concatenate([[0], s[:-1]])
    start = (s != 0) & (s != prev)
    return s, np.cumsum(start), start


def trade_stats(w, r, cost_bps):
    """Number of trades (runs of a constant non-zero sign) and their hit rate (net P&L of the run > 0, exit cost included)."""
    w = np.asarray(w, np.float64)
    s, rid, start = runs_of(w)
    n = int(start.sum())
    if n == 0:
        return {"n_trades": 0, "hit": float("nan")}
    dw = np.abs(np.diff(w, prepend=0.0))
    pnl = np.bincount(rid, weights=w * r - cost_bps * 0.5e-4 * dw, minlength=n + 1)[1:]
    return {"n_trades": n, "hit": float(np.mean(pnl > 0))}


def months_of(hour_t):
    return np.asarray(pd.DatetimeIndex(hour_t).to_period("M").astype(str))


def monthly_table(r_hour, hour_t):
    m = months_of(hour_t)
    df = pd.DataFrame({"m": m, "lr": np.log1p(r_hour)}).groupby("m")["lr"].sum()
    return np.expm1(df)


def evaluate(w, grid, cost_bps, sl=slice(None)):
    """Hourly net returns of the minute position w at cost_bps and the headline numbers on hour slice sl."""
    g, u, _ = book(w, grid.r)
    rh = net(g, u, cost_bps)[sl]
    st = equity_stats(rh)
    years = len(rh) / HOURS_PER_YEAR
    return rh, st, years


def null_cum(w, grid, cost_bps, sl, draws=1000, seed=0, chunk=100, flips=None):
    """Random-side null on the SAME exposure path: the signs of the position's runs (maximal runs of a constant non-zero sign) are flipped by
    fair coins, magnitudes and times kept; the cost is the actual path's turnover cost. Returns the draws' cumulative returns."""
    from scipy import sparse
    w = np.asarray(w, np.float64); s, rid, start = runs_of(w)
    gm = w * grid.r; hour = np.arange(grid.n) // 60
    cost_h = cost_bps * 0.5e-4 * hsum(np.abs(np.diff(w, prepend=0.0)))
    nr = int(rid.max()) if len(rid) else 0
    df = pd.DataFrame({"h": hour, "run": rid, "g": gm}).groupby(["h", "run"], sort=True)["g"].sum().reset_index()
    sh = sparse.csr_matrix((np.ones(len(df)), (df["h"].values, np.arange(len(df)))), shape=(grid.nh, len(df)))
    rng = np.random.default_rng(seed); out = []
    if flips is not None:                                                                       # explicit flips [d, nr + 1] (tests)
        draws, chunk = len(flips), len(flips)
    for lo in range(0, draws, chunk):
        d = min(chunk, draws - lo)
        fl = flips if flips is not None else np.where(rng.random((d, nr + 1)) < 0.5, 1.0, -1.0)
        fl = fl.copy(); fl[:, 0] = 1.0                                                          # run 0 = before the first trade (flat)
        gross = sh @ (df["g"].values[:, None] * fl[:, df["run"].values].T)                      # [nh, d]
        rh = gross - cost_h[:, None]
        rh = rh[sl]
        out.append(np.expm1(np.log1p(rh).sum(0)))
    return np.concatenate(out)


def flip_runs(w, flips_by_run):
    """The flipped path itself (used by the tests): multiply each run's minutes by its sign."""
    s, rid, _ = runs_of(w)
    return np.asarray(w, np.float64) * flips_by_run[rid]


def leverage_for_dd(r_hour, cap=0.05):
    """Largest constant leverage L (bisection) with max DD of the leveraged hourly net returns below cap; its cumulative return."""
    r = np.asarray(r_hour, float)
    lo, hi = 0.0, 500.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid * r.min() > -1 and equity_stats(mid * r)["max_dd"] < cap:
            lo = mid
        else:
            hi = mid
    return float(lo), equity_stats(lo * r)["total_return"]


# ------------------------------------------------------------------ order guard
def assert_selection_committed(path, repo=None):
    """selection.json must be tracked and unmodified (clean) in git before any look: raises otherwise. Returns the sha256 of the file."""
    import hashlib
    repo = repo or os.path.dirname(os.path.abspath(path))
    rel = os.path.relpath(os.path.abspath(path), repo).replace("\\", "/")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo, capture_output=True, text=True)
    if tracked.returncode != 0:
        raise AssertionError(f"{rel} is not tracked: commit selection.json before any look")
    st = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=repo, capture_output=True, text=True).stdout.strip()
    if st:
        raise AssertionError(f"{rel} is modified or staged ({st}): commit it unchanged before any look")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
