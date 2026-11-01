"""Rule candidates and baselines (SPEC 'track1', 'track2', 'baselines', 'strategy_layer' 2 and 4).

Every rule is a causal signal array over bar indices of the full dev (or full) series plus a policy object for engine E.
No rule here has a free parameter fitted on data: all constants are fixed by the SPEC.

Track 1
  T1-FADEV     vol-gated fade: |z15| (sigma60 units) in the top 2% of the trailing 1,440 minutes AND the trailing
               30-day percentile of sigma60 >= 2/3; fade; maker K = 3 then cancel; taker exit 15 bars after the fill;
               same-side signal during a hold resets the exit clock; funding skip (>= 3 bps expected payment).
  plain fade   calibration section 6 rule on the 1-min grid: |z15| (EWMA lambda 0.94 sigma) in the top 2% of the
               trailing 1,440 minutes; fade; hold 15; no overlap; same order types as FADEV.
  intraday trend  sign(EMA15 - EMA60) of 1-min closes, re-decided at every 15-min close, |x| = 1, maker K = 3, cancel.
Track 2
  T2-P         daily at the 23:59 close: tau = mean_L sign(ln C_d - ln C_{d-L}), L in {7, 14, 28, 56} days;
               x* = round(tau * min(1, 0.5 / sigma_ann), 1); trade if |x* - x| >= 0.2; maker, chase after 15 bars.
  TSMOM unscaled  x = tau, traded when tau changes.
  vol-target long x = round(min(1, 0.5 / sigma_ann), 1), band 0.2.
  60-min fade  |z60| (sigma_t units) in the top 2% of the trailing 30 days of 1-min values; fade; hold 60; maker
               K = 5 then cancel; taker exit 60 bars after the fill.
Shared: B&H perp (x = 1, maker, chase after 15), cash, static-exposure-matched (ConstPolicy), carry hurdle (funding sum).
"""
import numpy as np
import pandas as pd

from features import ewm

DAY = 86400
FUND_SKIP_BPS = 3.0


def rolling_rank(x, w):
    """Causal percentile rank of x_t within x_{t-w+1..t} (pandas average ties); NaN until the window is full."""
    return pd.Series(x).rolling(w, min_periods=w).rank(pct=True).to_numpy()


def log_ret(lc, L):
    out = np.full_like(lc, np.nan)
    out[L:] = lc[L:] - lc[:-L]
    return out


def fund_last_bps(b):
    """Last settled funding print (bps) at or before each bar (a print at bar t settles at its open)."""
    ev = np.flatnonzero(b.funding != 0)
    k = np.searchsorted(ev, np.arange(len(b.t)), side="right")
    return np.where(k > 0, b.funding[ev[np.maximum(k - 1, 0)]] * 1e4, 0.0)


def print_ahead(b, span):
    """True if a scheduled funding time (00/08/16 UTC) falls on a bar in (i, i + span]; schedule only, no rates."""
    sched = (b.t % 28800 == 0).astype(np.int64)
    cs = np.r_[0, np.cumsum(sched)]
    n = len(sched)
    i = np.arange(n)
    hi = np.minimum(i + span, n - 1)
    return (cs[hi + 1] - cs[i + 1]) > 0


# ----------------------------------------------------------------------------------------------- signals

def sig_fadev(b):
    lc = np.log(b.c)
    r = np.r_[0.0, np.diff(lc)]
    s60 = np.maximum(np.sqrt(ewm(r * r, 60)), 1e-8)
    z15 = log_ret(lc, 15) / (s60 * np.sqrt(15))
    rk = rolling_rank(np.abs(z15), 1440)
    vp = rolling_rank(s60, 43200)
    hit = (rk >= 0.98) & (vp >= 2.0 / 3.0)
    side = np.where(hit, -np.sign(z15), 0.0).astype(np.int8)          # fade the move
    fl = fund_last_bps(b)
    ahead = print_ahead(b, 3 + 15 + 1)
    pay = side * fl                                                    # expected payment (bps) if the side is held
    entry = np.where(ahead & (pay >= FUND_SKIP_BPS), 0, side).astype(np.int8)
    return side, entry


def sig_plain_fade(b):
    lc = np.log(b.c)
    r = np.r_[0.0, np.diff(lc)]
    sig = np.maximum(np.sqrt(ewm(r * r, 2.0 / 0.06 - 1.0)), 1e-8)      # RiskMetrics lambda = 0.94
    z15 = log_ret(lc, 15) / (sig * np.sqrt(15))
    rk = rolling_rank(np.abs(z15), 1440)
    side = np.where(rk >= 0.98, -np.sign(z15), 0.0).astype(np.int8)
    return side


def sig_fade60(b, sigma):
    lc = np.log(b.c)
    z60 = log_ret(lc, 60) / (sigma * np.sqrt(60))
    rk = rolling_rank(np.abs(z60), 43200)
    return np.where(rk >= 0.98, -np.sign(z60), 0.0).astype(np.int8)


def sig_intraday_trend(b):
    """Target per bar at 15-min closes (hh:14, :29, :44, :59); NaN elsewhere."""
    e15, e60 = ewm(b.c, 15), ewm(b.c, 60)
    tgt = np.sign(e15 - e60)
    dec = (b.t % 900) == 840
    return np.where(dec, tgt, np.nan)


def daily_closes(b):
    """Per bar: (decision mask at 23:59, daily index d of each bar, daily log closes C over a complete calendar)."""
    day = b.t // DAY
    d0 = day[0]
    nd = int(day[-1] - d0 + 1)
    last = np.full(nd, -1, np.int64)
    last[day - d0] = np.arange(len(day))                                # last bar of each calendar day
    C = np.full(nd, np.nan)
    ok = last >= 0
    C[ok] = np.log(b.c[last[ok]])
    C = pd.Series(C).ffill().to_numpy()
    dec = (b.t % DAY) == 86340
    return dec, (day - d0).astype(np.int64), C


def tsmom_daily(C):
    """tau_d and sigma_ann_d from closes up to and including day d."""
    tau = np.zeros_like(C)
    for L in (7, 14, 28, 56):
        lagged = np.r_[np.full(L, np.nan), C[:-L]]
        tau += np.sign(np.nan_to_num(C - lagged))
    tau /= 4.0
    rd = np.r_[0.0, np.diff(C)]
    s_ann = np.sqrt(365.0 * np.maximum(ewm(rd * rd, 30), 1e-12))
    valid = np.arange(len(C)) >= 56
    return np.where(valid, tau, 0.0), s_ann, valid


# ----------------------------------------------------------------------------------------------- policies

class EventPolicy:
    """Event fade: enter at a signal (maker, K bars, cancel), taker exit H bars after the fill, no pyramiding.
    extend=True: a same-side signal during a hold resets the exit clock to signal bar + H; opposite signals ignored."""

    def __init__(self, entry_side, H, K, extend_side=None):
        self.entry, self.H, self.K = entry_side, int(H), int(K)
        self.ext = extend_side
        self.chase_entries = False
        self.state, self.exit_at, self.pos = "flat", -1, 0
        self.n_ext = 0
        self.dec_bar = -1

    def on_fill(self, i, kind, st):
        if kind == "cancel":
            if st.q == 0.0:
                self.state = "flat"
            else:                                  # never strand in 'entry' with a stray position: exit it now
                self.state, self.exit_at, self.pos = "hold", i, int(np.sign(st.q))
        elif kind == "increase" and st.q != 0.0 and self.state == "entry":
            # a fill at the decision bar itself (nominal fills) holds H bars like a next-bar fill
            hold_end = i + self.H if i == self.dec_bar else i + self.H - 1
            self.state, self.exit_at, self.pos = "hold", hold_end, int(np.sign(st.q))
        elif st.q == 0.0:
            self.state = "flat"

    def decide(self, i, x, st):
        if self.state == "hold":
            if self.ext is not None and self.ext[i] == self.pos and i + self.H > self.exit_at:
                self.exit_at = i + self.H
                self.n_ext += 1
            if i >= self.exit_at:
                self.state = "exiting"
                return 0.0
            return None
        if self.state != "flat":
            return None
        if st.q != 0.0:                            # never leave a stray position (e.g. after a cap trim) open
            self.state = "exiting"
            return 0.0
        s = self.entry[i]
        if s != 0:
            self.state, self.dec_bar = "entry", i
            return float(s)
        return None


class StaticPolicy:
    """B5 static-exposure-matched: hold x = mx (constant exposure): trade to mx at the first decision, then at 23:59
    closes whenever |x - mx| > band (maker, chase after 15 bars)."""

    def __init__(self, mx, dec, band=0.02, K=15, chase=True):
        self.mx, self.dec, self.band, self.K, self.chase_entries = float(mx), dec, band, K, chase
        self.started = False

    def decide(self, i, x, st):
        if not self.started:
            self.started = True
            return self.mx
        if self.dec[i] and abs(x - self.mx) > self.band:
            return self.mx
        return None


class SignChangePolicy:
    """Target per bar (NaN = no decision); trade only when the target sign differs from the held sign (|x| = 1)."""

    def __init__(self, targets, K, chase):
        self.tg, self.K, self.chase_entries = targets, int(K), bool(chase)

    def decide(self, i, x, st):
        v = self.tg[i]
        if np.isnan(v):
            return None
        held = 0.0 if abs(x) < 0.05 else np.sign(x)
        if v == held:
            return None
        return float(v)


class DailyPolicy:
    """At 23:59 closes: target from a per-day array; trade if |target - x| >= band (band = 0: on any change of target)."""

    def __init__(self, dec, dayidx, target_by_day, band=0.2, K=15, chase=True, on_change=False):
        self.dec, self.di, self.tbd, self.band = dec, dayidx, target_by_day, band
        self.K, self.chase_entries, self.on_change = K, chase, on_change
        self.last = None

    def decide(self, i, x, st):
        if not self.dec[i]:
            return None
        tgt = float(self.tbd[self.di[i]])
        if not np.isfinite(tgt):
            return None
        if self.on_change:
            if self.last is not None and tgt == self.last:
                return None
            self.last = tgt
            return None if abs(tgt - x) < 1e-9 else tgt
        return tgt if abs(tgt - x) >= self.band - 1e-12 else None


# ----------------------------------------------------------------------------------------------- factory

def build_signals(b, sigma):
    side, entry = sig_fadev(b)
    dec, di, C = daily_closes(b)
    tau, s_ann, valid = tsmom_daily(C)
    scale = np.minimum(1.0, 0.5 / s_ann)
    return {"fadev_side": side, "fadev_entry": entry, "plain_fade": sig_plain_fade(b), "fade60": sig_fade60(b, sigma),
            "intraday_trend": sig_intraday_trend(b), "dec": dec, "dayidx": di,
            "t2p": np.where(valid, np.round(tau * scale, 1), np.nan), "tsmom": np.where(valid, tau, np.nan),
            "voltarget": np.where(valid, np.round(scale, 1), np.nan), "tau": tau, "s_ann": s_ann}


def policies(S, track):
    """name -> zero-arg factory of a fresh policy (policies are stateful)."""
    import sim
    if track == 1:
        return {
            "T1-FADEV": lambda: EventPolicy(S["fadev_entry"], 15, 3, extend_side=S["fadev_side"]),
            "plain_fade": lambda: EventPolicy(S["plain_fade"], 15, 3),
            "intraday_trend": lambda: SignChangePolicy(S["intraday_trend"], 3, False),
            "B&H": lambda: sim.ConstPolicy(1.0, K=15, chase=True),
            "cash": lambda: sim.ConstPolicy(0.0, K=1, chase=False),
        }
    return {
        "T2-P": lambda: DailyPolicy(S["dec"], S["dayidx"], S["t2p"], band=0.2),
        "tsmom_unscaled": lambda: DailyPolicy(S["dec"], S["dayidx"], S["tsmom"], on_change=True),
        "voltarget_long": lambda: DailyPolicy(S["dec"], S["dayidx"], S["voltarget"], band=0.2),
        "fade60": lambda: EventPolicy(S["fade60"], 60, 5),
        "B&H": lambda: sim.ConstPolicy(1.0, K=15, chase=True),
        "cash": lambda: sim.ConstPolicy(0.0, K=1, chase=False),
    }


def carry_hurdle(b, start, end, haircut_bps_yr=30.0):
    """Delta-neutral cash-and-carry reference: sum of funding prints in [start, end) minus 30 bps per year (log)."""
    f = b.funding[start:end].sum()
    years = (b.t[end - 1] - b.t[start]) / (365.25 * DAY)
    return float(f - haircut_bps_yr * 1e-4 * years)
