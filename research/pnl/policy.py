"""Policy layer (SPEC 'strategy_layer'): every candidate becomes a target exposure in [-1, 1] at its decision bars.

HysteresisPolicy: cost-aware state machine for forecasters (mu in bps). BandPolicy: no-trade band for PnL nets.
Both are driven by engine E (sim.simulate) through decide(i, x, st). Arrays are indexed by bar index i; per-bar
parameter arrays let the model (and its cadence, hold, threshold and maker window) switch at fold boundaries while
the position carries across.
"""
import numpy as np

C_RT = 8.0          # effective C7-B round trip in bps (SPEC: about 7.9)


def _arr(v, n, dtype):
    return np.full(n, v, dtype) if np.isscalar(v) else np.asarray(v, dtype)


class HysteresisPolicy:
    """State s in {-1, 0, +1}, size |x| = 1. From flat: enter sign(mu) if |mu| > thr. In a position, after a minimum
    hold of H minutes: exit if s * mu <= 0; flip if s * mu < -thr; otherwise hold. thr = theta * C_RT (bps).
    mu: forecast in bps per bar (NaN = no forecast); decisions where `dec` is True."""

    def __init__(self, mu, dec, H, thr, K=1, chase=False):
        n = len(mu)
        self.mu, self.dec = mu, dec
        self.H, self.thr, self.Ka = _arr(H, n, np.int64), _arr(thr, n, np.float64), _arr(K, n, np.int64)
        self.K, self.chase_entries = int(self.Ka.max()), chase
        self.s, self.since = 0, -10 ** 9

    def K_at(self, i):
        return self.Ka[i]

    def on_fill(self, i, kind, st):
        if kind == "cancel":
            if st.q == 0.0:
                self.s = 0
        elif st.q != 0.0 and kind in ("increase", "flip") and int(np.sign(st.q)) == self.s:
            self.since = i                         # the hold counts from the fill; a stale fill never resets the state

    def decide(self, i, x, st):
        if not self.dec[i]:
            return None
        m = self.mu[i]
        if not np.isfinite(m):
            return None
        thr = self.thr[i]
        if self.s == 0:
            if abs(m) > thr:
                self.s, self.since = int(np.sign(m)), i
                return float(self.s)
            return None
        if i - self.since < self.H[i]:
            return None
        if self.s * m < -thr:
            self.s, self.since = -self.s, i
            return float(self.s)
        if self.s * m <= 0:
            self.s = 0
            return 0.0
        return None


class BandPolicy:
    """Trade the drifted exposure to the nearest edge of [tau - b, tau + b] (clipped to [-1, 1]), rounded to the 0.05
    grid toward the band interior (band_target)."""

    def __init__(self, tau, band, dec, K=1, chase=False, step=0.05):
        n = len(tau)
        self.tau, self.band, self.dec, self.Ka = tau, band, dec, _arr(K, n, np.int64)
        self.K, self.chase_entries, self.step = int(self.Ka.max()), chase, step

    def K_at(self, i):
        return self.Ka[i]

    def decide(self, i, x, st):
        if not self.dec[i]:
            return None
        lo = max(-1.0, self.tau[i] - self.band[i])
        hi = min(1.0, self.tau[i] + self.band[i])
        if lo <= x <= hi:
            return None
        tgt = band_target(x, lo, hi, self.step)
        return None if abs(tgt - x) < 1e-9 else float(np.clip(tgt, -1, 1))


def band_target(x, lo, hi, step):
    """Nearest band edge rounded to the step grid TOWARD the band interior (so the new x is inside the band); if no
    grid point lies in [lo, hi], the unrounded edge."""
    if step <= 0:
        return lo if x < lo else hi
    g = np.ceil(lo / step - 1e-9) * step if x < lo else np.floor(hi / step + 1e-9) * step
    return g if lo - 1e-12 <= g <= hi + 1e-12 else (lo if x < lo else hi)
