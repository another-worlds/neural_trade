"""Policy layer (SPEC 'strategy_layer'): every candidate becomes a target exposure in [-1, 1] at its decision bars.

HysteresisPolicy: cost-aware state machine for forecasters (mu in bps). BandPolicy: no-trade band for PnL nets.
Both are driven by engine E (sim.simulate) through decide(i, x, st).
"""
import numpy as np

C_RT = 8.0          # effective C7-B round trip in bps (SPEC: about 7.9)


class HysteresisPolicy:
    """State s in {-1, 0, +1}, size |x| = 1. From flat: enter sign(mu) if |mu| > theta * C_RT. In a position, after a
    minimum hold of H minutes: exit if s * mu <= 0; flip if s * mu < -theta * C_RT; otherwise hold.
    mu: forecast in bps per bar index (NaN = no forecast); decisions at bars where `decide_mask` is True."""

    def __init__(self, mu, decide_mask, H, theta, K=3, chase=False, offset=0):
        self.mu, self.mask, self.H, self.thr = mu, decide_mask, int(H), theta * C_RT
        self.K, self.chase_entries, self.offset = K, chase, offset
        self.s, self.since = 0, -10 ** 9

    def on_fill(self, i, kind, st):
        if kind == "cancel" and st.q == 0.0:
            self.s = 0
        elif st.q != 0.0 and kind in ("increase", "flip"):
            self.s, self.since = int(np.sign(st.q)), i

    def decide(self, i, x, st):
        j = i - self.offset
        if not self.mask[j]:
            return None
        m = self.mu[j]
        if not np.isfinite(m):
            return None
        if self.s == 0:
            if abs(m) > self.thr:
                self.s, self.since = int(np.sign(m)), i
                return float(self.s)
            return None
        if i - self.since < self.H:
            return None
        if self.s * m < -self.thr:
            self.s, self.since = -self.s, i
            return float(self.s)
        if self.s * m <= 0:
            self.s = 0
            return 0.0
        return None


class BandPolicy:
    """Trade the drifted exposure to the nearest edge of [tau - b, tau + b] (clipped to [-1, 1]), rounded to 0.05."""

    def __init__(self, tau, band, decide_mask, K=1, chase=False, offset=0, step=0.05):
        self.tau, self.band, self.mask = tau, band, decide_mask
        self.K, self.chase_entries, self.offset, self.step = K, chase, offset, step

    def decide(self, i, x, st):
        j = i - self.offset
        if not self.mask[j]:
            return None
        lo = max(-1.0, self.tau[j] - self.band[j])
        hi = min(1.0, self.tau[j] + self.band[j])
        if lo <= x <= hi:
            return None
        tgt = round((lo if x < lo else hi) / self.step) * self.step
        return None if abs(tgt - x) < 1e-9 else float(np.clip(tgt, -1, 1))
