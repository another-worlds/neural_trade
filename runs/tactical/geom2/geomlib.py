"""Geometry features for the night program (nt-tactical-geom2): a generalisation of lab/geom_lab.py's geometry to all 14
indicator families at textbook periods and several crossing windows. Everything is computed from ONE window [N, L, 5]
(OHLCV, L = 60) and only from bars inside it (causal: bars after the window end cannot enter; tested in test_geomlib.py).

Per channel x (a series over the window), at the window's last bar:
  slope_k   (x[-1] - x[-1-k]) / (k * sd(diff x))  for k in SLOPES                      (all channels)
  squeeze   sd(x[-10:]) / sd(x)                                                         (all channels)
  pos       arcsinh of the signed distance to the reference: price channels (close - x) / sd(diff close);
            oscillators (x - center) / sd(diff x)                                       (price and oscillator channels)
  cross_w   (tanh(z[-1]) - tanh(z[-1-w])) / 2 with z the signed distance above, w in CROSS (smooth crossing in the last w bars)
Channel kinds: price (compared with the close), osc (compared with its natural centre), other (slope and squeeze only).
`geom(W, names=None, cross=(5,))` returns [N, F]; `base5` is the old H45 geometry (5 channels, one crossing window) and equals
lab/geom_lab.fs_geom up to the channel definitions (EMA20, RSI14, MACD hist, SMA20, Stoch %K 14)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

SLOPES = (3, 10)
CHUNK = 8000
BASE5 = ("ema20", "rsi14", "macd_hist", "sma20", "stoch14")


def ema(x, p):
    a = 2.0 / (p + 1.0); out = np.empty_like(x); out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = a * x[:, t] + (1 - a) * out[:, t - 1]
    return out


def _roll(x, p, fn):
    """Trailing window statistic over axis 1; the first p-1 positions use the first value repeated (a prefix-only rule)."""
    xp = np.concatenate([np.repeat(x[:, :1], p - 1, axis=1), x], axis=1)
    return fn(swv(xp, p, axis=1), axis=2)


def _pad1(x):
    return np.concatenate([x[:, :1], x], axis=1)


def channels(W):
    """name -> (series [N, L], kind, center). Every series at position t uses bars <= t only."""
    W = W.astype(np.float64)
    o, h, l, c, v = (W[..., i] for i in range(5))
    ch = {}
    d = np.diff(c, axis=1)
    for p in (5, 10, 20, 40):
        ch[f"ema{p}"] = (ema(c, p), "price", None)
    sma20 = _roll(c, 20, np.mean); sd20 = _roll(c, 20, np.std)
    ch["sma20"] = (sma20, "price", None)
    ch["bb_up"] = (sma20 + 2 * sd20, "price", None); ch["bb_lo"] = (sma20 - 2 * sd20, "price", None)
    pc = np.concatenate([c[:, :1], c[:, :-1]], axis=1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr10, atr14 = ema(tr, 10), ema(tr, 14)
    e20 = ch["ema20"][0]
    ch["kel_up"] = (e20 + 2 * atr10, "price", None); ch["kel_lo"] = (e20 - 2 * atr10, "price", None)
    ch["don_up"] = (_roll(h, 20, np.max), "price", None); ch["don_lo"] = (_roll(l, 20, np.min), "price", None)
    tp = (h + l + c) / 3
    ch["vwap20"] = (_roll(tp * v, 20, np.sum) / (_roll(v, 20, np.sum) + 1e-12), "price", None)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    ch["rsi14"] = (_pad1(100 - 100 / (1 + ema(up, 14) / (ema(dn, 14) + 1e-12))), "osc", 50.0)
    macd = ema(c, 12) - ema(c, 26)
    ch["macd_line"] = (macd, "osc", 0.0); ch["macd_hist"] = (macd - ema(macd, 9), "osc", 0.0)
    lo14, hi14 = _roll(l, 14, np.min), _roll(h, 14, np.max)
    ch["stoch14"] = ((c - lo14) / (hi14 - lo14 + 1e-12), "osc", 0.5)
    ma_tp = _roll(tp, 20, np.mean); md = _roll(np.abs(tp - ma_tp), 20, np.mean)   # mean deviation about the trailing mean
    ch["cci20"] = (np.clip((tp - ma_tp) / (0.015 * md + 1e-12) / 100, -5, 5), "osc", 0.0)
    dtp = np.diff(tp, axis=1); flow = (tp * v)[:, 1:]
    pos = _roll(flow * (dtp > 0), 14, np.sum); neg = _roll(flow * (dtp < 0), 14, np.sum)
    ch["mfi14"] = (_pad1(pos / (pos + neg + 1e-12)), "osc", 0.5)
    uh, dl = np.diff(h, axis=1), -np.diff(l, axis=1)
    pdm = np.where((uh > dl) & (uh > 0), uh, 0.0); mdm = np.where((dl > uh) & (dl > 0), dl, 0.0)
    atrw = ema(tr[:, 1:], 14) + 1e-12
    pdi, mdi = 100 * ema(pdm, 14) / atrw, 100 * ema(mdm, 14) / atrw
    ch["dmi_diff"] = (_pad1(pdi - mdi), "osc", 0.0)
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    ch["adx14"] = (_pad1(ema(dx, 14)), "other", None)
    ch["atr14"] = (atr14, "other", None)
    ch["bb_width"] = (4 * sd20, "other", None)
    obv = np.concatenate([np.zeros((len(c), 1)), np.cumsum(np.sign(d) * v[:, 1:], axis=1)], axis=1) / (np.cumsum(v, axis=1) / np.arange(1, v.shape[1] + 1) + 1e-12)
    ch["obv"] = (obv, "other", None)
    return ch


def _geom_chunk(W, names, cross):
    ch = channels(W)
    names = names or tuple(ch)
    c = W[..., 3].astype(np.float64); sdc = np.diff(c, axis=1).std(1) + 1e-12
    f = []
    for name in names:
        x, kind, center = ch[name]
        sx = np.diff(x, axis=1).std(1) + 1e-12
        for k in SLOPES:
            f.append((x[:, -1] - x[:, -1 - k]) / (k * sx))
        if kind == "price":
            z = (c - x) / sdc[:, None]
        elif kind == "osc":
            z = (x - center) / sx[:, None]
        if kind != "other":
            f.append(np.arcsinh(z[:, -1]))
            for w in cross:
                f.append((np.tanh(z[:, -1]) - np.tanh(z[:, -1 - w])) / 2)
        f.append(x[:, -10:].std(1) / (x.std(1) + 1e-12))
    return np.stack(f, 1)


def geom(W, names=None, cross=(5,)):
    """Geometry features of windows W [N, L, 5]; NaN-free."""
    out = [_geom_chunk(W[i:i + CHUNK], names, cross) for i in range(0, len(W), CHUNK)]
    return np.nan_to_num(np.concatenate(out).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def geom_base5(W):
    return geom(W, BASE5, (5,))


def geom_rich(W):
    return geom(W, None, (3, 5, 10, 20))


def geom_names(names=None, cross=(5,)):
    ch = channels(np.random.default_rng(0).normal(size=(2, 60, 5)).cumsum(1) + 100)
    names = names or tuple(ch); out = []
    for n in names:
        kind = ch[n][1]
        out += [f"{n}.slope{k}" for k in SLOPES]
        if kind != "other":
            out += [f"{n}.pos"] + [f"{n}.cross{w}" for w in cross]
        out.append(f"{n}.squeeze")
    return out
