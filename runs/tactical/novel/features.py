"""Novel geometry features N1-N7 (runs/tactical/novel/SPEC.md). Vectorised over windows W [N, 60, 5] (open, high, low, close,
volume), per window only (no statistic crosses windows), causal (a window's features read only that window), scale-free
(multiplying the prices by a constant leaves them unchanged), NaN-safe (nan_to_num at the end).

Conventions fixed before any run (where the SPEC leaves room):
- Heavy-tailed ratios go through a signed log1p (`slog`): monotone, so boosting is unaffected; it keeps the logistic readout usable.
- Every sd is std + 1e-9 * mean|close| (a flat tape gives 0 / eps = 0, not 0 / 0); the guard scales with the price, so scale-free.
- N1: the three axes are computed on all 60 bars (close - EMA20, RSI14, MACD(12,26,9) histogram), each z-scored over the
  60-bar window, and the curve is the last 20 points. Curvature = turning angle / mean adjacent segment length; torsion = signed
  angle between consecutive binormals about the shared tangent / segment length.
- N2: paths on 60 bars; X = log close, Y in {cumulative signed volume, RSI, close - EMA20, high - low}; the last k+1 points
  (k = 15, 30; all 60 points for k = 60). area = 2 * Levy area / (TV_X * TV_Y), in [-1, 1].
- N3: s = EMA5(close); v = diff s, a = diff v, each divided by its own window std (no centering); theta = atan2(a, v).
- N4: hulls of (t, high) and (t, low) over bars 0..58 (the last bar is left out so its lines are projected, not fitted to it);
  the last hull edges are extended one bar to bar 59; position = (close59 - L59) / (U59 - L59), clipped to [-2, 3].
- N5: 40 midpoint quantiles of the 1-bar returns; halves = returns 0..28 and 30..58; recent = the last 15 returns against the
  44 before them; the signed median shift is (later - earlier) / sd.
- N6: returns in window-sd units; six sub-windows of 10 bars (9 returns each) -> (mu, sigma), sigma floored at 0.05; Fisher-Rao
  distance of the Gaussian family = sqrt(2) * hyperbolic distance in (mu / sqrt(2), sigma).
- N7: quadratic coefficient of the fit to the last k log closes (k = 15, 30, 60) in bar units, / sd of the 1-bar log returns.
"""
import numpy as np

EPS = 1e-9
NAMES = {
    "N1": ["kappa_mean", "kappa_end", "tau_end", "turning", "tortuosity"],
    "N2": [f"levy_{y}_{k}" for k in (15, 30, 60) for y in ("vol", "rsi", "ema", "rng")],
    "N3": ["cos_theta", "sin_theta", "omega_end", "winding30", "log_radius"],
    "N4": ["slope_up", "slope_lo", "pos", "dist_up", "dist_lo", "hull_area", "hull_verts"],
    "N5": ["w1_halves", "w1_15v44", "med_halves", "med_15v44"],
    "N6": ["fr_path", "fr_net", "fr_net_signed"],
    "N7": ["convex_15", "convex_30", "convex_60"],
}
FAMILIES = list(NAMES)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def ema(x, p):
    a = 2.0 / (p + 1.0); out = np.empty_like(x); out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = a * x[:, t] + (1 - a) * out[:, t - 1]
    return out


def _unpack(W):
    W = np.asarray(W, np.float64)
    return W[..., 0], W[..., 1], W[..., 2], W[..., 3], W[..., 4]


def _zs(x, ref):
    """z-score along axis 1 (mean removed, divided by std + eps scaled by `ref`, a per-window price scale)."""
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + EPS * ref[:, None])


def rsi_series(c, p=14):
    """RSI14 minus 50 over 50, a series on the 60 bars (bar 0 repeats bar 1)."""
    d = np.diff(c, axis=1); up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    rs = ema(up, p) / (ema(dn, p) + 1e-12 * np.abs(c).mean(1, keepdims=True))
    r = (100 - 100 / (1 + rs) - 50) / 50
    return np.concatenate([r[:, :1], r], 1)


# ------------------------------------------------------------------ N1 Frenet geometry of a 3-D curve
def frenet(P):
    """Discrete Frenet quantities of polylines P [N, n, 3]. Returns kappa [N, n-2] (at vertices 1..n-2), tau [N, n-3] (between
    consecutive vertices), phi (turning angles), arc length and chord length."""
    e = np.diff(P, axis=1); ln = np.linalg.norm(e, axis=2); t = e / (ln[..., None] + 1e-300)
    cr = np.cross(t[:, :-1], t[:, 1:]); s = np.linalg.norm(cr, axis=2)
    phi = np.arctan2(s, np.einsum("nij,nij->ni", t[:, :-1], t[:, 1:]))
    ds = 0.5 * (ln[:, :-1] + ln[:, 1:]); kappa = phi / (ds + 1e-300)
    b = cr / (s[..., None] + 1e-300)
    psi = np.arctan2(np.einsum("nij,nij->ni", np.cross(b[:, :-1], b[:, 1:]), t[:, 1:-1]),
                     np.einsum("nij,nij->ni", b[:, :-1], b[:, 1:]))
    tau = psi / (ln[:, 1:-1] + 1e-300)
    chord = np.linalg.norm(P[:, -1] - P[:, 0], axis=1)
    return kappa, tau, phi, ln.sum(1), chord


def n1(W):
    o, h, l, c, v = _unpack(W); ref = np.abs(c).mean(1)
    ax = [c - ema(c, 20), rsi_series(c, 14)]
    m = ema(c, 12) - ema(c, 26); ax.append(m - ema(m, 9))
    refs = [ref, np.ones_like(ref), ref]                    # RSI is dimensionless: its guard must not scale with the price
    P = np.stack([_zs(a, r_)[:, -20:] for a, r_ in zip(ax, refs)], 2)
    kappa, tau, phi, arc, chord = frenet(P)
    return np.stack([slog(kappa.mean(1)), slog(kappa[:, -1]), slog(tau[:, -1]), phi.sum(1),
                     np.log(arc / (chord + 1e-9) + 1e-12)], 1)


# ------------------------------------------------------------------ N2 Levy areas
def levy(X, Y):
    """Normalised signed Levy area of the polylines (X_i, Y_i), [N, n] each: 2 A / (TV_X TV_Y), in [-1, 1]."""
    dx, dy = np.diff(X, axis=1), np.diff(Y, axis=1)
    A = 0.5 * (((X[:, :-1] - X[:, :1]) * dy) - ((Y[:, :-1] - Y[:, :1]) * dx)).sum(1)
    tv = np.abs(dx).sum(1) * np.abs(dy).sum(1)
    return np.where(tv > 1e-300, 2 * A / np.maximum(tv, 1e-300), 0.0)


def n2(W):
    o, h, l, c, v = _unpack(W)
    X = np.log(np.maximum(c, 1e-300)); d = np.diff(c, axis=1)
    cumv = np.concatenate([np.zeros((len(c), 1)), np.cumsum(np.sign(d) * v[:, 1:], axis=1)], 1)
    Ys = [cumv, rsi_series(c, 14), c - ema(c, 20), h - l]
    out = []
    for k in (15, 30, 60):
        n = k + 1 if k < 60 else 60
        for Y in Ys:
            out.append(levy(X[:, -n:], Y[:, -n:]))
    return np.stack(out, 1)


# ------------------------------------------------------------------ N3 phase portrait with winding
def n3(W):
    o, h, l, c, v = _unpack(W)
    s = ema(c, 5); vel = np.diff(s, axis=1); acc = np.diff(vel, axis=1)
    sv = vel.std(1, keepdims=True) + 1e-12 * np.abs(c).mean(1, keepdims=True)
    sa = acc.std(1, keepdims=True) + 1e-12 * np.abs(c).mean(1, keepdims=True)
    vn, an = vel[:, 1:] / sv, acc / sa                     # aligned on the same bars (a_t = v_t - v_{t-1})
    th = np.arctan2(an, vn); dth = np.arctan2(np.sin(np.diff(th, axis=1)), np.cos(np.diff(th, axis=1)))
    r = np.sqrt(vn[:, -1] ** 2 + an[:, -1] ** 2)
    return np.stack([np.cos(th[:, -1]), np.sin(th[:, -1]), dth[:, -1], dth[:, -30:].sum(1), np.log(r + 1e-3)], 1)


# ------------------------------------------------------------------ N4 convex hull
def upper_hull(y):
    """Upper concave hull of the points (t, y[:, t]) for t = 0..T-1, vectorised over rows. Returns the area under the hull, the
    number of vertices and the slope of the last edge (into t = T-1). Collinear points are not vertices."""
    N, T = y.shape; cur = np.zeros(N, int); area = np.zeros(N); nv = np.ones(N); last = np.zeros(N)
    idx = np.arange(T)[None, :]; rows = np.arange(N); act = cur < T - 1
    while act.any():
        yc = y[rows, cur]; dt = (idx - cur[:, None]).astype(float)
        sl = np.where(dt > 0, (y - yc[:, None]) / np.where(dt > 0, dt, 1.0), -np.inf)
        m = sl.max(1); ok = sl >= (m - 1e-12 * (1.0 + np.abs(m)))[:, None]
        nxt = T - 1 - np.argmax(ok[:, ::-1], axis=1)       # the farthest among ties
        nxt = np.where(act, nxt, cur)
        w = nxt - cur
        area += 0.5 * (yc + y[rows, nxt]) * w; nv += act
        hit = act & (nxt == T - 1); last = np.where(hit, m, last)
        cur = nxt; act = cur < T - 1
    return area, nv, last


def n4(W):
    o, h, l, c, v = _unpack(W); ref = np.abs(c).mean(1); sd = np.diff(c, axis=1).std(1) + EPS * ref
    hh, ll = h[:, :-1], l[:, :-1]; T = hh.shape[1]
    aU, nU, sU = upper_hull(hh); aL, nL, sL = upper_hull(-ll); sL, aL = -sL, -aL
    U = hh[:, -1] + sU; L = ll[:, -1] + sL                  # the last edges extended to bar 59
    wd = U - L; pos = np.where(wd > 1e-9 * ref, (c[:, -1] - L) / np.where(wd > 1e-9 * ref, wd, 1.0), 0.5)
    rng = hh.max(1) - ll.min(1)
    area = (aU - aL) / (rng * (T - 1) + EPS * ref)
    return np.stack([slog(sU / sd), slog(sL / sd), np.clip(pos, -2, 3), slog((c[:, -1] - U) / sd), slog((c[:, -1] - L) / sd),
                     area, nU + nL], 1)


# ------------------------------------------------------------------ N5 Wasserstein shift
_Q = (np.arange(40) + 0.5) / 40


def _w1(a, b, sd):
    qa, qb = np.quantile(a, _Q, axis=1), np.quantile(b, _Q, axis=1)
    return np.abs(qa - qb).mean(0) / sd


def n5(W):
    o, h, l, c, v = _unpack(W); r = np.diff(c, axis=1) / c[:, :-1]; sd = r.std(1) + EPS
    a, b = r[:, :29], r[:, 30:]; p, q = r[:, :-15], r[:, -15:]
    return np.stack([slog(_w1(a, b, sd)), slog(_w1(p, q, sd)), slog((np.median(b, 1) - np.median(a, 1)) / sd),
                     slog((np.median(q, 1) - np.median(p, 1)) / sd)], 1)


# ------------------------------------------------------------------ N6 Fisher-Rao path
def fr_dist(mu1, s1, mu2, s2):
    """Fisher-Rao distance between N(mu1, s1^2) and N(mu2, s2^2): sqrt(2) * hyperbolic distance of (mu / sqrt 2, sigma)."""
    dx = (mu1 - mu2) / np.sqrt(2.0)
    return np.sqrt(2.0) * np.arccosh(np.maximum(1.0 + (dx ** 2 + (s1 - s2) ** 2) / (2.0 * s1 * s2), 1.0))


def n6(W):
    o, h, l, c, v = _unpack(W); r = np.diff(c, axis=1) / c[:, :-1]; r = r / (r.std(1, keepdims=True) + EPS)
    mus = np.stack([r[:, 10 * k:10 * k + 9].mean(1) for k in range(6)], 1)
    sig = np.maximum(np.stack([r[:, 10 * k:10 * k + 9].std(1) for k in range(6)], 1), 0.05)
    d = fr_dist(mus[:, :-1], sig[:, :-1], mus[:, 1:], sig[:, 1:]); net = fr_dist(mus[:, 0], sig[:, 0], mus[:, -1], sig[:, -1])
    return np.stack([np.log1p(d.sum(1)), np.log1p(net), np.sign(mus[:, -1] - mus[:, 0]) * np.log1p(net)], 1)


# ------------------------------------------------------------------ N7 convexity (control)
def quad_coef(y):
    """Quadratic coefficient (per bar^2) of the least-squares parabola through y [N, k]."""
    k = y.shape[1]; t = np.arange(k) - (k - 1) / 2.0; t2 = t ** 2 - (t ** 2).mean()
    return (y * t2).sum(1) / (t2 ** 2).sum()


def n7(W):
    o, h, l, c, v = _unpack(W); y = np.log(np.maximum(c, 1e-300)); sd = np.diff(y, axis=1).std(1) + EPS
    return np.stack([quad_coef(y[:, -k:]) / sd for k in (15, 30, 60)], 1)


FN = {"N1": n1, "N2": n2, "N3": n3, "N4": n4, "N5": n5, "N6": n6, "N7": n7}


def compute(W, fams=None, chunk=20000):
    """{family: float32 array [N, k]} for the windows W [N, 60, 5]."""
    fams = fams or FAMILIES; out = {f: [] for f in fams}
    for i in range(0, len(W), chunk):
        w = np.asarray(W[i:i + chunk], np.float64)
        for f in fams:
            out[f].append(np.nan_to_num(FN[f](w), nan=0.0, posinf=0.0, neginf=0.0))
    return {f: np.concatenate(v).astype(np.float32) for f, v in out.items()}
