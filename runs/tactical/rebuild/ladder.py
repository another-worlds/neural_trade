"""Rebuild ladder (PLAN 9, AUDIT 5): a small Keras (TF 2.10) model family on the lab's cached windows, one step per flag.
Every step is scored with lab.evaluate_slice on the 24 slices and appended to results.jsonl (same per-slice schema as the lab).
Selection and early stopping use an INNER validation split = the last 15% of each slice's TRAIN block (a 20-row gap before it);
the val block is only ever scored. CPU only; seeds fixed.

  L0  sklearn logistic regression on tb7, C=0.1 (the bar)
  L1  the same features and L2 in Keras: Dense(1) zero-init per horizon, masked BCE, Adam full batch to convergence
  L3  the three horizons trained jointly (Dense(3), summed masked BCE) = three independent logistic heads
  L5  `rich` features (43), linear; L2 (C) per horizon chosen on the inner validation split
  L6  L5 + residual MLP (Dense 32 gelu, dropout 0.2, zero-init output) added to the linear logit
  L7  L5 + GRU(32) over train-standardised scale-free per-bar channels, zero-init Dense(3) added to the linear logit

usage: python ladder.py --step L0|L1|L3|L5|L6|L7 [--slices N] [--tag name]      python ladder.py --report"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, glob, json, math, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab"))
import lab  # noqa: E402  (chdirs to the repo root; CACHE is the shared slice cache)
import tensorflow as tf  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = os.environ.get("LADDER_OUT", "runs/tactical/rebuild/results.jsonl")
C0 = 0.1                                   # the L2 of the bar (sklearn C)
CGRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
INNER, GAP = 0.15, lab.GAP
LIN_STEPS, LIN_LR = 2000, 1e-2             # full-batch Adam, cosine-decayed: converges on <=43 standardised features
NL_LR, NL_BATCH, NL_EPOCHS, NL_PATIENCE = 1e-3, 512, 30, 4


# ------------------------------------------------------------------ data
def load_slice(path, fs_name):
    D = np.load(path); db = float(D["deadband"]); rtr = D["rtr"]
    Y = (rtr > 0).astype(np.float32); M = (np.abs(rtr) > db).astype(np.float32)
    fs = lab.SETS[fs_name]
    Ftr, Fva = np.nan_to_num(fs(D["Wtr"])), np.nan_to_num(fs(D["Wva"]))
    sc = StandardScaler().fit(Ftr)
    n = len(Ftr); cut = int(round((1 - INNER) * n))
    return dict(D=D, Ftr=sc.transform(Ftr).astype(np.float32), Fva=sc.transform(Fva).astype(np.float32), Y=Y, M=M,
                itr=np.arange(0, cut - GAP), iva=np.arange(cut, n))


def seq_channels(W):
    """[N,60,5] scale-free per-bar channels: 1-bar return / sd, (close - EMA_p)/sd for p = 5, 20, (high-low)/sd,
    log1p(volume / window mean volume). sd = the window's sd of 1-bar changes (as the lab's features)."""
    o, h, l, c, v, d, sd = lab.parts(W); s = sd[:, None]
    r1 = np.concatenate([np.zeros((len(c), 1)), d], 1) / s
    ch = [r1, (c - lab.ema(c, 5)) / s, (c - lab.ema(c, 20)) / s, (h - l) / s, np.log1p(v / (v.mean(1, keepdims=True) + 1e-12))]
    return np.nan_to_num(np.stack(ch, 2)).astype(np.float32)


def add_seq(S):
    Qtr, Qva = seq_channels(S["D"]["Wtr"]), seq_channels(S["D"]["Wva"])
    mu, sd = Qtr.mean((0, 1)), Qtr.std((0, 1)) + 1e-6          # per channel over the train block, NOT per window
    S["Qtr"], S["Qva"] = (Qtr - mu) / sd, (Qva - mu) / sd


# ------------------------------------------------------------------ models and the one training loop
def make_model(F, k=3, mlp=False, seq=False):
    """Linear logit (zero-init, the L2-penalised part) plus optional zero-init residual branches. Returns (model, linear layer)."""
    L = tf.keras.layers
    x = L.Input((F,)); lin = L.Dense(k, kernel_initializer="zeros", name="lin"); logit = lin(x); ins = [x]
    if mlp:
        h = L.Dropout(0.2)(L.Dense(32, activation="gelu")(x)); logit = L.Add()([logit, L.Dense(k, kernel_initializer="zeros")(h)])
    if seq:
        q = L.Input((60, 5)); g = L.Dropout(0.2)(L.GRU(32)(q)); logit = L.Add()([logit, L.Dense(k, kernel_initializer="zeros")(g)])
        ins.append(q)
    return tf.keras.Model(ins, logit), lin


def masked_bce(logit, y, m):
    """Per-column mean BCE over the masked (outside the deadband) rows."""
    l = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)
    return tf.reduce_sum(l * m, 0) / tf.maximum(tf.reduce_sum(m, 0), 1.0)


def fit(model, lin, X, Y, M, lam, *, lr, epochs, batch=None, val=None, patience=0, decay=True, seed=0):
    """Adam on sum_h masked-BCE_h + sum_h lam_h * ||W_h||^2 (W = the linear kernel; lam_h = 1/(2 C n_h) matches sklearn's C).
    batch=None: full batch, `epochs` steps. val=(X, Y, M): evaluate every epoch, keep the best weights (epoch 0 = start),
    stop after `patience` epochs without gain. Returns (best_epoch, best_val)."""
    n = len(Y); lam = tf.constant(lam, tf.float32); spe = 1 if batch is None else n // batch
    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(lr, epochs * spe, alpha=1e-2) if decay else lr)
    kern = lin.kernel

    @tf.function
    def step(xb, yb, mb):
        with tf.GradientTape() as t:
            loss = tf.reduce_sum(masked_bce(model(xb, training=True), yb, mb)) + tf.reduce_sum(lam * tf.reduce_sum(kern ** 2, 0))
        vs = model.trainable_variables; opt.apply_gradients(zip(t.gradient(loss, vs), vs))

    @tf.function
    def vloss(xb, yb, mb):
        return tf.reduce_sum(masked_bce(model(xb, training=False), yb, mb))

    def score():
        return float(vloss([tf.constant(x) for x in val[0]], tf.constant(val[1]), tf.constant(val[2])))

    Xt, Yt, Mt = [tf.constant(x) for x in X], tf.constant(Y), tf.constant(M); rs = np.random.RandomState(seed)
    best = (score() if val else math.inf, model.get_weights(), 0); bad = 0
    for ep in range(1, epochs + 1):
        if batch is None:
            step(Xt, Yt, Mt)
        else:
            perm = rs.permutation(n)
            for i in range(spe):
                ix = perm[i * batch:(i + 1) * batch]; step([x[ix] for x in X], Y[ix], M[ix])
        if val:
            s = score()
            if s < best[0] - 1e-6:
                best = (s, model.get_weights(), ep); bad = 0
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    if val:
        model.set_weights(best[1])
    return best[2], best[0]


def lam_vec(C, M):
    """L2 coefficient per horizon column for sklearn-equivalent strength C (scalar or per column)."""
    return 1.0 / (2.0 * np.asarray(C, np.float64) * M.sum(0))


def predict(model, X):
    return tf.sigmoid(model([tf.constant(x) for x in X], training=False)).numpy().astype(np.float64)


# ------------------------------------------------------------------ steps (each returns P_val [N,3] for a slice)
def step_L0(S):
    D, P = S["D"], np.zeros((len(S["Fva"]), 3))
    for i in range(3):
        m = S["M"][:, i] > 0
        P[:, i] = LogisticRegression(C=C0, max_iter=1000).fit(S["Ftr"][m], S["Y"][m, i].astype(int)).predict_proba(S["Fva"])[:, 1]
    return P


def step_L1(S):
    P = np.zeros((len(S["Fva"]), 3))
    for i in range(3):
        y, m = S["Y"][:, i:i + 1], S["M"][:, i:i + 1]
        model, lin = make_model(S["Ftr"].shape[1], k=1)
        fit(model, lin, [S["Ftr"]], y, m, lam_vec(C0, m), lr=LIN_LR, epochs=LIN_STEPS)
        P[:, i] = predict(model, [S["Fva"]])[:, 0]
    return P


def step_L3(S):
    model, lin = make_model(S["Ftr"].shape[1])
    fit(model, lin, [S["Ftr"]], S["Y"], S["M"], lam_vec(C0, S["M"]), lr=LIN_LR, epochs=LIN_STEPS)
    return predict(model, [S["Fva"]])


def l5_stage(S):
    """Choose C per horizon on the inner split. Returns (Cs, inner weights [kernel, bias] at the chosen Cs, full-train weights)."""
    F, Y, M, itr, iva = S["Ftr"], S["Y"], S["M"], S["itr"], S["iva"]; ker, bia, val = [], [], []
    for C in CGRID:
        model, lin = make_model(F.shape[1])
        fit(model, lin, [F[itr]], Y[itr], M[itr], lam_vec(C, M[itr]), lr=LIN_LR, epochs=LIN_STEPS)
        w = model.get_weights(); ker.append(w[0]); bia.append(w[1])
        val.append(masked_bce(model(F[iva]), Y[iva], M[iva]).numpy())
    pick = np.argmin(np.array(val), 0)                       # [3]
    Cs = np.array(CGRID)[pick]
    inner = [np.stack([ker[pick[h]][:, h] for h in range(3)], 1), np.array([bia[pick[h]][h] for h in range(3)], np.float32)]
    model, lin = make_model(F.shape[1])
    fit(model, lin, [F], Y, M, lam_vec(Cs, M), lr=LIN_LR, epochs=LIN_STEPS)
    return Cs, inner, model.get_weights(), model


def step_L5(S):
    Cs, _, _, model = l5_stage(S); S["info"] = dict(C=[float(c) for c in Cs])
    return predict(model, [S["Fva"]])


def residual_step(S, mlp, seq):
    """L6 / L7: start at L5's solution, add the zero-init branch, early-stop on the inner split, refit on the full train block
    for the best epoch count (0 = the branch never helped: the L5 model is returned)."""
    Cs, inner, full, l5model = l5_stage(S); F, Y, M, itr, iva = S["Ftr"], S["Y"], S["M"], S["itr"], S["iva"]
    Xs = [F] + ([S["Qtr"]] if seq else []); tf.keras.utils.set_random_seed(0)
    model, lin = make_model(F.shape[1], mlp=mlp, seq=seq); lin.set_weights(inner)
    be, bv = fit(model, lin, [x[itr] for x in Xs], Y[itr], M[itr], lam_vec(Cs, M[itr]), lr=NL_LR, epochs=NL_EPOCHS,
                 batch=NL_BATCH, val=([x[iva] for x in Xs], Y[iva], M[iva]), patience=NL_PATIENCE, decay=False)
    S["info"] = dict(C=[float(c) for c in Cs], best_epoch=be)
    Xv = [S["Fva"]] + ([S["Qva"]] if seq else [])
    if be == 0:
        return predict(l5model, [S["Fva"]])
    tf.keras.utils.set_random_seed(0)
    model, lin = make_model(F.shape[1], mlp=mlp, seq=seq); lin.set_weights(full)
    fit(model, lin, Xs, Y, M, lam_vec(Cs, M), lr=NL_LR, epochs=be, batch=NL_BATCH, decay=False)
    return predict(model, Xv)


STEPS = {"L0": (step_L0, "tb7"), "L1": (step_L1, "tb7"), "L3": (step_L3, "tb7"), "L5": (step_L5, "rich"),
         "L6": (lambda S: residual_step(S, True, False), "rich"), "L7": (lambda S: residual_step(S, False, True), "rich")}


# ------------------------------------------------------------------ run and report
def run(step, n_slices=None, tag=None):
    fn, fs_name = STEPS[step]; t0 = time.time(); rng = np.random.default_rng(0); per = {}; infos = []; fit_s = 0.0
    paths = sorted(glob.glob(lab.CACHE))[:n_slices]
    for p in paths:
        tf.keras.utils.set_random_seed(0)
        S = load_slice(p, fs_name)
        if step == "L7":
            add_seq(S)
        t1 = time.time(); P = fn(S); fit_s += time.time() - t1
        for k, v in lab.evaluate_slice(S["D"], P, rng).items():
            per.setdefault(k, []).append(float(v))
        infos.append(S.get("info", {}))
        print(f"{step} {os.path.basename(p)} auc3 {per['auc3'][-1]:.4f}", flush=True)
    res = {"step": step, "tag": tag or step, "n_slices": len(paths), "seconds": round(time.time() - t0, 1),
           "fit_seconds": round(fit_s, 1), "per_slice": {k: [round(x, 5) for x in v] for k, v in per.items()}, "info": infos}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    print(f"{step}: mean3 {np.mean(per['auc3']):.4f}  fit {res['fit_seconds']} s  total {res['seconds']} s")
    return res


def tci(v):
    v = np.asarray(v, float); m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - h, m + h


def report():
    latest = {}
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l)
        if r["n_slices"] == 24:
            latest[r["step"]] = r
    order = [s for s in ("L0", "L1", "L3", "L5", "L6", "L7") if s in latest]
    print("step | mean3 AUC [95% CI] | d vs prev [CI] | d vs L0 [CI] | h1 ll - const | top10% hit | bps (null95) | fit s")
    prev = None
    for s in order:
        r = latest[s]; ps = r["per_slice"]; a = np.array(ps["auc3"]); m = tci(a)
        d = lambda base: "%+.4f [%+.4f,%+.4f]" % tci(a - np.array(latest[base]["per_slice"]["auc3"])) if base else "-"
        ll = np.mean(np.array(ps["ll_h1"]) - np.array(ps["ll_const_h1"]))
        print(f"{s} | {m[0]:.4f} [{m[1]:.4f},{m[2]:.4f}] | {d(prev)} | {d('L0') if s != 'L0' else '-'} | {ll:+.4f} | "
              f"{np.mean(ps['hon10_hit']):.4f} | {np.mean(ps['hon10_bps']):+.2f} ({np.mean(ps['hon10_null95']):+.2f}) | {r['fit_seconds']}")
        prev = s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--step", choices=sorted(STEPS)); ap.add_argument("--slices", type=int)
    ap.add_argument("--tag"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    report() if a.report else run(a.step, a.slices, a.tag)
