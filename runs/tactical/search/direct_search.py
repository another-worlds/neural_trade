"""Hypothesis C (lead, 2026-10-08): the indicator parameter space is small and bounded, so search it DIRECTLY (Optuna,
TPE) with a simple readout - logistic regression on indicator values at the window's end - instead of gradient descent
through a 300k-parameter network. This is the VISION's "manual search" baseline that learned indicators must beat.
Fixed before the run:
  data      the long block of the hill-climb (excerpt, MAX_SEQUENCE_COUNT 126,000, fold -2): train block -> fit, val block -> score
  target    h1 (15 min) direction, deadband 5 bps, AUC
  features  one instance each of EMA distance, RSI, MACD histogram, Bollinger %b, ATR, Stochastic %K, CCI (7 families, 9 periods),
            computed on each 60-bar OHLCV window (periods 2..60), plus the 6 trailing returns of DIRECTION_SKIP
  search    150 TPE trials maximise the mean val AUC over the 3 SEARCH slices (climb slices 1, 3, 5);
            the 3 CHECK slices (climb 2, 4, 6) are scored once for: textbook periods, the searched periods, and the network
            (hc5_noprice3, h1 direction AUC, mean of its 2 seeds) - none of them chooses anything on the check slices.
usage: python direct_search.py"""
import json, os, sys, time
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import neural_trade  # noqa: F401
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor
from neural_trade.experiments import screen as S

sys.path.insert(0, r"D:\nt\nt_tactical\runs\tactical\hc4"); sys.path.insert(0, r"D:\nt\nt_tactical\runs\tactical")
import make_hc4 as H

SEARCH, CHECK = H.CLIMB[0::2], H.CLIMB[1::2]
OV = dict(H.WEEK)
TEXTBOOK = {"ema": 20, "rsi": 14, "macd_f": 12, "macd_s": 26, "macd_g": 9, "bb": 20, "atr": 14, "stoch": 14, "cci": 20}


def ema(x, p):  # x [N, L] -> EMA along axis 1, value at each step
    a = 2.0 / (p + 1.0); out = np.empty_like(x); out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = a * x[:, t] + (1 - a) * out[:, t - 1]
    return out


def feats(W, prm):
    """W [N, 60, 5] OHLCV window (window-relative scaling does not matter: every feature is scale-free)."""
    o, h, l, c = W[..., 0], W[..., 1], W[..., 2], W[..., 3]
    sd = np.std(np.diff(c, axis=1), axis=1) + 1e-9
    f = []
    f.append((c[:, -1] - ema(c, prm["ema"])[:, -1]) / sd)
    d = np.diff(c, axis=1); up = np.clip(d, 0, None); dn = np.clip(-d, 0, None)
    rs = ema(up, prm["rsi"])[:, -1] / (ema(dn, prm["rsi"])[:, -1] + 1e-9); f.append(100 - 100 / (1 + rs))
    fs, ss = sorted((prm["macd_f"], prm["macd_s"]))
    macd = ema(c, fs) - ema(c, max(ss, fs + 1)); sig = ema(macd, prm["macd_g"]); f.append((macd - sig)[:, -1] / sd)
    p = prm["bb"]; win = c[:, -p:]; f.append((c[:, -1] - win.mean(1)) / (2 * win.std(1) + 1e-9))
    tr = np.maximum(h[:, 1:] - l[:, 1:], np.maximum(np.abs(h[:, 1:] - c[:, :-1]), np.abs(l[:, 1:] - c[:, :-1])))
    f.append(ema(tr, prm["atr"])[:, -1] / sd)
    p = prm["stoch"]; lo, hi = l[:, -p:].min(1), h[:, -p:].max(1); f.append((c[:, -1] - lo) / (hi - lo + 1e-9))
    p = prm["cci"]; tp = (h + l + c)[:, -p:] / 3; f.append((tp[:, -1] - tp.mean(1)) / (0.015 * np.abs(tp - tp.mean(1, keepdims=True)).mean(1) + 1e-9))
    for k in (1, 5, 10, 15, 20, 30):
        f.append((c[:, -1] - c[:, -1 - k]) / sd)
    return np.nan_to_num(np.stack(f, 1))


DATA = {}
for de in H.CLIMB:
    t = time.time()
    cfg = Config.from_yaml("configs/default.yaml").override(**OV, DATA_END=de, SEED=0)
    cache = {}; S._load_cached(cfg, cache)
    X_seq, y_seq, lc_seq, ext, X_model = S._windowed_cached(cfg, cache)
    dp = DataProcessor(cfg); dp.prepare_datasets_from_windows(X_seq, y_seq, lc_seq, ext, X_model=X_model)
    fo = dp.fold; db = float(cfg.DIR_DEADBAND_BPS) / 1e4
    def lab(idx):
        r = y_seq[idx, 1] / lc_seq[idx].reshape(-1); m = np.abs(r) > db; return (r > 0).astype(int), m
    DATA[de] = {"Wtr": X_model[fo.train].astype(np.float64), "Wva": X_model[fo.val].astype(np.float64),
                "ytr": lab(fo.train), "yva": lab(fo.val)}
    print(f"loaded {de} in {time.time()-t:.1f}s train {len(fo.train)} val {len(fo.val)}", flush=True)


def score(prm, slices):
    aucs = []
    for de in slices:
        D = DATA[de]; Ftr, Fva = feats(D["Wtr"], prm), feats(D["Wva"], prm)
        (ytr, mtr), (yva, mva) = D["ytr"], D["yva"]
        sc = StandardScaler().fit(Ftr[mtr]); m = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(Ftr[mtr]), ytr[mtr])
        aucs.append(roc_auc_score(yva[mva], m.predict_proba(sc.transform(Fva[mva]))[:, 1]))
    return float(np.mean(aucs)), [float(a) for a in aucs]


def objective(trial):
    prm = {k: trial.suggest_int(k, 2, 60) for k in TEXTBOOK}
    return score(prm, SEARCH)[0]


optuna.logging.set_verbosity(optuna.logging.WARNING)
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
st.enqueue_trial(TEXTBOOK)
t0 = time.time(); st.optimize(objective, n_trials=150)
best = st.best_params
res = {"search_slices": SEARCH, "check_slices": CHECK, "trials": 150, "seconds": round(time.time() - t0),
       "textbook": TEXTBOOK, "searched": best,
       "search_auc": {"textbook": score(TEXTBOOK, SEARCH), "searched": score(best, SEARCH)},
       "check_auc": {"textbook": score(TEXTBOOK, CHECK), "searched": score(best, CHECK)}}
net = {}
for f in __import__("glob").glob("runs/tactical/screens/hc5_noprice3/results*.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); net.setdefault(r["data_end"], []).append(r["direction_auc"]["h1"]["auc"])
res["check_auc"]["network_hc5_noprice3_h1"] = (float(np.mean([np.mean(net[s]) for s in CHECK if s in net])), [float(np.mean(net[s])) for s in CHECK if s in net])
json.dump(res, open("runs/tactical/search/direct_search.json", "w"), indent=1)
print(json.dumps(res, indent=1))
