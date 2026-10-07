"""Why does the slice DATA_END=2022-04-19T07:00 give every configuration a high direction AUC (base mean 0.67)?
Marks the train / val block time ranges of that screen trial, describes the price path in each, and scores simple
model-free rules on the same val block with the same label mask as screen._direction_auc. Contrast: two slices
where every configuration was below 0.5. CPU only."""
import json, sys
import numpy as np
import neural_trade  # noqa: F401
from sklearn.metrics import roc_auc_score
from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor
from neural_trade.experiments import screen as S

OVER = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
        "MAX_SEQUENCE_COUNT": 4500, "FOLD_INDEX": -2, "BATCH_SIZE": 64, "SEED": 0}
SLICES = sys.argv[1:] or ["2022-04-19T07:00:00", "2022-10-23T20:00:00", "2024-07-09T22:00:00"]
out = {}
for de in SLICES:
    cfg = Config.from_yaml("configs/default.yaml").override(**OVER, DATA_END=de)
    cache = {}
    df, close = S._load_cached(cfg, cache)
    X_seq, y_seq, lc_seq, ext, X_model = S._windowed_cached(cfg, cache)
    dp = DataProcessor(cfg)
    dp.prepare_datasets_from_windows(X_seq, y_seq, lc_seq, ext, X_model=X_model)
    f = dp.fold
    closes = np.asarray(close, dtype=float).reshape(-1)
    idx = df.index if hasattr(df, "index") else None

    def when(lc_block):
        # locate the block's anchor bars in the full close series by matching the first 3 last-close values
        a = lc_block.reshape(-1)
        cand = np.where(np.isclose(closes, a[0]))[0]
        for c in cand:
            if c + len(a) <= len(closes) and np.allclose(closes[c:c + 3], a[:3]):
                return int(c), str(idx[c]) if idx is not None else None, str(idx[c + len(a) - 1]) if idx is not None else None
        return None, None, None

    res = {}
    for name, ids in (("train", f.train), ("val", f.val)):
        X, y, lc = X_seq[ids], y_seq[ids], lc_seq[ids]
        start, t0, t1 = when(lc)
        p = lc.reshape(-1)
        r = {"n": int(len(ids)), "start": t0, "end": t1, "price_first": float(p[0]), "price_last": float(p[-1]),
             "return_pct": float(100 * (p[-1] / p[0] - 1)), "range_pct": float(100 * (p.max() / p.min() - 1)),
             "vol_1m_bps": float(1e4 * np.std(np.diff(np.log(p))))}
        db = float(cfg.DIR_DEADBAND_BPS) / 1e4
        ups = []
        for i, h in enumerate(cfg.HORIZON_STEPS):
            ret = y[:, i] / lc.reshape(-1)
            m = np.abs(ret) > db
            ups.append(float((ret[m] > 0).mean()))
        r["up_share_h0_h2"] = ups
        if name == "val":
            logx = np.log(X)
            rules = {}
            for k in (1, 5, 10, 15, 30, 59):
                mom = logx[:, -1] - logx[:, -1 - k]
                aucs = []
                for i, h in enumerate(cfg.HORIZON_STEPS):
                    ret = y[:, i] / lc.reshape(-1); m = np.abs(ret) > db
                    yt = (ret[m] > 0).astype(int)
                    aucs.append(float(roc_auc_score(yt, mom[m])) if len(set(yt)) == 2 else None)
                rules[f"momentum_{k}"] = {"auc_h0_h2": aucs, "mean": float(np.mean(aucs))}
            r["rules"] = rules
            # autocorrelation of 10-bar returns inside the val block (trend or mean reversion?)
            r10 = np.diff(np.log(p[::10]))
            r["acf1_10bar"] = float(np.corrcoef(r10[:-1], r10[1:])[0, 1])
        res[name] = r
    out[de] = res
    v = res["val"]
    best = max(v["rules"].items(), key=lambda kv: abs(kv[1]["mean"] - 0.5))
    print(f"\n== {de}")
    for name in ("train", "val"):
        b = res[name]
        print(f"  {name:5s} {b['start']} -> {b['end']}  n={b['n']}  price {b['price_first']:.0f} -> {b['price_last']:.0f} "
              f"({b['return_pct']:+.2f}%), range {b['range_pct']:.2f}%, 1-min vol {b['vol_1m_bps']:.1f} bps, up share {['%.2f' % u for u in b['up_share_h0_h2']]}")
    print("  val rules (AUC of the k-bar momentum sign; < 0.5 = reversal works):", {k: round(x["mean"], 3) for k, x in v["rules"].items()})
    print(f"  val acf of 10-bar returns {v['acf1_10bar']:+.3f}; strongest rule {best[0]} {best[1]['mean']:.3f}")
json.dump(out, open("runs/tactical/slice_2022_04_19.json", "w"), indent=1, default=str)
