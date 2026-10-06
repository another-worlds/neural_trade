"""logreg_lags reference on the SAME train/val blocks as the hill-climb screen trials (one value per slice;
deterministic, no seeds). Mean of val direction AUC over h0-h2, same deadband mask as screen._direction_auc."""
import json, sys, yaml
import numpy as np
import neural_trade  # noqa: F401  (before tensorflow)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor
from neural_trade.evaluation.baselines import lag_features
from neural_trade.experiments import screen as S
from neural_trade.metrics.direction_labels import direction_labels_np

spec_path = sys.argv[1]
spec = yaml.safe_load(open(spec_path))
base = Config.from_yaml("configs/default.yaml").override(**spec["overrides"])
cache, out = {}, {}
for de in spec["slices"]:
    cfg = base.override(DATA_END=de, SEED=0)
    S._load_cached(cfg, cache)
    X_seq, y_seq, lc_seq, ext, X_model = S._windowed_cached(cfg, cache)
    dp = DataProcessor(cfg)
    dp.prepare_datasets_from_windows(X_seq, y_seq, lc_seq, ext, X_model=X_model)
    f = dp.fold
    Xtr, ytr, ltr = X_seq[f.train], y_seq[f.train], lc_seq[f.train]
    Xva, yva, lva = X_seq[f.val], y_seq[f.val], lc_seq[f.val]
    db = float(cfg.DIR_DEADBAND_BPS)
    labtr = direction_labels_np(ytr, ltr, db)
    sc = StandardScaler().fit(lag_features(Xtr))
    Ftr, Fva = sc.transform(lag_features(Xtr)), sc.transform(lag_features(Xva))
    aucs = {}
    for i, h in enumerate(("h0", "h1", "h2")):
        lab, mask = labtr[h]
        m = LogisticRegression(C=1.0, max_iter=1000).fit(Ftr[mask], lab[mask])
        ret = yva[:, i] / np.where(np.abs(lva) > 1e-9, lva, np.nan)
        mk = np.isfinite(ret) & (np.abs(ret) > db / 10000.0)
        yt = (ret[mk] > 0).astype(int)
        aucs[h] = float(roc_auc_score(yt, m.predict_proba(Fva)[mk, 1])) if len(np.unique(yt)) == 2 else None
    out[de] = {"auc": aucs, "mean": float(np.mean([v for v in aucs.values() if v is not None])),
               "n_train": int(len(Xtr)), "n_val": int(len(Xva))}
    print(de[:10], out[de], flush=True)
json.dump(out, open("runs/tactical/hc_logreg.json", "w"), indent=1)
print("mean over slices %.4f" % np.mean([v["mean"] for v in out.values()]))
