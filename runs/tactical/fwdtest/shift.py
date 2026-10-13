"""Per-feature shift of 2025-10 (calibration month) under the frozen training scaler. Features only; no label, no score."""
import json, time
import numpy as np, pandas as pd
import fwdtest_lib as T
t0 = time.time()
mdl, sc, ntr, span = T.train_frozen()
s = T.load_fwd_bars(); X, t, r, _ = T.anchors(s)
ti = pd.DatetimeIndex(t.astype("datetime64[ns]")); m = np.asarray((ti >= "2025-10-01") & (ti < "2025-11-01"))
Z = sc.transform(np.asarray(X[m], np.float32)); names = T.feature_names()
rows = [{"i": i, "name": names[i], "volume_family": i in T.VOLUME_IDX, "train_mean": float(sc.mean_[i]), "train_std": float(sc.scale_[i]),
         "new_z_mean": float(Z[:, i].mean()), "new_z_std": float(Z[:, i].std()), "frac_clipped_abs_z_gt_8": float((np.abs(Z[:, i]) > 8).mean())} for i in range(43)]
out = {"n_train_rows": ntr, "train_span": span, "n_oct_anchors": int(m.sum()), "first_oct_anchor": str(ti[m].min()), "features": rows,
       "volume_families_mean_abs_z_mean": float(np.mean([abs(r_["new_z_mean"]) for r_ in rows if r_["volume_family"]])),
       "other_families_mean_abs_z_mean": float(np.mean([abs(r_["new_z_mean"]) for r_ in rows if not r_["volume_family"]])),
       "volume_families_mean_z_std": float(np.mean([r_["new_z_std"] for r_ in rows if r_["volume_family"]])),
       "other_families_mean_z_std": float(np.mean([r_["new_z_std"] for r_ in rows if not r_["volume_family"]]))}
json.dump(out, open("shift_2025-10.json", "w"), indent=1)
with open("shift_2025-10.txt", "w") as f:
    f.write(f"training rows {ntr}, span {span}; October anchors {int(m.sum())} (first {out['first_oct_anchor']})\n")
    f.write(f"{'feature':22s} {'vol':3s} {'z_mean':>8s} {'z_std':>8s} {'clip>8':>7s}\n")
    for r_ in sorted(rows, key=lambda r_: (not r_["volume_family"], -abs(r_["new_z_mean"]))):
        f.write(f"{r_['name']:22s} {'V' if r_['volume_family'] else '':3s} {r_['new_z_mean']:8.3f} {r_['new_z_std']:8.3f} {r_['frac_clipped_abs_z_gt_8']:7.4f}\n")
    f.write(f"mean |z_mean|: volume {out['volume_families_mean_abs_z_mean']:.3f}, other {out['other_families_mean_abs_z_mean']:.3f}; mean z_std: volume {out['volume_families_mean_z_std']:.3f}, other {out['other_families_mean_z_std']:.3f}\n")
print(open("shift_2025-10.txt").read()); print(round(time.time() - t0, 1), "s")
