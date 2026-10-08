"""How the 9 heads relate on saved validation predictions (screen run.save_predictions npz), long-block base runs.
Questions: (1) does the direction head know more than the sign of the price head, or than the Gaussian P(up) implied
by price + variance; (2) is the price head ranked right but scaled wrong (shrinkage beta, its skill at the best beta);
(3) is it biased; (4) do the horizons' heads agree; (5) variance ranking. Prints per horizon means over runs."""
import glob, json, sys
import numpy as np
from math import erf, sqrt
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

spec = sys.argv[1] if len(sys.argv) > 1 else "cand_c2_base"
files = sorted(glob.glob(f"runs/tactical/screens/{spec}/preds/*.npz"))
ncdf = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
acc = {h: {} for h in range(3)}
for f in files:
    d = np.load(f)
    y, lc = d["y"], d["last_close"].reshape(-1)
    db = float(d["deadband_bps"]) / 1e4 if "deadband_bps" in d else 5e-4
    ps = float(d["pred_scale"]) if "pred_scale" in d else 1.0
    for h in range(3):
        yt, dl, pu, vr = y[:, h], d[f"delta_h{h}"], d[f"p_up_h{h}"], d[f"var_h{h}"]
        ret = yt / lc; m = np.abs(ret) > db; lab = (ret[m] > 0).astype(int)
        sig = np.sqrt(np.maximum(vr, 1e-12)) * ps
        g_up = ncdf(dl / np.maximum(sig, 1e-12))            # Gaussian P(up) from price + variance heads
        r = acc[h]
        def put(k, v): r.setdefault(k, []).append(v)
        put("auc_direction_head", roc_auc_score(lab, pu[m]))
        put("auc_price_head", roc_auc_score(lab, dl[m]))
        put("auc_gauss_from_price_var", roc_auc_score(lab, g_up[m]))
        put("corr_dir_vs_price", np.corrcoef(pu, dl)[0, 1])
        beta = float(np.dot(dl, yt) / max(np.dot(dl, dl), 1e-12))
        put("shrink_beta", beta)
        mse0 = np.mean(yt ** 2)
        put("skill_raw", 1 - np.mean((yt - dl) ** 2) / mse0)
        put("skill_at_best_beta", 1 - np.mean((yt - max(beta, 0) * dl) ** 2) / mse0)
        put("corr_price_y", np.corrcoef(dl, yt)[0, 1])
        put("bias_pred_minus_true_mean_over_sd", (dl.mean() - yt.mean()) / yt.std())
        put("sd_pred_over_sd_true", dl.std() / yt.std())
        put("spearman_var_err2", spearmanr(vr, (yt - dl) ** 2).correlation)
        put("spearman_var_absy", spearmanr(vr, np.abs(yt)).correlation)
        put("mean_abs_p_minus_half", np.mean(np.abs(pu - 0.5)))
    for a, b in ((0, 1), (1, 2)):
        acc[a].setdefault(f"corr_price_h{a}_h{b}", []).append(np.corrcoef(d[f"delta_h{a}"], d[f"delta_h{b}"])[0, 1])
        acc[a].setdefault(f"corr_pup_h{a}_h{b}", []).append(np.corrcoef(d[f"p_up_h{a}"], d[f"p_up_h{b}"])[0, 1])
out = {f"h{h}": {k: round(float(np.mean(v)), 4) for k, v in acc[h].items()} for h in range(3)}
out["runs"] = len(files)
print(json.dumps(out, indent=1))
json.dump(out, open(f"runs/tactical/probe/heads_interplay_{spec}.json", "w"), indent=1)
