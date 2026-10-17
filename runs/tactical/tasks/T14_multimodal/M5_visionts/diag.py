"""M5 diagnostics (CPU): bias, rmse, correlation of the point forecast against the target and persistence (as M2/diag.txt), and of the seasonal term."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, anchors as AN, view_cache as VC
for s in ("cut", "fwd"):
    b = AN.load_base(s); y = b.A["T"]["A"] + b.A["LV0"]
    for name, col in (("M5_visionts", "point_log_rv"), ("M5_visionts", "vts_logrv_minus_last")):
        v = VC.read_view(name, s).reindex(b.ta); x = v[col].to_numpy(); m = b.ok & np.isfinite(x)
        if col == "point_log_rv":
            e = x[m] - y[m]; e0 = b.A["LV0"][m] - y[m]
            print(s, "vts point - target: mean %.3f sd %.3f rmse %.3f | persistence rmse %.3f | corr(vts,target) %.3f corr(LV0,target) %.3f | n %d" % (e.mean(), e.std(), np.sqrt((e**2).mean()), np.sqrt((e0**2).mean()), np.corrcoef(x[m], y[m])[0, 1], np.corrcoef(b.A["LV0"][m], y[m])[0, 1], m.sum()))
    v = VC.read_view("M5b_visionts_seasonal", s).reindex(b.ta)["vts_seasonal_resid"].to_numpy(); m = b.ok & np.isfinite(v)
    print(s, "seasonal resid: sd %.3f corr with (target - LV0) %.3f" % (v[m].std(), np.corrcoef(v[m], (y - b.A["LV0"])[m])[0, 1]))
