"""M5 figure step 1 (mm_env, GPU for one anchor): the image VisionTS sees and its reconstruction for the anchor 2026-03-20 14:00 UTC
(the lower-median-vol day of 2026-03 by M2's rule, read from M2_ttm/day_data.csv; the anchor is the rule's 14:00 UTC), plus the day's hourly series.
Writes fig_data.npz, day_data.csv, day_price.csv, ctx_5m.csv. Series (i) ln rv5 is shown (the image of series (ii) is the second channel of the same pass)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "core"))
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf")
import numpy as np, pandas as pd, torch
import anchors as AN, view_cache as VC
import m5_lib as L
from build_view import load_model
TA = pd.Timestamp("2026-03-20 14:00")
m2 = pd.read_csv(os.path.join(HERE, "..", "M2_ttm", "day_data.csv"), index_col=0, parse_dates=True)
assert m2.index[0].date() == TA.date() and len(m2) == 24
v = VC.read_view("M5_visionts", "fwd")
d = m2[["real_logrv", "v3_logrv", "real_vol_bps", "v3_vol_bps"]].join(v[["point_log_rv"]].rename(columns={"point_log_rv": "vts_logsum"}))
d["vol_bps_vts_logsum"] = 1e4 * np.sqrt(np.exp(d["vts_logsum"]))
d.to_csv(os.path.join(HERE, "day_data.csv"))
store = AN.get_store("fwd"); frame = L.make_frame(store)
idx, X = L.context(frame, TA)
model = load_model()
with torch.no_grad():
    y, img_in, img_rec = model(torch.tensor(X[None], dtype=torch.float32).cuda(), export_image=True)
y = y[0].cpu().numpy(); img_in = img_in[0, 0].cpu().numpy()[..., 0]; img_rec = img_rec[0, 0].cpu().numpy().mean(-1)   # series (i); grey
npi = model.num_patch_input * model.patch_size
fut = pd.date_range(TA, periods=12, freq="5min"); real = frame.reindex(fut)["lrv"].to_numpy(float)
np.savez(os.path.join(HERE, "fig_data.npz"), img_in=img_in, img_rec=img_rec, npi=npi, y_rv=y[:, 0], y_vol=y[:, 1], real_rv=real, ctx_idx=idx.values.astype("datetime64[ns]"), ctx_rv=X[:, 0], ctx_vol=X[:, 1], fut=fut.values.astype("datetime64[ns]"))
m5 = store.m5(); m5.loc[TA.normalize():TA.normalize() + pd.Timedelta(hours=24) - pd.Timedelta(minutes=5), "close"].to_csv(os.path.join(HERE, "day_price.csv"))
print("anchor", TA, "image", img_in.shape, "input columns", npi, "forecast ln rv5", y[:, 0].round(2), "real", real.round(2))
