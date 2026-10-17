"""Step 1 of the figure (mm_env): pick the median-vol day of 2026-03 by rule, write day_data.csv (hourly) and day_price.csv (5-min close).
Rule: among the days of 2026-03 with 24 ok hourly anchors, the day whose mean hourly realised vol (sqrt of the hour's RV) is the median (lower median)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN, view_cache as VC, evaluate as EV
base = AN.load_base("fwd"); ta = base.ta; ok = base.ok
y = base.A["T"]["A"] + base.A["LV0"]            # realised log RV of the hour [ta, ta+1h)
v = VC.read_view("M1_chronos2", "fwd")
mo = pd.Period("2026-03", "M")
rows = EV.run_A(base, ok, [mo], None, {"V3": None}, keep_preds=True)[0]
te, p = rows["te"], rows["preds"]["V3"]
df = pd.DataFrame({"real_logrv": y[te], "v3_logrv": p + base.A["LV0"][te]}, index=ta[te])
df = df.join(v[["ch_q10", "ch_q50", "ch_q90"]])
vol = lambda x: 1e4 * np.sqrt(np.exp(x))      # hourly vol in bps
for c in list(df.columns): df[c.replace("logrv", "vol_bps").replace("ch_", "ch_vol_")] = vol(df[c])
day = df.index.normalize(); cnt = df.groupby(day).size(); full = cnt[cnt == 24].index
dm = df.loc[day.isin(full)].groupby(df.index[day.isin(full)].normalize())["real_vol_bps"].mean().sort_values()
chosen = dm.index[(len(dm) - 1) // 2]; print("days", len(dm), "chosen", chosen.date(), "mean hourly vol bps", dm[chosen])
d = df[df.index.normalize() == chosen]; d.to_csv(os.path.join(HERE, "day_data.csv"))
m5 = AN.get_store("fwd").m5(); pr = m5.loc[chosen:chosen + pd.Timedelta(hours=24) - pd.Timedelta(minutes=5), "close"]
pr.rename("close").to_csv(os.path.join(HERE, "day_price.csv"))
# bias of Chronos's point forecast on the whole fwd+cut ok set (diagnostic for the 50/50 average)
for s in ("cut", "fwd"):
    b = AN.load_base(s); vv = VC.read_view("M1_chronos2", s).reindex(b.ta); yy = b.A["T"]["A"] + b.A["LV0"]; m = b.ok
    e = vv["ch_q50"].to_numpy()[m] - yy[m]; e0 = b.A["LV0"][m] - yy[m]
    print(s, "ch_q50 - target: mean %.3f sd %.3f rmse %.3f | persistence rmse %.3f | corr(ch_q50,target) %.3f corr(LV0,target) %.3f" % (e.mean(), e.std(), np.sqrt((e**2).mean()), np.sqrt((e0**2).mean()), np.corrcoef(vv["ch_q50"].to_numpy()[m], yy[m])[0,1], np.corrcoef(b.A["LV0"][m], yy[m])[0,1]))
