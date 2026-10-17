"""T14 harness proof: two dummy views through the real path (cache -> evaluate).

  D0_lag   last hour's log RV from `bars_before` (61 one-minute bars) and from the cached 5-minute RV series, and `point_log_rv` = that value
           (a persistence forecast). V3 already contains it: the increment must be about 0.
  D1_leak  next hour's log RV and the next hour's max |move| in bps, computed from bars AFTER the anchor (a deliberate leak): the increment
           must be huge on both targets, which shows the harness detects look-ahead.

    python run_dummy.py            # writes the views, the reference baselines and the two dummy results into core/dummy_results/
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors as AN  # noqa: E402
import evaluate as EV  # noqa: E402
import view_cache as VC  # noqa: E402
from _libs import T10  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dummy_results")


def lag_view(store, ta):
    """last hour's log RV (T01's scale) three ways: from the last 61 one-minute bars, from the cached 5-minute series, as the point forecast."""
    lag = np.full(len(ta), np.nan); lag5 = np.full(len(ta), np.nan)
    rv5 = store.rv5()
    for i, t in enumerate(ta):
        b = store.bars_before(t, 61, "1min")
        if len(b) == 61:
            lr = np.diff(np.log(b["close"].to_numpy())); lag[i] = np.log(np.sum(lr ** 2) + AN.EPS_RV)
    # 5-minute path: the hour that ends with the last complete bin before ta
    end = rv5.index.searchsorted(ta - pd.Timedelta(minutes=5), side="right") - 1
    ok = end >= 0
    lag5[ok] = rv5["logrv_1h"].to_numpy()[end[ok]]
    return pd.DataFrame({"lag_logrv": lag, "lag_logrv_5m": lag5, EV.VC.POINT_COL: lag}, index=pd.DatetimeIndex(ta, name="ta"))


def leak_view(store, ta):
    """next hour's log RV and max |move| in bps from the bars at and after ta (look-ahead on purpose)."""
    m1 = store.m1; c = m1["close"].to_numpy(); lr = np.concatenate([[0.0], np.diff(np.log(c))]); cs2 = np.concatenate([[0.0], np.cumsum(lr ** 2)])
    pos = m1.index.searchsorted(ta, side="left"); n = len(m1); good = (pos >= 1) & (pos + 60 < n)
    nxt = np.full(len(ta), np.nan); mv = np.full(len(ta), np.nan)
    p = pos[good]; nxt[good] = np.log(cs2[p + 60] - cs2[p] + AN.EPS_RV)
    mv[good] = 1e4 * T10.hit_labels(m1["high"].to_numpy(), m1["low"].to_numpy(), c, p, 60)
    return pd.DataFrame({"leak_next_logrv": nxt, "leak_move_bps": mv}, index=pd.DatetimeIndex(ta, name="ta"))


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); checks = {}
    for src in ("cut", "fwd"):
        anc = AN.load_anchors(src); store = AN.get_store(src); ta = anc.index
        d0 = lag_view(store, ta); VC.write_view("D0_lag", src, d0, note="last hour's log RV (bars_before and the 5-minute series); expected increment ~0")
        d1 = leak_view(store, ta); VC.write_view("D1_leak", src, d1, note="deliberate look-ahead: next hour's log RV and max |move| bps; expected huge increment")
        base = AN.load_base(src); okm = base.ok & d0["lag_logrv"].notna().to_numpy()
        checks[src] = {"lag_vs_LV0_max_abs": float(np.max(np.abs(d0["lag_logrv"].to_numpy()[okm] - base.A["LV0"][okm]))),
                       "lag5m_vs_LV0_max_abs": float(np.nanmax(np.abs(d0["lag_logrv_5m"].to_numpy()[okm] - base.A["LV0"][okm]))),
                       "leak_vs_target_max_abs": float(np.nanmax(np.abs(d1["leak_next_logrv"].to_numpy()[okm] - (base.A["T"]["A"][okm] + base.A["LV0"][okm])))),
                       "leak_move_vs_T10_label_max_abs_bps": float(np.nanmax(np.abs(d1["leak_move_bps"].to_numpy()[okm] - 1e4 * base.B["move"][okm])))}
    print("checks", json.dumps(checks, indent=1)); json.dump(checks, open(os.path.join(OUT, "dummy_checks.json"), "w"), indent=1)
    EV.run_baselines(os.path.join(HERE, "reference"))
    out = {}
    for nm in ("D0_lag", "D1_leak"):
        r = EV.evaluate_view(nm, out_dir=os.path.join(OUT, nm))
        out[nm] = {lk: {"A_inc": r["looks"][lk]["A"]["increment"]["view"]["mean"], "B_rel": r["looks"][lk]["B"]["rel_improvement"]["view"]["mean"]} for lk in r["looks"]}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
