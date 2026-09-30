"""What did the 360-day models compute? Surrogate analysis of their outputs (CPU, stored predictions only).

    CUDA_VISIBLE_DEVICES=-1 python runs/experiments/micro_loop_v1/interpret_c1.py   # writes interpret_c1.json

For each of the six long_360d_stab models (dev folds -3 / -2 x seeds 0-2; the leader C1 is f-2 s0), the model's outputs
on its calibration and dev blocks are explained by simple features of the close series it sees (the model's input is
the last 60 closes): trailing log returns over 1-60 bars, realised volatility, distance to moving averages, RSI, the
position in the 60-bar range. Explained outputs: the traded signal (confidence-weighted P(up)), calibrated P(up) per
horizon, and the log predicted sigma (h1). Surrogates are fitted on the calibration block and scored on the dev block
(R^2 out of sample): a linear model on standardised features, a depth-3 regression tree (readable rules) and gradient
boosting (an upper bound on what these features explain). Partial dependence on the dev block puts the model's mean
output next to the realised forward return per feature decile, so a learned pattern can be checked against the data.
Nothing here chooses anything; fold -1 is not read (D-020).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text

import neural_trade  # noqa: F401
from neural_trade.experiments.scorer import BlockSignals, load_block

ROOT = Path(__file__).resolve().parents[3]
STAB = ROOT / "runs/scenarios/long_360d_stab"
LEADER = "20260930T094257Z-dce15ed-e3669618-default__f-2__s0"
OUT = Path(__file__).with_name("interpret_c1.json")
WARM = 240  # bars dropped at each block start (features need history)


def features(close: np.ndarray) -> pd.DataFrame:
    c = pd.Series(np.log(close))
    r1 = c.diff()
    f = {}
    for k in (1, 3, 5, 10, 15, 30, 60):
        f[f"ret_{k}"] = 1e4 * (c - c.shift(k))
    f["vol_15"] = 1e4 * r1.rolling(15).std()
    f["vol_60"] = 1e4 * r1.rolling(60).std()
    f["vol_ratio"] = r1.rolling(15).std() / r1.rolling(240).std()
    px = pd.Series(close)
    for k in (5, 10, 30, 60):
        f[f"dist_sma{k}"] = 1e4 * (px / px.rolling(k).mean() - 1)
    d = px.diff()
    up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    hi, lo = px.rolling(60).max(), px.rolling(60).min()
    f["range_pos_60"] = (px - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame(f)


def block(d: Path, name: str):
    frame, bars, extra = load_block(d / name)
    return frame, np.asarray(bars.close, dtype=float), extra


def targets(sig_frame) -> dict:
    return {"signal": np.asarray(sig_frame.weighted_direction, dtype=float),
            "p_h0": np.asarray(sig_frame.p[:, 0], dtype=float), "p_h1": np.asarray(sig_frame.p[:, 1], dtype=float),
            "p_h2": np.asarray(sig_frame.p[:, 2], dtype=float),
            "log_sigma_h1": np.log(np.asarray(sig_frame.sigma_ret[:, 1], dtype=float))}


def r2(y, yhat) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def analyse(d: Path, detail: bool) -> dict:
    cal, cclose, _ = block(d, "predictions_cal.npz")
    oos, oclose, extra = block(d, "predictions_oos.npz")
    sig = BlockSignals.build(cal, oos)
    Fc, Fo = features(cclose), features(oclose)
    Tc, To = targets(sig.cal), targets(sig.oos)
    ok_c = Fc.notna().all(axis=1).to_numpy() & (np.arange(len(Fc)) >= WARM)
    ok_o = Fo.notna().all(axis=1).to_numpy() & (np.arange(len(Fo)) >= WARM)
    for T, ok in ((Tc, ok_c), (To, ok_o)):
        for v in T.values():
            ok &= np.isfinite(v)
    Xc, Xo = Fc[ok_c], Fo[ok_o]
    mu, sd = Xc.mean(), Xc.std()
    Zc, Zo = (Xc - mu) / sd, (Xo - mu) / sd
    out = {"cell": d.name, "n_cal": int(ok_c.sum()), "n_dev": int(ok_o.sum()), "surrogates": {}}
    for key in Tc:
        yc, yo = Tc[key][ok_c], To[key][ok_o]
        lin = LinearRegression().fit(Zc, yc)
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500).fit(Xc, yc)
        gb = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.1).fit(Xc, yc)
        s = {"r2_linear": round(r2(yo, lin.predict(Zo)), 4), "r2_tree3": round(r2(yo, tree.predict(Xo)), 4),
             "r2_boosting": round(r2(yo, gb.predict(Xo)), 4), "sd_dev": float(np.std(yo)),
             "coef_std": {k: round(float(v) * 1e3 if key != "log_sigma_h1" else float(v), 4) for k, v in zip(Xc.columns, lin.coef_)},
             "corr_dev": {k: round(float(np.corrcoef(Xo[k], yo)[0, 1]), 4) for k in Xo.columns}}
        if detail:
            s["tree_rules"] = export_text(tree, feature_names=list(Xc.columns), decimals=2)
        out["surrogates"][key] = s
    if detail:
        # partial dependence on the dev block: the model's mean output and the realised forward return per decile
        y = np.asarray(oos.y, dtype=float)[ok_o]
        close_o = oclose[ok_o]
        fwd = {h: 1e4 * y[:, i] / close_o for i, h in enumerate(("h0", "h1", "h2"))}
        sigv, pv = To["signal"][ok_o], To["log_sigma_h1"][ok_o]
        pd_out = {}
        for k in Xo.columns:
            x = Xo[k].to_numpy()
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
            j = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
            rows = []
            for b in range(len(edges) - 1):
                m = j == b
                if m.sum() < 50:
                    continue
                rows.append({"x": round(float(np.median(x[m])), 3), "n": int(m.sum()),
                             "signal": round(float(sigv[m].mean()), 5),
                             "sigma_bps": round(float(np.exp(pv[m]).mean() * 1e4), 3),
                             "fwd_h0": round(float(fwd["h0"][m].mean()), 3), "fwd_h1": round(float(fwd["h1"][m].mean()), 3),
                             "fwd_h2": round(float(fwd["h2"][m].mean()), 3),
                             "up_h1": round(float((fwd["h1"][m] > 0).mean()), 4),
                             "absmove_h1": round(float(np.abs(fwd["h1"][m]).mean()), 3)})
            pd_out[k] = rows
        ts = np.asarray(extra["anchor_timestamp"]).astype(str)[ok_o]
        hours = np.array([int(t[11:13]) for t in ts])
        pd_out["hour_utc"] = [{"x": h, "n": int((hours == h).sum()), "signal": round(float(sigv[hours == h].mean()), 5),
                               "sigma_bps": round(float(np.exp(pv[hours == h]).mean() * 1e4), 3),
                               "absmove_h1": round(float(np.abs(fwd["h1"][hours == h]).mean()), 3),
                               "fwd_h1": round(float(fwd["h1"][hours == h].mean()), 3)} for h in range(24)]
        out["partial_dependence"] = pd_out
        # how the three horizons' calibrated P(up) relate, and how the signal relates to each
        P = np.column_stack([To["p_h0"][ok_o], To["p_h1"][ok_o], To["p_h2"][ok_o], sigv])
        out["p_corr"] = np.corrcoef(P.T).round(3).tolist()
    return out


def main() -> None:
    cells = [analyse(Path(p), Path(p).name == LEADER) for p in sorted(glob.glob(str(STAB / "2026*")))]
    # consistency: correlation of the standardised signal coefficients across the six models
    names = list(cells[0]["surrogates"]["signal"]["coef_std"].keys())
    C = np.array([[c["surrogates"]["signal"]["coef_std"][k] for k in names] for c in cells])
    S = np.array([[c["surrogates"]["log_sigma_h1"]["coef_std"][k] for k in names] for c in cells])
    res = {"features": names, "cells": cells, "coef_corr_signal": np.corrcoef(C).round(3).tolist(),
           "coef_corr_sigma": np.corrcoef(S).round(3).tolist()}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for c in cells:
        s, v = c["surrogates"]["signal"], c["surrogates"]["log_sigma_h1"]
        top = sorted(s["coef_std"].items(), key=lambda kv: -abs(kv[1]))[:4]
        topv = sorted(v["coef_std"].items(), key=lambda kv: -abs(kv[1]))[:3]
        print(c["cell"][-12:], "signal R2 lin/tree/gb", s["r2_linear"], s["r2_tree3"], s["r2_boosting"], "top", top)
        print("   sigma R2", v["r2_linear"], v["r2_tree3"], v["r2_boosting"], "top", topv)
    print("signal coef corr across models\n", np.array(res["coef_corr_signal"]))
    print("sigma coef corr across models\n", np.array(res["coef_corr_sigma"]))
    lead = next(c for c in cells if c["cell"] == LEADER)
    print(lead["surrogates"]["signal"]["tree_rules"])
    print("p corr", lead["p_corr"])


if __name__ == "__main__":
    main()
