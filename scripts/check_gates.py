"""Judge the remediation plan's M1-M4 stop/go gates, clause by clause, as the plan words them.

    python scripts/check_gates.py [runs/gates]

Expects the run directories written by scripts/gate_run.py:
    m1a  2 epochs, physics lambdas at 0      (M1 run (a))
    m1b  2 epochs, defaults                  (M1 run (b))
    m2   5 epochs, defaults                  (M2)
    m3   20 epochs, defaults                 (M3 and M4, first attempt - kept as history)
    m4   20 epochs, defaults, after the normalised-conformal fix (M3 and M4, second attempt)
    m5   20 epochs, defaults, after the BCE direction loss + direction skip (M3 and M4, current)
The newest finished attempt is judged; earlier ones are printed as history (M3@m3, ...).
Missing runs are reported as NOT RUN, never as passes. Exit 0 only when every clause of every
milestone whose run exists passes, and no clause is PENDING.

Each clause quotes its source in the plan (C:/Users/Step/.claude/plans/write-the-plan-to-rosy-stonebraker.md,
sections A1-A4). An earlier version of this script checked weaker proxies (e.g. "PIT-KS is
finite" instead of "val_pit_ks_h1 < 0.2") and read the wrong CSV by default; it overstated
M2 as passed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/gates")
PHYSICS_TRAIN = ("t_perp_loss", "casimir_loss", "hd_loss", "ife_loss", "vac_overflow_loss")
PERIOD_PREFIXES = ("ma_period_", "rsi_period_", "bb_period_", "macd_")

results = []  # (milestone, clause, status, detail)


def record(ms, clause, status, detail=""):
    results.append((ms, clause, status, detail))
    print(f"  {status:8s} {clause}{('  -- ' + detail) if detail else ''}")


def check(ms, clause, ok, detail=""):
    record(ms, clause, "PASS" if ok else "FAIL", detail)


def load(name):
    d = ROOT / name
    if not (d / "training_log.csv").exists():
        return None
    if not (d / "analytics.json").exists():  # still training: analytics are written at the end
        return {"dir": d, "in_progress": True}
    run = {"dir": d, "log": pd.read_csv(d / "training_log.csv")}
    ph = d / "indicator_params_history.csv"
    run["params"] = pd.read_csv(ph) if ph.exists() else None
    an = d / "analytics.json"
    run["an"] = json.loads(an.read_text()) if an.exists() else None
    return run


def fmt(v):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def m1(run, label, defaults):
    ms = f"M1{label}"
    log, params = run["log"], run["params"]
    print(f"\n{ms}  ({run['dir'].name}, {len(log)} epochs)  "
          f"plan A1: 'train_and_evaluate(force=True, epochs=2) twice - (a) physics lambdas overridden to 0, (b) defaults'")
    check(ms, "log_val_loss changes between epochs", log["val_loss"].round(9).nunique() > 1,
          f"val_loss={list(np.round(log['val_loss'], 5))}")
    check(ms, "log_nonfinite_grad_steps == 0", float(log["nonfinite_grad_steps"].max()) == 0.0,
          f"max={log['nonfinite_grad_steps'].max()}")
    if params is not None:
        cols = [c for c in params.columns if c.startswith(PERIOD_PREFIXES)]
        check(ms, "18 learned periods finite", len(cols) == 18 and bool(np.isfinite(params[cols].values).all()),
              f"{len(cols)} period columns")
        moved = int((params[cols].iloc[-1] != params[cols].iloc[0]).sum()) if len(params) > 1 else 0
        check(ms, ">= 12 periods change between epoch 0 and 1 (S4)", moved >= 12, f"{moved} moved")
    else:
        record(ms, "learned periods", "FAIL", "indicator_params_history.csv missing")
    for h in ("h0", "h1", "h2"):
        tr, va = log[f"extended_{h}"], log[f"val_extended_{h}"]
        ok = bool(((tr > 0) & (tr < 3)).all()) and bool((np.abs(tr - 1.333295) > 1e-4).all()) \
            and bool((np.abs(tr.values - va.values) > 1e-9).any())
        check(ms, f"log_extended_{h} in (0,3), != 1.333295, train != val (S2)", ok,
              f"train={list(np.round(tr, 4))} val={list(np.round(va, 4))}")
    up = log["train_pred_up_rate_h1"]
    check(ms, "log_train_pred_up_rate_h1 not in {0,1}", bool(((up > 0) & (up < 1)).all()),
          f"{list(np.round(up, 4))}")
    mcc = log["val_dir_mcc_h1"]
    # "GO when both show ... log_val_dir_mcc_h1 != 0.0 exactly": judged on the run's final epoch
    # (the model the run ends with); every epoch is printed so an early constant head is visible.
    check(ms, "log_val_dir_mcc_h1 != 0.0 exactly (final epoch)", float(mcc.iloc[-1]) != 0.0,
          f"per epoch {list(np.round(mcc, 4))}")
    if defaults:
        cas = log["casimir_loss"]
        check(ms, "(b) log_casimir_loss > 0 (price heads non-zero)", bool((cas > 0).all()),
              f"{list(np.round(cas, 6))}")


def m2(run):
    ms = "M2"
    log, an = run["log"], run["an"]
    print(f"\nM2  ({run['dir'].name}, {len(log)} epochs)  plan A2: 'M2 stop/go (5 epochs)'")
    check(ms, "run is 5 epochs", len(log) >= 5, f"{len(log)} epochs")
    phys = log[list(PHYSICS_TRAIN)]
    check(ms, "physics columns finite", bool(np.isfinite(phys.values).all()))
    check(ms, "physics columns >= 0", bool((phys.values >= -1e-7).all()))
    const = [c for c in PHYSICS_TRAIN if log[c].nunique() <= 1]
    check(ms, "physics columns non-constant", not const, f"constant: {const}" if const else "")
    check(ms, "log_val_vac_overflow_loss == 0", bool(np.allclose(log["val_vac_overflow_loss"], 0.0, atol=1e-8)))
    for col in ("val_nll_loss", "val_crps_loss"):
        if col in log:
            v = log[col]
            check(ms, f"{col} improves over epoch 0 (last < first)", float(v.iloc[-1]) < float(v.iloc[0]),
                  f"{list(np.round(v, 4))}")
        else:
            record(ms, f"{col} improves over epoch 0", "FAIL", "column absent")
    pk = log["val_pit_ks_h1"]
    check(ms, "log_val_pit_ks_h1 < 0.2 (last epoch)", float(pk.iloc[-1]) < 0.2, f"{list(np.round(pk, 4))}")
    if an is None:
        record(ms, "analytics (corr var/err^2, var dispersion)", "FAIL", "analytics.json missing")
        return
    h1 = an["horizons"]["h1"]
    check(ms, "corr(var_scaled, error_scaled^2) h1 > 0.10", h1["corr_var_err2_scaled"] > 0.10,
          fmt(h1["corr_var_err2_scaled"]) + "  (HEAD 0.032)")
    check(ms, "std(var_h1)/mean(var_h1) > 0.05", h1["var_dispersion"] > 0.05, fmt(h1["var_dispersion"]))


def m3(run, ms="M3"):
    log, an = run["log"], run["an"]
    print(f"\n{ms}  ({run['dir'].name}, {len(log)} epochs run)  plan A3: 'M3 stop/go (20 epochs)'")
    if an is None:
        record(ms, "analytics", "FAIL", "analytics.json missing")
        return
    check(ms, "run requested 20 epochs", an["epochs_requested"] >= 20, f"{an['epochs_requested']} requested")
    h1 = an["horizons"]["h1"]
    check(ms, "EV(delta) h1 > 0", h1["ev_delta"] > 0, fmt(h1["ev_delta"]) + "  (HEAD 0.0000)")
    check(ms, "ROC-AUC h1 > 0.52 (all test rows)", h1["roc_auc"] > 0.52,
          f"{fmt(h1['roc_auc'])} on {h1['n_test']} rows; masked {fmt(h1['roc_auc_masked'])}")
    trades = an.get("total_trades")
    if trades is None:
        record(ms, "Total Trades > 0", "PENDING", an.get("total_trades_note", "not computed"))
    else:
        check(ms, "Total Trades > 0", trades > 0, str(trades))
    best = an["best_epoch"]
    mcc = float(log["val_dir_mcc_h1"].iloc[best])
    gmcc = float(log["val_gauss_dir_mcc_h1"].iloc[best])
    check(ms, "best-epoch log_val_dir_mcc_h1 > 0.02", mcc > 0.02, f"{mcc:.4f} at epoch {best}")
    check(ms, "best-epoch log_val_gauss_dir_mcc_h1 > 0", gmcc > 0, f"{gmcc:.4f} at epoch {best}")


def m4(run, ms="M4"):
    an = run["an"]
    print(f"\n{ms}  ({run['dir'].name})  plan A4: 'test coverage at alpha=0.1 in [0.87, 0.93] all horizons; "
          "early stopping fires on a plateaued run; bare pytest green'")
    if an is None or not an.get("calibration_fitted"):
        record(ms, "conformal coverage on test", "FAIL", "calibration pipeline not fitted")
    else:
        for h in ("h0", "h1", "h2"):
            cov = an["horizons"][h]["coverage90"]
            check(ms, f"test coverage@90 {h} in [0.87, 0.93]", cov is not None and 0.87 <= cov <= 0.93, fmt(cov))
    record(ms, "early stopping fires on a plateaued run", "INFO",
           f"this run: {an['epochs_run']}/{an['epochs_requested']} epochs, early_stopped={an['early_stopped']}; "
           "the plateau case is pinned by tests/test_train_smoke.py::test_early_stopping_fires_on_a_plateau")
    record(ms, "bare pytest green", "INFO", "judged by the pytest run, not by this script")


def main():
    print(f"gate runs under: {ROOT.resolve()}")
    attempts = ("m3", "m4", "m5")          # successive M3/M4 attempts; the newest finished one is judged
    runs = {n: load(n) for n in ("m1a", "m1b", "m2", *attempts)}
    finished = [a for a in attempts if runs[a] is not None and not runs[a].get("in_progress")]
    current = finished[-1] if finished else "m3"
    plan = [("m1a", lambda r: m1(r, "a", False)), ("m1b", lambda r: m1(r, "b", True)), ("m2", m2)]
    for a in attempts:
        if a == current:
            plan.append((a, lambda r: (m3(r), m4(r))))
        elif runs[a] is not None:
            plan.append((a, lambda r, a=a: (m3(r, f"M3@{a}"), m4(r, f"M4@{a}"))))
    for name, fn in plan:
        if runs[name] is None:
            print(f"\n{name}: NOT RUN")
            results.append((name, "run exists", "NOT RUN", ""))
        elif runs[name].get("in_progress"):
            print(f"\n{name}: IN PROGRESS (no analytics.json yet)")
            results.append((name, "run finished", "NOT RUN", "in progress"))
        else:
            fn(runs[name])

    print("\n" + "=" * 78)
    by_ms = {}
    for ms, _clause, status, _ in results:
        by_ms.setdefault(ms, []).append(status)
    for ms, st in by_ms.items():
        n_fail = st.count("FAIL")
        n_pend = st.count("PENDING") + st.count("NOT RUN")
        verdict = "PASS" if n_fail == 0 and n_pend == 0 else ("FAIL" if n_fail else "INCOMPLETE")
        print(f"  {ms:5s} {verdict:10s} {st.count('PASS')} pass, {n_fail} fail, {n_pend} pending/not run")
    bad = [r for r in results if r[2] in ("FAIL", "PENDING", "NOT RUN") and "@" not in r[0]]
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
