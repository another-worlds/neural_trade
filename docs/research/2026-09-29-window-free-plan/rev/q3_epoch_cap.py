"""Should-fix 1: an epoch cap fixed from the dev runs' served epochs, and how often it would cap a judged run.

Measured (DEV fold -2 only): the served (best val_loss) epoch and whether early stopping fired, for the 42 v1 runs
(EPOCHS 20, EARLY 6). A capped run's served epoch is right-censored (the true one is >= the observed).
Rule under test: EPOCHS = ceil(k x max over the D dev runs of their served epoch) + EARLY; a judged run is
capped iff served_j + EARLY > EPOCHS, i.e. served_j > k x max(served_dev) (up to rounding).
The v1 curves cannot show it beyond 20 epochs, so the rate is modelled: ln(served) ~ N(mu, sigma) per run, plus
a fold-level shift ~ N(0, sigma_f) shared by the runs of one fold (dev and judged folds differ), sigma from
(a) a censored-normal MLE on the 42 runs (seed effect included), (b) 0.40 (the measured within-condition SD of
ln reach epoch, q3_probe_target), (c) 0.60; sigma_f in {0, 0.3} (Estimate). P(capped) does not depend on mu.
Scaling to 7-day blocks: fold -2 has 90 steps per epoch, a 7-day block 40; step-matched x 90/40 = 2.25,
epoch-matched x 1 (which one holds is what the probe measures).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q3_epoch_cap.py
Writes q3_epoch_cap.json."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import optimize, stats

from common import CAP_V1, EARLY, OUT, STEPS_7DAY, STEPS_V1_FOLD_M2, load_dev_runs

recs = load_dev_runs()


def censored_mle(x, cens):
    x = np.log(np.asarray(x, float)); cens = np.asarray(cens, bool)

    def nll(p):
        mu, ls = p; s = math.exp(ls)
        return -(stats.norm.logpdf(x[~cens], mu, s).sum() + stats.norm.logsf(x[cens], mu, s).sum())
    r = optimize.minimize(nll, [np.log(18), np.log(0.4)], method="Nelder-Mead")
    return float(r.x[0]), float(math.exp(r.x[1]))


def main():
    served = np.array([r["served"] for r in recs]); cens = np.array([not r["early_stopped"] for r in recs])
    mu_hat, s_hat = censored_mle(served, cens)
    out = {"source": "42 v1 runs, DEV fold -2", "measured": {
        "capped_runs(no early stop within 20)": int(cens.sum()), "runs": len(recs),
        "served_epoch_early_stopped_runs": sorted(served[~cens].tolist()),
        "served_epoch_capped_runs_quantiles": {q: float(np.quantile(served[cens], q)) for q in (0.1, 0.5, 0.9)},
        "censored_mle_ln_served": {"mu": mu_hat, "sigma": s_hat, "median_epochs": math.exp(mu_hat)},
        "served_in_updates_v1(90/epoch)": {"median_capped_lower_bound": float(np.median(served[cens]) * STEPS_V1_FOLD_M2)},
        "7day_equivalent_epochs": {"epoch_matched": [int(served.min()), CAP_V1], "step_matched_x2.25": [round(served.min() * STEPS_V1_FOLD_M2 / STEPS_7DAY, 1), round(CAP_V1 * STEPS_V1_FOLD_M2 / STEPS_7DAY, 1)]},
        "rule_cap_if_dev_like_v1(k=2, D=3 capped dev runs, served 17-20)": [2 * 17 + EARLY, 2 * 20 + EARLY],
    }}
    rng = np.random.default_rng(3)
    REPS = 200000
    mc = {}
    for sname, sig in (("mle", s_hat), ("0.40", 0.40), ("0.60", 0.60)):
        for sig_f in (0.0, 0.3):
            for D in (3, 6):
                for k in (1.5, 2.0, 3.0):
                    dev = rng.normal(0, sig, (REPS, D)) + rng.normal(0, sig_f, (REPS, 1))
                    jud = rng.normal(0, sig, REPS) + rng.normal(0, sig_f, REPS)
                    p_run = float(np.mean(jud > math.log(k) + dev.max(1)))
                    # share of studies where more than 25% of 10 judged runs of one arm are capped (fold shifts differ per fold)
                    J = rng.normal(0, sig, (20000, 10)) + rng.normal(0, sig_f, (20000, 10))
                    devs = rng.normal(0, sig, (20000, D)) + rng.normal(0, sig_f, (20000, 1))
                    capped = (J > math.log(k) + devs.max(1, keepdims=True)).sum(1)
                    mc[f"sigma={sname}({sig:.2f})|sigma_f={sig_f}|D={D}|k={k}"] = {
                        "P(judged run capped)": round(p_run, 4),
                        "P(>25% of 10 runs capped)": round(float(np.mean(capped > 2.5)), 4),
                        "P(any of 10 runs capped)": round(float(np.mean(capped > 0)), 4)}
    out["model"] = mc
    (OUT / "q3_epoch_cap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
