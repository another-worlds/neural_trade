"""O11 part c DIAGNOSTIC (descriptive; not part of the pre-registered test, no choice is made from it): the same walk-forward quantile regression with only
[ln sigma_hat] as the feature ("sig1"), to see whether the failure of the full candidate comes from the extra geometry features or from the method.
Scored on B and C like the main run; writes diag.jsonl and prints the comparison with the base and the candidate of results.jsonl."""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import o11_vol_lib as L  # noqa: E402
import common_run as R  # noqa: E402
from sklearn.linear_model import QuantileRegressor  # noqa: E402
O3 = L.O3; OUT = os.path.join(HERE, "diag.jsonl")


def sig1(F, tr, m, te):
    sub = tr[::L.QR_STRIDE]; x = (0.5 * O3.ridge_pred(m, L.P7.getX(F, sub, "L1")))[:, None]; mu, sd = x.mean(0), x.std(0) + 1e-12
    xt = ((0.5 * O3.ridge_pred(m, L.P7.getX(F, te, "L1")))[:, None] - mu) / sd; y = F[f"r{L.H_C}"][sub]
    q = {tau: QuantileRegressor(quantile=tau, alpha=L.QR_ALPHA, solver="highs").fit((x - mu) / sd, y) for tau in (L.Q_LO, L.Q_HI)}
    return q[L.Q_LO].predict(xt), q[L.Q_HI].predict(xt)


def main():
    R.nice(); Fc = L.load_cut(); Ff = L.load_fwd_frame(); F = L.combine(Fc, Ff); rows = []
    for span, months in (("B", L.B_MONTHS), ("C", L.C_MONTHS[1:])):
        for mo in months:
            m0 = L.month_bounds(mo)[0]; m0_ = L.TRAIN_END if span == "C" else m0
            tr = O3.train_rows(F["ta"], F["ok60"], 60, m0_); m = L.l1_fit(F, tr)
            te = O3.test_rows(F["ta"], F["ok60"], m0, L.month_bounds(mo)[1]); lo, hi = sig1(F, tr, m, te)
            rows.append({"span": span, "month": mo, "model": "sig1", **L.tail_row(F["r60"][te], lo, hi)})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    d = pd.DataFrame(rows); res = pd.DataFrame([json.loads(x) for x in open(os.path.join(HERE, "results.jsonl"), encoding="utf-8")])
    J = L.B_MONTHS + L.C_MONTHS[1:]
    for nm, df_ in (("sig1", d), ("cand", res)):
        x = df_[(df_.model == nm) & df_.month.isin(J)].set_index("month"); b = res[(res.model == "base") & res.month.isin(J)].set_index("month")
        s = L.series_stats((x.pin_bps - b.pin_bps).reindex(J).values)
        print(f"{nm}-base pinball J (24 months): {L.fmt(s, 3)}; pooled hit {x.n_lo.sum() / x.n.sum():.4f} / {x.n_hi.sum() / x.n.sum():.4f}")


if __name__ == "__main__":
    main()
