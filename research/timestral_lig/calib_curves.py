"""Accuracy of 'follow' trades by confidence (top-k by trailing rank of |z|) and horizon alignment, calibration vs Test (descriptive)."""
import json, os
import numpy as np, torch
import train, calibrate as C

KEEPS = (1.0, 0.3, 0.1, 0.05, 0.02)
out = {}
for ds in ("repo", "post"):
    X = train.Ctx(ds); D = X.D; lc = D.lc
    cal = np.unique(np.concatenate([D.train[::2], D.v1, D.v2])); v12 = np.unique(np.concatenate([D.v1, D.v2])); te = D.test
    sets = {"A0 zero-shot": ((C.model_signals(X.q0(cal)), cal), (C.model_signals(X.q0(te)), te)),
            "A2": ((C.model_signals(C.lig_predictions(X, "A2", v12)), v12), (C.model_signals(C.lig_predictions(X, "A2", te)), te)),
            "price-only (follow = momentum)": ((C.price_signals(lc, cal, D.sigma), cal), (C.price_signals(lc, te, D.sigma), te))}
    out[ds] = {}
    for name, ((sc, oc), (st, ot)) in sets.items():
        yc, yt = C.targets(lc, oc), C.targets(lc, ot)
        for H in C.HS:
            for align in ("none", "agree3"):
                if ((sc["dir"][H] != 0) & sc["align"][align]).sum() < 30: continue
                row = []
                for k in KEEPS:
                    r = []
                    for sig, org, y in ((sc, oc, yc), (st, ot, yt)):
                        pos, d = C.trade_index(sig, org, H, align, "z", k, "follow"); p = d * y[H][pos]; dd = p[p != 0]
                        r.append((round(float((dd > 0).mean()), 3) if len(dd) else None, int(len(dd))))
                    row.append(r)
                out[ds][f"{name} | H={H} | {align}"] = row
json.dump(out, open(os.path.join(train.WORK, "calibration_curves.json"), "w"), indent=1)
for ds, d in out.items():
    print(f"\n#### {ds}: accuracy of 'follow' trades, calibration -> Test (decided trades), by confidence keep fraction")
    print("| Model | H | Alignment | " + " | ".join(f"top {k:.0%}" for k in KEEPS) + " |")
    print("|---|---|---|" + "---|" * len(KEEPS))
    for key, row in d.items():
        name, H, al = key.split(" | ")
        cells = [f"{a[0]:.1%}→{b[0]:.1%} ({b[1]})" if a[0] is not None and b[0] is not None else "–" for a, b in row]
        print(f"| {name} | {H[2:]} | {al} | " + " | ".join(cells) + " |")
