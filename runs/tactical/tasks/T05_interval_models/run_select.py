"""T05 selection (SPEC.md): monthly walk-forward on the cut file, months 2019-01..2024-06, six models. Writes selection.json (to be committed
BEFORE run_lookA.py runs; that script asserts it) and tables_selection.txt. Usage: python run_select.py"""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t05_lib as T  # noqa: E402

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def main():
    t0 = time.time(); log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    s = T.load(T.CUT_CSV, T.SEL_LOAD); B = T.make_data(s); log(f"bars {len(s)} {s.index.min()} .. {s.index.max()}, anchors {len(B['D'].anch)}")
    df = T.walk_forward(B, T.SEL_MONTHS, T.MODELS, log=log)
    summ = T.summarise(df)
    qual = {m: bool(0.88 <= v["pooled_cover"] <= 0.92) for m, v in summ.items()}
    cands = [m for m in ("M2", "M3", "M4", "M5") if qual[m]]
    chosen = min(cands, key=lambda m: summ[m]["iscore_bps"]) if cands else None
    sel = {"spec": "runs/tactical/tasks/T05_interval_models/SPEC.md @7a02d55", "months": [T.SEL_MONTHS[0], T.SEL_MONTHS[-1]], "n_months": int(df.month.nunique()),
           "summary": summ, "qualifies_pooled_cover_088_092": qual, "candidates_excluding_M0_M1": cands, "chosen": chosen,
           "qr_stride": T.QR_STRIDE, "code_head_when_run": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "runtime_s": time.time() - t0}
    json.dump(sel, open(f"{HERE}/selection.json", "w"), indent=1)
    df.to_json(f"{HERE}/selection_rows.json", orient="records", indent=0)
    lines = ["== T05 selection: monthly walk-forward 2019-01..2024-06 (66 months), T=60, 90% intervals ==",
             "model  mean_iscore_bps  pooled_cover  monthly_cover_sd  mean_width_bps  qualifies"]
    for m in T.MODELS:
        v = summ[m]; lines.append(f"{m:5s} {v['iscore_bps']:14.3f} {v['pooled_cover']:12.4f} {v['cover_sd']:16.4f} {v['width_bps']:14.2f}  {qual[m]}")
    lines.append(f"chosen (lowest score among M2-M5 with pooled coverage in [0.88, 0.92]): {chosen}")
    p0 = df[df.model == "M0"].sort_values("month").iscore_bps.to_numpy()
    for m in T.MODELS[1:]:
        d = df[df.model == m].sort_values("month").iscore_bps.to_numpy() - p0
        mu, lo, hi, _ = T.L3.ci(d); lines.append(f"  {m} - M0 mean monthly iscore diff {mu:+.3f} [{lo:+.3f},{hi:+.3f}] bps, months lower {int((d < 0).sum())}/{len(d)} (descriptive)")
    open(f"{HERE}/tables_selection.txt", "w").write("\n".join(lines)); print("\n".join(lines))
    log("done")


if __name__ == "__main__":
    main()
