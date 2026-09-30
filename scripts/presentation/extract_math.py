"""Data for the maths presentation (presentations/4_math_report.html): training dynamics of the six 360-day models,
their learned indicator periods per epoch, and every level-1 screen trial (health and the knobs it varied).

    python scripts/presentation/extract_math.py      # -> docs/presentation/data_math.json

Reads committed run files only; dev folds only (D-020). Analytic curves (gradients of each loss term, the Adam bound,
the AUC -> achievable BCE relation, the EWMA kernel) are computed in the page itself. Numbers quoted from the research
notes live in docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md).
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAB = REPO / "runs/scenarios/long_360d_stab"
OUT = REPO / "docs/presentation/data_math.json"
KNOBS = ("LR", "GRAD_CLIP_NORM", "ADAM_BETA1", "ADAM_BETA2", "INDICATOR_LR_MULT", "LAMBDA_SOFT_ECE",
         "LAMBDA_VAR", "LAMBDA_NLL_OUTER", "LAMBDA_DIR_OUTER", "LAMBDA_CRPS", "LAMBDA_POINT", "LAMBDA_COHERENCE",
         "LAMBDA_PNL", "DIRECTION_LOSS", "LAMBDA_VOL", "LAMBDA_T_PERP", "LAMBDA_HD")


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if v == v else None


def runs() -> list:
    out = []
    for d in sorted(Path(p) for p in glob.glob(str(STAB / "2026*"))):
        tl = list(csv.DictReader(open(d / "training_log.csv", encoding="utf-8")))
        ih = list(csv.DictReader(open(d / "indicator_params_history.csv", encoding="utf-8")))
        st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        init = json.loads((d / "period_init.json").read_text(encoding="utf-8"))["periods"]
        names = [k for k in ih[0] if k.startswith(("ma_", "macd_", "rsi_", "bb_"))]
        out.append({
            "name": d.name, "fold": d.name.split("__")[1], "seed": d.name.split("__")[2],
            "served_epoch": st.get("weights_epoch"), "sec_per_step": st.get("sec_per_step"),
            "epoch": [int(float(r["epoch"])) + 1 for r in tl],
            "grad_norm": [num(r["grad_global_norm"]) for r in tl], "loss": [num(r["loss"]) for r in tl],
            "val_loss": [num(r["val_loss"]) for r in tl], "lr": [num(r["lr_used"]) for r in tl],
            "lr_indicator": [num(r.get("log_lr_indicator_used")) for r in ih],
            "nonfinite": [num(r["nonfinite_grad_steps"]) for r in tl],
            "val_dir_loss": {h: [num(r.get(f"val_dir_loss_{h}")) for r in tl] for h in ("h0", "h1", "h2")},
            "periods": {k: [num(r[k]) for r in ih] for k in names}, "period_init": {k: num(init.get(k)) for k in names}})
    return out


def screens() -> list:
    rows = []
    for d in sorted(glob.glob(str(REPO / "runs/screens/l1_*"))):
        block = Path(d).name.replace("l1_", "")
        for f in sorted(glob.glob(str(Path(d) / "results.shard-*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                o = json.loads(line)
                h = o.get("health") or {}
                shares = h.get("loss_term_shares") or {}
                top = max(shares, key=shares.get) if shares else None
                cd = o.get("config_diff") or {}
                rows.append({"block": block, "passed": bool(o.get("passed")), "seed": o.get("seed"),
                             "reason": (str((o.get("reasons") or [""])[0]).split(" ")[0] or None),
                             "clipped": num(h.get("clipped_share")), "gmean": num(h.get("grad_global_norm_mean")),
                             "gmax": num(h.get("grad_global_norm_max")), "nonfinite": h.get("nonfinite_grad_steps"),
                             "drop": num(h.get("train_loss_drop")), "top_term": top,
                             "top_share": num(shares.get(top)) if top else None,
                             "knobs": {k: (cd[k] if isinstance(cd.get(k), str) else num(cd.get(k))) for k in KNOBS if k in cd}})
    return rows


def main() -> None:
    data = {"runs": runs(), "screens": screens()}
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    n = len(data["screens"])
    print("wrote", OUT, round(OUT.stat().st_size / 1e6, 2), "MB;", len(data["runs"]), "runs,", n, "screen trials,",
          sum(r["passed"] for r in data["screens"]), "passed")


if __name__ == "__main__":
    main()
