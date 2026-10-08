"""Round hc7 (owner, 2026-10-08): the decisive indicator questions and the lead's hypotheses A and B, long block, 6 climb
slices x 2 seeds, all 9 heads, fixed before any run. Base for direction/confidence: hc5_noprice3 (no price, 3 horizons,
all 14 families learned); for price: cand_c2_base (with price).
  D  (owner)  none         no indicator family at all                       -> are indicators needed at all?
  D  (owner)  frozen       all 14 frozen at textbook values (LR/grad x1e-9, no per-window shift) -> is learning the periods needed?
  A  (lead)   thin_learned MODEL_NAME linear_indicators: pooled indicators -> LayerNorm -> linear heads (the readout cannot
                           absorb what the indicators miss), periods learned
  A  (lead)   thin_frozen  the same, frozen at textbook values           -> with a thin readout, do learned periods beat textbook?
  B  (lead)   fn_dir       a direction-only network: RSI + MACD (trend), only the direction loss (NLL, CRPS, physics off)
  B  (lead)   fn_conf      a confidence-only network: ATR, Bollinger, Keltner, Donchian (volatility), NLL + CRPS + t_perp,
                           direction loss off
  B  (lead)   fn_price     a price-only network: MA, price head on, point loss only (direction and variance losses off)
Each B network is judged only on its own function (fn_dir on direction, fn_conf on confidence, fn_price on the price head).
usage: python make_hc7.py -> configs/tactical/hc7_*.yaml, runs/tactical/hc7/tasks.txt"""
import os, subprocess, sys
import yaml
os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/ind")
import make_ind as I

PY = sys.executable
empty = {k: [] for k in I.FAMS_CLOSE}
FROZEN = {"INDICATOR_LR_MULT": 1e-9, "INDICATOR_GRAD_MULT": 1e-9, "ADAPTIVE_INDICATORS": False}
PHYS0 = {"LAMBDA_T_PERP": 0.0, "LAMBDA_CASIMIR": 0.0, "LAMBDA_HD": 0.0, "LAMBDA_IFE": 0.0, "LAMBDA_VAC_OVERFLOW": 0.0}
fam = lambda **kw: {**empty, **kw}
V = {
    "none": {"PRICE_HEAD": "none", "INDICATOR_FAMILIES": dict(empty)},
    "frozen": {"PRICE_HEAD": "none", **FROZEN},
    "thin_learned": {"PRICE_HEAD": "none", "MODEL_NAME": "linear_indicators"},
    "thin_frozen": {"PRICE_HEAD": "none", "MODEL_NAME": "linear_indicators", **FROZEN},
    "fn_dir": {"PRICE_HEAD": "none", "INDICATOR_FAMILIES": fam(rsi=I.FAMS_CLOSE["rsi"], macd=I.FAMS_CLOSE["macd"]),
               "LAMBDA_VAR": 0.0, "LAMBDA_CRPS": 0.0, **PHYS0},
    "fn_conf": {"PRICE_HEAD": "none", "INDICATOR_FAMILIES": fam(bb=I.FAMS_CLOSE["bb"], atr=I.DEFAULT_OHLCV["atr"],
               keltner=I.DEFAULT_OHLCV["keltner"], donchian=I.DEFAULT_OHLCV["donchian"]), "LAMBDA_DIR": 0.0,
               "LAMBDA_CASIMIR": 0.0, "LAMBDA_HD": 0.0, "LAMBDA_IFE": 0.0},
    "fn_price": {"INDICATOR_FAMILIES": fam(ma=I.FAMS_CLOSE["ma"]), "LAMBDA_DIR": 0.0, "LAMBDA_VAR": 0.0, "LAMBDA_CRPS": 0.0,
                 "LAMBDA_EXTENDED_TREND": 0.0, "LAMBDA_COHERENCE": 0.0, **PHYS0},
}
names = []
for n, o in V.items():
    subprocess.run([PY, "runs/tactical/hc4/make_hc4.py", "tmp"], check=True, capture_output=True)
    s = yaml.safe_load(open("configs/tactical/hc4_tmp.yaml"))
    s["name"] = f"hc7_{n}"; s["overrides"].update(o)
    yaml.safe_dump(s, open(f"configs/tactical/hc7_{n}.yaml", "w"), sort_keys=False)
    names.append(f"hc7_{n}")
os.remove("configs/tactical/hc4_tmp.yaml")
open("runs/tactical/hc7/tasks.txt", "w").write("".join(f"{n} {i}/3\n" for n in names for i in range(3)))
print(" ".join(names))
