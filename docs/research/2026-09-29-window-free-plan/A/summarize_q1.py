"""Print compact tables from q1_results.json (for FINDINGS.md)."""
import json
import os

import numpy as np

R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q1_results.json")))
print("S =", R["target_scale_S_dollars"], "window_repro", R["window_repro"])
P = R["precision"]
for T in ("T30720", "T43008"):
    for alpha in ("constant_alpha", "per_bar_alpha"):
        for ch in ("d_ema_minus_close", "bb_var_ewma_d2", "rsi_gain_ewma"):
            print(f"\n{T} {alpha} {ch}: rel err (max|err|/max|state|) per period; max|state| from V1 C64")
            names = list(P[T]["C64"][alpha]["V1_segsum_hier_mulsum"][ch].keys())
            print(f"{'variant':34s}" + "".join(f"{n[:9]:>10s}" for n in names))
            ms = [P[T]["C64"][alpha]["V1_segsum_hier_mulsum"][ch][n]["max_abs_state"] for n in names]
            print(f"{'max|state|':34s}" + "".join(f"{m:10.3g}" for m in ms))
            for C in ("C64", "C128"):
                for v, e in P[T][C][alpha].items():
                    row = [e[ch][n]["rel_err"] for n in names]
                    flag = "".join("" if e[ch][n]["PASS"] else "" for n in names)
                    fails = sum(not e[ch][n]["PASS"] for n in names)
                    print(f"{(C + ' ' + v)[:34]:34s}" + "".join(f"{r:10.1e}" for r in row) + f"  fails={fails}")
        # rsi value
        e = P[T]["C64"][alpha]["V1_segsum_hier_mulsum"]["rsi_value_points_max_abs_err_after_1000"]
        print(f"{T} {alpha} RSI value abs err (points) V1 C64:", {k: f"{v:.1e}" for k, v in e.items()})
        e = P[T]["C64"][alpha]["V1e_tf32_rn_emulated"]["rsi_value_points_max_abs_err_after_1000"]
        print(f"{T} {alpha} RSI value abs err (points) TF32rn C64:", {k: f"{v:.1e}" for k, v in e.items()})
        for v in ("V1_segsum_hier_mulsum", "V1e_tf32_rn_emulated", "V3_mat2_global_decayinput"):
            m = P[T]["C64"][alpha][v].get("macd")
            if m:
                for part, d in m.items():
                    print(f"  {T} {alpha} {v} {part}:", {k: (f"{c['rel_err']:.1e}", c["PASS"]) for k, c in d.items()})
print("\nsplit:")
for k, v in R["split_invariance"].items():
    rel = max(c["rel_err"] for c in v["d_ema_minus_close"]["per_channel"].values())
    relv = max(c["rel_err"] for c in v["bb_var_ewma_d2"]["per_channel"].values())
    print(f"  {k}: max rel d {rel:.1e}, var {relv:.1e}, PASS {v['ALL_PASS']}, bitdiff {v['d_ema_minus_close']['n_elements_bitwise_different']}, first half bitwise {v['d_ema_minus_close']['bitwise_equal_first_half']}")
print("\ngradients:")
for k, v in R["gradients"].items():
    print(" ", k, "fd conv max", f"{max(v['fd_eps_1e-4_vs_1e-5_rel'].values()):.1e}",
          "tf64 vs fd max", f"{max(v['tf_float64_autodiff_vs_fd_rel'].values()):.1e}")
    for vn, e in v.items():
        if isinstance(e, dict) and "rel_err_vs_fd" in e:
            print(f"    {vn}: max {max(e['rel_err_vs_fd'].values()):.1e} PASS {e['PASS_1e-3']}",
                  {k2: f"{x:.0e}" for k2, x in e["rel_err_vs_fd"].items()})
print("\ncausality:", {k: (v["perturb_after_t_bitwise_unchanged"], v["future_jacobian_exactly_zero"]) for k, v in R["causality"].items()})
