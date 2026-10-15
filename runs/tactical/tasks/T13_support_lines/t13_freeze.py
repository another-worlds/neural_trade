"""T13 freeze (SPEC.md "Prospective"): A1 (with H3 and AS), B1 (with L1) and C1 (with V3 and AR) fitted ONCE on the Binance file 2025-10-02..2026-09-30 with
the pair chosen by selection.json; written to frozen_B.json with its sha256. Reads no bar after 2026-09-30 23:59 and no October-2026 target.
Also the JSON <-> model conversions used by score_B.py."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import t13_lib as L


# ---------------------------------------------------------------- JSON <-> models
def _lst(a):
    return [float(v) for v in np.asarray(a).ravel()]


def _logit_to(m):
    return None if m is None else {"mu": _lst(m[0]), "sd": _lst(m[1]), "coef": _lst(m[2]), "ic": float(m[3])}


def _logit_from(j):
    return None if j is None else (np.array(j["mu"]), np.array(j["sd"]), np.array(j["coef"]), float(j["ic"]))


def a_to_json(m):
    h = m["har"]
    return {"pair": m["pair"], "n_train": m["n_train"], "last_label_end": m["last_label_end"],
            "har": {"mu": _lst(h["mu"]), "sd": _lst(h["sd"]), "coef": _lst(h["coef"]), "ic": float(h["ic"]), "c": float(h["c"]), "plus": bool(h["plus"])},
            "h3": _logit_to(m["h3"]), "a1": _logit_to(m["a1"]), "as": _logit_to(m["as"])}


def a_from_json(j):
    h = j["har"]
    return {"pair": j["pair"], "n_train": j["n_train"], "last_label_end": j["last_label_end"],
            "har": {"mu": np.array(h["mu"]), "sd": np.array(h["sd"]), "coef": np.array(h["coef"]), "ic": h["ic"], "c": h["c"], "plus": h["plus"]},
            "h3": _logit_from(j["h3"]), "a1": _logit_from(j["a1"]), "as": _logit_from(j["as"])}


def _ridge_o3_to(m):
    return {"mu": _lst(m["mu"]), "sd": _lst(m["sd"]), "w": _lst(m["w"]), "ym": float(m["ym"])}


def _ridge_o3_from(j):
    return {"mu": np.array(j["mu"]), "sd": np.array(j["sd"]), "w": np.array(j["w"]), "ym": float(j["ym"])}


def b_to_json(st):
    return {"pair": st["pair"], "n_tr": st["n_tr"], "first_ta": st["first_ta"], "last_label_end": st["last_label_end"], "m": {k: _ridge_o3_to(v) for k, v in st["m"].items()}}


def b_from_json(j):
    return {"pair": j["pair"], "n_tr": j["n_tr"], "first_ta": j["first_ta"], "last_label_end": j["last_label_end"], "m": {k: _ridge_o3_from(v) for k, v in j["m"].items()}}


def c_to_json(m):
    return {"pair": m["pair"], "n_tr": m["n_tr"], "last_label_end": m["last_label_end"], "ar": _lst(m["ar"]), "V3": m["V3"], "C1": m["C1"]}


def c_from_json(j):
    return {"pair": j["pair"], "n_tr": j["n_tr"], "last_label_end": j["last_label_end"], "ar": list(j["ar"]), "V3": j["V3"], "C1": j["C1"]}


# ---------------------------------------------------------------- freeze
def load_freeze_bars(path=None):
    return L.L12.load_upto(path or L.guard.FWD_CSV, L.F_LO, L.F_HI)


def freeze(bars, pairs, selection_sha):
    """pairs = {'A': (bar, nb), 'B': ..., 'C': ...}. bars: Binance 1-minute bars F_LO..F_HI."""
    assert bars.index.max() <= L.F_HI and bars.index.max() < L.F_M0
    art = {"spec": "T13 SPEC.md (1869df6) freeze", "selection_json_sha256": selection_sha, "pairs": {k: L.pair_key(v) for k, v in pairs.items()},
           "m0": str(L.F_M0), "last_bar": str(bars.index.max()), "n_bars": int(len(bars)), "train_from": str(L.TRAIN_FROM_B)}
    P = L.L10.prep(bars, "fwd"); B = L.L10.build(P, L.T_MIN); Bp = L.a_pair_B(P, B, pairs["A"]); days = (L.F_M0 - L.TRAIN_FROM_B) / pd.Timedelta(days=1)
    art["A"] = a_to_json(L.a_fit(Bp, P, L.T_MIN, L.F_M0, days, pairs["A"])); del P, B, Bp
    F = L.build_B_frame(bars, L.O5.FREEZE_FIRST, pairs=[pairs["B"]]); art["B"] = b_to_json(L.b_fit(F, L.F_M0, pairs["B"])); del F
    BC = L.build_C(bars, L.T01.WARM, "binance", pairs=[pairs["C"]]); art["C"] = c_to_json(L.c_fit(BC, pairs["C"], L.F_M0, start=L.TRAIN_FROM_B))
    art["sha256"] = L.sha_of(art)
    return art


if __name__ == "__main__":
    sel_path = os.path.join(HERE, "selection.json"); commit = L.assert_selection_committed(sel_path); sel = json.load(open(sel_path))
    assert os.path.exists(os.path.join(HERE, "results.json")), "run run_judge.py (J1, J2) before the freeze"
    bars = load_freeze_bars(); print("selection committed in", commit, "| bars", len(bars), bars.index.min(), bars.index.max(), flush=True)
    pairs = {p: L.pair_from_key(sel["chosen"][p]) for p in "ABC"}
    art = freeze(bars, pairs, L.sha_file(sel_path)); json.dump(art, open(os.path.join(HERE, "frozen_B.json"), "w"), indent=1)
    open(os.path.join(HERE, "frozen_B.sha256"), "w").write(art["sha256"] + "\n")
    print({k: v for k, v in art.items() if k not in ("A", "B", "C")})
    print("A n_train", art["A"]["n_train"], art["A"]["last_label_end"], "| B n_tr", art["B"]["n_tr"], art["B"]["first_ta"], art["B"]["last_label_end"], "| C n_tr", art["C"]["n_tr"], art["C"]["last_label_end"])
