"""O2 analysis (SPEC.md): 8-epoch effect of the physics terms on the variance head, 2 seeds x 6 slices, and d8 - d4.
usage: python analyze_o2.py score|report   (CPU). Imports analyze_phys7 scoring unchanged.
4-epoch references: seed 0 = runs/tactical/phys7/scores_phys7_none_{on,off}.json, seed 1 = runs/tactical/phys7b/scores_phys7b_none_*.json."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # runs/tactical
sys.path.insert(0, os.path.join(ROOT, "phys7"))
import analyze_phys7 as a  # noqa: E402


def rows8(c):
    return json.load(open(os.path.join(HERE, f"scores_o2_none_{c}.json")))


def rows4(c):
    r = [x for x in json.load(open(os.path.join(ROOT, "phys7", f"scores_phys7_none_{c}.json"))) if x["seed"] == 0]
    r += [x for x in json.load(open(os.path.join(ROOT, "phys7b", f"scores_phys7b_none_{c}.json"))) if x["seed"] == 1]
    return r


def onoff(R, key, who="net"):
    on = a.per_slice(R["on"], who, key)
    off = a.per_slice(R["off"], who, key)
    return a.paired(on, off)


def tab(R, tag, res):
    print(f"== {tag}: runs on/off {len(R['on'])}/{len(R['off'])}")
    for key, lab in (("spearman", "Spearman"), ("qlike", "QLIKE")):
        d, s = onoff(R, key)
        res[f"{tag}_{key}_on_minus_off"] = a.tci(d)
        print(lab, "on-off per slice", np.round(d, 4).tolist(), "mean/lo/hi/sd/n", [round(x, 4) for x in a.tci(d)], "up", int((d > 0).sum()))
        for hi, h in enumerate(a.HZ):
            d2, _ = a.paired({k: v[hi] for k, v in a.per_slice(R["on"], "net", key + "_h").items()},
                             {k: v[hi] for k, v in a.per_slice(R["off"], "net", key + "_h").items()})
            res[f"{tag}_{key}_h{hi}_on_minus_off"] = a.tci(d2)
            print(f"   h{hi} ({h})", [round(x, 4) for x in a.tci(d2)])
        for c in ("on", "off"):
            d3, _ = a.paired(a.per_slice(R[c], "net", key), a.per_slice(R[c], "har_tod", key))
            m = float(np.mean(list(a.per_slice(R[c], "net", key).values())))
            res[f"{tag}_{c}_{key}_net_minus_har"] = a.tci(d3) + [m]
            print(f"NET-HAR+tod physics {c} {lab}", [round(x, 4) for x in a.tci(d3)], "net mean", round(m, 4))
    for c in ("on", "off"):
        print("constant-variance runs", c, [r["slice"] for r in R[c] if any(r["info"]["const_var"])])


def main_report():
    R8 = {c: rows8(c) for c in ("on", "off")}
    R4 = {c: rows4(c) for c in ("on", "off")}
    res = {}
    tab(R8, "8ep_seeds01", res)
    tab(R4, "4ep_seeds01", res)
    for s in (0, 1):
        tab({c: [r for r in R8[c] if r["seed"] == s] for c in R8}, f"8ep_seed{s}", res)
    # verdict inputs
    d8, s8 = onoff(R8, "spearman")
    d4, s4 = onoff(R4, "spearman")
    dd = d8 - d4 if s8 == s4 else None
    assert dd is not None, "slice sets differ"
    res["d8_minus_d4_spearman"] = a.tci(dd)
    print("== d8 - d4 (Spearman on-off), per slice", np.round(dd, 4).tolist(), [round(x, 4) for x in a.tci(dd)])
    q8, _ = onoff(R8, "qlike")
    q4, _ = onoff(R4, "qlike")
    res["d8_minus_d4_qlike"] = a.tci(q8 - q4)
    print("d8 - d4 QLIKE", [round(x, 4) for x in a.tci(q8 - q4)])
    # net minus HAR 8 vs 4 per cell
    for c in ("on", "off"):
        for key in ("spearman", "qlike"):
            g8 = a.paired(a.per_slice(R8[c], "net", key), a.per_slice(R8[c], "har_tod", key))[0]
            g4 = a.paired(a.per_slice(R4[c], "net", key), a.per_slice(R4[c], "har_tod", key))[0]
            res[f"gap_{c}_{key}_8_minus_4"] = a.tci(g8 - g4)
            print(f"net-HAR {c} {key}: 8ep {np.mean(g8):+.4f}  4ep {np.mean(g4):+.4f}  change {[round(x, 4) for x in a.tci(g8 - g4)]}")
    # epochs run / served epoch per run
    print("== epochs run / served epoch per run")
    for c in ("on", "off"):
        for f in sorted(glob.glob(os.path.join(ROOT, "screens", f"o2_none_{c}", "results*.jsonl"))):
            for l in open(f, encoding="utf-8"):
                x = json.loads(l)
                keys = [k for k in x if "epoch" in k.lower()]
                print(c, x.get("data_end"), x.get("seed"), {k: x[k] for k in keys}, "wall_s", x.get("wall_s"))
    json.dump(res, open(os.path.join(HERE, "report_tables.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        for c in ("on", "off"):
            a.score(f"o2_none_{c}", ROOT)  # writes into phys7/ ; moved here
            os.replace(os.path.join(ROOT, "phys7", f"scores_o2_none_{c}.json"), os.path.join(HERE, f"scores_o2_none_{c}.json"))
    else:
        main_report()
