"""phys7b analysis (SPEC.md): 3-seed means for the primary pair. Seed 0 = scores of the phys7 runs (scores_phys7_none_*.json),
seeds 1, 2 = phys7b. usage: python analyze_phys7b.py score|report  (CPU)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P7 = os.path.join(os.path.dirname(HERE), "phys7")
sys.path.insert(0, P7)
import analyze_phys7 as a  # noqa: E402

ROOT = os.path.dirname(HERE)  # runs/tactical


def rows(c, seeds):
    out = []
    if 0 in seeds:
        out += [r for r in json.load(open(os.path.join(P7, f"scores_phys7_none_{c}.json"))) if r["seed"] == 0]
    f = os.path.join(HERE, f"scores_phys7b_none_{c}.json")
    if os.path.exists(f):
        out += [r for r in json.load(open(f)) if r["seed"] in seeds]
    return out


def tables(seeds, tag, res):
    R = {c: rows(c, seeds) for c in ("on", "off")}
    print(f"== {tag}: seeds {seeds}; runs on/off {len(R['on'])}/{len(R['off'])}")
    for key, lab in (("spearman", "Spearman"), ("qlike", "QLIKE")):
        on = a.per_slice(R["on"], "net", key)
        off = a.per_slice(R["off"], "net", key)
        d, s = a.paired(on, off)
        res[f"{tag}_{key}_on_minus_off"] = a.tci(d)
        print(lab, "on-off per slice", np.round(d, 4).tolist(), "mean/lo/hi/sd/n", [round(x, 4) for x in a.tci(d)], "up", int((d > 0).sum()))
        for hi, h in enumerate(a.HZ):
            d2, _ = a.paired({k: v[hi] for k, v in a.per_slice(R["on"], "net", key + "_h").items()},
                             {k: v[hi] for k, v in a.per_slice(R["off"], "net", key + "_h").items()})
            res[f"{tag}_{key}_h{hi}_on_minus_off"] = a.tci(d2)
            print(f"   h{hi} ({h})", [round(x, 4) for x in a.tci(d2)])
        for c in ("on", "off"):
            d3, _ = a.paired(a.per_slice(R[c], "net", key), a.per_slice(R[c], "har_tod", key))
            res[f"{tag}_{c}_{key}_net_minus_har"] = a.tci(d3)
            m = float(np.mean(list(a.per_slice(R[c], "net", key).values())))
            print(f"NET-HAR+tod physics {c} {lab}", [round(x, 4) for x in a.tci(d3)], "net mean", round(m, 4))
    for c in ("on", "off"):
        bad = [r["slice"] for r in R[c] if any(r["info"]["const_var"])]
        print("constant-variance runs", c, bad)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        for c in ("on", "off"):
            a.score(f"phys7b_none_{c}", ROOT)  # writes into phys7/ (HERE of analyze_phys7); moved
            os.replace(os.path.join(P7, f"scores_phys7b_none_{c}.json"), os.path.join(HERE, f"scores_phys7b_none_{c}.json"))
    else:
        res = {}
        tables((0, 1, 2), "seeds012", res)
        tables((1, 2), "seeds12", res)
        tables((0,), "seed0", res)
        json.dump(res, open(os.path.join(HERE, "report_tables.json"), "w"), indent=1)
