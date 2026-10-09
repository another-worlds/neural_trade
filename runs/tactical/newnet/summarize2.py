"""newnet2 table: python summarize2.py [tag-substring] [--only-slices climb6]
One row per record of results.jsonl (variant x span x seed). `d_auc3` is the paired difference to the regression on the same slices and span;
its 95% t-interval is over slices. Tails: honest top-10% (plain 'hon', magnitude-filtered 'mon') in bps against the random-sign null95."""
import json
import sys

import numpy as np
from scipy import stats

CLIMB6 = ["2017-05-02T12", "2018-11-16T02", "2020-03-30T04", "2021-10-13T18", "2023-04-29T08", "2024-11-11T23"]


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return f"{v.mean():+.4f}" if len(v) else "n/a"
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():+.4f} [{v.mean() - h:+.4f},{v.mean() + h:+.4f}]"


def pick(r, key, idx):
    return [r["per_slice"][key][i] for i in idx]


def rows(path="results.jsonl", sub=None, only6=False):
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        tag = r.get("tag", "")
        if sub and sub not in tag and not (sub == "base" and tag == ""):
            continue
        names = r["slices"]
        idx = [i for i, n in enumerate(names) if (n in CLIMB6 or not only6)]
        if not idx:
            continue
        ps = r["per_slice"]
        out.append(dict(tag=tag or "orig", arch=r["arch"], span=r["span"], seed=r["seed"], n=len(idx), params=r["params"]["total"],
                        sec=r["seconds"], auc3=np.mean(pick(r, "auc3", idx)), auc_lin=np.mean(pick(r, "auc3_lin", idx)),
                        d=ci(pick(r, "d_auc3", idx)), llgap=np.mean(np.array(pick(r, "ll_h1", idx)) - np.array(pick(r, "ll_const_h1", idx))),
                        hon=np.nanmean(pick(r, "hon10_bps", idx)), honnull=np.nanmean(pick(r, "hon10_null95", idx)),
                        mon=np.nanmean(pick(r, "mon10_bps", idx)), monnull=np.nanmean(pick(r, "mon10_null95", idx)),
                        monlin=np.nanmean(pick(r, "mon_lin10_bps", idx)), vol=np.mean(pick(r, "vol_rho", idx)),
                        volb=np.mean(pick(r, "vol_rho_base", idx)), span_days=np.mean([r["slice_info"][i]["days"] for i in idx]) if r["slice_info"] else float("nan"),
                        warm=r.get("warm", 0), tl=r.get("train_lin", False)))
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    for x in rows(sub=a[0] if a else None, only6="--only-slices" in sys.argv):
        print(f"{x['tag']:18s} {x['arch']:6s} {x['span']:3s} seed {str(x['seed']):3s} n={x['n']:2d} p={x['params']:5d} {x['sec']:6.0f}s | mean3 {x['auc3']:.4f} (reg {x['auc_lin']:.4f}) "
              f"| d_auc3 {x['d']} | ll-const {x['llgap']:+.4f} | top10 {x['hon']:+.2f} (null95 {x['honnull']:+.2f}) | mag-filt {x['mon']:+.2f} "
              f"(null95 {x['monnull']:+.2f}; reg {x['monlin']:+.2f}) | vol rho {x['vol']:.3f} (base {x['volb']:.3f}) | mean days {x['span_days']:.0f}")
