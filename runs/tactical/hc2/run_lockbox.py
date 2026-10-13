"""usage: python run_lockbox.py      THE ONE LOOK (SPEC.md): winner and baseline on the 12 lockbox months, 2 bps. Refuses without winner.json, without the
look caches (run_preds.py look, run_paths.py look) and when lockbox_result.json already exists."""
import sys, os, json, time
import numpy as np
from scipy import stats as sps
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hc2lib as H  # noqa: E402

OUT = f"{HERE}/lockbox_result.json"


def verdict_numbers(ev, cfg, draws=500):
    fs = H.full_stats(ev, cfg)
    ms = np.array(fs["monthly_net_bps"]); k = len(ms); se = ms.std(ddof=1) / np.sqrt(k); tq = sps.t.ppf(0.975, k - 1)
    fs["monthly_mean_bps"] = float(ms.mean()); fs["monthly_ci95"] = [float(ms.mean() - tq * se), float(ms.mean() + tq * se)]
    cum, dd, hit = H.random_side_null(ev, cfg, draws)
    fs["null"] = {"draws": draws, "cum_median": float(np.median(cum)), "cum_p95": float(np.quantile(cum, .95)), "dd_median": float(np.median(dd)),
                  "dd_p05": float(np.quantile(dd, .05)), "hit_p95": float(np.quantile(hit, .95)), "p_cum_ge_actual": float((cum >= fs["cum_pct"]).mean())}
    return fs


if __name__ == "__main__":
    win = json.load(open(f"{HERE}/winner.json"))["winner"]
    if os.path.exists(OUT):
        sys.exit("refused: the one look was taken (lockbox_result.json exists)")
    t0 = time.time(); inp = H.Inputs()
    dev = H.Evaluator(H.Store(inp)); look = H.Evaluator(H.Store(inp, "look", look=True))
    res = {"lockbox_months": list(H.MO.LOCKBOX), "winner_cfg": win, "baseline_cfg": H.BASELINE}
    for name, cfg in (("winner", win), ("baseline", H.BASELINE)):
        res[name] = {"cfg": H.ckey(cfg), "dev": H.full_stats(dev, cfg), "lockbox": verdict_numbers(look, cfg)}
    w, b = res["winner"]["lockbox"], res["baseline"]["lockbox"]
    same = H.ckey(win) == H.ckey(H.BASELINE)
    crit = {"1 mean monthly net bps, 95% t-interval lower bound > 0": bool(w["monthly_ci95"][0] > 0),
            "2 lockbox Calmar above the baseline's": bool((not same) and w["calmar"] > b["calmar"]),
            "3 cumulative return above the winner's random-side null p95": bool(w["cum_pct"] > w["null"]["cum_p95"])}
    res["criteria"] = crit; res["PASS"] = all(crit.values()); res["winner_is_baseline"] = same; res["seconds"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=1)
    for name in ("winner", "baseline"):
        for ph in ("dev", "lockbox"):
            s = res[name][ph]
            print(f"{name:8s} {ph:7s} n={s['n_trades']} hit={s['hit']:.3f} mean={s['mean_net_bps']:.2f}bps cum={s['cum_pct']:.1f}% dd={s['max_dd_pct']:.1f}% calmar={s['calmar']:.2f} "
                  f"months hit>60={s['months_hit_gt_60']}/{s['months_with_trades']} dd<5={s['months_dd_lt_5']}/{s['months_with_trades']} w_dd5={s['frontier']['w_dd5']} cum@w={s['frontier']['cum_at_w_dd5']:.1f}")
    print(crit, "PASS" if res["PASS"] else "NOT PASSED")
