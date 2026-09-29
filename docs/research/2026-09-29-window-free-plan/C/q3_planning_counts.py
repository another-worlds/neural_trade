"""Q3: planning pair counts (1 seed per fold, so each pair carries the arm x fold interaction) for the
recommended primaries, with a 1.25 inflation of the v1 pair SD for 7-day training blocks (an ESTIMATE:
the training-block effect was not measured). One-sided alpha 0.05, power 0.80, true difference 0.
Writes q3_planning_counts.json."""
import json, math
from pathlib import Path
from scipy import stats
HERE = Path(__file__).resolve().parent
noise = json.loads((HERE / "q2_noise_v1.json").read_text())
def comp(m): return noise["metrics"][m]["components"]["pooled"]
def n80(delta, sd):
    for n in range(2, 5000):
        se = sd / math.sqrt(n)
        if 1 - stats.nct.cdf(stats.t.ppf(0.95, n - 1), n - 1, delta / se) >= 0.80:
            return n
out = {}
for m, edges in (("logcrps_mean", {"planning_latest_test_run(sizing only)": 0.019519, "dev_-2_checkpoint_today_like": 0.041782, "dev_-2_checkpoint_all12": 0.040387}),
                 ("logcrps_h1", {"planning_latest_test_run(sizing only)": 0.016856, "dev_-2_checkpoint_today_like": 0.041560, "dev_-2_checkpoint_all12": 0.040197})):
    c = comp(m)
    for infl in (1.0, 1.25, 1.5):
        sd = math.sqrt((c["sd_pair_nominal_seed"] * infl) ** 2 + 2 * c["s_cp"] ** 2)
        for name, E in edges.items():
            for frac in (1/3, 1/2):
                out[f"{m}|infl{infl}|{name}|frac{frac:.2f}"] = {"sd_per_pair": round(sd, 5), "margin": round(E * frac, 5), "pairs_80pct": n80(E * frac, sd)}
(HERE / "q3_planning_counts.json").write_text(json.dumps(out, indent=2))
for k, v in out.items():
    print(k, v)
