"""Q3 (0): probability of each probe classification given the measured reach-epoch noise.
Measured (q3_probe_target.json): SD of ln(reach epoch) across v1 dev-fold runs ~0.40 (within condition).
Model: ln(mean-seed reach ratio) ~ N(ln rho, 0.40^2 * 2 / S). Epoch-bound iff ratio <= 1.5, update-bound iff >= 3.
Writes q3_probe_classification.json."""
import json, math
from pathlib import Path
from scipy import stats
sd_run = json.loads(Path(__file__).with_name("q3_probe_target.json").read_text())["v1_dev_fold_-2"]["frac0.05"]["within_condition_sd_log_reach_epoch"]
out = {"sd_log_reach_epoch_per_run": sd_run, "table": {}}
for S in (3, 4, 5, 6):
    se = sd_run * math.sqrt(2.0 / S)
    for rho in (1.0, 1.2, 1.5, 2.0, 3.0, 4.0):
        p_epoch = stats.norm.cdf((math.log(1.5) - math.log(rho)) / se)
        p_update = 1 - stats.norm.cdf((math.log(3.0) - math.log(rho)) / se)
        out["table"][f"S{S}_rho{rho}"] = {"epoch_bound": round(p_epoch, 3), "update_bound": round(p_update, 3),
                                          "inconclusive": round(1 - p_epoch - p_update, 3)}
Path(__file__).with_name("q3_probe_classification.json").write_text(json.dumps(out, indent=2))
for k, v in out["table"].items():
    print(k, v)
