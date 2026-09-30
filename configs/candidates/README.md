# Saved candidates (owner 2026-09-30: "сохрани 1-3")

The top three rows of the zero-cost ranking (D-044), saved so they survive and can be re-checked. All three use
the six 360-day models of `configs/scenarios/long_360d_stab.yaml` (BTC/USDT 1-minute, 60-bar window, horizons
10/15/20 min, trained at dce15ed on folds -3 / -2 x seeds 0-2). Numbers are on the dev blocks (~32 days each);
the newest fold (-1) is still untouched.

| id | what | mean return per block | profitable blocks | mean win share | max drawdown |
|---|---|---|---|---|---|
| C1 | each model -> calibrated_quantile 0.9, full size | +10.2% | 6/6 | 54.1% | 6.45% |
| C2 | each model -> calibrated_quantile 0.95, full size | +8.6% | 6/6 | 53.8% | 5.45% |
| C3 | seed ensemble per fold -> calibrated_quantile 0.9, size 0.7 | +6.0% | 2/2 | 54.2% | 4.5% |

- Model files (weights, calibration, config, stored predictions): `saved_models/long_360d/` (git-ignored,
  machine-local, 89 MB). `manifest.json` holds every file's sha256 and every cell's reproduced numbers.
- Re-check: `python configs/candidates/save_candidates.py` (copies only missing files, re-verifies the sha256 of
  every copy against the run store, recomputes all three candidates from the saved predictions).
- None meets the owner's goal in full (win share >= 60%); C3 meets drawdown < 5% with profit on both folds.
