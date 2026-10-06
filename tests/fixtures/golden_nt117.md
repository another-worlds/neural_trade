# NT-117 golden recording

`tests/fixtures/golden_nt117.npz` is `scripts/golden_run.py record` on CPU after `LAMBDA_SOFT_ECE` and `LAMBDA_VOL` ship at 0 (D-057). The recording from before that edit is not in git. `verify` against it is not the pass condition: this loss change is supposed to move the numbers.

Both recordings have 311 arrays. 39 are equal and 272 differ. No key was added or removed.

Equal: the five calibration weights that stay on their configured value (`lambda_casimir`, `lambda_extended_trend`, `lambda_hd`, `lambda_ife`, `lambda_t_perp`), the learning-rate history, the true up-rates, and the retired local / global / vac / ife / pnl history rows.

Differ, by prefix (equal / differ): `calib_lambda` 5/9, `coverage` 0/3, `hist` 34/197, `period` 0/54, `pred` 0/9.

The two edited weights, after the calibration pass:

- `calib_lambda/lambda_soft_ece`: 2.01468 to 0. The pass skips an inactive term.
- `calib_lambda/lambda_vol`: 3.84105 to 0.1. The configured value is 0, and the pass then applies its clamp floor of 0.1.

The other calibrated weights move with them: `lambda_short` 2.09347 to 2.18932, `lambda_point` 0.940436 to 0.983495, `lambda_long` 0.642615 to 0.672037, `lambda_dir` 0.736384 to 0.7701, `lambda_var` 0.405569 to 0.424139, `lambda_crps` 0.847212 to 0.886002, `ref_loss` 0.727969 to 0.7613. Test predictions, learned periods, coverage, and the remaining history rows differ. The largest prediction gap is `pred/delta/h2`, max abs 240.886.
