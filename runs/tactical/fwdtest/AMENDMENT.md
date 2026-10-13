# AMENDMENT to FORWARD_SPEC.md for the forward test on Binance klines (written before any forward score)

FORWARD_SPEC.md (frozen at 1bda790) stays in force except where this file amends it. Written 2026-10-13, committed before the
run script was executed. No forward label, return or score had been computed at the time of writing; the only forward-data
numbers computed are the data-quality report and the feature shift of 2025-10 (features only, `shift_2025-10.txt`), both
committed with this file.

## Source mismatch (this is a CROSS-SOURCE test: weaker evidence)

The forward bars come from Binance spot BTCUSDT 1-minute klines (data.binance.vision, checksums verified). The training bars
come from the main file `Bitcoin_BTCUSDT.csv`, used through its cut copy. The two are NOT the same feed. Check on 2025-06
(a non-protected month; `source_check_2025-06.json`, `source_check_2025-06_diag.json`): 43,200 rows each, identical
timestamps, but only 0.009% of closes are equal, no row matches on all five fields; max abs diff open 265.7, high 276.8,
low 248.0, close 291.6, volume 447.4; close difference (Binance minus main) mean -46.8, std 23.6; 1-minute return correlation
0.92, hourly 0.9987; the main file has integer prices only (Binance 6.8%), 6% of Binance's volume sum and volume correlation
0.44. The venue of the main file was not identified. Consequence: the frozen model, trained on the main file's feed, is scored
on a different feed. The features are mostly scale-free (returns in volatility units, oscillators, volume relative to the
window mean), so the shift is expected to be modest, but any forward result is a statement about this cross-feed transfer, not
a clean out-of-time test of the candidate on its own source. The verdict is labelled accordingly and is weaker evidence than
FORWARD_SPEC intended.

## Amended procedure

1. Training, once: the model (logistic regression, C = 0.1, rich features, deadband, StandardScaler, clip +-8, H = 4 hours)
   is fitted ONCE on the 730 days ending 2025-07-20 23:59 of the cut copy (training anchors whose label bar closes by
   2025-07-21 00:00, i.e. no protected bar), instead of the monthly refit of FORWARD_SPEC (which would need the protected span),
   and is frozen. No refit afterwards.
2. Features for the new months use only bars of the new file: 1-hour bars from the new file, a 60-bar window per anchor.
   Every anchor whose window would reach back before 2025-10-01 is dropped (the first anchor is at 2025-10-03 11:00).
3. Threshold: for month m the entry threshold is the 90th percentile of |P - 0.5| over ALL anchors of month m-1 (the spec's
   "top 10%", with the previous month's quantile instead of the month's own). 2025-10 is calibration-only (no trade); the
   scored months are 2025-11 up to the last COMPLETE month in the downloaded file.
4. Trades: signal at the hour bar; long if P > 0.5 else short; entry at the open of the next 1-hour bar, exit at the close of
   the 4th bar; non-overlapping (skip while a position is open); 1x notional, additive equity; round trip 2 bps (also 0, 1, 5).
5. Success criteria, unchanged, at 2 bps: (1) mean monthly net bps > 0; (2) 95% t-interval (n_months - 1 d.o.f.) lower bound > 0;
   (3) max drawdown < 15%; hit reported. With fewer than 6 scored months the verdict is "inconclusive". Positive mean with the
   interval including 0 is "inconclusive", not "refuted". All reported with the cross-source label above.
6. Also reported, not criteria: the owner's goal metrics (hit > 60% overall, max drawdown < 5%, number of scored months with
   monthly hit > 60% and with in-month drawdown < 5%, equity restarting each month); the random-side null on the same trades
   (2,000 coin-flip draws per trade; cumulative return and hit median/p95, and the share of draws with cumulative return at or
   above the actual); the leverage frontier: per-trade net return (2 bps included) multiplied by leverage L = 1, 2, 3, 5, 10,
   20 (additive equity, cost scales with L), cumulative return and max drawdown, and the largest L with max drawdown < 5% and
   its cumulative return.

## Arms

- **frozen** (the only verdict arm): exactly the recipe above, with the training StandardScaler.
- **diag_oct_scaler** (one pre-declared diagnostic arm, NOT a verdict, no selection between arms): the same frozen model
  (same coefficients) but each feature standardised with the mean and std of the calibration month 2025-10 only (all its
  anchors), then clipped at +-8. It uses no data after 2025-10 for the scaler. It exists to show how much of any result is
  feed shift. Both arms are reported; if they disagree the frozen arm decides.

## Per-feature shift for 2025-10

`shift_2025-10.txt` / `.json`: for the 685 anchors of 2025-10, the z-mean and z-std of every feature under the frozen
training scaler (volume families listed first: mfi_14, mfi_30, log_vol_ratio_last, log_vol_ratio_5, obv_slope_10,
vwap_dev_20). Summary: mean |z-mean| 0.103 for the volume families against 0.070 for the others; mean z-std 0.893 against
1.005; largest |z-mean| 0.162 (obv_slope_10); no feature is clipped in more than 0.3% of rows. The shift is small, so the
mismatch is mostly benign for the standardised inputs, but the cross-feed caveat stays.

## Tests (seconds; `test_fwdtest.py`, `tests.log`)

No bar between 2025-07-21 and 2025-09-30 is read (loaders assert, on synthetic and the real files); the training span ends
2025-07-20 (loader assert and label-closed check); next-open return; the threshold uses only the previous month; the first
forward anchor is after the warm-up.

## Not allowed after the first number

No change of any setting, arm, threshold rule or criterion. A different configuration is a new candidate with its own spec.
