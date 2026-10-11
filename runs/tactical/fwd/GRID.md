# Forward candidate: pre-declared selection grid (written before any number was computed)

Purpose: choose ONE frozen configuration for a future forward test on data after 2025-09-29. The look
2024-07..2025-09 is spent (H56/H58/H61) and is NOT read here. No bar on or after 2024-07-01 is loaded.

Data and walk-forward: `Bitcoin_BTCUSDT.csv` 1-minute bars resampled to 1 hour, truncated at load to timestamps
< 2024-07-01. Monthly walk-forward of `runs/tactical/recent/wf.py` (split_masks purge, deadband mask on training rows,
StandardScaler fitted on train, clip 8, `_fit` models, anchors every hour, window 60 bars). Selection months: every
month 2019-01 .. 2024-06 that has enough rows (66 months at most).

Entry: next-open (open of the bar after the signal bar, as `runs/tactical/nextopen/`); exit: close of the H-th bar
(holding period H hours). Label return and the training labels use the same next-open return.

Grid (24 cells):
- features: tb7, rich
- model: logistic (C = 0.1), L1 (liblinear, C chosen on the tail 20% of the training rows)
- training length: 365 d, 730 d
- horizon H: 1, 2, 4 hours
- trade selection: top 10% / top 5% of the month's own confidence |P - 0.5| (the month's own quantile, as wf.py),
  then a non-overlapping sequence (eqlib.non_overlapping: scan by entry, skip while a position is open); this
  gives 48 candidates.
- deadband for the training mask: 5e-4 * sqrt(H * 60 / 15) (wf.deadband's formula with the cell's H).

Score (fixed now): for each selection month, the sum of the net bps of the month's trades (trade entered in that
month, round-trip cost 2 bps, 1x notional); the score is the mean of that monthly number over the selection
months. The single best candidate by score is frozen. Tie (scores within 0.5 bps): the lower max drawdown (additive
equity, eqlib convention) at 2 bps.

Nothing else is tuned. In-sample-of-selection numbers (hit, bps, drawdown at 0/1/2/5 bps) are reported for the
chosen candidate and are labelled as such.
