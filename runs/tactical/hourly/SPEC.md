# Hourly triple-barrier study: SPEC (fixed before any run, 2026-10-09 ~19:30)

Owner, 2026-10-09: "да займись этим активно. построй эксперименты до завтрашнего утра" (after H33: the triple barrier with a
meta model earned +7..+19 bps per trade above the random-side null on hourly bars).

## Data and honesty

- Bitcoin_BTCUSDT.csv resampled to 1-hour bars, 2017-01-01 .. 2025-09-29.
- **Dev period:** bars up to 2024-06-30. Every choice (barrier geometry, features, models, thresholds) is made here, by walk-forward:
  8 folds over the dev anchors from 30% on, each trained on everything before it minus a 2T-bar gap.
- **Held-out period:** 2024-07-01 .. 2025-09-29 (~15 months), never looked at during the grid. Touched ONCE, after the grid, for the
  top 3 dev configurations: each is trained on all dev data (minus the gap) and scored on the held-out period.
- Costs 0 (D-044). Net bps at 5 and 10 bps per round trip are reported for information only.

## Trade definition

Entry at the close of the window's last bar; sigma = std of the last 60 hourly log returns; take-profit at +TP * sigma * sqrt(T),
stop at -SL * sigma * sqrt(T) (in the trade's direction), time limit T bars; the first touched by the following bars' high / low wins
(both in one bar: the stop); otherwise the return at T.

## Grid (dev only)

- Geometry: T in {4, 8, 12, 24, 48} h x (TP, SL) in {(1,1), (1.5,1.5), (2,1), (1,2), (2,2)}, with tb7 features, logistic primary on
  the sign at T, meta boosting (25 runs).
- Then, at the best 3 geometries of that stage by the dev ranking below: features {tb7, rich, ctx (rich + hourly context: returns over
  1/4/12/24/72/168 h over vol, vol ratios, position in the 24 h and 168 h range, hour-of-day and day-of-week sin/cos, volume ratio)} x
  primary {logistic, boosting} x primary target {sign at T, barrier outcome} x meta features {base, base + predicted log|move|}.
- Trade selection: the primary's top 20% confidence (threshold from train) and the meta model's top 20% (threshold on the last 25%
  of train).

## Ranking and success (fixed now)

- Dev rank of a configuration: the mean over the 8 folds of (meta bps - random-side null95 bps); ties by the lower 95% bound of the
  meta bps.
- **Held-out success:** the dev winner's held-out meta bps is above its random-side null95 AND its 95% interval over calendar months
  is above 0. The 2nd and 3rd are reported, not judged. A failure is recorded as such; nothing is re-tuned on the held-out period.
