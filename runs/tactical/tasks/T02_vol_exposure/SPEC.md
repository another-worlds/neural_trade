# T02: volatility-timed exposure to BTC (risk management of buy-and-hold)

Lead, 2026-10-13. Committed BEFORE any result. REVIEW_2026-10-13.md U2. This is a size-of-moves use of the
volatility forecast. It does not take a side.

## Question
Can a volatility forecast cut buy-and-hold's drawdown by more than plain de-leveraging does at the same average
exposure? H78 and H79 used volatility to size a DIRECTION signal. Timing market exposure itself was never tested.

## Data
The same files, the cutoff and the never-read rule as T01. Hourly bars are built from 1-minute bars. Entry and exit are
at the next hour's open.

## Strategy family (12 configurations, fixed here)
- Long-only exposure w_t in [0, 1], rebalanced hourly. Cost: 2 bps per unit of turnover (0 and 5 are also reported).
- **Forecast:** the next-24-hour realised volatility. It comes from a HAR ridge on log RV over 1 h, 4 h, 24 h and 7 d, plus hour of day.
  The ridge is fitted on the 365 days before each month (selection), or frozen at 2025-07-20 (forward).
- **Rules:**
  - (a) vol targeting: w = min(1, sigma* / sigma_hat). sigma* is the median forecast of the training span, times k
    for k in {0.5, 0.75, 1.0, 1.5}. That gives 4 configurations.
  - (b) a high-vol switch: w = 0 when sigma_hat is above the training span's q-quantile, else 1. q is in {0.7, 0.8, 0.9}.
    That gives 3 configurations.
  - (c) a vol-change switch from T01's frozen S1 model: w = 0 for the next hour when the predicted dA is above the
    training q-quantile, else 1. q is in {0.8, 0.9}. That gives 2 configurations.
  - (a) with a no-trade band: rebalance only when |w_new - w| > 0.2, for k in {0.75, 1.0, 1.5}. That gives 3 configurations.
- **Null for each configuration:** constant exposure c equal to that configuration's mean w over the same period
  (plain de-leveraging).

## Selection and looks
- **Selection:** 2019-01..2024-06 on the cut main file. The ONE configuration with the highest Calmar ratio of
  (strategy / its constant null) at 2 bps is chosen and written to `selection.json` before any look.
- **Look 1 (descriptive; this period was seen by earlier studies):** 2024-07..2025-07-20.
- **Look 2, the verdict:** the Binance forward data 2025-10..2026-09. The forecast is frozen as fitted on the 365 days to
  2025-07-20, and so are its quantile thresholds.

## Pre-registered criterion (Look 2, at 2 bps)
**Success** iff all three hold:
1. max drawdown is <= 0.8 x the constant null's max drawdown;
2. Calmar (annualised return / max DD) is above the constant null's;
3. Calmar is above buy-and-hold's.

Also reported:
- the owner's DD < 5% line;
- the exposure needed for DD < 5%;
- monthly returns;
- turnover;
- a block-bootstrap interval (blocks of 1 week) of the Calmar difference against the null.

## Tests (seconds)
- the never-read span;
- the forecast uses only past bars;
- next-open fills;
- the thresholds come from the training span only;
- the null's exposure equals the configuration's mean exposure.

## Outputs
`runs/tactical/tasks/T02_vol_exposure/`: code, `selection.json` (committed before the looks), `results.json`,
`tables.txt`, equity PNGs (`visualization/theme.py` colours), `REPORT.md`. CPU only.
