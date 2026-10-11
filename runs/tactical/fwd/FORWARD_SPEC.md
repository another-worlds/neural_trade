# FORWARD_SPEC: the frozen candidate for a forward test (frozen 2026-10-11)

Chosen by the pre-declared GRID.md (commit e2c4be7) on selection months 2019-01..2024-06 ONLY; the spent look
2024-07..2025-09 was never read (no bar on or after 2024-07-01 is loaded by the selection code; test
`test_no_bar_on_or_after_cutoff`). Selection results are in-sample-of-selection and are NOT evidence of skill.

## Configuration (exact)
- Bars: 1-hour, resampled from the 1-minute Binance BTC/USDT file; window 60 bars; one anchor per hour.
- Features: `rich` (`runs/tactical/lab/lab.py fs_rich`, 43 features, NaN/inf -> 0).
- Model: logistic regression, C = 0.1, StandardScaler fitted on the training rows, clip +-8 (`recent/wf.py _fit`).
- Walk-forward: monthly; train on the 730 days before the month (label bar closed by the month start, `wf.split_masks`
  purge); training rows with |next-open return| <= deadband 5e-4*sqrt(H*60/15) removed.
- Horizon H = 4 hours. Entry at the OPEN of the bar after the signal bar; exit at the close of the 4th bar.
- Trade selection: per test month the top 10% of |P - 0.5| (the month's own quantile); long if P > 0.5 else short;
  non-overlapping sequence (skip a signal while a position is open; 1x notional, additive equity).
- Cost: 2 bps round trip.
- Code: branch `nt-tactical-fwd`, commit e678b36c05b5fef51e742597f2a0a6e88c9f52c2 (`runs/tactical/fwd/fwdlib.py`,
  `forward.py`, selection `run_grid.py` + `select_freeze.py`; `grid.jsonl` sha256 prefix b8d420d08a11e73c).

## Selection-period numbers (IN-SAMPLE-OF-SELECTION, 2019-01..2024-06, 66 months, 2,651 trades)
| cost bps | hit | mean net bps/trade | mean net bps/month | max DD % | months with hit > 60% |
|---|---|---|---|---|---|
| 0 | 0.604 | +14.3 | +574 | 11.6 | 33/66 |
| 1 | 0.598 | +13.3 | +534 | 11.8 | 33/66 |
| 2 | 0.593 | +12.3 | +494 | 12.1 | 32/66 |
| 5 | 0.571 | +9.3 | +373 | 14.6 | 22/66 |
The grid had 48 candidates; the winner of 48 on 66 months carries selection optimism. The edge is largest in the
high-volatility 2019-2021 months; expect the forward number to be much smaller.

## Success criteria for the forward test (fixed now)
Data: bars strictly after 2025-09-29; test months start at 2025-10 (the script refuses earlier months). At least 6
COMPLETE months. All of these must hold at 2 bps round trip:
1. mean monthly net bps (sum of the month's trades' net bps) > 0;
2. the 95% t-interval (n_months - 1 d.o.f.) of the monthly net bps has a lower bound > 0;
3. max drawdown (additive equity, 1x notional) < 15% (selection-period value 12.1%);
4. the hit rate is reported (selection period 0.593), not a pass criterion; also reported: trades, mean net
   bps/trade, cumulative return, the per-month table.
With only 6 months the interval is wide: failing criterion 2 with a positive mean is "inconclusive", not
"refuted"; criteria 1-3 all met is "not refuted", and further months extend the test. No re-selection, no change of
any setting after the first forward number is seen; a different configuration is a new candidate with its own spec.

## How to run (one command)
Extend `Bitcoin_BTCUSDT.csv` (same columns timestamp,open,high,low,close,volume; 1-minute; it must still contain the
2023-2025 history for the 730-day training window), then:
`python runs/tactical/fwd/forward.py --csv <file> --first 2025-10 --last <last complete month>`
Output: `runs/tactical/fwd/forward_<first>_<last>.json` with the criteria and PASS flag. Running it needs the data
after 2025-09-29, which the owner must supply (D-050).
