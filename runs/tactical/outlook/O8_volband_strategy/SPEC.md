# O8: a trading strategy built on the geometry-enhanced volatility forecast, backtested on independent data

Outlook lead, 2026-10-13. Committed BEFORE any strategy number is computed. Owner (2026-10-13): "придумай и построй стратегию на
основе нашего нового подхода, проведи бэктест на независимых данных". Tactical, exploratory; CPU only.

## The approach and the idea
Our new approach predicts the SIZE of the next 15-60 minutes, not the sign. It is L1 of O5: a ridge on S3 + the EWMA bank + the
geometry G1 + G15, with target ln RV (QA PASS in O1 and O5). A spot strategy can use a size forecast without any direction skill
by placing orders at a distance that scales with the forecast:
- **REV (band reversion):** rest a buy limit below the price and a sell limit above it. On a fill, aim back at the anchor price.
  This earns when excursions of the forecast size revert.
- **BRK (band breakout):** rest a buy stop above and a sell stop below. On a fill, ride the move. This earns when excursions continue.

Which of the two works is decided on the selection months only. A better size forecast should put the bands where excursions
actually end (REV) or begin (BRK). That is the testable claim "our approach matters": the same strategy with a size forecast
WITHOUT geometry must do worse.

## Size forecast
- sigma_H(t) = sqrt(exp(mu_H(t))), the forecast standard deviation of the log return over the next H minutes, H in {15, 60}.
  mu_H is the L1 prediction of ln RV(t, t+H) from bars < t.
- **Selection and tests A/B:** monthly walk-forward fits on the 365 days before each month (as O5).
- **Test C:** one frozen fit on the 365 days ending 2025-07-20 23:59. Features on the forward file are computed from forward bars
  only, after the warm-up. Nothing is fitted on forward data.
- **Arms of the size forecast** (same orders, different sigma):
  - **L1:** the approach.
  - **R0:** S3 only, no geometry, no bank.
  - **TRAIL:** sqrt(RV over the last H minutes).
  - **SHUF:** L1's sigma permuted at random within each month (keeps the level distribution, destroys the timing; seed 0). This is the null.

## Mechanics (1-minute OHLC, every fill pessimistic)
- **Decision times:** every 15 minutes on the clock (:00, :15, :30, :45), at the open of that minute's bar. The anchor is
  m = open of the decision bar, and sigma uses bars before that bar.
- **One position at a time.** While flat, the decision places both orders, valid for the next 15 bars. They are cancelled at the
  next decision time, and the first one to fill wins. While in a position, no new orders are placed.
- **REV entries:** buy limit at L = m * (1 - k * sigma_H), sell (short) limit at U = m * (1 + k * sigma_H).
  - A limit fills only if the bar trades THROUGH it by at least 1 bp (low <= L * (1 - 1e-4), or high >= U * (1 + 1e-4)).
  - The fill price is the limit price. If the bar OPENS beyond the level, the fill is at the open, which is better for a limit and is allowed.
- **BRK entries:** buy stop at U, sell stop at L.
  - The order triggers when high >= U (or low <= L).
  - The fill price is max(U, open) for a buy and min(L, open) for a sell, so a gap fills at the worse price.
- **Exits** (checked from the bar AFTER the entry bar):
  - **REV target:** a limit at m, with the same 1 bp trade-through rule.
  - **BRK target:** none.
  - **Stop:** at s * sigma_H beyond the entry price (s in {1, none}). A stop fills at the stop price, or at the open if the bar opens beyond it.
  - **Time stop:** at the open of bar entry + H.
  - **Ties:** if a target and a stop can both fill in one bar, the stop fills first.
- **Costs:** 2 bps round trip in the primary (the owner's 2026-10-13 hill-climb choice). 0 and 5 bps are also reported (D-044: costs are 0 unless the study sets them).
- **Returns:** a position's return is the log price change in its direction minus the cost. A month's return is the sum over its
  positions, with capital constant at 1x and no compounding.
- **Drawdown:** taken on the cumulative sum of position returns within each test span.

## Configurations (24) and the choice
family {REV, BRK} x k {1.0, 1.5, 2.0} x H {15, 60} x stop s {1, none}, all with the L1 arm.
**Choice:** on the selection months 2021-01 .. 2024-06 (42), the configuration with the highest t-statistic of the monthly net
return at 2 bps. Only that configuration goes to the tests. The choice is written to `choice.json` and committed BEFORE any test
month is computed. The test runs are a separate script invocation that reads `choice.json`.

## Independent tests (no strategy of this kind was ever run on them)
- **B: 2024-07 .. 2025-07-20** (13 months, the last one partial). Walk-forward sigma.
- **C: 2025-10-05 .. 2026-10-12** (the Binance forward file, about 12.3 months). Frozen sigma. 2026-10 is partial (12 days) and has never been read by any study.
- **A: 2018-07 .. 2020-12** (30 months). Walk-forward sigma. Reported separately: that market had a strong short-term reversal (H51, H59), so A is not the verdict.
- **Caveat:** L1's volatility RANKING was judged on A, B and C (O1, O5), and C's months were read by T01-T04. Their prices were never
  used to choose or tune a strategy. The 2026-10 days are fully unseen.

## Pre-registered criterion (pooled B + C, unit = month, 95% t-interval)
1. **Profitable:** the mean monthly net return of the chosen configuration (L1 arm, 2 bps) is > 0, with its lower bound > 0.
2. **Not luck:** L1 minus SHUF, paired by month, has a mean > 0 with its lower bound > 0.
3. **The approach matters:** L1 minus R0, paired by month, has a mean > 0 with its lower bound > 0.

Verdicts:
- **"strategy works on independent data"** iff 1 and 2 hold;
- **"geometry adds to the strategy"** iff 3 holds, reported whatever 1 and 2 give.

B and C are also reported separately, along with A.

## Also reported (descriptive)
- per test span and arm:
  - number of trades;
  - hit rate (share of positions with a net return > 0);
  - mean and median trade in bps;
  - monthly returns;
  - Sharpe of daily returns (annualised, sqrt 365);
  - maximum drawdown;
  - Calmar;
  - long/short split;
  - exit reasons;
  - buy-and-hold of the same span;
- the owner's target (hit > 60 %, drawdown < 5 %), marked met or not;
- the selection table of all 24 configurations.

## Tests (seconds)
- sigma at decision t uses bars < t only, and the frozen fit reads no bar after 2025-07-20 23:59. No bar from 2025-07-21..2025-09-30 is read.
- fill rules on hand-made bars:
  - a limit touched but not traded through does not fill;
  - a gap opens beyond the level;
  - a stop and a target in the same bar resolve to the stop;
  - the time stop exits at the open of entry + H;
  - only one position at a time;
- `choice.json` exists, and its commit precedes the commit of any test result. The run script refuses to score the tests without it.
- SHUF keeps each month's sigma multiset.

## Outputs
`runs/tactical/outlook/O8_volband_strategy/`: code, `selection.jsonl`, `choice.json`, `tests.jsonl`, `trades_*.parquet` or `.csv.gz`
(chosen configuration, every arm), `tables.txt`, the equity and drawdown figures (theme colours), `run.log`, `tests.log`, `REPORT.md`.
A new section in the outlook HTML report draws the strategy on real BTC windows (orders, fills, exits) from the selection and test spans.
