# Optimising the network and the trading strategy for PnL

| | |
|---|---|
| Date | 2026-11-01 |
| Task | Optimise the network and the trading strategy for net PnL |
| Setup chosen by the user | Binance USD-M BTCUSDT perp, standard fees (maker 2 bps, taker 5 bps; primary round trip about 7 bps = maker in, taker out). Funding paid. Shorts allowed. At most 1x exposure. Two tracks tested separately: Track 1 = 1/5/15-minute horizons, Track 2 = open search from 1 minute to 1 day. Both: fix neural_trade's `model.py` in place AND build a new PnL-trained network. Data 2020-01 to 2026-10, last 12 months sealed. |
| Design | Three independent designs scored by a judge panel; the synthesis is frozen in `research/pnl/SPEC.md`. Every later change is in `research/pnl/CHANGES.log`. |
| Code | `research/pnl/` (simulator, features, folds, rules, ridge, PnL band net, gates, seal runner, tests) |
| Prepared by | Claude Code (AI) |

---

## 1. Answer

__ANSWER__

---

## 2. What was built

### 2.1 Data and the seal

- **Data.** 1-minute BTCUSDT perp bars from data.binance.vision, 2020-01-01 to 2026-10-30 (3.59 million bars), and every funding print.
- **Development period.** 2020-01 to 2025-10. Out-of-sample (OOS) scoring uses 2021-01-01 to 2025-10-31: 1,765 days, about 4.8 years.
- **Seal.** 2025-11-01 to 2026-10-30. No code could read it before the pre-registration was committed: `data.load` refuses unless the committed `prereg.json`, its SHA-256 and the run-once lock all exist.
- **Data flags.** The 2026-10 funding prints are rebuilt from the premium index (error about 0.02 bps per print). The 2026-10-28 kline file is missing (a 1-day gap in the seal).

### 2.2 The exact simulator (engine E)

Every reported number comes from one event-driven simulator on 1-minute bars (`sim.py`).

- **Decisions** use data up to the close of bar t. The earliest fill is in bar t+1.
- **Maker fills** need a strict trade-through by 1 tick in the next K bars (K = 1, 3 or 15). Unfilled Track 1 entries are cancelled. Track 2 orders chase to taker after 15 bars.
- **Taker fills** are at the next open.
- **Funding** is charged to the position held at the end of the bar before each print.
- **The 1x cap** is enforced with trims at |x| > 1.02 at a decision and > 1.10 at any close.
- **Scenarios.** Three fee scenarios: C7 (maker in, taker out; primary), C4 (maker both ways), C10 (taker both ways). Two fill scenarios: B (base) and S (stress: 1 bps trade-through, +1 bar latency, +1 bps taker slippage).
- **Primary scenario.** C7-B. The decision was never moved to a cheaper scenario.

### 2.3 Candidates (pre-registered)

| Track | Candidate | What it is | Free parameters |
|---|---|---|---|
| 1 | T1-FADEV | Fade a top-2% 15-minute move when volatility is in its top third; maker in, taker out after 15 minutes | none |
| 1 | T1-LIN | Ridge forecast of the H-minute return from 42 intraday features; cost-aware hysteresis (enter if \|mu\| > theta x 8 bps) | 45 configurations chosen in-fold by purged CV |
| 1 | T1-NET | PnL band net (below), h = 1/5/15 | 12 cells chosen in-fold on inner validation |
| 1 | T1-NT | neural_trade after the fixes, through the same hysteresis (theta = 1) | h chosen on its own validation |
| 2 | T2-P | Fixed daily trend: mean sign of the 7/14/28/56-day returns, 50% vol target | none |
| 2 | T2-LIN | Ridge on 55 features at 5-minute origins, H from 1 minute to 1 day | 90 configurations chosen in-fold |
| 2 | T2-NET | PnL band net, h = 60/240/1440 plus the Track 1 cells | 24 cells chosen in-fold |

Baselines run through the same simulator: cash, buy-and-hold (B&H), plain fade, intraday trend, TSMOM unscaled, vol-target long, 60-minute fade, a constant-exposure match for every candidate, and a carry reference.

### 2.4 The PnL band network

The network is trained on money, not on accuracy.

- **Heads.** Target exposure tau = tanh(.), a no-trade half-width b, and an auxiliary variance head.
- **Band layer.** The position moves only when it leaves [tau - b, tau + b]. It has an exact custom gradient, checked by gradcheck.
- **Loss.** A de-meaned daily mean-variance utility. Exact costs are charged per unit of increase (3 bps) and decrease (5 bps). Funding is charged in the loss.
- **Variants.** LIN (130-170 parameters) or MLP (5-6k parameters), cost multiplier kappa in {1, 2}, 5 seeds per cell, early stopping on inner validation, abstain (cash) if the inner-validation utility is <= 0.
- **Pipeline checks passed.**
  - A planted signal (IC 0.1) is learned and makes held-out money for LIN and MLP.
  - On shuffled returns the net stays at cash.

### 2.5 Walk-forward and gates

- **Walk-forward.** Annual expanding folds: train up to the end of year Y-1, then trade year Y with the model frozen. All test years are chained into one continuous OOS run.
- **Gates.** A candidate is deployable only if it passes all five gates on dev OOS:
  - G1: selection-adjusted timing skill, p <= 0.05. This is a circular-shift null with max-t over the track's candidates.
  - G2: the bootstrap 95% lower bound of mean daily net > 0.
  - G3: it beats the track's baselines on point estimate.
  - G4: positive in at least 3 of 5 years and in both halves.
  - G5: positive under the stress fills.
- **If nothing passes.** The track deploys cash. Its highest-t candidate is scored on the seal as information only.

---

## 3. Dev results (OOS 2021-01-01..2025-10-31)

__DEV_TABLES__

### 3.1 What the numbers say

**Track 1 (1-15 minutes).**
- **Dense intraday rules lose at every fee tier.**
  - The vol-gated fade (T1-FADEV) earns +1.0 bps gross per trade and loses 6.0 bps net per trade (95% CI -7.7 to -4.2), about -76%/yr at C7.
  - At maker/maker (C4) it still loses -51%/yr. Even at a 0 bps maker fee it loses -4%/yr (`results/diag_fees.json`).
  - The plain fade and the intraday trend are worse.
  - These rules trade 1,200-8,000 times a year. A 7-8 bps round trip is far larger than their 0-2 bps gross edge.
- **T1-LIN has real timing skill, but only inside the first minute.**
  - The ridge forecaster trades only about 34 times a year, almost all in extreme volatility: at entry, sigma is about 11x its dev median.
  - It fades the last 1-5 minute move in 98% of trades. It nets +24 bps per trade (95% CI +9.6 to +38.7).
  - At C7-B it makes +7.7%/yr with a Sharpe of 1.12, a max drawdown of 5% and a selection-adjusted p of 0.001.
  - It loses -6.0%/yr under the stress fills (G5 fails).
  - The decomposition shows the cause: maker entries plus one bar of latency. Delayed maker orders fill mostly when the crash continues (adverse selection) and miss the reversals. The 1 bps trade-through and the 1 bps slippage cost little.
  - With taker entries the edge partly survives latency: +0.96 bps/day at C10 with one bar of latency, and +0.77 bps/day in the full C10-S stress.
  - **Caveat: concentration.** 129 of its 162 trades fall in 2021, mostly around the May 2021 crash. It made no trades in 2022.
- **T1-NET (PnL band net) learns a little, not enough.**
  - All in-fold choices were LIN cells at h = 5 or 15 minutes. +8.2%/yr at C7-B, Sharpe 0.52, p 0.001.
  - Its bootstrap lower bound is below 0 (G2 fails), and it loses under stress (G5 fails).
  - Its gross edge is about 2.8 bps per trade, net 1.4.
- **Every Track 1 MLP stayed at cash.** The declared extended-warm-up variant changed nothing (T1-NET-W = T1-NET). With costs in the loss, the MLP learns tau close to 0. That is a valid "no edge worth its cost" answer, not a training failure.

**Track 2 (1 minute to 1 day).**
- **The trend family earns about what buy-and-hold earns, with smaller drawdowns, and nothing beats it.**
  - T2-P (fixed 7-56-day trend, 50% vol target): +12.7%/yr, Sharpe 0.35, max drawdown 47%.
  - Unscaled TSMOM: +17.2%/yr, Sharpe 0.39. B&H: +15.6%/yr, Sharpe 0.25, max drawdown 80%.
  - None of these differences is significant (paired CIs include 0).
  - T2-P fails G3 because unscaled TSMOM has the higher Sharpe.
- **T2-LIN and T2-NET add nothing.**
  - Sharpe 0.16 and 0.28; selection-adjusted p 0.16 and 0.19.
  - T2-NET's result is fragile. It held a net short position on average (mean x -0.33) through a bull market. The descriptive warm-up variant changed one fold's cell choice and turned +12.3%/yr into -2.0%/yr.
- **Carry reference.** A delta-neutral funding carry (sum of prints minus 30 bps/yr) earned about 57% log over the 4.8 years, about 12%/yr. It has no price risk, but it shrank to about 4% in 2025. The spot leg is not simulated, so this is a hurdle, not a result.


---

## 4. The sealed test

__SEAL__

---

## 5. What this means for making money

**At standard fees, with execution judged on 1-minute bars, nothing in this study is ready to trade.** The answer to "optimise for PnL" is therefore mostly a list of what to stop and one lead to test properly.

1. **Stop trading dense 1-15-minute signals at this fee tier.**
   - Every rule that trades hundreds or thousands of times a year loses: fades, trend and ridge at 5-minute cadence.
   - The gross edge is 0-3 bps per trade; the round trip is 7-8 bps.
   - Lower fees do not rescue the fades. Even a 0 bps maker fee leaves them negative.
2. **Treat the crash-reversal lead (T1-LIN) as a hypothesis for tick data, not as a strategy.**
   - It is the only result with selection-adjusted skill (p 0.001) and a positive bootstrap bound.
   - But it is rare (about 34 trades a year), concentrated in one crash regime (2021), and it needs execution within the first minute.
   - The next step is concrete:
     - (a) Download aggTrades/bookTicker for the 2021-2025 crash days.
     - (b) Re-test the taker-entry version with real latency (tens of milliseconds, not a 1-minute bar) and real spread and depth.
     - (c) If it holds, paper-trade it with a pre-registered sequential test.
   - Expect few trades and long gaps between them.
3. **For the daily horizon, choose by risk, not by skill.**
   - Trend-following and vol-targeting gave buy-and-hold-like returns with drawdowns of 47-51% instead of 80%.
   - None beat buy-and-hold or one another with any significance in 4.8 years, and the seal cannot confirm a Sharpe below about 2.5.
   - Holding trend or vol-targeted BTC instead of cash is a risk preference, not a found edge.
4. **The PnL-trained network works as a method, and the data gave it nothing to learn at these horizons.**
   - It learns a planted signal and refuses to trade noise. On real data it chose small or flat positions.
   - More capacity (MLP) did not help.
   - More data (other assets, other venues) or more information (order book, cross-venue basis) is the lever, not more architecture.
5. **Funding carry is the hurdle to beat.** A delta-neutral carry earned about 12%/yr over 2021-2025 with no price risk, but only about 4% in 2025. Any active strategy must beat it after costs.


---

## 6. Process, reviews and deviations

- **Design.**
  - Three independent designs went to a judge panel, and the synthesis was frozen in `SPEC.md` before any data was modelled.
  - Every change after that is in `CHANGES.log`, with its reason and with whether results existed at the time.
- **Two code reviews before the gates were computed.** Each had independent reviewers, and each finding went to an adversarial verifier.
  - 15 defects were confirmed and fixed with regression tests (61 tests pass).
  - The most important was the exposure used by the skill test: it was one bar behind the position that earns each bar.
  - One bug was found after the first full dev run: under C4 the cap trim ignored a resting maker exit, so it made the fades look break-even at maker/maker. It used the one documented full rerun; the C7 rows did not change.
- **Harness checks (all pass).**
  - Causality: 20 random cuts, noise and permutation.
  - Funding alignment.
  - Purged fold manifest.
  - A deliberate one-bar leak gives +1,450 bps/day against -0.5 honest.
  - Shuffled labels give 0 within 2 SE.
  - Seal truncation.
  - Reconciliation of the skill test with engine E: B&H 0.007%/yr, T2-P 0.16%/yr. The plain fade shows 5.4%/yr; this is the cost of late maker fills, which the per-bar exposure path cannot see.
- **Network pipeline checks.**
  - A planted IC 0.1 signal is learned (positive held-out net for LIN and MLP).
  - Shuffled returns give cash.
  - The band gradient passes gradcheck on all branches.
- **Not done or limited.**
  - Fills at bar resolution: no queue position, no L2.
  - One asset (BTC), one venue.
  - The learnable-indicator variant of the net was pre-declared to run only if a net passed all gates; none did.

