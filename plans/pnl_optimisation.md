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

**At standard perp fees, no strategy or network passed the pre-registered tests. Both tracks deploy cash. The one skilled lead found on dev failed on the sealed year.**

| | Track 1 (1-15 min) | Track 2 (1 min-1 day) |
|---|---|---|
| Dev gates (4.8 years OOS) | Nothing passes all five. T1-LIN passes four: real timing skill (p 0.001), +7.7%/yr, Sharpe 1.1. It fails the stress gate. | Nothing passes. Best: T2-P, +12.7%/yr, Sharpe 0.35, p 0.14; unscaled TSMOM is higher (Sharpe 0.39). |
| Deployed | **Cash** | **Cash** |
| Sealed year (2025-11..2026-10, information only) | T1-LIN: **KILL** (-6.8 bps/day; about -25%/yr). Its taker variant (S1) also lost. | T2-P: **CONSISTENT** (+33% log, Sharpe 1.1, while BTC lost 39% log). **Not confirmed**: dev plus seal is not significant. |

1. **Dense short-horizon trading cannot pay for itself here.**
   - Every 1-15-minute rule that trades often loses after costs. This holds for fades, trend, ridge at 5-minute cadence and the neural nets.
   - The loss holds at maker/maker fees and even at a 0 bps maker fee.
   - The gross edge is 0-3 bps per trade; a round trip costs 7-8 bps.
2. **The only skill found was crash-reversal timing, and it did not survive.**
   - On dev the ridge forecaster faded extreme 1-minute moves about 34 times a year and earned +24 bps per trade.
   - But it needed fills inside the first minute, and 80% of its trades came from 2021.
   - On the seal, its frozen model traded 254 times and lost.
3. **The daily trend family behaves as expected and is not proven skill.**
   - It earned about as much as buy-and-hold on dev, with smaller drawdowns (47% against 80%).
   - In the sealed year BTC fell, the trend went short, and T2-P gained.
   - That fits trend-following, but 5.8 years cannot show that it beats buy-and-hold or cash with significance. The trend family was also seen earlier in this repository (disclosed).
4. **The networks.**
   - The new PnL-trained band net works as a method: it learns a planted signal and refuses to trade noise. On real data it found too little to trade on: +8.2%/yr on dev, not significant, about 0 on the seal.
   - neural_trade's `model.py` got all 12 specified fixes plus tests. Its forecasts now have a small positive information coefficient (IC 0.02-0.05). That is not enough to trade, and it is retired from trading by the pre-registered rule K2.

**Recommendation.** Do not trade any of these candidates with money.
- If you continue, do only two cheap things:
  - Paper-trade the daily trend rule (T2-P) for 12 months, with its pre-registered Sharpe test, as a risk choice against holding BTC.
  - Re-test crash reversals only on tick or L2 data with real latency.
- Neither is likely to beat a simple funding-carry hurdle (about 12%/yr on 2021-25, 4% in 2025) by enough to matter.


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

Annualised = mean daily net log return x 365. G1 p = the candidate's selection-adjusted skill p-value. Full statistics: `research/pnl/results/gate.json`, `gate.md`, `dev_runs.json`.

### Track 1: dev OOS 2021-01-01..2025-10-31 (1,765 days), net of fees and funding

| strategy | C7-B %/yr | Sharpe C7-B | C7-S %/yr | C4-B %/yr | C10-B %/yr | trades/yr | mean x | G1 p | gates passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T1-FADEV | -75.5 | -3.12 | -157.8 | -51.2 | -105.1 | 1,211 | 0.00 | 1.000 | G3 |
| T1-LIN | 7.7 | 1.12 | -6.0 | 7.9 | 6.7 | 34 | 0.00 | 0.001 | G1, G2, G3, G4 |
| T1-NET | 8.2 | 0.52 | -8.6 | 10.9 | 4.9 | 642 | -0.05 | 0.001 | G1, G3, G4 |
| T1-NT | 7.1 | 0.35 | -45.4 | 16.5 | -5.0 | 444 | 0.01 | 0.001 | G1, G3 |
| T1-NET-W | 8.2 | 0.52 | -8.6 | 10.9 | 4.9 | 642 | -0.05 | - | G3, G4 |
| plain_fade | -214.8 | -9.21 | -328.9 | -147.5 | -303.5 | 3,242 | -0.00 | - | - |
| intraday_trend | -674.2 | -11.56 | -1,010.0 | -533.0 | -861.3 | 8,131 | 0.02 | - | - |
| B&H | 15.7 | 0.25 | 15.7 | 15.7 | 15.7 | 0 | 1.04 | - | - |
| cash | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | - | 0.00 | - | - |

p_sel (track) = 0.001; deployed: **CASH**; information only: T1-LIN

### Track 2: dev OOS 2021-01-01..2025-10-31 (1,765 days), net of fees and funding

| strategy | C7-B %/yr | Sharpe C7-B | C7-S %/yr | C4-B %/yr | C10-B %/yr | trades/yr | mean x | G1 p | gates passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T2-P | 12.7 | 0.35 | 11.1 | 13.5 | 11.8 | 37 | 0.11 | 0.141 | G4, G5 |
| T2-LIN | 9.3 | 0.16 | -9.8 | 19.2 | 0.4 | 449 | 0.17 | 0.158 | G4 |
| T2-NET | 12.3 | 0.28 | 7.3 | 14.0 | 10.6 | 98 | -0.33 | 0.189 | G5 |
| T2-NET-W | -2.0 | -0.04 | -14.1 | 0.3 | -4.9 | 144 | -0.34 | - | none |
| tsmom_unscaled | 17.2 | 0.39 | 15.1 | 18.1 | 16.2 | 37 | 0.11 | - | - |
| voltarget_long | 15.3 | 0.33 | 15.3 | 15.3 | 15.3 | 0 | 0.82 | - | - |
| fade60 | -56.9 | -2.11 | -103.0 | -45.0 | -70.2 | 588 | 0.00 | - | - |
| B&H | 15.7 | 0.25 | 15.7 | 15.7 | 15.7 | 0 | 1.04 | - | - |
| cash | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | - | 0.00 | - | - |

p_sel (track) = 0.141; deployed: **CASH**; information only: T2-P

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


### 3.2 The networks

**PnL band net (new, `research/pnl/net.py`, `train.py`).**
- **Training.** 36 jobs: 2 tracks x 6 folds x 3 cadences, each with 4 cells x 5 seeds.
- **Track 1.**
  - The in-fold choice took LIN cells at h = 5 minutes in four folds and h = 15 in one, from inner-validation utility.
  - Every MLP cell stayed at cash, also in the declared variant with a longer no-band warm-up.
  - With trading costs inside the loss, the MLP learned target exposures close to 0. That is the net saying "no edge worth its cost".
- **Track 2.**
  - The MLP cells did trade, and the in-fold choice mostly took the daily MLP.
  - The dev result (+12.3%/yr) was fragile: one changed cell choice in one fold (the variant) turned it into -2.0%/yr.
- **Pipeline checks.** The planted signal is learned and the shuffled null stays flat, so the method is sound. The market data just held too little signal for it.

**neural_trade (`model.py`, fixed in place).**
- **Fixes.** All 12 specified fixes landed with tests:
  - live indicators; exact momentum-period inverse; LayerNorm; unit-free 6-channel input;
  - unbiased losses (alpha 0.5, median quantile, 7 bps deadband with a deadband-aware alignment term);
  - no lambda calibration; purged validation with early stopping; no sequence cap; reproducible bundles; a causal `predict_at`.
- **Tests.** 62 tests pass. One test fails and is kept at its bar (marked as an expected failure): a planted weak 5-minute signal is learned to OOS IC 0.036, below the required 0.05. The model learns only part of a weak signal.
- **Measured skill.** OOS skill pooled over 2021-2025 (95% CIs from a 5-day-block bootstrap):

| | h = 1 min | h = 5 min | h = 15 min |
|---|---:|---:|---:|
| Spearman IC | +0.022 [0.019, 0.024] | +0.042 [0.039, 0.045] | +0.047 [0.043, 0.051] |
| AUC (\|r\| >= 7 bps) | 0.517 | 0.528 | 0.532 |
| Share of minutes with \|mu\| > 8 bps | 0.3% | 2.1% | 6.7% |

- **Is the skill real?** The IC is small but its CI excludes 0 in almost every fold. That is better than the "indistinguishable from 0" the design expected.
- **Is it tradable?** No.
  - Its forecasts rarely clear the 8 bps cost bar. From 2023 on almost none do.
  - Through the common hysteresis layer (T1-NT) it made +7.1%/yr at C7-B but -45%/yr under the stress fills.
  - T1-LIN beat it on net, so the pre-registered rule K2 retires it from trading use.
  - Its fold-S selection abstained, so it was flat on the seal.

---

## 4. The sealed test

**Protocol.** These steps ran in this order:
1. The pre-registration (`research/pnl/prereg.json`, `prereg.md`) was committed and pushed (commit 1630cd3). It held both deployment decisions (cash), the information-only candidates, the verdict rules, one secondary hypothesis and the SHA-256 of 36 data, feature and model files.
2. `run_seal.py` ran once.
   - Its guard checked a clean tree, the pre-registered code commit, the hashes of every file it loads, and that no lock or result existed. It then wrote the lock, and only after that did `data.load(seal=True)` open the seal.
   - The features recomputed on the full data were bit-identical to the cached dev features (a second causality proof).
3. One access line is in `results/seal_access.log`.

| strategy (frozen fold-S model) | C7-B bps/day (full) | C7-B bps/day (clean) | Sharpe | C7-S | C10-S | trades |
|---|---:|---:|---:|---:|---:|---:|
| T1-LIN (information only) | -6.81 | -7.72 | -1.73 | -11.97 | -8.19 | 254 |
| T1-NET | +0.24 | +0.79 | 0.06 | -1.27 | -0.99 | 372 |
| T1-FADEV | -19.99 | -19.49 | -4.79 | -35.59 | -35.62 | 1,157 |
| T1-NT (abstained: cash) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| T2-P (information only) | +9.14 | +8.94 | 1.10 | +8.74 | +8.65 | 44 |
| TSMOM unscaled | +9.48 | +9.34 | 1.11 | +9.08 | +8.96 | 44 |
| T2-NET | +5.50 | +5.96 | 0.57 | +3.77 | +3.88 | 100 |
| T2-LIN | -11.40 | -10.02 | -1.09 | -10.57 | -13.83 | 187 |
| Vol-target long | -12.25 | -11.32 | -1.13 | -12.22 | -12.25 | 1 |
| B&H perp | -10.62 | -9.37 | -0.87 | -10.62 | -10.61 | 1 |

Clean seal = the seal without the two windows studied earlier (about 313 days).

**Verdicts (both information only, because both tracks deployed cash):**
- **T1-LIN: KILL.**
  - The mean is below 0 on the full and the clean seal; the bootstrap p is 0.92.
  - The fold-S model chose H = 15 minutes with a low threshold, so it traded 254 times, not about 34.
  - The edge did not carry over.
  - The secondary hypothesis S1 (taker entries survive latency) is **not supported**: -8.2 bps/day, p 0.95.
  - Under the pre-registered kill rule, T1-LIN is dropped for this venue and fee tier.
- **T2-P: CONSISTENT, not CONFIRMED.**
  - Its seal Sharpe of 1.10 is far above the falsification bound (-1.46 full, -1.57 clean), and its seal PnL is positive (+33% log).
  - The pooled dev-plus-seal lower bound is still below 0 (-1.55 bps/day), so it is not confirmed.
  - Read with care:
    - BTC fell in the sealed year, which is the regime where trend-following gains.
    - Unscaled TSMOM did as well (Sharpe 1.11).
    - The trend family was computed on this period in earlier work, and the designers may know the 2025-26 price path (both disclosed in the pre-registration).
  - Under the pre-registered rules, CONSISTENT means only "paper-trade for at least 12 months", never capital.


---

## 5. What this means for making money

**At standard fees, with execution judged on 1-minute bars, nothing in this study is ready to trade.** The answer to "optimise for PnL" is therefore mostly a list of what to stop and one lead to test properly.

1. **Stop trading dense 1-15-minute signals at this fee tier.**
   - Every rule that trades hundreds or thousands of times a year loses: fades, trend and ridge at 5-minute cadence.
   - The gross edge is 0-3 bps per trade; the round trip is 7-8 bps.
   - Lower fees do not rescue the fades. Even a 0 bps maker fee leaves them negative.
2. **The crash-reversal lead (T1-LIN) is killed at this venue and fee tier.** It lost on the seal, and so did its taker version. If anyone revisits it, it must be a new hypothesis on tick data, not a strategy.
   - On dev it was the only result with selection-adjusted skill (p 0.001) and a positive bootstrap bound.
   - But it is rare (about 34 trades a year), concentrated in one crash regime (2021), and it needs execution within the first minute.
   - The next step is concrete:
     - (a) Download aggTrades/bookTicker for the 2021-2025 crash days.
     - (b) Re-test the taker-entry version with real latency (tens of milliseconds, not a 1-minute bar) and real spread and depth.
     - (c) If it holds, paper-trade it with a pre-registered sequential test.
   - Expect few trades and long gaps between them.
3. **For the daily horizon, choose by risk, not by skill.**
   - Trend-following and vol-targeting gave buy-and-hold-like returns with drawdowns of 47-51% instead of 80%.
   - None beat buy-and-hold or one another with any significance in 4.8 years. The seal can only falsify, and it did not: T2-P was CONSISTENT in a falling year.
   - Holding trend-following BTC instead of cash or buy-and-hold is a risk preference with a plausible but unproven edge. The honest next step is 12 months of paper trading with the pre-registered test.
4. **The PnL-trained network works as a method, but the data gave it nothing to learn at these horizons** (about 0 on the seal for T1-NET; +5.5 bps/day for T2-NET, in line with the trend family).
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
  - The first review confirmed 15 defects; the second, focused review confirmed 4 more.
  - The lead found several others: residue orders, size thresholds under ruin, a stray position under C4, and three mis-specified checks.
  - All were fixed with regression tests (61 simulator, statistics and net tests pass).
  - The most important was the exposure used by the skill test: it was one bar behind the position that earns each bar.
  - The C4 stray-position bug was found after the first full dev run. It had made the fades look break-even at maker/maker. It used one documented full rerun (of 2 allowed); the C7 gate rows did not change.
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

