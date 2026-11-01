# Optimising the network and the trading strategy for PnL

| | |
|---|---|
| Date | 2026-11-01 |
| Task | Optimise the network and the trading strategy for net PnL |
| Setup chosen by the user | Binance USD-M BTCUSDT perp, standard fees (maker 2 bps, taker 5 bps; the primary round trip is about 7 bps, maker in and taker out). Funding paid. Shorts allowed. At most 1x exposure. Two tracks tested separately: Track 1 = 1/5/15-minute horizons; Track 2 = open search from 1 minute to 1 day. Both: fix neural_trade's `model.py` in place AND build a new PnL-trained network. Data 2020-01 to 2026-10, last 12 months sealed. |
| Design | Three independent designs (protocol, network and strategy angles), scored by one judge. The synthesis is frozen in `research/pnl/SPEC.md`. Every later change is in `research/pnl/CHANGES.log`. |
| Code and results | `research/pnl/` (simulator, features, folds, rules, ridge, PnL band net, gates, seal runner, tests); `research/pnl/results/` |
| Prepared by | Claude Code (AI) |

---

## 1. Answer

**At standard perp fees, no strategy or network passed the pre-registered tests. Both tracks deploy cash. The one lead with timing skill on dev failed on the sealed year.**

| | Track 1 (1-15 min) | Track 2 (1 min-1 day) |
|---|---|---|
| Dev gates (4.8 years walk-forward OOS; earlier work in this repository looked at the same years) | Nothing passes all five gates. T1-LIN passes four: timing p 0.001 (the floor of the test; T1-NET and T1-NT get the same), +7.7%/yr, Sharpe 1.1, 85% of it from 2021. It fails the stress gate. | Nothing passes. Best: T2-P, +12.7%/yr, Sharpe 0.35, p 0.14. Unscaled TSMOM has the higher Sharpe (0.39). |
| Deployed | **Cash** | **Cash** |
| Sealed year 2025-11..2026-10 (information only) | T1-LIN: **KILL**, -6.8 bps/day (about -25%/yr). Its taker variant (S1) also lost. | T2-P: **CONSISTENT**, +33% log, Sharpe 1.1, while a long perp lost 39% log after funding. **Not confirmed**: dev plus seal is not significant. **Not blind**: earlier work computed the trend family on the whole seal, and its 5 best days made more than its whole-year gain. |

1. **The fast rules lose, and the fast models do not hold up.**
   - **Fast rules.** The fades and the intraday trend lose at every fee tier, also at maker/maker and at a 0 bps maker fee. Their gross edge is -1 to +1 bps per trade, and a full-size round trip costs 7 bps.
   - **Fast models.** The models that trade a few hundred times a year made +7 to +9%/yr at C7-B on dev: T1-NET, T1-NT, and the ridge at 5-minute origins (T2-LIN, which chose 4-hour and daily horizons).
     - None of them passes the bootstrap test for net profit.
     - All three lose under the stress fills.
     - From 2022 on, T1-NET made +1.4%/yr and T1-NT -1.8%/yr. T2-LIN made +9.7%/yr, but with very large swings (Sharpe 0.16 over dev; -11.4 bps/day on the seal).
2. **The only lead with timing skill and a positive profit bound was crash-reversal timing, and it failed on the seal.**
   - On dev, the ridge forecaster (T1-LIN) faded extreme 1-minute moves about 34 times a year and earned +24 bps per trade.
   - But it needed fills inside the first minute, and 80% of its trades came from 2021.
   - On the seal, its frozen fold-S model traded 254 times and lost. That model chose H = 15 minutes and a low threshold; the dev folds used H = 1 or 5 and higher thresholds. So the seal killed the selection procedure. It did not test the dev configuration itself.
3. **The daily trend family behaves as trend-following does, and its edge is not proven.**
   - On dev, T2-P earned less than buy-and-hold (+12.7%/yr against +15.7%/yr) at about half the exposure. Its drawdown was 46% against 80%.
   - In 2025 (to October) T2-P lost 29% log while buy-and-hold gained 12%. At the end of dev it was still below its peak after 597 days.
   - In the sealed year BTC fell, the trend went short, and T2-P gained.
   - 5.8 years cannot show that it beats buy-and-hold or cash. Earlier work computed the trend family on 2022-01 to 2026-11, so neither T2-P result is a blind test (disclosed).
4. **The networks.**
   - **PnL band net.** The new PnL-trained band net passes its pipeline checks: it learns a planted signal of IC 0.1 and stays at cash on noise. On real data it found no reliable edge.
     - T1-NET: +8.2%/yr on dev, with a bootstrap lower bound below 0; about 0 on the seal.
     - T2-NET: +12.3%/yr on dev, not significant and fragile. +5.5 bps/day on the seal, but a constant short of its average size would have made 2.3 bps/day of that while BTC fell.
   - **neural_trade.** `model.py` got all 12 specified fixes, with tests. Its forecasts now have a small positive information coefficient (IC 0.02-0.05, falling over time). That is not enough to trade, and the pre-registered rule K2 retires it from trading.

**Recommendation.** Do not trade any of these candidates with money.
- If you continue, do only two cheap things:
  - **Paper-trade T2-P.** Paper-trade the daily trend rule (T2-P) for 12 months, as a risk choice against holding BTC. Write and commit the paper-trading test before the first paper trade, because this study did not pre-register one. One year can show skill only if the Sharpe is above about 2.5.
  - **Re-test crash reversals** only on tick or L2 data with real latency, as a new hypothesis.
- **Compare both with funding carry.** A funding-carry hurdle earned about 12%/yr on 2021-25 (more than half of it in 2021), about 4% in 2025 and about 3% in the sealed year. This study did not test whether either beats it.

---

## 2. What was built

### 2.1 Data and the seal

- **Data.** 1-minute BTCUSDT perp bars from data.binance.vision, 2020-01-01 to 2026-10-30 (3.59 million bars), and every funding print.
- **Development.** 2020-01 to 2025-10. Out-of-sample (OOS) scoring uses 2021-01-01 to 2025-10-31: 1,765 days, about 4.8 years, walk-forward.
- **Seal.** 2025-11-01 to 2026-10-30.
  - No code in this study could read the seal before we committed the pre-registration. `data.load` refuses unless the committed `prereg.json`, its SHA-256 and the run-once lock all exist.
  - Earlier work in this repository did study two windows inside the seal, and it computed the trend family over all of it (section 4).
- **Data flags.** The 2026-10 funding prints are rebuilt from the premium index (error about 0.02 bps per print). The 2026-10-28 kline file is missing (a 1-day gap in the seal).

### 2.2 The exact simulator (engine E)

Every reported PnL number, except the funding-carry reference, comes from one event-driven simulator on 1-minute bars (`sim.py`, engine E).

- **Decisions.** A decision uses data up to the close of bar t. The earliest fill is in bar t+1.
- **Maker fills.** A maker fill needs a strict trade-through by 1 tick within the next K bars (K = 1, 3 or 15).
  - Unfilled Track 1 entries are cancelled.
  - Track 2 orders chase to taker after 15 bars.
- **Taker fills** are at the next open.
- **Funding.** The position held at the end of the bar before each print pays it.
- **The 1x cap.** The simulator trims to 1x at |x| > 1.02 at a decision and at |x| > 1.10 at any close.
- **Fee scenarios.** C7 = maker in, taker out (primary). C4 = maker both ways. C10 = taker both ways.
- **Fill scenarios.** B = base. S = stress: 1 bps trade-through, +1 bar latency, +1 bps taker slippage.
- **Primary scenario.** C7-B. We never moved a decision to a cheaper scenario.

### 2.3 Candidates (pre-registered)

| Track | Candidate | What it is | Free parameters |
|---|---|---|---|
| 1 | T1-FADEV | Fade a top-2% 15-minute move when volatility is in its top third; maker in, taker out after 15 minutes | none |
| 1 | T1-LIN | Ridge forecast of the H-minute return from 42 intraday features; cost-aware hysteresis (enter if \|mu\| > theta x 8 bps) | 45 configurations, chosen in-fold by purged CV |
| 1 | T1-NET | PnL band net (below), h = 1/5/15 | 12 cells, chosen in-fold on inner validation |
| 1 | T1-NT | neural_trade after the fixes, through the same hysteresis (theta = 1) | h, chosen on its own validation |
| 2 | T2-P | Fixed daily trend: mean sign of the 7/14/28/56-day returns, 50% vol target | none |
| 2 | T2-LIN | Ridge on 55 features at 5-minute origins, H from 1 minute to 1 day | 90 configurations, chosen in-fold |
| 2 | T2-NET | PnL band net, h = 60/240/1440, plus the Track 1 cells | 24 cells, chosen in-fold |

Baselines run through the same simulator:
- cash, buy-and-hold (B&H);
- plain fade, intraday trend, unscaled TSMOM, vol-target long, 60-minute fade;
- a constant-exposure match for every candidate;
- a funding-carry reference (a sum of funding prints, not simulated).

### 2.4 The PnL band network

The network is trained on money, not on accuracy.

- **Heads.**
  - A target exposure tau = tanh(.).
  - A no-trade half-width b.
  - An auxiliary variance head.
- **Band layer.** The position moves only when it leaves [tau - b, tau + b]. The layer has an exact custom gradient, which passes gradcheck.
- **Loss.** A de-meaned daily mean-variance utility.
  - Exact costs: 3 bps per unit of increase and 5 bps per unit of decrease.
  - Funding is charged in the loss.
- **Variants and training.**
  - LIN (130-170 parameters) or MLP (5-6k parameters).
  - Cost multiplier kappa in {1, 2}.
  - 5 seeds per cell; early stopping on inner validation.
  - Cash if the inner-validation utility is <= 0.

### 2.5 Walk-forward and gates

- **Walk-forward.** Annual expanding folds: train up to the end of year Y-1, then trade year Y with the model frozen. All test years are chained into one continuous OOS run.
- **Gates.** A candidate is deployable only if it passes all five on dev OOS:
  - **G1.** Selection-adjusted timing skill, p <= 0.05. The test compares the timing with the same positions shifted in time, at the same costs, with a max over the track's candidates. It does not test whether net PnL is above 0.
  - **G2.** The bootstrap 95% lower bound of mean daily net > 0.
  - **G3.** It beats the track's gate comparators on point estimate:
    - Track 1: plain fade and intraday trend, on mean daily net.
    - Track 2: B&H, unscaled TSMOM and vol-target long, on Sharpe.
  - **G4.** It is positive in at least 3 of 5 years and in both halves.
  - **G5.** It is positive under the stress fills.
- **If nothing passes.** The track deploys cash. Its highest-t candidate is scored on the seal as information only.

---

## 3. Dev results (walk-forward OOS 2021-01-01..2025-10-31)

Annualised = mean daily net log return x 365. G1 p = the candidate's selection-adjusted timing p-value. Full statistics are in `research/pnl/results/gate.json`, `gate.md` and `dev_runs.json`.

### Track 1: dev OOS 2021-01-01..2025-10-31 (1,765 days), net of fees and funding

| strategy | C7-B %/yr | Sharpe C7-B | C7-S %/yr | C4-B %/yr | C10-B %/yr | trades/yr | mean x | G1 p | gates passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T1-FADEV | -75.5 | -3.12 | -157.8 | -51.2 | -105.1 | 1,211 | 0.00 | 1.000 | G3 |
| T1-LIN | 7.7 | 1.12 | -6.0 | 7.9 | 6.7 | 34 | 0.00 | 0.001 | G1, G2, G3, G4 |
| T1-NET | 8.2 | 0.53 | -8.6 | 10.9 | 4.9 | 642 | -0.05 | 0.001 | G1, G3, G4 |
| T1-NT | 7.1 | 0.35 | -45.4 | 16.5 | -5.0 | 444 | 0.01 | 0.001 | G1, G3 |
| T1-NET-W | 8.2 | 0.53 | -8.6 | 10.9 | 4.9 | 642 | -0.05 | - | G3, G4 |
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
- **The fades and the intraday trend lose at every fee tier.**
  - The vol-gated fade (T1-FADEV) earns +1.0 bps gross per trade and loses 6.0 bps net per trade (95% CI: a loss of 4.2 to 7.7 bps). That is a loss of about 76%/yr at C7.
  - At maker/maker (C4) it still loses 51%/yr. Even at a 0 bps maker fee it loses 4%/yr (`results/diag_fees.json`).
  - The plain fade and the intraday trend lose more.
  - These rules trade 1,200 to 8,000 times a year. A 7-8 bps round trip is far larger than their gross edge of -1 to +1 bps per trade.
- **T1-LIN showed timing skill on dev, but only with fills inside the first minute. It failed on the seal (section 4).**
  - The ridge forecaster trades only about 34 times a year. Mostly this is in extreme volatility: the median sigma at entry is about 11x its dev median.
  - It fades the last 1-5-minute move in 98% of trades. It nets +24 bps per trade (95% CI +9.6 to +38.7).
  - At C7-B it made +7.7%/yr on dev, with a Sharpe of 1.12 and a max drawdown of 5%.
  - Its statistics:
    - Timing-skill p: 0.001, the smallest p the test can give.
    - Bootstrap p for its net return: 0.009.
    - Deflated Sharpe after the 76 trials of Track 1: 0.52 (0.95 would be significant).
  - It loses 6.0%/yr under the stress fills (G5 fails).
  - **The cause is maker entries plus one bar of latency.** Delayed maker orders fill mostly when the crash continues (adverse selection) and miss the reversals. The 1 bps trade-through and the 1 bps slippage cost little.
  - **A post-hoc diagnostic on taker entries.**
    - With taker entries, the dev edge partly survived latency: +0.96 bps/day at C10 with one bar of latency, and +0.77 bps/day in the full C10-S stress.
    - The diagnostic is noisy: two bars of latency gave more than one bar (+1.71 bps/day).
    - On the seal the same idea (S1) lost 8.2 bps/day.
- **T1-NET (PnL band net) learns a little, not enough.**
  - +8.2%/yr at C7-B, Sharpe 0.53. Timing p 0.001; bootstrap p for net return 0.12, so G2 fails. It also loses under stress (G5 fails).
  - Its gross edge is about 2.8 bps per trade, net 1.4.
- **Most dev profit came in 2021.**
  - From 2022-01-01 on: T1-LIN made +1.4%/yr, T1-NET +1.4%/yr and T1-NT -1.8%/yr.
  - 2021 gave 85% of the T1-LIN dev net and 86% of the T1-NET dev net.
- **No Track 1 MLP was ever chosen** (details in 3.2).

**Track 2 (1 minute to 1 day).**
- **The trend family earns returns close to buy-and-hold, with smaller drawdowns. Nothing beats it with significance.**
  - T2-P (fixed 7-56-day trend, 50% vol target): +12.7%/yr, Sharpe 0.35, max drawdown 46%, at mean |x| 0.54.
  - Unscaled TSMOM: +17.2%/yr, Sharpe 0.39.
  - B&H: +15.7%/yr, Sharpe 0.25, max drawdown 80%.
  - None of these differences is significant: the paired CIs include 0.
  - T2-P fails G3 because unscaled TSMOM has the higher Sharpe.
  - Vol-target long had a 72% drawdown.
- **T2-LIN and T2-NET add nothing.**
  - Sharpe 0.16 and 0.28; selection-adjusted p 0.16 and 0.19.
  - T2-NET's result is fragile:
    - It held a net short position on average (mean x -0.33, mean |x| 0.59) through a bull market.
    - From 2022 on it lost 0.9%/yr.
    - The descriptive warm-up variant changed one fold's cell choice and turned +12.3%/yr into -2.0%/yr.
- **Carry reference.** A delta-neutral funding carry (sum of prints minus 30 bps/yr) earned about 57% log over the 4.8 years, about 12%/yr.
  - More than half of it (30% log) came in 2021. The later years gave about 4%, 8%, 12% and 4% (2025, to October).
  - It has no directional price risk, but it has basis, exchange and funding-sign risk.
  - The spot leg is not simulated, so this is a hurdle, not a result.

### 3.2 The networks

**PnL band net (new; `research/pnl/net.py`, `train.py`).**
- **Training.** 36 jobs: 2 tracks x 6 folds x 3 cadences, each with 4 cells x 5 seeds.
- **Track 1 cell choice.** The in-fold choice took LIN cells at h = 5 minutes in four of the five dev folds and h = 15 in one (2023). Fold S also took h = 5.
- **The MLP on Track 1.**
  - In the main run no dev-fold MLP cell traded. The MLPs froze at the flat start: the hard band passes no gradient while the position does not move. One fold-S cell traded slightly.
  - With a longer no-band warm-up (the declared variant, T1-NET-W), 6 of 30 dev-fold MLP cells traded. None beat the best LIN cell on inner validation, so no choice changed.
  - We cannot tell whether this means "no edge worth its cost" or a training limit.
- **Track 2.**
  - The in-fold choice took an MLP in every fold, mostly the daily one.
  - This study has no LIN-only Track 2 run, so it cannot say whether the MLP helped.
- **Pipeline checks.** The net learns a planted signal of IC 0.1 and stays flat on shuffled returns. The IC that neural_trade found on real data is 2-5 times smaller (0.02-0.05). So the checks do not show whether the market held too little signal or the net cannot find a weak one.

**neural_trade (`model.py`, fixed in place).**
- **Fixes.** All 12 specified fixes landed, with tests:
  - live indicators; the exact momentum-period inverse; LayerNorm; a unit-free 6-channel input;
  - unbiased losses (alpha 0.5, the median quantile, a 7 bps deadband with a deadband-aware alignment term);
  - no lambda calibration; purged validation with early stopping; no sequence cap; reproducible bundles; a causal `predict_at`.
- **Tests.** 62 tests pass.
  - One test fails and is kept at its bar, marked as an expected failure. A planted weak 5-minute signal is learned only to OOS IC 0.036, below the required 0.05.
  - So the model learns only part of a weak signal.
- **OOS skill, pooled over 2021-2025** (95% CIs from a 5-day-block bootstrap):

| | h = 1 min | h = 5 min | h = 15 min |
|---|---:|---:|---:|
| Spearman IC | +0.022 [0.019, 0.024] | +0.042 [0.039, 0.045] | +0.047 [0.043, 0.051] |
| IC by year 2021 / 22 / 23 / 24 / 25 | 0.042 / 0.020 / 0.027 / -0.005 / 0.011 | 0.061 / 0.047 / 0.051 / 0.032 / 0.029 | 0.058 / 0.040 / 0.067 / 0.045 / 0.035 |
| AUC (\|r\| >= 7 bps) | 0.517 | 0.528 | 0.532 |
| Share of minutes with \|mu\| > 8 bps | 0.3% | 2.1% | 6.7% |

- **Is the skill real?** The IC is small. Its CI is above 0 in 14 of the 15 fold-horizon cells. In 2024 at h = 1 it is below 0 (-0.005). The IC also falls over time.
- **Is it tradable?** No.
  - Its forecasts rarely clear the 8 bps cost bar. From 2023 on, almost none do.
  - Through the common hysteresis layer (T1-NT) it made +7.1%/yr at C7-B, but it lost 45%/yr under the stress fills, and it lost 1.8%/yr from 2022 on.
  - T1-LIN beat it on net, so the pre-registered rule K2 retires it from trading.
  - Its fold-S selection abstained, so it was flat on the seal.

---

## 4. The sealed test

**Protocol.**
1. We committed and pushed the pre-registration (`research/pnl/prereg.json`, `prereg.md`; commit 1630cd3). It holds:
   - both deployment decisions (cash) and the information-only candidates;
   - the verdict rules and one secondary hypothesis;
   - the SHA-256 of 36 data, feature and model files.
2. `run_seal.py` ran once.
   - Its guard checked a clean tree, the pre-registered code commit, the hashes of every file it loads, and that no lock or result existed.
   - Then it wrote the lock. Only after that did `data.load(seal=True)` open the seal.
   - The features recomputed on the full data were bit-identical to the cached dev features. This is a second causality proof.
3. `results/seal_access.log` holds one access line.

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

The clean seal is the 313 of the 363 seal days outside the two windows studied earlier (2025-11-01..11-10 and 2026-09-20..10-30). The long perp's -39% log includes about 3.5% of funding paid; the price itself fell about 35% log.

**Verdicts.** Both are information only, because both tracks deployed cash.
- **T1-LIN: KILL.**
  - The mean is below 0 on the full and on the clean seal; the bootstrap p is 0.92.
  - The fold-S model chose H = 15 minutes with theta 0.5, so it traded 254 times, not about 34.
  - The secondary hypothesis S1 (taker entries survive latency) is **not supported**: -8.2 bps/day, p 0.95.
  - The pre-registered drop rule covers only a deployed candidate. T1-LIN was information only, so this KILL is recorded as information. We drop it for this venue and fee tier on the same logic, not because a pre-registered rule requires it.
- **T2-P: CONSISTENT, not CONFIRMED.**
  - Its seal Sharpe of 1.10 is above the falsification bound (-1.46 full, -1.57 clean). Its seal PnL is positive (+33% log).
  - **This test is weak.**
    - One year can detect only a Sharpe above about 2.5.
    - Before the seal opened, the SPEC put the chance of a positive seal year at about 60-65%.
  - **The pooled bound is below 0.** The pooled dev-plus-seal lower bound is still below 0 (-1.55 bps/day), so the result is not confirmed.
  - **Read with care.**
    - BTC fell in the sealed year, which is the regime where trend-following gains.
    - The 5 best days of T2-P made +35% log, which is more than its +33% total. The other 358 days lost about 2%.
    - Unscaled TSMOM did as well (Sharpe 1.11). This is not independent support: on the seal its daily returns have a correlation of 0.995 with T2-P.
    - Earlier work computed the trend family on this period, and the designers may know the 2025-26 price path. Both are disclosed in the pre-registration.
  - Under the pre-registered rules, CONSISTENT means only "paper-trade for at least 12 months", never capital.

---

## 5. What this means for making money

**At standard fees, with execution judged on 1-minute bars, nothing in this study is ready to trade.** The answer to "optimise for PnL" is mostly a list of what to stop, and two cheap next steps.

1. **Stop trading fast signals at this fee tier.**
   - The fades and the intraday trend lose at every tier. Their gross edge is -1 to +1 bps per trade; a full-size round trip costs 7 bps.
   - Lower fees do not rescue the fades. Even a 0 bps maker fee leaves them negative.
   - **Lower fees do help the nets on dev.**
     - With a 0 bps maker fee both ways, T1-NET made +17.4%/yr (Sharpe 1.12; descriptive only).
     - This depends on maker fills: under the stress fills at C4 it lost 13.6%/yr. Check it on tick or L2 data before you rely on it.
2. **The crash-reversal lead (T1-LIN) is dropped at this venue and fee tier.**
   - On dev it was the only candidate that passed both the timing test and the bootstrap bound.
   - It lost on the seal, and so did its taker version.
   - If anyone revisits it, it must be a new hypothesis on tick data, with real latency (tens of milliseconds) and real spread and depth.
3. **For the daily horizon, choose by risk, not by skill.**
   - Trend-following (T2-P and unscaled TSMOM) gave returns close to buy-and-hold, with drawdowns of 46-51% instead of 80%.
   - Vol-target long had a 72% drawdown on dev, and on the seal it lost more than buy-and-hold.
   - None beat buy-and-hold or one another with significance in 4.8 years. The seal can only falsify, and it did not: T2-P was CONSISTENT in a falling year.
   - Holding trend-following BTC instead of cash or buy-and-hold is a risk choice. Its edge is not proven.
   - The next step is 12 months of paper trading, with a test that you commit before the paper trading starts.
4. **The PnL-trained network passes its pipeline checks, but it found no reliable edge at these horizons.**
   - T1-NET was about 0 on the seal.
   - T2-NET made +5.5 bps/day on the seal. A constant short of its average size would have made 2.3 bps/day of that while BTC fell.
   - We expect more data (other assets, other venues) or more information (order book, cross-venue basis) to matter more than more architecture. This study did not test that.
5. **Funding carry is the hurdle to beat.**
   - A delta-neutral carry earned about 12%/yr over 2021-2025 (more than half of it in 2021), about 4% in 2025 and about 3% in the sealed year.
   - It has no directional price risk, but it has basis and exchange risk, and its spot leg is not simulated.
   - Any active strategy must beat it after costs.

---

## 6. Process, reviews and deviations

- **Design.**
  - Three independent designs went to one judge. We froze the synthesis in `SPEC.md` before this study modelled any data.
  - Earlier work in this repository had already looked at the dev data, and at the trend family over 2022-01 to 2026-11 (pre-registration disclosures).
  - Every change after that is in `CHANGES.log`, with its reason and with whether results existed at the time.
- **Two code reviews.**
  - **First review.** It ran before any full dev result, with independent reviewers and an adversarial verifier for each finding. Its fixes are in `CHANGES.log`:
    - 10 simulator, policy, check and net defects;
    - the statistics and seal-protocol fixes;
    - the Deflated Sharpe input.
    - The most important: the exposure used by the skill test was one bar behind the position that earns each bar.
  - **Second review.** This focused review ran after we computed the first dev gates (commit edd8e88) and before the full rerun. It confirmed 4 more defects.
  - **Our own finds.** We also found residue orders, size thresholds under ruin, a stray position under C4, and three mis-specified checks.
  - **Tests.** The simulator, policy and net defects were fixed with regression tests; 61 simulator, statistics, metrics and net tests pass. The three checks were corrected in `checks.py`.
- **One documented full rerun** (of the 2 allowed).
  - The cause was the C4 stray-position bug, found after the first full dev run. It had made the fades look break-even at maker/maker.
  - No candidate's C7 row changed. The B&H and intraday-trend comparator rows moved slightly (B&H from +15.55 to +15.69%/yr), and no gate result changed.
- **Harness checks (all pass).**
  - Causality: 20 random cuts on one 150k-bar window, noise and permutation. The SPEC asked for 1,000 cuts; this reduction is now noted in `CHANGES.log`.
  - Funding alignment.
  - The purged fold manifest.
  - A deliberate one-bar leak gives +1,450 bps/day, against -0.5 honest.
  - Shuffled labels give 0 within 2 SE.
  - Seal truncation.
  - Reconciliation of the skill test with engine E: B&H 0.007%/yr and T2-P 0.16%/yr. The plain fade shows 5.4%/yr. That is the cost of late maker fills, which the per-bar exposure path cannot see.
- **Network pipeline checks.**
  - A planted IC 0.1 signal is learned (held-out net is positive for LIN and MLP).
  - Shuffled returns give cash.
  - The band gradient passes gradcheck on all branches.
- **Not done or limited.**
  - Fills at bar resolution: no queue position and no L2.
  - One asset (BTC) and one venue.
  - The learnable-indicator variant of the net was pre-declared to run only if a net passed all gates. None did.
  - **neural_trade seal path.** The neural_trade adapter has no seal-reading path: the permission system blocked adding one. It was not needed, because T1-NT's fold-S model abstained (cash).
