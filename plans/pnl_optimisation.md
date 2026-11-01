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

__DEV_FINDINGS__

---

## 4. The sealed test

__SEAL__

---

## 5. What this means for making money

__MONEY__

---

## 6. Process, reviews and deviations

__PROCESS__
