# Strategy architectures for this model's outputs

Research note, 2026-09-29, branch `nt-005-strategy-study`. The owner asked: "Evaluate the trading strategy
structure. Research best trading strategy architectures targeting our model data." This is the research
half: which strategy architectures to implement and test, how to pick one honestly, and what the engine
needs. It implements nothing, ran no GPU job and used no reference test-block number to choose anything.
Every number computed here is labelled with its data; estimates are labelled as estimates.

## Summary

**Directional trading of this model at 10-20 minutes cannot break even at 26 bps round trip.** The cost is
1.0-1.3 standard deviations of the 10-20 minute move (sd 20-27 bps on the reference file before the test
block). To break even the forecast needs a signal-return correlation of at least 1.2 when it trades on every
bar, which is impossible. If it trades only its 2% most confident bars, it still needs about 0.36-0.49. The
model has about 0.01-0.015 (test AUC 0.500-0.507). The best simple baseline has about 0.04-0.12 on its
best folds. So the model is short by a factor of 25 or more, and the gap is structural. Stop and take-profit
geometry cannot close it (section 1.4). Longer holds lower the bar (about 0.18 correlation at one day on
every bar). But the model makes no forecasts at those horizons, and simple momentum or reversal rules on
2022-2025 BTC show no gross edge above about 1 bp per trade at 15-240 minutes.

**The variance head is the model's only edge, and a trivial forecaster may beat it.** An EWMA of squared
1-minute returns gets a CRPSS of +0.06 to +0.07 against a constant variance on the reference file before its
test block. The model gets +0.009 to +0.020 on its test block. These are different blocks, so this is a
warning, not a verdict (section 2.0). Until a same-block comparison exists, every variance-driven
architecture below must run twice: once with the model's sigma and once with an EWMA sigma (its
"model-free twin").

**Ranked shortlist** (details in section 3). K counts the candidate configurations that compete for the
winner. The twins and the baselines do not compete.

| rank | architecture | model output used | engine mode | K |
|---|---|---|---|---|
| 1 | Volatility-targeted long exposure: e = min(1, sigma* / sigma_hat), hourly decisions, rebalancing band | sigma (h2) | target-exposure (new) | 2 |
| 2 | Volatility-regime stand-aside on buy-and-hold: long, flat while sigma_hat is above a cal quantile, with hysteresis | sigma (h2) | today's engine | 2 |
| 3 | Net-edge fractional Kelly aim with a no-trade band (Garleanu-Pedersen / de Lataillade et al. form) | P(up) and sigma (mu_hat = sigma x Phi^-1(p)); served delta as an option once NT-007 decides | target-exposure (new); discrete twin in today's engine | 2 |
| 4 | Variance-gated classic TA primaries (a meta-labelling filter with a one-feature secondary) | sigma (gate) | today's engine | 4 |
| 5 | Meta-labelling with a fitted secondary classifier | P(up), sigma, intervals as features | today's engine | deferred: the calibration block holds too few labels (section 2.6) |
| 6 | Volatility-scaled time-series momentum at daily lookbacks | none (a manual baseline) | target-exposure | deferred: needs the long-history folds (NT-041) |

Also on the leaderboard: the incumbent `calibrated_quantile` (1 row, K = 1), for 11 candidates in all. The
baselines: buy-and-hold, always-flat, the size-matched random null (discrete) or a circular-shift timing null
(exposure), and the EWMA twins of ranks 1, 2 and 4 (8 rows).

**Expected outcome (an estimate).** A clear negative result is the most likely outcome. Ranks 1 and 2 are
long-biased overlays on buy-and-hold. The literature says they reduce drawdowns but rarely raise the Sharpe
ratio net of costs outside equity-market portfolios. On 2022-2025 BTC a daily-rebalanced volatility target
scored 0.47 net Sharpe against 0.45 for buy-and-hold, and 0.53 against 0.61 in 2025. Rank 3 will hold a
flat book until the direction heads improve (NT-003), which is its purpose. It is the correct replacement for
quantile entries. If no candidate passes the guard-rails, the honest winner is `always_flat` ("do not trade
the model"), and the report states that (section 4.5).

**Selection protocol** (section 4). Pre-register the K = 11 grid. Fit every threshold on each fold's
calibration block. Rank by the mean net Sharpe over 2 dev folds x 3 seeds. Disqualify any row that fails
these paired guard-rails: beats always-flat, beats buy-and-hold, beats the matched random or timing null at
the 95th percentile, drawdown no worse than buy-and-hold, and at least 20 trades or rebalances. Report the
Deflated Sharpe Ratio for K = 11. On 10 dev days the expected best of 11 zero-skill Sharpe ratios is about
10 (annualised), so a dev Sharpe below that is not evidence. The seeds reduce model noise but not market
noise. Only more folds (the long history, NT-041) fix that.

**Engine needs** (section 5). A target-exposure backtest mode (a strategy returns a target exposure in
[-1, 1] at a bar's close, filled at the next open, rebalanced only outside a band, with per-bar cost
accounting). Model-free EWMA sigma in `SignalFrame`. A circular-shift timing null. A vol-targeted
buy-and-hold baseline. Stored calibration and out-of-sample prediction frames, so that strategy grids run on
the CPU without retraining. `assert_no_lookahead` extended to exposure decisions.

## 0. What the strategy layer gets today, and what it does with it

- **Per bar, per horizon (h0/h1/h2 = 10/15/20 bars):** the calibrated P(up); the served delta, which is
  beta x the raw head, with beta fit on the calibration block. On the latest run beta is 0.30 / 0.07 / 0.33
  (`eval_report_test.md`), and on earlier runs beta_h1 was 0, so the served delta is often exactly 0. The raw
  heads, whose Pearson correlation with the outcome is negative on test (-0.007 / -0.033 / -0.038). Sigma in
  $. Conformal 90% intervals, carried in `PredictionFrame.intervals` but not passed into `SignalFrame`.
- **Measured skill on the latest test block** (run `20260929T081632Z-426de4f-dirty-aba344d6`, 7,236 bars;
  the numbers are reported here, not used for choices):
  - Direction: AUC 0.506 / 0.500 / 0.507 and MCC 0.004 / -0.018 / 0.015. The model loses to `logreg_lags`
    on every direction metric, all within noise. The confidence gap is noise on all three horizons.
  - Price heads: skill against a zero prediction is negative on all three (served and raw).
  - Variance: CRPSS against a constant variance +0.020 / +0.014 / +0.009 (DM z +4.9 / +2.9 / +1.3). The
    var / err^2 Spearman is about 0.23-0.25. Coverage is 0.904 / 0.909 / 0.912.
- **The engine** (`strategy/backtest.py`) is all-or-nothing and discrete. There is one position at a time,
  sized once at entry, left while `max_hold`, a stop, a take-profit or `exit_signal` has not fired, with no
  resizing. Costs are 13 bps per side on the notional. Sharpe is annualised from per-bar equity returns,
  flat bars included.
- **Today's strategies** read the direction features: weighted P(up), agreement, consensus, strength and
  delta signs. The variance enters only as "confidence" weights and stop distances. On the latest test block
  `calibrated_quantile` made 156 trades: 156 x 26 bps is about 41% of equity in costs, which is the whole of
  the -33.8% net (the gross was about buy-and-hold's). This structure spends its turnover on the one output
  that has no skill and ignores the one output that has some.
- **Evaluation:** the engine scores one strategy per run on that fold's out-of-sample block
  (`experiments/scorer.py`), with its knobs fitted on the fold's calibration block. The run directories do
  not store the prediction frames, so today a strategy grid needs the TrainResult in memory, which means
  retraining (section 5.6).

## 1. Break-even arithmetic

### 1.1 Data and definitions

The volatility numbers come from the bundled `binance_btcusdt_1min_ccxt.csv` **with its last 7,400 bars
(fold -1's test block and gap) removed**, 36,100 bars from 2025-10-11 to 2025-11-05. They are checked against
the long history `Bitcoin_BTCUSDT.csv` for 2022-01-01 to 2025-09-29, which ends before the 30-day file starts.
The scripts ran in the session scratchpad on the CPU and are not committed. The tables below give everything
they printed.

- c = 26 bps round trip. sigma_H is the standard deviation of the non-overlapping H-bar log return.
- **Every-bar sign bet with signal correlation rho.** If the signal s and the return r are jointly Gaussian,
  E[sign(s) r] = rho sigma_H sqrt(2/pi). Break-even: rho* = c / (sigma_H sqrt(2/pi)).
- **Trading only the most confident fraction q** (q/2 in each tail, beyond z = Phi^-1(1 - q/2)): the gross
  edge is E = rho sigma_H phi(z) / (q/2), so rho*_q = c (q/2) / (sigma_H phi(z)).
- **Hit rate with symmetric win and loss sizes:** p* = 1/2 + c / (2 E|r_H|).
- **Gaussian read-out of a calibrated P(up):** if r_H ~ N(mu, sigma_H^2), then P(up) = Phi(mu / sigma_H), so
  mu_hat = sigma_H Phi^-1(p). An entry pays for itself only when p >= Phi(c / sigma_H), or p <= 1 - Phi(c / sigma_H).
- **From rho to AUC and accuracy** (simulated bivariate normal, 400,000 draws): rho 0.02 gives accuracy 0.506
  and AUC 0.508; rho 0.05 gives 0.517 / 0.524; rho 0.10 gives 0.532 / 0.545; rho 0.20 gives 0.563 / 0.589;
  rho 0.30 gives 0.596 / 0.635. So the model's test AUC of 0.500-0.507 is rho of about 0.00-0.015. The
  `logreg_lags` range of 0.52-0.56 (direction_v1 REPORT, by fold) is rho of about 0.04-0.12.

### 1.2 The table (reference file, before the test block)

| H (bars) | sigma_H (bps) | E\|r_H\| (bps) | c / sigma_H | rho* every bar | rho* top 10% | rho* top 2% | hit rate p* | P(up) needed, Gaussian read-out |
|---|---|---|---|---|---|---|---|---|
| 10 | 20.1 | 13.6 | 1.29 | 1.62 | 0.63 | 0.49 | 1.46 (impossible) | 0.90 |
| 15 | 23.8 | 16.4 | 1.09 | 1.37 | 0.53 | 0.41 | 1.29 (impossible) | 0.86 |
| 20 | 27.1 | 18.7 | 0.96 | 1.21 | 0.47 | 0.36 | 1.20 (impossible) | 0.83 |
| 60 | 49.2 | 33.7 | 0.53 | 0.66 | 0.26 | 0.20 | 0.89 | 0.70 |
| 240 | 87.8 | 60.3 | 0.30 | 0.37 | 0.14 | 0.11 | 0.72 | 0.62 |
| 1440 | 183.8 | 145.4 | 0.14 | 0.18 | 0.07 | 0.05 | 0.59 | 0.56 |

The 2022-2025 history gives the same picture: sigma_15 = 27.9 bps, c / sigma_15 = 0.93, rho* = 1.17 on every
bar and 0.35 on the top 2%. The 2025 history gives sigma_15 = 23.2 and c / sigma_15 = 1.12. The
direction_v1 figure "15-minute sd about 21 bps" is from another block and agrees within noise.

### 1.3 Reading it

- **Every bar at 10-20 minutes:** rho* > 1. No forecast can break even. The cost exceeds one standard
  deviation of the move it is trying to catch.
- **Selective at 10-20 minutes:** trading only the top 2% of bars (about 145 entries per 7,236-bar block)
  needs rho of about 0.36-0.49 over all bars, which is roughly an overall AUC of 0.7 or more. The model has
  about 0.01, the logistic baseline's best fold about 0.12. This also assumes the edge is concentrated in the
  tails exactly as a Gaussian predicts, which is generous.
- **Calibrated probabilities make the gap visible:** at 20 minutes an entry breaks even only when calibrated
  P(up) >= 0.83 (or <= 0.17) in median volatility. In the top volatility quintile (15-minute move sd about
  40 bps, 2025 data) the bar is P(up) >= 0.74. A head whose test AUC is 0.507 does not produce such
  probabilities, and if it did they would be miscalibrated. The Gaussian read-out rule of rank 3 therefore
  stays flat, and that is the correct behaviour.
- **Longer holds** lower the bar (rho* of about 0.18 on every bar at one day), because the move grows like
  sqrt(H) while the cost stays fixed. But the model forecasts only 10-20 bars ahead. Its signals decay well
  before 240 bars, since the reversal and momentum in direction_v1 live at 5-60 bars. Holding a 15-minute
  signal for 4 hours dilutes its edge by about sqrt(15/240) = 1/4 while lowering rho* by about 4: a wash at
  best.
- **Is directional trading of this model viable?** No: not at 10-20 minutes with 26 bps, and not by holding
  longer. It would take a direction model with rho >= 0.1 on longer horizons, together with a cheaper venue.
  Both are outside the current model.

### 1.4 Why exit rules cannot rescue it

Under a price with no predictable drift (a martingale), no stop, take-profit or time exit changes the
expected gross P&L of a trade. That is the optional stopping theorem. Exits reshape the distribution: many
small wins and few large losses, or the reverse. They do not change the mean, and each extra exit and
re-entry adds another 26 bps. Every notebook strategy's TP / SL / TP1 structure is therefore neutral in
expectation before costs and negative after them. The 13 bps per side is fixed in bps, so it is paid on
every entry whatever the stop geometry.

### 1.5 Empirical check: simple rules on 2022-2025 (gross edge per trade)

The sign of the trailing L-bar return, held H bars, non-overlapping. Net = gross - 26.

| L | H | trades | gross bps / trade, 95% CI |
|---|---|---|---|
| 15 | 15 | 131,326 | -0.33 [-0.48, -0.18] (tiny reversal) |
| 60 | 15 | 131,323 | -0.00 [-0.16, +0.15] |
| 60 | 60 | 32,830 | -0.57 [-1.17, +0.04] |
| 240 | 60 | 32,827 | +0.06 [-0.54, +0.67] |
| 240 | 240 | 8,206 | -0.48 [-2.85, +1.89] |
| 1440 | 1440 | 1,366 | -5.3 [-19.6, +9.0] |
| 4320 | 1440 | 1,364 | +9.5 [-4.8, +23.8] |

Extreme-move rules (the trailing L-bar move beyond z x EWMA sigma, followed or faded, held H) give gross
edges from -4.6 to +9.6 bps, all with CIs covering 0. The largest mean, +9.6 at L = H = 60 with z = 2.58 on
170 trades, is still 16 bps short of the cost.

One cell stood out: daily momentum in the lowest-volatility quintile, +33 bps [+10, +56] on 273 trades. It
was one of about 30 cells examined, and the matching high-volatility cell was -28. It is a multiple-testing
candidate, not evidence, and is not used below.

The published intraday BTC momentum result (Shen, Urquhart and Wang 2022: the first half-hour of a volume
session predicts the last half-hour) is a session-level effect with one trade per day. Nothing here
contradicts it, and it does not use this model's outputs.

### 1.6 Cost sensitivity (not a proposal to change defaults)

rho* is linear in c. The reference costs, 10 bps fee + 3 bps spread and slippage per side, match Binance spot
VIP 0 taker fees (0.10%). The USDⓈ-M perpetual taker fee is 0.05% (maker 0.02%) (Binance fee schedule). At 8
bps per side (futures taker + 3) every rho* above falls to 0.62x, and that still leaves 10-20-minute trading
impossible. Changing the default cost profile changes trading behaviour, which is the owner's call. The
report should instead show each row's **break-even round-trip cost**, c* = gross P&L / traded notional,
which NT-005 (b) already asks for.

## 2. The architectures

### 2.0 First: which sigma?

Every architecture that can use this model's edge uses sigma, so the first question is whether the model's
sigma beats a free one. CRPSS against a constant variance for the 15-bar move: the constant and the EWMA
scale are fitted on the first half of the data and scored on the second half. The CRPS is Gaussian with zero
mean.

| data | EWMA hl 60 | EWMA hl 240 | EWMA hl 1440 | EWMA x time-of-day, hl 240 |
|---|---|---|---|---|
| reference file before the test block | +0.068 | +0.061 | +0.034 | +0.058 |
| 2025 Jan-Sep | +0.153 | +0.144 | +0.130 | +0.150 |

The model on its test block scores +0.014 at h1. That is a different block, and a constant fitted on the
train block rather than on a first half, so it is not a same-block result. But the gap is 4-5x. The model's
var / err^2 Spearman is 0.23-0.25; the EWMA's Spearman(sd, |r|) is 0.36-0.40 on the pre-test reference data.
(These are different statistics, so this comparison is only indicative.) This is well established:
realised-variance models such as HAR-RV (Corsi 2009) are hard to beat at short horizons.

**Consequence:** (a) The evaluation report needs EWMA and HAR variance baselines on the same block, with the
DM test it already runs against `const_var`. That is a finding for the backlog, and it bears on the
indicator-learning purpose too. (b) Every variance-driven strategy below runs with `sigma_source` in {model,
ewma}. The EWMA row is its model-free twin, and "the model adds value" is claimed only when the model row
beats its twin in a paired comparison (D-025).

### 2.1 Volatility-managed / volatility-targeted exposure

- **Mechanism:** hold e_t = min(L, sigma* / sigma_hat_t) of a long position, which is volatility targeting.
  Moreira and Muir (2017) scale by 1 / sigma_hat^2 instead. Risk is cut when predicted volatility is high.
  The Sharpe ratio rises only if expected returns do not rise in proportion to variance (Moreira and Muir
  2017).
- **Output used:** sigma (h2 is the longest horizon, so the least noise per unit of turnover), in return
  units (sigma_$ / close / sqrt(20) per bar).
- **Evidence:** Moreira and Muir (2017) report higher Sharpe ratios for the market and several factors. The
  critiques:
  - Cederburg et al. (2020): across 103 portfolios, real-time out-of-sample versions "generally earn lower
    certainty equivalent returns and Sharpe ratios than the unmanaged portfolios".
  - Liu, Tang and Zhou (2019): the original scaling constant used look-ahead. Corrected, drawdowns reach
    68-93%.
  - Barroso and Detzel (2021): after transaction costs, only the market portfolio's benefit survives.
  - Harvey et al. (2018), on 60 assets: vol targeting raises the Sharpe ratio for risk assets (equity,
    credit), with negligible effect elsewhere, and reliably shrinks the left tail.
  - Wang and Yan (2021): scaling by downside volatility does better than scaling by total volatility.
- **Our check** (2022-2025 BTC, EWMA sigma, 13 bps per side, long-only, cap 1). All the gain, where there is
  any, is in drawdown:

  | rebalancing | net Sharpe | buy-and-hold Sharpe | turnover | max drawdown (vol-targeted vs buy-and-hold) |
  |---|---|---|---|---|
  | daily, band 0.10 | 0.47 | 0.45 | 12.7x / yr | 0.60 vs 0.68 |
  | hourly, band 0.25 | 0.22 | 0.45 | 38.6x / yr | |
  | every bar, band 0 | -0.73 | 0.45 | 322x / yr | |

  The 2025 sample alone: daily 0.53 against 0.61.
- **Why it could work here:** it uses the variance forecast, the one output with an edge, and needs no
  direction skill. Its turnover can be held to a few bps per 5-day block.
- **Why it may not:** intraday BTC shows no clear "high volatility means a worse return per unit of risk"
  pattern. The next-60-bar mean return by trailing-volatility quintile, 2022-2025, was +0.57 / +0.09 / +0.01
  / -0.58 / +1.29 bps, all within about 2 standard errors of 0. The gain Moreira and Muir found comes from
  monthly equity-factor dynamics, not from 1-minute crypto. And a 20-minute forecast is mostly its trailing
  volatility, so its timing adds little over an EWMA.
- **Fitting on the calibration block:** sigma* = the median of sigma_hat on the calibration block (so the
  mean exposure is about 0.8-0.9). L = 1 is fixed (spot, no leverage). The band b and the decision cadence
  are the only knobs.
- **Turnover and cost drag (an estimate):** hourly decisions with band 0.25 turn over about 0.5x equity per
  5-day block, about 7 bps of cost. The model's 20-minute sigma is noisier than an hl-240 EWMA, so expect
  more (the band absorbs it).
- **Failure modes:** in a 5-day block the result is dominated by the drift (buy-and-hold made +4.4% on the
  latest test block), so it must be judged **paired** against buy-and-hold (section 4.3). A wrong sigma*
  from a quiet calibration block leaves the exposure capped at 1 all the time, which is just buy-and-hold. A
  whipsaw with no band pays 42% a year.
- **Engine:** needs the target-exposure mode (section 5).

### 2.2 Variance-forecast regime filter (stand aside or trade by predicted volatility)

There are two opposite uses, and they should not be confused.

- **Risk filter on a long book (rank 2):** go flat when sigma_hat is above its q-quantile on the calibration
  block, and re-enter below a lower quantile (hysteresis). This is the binary version of 2.1, a
  "volatility-timing threshold". Today's engine can express it:
  - `decide` returns LONG while sigma_hat < q_in;
  - `exit_signal` fires when sigma_hat > q_out;
  - `max_hold` is set very large.

  Knobs: q_out in {0.80, 0.95}, with q_in = q_out - 0.10. The literature and failure modes are those of 2.1.
  The binary version turns over more per unit of risk change than the banded continuous one (Barroso and
  Detzel 2021 treat discrete switching as a cost-mitigation variant).
- **Tradeability filter on a directional rule (rank 4):** trade only when sigma_hat is high. The cost is a
  smaller share of the move there (c / sigma_15 is 0.65 in the top quintile against 2.74 in the bottom one,
  2025 data), so a given rho is worth more bps. The gate raises the gross edge per trade only in proportion
  to sigma, and it helps only when rho > 0. On 2022-2025, momentum gated to the top volatility quintile was
  -2.3 bps [-4.4, -0.1] at L = H = 60: no hidden edge surfaced.

### 2.3 Cost-aware no-trade bands and partial rebalancing toward an aim portfolio

- **Theory:**
  - Proportional costs: the optimal policy is a no-trade region around the frictionless target, and one
    trades only to its boundary (Davis and Norman 1990). For small costs its width scales as cost^(1/3)
    (Janecek and Shreve 2004).
  - Predictable returns with quadratic costs: trade partially toward an **aim portfolio**, a weighted
    average of the current and expected future Markowitz portfolios, with slower signals weighted more
    (Garleanu and Pedersen 2013).
  - Linear costs with a position cap: the optimum switches between the caps when the predictor crosses a
    threshold, and with a quadratic risk penalty that threshold becomes the no-trade band (de Lataillade,
    Deremble, Potters and Bouchaud 2012).
- **Mapping to our outputs:** aim a_t = f x mu_hat_t / sigma_hat_t^2, clipped to [-1, 1], where mu_hat_t is the
  expected H-bar return. From P(up) via the Gaussian read-out, mu_hat = sigma_hat Phi^-1(p). From the served
  delta, it is a per-horizon choice (NT-007). Trade only when |a_t - e_t| > b, and then to the band's edge.
- **Why it matters here** even though it will barely trade: this is the only architecture whose "do nothing"
  answer is principled. With mu_hat of a few bps against a 26 bps round trip, the band holds the book flat.
  A future direction model (NT-003) is monetised correctly by the same code with no new knobs. It replaces
  quantile entries, which force about 10% of bars into trades whatever their edge.
- **Engine:** target-exposure mode. The discrete twin for today's engine is an entry only when
  |mu_hat_H| > c + k sigma_hat_H x SE, held H bars. It is the NT-005 variant "enter only when |expected
  move| > cost + k x sigma". Expect 0 trades on the current model (section 1.3).

### 2.4 Kelly and fractional Kelly sizing from P(up) and sigma

- **Theory:** the growth-optimal fraction is f* = mu / sigma^2 for a continuous asset (Kelly 1956; Thorp 2006),
  and p - (1 - p) / b for a binary bet with odds b. Full Kelly is very risky in the short run under estimation
  error. Fractional Kelly (blending with cash) trades growth for much smoother paths (MacLean, Thorp and Ziemba
  2010).
- **Here:** Kelly without costs is the aim of 2.3 with f = 1, and it is dangerous. A 2 bps edge on a 27 bps,
  20-minute sigma gives f* of about 2.7, a 270% position churned every 20 minutes. **Kelly must be applied to
  the net edge over the holding period:**
  f = f_frac x max(|mu_hat_H| - c, 0) / sigma_hat_H^2, with sign(mu_hat_H), clipped to [-1, 1].

  With the current heads this is 0 on almost every bar. So Kelly is not a separate architecture here but the
  sizing rule inside rank 3, with f_frac in {0.25, 0.5}.
- **Failure modes:** a miscalibrated P(up) turns straight into over-sizing, so calibration is a
  precondition (the model's ECE of 0.02-0.04 is already comparable to its whole edge). Short blocks cannot
  validate a growth criterion.

### 2.5 Expected-edge-over-cost entry filters with longer holds

- This is the discrete form of 2.3 / 2.4: enter when the expected H-bar net edge is positive by a margin,
  hold H bars, and never exit early on a signal (early exits pay another 26 bps for an edge that section 1.4
  says is 0 in expectation).
- **Longer holds** help only if the signal's edge decays more slowly than sqrt(H). Section 1.3 shows it does
  not for a 10-20-bar signal. Holding h2 (20 bars) instead of h1 does not change the per-entry cost: each entry still pays
  26 bps.
- It is included (inside rank 3) because NT-005 names it and because its expected result, 0 trades, is the
  correct null behaviour.

### 2.6 Meta-labelling

- **Mechanism** (Lopez de Prado 2018, ch. 3; Joubert 2022; Meyer, Barziy and Joubert 2023): a primary rule
  chooses the side. A secondary classifier, trained on the primary's own trades, predicts P(this trade is
  profitable after costs). It filters the trades and sizes them, and calibrating it matters for sizing.
- **Output used:** our P(up), sigma, the interval width and the agreement features as the secondary's inputs;
  the primary is a TA rule or the trailing-returns logistic model.
- **Why not now:** the secondary is fitted on labelled primary trades. A fold's calibration block holds about
  2,000-2,900 bars, which is about 130-190 non-overlapping 15-bar outcomes, and a primary that trades 5% of
  bars labels about 100-150 trades. That is too few to fit and calibrate a classifier without overfitting.
  The one feature that is estimable is a monotone threshold on sigma_hat, and that is rank 4. Fitting the
  secondary on the training block instead would reuse data the network was fit on. Revisit this when the
  long history (NT-041) gives thousands of primary trades per fold.
- **Failure mode:** meta-labelling improves the precision of a primary that already has an edge. It cannot
  create one. With no gross edge in the primaries (section 1.5), it would only choose which losing trades
  to skip.

### 2.7 Trend or mean-reversion primaries gated or sized by the variance forecast

- **Primaries:** the classic TA rules that NT-033 builds as manual baselines (moving-average cross, RSI
  threshold, Bollinger breakout, with the same search budget). Time-series momentum (Moskowitz, Ooi and
  Pedersen 2012) scales positions by 1 / sigma. Kim, Tse and Wald (2016) show its alpha comes mostly from the
  volatility scaling, not from the momentum.
- **Gate or size with the model:** enter only when sigma_hat >= its cal-block quantile q (rank 4), or size
  each entry by sigma* / sigma_hat. This is exactly the yardstick's "learned indicators against textbook
  rules" question, asked about the variance head: does the model's sigma make a textbook rule better than
  the same rule with an EWMA gate, or with none?
- **Expectation (an estimate):** the primaries show no gross edge at intraday horizons (section 1.5).
  Gating by volatility did not reveal one, so the likely result is negative. It is still worth running
  because it costs 4 candidate configurations and answers the yardstick's question directly.
- **Engine:** today's engine (discrete entries), since gating is an entry filter. Sizing by sigma* / sigma_hat
  uses `Order.size_frac`. The size-matched random null already accounts for sizes.

### 2.8 Others considered

- **Variance monetised directly** (options, straddles, variance swaps on BTC). This is the natural home of a
  variance edge. It needs an options venue and data, which VISION puts outside the MVP. Noted, not proposed.
- **Market making with a sigma-driven quote width.** A variance forecast sets spreads and inventory limits
  in market-making models. It needs order-book data and maker fills, outside the MVP (VISION "Not in the
  MVP").
- **Volatility-scaled stops (ATR-style).** A risk tool, not an edge (section 1.4). Keep sigma-based stops
  only as a tail guard in discrete strategies.
- **The four notebook strategies** (`threshold_spike`, `enhanced_multi_horizon`, `liberal`,
  `calibrated_quantile`) are directional on heads with no skill, with 13-22 knobs each. Tuning them would
  inflate K for nothing. Keep `calibrated_quantile` as the incumbent row. Leave the others registered and
  untuned; deleting them is not proposed (D-029).

## 3. The ranked shortlist with parameterisations

Common to all: the thresholds are fitted per fold per seed on that fold's calibration `SignalFrame`.
sigma_hat_t is the sigma of h2 in return units, s.sigma[t, 2] / s.close[t]. `sigma_source` in {model, ewma},
where ewma is the model-free twin, a baseline that does not compete. The decision cadence is every 60 bars
for exposure strategies (fixed, not a knob).

1. **VT: volatility-targeted long** (`vol_target`, exposure mode). The target is
   e_t = min(1, sigma* / sigma_hat_t), with sigma* = median(sigma_hat) on the calibration block. Rebalance at
   the next open when |e_t - e_held| > b. **Grid: b in {0.10, 0.25}, K = 2** (+2 EWMA twins). Judged paired
   against buy-and-hold.
2. **RS: regime stand-aside** (`vol_regime_long`, today's engine). Long while sigma_hat < q_in; exit when
   sigma_hat > q_out. q_out is the calibration quantile in {0.80, 0.95}, q_in the calibration quantile
   (q_out - 0.10). `max_hold` = the block length. **K = 2** (+2 twins). This can run before the exposure mode
   exists: it is the cheap precursor of VT.
3. **NK: net-edge fractional-Kelly aim with a no-trade band** (`net_edge_kelly`, exposure mode).
   - mu_hat_t = sigma_$[t, h] x Phi^-1(p[t, h]) / close, for h = h2. The source is calibrated P(up); the
     served delta is an option once NT-007 decides.
   - aim_t = sign(mu_hat) x f x max(|mu_hat| - c, 0) / sigma_hat_H^2, clipped to [-1, 1], with c = 26 bps and
     sigma_hat_H the h2 sigma.
   - Trade to the band's edge when |aim - e| > 0.10.
   - **Grid: f in {0.25, 0.5}, K = 2.** Expected: flat on almost every bar with today's heads.
   - The discrete twin `edge_over_cost` for today's engine (entry when |mu_hat_H| > c, hold H = 20) can
     replace NK until the exposure mode exists (K = 1 then).
4. **GT: variance-gated TA primaries** (`gated_ta`, today's engine). The primaries are NT-033's MA cross and
   Bollinger breakout at textbook parameters, not tuned here. The ungated primaries are NT-033's baseline
   rows. Enter only when sigma_hat >= the calibration quantile q; exit on the primary's reversal or after
   `max_hold` = 60. **Grid: primary {MA cross, Bollinger} x q {0.5, 0.8}, K = 4** (+4 twins).
5. **ML: meta-labelling with a fitted secondary**: deferred to the long-history folds (section 2.6).
6. **TSMOM with volatility scaling at daily lookbacks**: a manual baseline for the long history. A 5-day
   block holds about 5 independent daily decisions, so it cannot be tested on the reference setup.

**Totals:** 2 + 2 + 2 + 4 + 1 (`calibrated_quantile`) = **K = 11 candidates**, plus 8 twins and 3 standard
baselines. Every strategy runs on stored predictions (section 5.6), so the grid costs CPU minutes and no GPU
time once the runs exist. The existing reference-scenario runs (3 seeds x 3 folds) are enough.

Order of implementation, by value per cost:

1. The same-block EWMA / HAR variance baseline in the report (it decides whether the model's sigma matters at all).
2. Stored prediction frames.
3. RS and GT and the discrete `edge_over_cost` (today's engine).
4. The exposure mode.
5. VT and NK.

## 4. Selection protocol

### 4.1 Pre-registration

Commit the candidate list (section 3, K = 11), their grids, the twins, the baselines, the guard-rails below,
the metrics and the "clear negative" rule before scoring any dev fold. Per D-020 the strategy grid is a
**sweep**: it picks a winner, and it is not a verdict that A beats B. Any "A beats B" claim (the model's sigma
beats EWMA, VT beats buy-and-hold) is a pre-registered study judged by the paired comparator (NT-032, D-025)
on judgement folds that no choice used. NT-005's "at most 3 variants" rule then applies to that study, not to
the sweep. How NT-005's acceptance maps onto this split is the lead's call.

### 4.2 Ranking

For each candidate and each (seed, dev fold) pair of the reference scenario (folds -3 and -2 x 3 seeds = 6
pairs), fit the calibration-block thresholds, then record the net Sharpe after the default costs. Rank by
the mean over the 6 pairs. Show the test fold -1 columns beside each row; they never rank (D-020).

### 4.3 Guard-rails (a row is disqualified if any fails)

All are computed on the dev folds with the pairs pooled, from per-bar net returns (the mean over seeds per
bar). The block bootstrap uses 240-bar blocks for exposure strategies and 80-bar blocks for discrete ones
(D-012).

1. **Beats always-flat:** mean dev net return > 0, with a bootstrap CI reported.
2. **Beats buy-and-hold, paired:** the per-bar difference (strategy - buy-and-hold) has a positive mean. The
   pairing removes the market path, which dominates both series. Without it, a long-biased overlay "wins"
   whenever the block went up.
3. **Beats its null at the 95th percentile on each dev fold.** For discrete strategies this is the existing
   size-matched random null (`random_same_frequency`). For exposure strategies it is a **circular-shift timing
   null**: the strategy's own exposure path shifted against the returns by random offsets (say 100 offsets
   drawn from [720, n - 720] bars). This keeps the exposure distribution, autocorrelation and turnover, and
   destroys only the timing.
4. **Max drawdown <= buy-and-hold's** on each dev fold (for long-biased rows), and <= 10% for the others.
5. **Activity floor:** at least 20 trades (discrete) or 20 rebalances (exposure) per dev fold, so that a
   winner is not one lucky trade.

**Model attribution (not a guard-rail, needed for any claim that the model adds value):** the row beats its
EWMA twin, paired.

### 4.4 How much a dev Sharpe can mean here

- A dev block is about 7,200 one-minute bars, about 5 days. The standard error of an annualised Sharpe
  estimated on T years is about sqrt(1 / T) (Lo 2002, iid case): **about 8.5 for one 5-day fold, about 6.0 for
  both dev folds pooled.**
- Seeds share the market path. They reduce the model's training noise, not this.
- The expected maximum of K zero-skill Sharpe z-scores (Bailey and Lopez de Prado 2014) is 1.05 for K = 4,
  1.66 for K = 12 and 2.53 for K = 100. With K = 11, the **best of 11 useless strategies is expected to show a
  dev Sharpe of about 10**, annualised.
- Report the **Deflated Sharpe Ratio** of the winner, with K = 11, T = the dev bars, and the skewness and
  kurtosis of its per-bar returns. With the twins counted too, K = 19 is the conservative choice. A winner
  counts as evidenced only if its DSR is >= 0.95. Otherwise it is "top of the leaderboard, not distinguishable
  from selection luck". Where a strategy family allows it, add the Probability of Backtest Overfitting
  (CSCV, Bailey, Borwein, Lopez de Prado and Zhu 2015) and the Harvey and Liu (2015) haircut as a
  cross-check. With 2 dev folds CSCV has almost no partitions, so the DSR is the primary number.
- The minimum-backtest-length argument (Bailey et al.) says the same thing more bluntly. Short blocks can
  reliably detect **losses from costs**, because cost drag is nearly deterministic (156 trades x 26 bps). They
  cannot reliably detect small positive edges. The long history (NT-041) is the only real fix: dozens of
  folds turn the standard error of 6 into about 1-2.

### 4.5 The winner and a clear negative

- **Winner:** the top row by mean dev net Sharpe **among the rows that pass every guard-rail**. Report it
  with its DSR, its test-fold columns (shown once, not used) and its break-even cost.
- **If no model-driven row passes,** the recorded winner is `always_flat` ("do not trade this model"). Buy-
  and-hold is a market exposure, not a use of the model: it is reported as the benchmark and never becomes
  "the model's winner". The report then gives NT-005 (b):
  - per architecture, the gross edge per trade (discrete) or per unit of turnover (exposure) in bps, with
    80-bar or 240-bar block-bootstrap CIs, against the 26 bps cost;
  - the break-even round-trip cost;
  - the model-against-EWMA twin comparison.
- **What counts as a clear negative:** every candidate fails guard-rail 1 or 2 on the dev folds, and the
  break-even cost of every discrete row lies below 26 bps with its CI excluding 26. That is a negative that
  more tuning will not reverse.

## 5. Engine requirements

### 5.1 A target-exposure mode

```python
class ExposureStrategy(Strategy):            # registered in Strategies with tag "exposure"
    decide_every: int = 60                   # bars between decisions (the cadence)
    band: float = 0.10                       # no-trade band in exposure units
    max_abs_exposure: float = 1.0            # spot: 1.0, no leverage

    def fit(self, cal: SignalFrame) -> "ExposureStrategy":
        """Set every threshold from the CALIBRATION block (sigma*, quantiles). Returns self."""

    def target(self, s: SignalFrame, t: int, current: float) -> float:
        """Target signed exposure in [-max_abs_exposure, max_abs_exposure], decided at bar t's close
        from s[:t+1] only; `current` is the exposure held at bar t's close (after drift)."""
```

`run_exposure_backtest(signals, bars, strategy, config)`:

1. **At the open of bar t:** if a pending target exists, rebalance to it. The traded notional is
   |target - current| x equity at the open. Fill at the mid moved adversely by spread + slippage; fee on
   the notional. Costs are booked on that bar.
2. **Holding:** the engine holds a signed **quantity** between rebalances, so the exposure drifts with the
   price, as a real position does. The band compares the target with the drifted exposure.
3. **At the close of bar t:** mark the equity. If t % decide_every == 0 and t >= warm-up, call `target`. If
   |target - current| > band, queue it for bar t+1's open. The "trade to the band's edge" variant is
   `current + sign(d) x (|d| - band)`, with d = target - current.
4. **No stops** in this mode at first (section 1.4). A later optional tail-stop overlay would be a separate
   knob.
5. **Output:** the same `BacktestResult` (equity, gross equity, per-bar `position`), with one record per
   rebalance (bar, from, to, notional, costs) instead of trades. The summary adds `n_rebalances`,
   `mean_abs_exposure`, `turnover` (the sum of |delta e|), `cost_drag` and `breakeven_cost_bps`
   (= gross P&L / traded notional x 1e4, per side x 2).

It fits the existing timing contract: decide at close t, fill at open t+1, costs of 13 bps per side, the
`BacktestConfig` cost fields unchanged. `run_backtest` dispatches on `isinstance(strategy, ExposureStrategy)`.

### 5.2 SignalFrame additions (all causal)

- `sigma_ret` [N, 3]: sigma_$ / close.
- `mu_gauss` [N, 3]: sigma_$ x Phi^-1(p), the Gaussian read-out of the expected move.
- `sigma_ewma` [N, 3]: the model-free twin, an EWMA of squared 1-minute log returns scaled to each horizon.
  It is computed on the **full** OHLC frame up to each anchor bar and then indexed at the anchors, so the
  block's first bars have a warm history. Its scale is fitted on the calibration block.
- `interval_lo` / `interval_hi` [N, 3] from `PredictionFrame.intervals`, so that meta-label features exist
  later.

`SignalFrame.build` then needs the bars or the frame, and the `sigma_source` switch selects `sigma` or
`sigma_ewma` in every variance strategy.

### 5.3 Baselines the report carries

buy-and-hold, always-flat, the size-matched random null (discrete) or the circular-shift timing null
(exposure), a **vol-targeted buy-and-hold with EWMA sigma** (the model-free VT), and each candidate's EWMA
twin.

### 5.4 The look-ahead self-test

Extend `assert_no_lookahead` so that for an `ExposureStrategy` the recorded decisions are
(bar, target, queued) tuples. The same perturb-after-t test must leave every decision up to t and the equity
up to t + 1 unchanged. Add three cases:

1. `sigma_ewma` built from perturbed bars after t must not change before t. This catches a centred or
   backward-filled EWMA.
2. `fit(cal)` receives only the calibration frame. A test passes a test frame with NaNs after t and expects
   identical thresholds.
3. The circular-shift null never feeds its shifted exposures back into the strategy.

### 5.5 Leaderboard fields

Per row: mean dev net Sharpe, the per-(seed, fold) values, the guard-rail pass / fail with its statistic,
the DSR with the K used, the break-even cost, the turnover, and the test columns (never ranking). The
leaderboard header states K (NT-031).

### 5.6 Stored predictions, so strategy grids run on the CPU

Write the calibration and out-of-sample `PredictionFrame`s of every run as a small npz (`from_npz` already
exists). At 7,236 x 3 horizons x (P(up) raw and calibrated, delta, variance, interval bounds) plus the close,
that is roughly 1-2 MB. Whether that counts as a light file under NT-010's policy is to be checked. Add a
`scenario rescore --strategy ... --params ...` path that rebuilds the SignalFrames and scores any strategy
without retraining. Without it every strategy configuration costs a training run.

## 6. What this research did not do

- It did not see any model predictions: the runs do not store them. The model's numbers are quoted from the
  test report. Nothing was chosen from them.
- It did not compare the model's sigma with an EWMA on the same block. That is the first follow-up.
- It ran no strategy through the project's engine. The vol-targeting and momentum numbers come from a
  simplified close-to-close simulation in the scratchpad (costs 13 bps per side on |delta exposure|), used
  only to size expectations.
- About 30 descriptive cells were computed on the long history (section 1.5). No single cell among them is
  evidence.

## Proposed follow-up items (for the lead to triage)

1. **P1:** EWMA and HAR-RV variance baselines in the evaluation report on the same block, with the DM test
   against the model's variance heads.
2. **P1:** stored calibration and out-of-sample prediction frames per run, and `scenario rescore` for
   strategy grids on the CPU.
3. **P1:** the target-exposure backtest mode, the timing null, the vol-targeted baseline and the look-ahead
   tests (section 5).
4. **NT-005** re-scoped around the shortlist and the protocol above, once 1-3 exist. It is then
   CPU-only on the existing reference runs.

## References

- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2015). The probability of backtest overfitting. *Journal of Computational Finance*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management* 40(5), 94-107. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Barroso, P. and Detzel, A. (2021). Do limits to arbitrage explain the benefits of volatility-managed portfolios? *Journal of Financial Economics* 140(3), 744-767. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3088828
- Binance. USDⓈ-M futures trading fee rate. https://www.binance.com/en/fee/futureFee (spot VIP 0: 0.10% maker and taker; USDⓈ-M: 0.02% maker, 0.05% taker)
- Cederburg, S., O'Doherty, M. S., Wang, F. and Yan, X. S. (2020). On the performance of volatility-managed portfolios. *Journal of Financial Economics* 138(1), 95-117. https://doi.org/10.1016/j.jfineco.2020.04.015
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* 7(2), 174-196. https://doi.org/10.1093/jjfinec/nbp001
- Davis, M. H. A. and Norman, A. R. (1990). Portfolio selection with transaction costs. *Mathematics of Operations Research* 15(4), 676-713. https://pubsonline.informs.org/doi/10.1287/moor.15.4.676
- de Lataillade, J., Deremble, C., Potters, M. and Bouchaud, J.-P. (2012). Optimal trading with linear costs. *Journal of Investment Strategies* 1(3), 91-115. https://arxiv.org/abs/1203.5957
- Garleanu, N. and Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance* 68(6), 2309-2340. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080
- Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. and Van Hemert, O. (2018). The impact of volatility targeting. *Journal of Portfolio Management* 45(1), 14-33. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175538
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1), 13-28. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489
- Janecek, K. and Shreve, S. E. (2004). Asymptotic analysis for optimal investment and consumption with transaction costs. *Finance and Stochastics* 8(2), 181-206. https://link.springer.com/article/10.1007/s00780-003-0113-4
- Joubert, J. F. (2022). Meta-labeling: theory and framework. *Journal of Financial Data Science*, Summer 2022. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4032018
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal* 35(4), 917-926.
- Kim, A. Y., Tse, Y. and Wald, J. K. (2016). Time series momentum and volatility scaling. *Journal of Financial Markets* 30, 103-124. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2786955
- Liu, F., Tang, X. and Zhou, G. (2019). Volatility-managed portfolio: does it really work? *Journal of Portfolio Management* 46(1), 38-51. https://www.ssrn.com/abstract=3283395
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36-52.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Ch. 3 meta-labelling, ch. 10 bet sizing, ch. 11-14 backtest statistics.)
- MacLean, L. C., Thorp, E. O. and Ziemba, W. T. (2010). Long-term capital growth: the good and bad properties of the Kelly and fractional Kelly capital growth criteria. *Quantitative Finance* 10(7), 681-687. https://www.researchgate.net/publication/227623956
- Meyer, M., Barziy, I. and Joubert, J. F. (2023). Meta-labeling: calibration and position sizing. *Journal of Financial Data Science* 5(2). https://www.pm-research.com/content/iijjfds/5/2/23
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance* 72(4), 1611-1644. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
- Moskowitz, T. J., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2), 228-250.
- Shen, D., Urquhart, A. and Wang, P. (2022). Bitcoin intraday time series momentum. *Financial Review* 57(2), 319-344. https://onlinelibrary.wiley.com/doi/10.1111/fire.12290
- Thorp, E. O. (2006). The Kelly criterion in blackjack, sports betting and the stock market. In *Handbook of Asset and Liability Management*, vol. 1. Elsevier.
- Wang, F. and Yan, X. S. (2021). Downside risk and the performance of volatility-managed portfolios. *Journal of Banking and Finance* 131. https://www.lehigh.edu/~xuy219/research/Downside.pdf

Project evidence: `runs/20260929T081632Z-426de4f-dirty-aba344d6/eval_report_test.md`;
`runs/experiments/direction_v1/REPORT.md` (sections 1 and 5); `docs/BACKLOG.md` NT-005, NT-007, NT-033;
`src/neural_trade/strategy/*.py`; `src/neural_trade/experiments/scorer.py`.
