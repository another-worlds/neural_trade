# What stops neural_trade from making money

| | |
|---|---|
| Date | 2026-11-01 |
| Task | Criticise what prevents the system from making money: arbitrary restrictions, unrealised architecture, defects, wrong objectives and untested routes. This includes the restrictions in our own research. |
| Method | Five independent critics (information set, execution, objective, model, strategy space). Each critic's findings went to a skeptic who re-checked every claim in the code and data. Three quantitative probes by the lead. |
| Result files | `research/money_blockers/critique.json` (48 blockers with verdicts), `probes.py` and `probes_output.txt` |
| Related | `plans/model_evaluation_timestral.md`, `plans/timestral_lig.md`, `plans/trading_calibration.md` |
| Prepared by | Claude Code (AI) |

---

## 1. Answer

**No single bug blocks the money. Three structural walls do, and each one alone is enough to stop profit.**

1. **The system trades where costs are larger than any achievable edge.**
   - The average move is 4 bps at 1 minute, 10 bps at 5 minutes and 16 bps at 15 minutes. A round trip costs 20 bps on spot (the venue the system uses) and 4–10 bps on perpetual futures (perps).
   - At 15 minutes and 4 bps, a rule needs 58–62% accuracy to break even. On spot it cannot break even at 15 minutes or less.
   - The best direction skill measured anywhere (AUC 0.54, correlation about 0.095) is worth about 1.8 bps per trade at 15 minutes. It clears 4 bps only at horizons of 4 hours or more.
2. **The system can neither learn nor measure an edge of realistic size.**
   - Training uses 2,880 sequences, which is 40 hours of data. In the last fold the training window is 100% weekend.
   - Training is 40 optimiser steps for about 385k parameters.
   - The notebook loaded 4.6 million bars (2017–2025) and kept 0.06% of them.
   - Our own Test windows were 6 and 14 days. A true 60%-accurate rule had only a 5–20% chance of passing our calibration gate and then scoring significantly on Test. A documented edge of 51–54% needs months to years of trades.
3. **The only real skill is variance. It has no path to money, and the decision layer uses it backwards.**
   - Variance skill is the only skill found: Timestral's QLIKE is 0.5–2.8% better than FHS.
   - The notebook trades only when predicted variance is low. That is where costs are hardest to beat: break-even accuracy is 80–93% in calm periods and 56–62% in the most volatile 5%.
   - The sizing helper is dead code. Nothing in the system can trade variance: there are no options and no implied-volatility data.

**Also: defects make the in-house backtest meaningless, but fixing them does not create profit.**
- The backtest charges no fees. With spot fees the logged run loses about 22%, not 1.7%.
- It uses 11× hidden leverage.
- It can only go short by construction.
- Its stop-loss and take-profit levels exist only on paper and are 10–50× the move scale.
- The learnable indicators do nothing in the forward pass, while the logs show "learned" periods.

**Routes that could make money do exist. None needs the current neural model, and none has passed a decisive test.**

| Route | Best evidence so far |
|---|---|
| A. Fade large moves when predicted volatility is high, on perps with cheap execution | +0.2 to +2 bps gross per trade in all 8 cells, but passive fills lose about 0.7 bps to adverse selection |
| B. Trend at 4-hour to daily horizons on many perps | Daily BTC trend: net Sharpe 0.67–0.82 against 0.32 buy-and-hold, CIs include 0 |
| C. Volatility risk premium, as a separate options business | +3.5–5.1 vol points per 30 days, Sharpe 0.8–1.1 before costs, crash blocks of −55 to −74 points |
| D. Funding carry | About 2.6%/yr above cash, shrinking toward zero in 2025–26 |

**Do first:** data before models. Download multi-year perp data, make horizon, venue, fees and fills harness parameters, select on net PnL, and run one pre-registered rule per route on a sealed holdout (Section 5).

---

## 2. Terms

| Term | Meaning |
|---|---|
| bps | Basis points; 1 bps = 0.01% |
| Round trip | Enter and exit one position; cost = fee per side x 2 (spot 10 bps per side; perp 2 bps maker, 5 bps taker) |
| Perp | Perpetual futures contract (here Binance USD-M BTCUSDT) |
| Break-even accuracy | Accuracy at which expected PnL after costs is 0. With wins and losses equal to the mean move: 0.5 + c / (2 E\|r\|). A Gaussian model gives a lower bar, for example 58% instead of 62% at 15 minutes and 4 bps. |
| IC | Information coefficient: correlation between forecast and realised return |
| Gate, power | Gate: the selection-adjusted rule from `trading_calibration.md`. Power: the chance that a test detects a real effect of a given size. |
| VRP | Volatility risk premium: implied volatility minus later realised volatility |
| QLIKE, FHS, EWMA | As in the zero-shot report: variance loss, filtered historical simulation, exponentially weighted moving average |

---

## 3. The blockers, ranked

The verdict column comes from the skeptic: confirmed, partly (the core holds, details corrected) or refuted (none
were). "Alone?" says whether removing that blocker by itself would plausibly create profit.

### Tier 1: structural (each one blocks profit)

| # | Blocker | Kind | Evidence | Why it blocks money | Verdict | Alone? |
|---|---|---|---|---|---|---|
| B1 | **Horizon × cost geometry.** 1/5/15 min, where the cost is 25–100% of the average move | restriction | Mean \|move\| 4.2/9.6/16.4/33/62/155 bps at 1/5/15/60/240/1440 min. Break-even accuracy at 4 bps: 97.5/71/62/56/53/51%. At 20 bps (spot) it is unreachable up to 15 min. Break-even IC at 4 bps: 0.79/0.36/0.21/0.10/0.06/0.03. The model can already resample to hours (`RESAMPLE_MINUTES`, model.py:43), but the research harness is fixed at 1/5/15 (`bench/common.py`, `calibrate.py`), so no test ever looked at h > 15 min. | A fixed cost in bps against a move that grows with √h: the edge needed falls as 1/√h, and the system works where the bar is highest. | confirmed | No, only with a skill at ≥ 1 h |
| B2 | **Venue and fee tier.** Binance spot at retail fees; shorts not bookable | restriction | dw_ccxt.py:14, :48 (spot BTC/USDT). Spot round trip 20 bps; perp 4 (maker) / 10 (taker). Perp volume is 6.4–8.7× spot. On perp prices the best fade nets −0.09 / −1.74 bps at 4 bps, −0.83 / −2.20 with a one-bar delay, and −6 to −9 at 10 bps. The notebook opens SHORT trades with no borrow cost. | Every rule found so far earns +1 to +7 bps gross, so it is dead at spot fees and marginal on perps. | partly | No; necessary, not sufficient |
| B3 | **Starved training.** 40 h of weekend data, 40 optimiser steps | restriction | `MAX_SEQUENCE_COUNT = 2880` (model.py:48, applied :348-357; SRS 3.2). The last fold trains on Sat 07:15 – Sun 23:14 UTC. 1-min variance in train is 2.3× lower than on weekdays. BATCH 576 × 5 batches × 8 epochs = 40 Adam steps for about 385k parameters. The notebook loaded 4,598,124 sequences (2017–2025) and kept 0.063%. dw_ccxt.py:31 fetches 30 days. 8+ years are free on data.binance.vision. | A 51–54% edge needs thousands of independent outcomes to learn. The training window holds about 24 non-overlapping 15-min moves above 30 bps. | confirmed | No, but it is a precondition |
| B4 | **Measurement cannot see a real edge** (our own research design) | research-design | Test windows of 6 and 14 days; deployed rules had 29–107 trades; minimum detectable gross edge 3.8–9.9 bps per trade. A true 60% rule passes gate × Test only 5–20% of the time. Trades needed for 80% power: 620 at 55%, 1,715 at 53%. A t of 1.96 needs Sharpe 15 over 6 days, 10 over 14 days and 2 over a year. The proposed 4-week forward test would again have power of only 0.3–0.6. The short windows were imposed by Timestral's release date; price-only rules and neural_trade could have used years. | Real small edges look like noise and are rejected. Noise passes only rarely, and when it does it looks like an edge. | confirmed | No, but it is the cheapest decisive step |
| B5 | **Variance skill unused and inverted** | unrealised | My probe (Section 4): predicted σ correlates 0.33–0.40 with realised \|move\|, the same before Test and on Test. Break-even accuracy at 4 bps is 80–93% (5 min) or 67–74% (15 min) in the calm half, and 56–62% in the top 5%. The notebook gates on conf = 1/(1+var) > 0.5 and skips "variance spikes", so it trades in calm periods. The 1.2×/0.6× sizing and the variance stops (helper_functions.py:61-86) are imported nowhere. The neural_trade variance head is itself weak (Spearman 0.05–0.16 against 0.31–0.40 for EWMA), and a free Parkinson EWMA already ranks \|r\| at 0.36–0.40. | Gross profit scales with move size, while costs are fixed in bps. The system picks the worst cost-to-move moments and never sizes by risk. | confirmed | No; it helps only together with a direction source (route A) |

### Tier 2: objective and decision layer

| # | Blocker | Evidence | Why it blocks money | Verdict | Alone? |
|---|---|---|---|---|---|
| B6 | **Objectives are not money.** Accuracy targets, losses with a built-in short bias, no "flat" class, no cost term | The last task maximised accuracy. A 56% rule lost 226 bps gross, and a 60%-hit fade nets −1.05 bps because average loss (21.1) exceeds average win (18.9). Focal α = 0.7 (model.py:113) puts the loss-optimal P(up) at 0.383 when the true P(up) is 0.5. The head ranges 0.14–0.49, so the notebook's LONG (> 0.65) can never fire and SHORT (< 0.35) is near mechanical: 9 of 9 logged trades were SHORT. Pinball τ = 0.4 (model.py:78): the 5-min median was DOWN-biased in 12 of 12 runs. `DIR_DEADBAND_BPS = 0` (model.py:119): 91% of 1-min labels are moves below 10 bps. | The system learns a short bet on the market, not an edge. Accuracy ignores payoff asymmetry and cost. | confirmed / partly | No: the unbiased head has AUC 0.49–0.51 |
| B7 | **Decisions key on the output with no skill** (the median) and never on the one with skill (the spread) | Every rule tested is the sign of the median, or \|median\| > cost. The calibration confidences are scale-free ranks, so bps magnitude never enters. Timestral-LIG shrank its median to the last price by design, then was judged descriptively on direction. An expected-value rule with an independent direction prior and an external volatility level was never pre-registered. In-sample, "top-2% fade only when E\|r₁₅\| ≥ 20 bps" gives n = 50, 64% hit, +8.0 bps net. That is motivation, not evidence. | The decision layer cannot use where the model is good. | partly | No; only with a direction source |
| B8 | **Model selection and stopping have no link to money** | Checkpoints are picked on val_loss, and the zero-gradient trend term is 99.9% of it in early runs. Early stopping cannot fire (patience ≥ epochs, model.py:819-826). The 480-bar validation fold is also the test fold and the backtest window. Runs 4–7 are bit-identical, and some validation sets had no UP labels (accuracy 1.0, MCC 0). | Reported metrics and backtests are noise or in-sample. | confirmed | No; needed for honest numbers |

### Tier 3: implementation defects (fix them to measure honestly; none creates profit)

| # | Defect | Evidence | Verdict |
|---|---|---|---|
| B9 | **Notebook backtest is not a PnL measurement** | No fees (cell 5 lines 310-315). 1 BTC per trade on a $10k account is 11.4× implicit leverage. With fees the logged run is about −$2,228 (−22%), not −$174. The variance mean and SD use centred windows (look-ahead; they did not fire in the logged run). TP/SL are "reference only", at +0.5%/+1.5%/−1% and 2–6% for the dynamic stop: in 30 minutes +0.5% is reached in 11–12% of windows and 2% in 0.1%. The hold is 30 bars against a 5-min forecast. The agreement filter is sign-blind (2/3 < 0.67, so it needs 3/3). | confirmed |
| B10 | **Learnable indicators are dead in the forward pass** (D1 and D7 go further than reported) | `logit + stop_gradient(logit)·19` = 20·logit forward (model.py:1073, 1085-1087, 1110-1112, 1135, 1147), so forward periods are 1.4e6–3e29. 34 of 39 EMA channels are constant inside each window, and the %B channels reach about 350, which saturates the GRU gates. indicator_params_history.csv shows "learned" periods of 4.9–58.8, because the logger reads the raw logit. Every logit drifts by the same 0.00427 per epoch, which is the L2 step alone. The momentum write-back has a fixed point at 1.948. The tests check shapes only. | confirmed |
| B11 | **Loss scale and controls** | The zero-gradient trend term (model.py:1644-1676) dominates the reference loss (raw USD²), so λ point/dir/var clamp at 1000 (model.py:783-804). Align, vol and L2 then fall below 1e-3 of the total. ReduceLROnPlateau uses factor 0.99. Scalers are rewritten on every call, and `force=False` reloads old weights (model.py:392-393, 843-848). There is no function that scores the latest bar. | confirmed / partly |
| B12 | **Inputs ignore what the CSV already holds** | Only the close reaches the model (model.py:244, 1219). The SRS (3.2.2, 3.5.1) promises OHLCV, returns, volatility and volume features; none is built. There is no time-of-day input, though mean \|1-min r\| varies 1.9× by UTC hour. Inputs are price levels (D8). A single-split check found that range terms improve a HAR variance forecast by 3–10 QLIKE points, but in LIG range and clock helped by only 0.38%, so the gain is uncertain. | partly |

### Tier 4: unrealised routes (outside the current system)

| Route | Evidence (critics' computations, verified) | What it needs | Kill criterion |
|---|---|---|---|
| **A. Volatility-gated reversal on perps** | Fade top-2% 15-min moves: gross +1.0 to +4.0 bps per trade on the month-long CSV, higher in the high-volatility half (+6.1/+4.6 vs +2.6/+0.8 on perp). In my probe, gating on Timestral σ gave positive gross in 8 of 8 cells (+0.15 to +2.0 bps). Passive maker entry fills 47–63% of signals and loses 0.6–0.7 bps to adverse selection. | 2+ years of 1-min perp klines and funding (free); a fill model; maker-tier fees | Net mean at 4 and 7 bps with CI below or including 0 on a sealed 6+ months: drop intraday direction for good |
| **B. Horizons of 4 h to 1 day, trend, many perps** | Daily BTC time-series momentum, 2022-01 to 2026-11, net of 5 bps per side: Sharpe 0.67–0.82 against 0.32 buy-and-hold. All CIs include 0, and 10 rules were tried. The literature points to weekly cross-sectional momentum and size in crypto (Liu & Tsyvinski 2021; Liu, Tsyvinski & Wu 2022, as cited by the critic). | Daily/4h klines for 10–30 perps, 3+ years; costs charged per position change | Pooled net Sharpe CI including 0 on held-out years |
| **C. Volatility risk premium (options)** | Deribit DVOL against forward 30-day realised volatility, 2022–2026: +6.8 vol points on average, implied above realised on 69% of days. As a variance-swap proxy per 30-day block: +3.5–5.1 points, Sharpe 0.77–1.13 before costs. The worst blocks are −55 to −74, and 2026 to date averages −3.2. Timing by implied minus trailing realised points the wrong way. My probe: +4–5 points (clean window), about 0 (repo window). | Deribit option quote history (e.g. Tardis), fees, spreads, delta hedging, tail sizing: a separate business | Short-vol backtest with real bid/ask and crash days: Sharpe CI including 0 |
| **D. Funding carry** | 0.60 bps per 8 h (6.6%/yr gross) in 2022–2026. About 2.6%/yr above cash (Sharpe about 1.6 before costs), shrinking: +7.0% in 2024, +0.9% in 2025, −0.6% in 2026. | Spot + perp, margin, basis mark-to-market | It sets a hurdle; it is not an opportunity at current funding |
| **E. Market making** | Bar-level fills have −1.0 to −3.4 bps markout before the 2 bps maker fee. The tick is about 0.001 bps, so the spread pays nothing. | L2/tick data, colocation, rebate tiers: a new system | Not viable at a retail tier |

---

## 4. Lead's probes (`research/money_blockers/probes_output.txt`)

Thresholds are fixed on pre-Test data, then applied to Test (zero-shot Timestral-28M, both sets). These numbers are descriptive.

**Predicted volatility vs. what costs demand** (break-even accuracy at 4 bps, pre-Test → Test)

| Regime by predicted σ | Repo, H=5 | Repo, H=15 | Clean, H=5 | Clean, H=15 |
|---|---|---|---|---|
| bottom 50% | 84% → 80% | 69% → 67% | 93% → 90% | 74% → 72% |
| 80–90% | 66% → 67% | 59% → 61% | 70% → 69% | 62% → 61% |
| top 5% | 60% → 60% | 57% → 56% | 62% → 62% | 58% → 58% |
| corr(predicted σ, \|move\|) | 0.40 → 0.40 | 0.37 → 0.38 | 0.42 → 0.40 | 0.38 → 0.33 |

**Fade the last move only in the top 10% of predicted σ** (time exit; gross bps per trade, t in brackets)

| | Repo, H=5 | Repo, H=15 | Clean, H=5 | Clean, H=15 |
|---|---|---|---|---|
| Pre-Test | +0.40 (0.49), 52.4% | +2.04 (1.03), 52.7% | +1.18 (1.66), 55.6% | +0.15 (0.09), 51.7% |
| Test | +1.26 (1.02), 51.3% | +1.75 (0.59), 50.0% | +0.92 (0.92), 53.8% | +1.00 (0.42), 51.7% |
| ±1σ bracket exit (Test) | +0.56 | −1.15 | +0.20 | +0.27 |

- **Edge size.** The sign is positive in all 8 time-exit cells, but the size is +0.2 to +2 bps against a 4 bps cost.
- **Brackets.** Brackets lower the gross in every cell.
- **Break-even fee.** The fee would have to be at most 0.2–2 bps per round trip, which needs a rebate tier.

**Implied vs later realised volatility** (Deribit DVOL against forward 24 h / 7 d realised): repo window +1.1 / +0.2 points (implied > realised in 50% of hours); clean window +4.9 / +4.2 points (72% / 96%).

---

## 5. What to do

1. **Stop measuring with the notebook.**
   - Add fees, explicit sizing and causal windows (B9).
   - Make horizon, venue, fee tier and fill model parameters of the research harness (B1, B2).
   - Small effort.
2. **Get the data (B3, B4).**
   - Download 2019–2026 1-min klines for BTCUSDT perp and spot, plus funding, from data.binance.vision. It is free.
   - Build a walk-forward harness with:
     - monthly refits and a purge;
     - net PnL as the selection objective;
     - costs charged on position changes;
     - taker fills, and maker fills with a trade-through rule and an adverse-selection markout;
     - HAC or block-bootstrap errors;
     - a power calculation before each test.
   - Seal the last 6 months.
3. **Run one pre-registered rule per route, once each, with stated kill criteria.**
   - Route A: the volatility-gated 15-min fade on perp, scored at 4 and 7 bps.
   - Route B: one daily or 4-hour trend rule on 10–30 perps, scored at 10 bps.
   - Route D: funding carry, measured as the hurdle every other route must beat.
   - Route C only if an options business is wanted. It needs option quote data and tail-risk limits.
4. **Only if a route passes,** decide whether model work is needed.
   - Routes B–D need no neural model.
   - For route A, a variance-first model (OHLC, clock, ≥ 512-bar context, unit-free targets) could sharpen the gate.
   - Fix B10–B12 and the loss biases in B6 (α = 0.5, τ = 0.5, deadband at cost) only if neural_trade stays in the stack.

**Do not:**
- add indicators or retrain at 1–15 min;
- tune thresholds on 1–2-week windows;
- optimise accuracy;
- expect the defect fixes in Tier 3 to create profit.

---

## 6. Limits of this critique

- **The probes and critic computations are descriptive.**
  - Most rest on one month per set, and several (route A, the E\|r\| gate) are in-sample or overlap the Test windows.
  - Route C and D numbers are before trading costs, margin and exchange risk.
- **Literature is cited as the critics reported it.** The two crypto-factor papers were not re-read here.
- **Our own earlier analyses carried restrictions that this critique counts as blockers.** They used:
  - short windows;
  - accuracy as the target;
  - 1-unit, non-overlapping, close-fill trades;
  - a 3-minute origin grid;
  - horizons of 15 minutes or less.

  Their verdicts stand within those limits. They do not show that no profitable strategy exists for this team.
