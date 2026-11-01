# What stops neural_trade from making money

| | |
|---|---|
| Date | 2026-11-01 |
| Task | Criticise what prevents the system from making money: arbitrary restrictions, unrealised architecture, defects, wrong objectives and untested routes. This includes the restrictions in our own research. |
| Method | Five independent critics (information set, execution, objective, model, strategy space). Each critic's findings went to a skeptic who re-checked every claim in the code and data. Three probes by the lead. A final number check of this document against all sources. |
| Result files | `research/money_blockers/critique.json` (48 blockers with verdicts), `probes.py`, `probes_output.txt` |
| Related | `plans/model_evaluation_timestral.md`, `plans/timestral_lig.md`, `plans/trading_calibration.md` |
| Prepared by | Claude Code (AI) |

---

## 1. Answer

**No single bug blocks the money. Three structural walls do, and each one alone is enough to stop profit.**

1. **The system trades where costs are larger than any edge found.**
   - **Moves are small at these horizons.** On the repo month the average move is 4 bps at 1 minute, 10 bps at 5 minutes and 16 bps at 15 minutes. On the quieter clean set it is 3, 7 and 12 bps.
   - **Costs are large.** A round trip costs 20 bps on spot, the venue the system uses, and 4–10 bps on perpetual futures (perps).
   - **The bar is out of reach at 15 minutes.** At 4 bps a rule needs 58–62% accuracy on the repo month and up to 67% on the clean set. On spot it cannot break even at 15 minutes or less.
   - **The best direction skill is far too small.** It is AUC 0.54, about 0.095 correlation, and worth about 1.8 bps per trade at 15 minutes. It would clear 4 bps only at horizons of 4 hours or more, and only if it carried over. No skill has been measured there.
2. **The training data is far too small, and our tests were too short for selective and long-horizon rules.**
   - **Training.** The model is capped at 2,880 sequences (48 hours). The last fold trains on 2,400 of them: 40 hours, all on a weekend. That is 40 optimiser steps for about 385k parameters. The notebook loaded 4.6 million bars (2017–2025) and kept 0.06% of them.
   - **Testing.** Dense 1–15-minute signals were tested well enough: they show no gross edge of even 1–2 bps. Selective rules and long horizons were not. A true 60%-accurate selective rule had only a 5–20% chance of passing our gate and then scoring significantly. A 53% edge at 4 hours needs about a year of trades to detect.
3. **The only skill large enough to matter is variance, and nothing turns it into money.**
   - **The variance skill is real.** Timestral's QLIKE is 0.5–2.8% better than FHS. Its predicted spread tracks the size of the next move (Pearson 0.33–0.42).
   - **It lowers the bar but does not raise accuracy.** In the most volatile 5% of moments, break-even accuracy falls to 56–62%. But accuracy does not rise with volatility, so fading moves in the high-volatility tercile still earns about 0 ± 2 bps gross.
   - **It sits at the wrong horizon.** Sizing and options need 1–30-day volatility forecasts; the skill is at 1–15 minutes.
   - **The current gate is backwards but nearly inert.** The notebook gate nominally favours low variance, which is the wrong way for costs. In practice it passes 90% of bars and uses neural_trade's own variance head, which has almost no skill.
   - **Two real direction signals exist, but they are worth only 0.2–3 bps per trade.** These are perp–spot basis reversion and fading large moves.

**Also: defects make the in-house backtest meaningless, but fixing them does not create profit.**
- The backtest charges no fees. With spot fees the logged run loses about 22%, not 1.7%.
- It uses 11× hidden leverage.
- It can only go short by construction.
- Its stops and take-profit levels exist only on paper and sit 1.5–18 standard deviations of the 30-minute move away.
- The learnable indicators do nothing in the forward pass, while the logs show "learned" periods.

**Routes that could make money do exist. None needs the current neural model, and none has passed a decisive test.**

| Route | Best evidence so far | Where it stands |
|---|---|---|
| A. Fade large moves on perps, gated by volatility | +0.2 to +2 bps gross per trade, positive in 4 of 4 pre-Test and 4 of 4 Test cells, none significant | Already loses in-sample at a realistic 7 bps (maker in, taker out). Needs a rebate tier to live. |
| B. Trend at 4-hour to daily horizons on many perps | Daily BTC trend: net Sharpe 0.67–0.82 against 0.32 buy-and-hold, all CIs include 0, 10 rules tried | Needs years of multi-asset data to confirm |
| C. Volatility risk premium, as a separate options business | +3.5–5.1 vol points per 30-day block, Sharpe 0.8–1.1 before costs, crash blocks of −55 to −74 points, negative in 2026 | Earns the premium itself; the model's skill adds nothing |
| D. Funding carry | About 2.6%/yr above cash in 2022–26, shrinking to about zero in 2025–26 | A hurdle for the other routes, not an opportunity now |

**Do first:** data before models. Download multi-year perp data, make horizon, venue, fees and fills harness parameters, and run one pre-registered rule per route on a holdout long enough to have power (Section 5).

---

## 2. Terms

| Term | Meaning |
|---|---|
| bps | Basis points; 1 bps = 0.01% |
| Round trip | Enter and exit one position. Spot costs 10 bps per side; the perp costs 2 bps per side as maker and 5 bps as taker. |
| Perp | Perpetual futures contract (here Binance USD-M BTCUSDT) |
| Break-even accuracy | The accuracy at which expected PnL after costs is 0. With wins and losses equal to the mean move it is 0.5 + c / (2 E\|r\|). A Gaussian model gives a lower bar, for example 58% instead of 62% at 15 minutes and 4 bps. |
| IC | Information coefficient: correlation between forecast and realised return |
| Dense, selective | Dense: a signal at every origin. Selective: trades only at the top few % of a score. |
| Power | The chance that a test detects a real effect of a given size |
| VRP | Volatility risk premium: implied volatility minus later realised volatility of the same tenor |
| QLIKE, FHS, EWMA | As in the zero-shot report: variance loss, filtered historical simulation, exponentially weighted moving average |

---

## 3. The blockers, ranked

The verdict column comes from the skeptics: confirmed, partly (the core holds, details corrected) or refuted (none
were). Where a blocker draws on several critic entries, both labels are shown. "Alone?" says whether removing that
blocker by itself would plausibly create profit.

### Tier 1: structural (each one blocks profit)

| # | Blocker | Kind | Evidence | Why it blocks money | Verdict | Alone? |
|---|---|---|---|---|---|---|
| B1 | **Horizon × cost geometry.** 1/5/15 min, where the cost is 25–100% of the average move | restriction | Repo month, mean \|move\| at 1/5/15/60/240/1440 min: 4.2/9.6/16.4/33/62/155 bps. Break-even accuracy at 4 bps: 97.5/71/62/56/53/51% (clean set: 67% at 15 min). At 20 bps (spot) it is unreachable up to 15 min. Break-even IC at 4 bps: 0.79/0.36/0.21/0.10/0.06/0.03. The model can already resample to hours (`RESAMPLE_MINUTES`, model.py:43). The research harness is fixed at 1/5/15 (`bench/common.py`, `calibrate.py`), so no test ever looked at h > 15 min. Variance ratios at 60 and 240 min have CIs that include 1, so no skill is known there yet. | A fixed cost in bps against a move that grows with √h: the edge needed falls as 1/√h, and the system works where the bar is highest. | confirmed / partly | No, only with a skill at ≥ 1 h |
| B2 | **Venue and fee tier.** Binance spot at retail fees; shorts not bookable | restriction | dw_ccxt.py:14, :48 (spot BTC/USDT). Spot round trip 20 bps; perp 4 (maker) / 10 (taker). Perp volume is 6.4–8.7× spot. On perp prices the best fade nets −0.09 / −1.74 bps at 4 bps, −0.83 / −2.20 with a one-bar delay, and −6 to −9 at 10 bps. The notebook opens SHORT trades with no borrow cost. | Every rule found so far earns +1 to +7 bps gross, so it is dead at spot fees and marginal on perps. | partly | No; necessary, not sufficient |
| B3 | **Starved training** | restriction | `MAX_SEQUENCE_COUNT = 2880` (model.py:48, applied :348-357; SRS 3.2), which is 48 h. The last fold trains on 2,400 sequences: 40 h, Sat 07:15 to Sun 23:14 UTC. Variance in that window is 2.3× lower than on weekdays. Optimiser steps: 5 batches of 576 per epoch × 8 epochs = 40 Adam steps, for about 385k parameters. The notebook loaded 4,598,124 sequences (2017–2025) and kept 0.063%. dw_ccxt.py:31 fetches 30 days. 8+ years are free on data.binance.vision. Note: lifting the cap alone breaks the model, because price-level inputs (BTC from about $3k to $120k+) turn into a regime label (D8). | A 51–54% edge needs thousands of independent outcomes. The 40-hour window holds about 24 non-overlapping 15-min moves above 30 bps. | confirmed / partly | No, but it is a precondition |
| B4 | **Tests too short for selective and long-horizon rules** (our own research design) | research-design | Dense 1–15-min signals were powered: the minimum detectable edge at H=1 over 6 days is 0.19 bps, and the dense tests rule out a 1–2 bps gross edge. Selective rules were not: deployed rules had 29–107 trades, so the minimum detectable gross edge was 3.8–9.9 bps per trade. A true 60% rule passes gate × Test only 5–20% of the time. Trades needed for 80% power: 620 at 55%, 1,715 at 53%. At 240 min, detecting 53% needs about 360 days out of sample. Detecting Sharpe 1 on one asset needs about 8 years. The short windows were imposed by Timestral's release date; price-only rules and neural_trade could have used years. | Any real edge in the selective or slow regime that still matters would look like noise. | confirmed / partly | No, but it is the cheapest decisive step |
| B5 | **The variance skill has no path to money** | unrealised | Probe (Section 4): predicted σ correlates 0.33–0.42 with realised \|move\| (Pearson), broadly stable from pre-Test to Test; the largest drop is 0.38 → 0.33. Break-even accuracy at 4 bps is 80–93% at 5 min (67–74% at 15 min) in the calm half, and 56–62% in the top 5%. But accuracy does not rise with σ: in the high-σ tercile the fade is 49.5–52.4% accurate at H=15, with gross −1.7 to +1.1 bps. Sizing by 1/σ made the fade worse (−2.28 → −2.81 bps per trade). Volatility targeting long BTC added a non-significant +0.07 Sharpe. The notebook gate (conf = 1/(1+var) > 0.5, spike skip) is backwards in principle, but it passes 90% of bars and uses neural_trade's head (Spearman 0.05–0.16). The 1.2×/0.6× sizing and the variance stops (helper_functions.py:61-86) are imported nowhere. | Volatility tells you when a move is big, not which way it goes. Without a direction source or a variance instrument at the right horizon, it cannot earn. | confirmed / partly | No |

### Tier 2: objective and decision layer

| # | Blocker | Evidence | Why it blocks money | Verdict | Alone? |
|---|---|---|---|---|---|
| B6 | **Objectives are not money.** Accuracy targets, losses with a built-in short bias, no "flat" class, no cost term | The last task maximised accuracy. A 56% rule lost 226 bps gross. A 60%-hit fade nets −1.05 bps, because the average loss (21.1) exceeds the average win (18.9). Focal α = 0.7 (model.py:113) puts the loss-optimal P(up) at 0.383 when the true P(up) is 0.5. The head ranges 0.14–0.49, so the notebook's LONG (> 0.65) can never fire and SHORT (< 0.35) is near mechanical: 9 of 9 logged trades were SHORT. Pinball τ = 0.4 (model.py:78): the 5-min median was DOWN-biased in 12 of 12 runs. `DIR_DEADBAND_BPS = 0` (model.py:119): 91% of 1-min labels are moves below 10 bps. Caveat: selecting on PnL has less power than selecting on accuracy (PnL per trade has an SD of about 40 bps at 15 min). | The system learns a short bet on the market, not an edge. Accuracy ignores payoff asymmetry and cost. | confirmed / partly | No: the unbiased head has AUC 0.49–0.51 |
| B7 | **Calibration ranks on scale-free confidence** | The calibration confidences (z, P(up) − 0.5, rank) are scale-free, so bps magnitude never enters `calibrate.py`. The earlier T-D rule (\|median\| > cost) is already an expected-value gate, and it correctly abstained for Timestral. Ranking the fade by bps instead of z lowered gross (+2.27 vs +2.95). An expected-value rule with an independent direction prior was never pre-registered. In-sample, "top-2% fade only when E\|r₁₅\| ≥ 20 bps" gives n = 50, 64% hit, +8.0 bps net. That is motivation, not evidence. | The decision layer cannot weigh move size against cost when selecting rules. | partly | No; only with a direction source |
| B8 | **Model selection and stopping have no link to money** | Checkpoints are picked on val_loss, and the zero-gradient trend term is 99.9% of it in early runs. Early stopping cannot fire (patience ≥ epochs, model.py:819-826). The 480-bar validation fold is also the test fold and the backtest window. Runs 4–7 are bit-identical, and some validation sets had no UP labels (accuracy 1.0, MCC 0). | Reported metrics and backtests are noise or in-sample. | confirmed / partly | No; needed for honest numbers |

### Tier 3: implementation defects (fix them to measure honestly; none creates profit)

| # | Defect | Evidence | Verdict |
|---|---|---|---|
| B9 | **Notebook backtest is not a PnL measurement** | No fees (cell 5 lines 310-315). 1 BTC per trade on a $10k account is 11.4× implicit leverage. With fees the logged run is about −$2,228 (−22%), not −$174. The variance mean and SD use centred windows (look-ahead; they did not fire in the logged run). TP/SL are "reference only": +0.5%/+1.5%/−1% are 1.5–4.5 SD of the 30-min move, and the 2–6% dynamic stop is 6–18 SD. The hold is 30 bars against a 5-min forecast. The agreement filter is sign-blind (2/3 < 0.67, so it needs 3/3). | partly / confirmed |
| B10 | **Learnable indicators are dead in the forward pass** (D1 and D7 go further than reported) | `logit + stop_gradient(logit)·19` = 20·logit forward (model.py:1073, 1085-1087, 1110-1112, 1135, 1147), so forward periods are 1.4e6–3e29. 34 of 39 EMA channels are constant inside each window. The %B channels have median \|value\| about 350 (up to about 5,400), which saturates the GRU gates. indicator_params_history.csv shows "learned" periods of 4.9–58.8, because the logger reads the raw logit. Every logit drifts by the same 0.00427 per epoch, which is the L2 step alone. The momentum write-back has a fixed point at 1.948. The tests check shapes only. A correct indicator bank inside Timestral (LIG) still found no direction skill. | confirmed |
| B11 | **Loss scale and controls** | The zero-gradient trend term (model.py:1644-1676) dominates the reference loss (raw USD²), so λ point/dir/var clamp at 1000 (model.py:783-804). Align, vol and L2 then fall below 1e-3 of the total. ReduceLROnPlateau uses factor 0.99. Scalers are rewritten on every call, and `force=False` reloads old weights (model.py:392-393, 843-848). There is no function that scores the latest bar. | confirmed / partly |
| B12 | **Inputs ignore what the CSV already holds** | Only the close reaches the model (model.py:244, 1219). The SRS (3.2.2, 3.5.1) promises OHLCV, returns, volatility and volume features; none is built. There is no time-of-day input, though mean \|1-min r\| varies 1.9× by UTC hour. Inputs are price levels (D8). Range terms improve a fitted HAR by 3–10 QLIKE points. Against a plain EWMA the gain is 5–8% at 15 min, mixed at 5 min and negative at 1 min. In LIG, range and clock helped repo QLIKE by 0.38% (descriptive) and not the clean set. | partly |

### Tier 4: unrealised routes (outside the current system)

| Route | Evidence (critics' computations, verified) | What it needs | Kill criterion |
|---|---|---|---|
| **A. Fade large moves on perps, gated by volatility** | Fade top-2% 15-min moves on the month-long CSV: gross +2.2 to +4.0 bps per trade on spot prices, t 1.6–2.2. On perp prices at 4 bps it nets −0.09 / −1.74. Splitting by volatility half (post hoc): +6.1/+4.6 vs +2.6/+0.8 bps. My probe gated on Timestral σ: positive gross in all 4 pre-Test and all 4 Test cells (+0.15 to +2.0 bps, spot prices, none significant). At a realistic 7 bps (maker in, taker out) it loses 3–5 bps per trade in-sample, and the high-volatility half nets −0.9 / −2.4. Passive maker entry fills 47–63% of signals and loses 0.6–0.7 bps to adverse selection. | 2+ years of 1-min perp klines and funding (free); a fill model; a maker-rebate fee tier | Net mean at the reachable fee tier with CI ≤ 0 on a sealed holdout of about 2 years (1,200–2,400 trades, SE about 1 bps per trade): drop intraday direction for good |
| **B. Horizons of 4 h to 1 day, trend, many perps** | Daily BTC time-series momentum, 2022-01 to 2026-11, net of 5 bps per side: Sharpe 0.67–0.82 against 0.32 buy-and-hold. All CIs include 0, and 10 rules were tried. The critic cites papers on weekly cross-sectional momentum and size in crypto (Liu & Tsyvinski 2021; Liu, Tsyvinski & Wu 2022). Those used 2014–2020 samples with illiquid coins, so performance on today's top-30 perps is unknown. | Daily/4h klines and funding for 10–30 perps, 3+ years. Costs charged per position change. Funding modelled: about 1–3 bps per position, roughly 3 bps a day, borne mostly by longs. | Pooled net Sharpe CI including 0 on held-out years (pooling across assets is needed for power) |
| **C. Volatility risk premium (options)** | Deribit DVOL against forward 30-day realised volatility, 2022–2026: +6.8 vol points on average, implied above realised on 69% of days. As a variance-swap proxy per 30-day block: +3.5–5.1 points, Sharpe 0.77–1.13 before costs. The worst blocks are −55 to −74, and 2026 to date averages −3.2. Timing by implied minus trailing realised points the wrong way. The model's 1–15-min skill is at the wrong horizon to help. | Deribit option quote history (e.g. Tardis), fees, spreads, delta hedging, tail sizing: a separate business | Short-vol backtest with real bid/ask and crash days: Sharpe CI including 0 |
| **D. Funding carry** | 0.60 bps per 8 h (6.6%/yr gross) in 2022–2026. About 2.6%/yr above cash (Sharpe about 1.6 before costs), shrinking: +7.0% in 2024, +0.9% in 2025, −0.6% in 2026. | Spot + perp, margin, basis mark-to-market | It sets a hurdle; it is not an opportunity at current funding |
| **E. Market making** | Bar-level fills have −1.0 to −3.4 bps markout before the 2 bps maker fee. The tick is about 0.001 bps, so the spread pays nothing. | L2/tick data, colocation, rebate tiers: a new system | Not viable at a retail tier |

Other verified signals that are real but too small:
- **Perp–spot basis reversion.** IC +0.035 to +0.060 at 1 minute, t about 7–10 in each half. It is worth +0.2 to +0.7 bps per trade and vanishes with a one-bar delay.
- **Perp order-flow imbalance.** IC +0.025 to +0.034, also gone after one bar.

---

## 4. Lead's probes (`research/money_blockers/probes_output.txt`)

Thresholds are fixed on pre-Test data, then applied to Test (zero-shot Timestral-28M, both sets, spot prices). These numbers are descriptive.

**Predicted volatility vs. what costs demand** (break-even accuracy at 4 bps, pre-Test → Test)

| Regime by predicted σ | Repo, H=5 | Repo, H=15 | Clean, H=5 | Clean, H=15 |
|---|---|---|---|---|
| bottom 50% | 84% → 80% | 69% → 67% | 93% → 90% | 74% → 72% |
| 80–90% | 66% → 67% | 59% → 61% | 70% → 69% | 62% → 61% |
| top 5% | 60% → 60% | 57% → 56% | 62% → 62% | 58% → 58% |
| Pearson corr(predicted σ, \|move\|) | 0.40 → 0.40 | 0.37 → 0.38 | 0.42 → 0.40 | 0.38 → 0.33 |

**Fade the last move only in the top 10% of predicted σ** (time exit; gross bps per trade, t in brackets, accuracy)

| | Repo, H=5 | Repo, H=15 | Clean, H=5 | Clean, H=15 |
|---|---|---|---|---|
| Pre-Test | +0.40 (0.49), 52.4% | +2.04 (1.03), 52.7% | +1.18 (1.66), 55.6% | +0.15 (0.09), 51.7% |
| Test | +1.26 (1.02), 51.3% | +1.75 (0.59), 50.0% | +0.92 (0.92), 53.8% | +1.00 (0.42), 51.7% |
| ±1σ bracket exit (Test) | +0.56 | −1.15 | +0.20 | +0.27 |

- **What volatility buys.** It lowers the bar (60% instead of 80–90%).
- **What it does not buy.** It does not raise accuracy, which stays at 50–56%.
- **Edge size.** The gross edge is +0.2 to +2 bps against a 4 bps cost, and brackets lower it in every cell.

**Implied vs later realised volatility** (Deribit DVOL, 30-day implied, against forward 24 h and 7 d realised). Repo window: +1.1 / +0.2 vol points (implied above realised in 50% of hours). Clean window: +4.9 / +4.2 points (72% / 96%). This comparison is descriptive and mismatches the horizons, so it is not a VRP estimate; rely on the 30-day block figures in route C.

---

## 5. What to do

1. **Stop measuring with the notebook.**
   - Add fees, explicit sizing and causal windows (B9).
   - Make horizon, venue, fee tier and fill model parameters of the research harness (B1, B2).
   - Small effort.
2. **Get the data and size the tests (B3, B4).**
   - Download 2019–2026 1-min klines for BTCUSDT perp and spot, funding, and daily klines for 10–30 liquid perps from data.binance.vision. It is free.
   - Build a walk-forward harness with:
     - monthly refits and a purge;
     - costs charged on position changes;
     - funding;
     - taker fills, and maker fills with a trade-through rule and an adverse-selection markout;
     - block-bootstrap errors.
   - Before each test, compute the power and set the holdout length from it:
     - about 2 years for route A;
     - several years pooled across assets for route B.
     - A 6-month seal is too short for either.
3. **Run one pre-registered rule per route, once each, with stated kill criteria (Section 3, Tier 4).**
   - Route A at the fee tier you can actually reach.
   - Route B with funding charged.
   - Route D as the hurdle.
   - Route C only if an options business with tail-risk limits is wanted.
   - Judge on net PnL. State the PnL target and its power in advance, because a PnL test has less power than an accuracy test.
4. **Only if a route passes,** decide on model work.
   - Routes B–D need no neural model.
   - For route A, a variance model at the trading horizon could sharpen the gate. It would use OHLC, the clock, a context of at least 512 bars and unit-free targets.
   - If neural_trade stays in the stack, fix in this order:
     1. return inputs (D8), before lifting the data cap;
     2. B10–B12;
     3. the loss biases in B6 (α = 0.5, τ = 0.5, deadband at cost).

**Do not:**
- add indicators or retrain at 1–15 min;
- tune thresholds on 1–2-week windows;
- optimise accuracy;
- size by 1/σ without an edge;
- expect the Tier 3 defect fixes to create profit.

---

## 6. Limits of this critique

- **The probes and critic computations are descriptive.**
  - Most rest on one month per set.
  - Several (route A, the volatility halves, the E\|r\| gate) are in-sample, post hoc, or overlap the Test windows.
  - Route C and D numbers are before trading costs, margin and exchange risk.
- **Literature is cited as the critics reported it.** The two crypto-factor papers were not re-read here.
- **Our own earlier analyses carried restrictions that this critique counts as blockers.** They used:
  - short windows;
  - accuracy as the target;
  - 1-unit, non-overlapping, close-fill trades;
  - a 3-minute origin grid;
  - horizons of 15 minutes or less.

  Their verdicts stand within those limits. They do not show that no profitable strategy exists for this team.
