# P&L-aware targets for the micro loop, and whether the owner's target is reachable

Research note, 2026-09-29, for NT-085 (the micro loop, D-041). The owner's goal (verbatim): "60< стабильной
предскзаткльной силы и drawdown <5 %", read as a stable hit rate above 60% with a maximum drawdown under 5%.
The owner named three causes of poor strategy results. Point 3 was "absence of the PnL in the models'
targets". This note researches point 3. It implements nothing and used no GPU. The only computation is a
CPU sizing script on `Bitcoin_BTCUSDT.csv` 2025-01-01 .. 2025-07-20, which ends before the micro loop's dev
block (2025-07-27 .. 2025-08-28) and far from fold -1. Nothing was chosen from any test block. Estimates are
labelled as estimates.

## Summary

**Reachability verdict.**

- **A stable hit rate above 60% on every bar, from OHLCV history, is not realistic.**
  - Our data, 10 hypotheses: no model beats a logistic regression on 3 lagged returns, whose AUC is
    0.51-0.53 (micro loop LOG).
  - Our CPU sizing below: that baseline hits 50.5-51.6% on all bars at 15-240 minutes.
  - The closest published study, Jaquart, Dann and Weinhardt (2021), used BTC at 1-60 minutes with
    technical, blockchain, sentiment and cross-asset features. It reached 51.5-56.0% accuracy, and every
    model lost money after a 30 bps round trip.
  - Published hit rates above 60% for BTC come from order-book data at horizons under a few seconds. Wang
    (2025) reports 71-73% at 0.5-1 s on Bybit. At those horizons the move is a fraction of one basis point,
    so a taker paying 13 bps per side cannot trade them.
- **"60% of TAKEN trades" (a selective, cost-aware filter) is a more realistic reading, but still unlikely
  from OHLCV, and on its own it is not a profit target.**
  - With symmetric profit-take and stop at b bps and a 26 bps round trip, the hit rate needed to break even
    is p* = 1/2 + 13/b. So a 60% hit rate only breaks even at b = 130 bps. It pays only on wider barriers,
    which means holds of hours to a day.
  - On the most confident 2-10% of bars, the logistic baseline reaches 53-57% hit, with a noise band of
    ±7-16 pp. Its gross edge is 0-4.5 bps per trade.
  - Estimate: a filter this selective trades roughly 0.5-2 times a day at 1-4 hour holds.
  - Telling 60% apart from the baseline's ~53% needs about 380 trades. That is about 6-20 months of trading,
    so a 30-day micro dev block cannot confirm the target at all. It can only show whether a directional
    signal exists (section 2.4).
- **Drawdown under 5% is a sizing question, not a model question.** Fractional sizing brings it under 5% if
  the edge is positive; always-flat meets it trivially. Without an edge, no sizing gives a positive PnL.
- **Information that moves the needle** is order flow, the order book and perpetual-futures lead-lag. The
  literature puts that signal at seconds to minutes. None of it is in `Bitcoin_BTCUSDT.csv`, and a new data
  source is outside the MVP (VISION "Not in the MVP"), so that is an owner decision.

**What P&L-aware training can and cannot do.**

- It can make the network spend its capacity on the bars and moves that matter for trading: large moves,
  net of cost, and the decision to stay flat.
- It cannot create information the inputs lack.
- The best evidence that a financial criterion beats a prediction criterion, Bengio (1997) and Lim, Zohren
  and Roberts (2019), comes from assets and horizons where the prediction criterion already had some edge.

**Ranked micro plan.** Order is value per cost. Each step runs on the micro layout; fold -1 stays untouched.

| rank | experiment | code? | GPU (estimate) | question |
|---|---|---|---|---|
| E1 | Cost-sensitive direction labels: `DIR_DEADBAND_BPS` 5 → 26 at horizons 60/120/240 | config only | ~25 GPU-min (2 variants x 3 seeds) | Does direction exist once small, untradable moves are masked out of training? |
| E2 | Mean-variance utility with a linear cost on the direction head's position 2p-1 (new objective `pnl_utility` in the Losses registry) | implementer item | ~40 GPU-min (3 variants x 3 seeds) | Does training on net P&L (owner point 3) produce a gross edge above cost? |
| E3 | Triple-barrier labels (±k·σ_ewma·√H, vertical barrier masked) plus a barrier strategy with the same geometry | implementer item | ~40 GPU-min (3 variants x 3 seeds) | Does a path-dependent, volatility-scaled "win before loss" label carry signal, and does the hit rate of taken trades reach p*? |
| E4 | Drawdown control by fractional sizing fitted on cal (net_edge_kelly f grid; a size grid for the barrier strategy) | config only (CPU rescore) | 0 | Only if E2 or E3 passes its trading gate: can MDD < 5% hold with net PnL > 0? |
| (E5) | A new information source (taker-buy volume, perpetual basis or funding, order-book imbalance) | owner decision first | n/a | Only if E1-E3 are negative: this is the only lever left in the literature |

**Recommended reading of the target, so that it can be checked** (for the owner to confirm):

> On walk-forward dev folds, the net hit rate of taken trades has a 95% lower bound ≥ 60% and ≥ p*(geometry),
> net PnL > 0 after the default costs, maximum drawdown < 5% of equity, and at least 100 trades per verdict.

## 1. P&L-aware targets and losses

### 1.0 What the codebase offers today (the attachment points)

- **The objective.** `Config.LOSS_NAME` resolves one objective from the Losses registry, once, when
  `CustomTrainModel` is built (`training/custom_model.py:123`).
  - An objective has the fixed signature `(model, x_window, y_true, y_pred, last_close, extended_trends,
    vacuum_overflow=None)` and returns the 34-slot `LossComponents` (`losses/registry.py`).
  - A new objective can wrap `custom_loss` and add a term. The registry, the train step and the
    physics terms stay untouched (D-002, D-003).
- **Forward returns are already available inside the objective.** r_H = y_true_raw / last_close, where
  y_true_raw = y_true x pred_scale + pred_mean (`losses/functions.py:494-496`). A P&L term therefore needs no
  new data for fixed-hold labels. Path-dependent labels (triple barrier) do: the objective sees only the three
  horizon end-points, not the path.
- **The heads.** The model returns a positional 9-tuple: price, direction and variance for each of h0, h1 and
  h2 (`models/gru_attention.py`). Serving, calibration and the evaluation frame all unpack these nine.
  Adding a head (a position head) touches every one of them, a much larger change than adding a term.
- **Cost-sensitive labels exist as config.** `DIR_DEADBAND_BPS` (default 5) masks every sample with
  |r_H| ≤ deadband out of the direction loss (`metrics/tf_direction.py:17-35`).
  - The same deadband is used by the calibration pipeline (`calibration/pipeline.py:148`), the `logreg_lags`
    baseline (`evaluation/baselines.py:57`) and the report (`evaluation/report.py:274`).
  - So changing it keeps model and baseline on the same labels.
- **What the physics terms touch.**
  - T-perp, Casimir, vacuum, IFE and vacuum-overflow act on the price and variance heads; HD acts on the
    variance heads (`losses/functions.py:661-712`).
  - None reads the direction heads. A term on the direction heads meets them only through the shared trunk.
- **Batches are fully shuffled** (`SHUFFLE_BUFFER: 0` means a full reshuffle in the micro and 360-day
  configs). Consecutive samples in a batch are not consecutive bars. So a Moody-style recurrent turnover term
  (position at t against position at t-1) cannot be computed inside a batch. A per-entry cost (enter, hold H,
  exit) can.

### 1.1 Differentiable utility and Sharpe losses (position = tanh(f(x)))

**Mechanism.**

- The network outputs a position a ∈ [-1, 1]. The loss is minus a financial criterion of a·r_H - c·|a|.
- **Earliest evidence.** Bengio (1997) trained networks directly on trading gains net of transaction costs
  for 35 Canadian stocks, and found this better than training on prediction error on noisy series.
- **Recurrent form.** Moody and Saffell (2001) maximise a differential Sharpe ratio with a recurrent
  position and turnover costs ("direct reinforcement").
- **Batch Sharpe.** Lim, Zohren and Roberts (2019) train on a batch Sharpe ratio of volatility-scaled
  positions over 88 futures.
  - Their Sharpe-optimised LSTM beat the benchmarks only up to 2-3 bps of cost.
  - At 10 bps without regularisation it scored a Sharpe of -5.3.
  - Putting the cost into the training return, which acts as a turnover regulariser, brought it to 0.91,
    about level with the benchmarks.
- **Portfolios.** Zhang, Zohren and Roberts (2020) apply the same Sharpe loss to portfolio weights.
- **Our cost is far larger.** The 26 bps round trip is 1.0-1.3 sigma of the 10-20-minute move. Lim et al.'s
  costs were a small fraction of their daily moves. Transfer from their results is an extrapolation.

**The form that fits this model: mean-variance utility with a linear cost.** A batch Sharpe ratio is poorly
suited here:

- it is a ratio of batch estimates, so its gradient depends on batch composition;
- over a shuffled batch it is a cross-section of unrelated times, not a time-series Sharpe.

An additive utility is SGD-friendly and has a known optimum:

```
r~ = r_H / sigma_H            (volatility-scaled forward return, winsorised at +-5)
c~ = c / sigma_H              (the 26 bps round trip in sigma units; cheaper in high volatility)
U  = mean( a * r~  -  c~ * |a|  -  (gamma / 2) * a^2 * r~^2 )
loss_pnl = -U
```

- Per sample, the maximiser is a* = clip((E[r~|x] - c~·sign)/(γ·E[r~²|x]), -1, 1). That is the net-edge
  fractional-Kelly aim that `net_edge_kelly` already trades.
- a* is 0 unless the expected move exceeds the cost. This is the no-trade region of the linear-cost
  literature (Garleanu and Pedersen 2013; de Lataillade et al. 2012).
- The quadratic term keeps the optimum inside (-1, 1). A linear utility has corner optima at ±1 and 0,
  which gives bang-bang, overconfident positions.
- σ_H is taken from the causal EWMA of the input bars or from the model's variance head under
  `tf.stop_gradient`. Without the stop-gradient the utility could lower |a| by inflating σ instead of
  sharpening the forecast. That would couple it to the T-perp and HD terms, which also pull σ.

**What it changes here, option A (recommended first).**

- Put the position on the direction head: a_i = 2p_i - 1 = tanh(logit_i / 2).
- New objective `pnl_utility`: `custom_loss(...)` + λ_pnl x Σ_i w_i x loss_pnl,i. Registered with
  `Losses.register_objective`, selected by `LOSS_NAME`, with new config fields `LAMBDA_PNL` and
  `PNL_GAMMA` and the cost taken from the backtest's cost fields.
- No new outputs.
- The existing temperature calibration on the cal block restores probability semantics after training. So
  every strategy that reads calibrated P(up) works unchanged: cq, edge_over_cost, net_edge_kelly.
- The magnitude-equalising λ calibration pass (`training/lambda_calibration.py`) must either skip
  `LAMBDA_PNL` or get its own damping field. Otherwise it rescales the new term to the others' magnitude.

**Option B (later, only if A shows signal).** A dedicated `position_h{i}` head (Dense(1, tanh)), which leaves
P(up) a pure BCE output. It changes the 9-tuple contract: model, serving, calibration, PredictionFrame and
stored predictions.

**Interaction with the existing terms.**

- BCE wants calibrated p. The utility wants p pulled to 0.5 wherever the edge is below cost, and pushed out
  where it exceeds cost. On a no-edge input the two agree: p ≈ 0.5.
- The physics terms see only the trunk's shared gradient (section 1.0).

**Failure modes.**

1. **The flat solution.** With no edge above cost, a ≡ 0 is the correct optimum, not a bug. This is the
   likely outcome here. To tell "flat because there is no edge" from "flat because optimisation stalled",
   log mean |a|, the share of |a| > 0.1, and the training-block utility.
2. **Fitting the training block's drift.** A 10-day block in a trend rewards a constant long. Lim et al.
   volatility-scale for the same reason.
   - Mitigation: demean r~ over the training block, or add a separate constant-exposure term, so that the
     utility rewards timing, not the block's drift.
   - Report the long/short balance and compare against buy-and-hold on dev.
3. **Gradient noise and heavy tails.** The gradient is ∝ r~, which is heavy-tailed. Winsorise and keep the
   batch large (2048 in the micro layout).
4. **Overlapping entries.** Adjacent windows share most of their H-bar path. The loss treats each window as
   an independent entry, which is fine for a selective-entry strategy, but its effective sample is about N/H
   (D-012).
5. **Overfitting in a backtest.** A P&L-trained model optimises the same quantity the backtest reports, so
   dev and test separation matters more (Bailey and Lopez de Prado 2014; Harvey and Liu 2015).

**Evidence of out-of-sample improvement.**

- For futures at daily horizons with costs ≤ 2-3 bps: positive (Lim et al. 2019).
- With costs of 10 bps: only with cost-in-loss regularisation, and then about equal to the benchmarks
  (Lim et al. 2019, exhibit 8).
- For intraday crypto at costs comparable to the move: no published evidence found in this session.

### 1.2 Triple-barrier labels (Lopez de Prado 2018, ch. 3)

**Mechanism.**

- From each entry, an upper barrier at +k·σ·√H, a lower barrier at -k·σ·√H and a vertical barrier at H bars.
- The label is which barrier is touched first.
- σ is a causal volatility estimate: here σ_ewma, or the model's own σ under stop-gradient.
- The label then describes the trade a barrier strategy actually takes, not the close-to-close sign.
- A cost-aware "no trade" class comes from:
  - setting the barriers so that b ≥ a multiple of the cost (b ≥ 2.5 x 26 bps below);
  - masking the vertical-barrier outcomes, or making them a third class.

**Sizing on our data** (CPU, 2025-01-01 .. 2025-07-20, entries every 60 bars, close entry, high/low touches):

| H | k | upper | lower | vertical | median barrier | hit rate needed at 26 bps, p* = 1/2 + 13/b |
|---|---|---|---|---|---|---|
| 60 | 1 | 25.2% | 24.4% | 50.4% | 36 bps | 0.858 |
| 60 | 2 | 5.7% | 6.5% | 87.8% | 73 bps | 0.679 |
| 240 | 1 | 26.4% | 27.7% | 45.8% | 73 bps | 0.679 |
| 240 | 2 | 7.3% | 8.7% | 83.9% | 145 bps | 0.589 |

Reading:

- Barriers wide enough to make a 60% hit rate profitable (b > 130 bps) are touched within 4 hours on only
  about 16% of entries.
- The label set is then small: about 16% of 10 days x 1,440 / 240 entries, or roughly 10 non-overlapping
  barrier outcomes per 10-day block. That is too few.
- The micro loop therefore needs H = 240 with k = 1 (73 bps, p* = 0.68), or the 360-day block.
- This is the concrete form of the "few labels" problem the strategy note raised for meta-labelling
  (strategy-architectures note section 2.6).

**What it changes here.**

- Labels come from the path, so the data layer must build them. `data/windowing.py` already reads
  max(HORIZON_STEPS) future bars per window. It would add a per-horizon barrier label and a "touched" mask to
  the dataset tuple. That is a fifth element, so `train_step`/`test_step` unpacking changes.
- Alternatively, carry the label and mask in spare columns of `extended_trends`. That is less clean.
- The direction head is trained on the barrier label with the vertical outcomes masked. This uses exactly
  the masking path of `DIR_DEADBAND_BPS`, with a different mask.
- The evaluation needs the same labels for `logreg_lags`, so the baseline stays comparable.
- **A barrier strategy with the training geometry.** The discrete engine already has intrabar TP/SL on
  high/low (`strategy/backtest.py:236-243`) and a max_hold. A new registered strategy `barrier_entry` needs:
  - entry when calibrated P(win) ≥ p*(b) + margin;
  - TP and SL at ±k·σ_ewma·√H;
  - max_hold H.
- **Interaction.** It affects only the direction heads' labels. The physics terms and the NLL on the price
  heads are unchanged.

**Failure modes.**

1. **Few labels at wide barriers** (the table above).
2. **Same-bar touches of both barriers.** The engine assumes the stop is hit first (`sl_first`); the labels
   must use the same rule.
3. **Martingale neutrality.** Under a driftless price, barrier geometry does not change the expected gross
   P&L (the strategy note, section 1.4). The label only helps if the inputs carry information about which
   barrier comes first.
4. **Labels below the cost.** Label thresholds below the round-trip cost label cost-sized noise as signal.
   That is the reason for b ≥ 2.5 x cost here.

**Evidence.**

- Lopez de Prado (2018) gives the method, not an out-of-sample test.
- A 2025 Financial Innovation paper reports positive crypto trading results after costs from CUSUM-filtered
  information bars, triple-barrier labels and deep learning. It could not be read in this session (paywall),
  so its design (single split, cost level) is unverified. It is cited as existence, not as evidence.

### 1.3 Cost-sensitive classification (only moves larger than the round trip count)

**Mechanism.**

- Elkan (2001): when errors have unequal costs, change the labels or weights so the classifier's decision
  threshold matches the cost.
- Here: train direction only on samples with |r_H| > 26 bps (the deadband), or weight each sample by
  |r_H| - c (a P&L-weighted BCE).

**Our data** (CPU sizing): the share of bars whose H-bar move exceeds 26 bps is 12.6% / 17.5% / 21.8% at
10 / 15 / 20 bars, 41.1% at 60, 55.2% at 120 and 66.3% at 240.

- At the reference horizons a 26 bps deadband throws away about 80% of the labels.
- At 1-4 hours it keeps half or more.

**What it changes.** `DIR_DEADBAND_BPS: 26` is config only (section 1.0).

- The trained head then estimates P(up | |r| > 26 bps).
- The model's own variance gives P(|r| > c) ≈ 2(1 - Φ(c/σ)).
- So P(profitable long at fixed hold) ≈ P(up | move) x P(|r| > c), a calibratable product.

**Failure modes.**

1. Direction metrics are then reported on the masked population, so their numbers are not comparable with
   earlier runs. The logreg baseline moves with them (same deadband), which keeps the comparison fair.
2. The mask discards information about small moves. If the signal lives in small moves, the variant loses.

**Evidence.** A standard ML result (Elkan 2001). No crypto-specific out-of-sample evidence found.

### 1.4 Meta-labelling on a primary rule

**Mechanism** (Lopez de Prado 2018 ch. 3; Joubert 2022; Meyer, Barziy and Joubert 2023):

- a primary rule sets the side;
- a secondary classifier predicts P(the primary's trade is profitable after costs) and sizes it.

**Not in the plan.**

- The primaries have no gross edge. Our CPU sizing: `logreg_lags` gross edge 0-4.5 bps per trade at 15-240
  minutes. The simple rules of the strategy note, section 1.5: about ±1 bp.
- Meta-labelling raises precision on a primary that already has an edge. It cannot create one (Joubert
  2022, and the strategy-architectures note section 2.6).
- The label shortage of 1.2 applies too.
- Revisit it if E1-E3 find a primary with a gross edge above cost.

### 1.5 Drawdown-aware objectives (CVaR, maximum drawdown)

**Mechanism.**

- CVaR is differentiable through the Rockafellar-Uryasev (2000) auxiliary variable α:
  CVaR_q(L) = min_α α + E[(L - α)+]/(1 - q). Deep hedging trains networks on this form (Buehler et al. 2019).
- Conditional drawdown (Chekhlov, Uryasev and Zabarankin 2005) needs the ordered P&L path.

**What it changes.**

- CVaR: one extra trainable scalar and a term in the `pnl_utility` objective. It replaces or complements the
  quadratic risk term.
- Drawdown: a sequential P&L path, which fully shuffled batches do not provide (section 1.0). It would need
  chronological batches, a data-pipeline change with its own speed cost (D-018).

**Why not now.**

- Drawdown is driven by sizing and by the sign of the edge. With no edge above cost, a drawdown-penalised
  loss converges to flat, as the utility already does.
- With an edge, fractional sizing fitted on cal (E4) controls drawdown without retraining. Drawdown scales
  roughly linearly with the Kelly fraction (MacLean, Thorp and Ziemba 2010).

**Evidence.** CVaR training works for hedging (Buehler et al. 2019). No evidence was found that
drawdown-penalised training beats post-hoc sizing for directional intraday trading.

## 2. Is the owner's target reachable on this data?

### 2.1 Our own evidence

- **Micro loop, 10 hypotheses.** No model variant beats `logreg_lags` in direction at 10 minutes to 5 hours,
  on 10 or 360 days of training, 60 or 240-bar windows, close-only or OHLCV plus 14 indicator families. The
  AUC is 0.51-0.53. Calibrated P(up) stays in [0.43, 0.58] on 98% of bars (H3).
- **Strategy study v1.** Every one of 20 strategies loses on dev. The best gross edge is +4.4 bps per trade
  against a 26 bps cost.
- **CPU sizing, this note.** Logistic regression on 3 volatility-scaled lagged H-bar returns, fitted on
  2025-01 .. 2025-04 and scored on 2025-05-01 .. 2025-07-20. n_eff = selected bars / H; the ± is the 95%
  band. Selected bars cluster in time, so the band is an optimistic estimate.

| H (bars) | bars traded | n_eff | hit rate | gross bps per trade |
|---|---|---|---|---|
| 15 | all | 7,679 | 51.3% ± 1.1 | +0.15 |
| 15 | top 10% | 767 | 53.2% ± 3.5 | +0.02 |
| 15 | top 2% | 153 | 53.9% ± 7.9 | +0.89 |
| 60 | all | 1,919 | 50.5% ± 2.2 | +0.91 |
| 60 | top 10% | 191 | 54.0% ± 7.1 | +1.88 |
| 60 | top 2% | 38 | 57.4% ± 15.9 | +1.50 |
| 240 | all | 479 | 51.6% ± 4.5 | -0.55 |
| 240 | top 10% | 47 | 55.3% ± 14.3 | +4.45 |
| 240 | top 2% | 9 | 50.5% ± 32.7 | +0.97 |

- **A hit rate in the mid-50s is not an edge.** Selective hit rates of 53-57% coexist with gross edges of
  0-4 bps, because winners and losers have nearly the same size at a fixed hold. That is why the owner's
  target must be paired with the payoff geometry, which p* = 1/2 + 13/b does.

### 2.2 The literature, intraday BTC direction from price and volume history

| source | data, horizon | inputs | result | after costs |
|---|---|---|---|---|
| Jaquart, Dann and Weinhardt (2021) | BTC, 2019-03 .. 2019-12, 1-60 min | technical, blockchain, sentiment, cross-asset | accuracy 51.5% (1 min) to 55.8% (60 min, ensemble); best single model 56.0% (LSTM, 60 min); predictability rises with horizon | top/bottom 1% quantile long-short: up to +116% over 3 months gross (LSTM, 60 min), **negative for every model** at a 30 bps round trip |
| Akyildirim, Goncu and Sensoy (2021) | 12 cryptocurrencies, 15 min to daily | price-based technical | "about 55-65%" average accuracy (abstract); SVM best | not verified in this session |
| Jaquart, Köpke and Weinhardt (2022) | 100 cryptocurrencies, daily, cross-sectional | price history | accuracy 52.9-54.1% | long-short Sharpe 3.1-3.2 after costs. The edge comes from **breadth** (100 coins, daily), not from the hit rate |

Reading:

- The best honest intraday BTC number close to our setup is about 56% at 60 minutes, with more inputs than
  we have, and it lost money after costs.
- Our logistic baseline, at 50.5-54% on the 2025 data, sits below it. Estimate: 2025 BTC is more efficient
  than 2019 BTC; Jaquart et al. (2021, discussion) cite evidence that bitcoin efficiency rises over time.
- The one route to a profit with a mid-50s hit rate in the literature is many independent bets (Grinold 1989):
  a cross-section of assets at daily horizons. A single instrument gets breadth only from trade count, and
  our cost caps that.

### 2.3 Sources that move the number

| source | evidence | horizon of the signal | usable at 13 bps per side? |
|---|---|---|---|
| Order-book depth and imbalance | Wang (2025), BTC/USDT Bybit, 100 ms snapshots: 71-73% binary accuracy at 0.5-1 s with 40 book levels, 58% with 5 levels; no P&L reported | under ~1 s | No: the move is far below the cost |
| Order flow and trade flow | Cont, Kukanov and Stoikov (2014): order-flow imbalance explains contemporaneous price changes; Silantyev (2019), BitMEX XBTUSD: trade-flow imbalance explains contemporaneous changes better; Kolm, Turiel and Westray (2023): predictable at very short horizons (Nasdaq equities) | seconds; contemporaneous | No for a taker; the signal decays within a minute |
| Perpetual futures leading spot | Alexander, Choi, Park and Sohn (2020): BitMEX derivatives lead price discovery; Plazuelo Pascual et al. (2025): futures tend to lead, mixed in high volatility | seconds to minutes | Only as an execution-speed signal; no evidence of a 60% hit at hours |
| Funding rates | Inan (2025): funding rates predict next-period funding rates | 8 h | No source found in this session showing funding predicts intraday BTC direction |
| Deep LOB models in general | Briola, Bartolucci and Aste (2024): high forecasting accuracy "does not necessarily correspond to actionable trading signals" | ticks | A warning, not a lever |

Reading:

- Every source with a large directional number works at seconds on order-book or trade data, where our
  13 bps taker cost exceeds the move.
- None of it is in the OHLCV file. Taker-buy volume is in Binance klines but not in `Bitcoin_BTCUSDT.csv`.
- Adding any of them is a new data source, outside the MVP (VISION). That makes it an owner decision (E5).

### 2.4 How many trades the target needs (why micro cannot confirm it)

- **Separating 60% from the baseline's ~53%** (one-sided α = 0.05, power 0.8) needs
  n ≈ (1.645 + 0.84)² x 0.24 / 0.07² ≈ 300-380 independent trades.
- **On a 30-day dev block at H = 240:**
  - about 180 non-overlapping outcomes in all;
  - on all of them the hit rate is known to ±7 pp;
  - a 10% selective filter leaves about 18 outcomes, ±23 pp.
- **Estimated trade frequency for a selective 1-4 hour strategy:** about 0.5-2 trades a day. The target
  therefore needs about 6-20 months of walk-forward dev data.
- **Consequence for the loop.**
  - The micro loop can test whether a directional signal exists: AUC against `logreg_lags` with n_eff
    noise, pooled over 3 seeds.
  - It can screen a trading result.
  - The owner's hit-rate target can only be confirmed on the long-history walk-forward folds (NT-041),
    judged under D-025.

### 2.5 Verdict

- **Stable > 60% hit on all bars from OHLCV history: not reachable.**
  - Our 10 hypotheses and the CPU sizing: 50.5-51.6% for the best baseline.
  - The closest literature: ≤ 56%, and unprofitable at 30 bps.
- **60% hit of taken trades, selective and cost-aware: unlikely from OHLCV, not excluded** at 1-4 hour
  holds with barriers ≥ 130 bps.
  - The baseline's selective hit rates are 53-57%, but with bands of ±7-16 pp.
  - It would be at about 0.5-2 trades a day.
  - It would be confirmable only on months of walk-forward data.
- **Drawdown < 5%: reachable by sizing whenever the edge is positive.**
- **The honest expectation (an estimate).** E1-E3 return negatives like iterations 1 and 2. That would point
  to the information source (E5), an owner decision, not to more modelling.

## 3. The micro-loop plan (D-041)

**Common protocol for every experiment.**

- **Layout.** Micro layout on `Bitcoin_BTCUSDT.csv`, as in `configs/scenarios/micro_horizons.yaml`: N_FOLDS 2,
  129,600 windows, fold -2 only, about 10 days of training, 30-day dev block 2025-07-27 .. 2025-08-28, batch
  2048, EPOCHS 30. Fold -1 is never scored.
- **Seeds.** 3 per variant (the I2-duel showed one seed's spread is as large as the effects).
- **Label.** Quick-sweep, descriptive. Each experiment gets a journal row in `runs/experiments/micro_loop_v1/LOG.md`.
- **GPU estimates** assume about 4-5 minutes per cell (the D-041 micro timing). They are not measured
  for the new objectives.

**Metrics, for every cell.**

- **Direction.** AUC per horizon minus `logreg_lags` on the same labels, with a block-bootstrap z.
- **Trading, on the dev block:**
  - net hit rate of taken trades (`hit_rate` in `strategy/performance.py`) with a ±1.96·sqrt(p(1-p)/n_trades) band;
  - gross edge per trade with its bootstrap CI;
  - net PnL after the default costs (13 bps per side, next-open fills);
  - maximum drawdown;
  - trade count;
  - buy-and-hold on the same block.

**Two pass lines, fixed now, before any run.**

- **Signal gate.** Seed-mean AUC - logreg_lags ≥ +0.02 on at least 2 of 3 horizons, pooled bootstrap
  z ≥ 2 on each of those, and no horizon worse than -0.01.
  - Today's deltas are -0.010 .. +0.010 (I2-duel), so this is a large bar.
- **Trading gate.**
  - Net PnL > 0 in at least 2 of 3 seeds.
  - Seed-mean gross edge per trade > 26 bps with a bootstrap CI lower bound > 13 bps.
  - At least 30 trades per cell.
  - Net hit rate of taken trades ≥ p*(geometry) at the point estimate. With 30 trades this is a
    screening signal, not proof.
- A pass on either gate sends the variant to the long-history walk-forward (NT-041) and a D-025 study. It is
  never used to change a default directly.

### E1: cost-sensitive direction labels (config only)

**Hypothesis.** Direction exists on moves larger than the round trip but is drowned by the 80-90% of small
moves at short horizons. Masking them out of the loss (and out of the baseline's labels) raises the AUC over
`logreg_lags`.

**Change.** A new scenario `configs/scenarios/micro_pnl_e1.yaml`:

- the micro layout;
- `HORIZON_STEPS: [60, 120, 240]` and `EXTENDED_TREND_PERIODS: [60, 120, 240]`;
- two variants: `db5` (control, `DIR_DEADBAND_BPS: 5`) and `db26` (`DIR_DEADBAND_BPS: 26`);
- seeds [0, 1, 2].

6 cells, about 25 GPU-min (an estimate).

**Scoring.**

- The run report: direction, which is on the masked labels for both model and baseline.
- `hitrate_buckets.py`: taken-trade hit rate by confidence bucket.
- `scenario rescore` with `edge_over_cost` (max_hold = H) and `net_edge_kelly` for net PnL and MDD.

**What a negative means.** Small, untradable moves were not hiding the signal. The inputs carry no
direction even on large moves at 1-4 hours. This strengthens the "information, not target" reading.

### E2: P&L utility objective on the direction heads (implementer item first)

**Implementer item.** New objective `pnl_utility` in `losses/functions.py` (section 1.1, option A):

- `LAMBDA_PNL`, `PNL_GAMMA` and `PNL_SIGMA_SOURCE` (ewma or model, stop-gradient) in `core/config.py` with
  metadata (NT-029);
- the λ-calibration pass skips `LAMBDA_PNL`;
- the return is winsorised at ±5σ and demeaned over the training block;
- the 34-slot `LossComponents` is kept, with the utility logged through an added component, or a documented
  reuse of a zero slot.

Acceptance:

1. On synthetic data with a planted edge above cost, mean |2p - 1| grows and the utility rises.
2. On synthetic data without an edge, the positions go to 0 (the flat solution).
3. The gradient is finite on the stability cases (D-026).
4. `sec_per_step` is not slower than `custom_loss` within noise (D-018).
5. `LOSS_NAME: custom_loss` runs give bit-identical results.

About one implementer day (an estimate).

**Hypothesis.** Training on net P&L (owner's point 3) moves the direction heads toward the bars where the
move exceeds the cost, and produces a dev gross edge per trade above cost that BCE alone does not.

**Change.** Scenario `micro_pnl_e2.yaml`:

- horizons [60, 120, 240];
- `LOSS_NAME: pnl_utility`;
- variants: `lam0` (control, identical to `custom_loss`), `lam_a` (λ_pnl set so the utility term is about
  0.5x the direction loss at initialisation) and `lam_b` (2x `lam_a`);
- `PNL_GAMMA` fixed at 1;
- seeds [0, 1, 2].

The two λ values are fixed from the initial loss magnitudes before training, not tuned on dev. 9 cells,
about 40 GPU-min (an estimate).

**Scoring.** As E1, plus the mean |2p_raw - 1|, the share of bars with |a| > 0.1, and the long share, to
diagnose the flat solution and drift fitting.

**What a negative means.**

- Adding the P&L to the target (owner's point 3) is not the bottleneck.
- The flat solution on the training block itself means the model finds no net edge even in-sample. That is
  the strongest possible form of this negative.

### E3: triple-barrier labels and a matching barrier strategy (implementer item first)

**Implementer item.**

1. In `data/windowing.py`: per-horizon barrier labels (upper-first 1, lower-first 0) and a touched mask from
   the close path, barriers ±k·σ_ewma·√H with a causal EWMA (half-life 60). Same-bar ties follow `sl_first`.
2. A config switch `DIRECTION_LABELS: sign | barrier` plus `BARRIER_K`. The objective uses the provided mask
   instead of the deadband.
3. The `logreg_lags` baseline trains on the same labels.
4. A registered discrete strategy `barrier_entry` with TP/SL at the same barriers, max_hold H, and entry at
   calibrated P(win) ≥ p*(b) + margin, with the margin fitted on cal.
5. A look-ahead test for the labels and the strategy (`assert_no_lookahead`).

About 1-1.5 implementer days (an estimate).

**Hypothesis.** "Which volatility-scaled barrier comes first" is more predictable from the inputs than the
close-to-close sign, and the taken-trade hit rate reaches p*(b).

**Change.** Scenario `micro_pnl_e3.yaml`:

- horizons [60, 120, 240];
- variants: `sign` (control), `bar_k1` (k = 1) and `bar_k15` (k = 1.5);
- seeds [0, 1, 2].

k = 2 is excluded on the micro layout: about 10 labels per training block (section 1.2). 9 cells, about
40 GPU-min (an estimate).

**Model-free bar, before the GPU runs.** The lead runs the logistic baseline on barrier labels on the CPU
(the E3 twin of section 2.1's table). This fixes the number E3 must beat.

**What a negative means.**

- Path-dependent, volatility-scaled labels do not reveal a signal the sign label hides.
- The barrier geometry can then only reshape the P&L distribution, as optional stopping predicts.

### E4: drawdown control by sizing (CPU only; only after a trading-gate pass)

**Hypothesis.** An edge that passes E2's or E3's trading gate keeps net PnL > 0 with MDD < 5% at a fractional
size.

**Change.** A strategy study YAML:

- `net_edge_kelly` with f ∈ {0.1, 0.25, 0.5};
- or `barrier_entry` with size ∈ {0.1, 0.25, 0.5};
- the size is chosen on the cal block as the largest with a simulated 95th-percentile MDD < 5%.

Rescored on the stored predictions: `neural-trade scenario rescore <scenario> --study <study>`.

**Pass.** Dev MDD < 5% with net PnL > 0 at the cal-chosen size.

**What a negative means.** The edge is too thin or too unstable to size under a 5% drawdown.

### E5: a new information source (owner decision; only if E1-E3 are negative)

- The cheapest candidate is Binance kline taker-buy volume (trade-flow imbalance at 1-minute resolution),
  then perpetual basis and funding. Order-book data is the most informative and the most expensive.
- Each needs a data item and the owner's approval (VISION "Not in the MVP": new data sources).
- The literature places their signal mostly at seconds (section 2.3). Estimate: even the cheapest may not
  reach 1-4 hour horizons.

**Not in the plan.**

- Meta-labelling (section 1.4).
- Drawdown-penalised training (section 1.5).
- A dedicated position head (option B), unless E2 passes.
- A batch-Sharpe loss (section 1.1).

## 4. What the model must output, and how the existing layer scores it

**For the goal to be checkable, each decision bar needs:**

1. **A side and a no-trade decision**: long, short or flat.
2. **A calibrated probability that the trade wins net of cost**, under the geometry it will be traded with:
   - fixed hold: P(side·r_H > c);
   - barrier: P(the favourable barrier is touched first).
   It is calibrated on the cal block, like today's temperature scaling.
3. **The geometry**: TP, SL and max hold, so that p*(b) is defined.
4. **A size**, for the drawdown part.

**How today's outputs map.**

- **Fixed hold.** The calibrated P(up) (with E1's deadband: P(up | |r| > c)) and the variance head's
  P(|r| > c) give P(win) ≈ P(up | move) x P(|r| > c).
  - `edge_over_cost` already enters on the Gaussian expected move exceeding the cost.
  - `net_edge_kelly` (NT-077 exposure mode) turns the net edge into a fractional-Kelly exposure with a
    no-trade band. It is the natural scorer for E2, whose optimum is the same aim (section 1.1).
- **Barrier.** E3's calibrated P(up-first) is P(win) for a long. `barrier_entry` trades it, and the discrete
  engine's intrabar TP/SL executes the geometry.

**Scoring the goal with what exists.**

- `neural-trade scenario rescore <scenario> --study <study>` re-scores stored predictions
  (`predictions_oos.npz`, `predictions_cal.npz`) on the CPU. Every threshold is fitted on cal.
- `strategy/performance.py` already reports:
  - `hit_rate`, the share of trades with net P&L > 0, which is the owner's "hit rate of taken trades" after
    costs;
  - `hit_rate_gross`;
  - `max_drawdown`;
  - the trade count.
- NT-077 added:
  - `breakeven_cost_bps`;
  - `gross_edge_per_trade_bps`;
  - the size-matched random null (discrete) and the circular-shift timing null (exposure).

**Missing, and small.**

1. The hit rate's noise band (±1.96·sqrt(p(1-p)/n) for non-overlapping trades) and the p*(geometry) column in
   the rescore summary or in an analysis script like `strategy_study_v1/analyze.py`.
2. An "owner target" row per configuration: lower bound of the net hit rate ≥ max(0.60, p*), net PnL > 0,
   MDD < 5%, n_trades ≥ 100.

Both are analysis-script work (lead or implementer) and need no model change.

## Proposed follow-up items (for the lead to triage)

1. **E1 scenario** (config only; lead or experimenter; about 25 GPU-min).
2. **Implementer item:** the `pnl_utility` objective and its config fields (E2, section 3).
3. **Implementer item:** triple-barrier labels, the `DIRECTION_LABELS` switch, the `barrier_entry` strategy,
   and the matching `logreg_lags` labels (E3).
4. **Small item:** the hit-rate noise band, p* and the owner-target row in rescore output (section 4).
5. **Owner question, only after E1-E3:** a new information source (E5). Recommendation: Binance
   taker-buy volume first, the cheapest.

## References

Opened or checked in this session (abstract, full text or search record):

- Akyildirim, E., Goncu, A. and Sensoy, A. (2021). Prediction of cryptocurrency returns using machine learning. *Annals of Operations Research* 297, 3-36. https://link.springer.com/article/10.1007/s10479-020-03575-y (abstract figure only)
- Anastasopoulos, A., Gradojevic, N., Liu, F., Maynard, A. and Tsiakas, I. (2024). Order flow and cryptocurrency returns. EFMA 2025 working paper. https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/OrderFlowpaper.pdf (daily cross-section; out-of-sample R² 0.66%; not used for intraday claims)
- Bengio, Y. (1997). Using a financial training criterion rather than a prediction criterion. *International Journal of Neural Systems* 8(4), 433-443. https://www.worldscientific.com/doi/10.1142/S0129065797000422
- Briola, A., Bartolucci, S. and Aste, T. (2024). Deep limit order book forecasting: a microstructural guide. arXiv:2403.09267. https://arxiv.org/abs/2403.09267
- Inan, E. (2025). Predictability of funding rates. SSRN 5576424. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424
- Jaquart, P., Dann, D. and Weinhardt, C. (2021). Short-term bitcoin market prediction via machine learning. *Journal of Finance and Data Science* 7, 45-66. https://doi.org/10.1016/j.jfds.2021.03.001 (full text read: Tables 3 and 5, section 4.3)
- Jaquart, P., Köpke, S. and Weinhardt, C. (2022). Machine learning for cryptocurrency market prediction and trading. *Journal of Finance and Data Science* 8. https://www.sciencedirect.com/science/article/pii/S2405918822000174 (abstract figures)
- Kolm, P. N., Turiel, J. and Westray, N. (2023). Deep order flow imbalance: extracting alpha at multiple horizons from the limit order book. *Mathematical Finance* 33, 1044-1081. https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413
- Lim, B., Zohren, S. and Roberts, S. (2019). Enhancing time series momentum strategies using deep neural networks. *Journal of Financial Data Science* 1(4). https://arxiv.org/abs/1904.04912 (full text read: section VI, exhibit 8)
- Plazuelo Pascual, J., Tardon Rubio, C., Toro Cebada, J. and Hernando Veciana, A. (2025). Price discovery in cryptocurrency markets. arXiv:2506.08718. https://arxiv.org/abs/2506.08718
- Silantyev, E. (2019). Order flow analysis of cryptocurrency markets. *Digital Finance* 1, 191-218. https://link.springer.com/article/10.1007/s42521-019-00007-w
- Wang, H. (2025). Exploring microstructural dynamics in cryptocurrency limit order books: better inputs matter more than stacking another hidden layer. arXiv:2506.05764. https://arxiv.org/abs/2506.05764 (HTML full text read: tables 1-4)
- Zhang, Z., Zohren, S. and Roberts, S. (2020). Deep learning for portfolio optimization. *Journal of Financial Data Science* 2(4). https://arxiv.org/abs/2005.13665
- "Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning" (2025). *Financial Innovation*. https://link.springer.com/article/10.1186/s40854-025-00866-w (search record only; full text and author list not accessible in this session)

Standard references cited from knowledge, not re-opened in this session:

- Alexander, C., Choi, J., Park, H. and Sohn, S. (2020). BitMEX bitcoin derivatives: price discovery, informational efficiency, and hedging effectiveness. *Journal of Futures Markets* 40(1), 23-43. https://doi.org/10.1002/fut.22050
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management* 40(5), 94-107. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Buehler, H., Gonon, L., Teichmann, J. and Wood, B. (2019). Deep hedging. *Quantitative Finance* 19(8), 1271-1291. https://arxiv.org/abs/1802.03042
- Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). Drawdown measure in portfolio optimization. *International Journal of Theoretical and Applied Finance* 8(1), 13-58. https://doi.org/10.1142/S0219024905002767
- Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1), 47-88. https://arxiv.org/abs/1011.6402
- de Lataillade, J., Deremble, C., Potters, M. and Bouchaud, J.-P. (2012). Optimal trading with linear costs. *Journal of Investment Strategies* 1(3). https://arxiv.org/abs/1203.5957
- Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of IJCAI 2001*, 973-978.
- Garleanu, N. and Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance* 68(6), 2309-2340. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080
- Grinold, R. C. (1989). The fundamental law of active management. *Journal of Portfolio Management* 15(3), 30-37.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1), 13-28. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489
- Joubert, J. F. (2022). Meta-labeling: theory and framework. *Journal of Financial Data Science*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4032018
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley (ch. 3: triple-barrier labels and meta-labelling).
- MacLean, L. C., Thorp, E. O. and Ziemba, W. T. (2010). Long-term capital growth: the good and bad properties of the Kelly and fractional Kelly capital growth criteria. *Quantitative Finance* 10(7), 681-687.
- Meyer, M., Barziy, I. and Joubert, J. F. (2023). Meta-labeling: calibration and position sizing. *Journal of Financial Data Science* 5(2).
- Moody, J. and Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks* 12(4), 875-889. https://doi.org/10.1109/72.935097
- Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk* 2(3), 21-41. https://doi.org/10.21314/JOR.2000.038

**Project evidence.**

- `runs/experiments/micro_loop_v1/LOG.md` (H1-H5, I2-duel, the readings);
- `runs/experiments/strategy_study_v1/REPORT.md`;
- `docs/research/2026-09-29-strategy-architectures/README.md` (sections 1.2-1.5, 2.6, 5);
- code: `losses/functions.py:490-809`, `losses/registry.py`, `metrics/tf_direction.py`,
  `models/gru_attention.py`, `training/custom_model.py:121-123`, `strategy/variance_strategies.py`,
  `strategy/backtest.py`, `strategy/performance.py:77-79`.

The CPU sizing script ran in the session scratchpad and is not committed; section 1.2's table and section
2.1's table are everything it printed that this note uses.
