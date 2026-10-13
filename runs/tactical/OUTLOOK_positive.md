# Positive outlook: geometry, physics, regression, synthetic interpretable indicators

**For the next tactical session.** Owner, 2026-10-13: "Я хочу позитивный аутлук насчет геометрии, физики, регрессии и
синтетических интерпретируемых индикаторов. Что тут получилось хорошо и где у нас перспективы. Этим займется следующая
сессия". The split of work: TACTICAL.md "Division of work".

Every point cites the journal ([LOG.md](LOG.md)). The negative side is in [REVIEW_2026-10-13.md](REVIEW_2026-10-13.md).
This file is the starting point: what worked, and where it leads.

## What worked

### Regression: the strongest, fastest and most interpretable model we have
- **Beats the default network.** A 13-feature logistic regression on scale-free textbook indicators and returns beats the
  default network's direction AUC by +0.036 [+0.006, +0.067] on the same blocks, in 3 s of CPU (H27).
- **Nothing beats it.** No network, MLP, GRU, PatchTST-lite or TCN beats it, from 11.5 days to 3 years of training (H28, H35, H36).
  The rebuilt Keras model equals sklearn exactly (H28), so we have a verified, minimal, readable direction model.
- **It is a precise instrument.** The monthly walk-forward lab measures a whole market history in seconds (H51-H53).
  It found the decay, the bid-ask share and the step breaks (H59, H72), and nothing else in the project could have.
- **It held a real edge.** On next-open fills it had a 59-60% top-10% hit in 2018-2020 (H59). The method was right for the
  market it saw.
- **Longer training helps.** 365 days beat 60 days in every cell (H56).

### Geometry: interpretable, and useful where the target is the size of moves
- **Calibration.** In the network, indicator geometry made direction better calibrated: log loss +0.058 [+0.019, +0.097]
  (H24), +0.052 with indgrad (H43). It reached the regression's scale-free form in a readable way.
- **Readable nonlinearity.** In the triple barrier, geometry lifts a logistic primary by +0.48 bps over rich features and
  +0.82 over shuffled ones, replicated on 2017-2019 (H47). A linear model with geometry matches boosting: interpretable
  nonlinearity.
- **Volatility.** Geometry adds about +0.02 Spearman over a bare HAR at minute scale (H47, +0.0165 [+0.009, +0.024]).
- **An honest TA reading.** The Bollinger "squeeze" measures volatility persistence (H47). We now know what the classic
  signal actually encodes.
- **Real BTC windows.** The report draws every geometry variant on real BTC windows (report/tactical_report.html), so the
  views exist.

### Physics terms (D-003): they help where they belong
- **Variance head.** On the variance head the physics terms help: +0.020 [+0.002, +0.037] Spearman with 6 of 6 slices up (H75),
  firmed to +0.0116 [+0.0031, +0.0202] on 3 seeds (H77). The pre-registered rule was met both times.
- **No harm.** They never hurt the variance head, and they close 15-50% of the gap to HAR + time of day.
- **Under-trained.** These were shrunk runs: 4 epochs, 6 slices. The effect was measured on under-trained nets, so the
  ceiling is unknown.
- **Direction is fed.** The gradient probe (H42) shows direction gets 30-45% of the trunk gradient. The losses are not
  starving it, so the physics terms are free to shape the variance side.

### Synthetic interpretable indicators: they learn volatility structure
- **Volatility structure.** Free-form learned indicator series (causal TCN, K = 12) capture volatility: +0.07..+0.12 Spearman over random series (H80).
- **Geometry on top.** Geometry on those series adds +0.013..+0.025 more (arm B).
- **Fairness gap.** The arms saw no realised-volatility statistics, which is why they trail HAR (H80 caveat).
- **Learned volatility.** Learned volatility predictors beat the best standard forecast on ranking:
  - minute ridge +0.014, tower +0.016..+0.019 over HAR + time of day (H39);
  - hourly boosting on context +0.045 [+0.022, +0.067] (H39);
  - the volatility tower beats trailing realised volatility by +0.034 (H36).
- **JEPA.** Masked-context JEPA learned volatility information that kept growing with pretraining, 0.193 → 0.242 (H41, H46).

## Prospects for the next session

Each point names the test that would decide it.

1. **Regression as the interpretable core.**
   - Add the regression direction model to the MVP registry. It is the "For the MVP lead" candidate already filed.
   - Use it as the frozen baseline every network must beat.
   - Readable coefficients per period show what changed in the market (H68 rotation).
   - Test: a registry model plus a paired MVP test (D-025).
2. **Geometry for the size and shape of moves.**
   - Point geometry at volatility, ranges and barrier hits, not at the sign (H47 volatility, H47 barrier).
   - Test: geometry over HAR with the in-window RV statistics included, monthly walk-forward, cutoff 2025-07-20, then
     the Binance forward data.
3. **Physics on the variance head, fully trained.**
   - H77 ran 4 epochs. Try a full-length training of the PRICE_HEAD-none network with and without physics, judged on
     variance Spearman and QLIKE against HAR.
   - It is the cleanest D-003 test available: physics judged on its natural axis.
4. **Synthetic indicators that see volatility.**
   - Repeat H80 with RV statistics among the generator's inputs, and volatility as the target.
   - Read the learned series as new, interpretable "volatility indicators" drawn on price (D-027 views).
   - Test: their gain over HAR + time of day, Holm, then forward.
5. **One combined, readable model.**
   - Regression for the side, a geometry + physics variance head for the size, and synthetic series as inputs.
   - Judge it as a risk model (interval coverage, CRPS, drawdown control), not by hit rate.

Constraints that still hold:
- cutoff 2025-07-20;
- the forward data only from the separate Binance file;
- next-open fills;
- thresholds from the past;
- SPEC before results.
