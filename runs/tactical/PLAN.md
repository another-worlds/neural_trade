# Tactical plan (living; the owner's ideas and the lead's hypotheses, newest first)

Owner, 2026-10-08: "обязательно пиши в план" - every idea the owner raises goes here with its status, so nothing is
forgotten. Status: idea / planned / running / done (journal row) / dropped (reason).

## 1. The network as an autonomous combinator of indicator "convolutions" (owner's original intent) - planned
Technical analysis draws lines over indicator values and reads their **geometry** (slope, crossings, the distance of
price to a line, price/indicator divergence, band squeeze before a breakout) to call trends. Today the 54 learned
indicator channels (learnable-period smoothing = learnable convolutions) are only mixed by GRU/attention/transformers;
no layer reads their geometry explicitly. Plan: a **geometry layer** over the learned indicators (per channel: slope over
k bars, sign changes / crossings between instances and with price, distance to price in volatility units, divergence),
feeding the direction head; combined with the confidence head (expected move size) the way TA uses "squeeze, then
breakout": confidence says *whether* a big move comes, geometry says *which way*. Needs code (implementer), a Config
switch, default off; judged by the hill-climb metric and by the ensemble strategy curves (section 3).

## 2. Remove the price head: 6 outputs (3 horizons) vs 2 outputs (1 horizon) - done (journal H19)
**Result:** without price the confidence improves (CRPSS +0.024, borderline) and direction AUC is unchanged; keep 3 horizons: their
ensemble reaches 60-61% on the most confident 5-10% of bars with the first gross move above the random null (CI still touches 0),
the 1-horizon net only 57%. Next: no-price 3-horizon as the new base for the indicator and loss rounds; fix the direction over-confidence (log loss).
Evidence: the price head has no information (corr 0.01-0.03, best-shrink skill ~0.003) yet six loss terms train it
through the trunk and the shared per-horizon tower; it adds nothing in head combinations (journal H14, H15).
Switches being built: PRICE_HEAD 'none' (no price layers; point, trend, coherence, IFE, vol, casimir, vac = 0; NLL/CRPS
with centre 0) and ACTIVE_HORIZONS (train a subset). Experiment (owner): the 3-horizon no-price network used as an
ensemble (direction + confidence on h0-h2) **vs** a 1-horizon no-price network (direction + confidence on h1), same
blocks, the 6x2 design; compared on h1 direction and confidence and on the ensemble strategy curves.

## 3. The model is an implicit ensemble: judge head combinations, not single heads - started
Owner: the strategy decides from all heads together; we only tested heads one by one. First numbers (H15): the mean
of the 3 horizons and "all 3 horizons agree" raise direction AUC by +0.005..+0.022; accuracy rises with the confidence
threshold (single h1 0.53 -> 0.58 on the top 10%; agree3 + high predicted variance 0.58 -> 0.61 on the top 2-5%), but the
gross move per trade stays inside the random null. Next: thresholds fitted on the calibration block (not the
validation block itself), a proper backtest with the size-matched random null, and the same curves for every variant.

## 4. Direction gated by the confidence head (regime) - idea
The variance head knows the size of the coming move (Spearman ~0.24 with |move|); direction is slightly better in the
high-variance third for h1/h2 (+0.007 AUC). A gate that switches direction behaviour by predicted volatility
(mean reversion in quiet regimes, as on 2022-04-16; continuation in trends) - needs code.

## 5. Loss architecture - hypotheses from the code and the probe (probe done: journal H16)
- **Measured (H16):** after 3 epochs the trunk gradient is NLL 31%, t_perp 27%, direction 11%; the indicators' gradient is
  NLL 30%, t_perp 28%, direction 10%, and the price terms (point, coherence, trend, IFE) and hd point against the total there.
  So the learned indicator periods are tuned mostly for the variance terms, while the direction - the owner's target for
  indicator geometry - gives them ~10% of their gradient. Hypotheses: (a) without the price terms (section 2) the indicator
  gradient stops fighting itself; (b) a larger direction weight or a direction-only gradient for the indicators (a separate
  indicator optimizer fed by the direction loss) tunes the periods for direction.
- The price head's six terms (incl. the extended-trend "momentum prior" that pulls the prediction towards the past move)
  feed noise into the shared trunk and the indicator gradients. Covered by section 2.
- Direction is over-confident: |p - 0.5| averages 0.10-0.12 at AUC 0.54; Brier 0.255-0.27 is worse than the constant
  0.25. Hypothesis: shrink (calibrate) P(up) or regularise the direction head; AUC unchanged, log loss and the metric up.
- Physics terms: the owner suspects they hurt; measured effect so far is nil either way (H12/H13: +0.36 noise units,
  CI over 0). The gradient probe will show each term's share and its conflict with CRPS / direction. D-003 (keep,
  fix the maths) is the owner's; any change goes to the owner with the evidence.

## 6. Model architecture
- One Dense(16) tower per horizon shared by price, direction and variance; a Flatten -> Dense(32) of about a million
  parameters before it, on ~16,560 training windows. Hypotheses: separate towers per head (or fewer heads, section 2);
  HEAD_POOL 'mean' instead of 'flatten' to cut the over-fitting layer. Widening the tower is not expected to help (NT-104:
  more capacity never beat a 3-lag logistic regression).

## 7. Indicator learning - the project's core claim, untested
54 learnable periods (14 families x 3), own optimizer (LR x5, grad x5), periods clipped to [2, 60], plus a per-window shift
(meta_adjust). There is no switch to freeze the periods at textbook values, so "learned beats textbook" was never
measured (NT-033 todo). Also: the indicator gradient comes from all loss terms, the noisy price head included. Plan: a
freeze switch (or INDICATOR_LR_MULT 0 if the code allows it), then learned vs frozen vs no per-window shift.

## 8. Indicator parameter search - formalisation (owner, 2026-10-08) - running

**The owner's intent:** the network is a search for indicator parameters and their links. **The problem (measured):** 54
bounded periods sit in front of ~300k network weights; the network absorbs everything, so the periods barely matter
(frozen = learned, H21) and their gradient is mostly from the confidence losses, ~10% from direction (H16).
**Principle:** the error must flow so that the indicator parameters carry the dominant share of the result.

Hypotheses, each with its test (criteria fixed before the runs):
- **A (lead) thin readout:** after the indicators only a linear head (MODEL_NAME linear_indicators, 7,610 parameters);
  learned vs frozen periods (hc7_thin_learned / hc7_thin_frozen). Success: learned beats frozen (CI over slices > 0).
- **B (lead, the owner's split by function):** separate networks per function, each with its own indicator families and
  only its own loss: direction <- RSI + MACD (hc7_fn_dir), confidence <- ATR, Bollinger, Keltner, Donchian + NLL/CRPS/t_perp
  (hc7_fn_conf), price <- MA + point loss (hc7_fn_price); each judged on its own function vs the base.
- **C (lead) direct search:** the space is finite, so search it directly (Optuna, 150 trials) with a logistic readout;
  the gradient search must beat it (runs/tactical/search/direct_search.py; search on 3 climb slices, check on the other 3).
- **D (owner) indicators as features -> freeze to the standard:** hc7_frozen (all 14 at textbook values) and hc7_none
  (no indicators) on the long block.
- **E (owner) indicators at the END, as a regulariser:** the network (transformer) predicts the future price PATH; the
  indicators are computed on the predicted path and on the real future, and their mismatch is an extra loss. Lead's
  notes: (1) the indicator parameters inside this loss must be fixed or variance-normalised, else a long period makes the
  target constant and the loss trivially small; (2) an indicator over [past + future] is dominated by the known past
  (lagging), so compute it on the future segment only (e.g. RSI / MA slope of the next 20 bars = the future path's trend
  direction and strength): a "shape" target that defines trend the way TA does. Needs code (a path head + the loss). - idea
- **F (owner) JEPA instead of the transformer:** a self-supervised encoder trained to predict the latent embedding of the
  next segment from the current window (target encoder by EMA, anti-collapse regulariser), pretrained on the whole 2017-2025
  history before each DATA_END (no labels needed), then small heads for direction and confidence. Lead's notes: promising for
  regime / volatility representations and for using the whole history; it cannot create direction information that the
  data lacks; a large build. Test: a linear probe on the frozen JEPA embedding vs our network on the same blocks. - idea
Proof of dominance for any winner: the indicator group's gradient share from its own function's loss >= 50% (probe) and the
learned periods move away from their starting values.
