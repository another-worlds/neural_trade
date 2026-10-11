## Designer A (physics/geometry)
# Designer A: eight hypotheses from loss physics and geometry for BTC/USDT 1-minute OHLCV, 2017-01..2025-09-29

Five of the eight do not predict direction: A2, A5, A6 and A7 predict a new target, and A1 asks when direction is predictable. A3, A4 and A8 change the loss or the representation, and are judged on volatility or path shape.

I ran no new literature searches. "Not found" means not found in the three sweeps above or in references I know (each closest work is named). Every numeric threshold is my proposal, to be frozen in a SPEC before any run.

## Protocol shared by all eight

**Date spans**
- **Selection:** monthly walk-forward with the `recent/wf.py` purge.
  - Direction-adjacent questions use 2021-01..2024-06 (42 months).
  - Questions with a new target may use 2018-01..2024-06 (78 months).
- **Confirmation for a new target:** one look at 2024-07..2025-07-20.
  - That span is spent only for direction hypotheses.
  - It stops at 2025-07-20 because the last 70 days (2025-07-21..2025-09-29) stay protected.
  - Each SPEC states its own contamination. H53 and H57 looked at 2023-25 volatility, so targets close to volatility (A3, A4, A5, A7) are partly contaminated there.
- **Confirmation for a direction-adjacent question:** none is possible until data after 2025-09 exists (owner question 1). Only selection-period evidence can be reported.

**Statistics**
- Effective samples: n_eff = N / T.
- Rough standard errors for one Spearman:
  - T = 20: about 0.005 over 18 months, about 0.02 per month.
  - T = 60: about 0.037 per month, about 0.004 over 78 months.
- Paired differences use t-intervals over months, with Holm correction across the horizons and tails in each SPEC.

**Baselines**
- Every new target is scored against what volatility alone predicts, never against a constant.
  - **Scale:** HAR-RV + time of day.
  - **Shape:** the Brownian law of the path, simulated at 1-minute discretisation with HAR σ̂.
  - **Distributions:** filtered historical simulation (FHS): σ̂_HAR times the empirical standardised distribution of the last 30 days.

**GPU**
- Every GPU job runs in chunks of at most 2 minutes (D-063).
- At most 2 heavy processes at once.

---

## A1. The arrow of time, split into a volatility part and a direction part (changes the question)

**Idea.** Train a classifier to tell 60-bar windows from their time reversals. Its log-odds estimate entropy production, the KL rate between the forward and reversed path measures; this is the idea of NEEP (Kim et al., PRL 2020).
- Time reversal of a price path means: reverse the order of the returns, negate them, swap O and C inside each bar, and keep H, L and volume.
- Two estimators:
  - EP_vol sees only |r|, range and volume (the Zumbach part).
  - EP_tot sees the signed path.
  - EP_dir = EP_tot − EP_vol.
- The question is not "predict direction". It is "where does structure exist that linear reversal cannot use?"

**Mechanism.**
- Weiss (1975): a stationary Gaussian linear process is time-reversible. The reversal edge of tb7 is essentially linear autocorrelation of standardised returns, so it is close to reversible structure.
- Irreversibility therefore measures non-Gaussian or nonlinear structure, which is exactly what tb7 cannot exploit.
- The naive version of this idea (H1 in the loss-physics sweep: "EP tracks the AUC decay") becomes a sharper and counter-intuitive prediction.

**Closest work.**
- Zumbach (2009, Quantitative Finance), time-reversal asymmetry of volatility: daily data, volatility only.
- Flanagan and Lacasa (2016): visibility graphs on daily data.
- NEEP: physical systems.
- **What is ours:** a learned estimator on 1-minute crypto; the split into a volatility part and a direction part; and EP_dir used to predict where nonlinear models beat linear ones.

**Falsifiable predictions.** P2 contradicts the naive hypothesis.
- **P1 (sanity):** EP_vol > 0 in every year, with the lower 95% CI above 0 (the Zumbach effect).
- **P2:** the monthly Spearman of EP_dir with the tb7 AUC excess is below 0.3 over 2018..2024-06. The AUC decay is not an arrow-of-time effect.
- **P3:** months in the top tercile of EP_dir show (boosting AUC − logistic AUC) at least +0.005 above the bottom tercile.

**Minimal experiment.**
- **Model:** a small 1D CNN or logistic classifier on σ̂-standardised windows. Forward and reversed copies of every window are balanced, so the classes are balanced by construction.
- **Training:** fit on the earlier years. Estimate EP per window on later months.
- **Controls:**
  - simulated Gaussian AR(1) with the same autocorrelation must give EP ≈ 0;
  - GARCH-t;
  - shuffled windows.
- **Baselines:** for P3, the boosting − logistic AUC difference per month comes from the existing `decay_lib` / `lab.py` code.

**Protocol.** Selection months only. This is a direction-adjacent question, so a forward confirmation needs data after 2025-09; that is stated in the SPEC.

**Cost.** About 0.5 GPU-hours in 2-minute chunks, plus about 1 CPU-hour.

**Risks and most likely failure.**
- **Most likely:** EP_dir ≈ 0 everywhere. Then P2 holds trivially and P3 is undefined. That is still a clean, evidenced statement that no nonlinear direction structure is left.
- **Artefacts:** tick rounding of prices, and windows that cross hour boundaries (the intraday clock is not time-symmetric). Remove the clock and use log prices.

## A2. Forecasting the martingale defect: serial dependence as a target that does not depend on direction (new target)

**Idea.** Over the next T bars, define

  Z₂ = Σ_{j−i≥2} r_i r_j / Σ r_i²

the future window's own realised autocorrelation, excluding lag 1.
- Under the martingale-difference hypothesis the numerator has conditional mean 0, whatever the volatility.
- Z₂ is invariant to flipping the sign of the whole path, so it says "trend or chop", not "up or down".
- Forecast it from sign-free features: |r| lags, RV, range, volume, clock, past realised Z₂ over 60/240/1440 bars, |EMA distance|, and past variance ratios.

**Mechanism.**
- Liquidation cascades should give positive autocorrelation; market making in quiet markets, negative.
- A spot trader cannot monetise this sign-free quantity directly, so it may not have been arbitraged away the way direction was.
- This bears on loose end 1 (volatility predictability grew while direction decayed).
- Dropping lag 1 removes most of the bid-ask bounce. A Z₃ variant (lags ≥ 3) checks the rest.

**Closest work.**
- Lo and MacKinlay (1988): variance-ratio tests. These are tests, not conditional forecasts.
- "Pockets of predictability" (Farmer, Schmidt and Timmermann): equities, direction.
- **What is ours:** a conditional forecast of a statistic of the future window that is mean-zero under the martingale hypothesis, on 1-minute crypto. Loose end 7 (quiet-regime reversal) becomes a target instead of a gate.

**Falsifiable predictions.**
- **P1:** at T = 60, the Spearman of (prediction, realised Z₂) beats a baseline by at least +0.02, with the lower 95% CI above 0 over the 78 selection months. The baseline is EWMA persistence of past Z₂ plus σ̂.
- **P2:** the gain does not decay. The slope of the yearly gain over 2018-2024 is no worse than −0.005 per year.

**Minimal experiment.**
- **Horizons:** T ∈ {20, 60}.
- **Models:** ridge regression and boosting.
- **Metrics:** Spearman and out-of-sample R².
- **Confirmation:** one look at 2024-07..2025-07-20. That span is unspent for this target, because the target is sign-symmetric.

**Cost.** About 2 CPU-hours.

**Risks and most likely failure.**
- **Most likely:** the predictability is only persistence of past Z₂, so it adds nothing over the baseline.
- **A second reading:** if Z₂ is predictable while the earlier boosting on tb7 found no direction gain, the trend/chop state cannot be turned into direction. That is informative, but it is not a trade.

## A3. Bellman consistency of the variance term structure: fixed-horizon TD instead of Monte Carlo regression (loss physics for volatility)

**Idea.** The expected cumulative variance V_h(t) = E_t[Σ_{k=1..h} r²_{t+k}] obeys a semigroup (Chapman–Kolmogorov) recursion:

  V_h(t) = E_t[r²_{t+1} + V_{h−1}(t+1)]

- HAR regresses on the realised Monte Carlo target Σ r², which is noisy.
- Fixed-horizon TD (FHTD) learns V_1..V_H by regressions one after another. Each one bootstraps on the previous estimate at t+1. This is a lower-variance target.
- With linear HAR features every step is a closed-form OLS.

**Mechanism.** Bootstrapping trades variance for bias.
- The gain should appear exactly where variance dominates: short rolling refits under non-stationarity.
- It should vanish with long training sets.

**Closest work.**
- De Asis et al. (AAAI 2020), "Fixed-horizon temporal difference methods": reinforcement learning.
- Corsi's HAR: the Monte Carlo regression.
- **What is ours:** FHTD applied to the realised-variance term structure; I know of no such use.

**Falsifiable prediction (an interaction).**
- At a 30-day rolling refit, FHTD beats direct HAR on QLIKE at h = 60 by at least 0.005, with the CI over months above 0.
- At a 365-day refit, |Δ| < 0.002.
- If the gain does not shrink with longer training, bootstrapping is not the cause, and the result goes against the mechanism.

**Minimal experiment.**
- **Features:** log-RV lags at 1, 5, 15, 60, 240 and 1440 minutes, plus Fourier terms of the clock.
- **Horizons:** h ≤ 240 at a 1-minute step. A geometric grid of h for speed is possible, but it needs the recursion over unequal steps.
- **Refit windows:** 7, 30, 90 and 365 days.
- **Metrics:** QLIKE and Spearman at 15, 60 and 240 minutes.
- **Optional extension:** volnet trained with TD targets.

**Cost.** About 2 CPU-hours.

**Risks and most likely failure.**
- **Most likely:** the HAR fit is already data-rich, so |Δ| < 0.002 at every refit length.
- **Bias:** TD bias from misspecified features can make TD worse.
- **Contamination:** the 2024-25 confirmation is partly contaminated (H53, H57).

## A4. The physics terms judged on their natural axis, with a Fisher-flat variance head (loss geometry; loose end 2)

**Idea.**
- **The geometry.** For the Gaussian scale family, the Fisher-Rao metric is flat in log σ: ds = √2 |d log σ|.
  - A variance head parametrised in v, with a floor, has stiff 1/v gradients and dead regions (AUDIT C11).
  - Reparametrising it as log v is the natural gradient for that family.
- **The design.** A 2 × 2 grid: {v head, log-v head} × {physics terms on, off}. Score it on QLIKE and Spearman of the variance against HAR on the same blocks, never on direction.
- **Why.** H42 showed the physics terms are aligned with NLL and CRPS and orthogonal to direction. Until now they were judged only on direction and the composite metric.

**Mechanism.** The value of the physics terms could be hidden by the stiff geometry of the head. They would then show up only in flat coordinates.

**Closest work.**
- InfoGeo Loss (ICML 2026): general data, MSE.
- NGBoost: natural gradient for boosting.
- **What is ours:** a factorial test of the project's six physics terms, on the axis they couple to, against HAR.

**Falsifiable predictions.**
- **P1:** the log-v head is better than the v head by at least 0.02 in QLIKE.
- **P2 (interaction):** the physics on−off difference is at least 0.005 in QLIKE under log-v, and smaller than its CI under v.
- **P3:** the net (any arm) improves Spearman over HAR by no more than +0.01. This is pre-registered as an expectation, not as a goal.

**Minimal experiment.**
- **Code:** a tactical switch `VAR_PARAM`, a Config change on `nt-tactical` only, made by the implementer.
- **Slices:** the 24 lab slices.
- **Comparison:** paired over slices; the SE of one slice's Spearman is about 0.045, so about 0.009 over 24.
- **Confirmation:** the 6 FINAL slices. They were used before for direction, not for variance; the SPEC says so.

**Cost.** 96 runs × 2 minutes ≈ 3.2 GPU-hours, or 1.6 GPU-hours with 12 slices.

**Risks and most likely failure.**
- **Most likely:** P1 holds, as a plain optimisation fix, and P2 is null. That is a clean negative for D-003 on its natural axis, and it goes to the owner.
- **Rules:** removing a physics term is the owner's decision (D-003); nothing here changes a default.

## A5. Omori aftershocks of HAR-standardised extremes (new target: hazard)

**Idea.** The event is 1{|r_1| > 5σ̂_1min}, a standardised 1-minute shock. Forecast P(at least one event in the next 60 minutes) with a hazard model:

  λ(t) = μ(σ̂_HAR, time of day) + Σ_i K0·m_iᵅ / (t − t_i + c)ᵖ

- The kernel is Omori's power law, with productivity scaling with the shock's magnitude m_i.
- HAR has fixed lags and no "time since the last shock" state.

**Mechanism.** In geophysics, aftershock rates decay as Omori's power law; financial markets show the same decay after crashes. If clustering stays after standardising by σ̂_HAR, then HAR, a crude multi-scale approximation of a power-law memory, misses the hazard structure.

**Closest work.**
- Lillo and Mantegna (2003, PRE): the Omori law after crashes.
- Petersen et al. (2010, PRE): the Omori, productivity and Båth laws.
- Hawkes VaR (arXiv 2510.21297): daily or intraday VaR.
- **What is ours:** a forecast of HAR-standardised exceedances at the minute scale, scored against the base rate that HAR implies.

**Falsifiable predictions.**
- **P1:** Brier skill of at least 0.01 (log-loss skill at least 2%) over a logistic on σ̂_HAR and the clock, with the CI over months above 0.
- **P2:** the fitted p lies in [0.6, 1.2] and moves by less than 0.2 between years.

**Minimal experiment.**
- **Fit:** MLE on 2018..2024-06.
- **Confirmation:** one look at 2024-07..2025-07-20; partly contaminated through HAR.
- **Reported, not pre-registered:** whether productivity is asymmetric between up-shocks and down-shocks.

**Cost.** About 1-2 CPU-hours.

**Risks and most likely failure.**
- **Most likely:** HAR's 1-, 5- and 15-minute lags already absorb the clustering, and the skill is below 0.01.
- **Event counts:** at 5σ̂ a 60-minute window may hold too few events per month. If so, the SPEC fixes the threshold at 4σ̂ before any run.

## A6. A generator of the path under an energy score, with a decomposition into scale, shape and endpoint (new target: path law)

**Idea.**
- **Generator:** an engression-style network: features plus noise in, a 20-step σ̂-standardised future path out.
- **Training loss:** the energy score on the path vector (cumulative returns at 5/10/15/20, running max and min, time of the maximum), with m = 16 samples.
- **Baselines:** the discretised Brownian path with σ̂, and FHS paths (bootstrapped standardised training paths).
- **Score decomposition:**
  - CRPS of the endpoint (direction plus scale);
  - CRPS of range / σ̂√T (shape);
  - CRPS of drawdown / σ̂√T (shape).
- **Optional upgrade:** a truncated level-3 signature-MMD score in TF.

**Mechanism.** The energy score is strictly proper on the joint law. If shape information beyond σ exists, it shows in the path-functional marginals but not in the endpoint.

**Closest work.**
- Pacchiardi et al. (scoring-rule training of generators).
- Engression (Shen and Meinshausen).
- Issa et al. (NeurIPS 2023): signature-kernel scores for neural SDEs.
- Sig-Wasserstein GANs (Liao et al.): daily or FX data, realism.
- **What is ours:** conditional intraday crypto, a Brownian/FHS null with σ̂ factored out, and the endpoint-vs-shape decomposition as the test.

**Falsifiable predictions.**
- **P1:** the energy score is at least 0.5% better than FHS, with the CI over months above 0.
- **P2 (the decomposition):** the endpoint CRPS gain stays within ±0.1%, while the range-ratio CRPS gain is at least 1%.

**Minimal experiment.**
- **Model:** an MLP generator.
- **Inputs:** HAR features, clock, volume, past range ratios.
- **Training:** selection months, checkpointed in 2-minute chunks.
- **Confirmation:** one look at 2024-07..2025-07-20, unspent for path law.

**Cost.** About 1-1.5 GPU-hours.

**Risks and most likely failure.**
- **Most likely:** the gain is all scale (σ̂ refinement), and the shape marginals tie with FHS.
- **Score insensitivity:** the energy score is weak at telling dependence structure apart. A variogram score (p = 0.5) is reported beside it.

## A7. Intraday VaR/ES under the elicitable FZ0 loss, with HAR, semivariance and Omori state (new target; the risk branch)

**Idea.** Estimate VaR and ES jointly for 15- and 60-minute returns, at α = 1% and 2.5%, for each tail separately.
- **Loss:** FZ0 (Fissler–Ziegel). ES alone is not elicitable; the pair (VaR, ES) is jointly elicitable.
- **Model:** σ̂_HAR times a shape multiplier from RS⁺/RS⁻ (semivariance), jump age, the A5 hazard and the clock; linear or a small MLP.
- **Baseline:** FHS.

**Mechanism.** If the tail shape is predictable, as opposed to the scale, a model that sees tail-specific state beats an FHS whose standardised quantile is fixed.

**Closest work.**
- Patton, Ziegel and Chen (2019, J. Econometrics): dynamic FZ0 models on daily data.
- Daily crypto FZ studies.
- **What is ours:** intraday horizons on 1-minute crypto, against FHS on HAR, with point-process state.

**Falsifiable predictions.**
- **P1:** FZ0 at least 1% better at 60 minutes in the 1% left tail; DM test p < 0.05 after Bonferroni over 2 tails × 2 α.
- **P2:** unconditional coverage within [0.8%, 1.2%]; Christoffersen and Acerbi–Székely Z2 tests not rejected.

**Minimal experiment.**
- **Exceedances at α = 1% and h = 60:** about 88 per year, about 570 over selection.
- **Confirmation:** one look at 2024-07..2025-07-20, flagged as partly contaminated.

**Cost.** About 2 CPU-hours (plus about 0.3 GPU-hours if the MLP is used).

**Risks and most likely failure.**
- **Most likely:** the whole gain comes from a better σ̂, and the shape multipliers are null.
- **Positioning:** the most practically useful item (drawdown control), but the least "unknown".

## A8. JEPA on a quotient path space: an information budget for shape given σ (JEPA)

**Idea.**
- **Context encoder:** reads the past 120 bars, σ̂-standardised.
- **Target encoder:** reads the future 20-60 bars, divided by their own realised σ and made invariant to the reflection path ↦ −path (symmetrised over the group {±1}).
  - The target therefore lives on the quotient of path space by scale × reflection.
  - The embedding cannot encode volatility or direction, only shape.
- **Collapse guard:** SIGReg instead of the EMA teacher.
- **Read-out:** a frozen linear probe on the embedding predicts Z₂ (A2), the range-ratio residual (A6), and the barrier-first-hit time ratio.

**Mechanism.** Earlier JEPA rounds (v1-v4) targeted raw future latents. Those latents were dominated by scale, which HAR already absorbs, and by direction, which does not exist. Taking the quotient removes both, so whatever the probe finds is the shape information the past window holds, within the probe's power.

**Closest work.**
- TS-JEPA; MTS-JEPA; LeJEPA / SIGReg (Balestriero and LeCun, 2025).
- **What is ours:** a group-invariant future target, and probes defined only on shape.

**Falsifiable predictions.**
- **P1:** the probe beats ridge on the hand-made features of A2/A6 by at least +0.01 Spearman on at least 2 of 3 shape targets.
- **P2:** effective rank ≥ 24 of 32 (no collapse).
- If P1 fails while P2 holds, conclude that no learned shape information exceeds the hand-made features at this probe's power.

**Minimal experiment.**
- **Pretraining:** 2018-2021 data, checkpointed in 2-minute chunks.
- **Probes:** walk-forward over 2022..2024-06.
- **Confirmation:** one look at 2024-07..2025-07-20, unspent for shape.
- **Order:** run only after A2 or A6 shows a non-zero shape signal. Without one, the prior is too low.

**Cost.** About 2-3 GPU-hours.

**Risks and most likely failure.**
- **Most likely:** a null result, as in JEPA v1-v4.
- **Tooling gap:** SIGReg needs a port to TF 2.10. It is small: characteristic functions along random 1-D projections.

---

## Order and decision tree

| Order | Hypothesis | Compute | What it decides |
|---|---|---|---|
| 1 | A2 (martingale defect) | about 2 CPU-h | Whether path shape beyond σ is predictable; it gates A6 and A8 |
| 2 | A1 (arrow of time) | about 0.5 GPU-h + about 1 CPU-h | Whether any nonlinear direction structure exists at all |
| 3 | A3 (TD on the variance term structure) | about 2 CPU-h | Whether bootstrapped targets beat HAR under refits |
| 4 | A5 (Omori hazard) | about 1-2 CPU-h | Whether tails cluster beyond HAR; its state feeds A7 |
| 5 | A7 (FZ0 VaR/ES) | about 2 CPU-h | Usable result for the risk branch |
| 6 | A6 (energy-score path generator) | about 1-1.5 GPU-h | Path law and the shape/scale/endpoint split; run if A2 is positive or ambiguous |
| 7 | A4 (Fisher-flat head, physics 2 × 2) | 1.6-3.2 GPU-h | The verdict on the physics terms on their natural axis, for the owner (D-003) |
| 8 | A8 (JEPA-PF) | 2-3 GPU-h | Only if A2 or A6 finds shape information |

**Totals (estimates):** about 9-11 CPU-hours and 5-8 GPU-hours.

**The one result that would change the project's direction:** a positive A2 or A6 combined with a P2 that holds (no decay). The predictable object would then be the shape of the path, not its direction or scale. That would move the risk branch toward stops and barriers set on forecast path geometry.

## Designer B (question/architecture)
# Designer B: changing the question, original architectures and JEPA. Seven hypotheses for a fresh experiment

## 0. Design principles

- **Change the question before changing the model.** H5-H65 show the model is not what limits direction skill. At 10-20 minutes, direction information on 1-minute BTC OHLCV appears to be close to exhausted. Every new target below is therefore built to be **invariant to the volatility level**, so that HAR cannot explain it by construction. A signal can only appear where the path departs from a time-changed martingale.
- **The geometric core comes from Dambis–Dubins–Schwarz.** Any continuous martingale is a Brownian motion run on a random clock, its quadratic variation. This splits "what is predictable" into two parts:
  - **the clock**: volatility and activity. HAR already captures it almost completely (H53, H55, H57).
  - **the shape of the path in its own clock**: trendiness, choppiness, jumps and drift. Under the martingale null this part cannot be predicted.

  Hypotheses H-1, H-2 and H-3 measure the second part directly. For HAR, its information is zero by construction.
- **Correction to an input (loss_physics, §3):** the claim "under detailed balance no direction edge is possible" is **false**. A stationary Gaussian ARMA process is time-reversible and still linearly predictable. Irreversibility measures only *non-Gaussian* temporal asymmetry, such as Zumbach or leverage effects and jumps. H-4 is designed with this correction.

**Shared protocol** (applies to all seven unless a hypothesis says otherwise):

| Element | Rule |
|---|---|
| Data | `Bitcoin_BTCUSDT.csv`, 1-minute OHLCV only (D-050) |
| Protected span | The last 70 days, 2025-07-21..2025-09-29, are never used |
| Walk-forward | Monthly, with a purge (`recent/wf.py split_masks`). Training uses the 365 days before each month (H56: 365 days beat 60) |
| Selection months | 2019-01..2024-06 for new targets (they were never scored, so early years are clean); 2021-01..2024-06 for direction |
| Confirmation | 2024-07-01..2025-07-20, **exactly one look, and only for targets never scored there**. Each hypothesis states whether its look is clean |
| Statistics | n_eff = N/T. Paired monthly differences with a t-interval. Holm across horizons and targets |
| SPEC | Committed before any result (as for H49/H56/H57/H63/H64) |
| Costs | Zero by default; a 1/2/5 bps grid is reported (D-044) |

---

## Summary

| # | Name | What changes | Compute | Confirmation look clean? | Prior |
|---|---|---|---|---|---|
| H-1 | **Trendiness in its own clock**: forecasting the scale-free shape of the path | Target: \|ΔP\|/range, volatility-invariant | CPU, ~1 h | **Yes** (never scored) | M |
| H-2 | **Omori aftershocks and foreshocks** on HAR-normalised tail exceedances | Target: the tail event; model: a point process | CPU plus a tiny TF fit, ~1-2 h | **Yes** | M |
| H-3 | **JEPA in business time (JEPA-DDS)** with SIGReg | Self-supervised target: the future path resampled in quadratic-variation time | GPU ~2-3 h | Yes, for shape targets | L-M |
| H-4 | **Arrow of time (NEEP)** as a measure of *when* the market is predictable | Question: when, not where | GPU ~1 h + CPU | No (direction), forward test only | L-M |
| H-5 | **Semigroup / TD loss** for the volatility term structure (Chapman–Kolmogorov) | Loss: bootstrapping on the model's own forecast | CPU ~1 h + GPU < 1 h | Partly (volatility was looked at) | M on data efficiency |
| H-6 | **Rotation or shrinkage of the edge**: geometry of the coefficient vector over 2017-2024 | Question: did the edge disappear or move? | CPU < 1 h | Not needed (descriptive) plus a forward test | H that it closes the question |
| H-7 | **Sign-free bracket strategy** on the H-1 forecast | Trading question with no direction | CPU < 1 h | Yes (the strategy class was never tested) | L-M |

At least four hypotheses change the target or the question (H-1, H-2, H-3, H-4). H-7 is a trading use of H-1.

---

## H-1. Trendiness in its own clock: is the scale-free shape of the next hour predictable?

**Idea.**
- The target for a horizon T is ρ_T = |C_{t+T} − C_t| / (max H − min L over (t, t+T]), with ρ ∈ [0, 1]. It answers "will the next hour trend or chop", not "up or down".
- For a time-changed Brownian motion with a clock independent of the path, the distribution of ρ **does not depend on the volatility level or the volatility path**. This follows from Brownian scaling, because the ratio is scale-free. So HAR carries no information about ρ under the null.
- Secondary targets, all scale-free: the within-horizon variance ratio VR = (r_T)² / Σ r²_1m, and the number of σ-sized directional changes (the directional-change intrinsic time of Glattfelder et al.).

**Mechanism.**
- Deviations from a martingale with a random clock give ρ: autocorrelation of 1-minute returns (reversal or momentum), local drift, and jumps (a jump gives a high ρ).
- The short-term reversal (H52) is the signed version of the same thing. Its **sign-free** projection, "the market is choppy now", might be more persistent than the sign itself, because it is a property of the regime and not of the last return. This has never been measured.

**Closest work, and how ours differs.**
- Lo–MacKinlay variance ratios and Kaufman's efficiency ratio are tests and descriptive indicators.
- Rolling Hurst exponents are used as features (estimator noise is large; Cont–Das 2203.13820).
- Glattfelder, Dupuis and Olsen describe intrinsic time descriptively.
- I found no out-of-sample *forecast* of the future scale-free path shape against a Brownian / HAR baseline on any HF asset (medium confidence).
- Project negative space: H54 (excursion asymmetry) is signed. Geometry and squeeze (H47) concern volatility persistence. The H49 novel features were judged on direction. ρ was never a target.

**Falsifiable prediction (numbers).**
- At T = 60 the challenger gives ΔSpearman ≥ +0.02 over the Brownian-microstructure baseline (average over selection months, lower bound of the 95% CI > 0).
- The largest gain is in 2019-20 (≈ +0.03..+0.05) and it decays to ≈ +0.015..+0.03 in 2023-24.
- **Sharp secondary prediction:** sign-free predictability decays more slowly than direction AUC. The ratio (2023-24 effect)/(2019-20 effect) for ρ exceeds the same ratio for AUC excess (0.53−0.5)/(0.595−0.5) ≈ 0.3.
- If ρ decays exactly in step with AUC, it is the same reversal effect and not a separate phenomenon. That is also informative.

**Minimal experiment.**

| | |
|---|---|
| Targets | ρ_T and VR_T for T ∈ {30, 60, 120} min, from 1-minute OHLC (H and L give the range exactly) |
| Baseline | A **"Brownian + microstructure" null**: ridge (logit link on ρ) on log σ̂_HAR, log(tick/σ̂√T), the average past \|close − next open\|/σ (a bounce proxy) and a Fourier clock (hour and minute within the hour, following Hansen–Kim–Kimbrough). A Monte Carlo calibration of E[ρ \| σ/tick] under Brownian motion plus bounce is a sanity check |
| Challenger features | Past ρ over 30/60/240, VR(5/15/60) over the last 240 bars, lag-1..5 autocorrelation of 1-minute returns, \|tb7 logit\| (the strength of the reversal signal, sign removed), volume surprise, time since the last ≥ 4σ bar, RS⁺/RS⁻ |
| Models | Ridge and HistGradientBoosting (CPU). Optional arm: a small MLP trained with an energy score on (ρ, VR) jointly |
| Metric | ΔSpearman (challenger − baseline); MSE skill on logit(ρ) |
| Criterion | Pre-registered: ΔSpearman ≥ +0.02 at T = 60, monthly CI > 0, Holm over 3 horizons × 2 targets |
| Protocol | Selection 2019-01..2024-06. **Confirmation 2024-07..2025-07-20 is clean**: ρ, VR and directional-change counts were never scored there |

**Cost.** CPU, about 1 hour including feature caches (`decay_lib` bar caches).

**Risks and most likely failure.**
- The baseline absorbs everything. ρ may be fully explained by σ/tick and the clock (microstructure). Then the result is a clean negative: "in its own clock the hour is Brownian".
- A second risk: past ρ is too noisy at a 60-240 bar window (the Cont–Das critique). The Spearman values will be about 0.01.

---

## H-2. Omori's law for BTC: HAR-normalised tail exceedances cluster as aftershocks and foreshocks

**Idea.**
- The event is E_t = 1{the price leaves the band C_t ± k·σ̂_HAR·√T within T}, with k chosen on the training set so the base rate is about 2%.
- The model is a point process: logit λ = HAR + clock baseline + Σ_i m(|z_i|)·(Δt_i + c)^(−p), over past exceedances i with learned (c, p) and a mark function m.
- A separate "foreshock" term counts small exceedances (k/2) over the last 30 minutes.
- The target is a **tail event after HAR normalisation**, not direction and not RV.

**Mechanism (physics).**
- Aftershock dynamics follow Omori–Utsu: a power-law decay in the rate after a shock. HAR is a sum of fixed-window averages and cannot represent the *time since the event* with a power-law kernel.
- If the market has memory of shocks (liquidations, stop cascades), the aftershock rate exceeds what HAR's smoothed σ̂ implies. The foreshock term tests the mirror law: a rising rate of small shocks before a large one.

**Closest work, and how ours differs.**
- Lillo & Mantegna 2003 (Omori law after crashes); Petersen, Wang, Havlin, Stanley 2010 PRE (Omori, productivity and Båth laws on 1-minute S&P data). Both are descriptive fits, with no out-of-sample forecast against HAR.
- Hawkes jump clustering in crypto VaR (2510.21297; abstract and summary only).
- Scaillet–Treccani–Trevisan 1704.08175 (Mt. Gox jumps, order flow).
- **Ours:** the events are exceedances *of the HAR forecast* (the residual of the clock, not raw jumps); the test is out-of-sample skill over HAR + clock on 1-minute BTC from 2017 to 2025; and it includes the foreshock term.
- The project never touched Hawkes, jumps or tail targets (§2 item 12).

**Falsifiable prediction.**
- Log-loss skill ≥ +2% (Brier skill ≥ +3%) over the HAR + clock logit at T = 30, k = q98, with monthly CI > 0.
- The fitted Omori exponent p is in [0.6, 1.3], as in the equity literature.
- The foreshock term adds ≥ +0.5% on its own.
- Secondary, exploratory: excitation is asymmetric, and down-exceedances excite more than up-exceedances (leverage). This is not a criterion, because the sign is close to direction.

**Minimal experiment.**

| | |
|---|---|
| Target | E_t for T ∈ {15, 30, 60}, k = q98 of the normalised maximum excursion on the training block |
| Baseline | Logistic on log σ̂_HAR (RV15..1440) + clock (Fourier: hour, minute within the hour, weekday) + log σ̂ × clock |
| Challenger | Baseline + an Omori kernel (TF, 4 parameters plus the mark; Adam, CPU) + a foreshock count; a variant with a sum of 3 exponentials (classical Hawkes) to compare power-law and exponential memory |
| Metric | Log-loss and Brier skill; reliability curve |
| Criterion | Pre-registered: log-loss skill ≥ +2% at T = 30, Holm over 3 horizons |
| Protocol | Selection 2019-01..2024-06 (n_eff ≈ 1,440 windows per month, ≈ 29 events per month, ≈ 1,900 events in total). **Confirmation 2024-07..2025-07-20 is clean** for this target. Caveat for the SPEC: H57 looked at RV in that period, but never at HAR-normalised exceedances |

**Cost.** CPU about 1 hour, plus a few minutes of TF on the CPU.

**Risks.**
- RV15 inside HAR already captures recent shocks. The added skill would then be about +0.5%: true but small.
- The k threshold depends on a σ̂ scale bias. Calibrating k on the training block per year removes this.
- Most likely failure: the Omori kernel works in sample, but out of sample the gain is inside the CI.

---

## H-3. JEPA in business time (JEPA-DDS): the target encoder sees the future path in quadratic-variation time

**Idea.**
- This is the only JEPA formulation that is new compared with v1-v4 (H34, H41, H46, H55). In those, the target was a latent of raw future bars, so the encoder learned σ, which HAR then absorbed (H55).
- Here the future path over 60 minutes is **resampled on a grid of equal quadratic-variation steps** (16-32 points) and normalised by the total QV. Under the continuous-martingale null (Dambis–Dubins–Schwarz) it is a standard Brownian path on [0, 1] and cannot be predicted.
- The context encoder reads the past 240 bars. The predictor predicts the target latent. Collapse is prevented by **SIGReg** (LeJEPA, 2511.08544: Epps–Pulley on random 1-D projections, one hyperparameter) instead of an EMA teacher and stop-gradient.
- The "clock" (wall time per QV quantum) is a separate stream with its own head. It is not in the shape target.

**Mechanism.**
- Removing the clock leaves the JEPA objective only one thing to learn: departures from martingale geometry (jumps, local drift, choppiness).
- The gap between the predictor's loss and the loss with **contexts permuted between samples** is a direct, model-based estimate of the predictable information in the shape of the path. It is a "ceiling" diagnostic on the axis where HAR is blind by construction.

**Closest work, and how ours differs.**
- TS-JEPA (2509.25449: raw patches, no finance).
- MTS-JEPA (2602.04643: anomalies).
- Phys-JEPA (2606.16076: a physical latent, weather and traffic).
- LeJEPA (images).
- The project's own JEPA v1-v4.
- I found nowhere a target encoder over a **time-changed (DDS) future path**, nor a JEPA probed for path invariants against a Brownian baseline. Medium confidence that this is new.

**Falsifiable prediction.**
- **(a)** The InfoNCE bound (K = 256 negatives) on I(context; shape in business time) is ≥ 0.005 nats per window, with a block-bootstrap CI > 0. For the clock stream the same bound is ≥ 0.05 nats (a positive control).
- **(b)** A frozen linear probe on the embedding improves Spearman on ρ_BT (H-1's trendiness computed in business time) by ≥ +0.01 over H-1's hand-made features. Without (b), JEPA adds nothing.
- My point estimate: (a) holds at about 0.005-0.02 nats, and (b) does not hold (+0.003). If (a) fails as well, that is a strong negative: **"at 60 minutes BTC is a martingale with a random clock; everything predictable is the clock"**. That would be a scientific result in its own right.

**Minimal experiment.**

| | |
|---|---|
| Pretraining | 2017-01..2020-12. Encoder: the existing patch transformer from `jepa/v4/model4.py`, 26k and 103k. Loss: smooth-L1 in latent space + SIGReg (λ from {0.02, 0.05}). 2 seeds |
| Probes | Fitted on 2021-01..2022-12; selection evaluation on 2023-01..2024-06 (monthly) |
| Probe targets | ρ_BT, ρ_T (H-1), the H-2 exceedance, drawdown/√QV, the number of directional changes |
| Baselines | (i) H-1 / H-2 hand-made features under ridge or logistic; (ii) the same JEPA with an **ordinary** (wall-clock) target, to isolate the effect of DDS; (iii) context-permutation for the bound |
| Criterion | Pre-registered: (a) bound CI > 0; (b) probe ΔSpearman ≥ +0.01 on ρ_BT, monthly CI > 0. Both are needed for "JEPA-DDS adds information" |
| Confirmation | 2024-07..2025-07-20, clean for the shape targets |

**Cost.** About 2-3 GPU-hours (2 seeds × 2 sizes × about 30-40 minutes) plus CPU probes. **This exceeds D-063's 2-minute limit per run.** It needs either the owner's consent (as before) or resumable 2-minute chunks with checkpoints.

**Risks.**
- Resampling with jumps: one 1-minute bar can carry several QV quanta. Linear interpolation creates artificial "straight segments", which a JEPA learns as a fake signal. Control: the same pipeline on synthetic Brownian motion with a GARCH clock and jumps must give a bound of about 0 for the shape (a mandatory negative control in the SPEC).
- SIGReg in TF 2.10 is new code: about 40 lines, plus a test on Gaussian data.
- Most likely failure: (a) is positive because of jumps, and (b) is null.

---

## H-4. Arrow of time: does local irreversibility mark *when* direction is predictable?

**Idea.**
- Train a small 1-D CNN (NEEP, Kim et al. PRL 2020) to tell forward 60-bar windows from **reversed** ones. The output averaged over a window is a local estimate of entropy production (EP).
- **Three controls separate the effects:**
  1. time + sign reversal (a true time reversal of the price path: increments in reverse order and negated);
  2. sign only (up/down mirror symmetry);
  3. order only.

  Only (1) minus (2) is "pure" irreversibility, without skewness.
- Questions:
  - **(i)** Does monthly EP follow the decay of AUC from 2018 to 2024?
  - **(ii)** Within a month, is the AUC of tb7 higher in the top EP quintile than in the bottom one?
  - **(iii)** What drives EP: volatility asymmetry (Zumbach: past trends predict future volatility, not the reverse) or return asymmetry?

**Mechanism and correction.** Reversibility does **not** rule out linear predictability: a Gaussian AR process is reversible. So EP is not an upper bound on AUC. It measures nonlinear asymmetry: stop and liquidation cascades (sharp moves followed by slow relaxation), Zumbach and leverage effects. If the remaining edge in tb7 has a nonlinear source (cascades), it should be concentrated where EP is high.

**Closest work, and how ours differs.**
- Kim et al. 2020 (NEEP, physical systems).
- Flanagan & Lacasa 2016 (daily data, visibility graphs).
- Lynch & Zumbach 2003; Zumbach 2009; Chicheportiche–Bouchaud (time-reversal asymmetry in volatility).
- No one has applied a neural EP estimator to 1-minute crypto as a gate on a direction model, nor split it into "time" and "sign" with controls. Never tested in the project.

**Falsifiable prediction.**
- **(i)** Spearman between monthly EP and monthly tb7 AUC over 2018-01..2024-06 (78 months) is ≥ +0.4. My actual bet: EP correlates with the volatility level (≥ 0.5) more than with AUC (< 0.3).
- **(ii)** Top-quintile AUC minus bottom-quintile AUC is ≥ +0.015, with monthly CI > 0, on 2021-01..2024-06 with next-open entry.
- **(iii)** More than 60% of the forward-versus-reversed classification accuracy remains when the input is |r| only (that is, it is the Zumbach effect, about volatility). This is a prediction about the nature of the effect.

**Minimal experiment.**

| | |
|---|---|
| Estimator | CNN with about 10k parameters, input 60 × (r/σ, \|r\|/σ, log volume), trained on 2017-2019 |
| Gate | EP quintiles computed with thresholds from the first half of each validation block (the `probe/honest_split.py` procedure) |
| Baseline | Unconditional tb7 AUC; a gate on σ̂ quintiles (to exclude "EP = volatility") |
| Criterion | (ii) at +0.015 **and** better than the σ̂ gate by ≥ +0.01 |
| Protocol | Direction: the 2024-07..2025-09 look is spent, so the verdict needs **forward data** (owner question 1). (i) and (iii) are descriptive and need no look |

**Cost.** About 1 GPU-hour (3 control classifiers × 2 seeds) and about 1 CPU-hour.

**Risks.** The most likely outcome is that EP is a proxy for volatility, the σ̂ gate is not beaten, and (ii) fails. Even then, (iii) gives the first quantified measure of time irreversibility on 1-minute BTC over 9 years, which is a separate scientific result.

---

## H-5. Semigroup (Chapman–Kolmogorov / TD) loss for the volatility term structure

**Idea.**
- Expected variance is additive in time (tower property): E_t[RV_{t,t+2T}] = E_t[RV_{t,t+T}] + E_t[E_{t+T}[RV_{t+T,t+2T}]].
- Instead of regressing directly on the noisy, heavily overlapping RV_{2T}, train the long head on the **bootstrapped target** y_T + Ŷ_T(t+T). The second term is the model's own forecast at t+T, with stop-gradient or a target network. This is TD learning (Sutton 1988 framed TD as multi-step prediction) on the horizon ladder 15 → 30 → 60 → 120 → 240 → 1440.
- Linear version: **LSTD-HAR**, with HAR coefficients for long horizons fitted by least-squares TD.
- Neural version: the existing volatility tower (`newnet`) with a TD loss.

**Mechanism.**
- A bootstrapped target has lower variance than a Monte Carlo target. With overlapping labels (FreDF: the bias of direct multi-step losses under autocorrelated labels) this matters most at long horizons and short training windows.
- The physics is the Markov semigroup property: a forecast that is coherent across horizons is a consistency constraint, not new information. So the prediction is about **data efficiency**, not the ceiling.

**Closest work, and how ours differs.**
- Sutton 1988; distributional RL (Bellemare 2017).
- Coherent temporal hierarchies (Rangapuram et al., AISTATS 2023: reconciliation, not TD).
- "HARd to beat" (2406.08041).
- I found no TD/LSTD estimation of the RV term structure in finance (medium confidence).
- The project's physics terms (t_perp and so on) were never judged on the volatility axis against HAR (§2 item 2). This hypothesis gives such a test with a clean physical meaning.

**Falsifiable prediction (with the mechanism's signature).**
- With 60-day training windows, LSTD-HAR or the TD tower at the 240- and 1440-minute horizons beats direct fitting on QLIKE by ≥ 1.0%.
- With 3-year windows the gain is ≤ 0.3%. A gain that shrinks with data is the signature of variance reduction. If the gain does not shrink, or is negative, then the TD bias (features that are not Markov-sufficient) outweighs it, and the hypothesis is rejected.

**Minimal experiment.**

| | |
|---|---|
| Target | log RV over 60/240/1440 minutes |
| Models | Direct HAR (`vol/volbench.py`) vs LSTD-HAR; the volnet tower, direct vs TD (target network, τ = 0.99) |
| Grid | Training windows of 30/60/180/365/1095 days |
| Metric | QLIKE and Spearman, paired by month |
| Criterion | Pre-registered: ≥ 1.0% QLIKE at 60 days for 240 **and** 1440 minutes; a monotone shrinking of the gain with the window (Spearman of gain against log window < 0) |
| Protocol | Selection 2019-01..2024-06. **The confirmation look is partly contaminated** (H53 and H57 looked at RV in 2024-25), so the internal signature (the curve of gain against window) is the main evidence. Forward data would be the clean check |

**Cost.** CPU about 1 hour (LSTD is closed-form); GPU about 0.5-1 hour for the neural arm.

**Risks.** HAR is already almost Markov-sufficient and its OLS is stable on 60 days, so the gain stays within noise. Practical value: longer volatility horizons for the owner's separate risk-management branch.

---

## H-6. Rotation or shrinkage: the geometry of the edge's decay

**Idea.**
- Fit tb7 on standardised features separately for each half-year from 2017H1 to 2024H1 (15 points), with bootstrap ellipses. Two geometric quantities: **the norm** of the coefficient vector (the strength of the edge) and **the angle** between neighbouring half-years and against 2019 (the direction of the edge).
- Shrinkage means a falling norm at a fixed angle: the edge is disappearing. Rotation means a stable norm and a growing angle: the edge is moving, and adaptation helps.
- Test: a Kalman / RLS time-varying logistic with learned process noise against a static fit over 365 days.

**Mechanism.** If the decay is drift in the mapping, a closed-form adaptive readout (the NG-RC and TTA family, rows 10 and 14 of the geometry_arch input) recovers part of the AUC. If it is shrinkage, no architecture will bring the edge back. This closes the "maybe a smarter model" branch with evidence.

**Closest work, and how ours differs.**
- Nagel 2025 (NBER w34104): random features reduce to recency-weighted momentum.
- TTA on daily FX and equity (2602.00073).
- Ours: the geometric split of decay into norm and angle on 1-minute BTC over 9 years.
- Partial negative space: H56 (365 days beat 60). That speaks against drift, but a fixed window is not an adaptive filter with learned noise.

**Falsifiable prediction.**
- My bet is shrinkage: ‖β‖(2024H1)/‖β‖(2019) ≤ 0.5 at cos(β_2019, β_2024H1) ≥ 0.8; the Kalman logistic gains ≤ +0.002 AUC.
- Rotation would mean cos < 0.6 between neighbouring half-years with a stable norm, and then a pre-registered gain from adaptation of ≥ +0.005 AUC on 2021-01..2024-06 with next-open entry.

**Minimal experiment.**
- tb7 features from `lab/decay_lib.py`; logistic per half-year; 200 bootstrap block resamples.
- Kalman logistic (Laplace approximation, process noise Q chosen on 2019-2020).
- A recency-weighted momentum baseline (the guard against Nagel's illusion).

**Protocol.** The norm and angle are descriptive. The adaptation gain is judged on selection months 2021-01..2024-06. Confirmation is forward data only (direction is spent).

**Cost.** CPU, under 1 hour.

**Risks.** None on the analysis side: it is decisive in either direction. The novelty is moderate; the value is that it closes a branch.

---

## H-7. A sign-free OCO bracket on the H-1 forecast: trade trendiness, not direction

**Idea.**
- If H-1 passes, its forecast can be traded **with no direction model at all**. At bar t place a buy-stop at C_t + aσ̂√T and a sell-stop at C_t − aσ̂√T (one cancels the other). After a trigger, take-profit is bσ̂√T and stop-loss is cσ̂√T in the direction of the trigger, with a time exit at T.
- The bracket's payoff is a pure bet on ρ (trendiness): it earns in trending windows and loses in choppy ones.
- Trade only in the top quintile of the H-1 forecast. The mirror variant (a fade bracket in the bottom quintile) is the "reversion in quiet regimes" idea from PLAN §4 in a sign-free form.

**Mechanism.** The project's trading failures (H48, H60, H61) were in choosing a side. Here the side comes from the market itself (the stop trigger), and the information is H-1's.

**Closest work, and how ours differs.**
- Breakout and channel systems (Donchian, Bollinger) are not conditioned on a trendiness *forecast*.
- Squeeze breakouts (H47) are volatility persistence, not shape.
- The project's triple barrier always took its side from a model or at random.

**Falsifiable prediction.** If H-1 gives ΔSpearman ≥ +0.02, the top-quintile bracket earns ≥ +1.0 bps per trade more than the same bracket on random bars with the same frequency and clock, above the 95th percentile of that null, with conservative fills.

**Minimal experiment.**
- Fills: a stop fills at max(stop, open of the bar that crosses it) for a buy, and the mirror for a sell. If entry and exit are both crossed inside one 1-minute bar, the trade counts as a loss.
- Non-overlapping trades; cost grid of 0/1/2/5 bps (`equity/eqlib.py`, `fwd/fwdlib.py`).
- (a, b, c) are chosen on selection months 2019-01..2024-06; every combination tried is counted for Holm.
- Confirmation: 2024-07..2025-07-20, one look. **This is clean**, because this strategy class (and its target) was never looked at there, although the period's prices were seen in direction work. The SPEC says so.

**Cost.** CPU, under 1 hour.

**Risks.** Stop entries on 1-minute data are slippage-sensitive. With "the worst open" fills the edge per trade may be eaten. Run it only after H-1 passes; otherwise it is a search for an edge without a reason.

---

## Recommended order and dependencies

1. **H-1 + H-2 + H-6** (CPU, about 3 hours in total, in parallel). They give the targets for H-3 and H-7 and close the "drift" question. The confirmation looks for H-1 and H-2 are clean.
2. **H-3 JEPA-DDS** only after H-1 and H-2. Its probes need their targets and hand-made baselines. The negative control on synthetic Brownian motion with a GARCH clock and jumps comes before the real pretraining.
3. **H-4** in parallel with H-3 (different GPU windows; at most 2 heavy processes, H43).
4. **H-5** for the risk-management branch (D-063 "a separate independent branch"), independent of the rest.
5. **H-7** only if H-1 passes its criterion.

Total: about 4-5 GPU-hours (H-3, H-4, H-5) and about 5-6 CPU-hours. **H-3 and H-4 do not fit D-063's 2-minute runs.** They need the owner's consent or resumable 2-minute chunks.

## Considered and rejected (with reasons)

- **A Fisher–Rao loss for a Gaussian head.** With a realised point instead of a distribution, the distance degenerates. In the (μ, log σ) parameterisation the Fisher matrix is diag(1/σ², 2), so the natural gradient for log σ equals NLL up to a constant. It is not a new hypothesis, only a preconditioning of μ.
- **Hit order and first-passage side relative to b/(a+b).** This is direction in disguise, and the 2024-25 look is spent.
- **Volume or activity as a target** (subordination, Ané–Geman). Volnet (H57) already takes half its gain from volume and range, so the area is saturated.
- **HNN / LNN / symplectic losses and PINN "price PDEs".** There is no conserved quantity or governing PDE.
- **Zero-shot Kronos or other foundation models.** The pretraining corpus very likely contains Binance BTC from our period, so the results would leak.

## Main risk across the whole set

Everything predictable at 10-60 minutes may be the clock: volatility, which HAR already captures. H-1 and H-3 are designed to give a **sharp, publishable negative** in that case: "in its own clock 1-minute BTC is a martingale". That would end the search for a direction model with evidence instead of with fatigue.