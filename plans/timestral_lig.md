# Timestral-LIG: learnable indicators and trend geometry inside Timestral

| | |
|---|---|
| Date | 2026-10-30 |
| Task | Inject learnable indicators and geometry inside Timestral; let the model (or us) govern architecture and losses |
| Method | Three independent designs plus a judge, then implementation, automatic checks, an 8-agent code review, a training ladder with controls on two data sets, pre-registered selection, one Test scoring |
| Prepared by | Claude Code (AI). All numbers come from `research/timestral_lig/results/`. |
| Related | [`plans/model_evaluation_timestral.md`](model_evaluation_timestral.md) (zero-shot evaluation) |

---

## 1. Verdict

**FAIL. Do not adopt Timestral-LIG.** The pre-registered pass rule fails, and no rung would pass it.

- **Selection on V2.** The pre-registered candidate is **A1**, the 10-parameter recalibration.
  - On the repo set, the indicator rung A2 was best, but it was not significantly better than A1. The Occam rule picked A1.
  - On the clean set, A2 was selected.
  - The common candidate is the lower rung, A1.
- **On Test, A1 fails P1, P2, P4, P5 and P6. It passes P3 and P7.**
  - Its QLIKE is 0.43% **worse** than zero-shot on the repo set (95% CI +0.10 to +0.77) and 0.37% better on the clean set (not significant).
  - Its pinball loss is 0.34% better on the repo set (significant) and 0.15% better on the clean set (not significant).
- **A2 would also fail (descriptive).** A2 is the full indicator and geometry rung at the quantile head. It would fail P1 and P3 and pass P2.
  - Pinball improves on both sets: −0.47% and −0.28%, both significant.
  - QLIKE does not improve significantly: −0.18% and −0.43%, both CIs include 0.
  - Tails get worse: 2 and 3 of 6 tail tests fail, against 1 for zero-shot.
- **Deeper injection showed no gain within the budget.** We fed features into the encoder tokens and the forecast query (A3), and added LoRA on attention (A4). With one seed each, both were worse than their start (A2 seed 0) at every V2 check from step 150 to 450, and early stopping ended them after 450 of 1,200 steps. On both sets the best checkpoint was step 0.
- **We could not tell a licence-free control apart from A2.** C-FHS-hi uses FHS quantiles, the same 62 features and a small recalibration, with no Timestral.
  - It is not significantly different from A2 on either set. The CIs are wide (A2 minus C-FHS-hi, QLIKE: repo −0.14 [−0.97, +0.67], clean +0.85 [−0.36, +2.20]), so this does not show that the two are equal.
  - Its QLIKE is lower on the clean set (−1.27% vs zero-shot, against −0.43% for A2). A2 is better on clean pinball at h = 1 (−0.44%, significant).
  - It fails 4 of 6 tail tests on the clean set.
- **The geometry losses showed no consistent benefit.** Without them (A2-g0), the clean V2 score was better (−1.33 vs −0.99). On Test the difference was mixed and mostly not significant (repo pinball 0.09% better with them).
- **What the model learned.**
  - The gates start at 0.88. The volatility, calendar, path-shape and Bollinger gates stayed near that value (0.75–0.94).
  - The EMA-distance, MACD, MACD-pairs, RSI and momentum gates fell to 0.49–0.57. The range and trend-geometry gates fell to 0.58–0.71.
  - It did not move the indicator periods much.
  - Its largest single change was to shrink Timestral's median toward the last price: 76–82% at h = 1 and 100% from h = 3–4 (A2). Part of this comes from our random-walk penalty on the median. Without the geometry losses (A2-g0), the shrink at h = 1 is only 49–74%.

**Why (our reading; not tested by a dedicated experiment).**
- Most indicators (51 of 62) are functions of the closes that the backbone already reads, and we could not detect what they add. A2-close (without range and clock) against A1 was not tested; its point estimates favour A2-close slightly.
- Only two groups carry information that is not in the closes: the high-low range and the clock. They helped repo QLIKE (−0.38% vs A2 without them, descriptive) and not clean QLIKE.
- Recalibration alone (A1) helped pinball (repo −0.34%, significant) but made repo QLIKE worse (+0.43%, significant).
- The V2 gains shrank on Test.
- None of the descriptive gains reaches the pre-registered bar.

**What to keep.**
- The indicator bank (`research/timestral_lig/indicators.py`) is a correct, causal, tested replacement for
  neural_trade's `LearnableIndicators`. It fixes defects D1 and D7 by design.
- An idea to test in neural_trade's risk path: add the high-low range and the time of day to the variance model. They helped repo QLIKE by 0.38% (descriptive); on the clean set the point estimate went the other way.
- Shrinking the median toward the last price removed most of Timestral's small negative price skill (MAE skill −0.006 → −0.001 at h = 1). We did not isolate its effect on pinball and QLIKE.

**Licence.** The trained adapters are derivatives of Timestral-28M under its Modified MIT licence. Clause 2 (the $20M
revenue condition) applies to them as to the model. We did not commit any weights. Use, including benchmark use,
needs written legal clearance first (see the zero-shot report, Section 1).

---

## Terms

| Term | Meaning |
|---|---|
| A0 | Zero-shot Timestral-28M, context 2048, with symmetric averaging (the reference) |
| A1-A4 | The ladder: A1 = recalibration G only; A2 = A1 + indicators at the quantile head (H); A3 = A2 + encoder tokens (E) and forecast query (D); A4 = A3 + LoRA (rank-4 low-rank adapters on attention weights) |
| A2-plc, A2-g0, A2-close | Controls: placebo features from another time; no geometry losses; no range and calendar groups |
| C-FHS, C-FHS-hi | Licence-free controls: FHS quantiles plus a conditional recalibration on the same 62 features, no Timestral. C-FHS uses offset learning rate 3e-4, C-FHS-hi 1e-2 |
| Train / V1 / V2 / Test | Data blocks per set; V1 (1 day) sets loss weights; V2 (3 days) does early stopping and selection; Test is scored once |
| V2 score | Mean over h = 1, 5, 15 of 0.5 x (QLIKE % change + pinball21 % change) against A0; negative = better |
| QLIKE, pinball21, FHS | As in the zero-shot report: variance score, 21-quantile pinball loss, filtered historical simulation |
| Seed ensemble | The mean of the seeds' quantile forecasts (it stays monotone) |
| Beats, promoted | Rung a beats (is promoted over) rung b if its V2 score improves by at least max(0.1%, 2 x a's seed SD) and the paired CI excludes 0 |
| Occam rule | The candidate is the lowest ladder rung (A0-A4) that the best ladder rung does not beat. Controls do not take part |
| Abort rule | From the design: if A3 and A4 seed 0 both stop at step 0 on both sets, train no more T2 seeds |
| Tail tests | 6 Newey-West (overlap-robust) exceedance tests: the 1% and 99% quantiles at h = 1, 5, 15. A test fails when p < 0.05 |
| Spec, design | The judged design from three independent design agents (fields: injection, indicators, geometry, losses, governance, protocol) |
| SD, pp, RMS, AUC, DST | Standard deviation (here: across seeds); percentage points; root mean square; area under the ROC curve for the return sign (0.5 = no skill); daylight-saving time |
| D1-D8, EWMA-t, metrics2.py | As in the zero-shot report; metrics2.py is the benchmark scorer in `research/tsfm_eval/bench/` |
| bps | Basis points; 1 bps = 0.01% |

---

## 2. What we built

### 2.1 Where the features enter

Timestral-28M stays frozen (27.8M parameters, loaded with `strict=True`, package not edited). Every new part starts at
exactly zero, so the model at step 0 is zero-shot Timestral. We checked this on every Test origin: the largest
difference is 0.0007 bps on the repo set and 0.0005 bps on the clean set (float32 storage rounding).

| Site | Where | Size | What it can do |
|---|---|---|---|
| E | added to the 64 encoder patch tokens | 33k | change what the transformer reads from each 32-minute patch |
| D | added to the forecast-query token (x 0.1) | 29k | steer what the decoder asks the context |
| H | added to the raw quantile-head channels, before the monotone transform | 25k | change the median (x 0.03) and the width of each quantile step, per horizon step |
| G | 10-parameter recalibration of the quantile fan | 10 | shrink the median toward the last price; widen or narrow each side, by step and by tail |
| LoRA | rank 4 on time attention `wQKV` and `wO`, blocks 8-11 | 37k | adapt the last four attention blocks |

The model keeps Timestral's symmetric averaging: each origin runs as two rows, `x` and `-x`, and the output is the
mean of the forecast and the mirrored forecast of `-x`. The mirror row gets mirrored features (every feature is odd or
even under `x -> -x`). We train the same averaged forecast that we score.

### 2.2 Features: learnable indicators and trend geometry

62 features in 11 groups. Each group has one gate `z = sigmoid(a)` that the model learns. Each feature is unit-free
(divided by a causal volatility, a ratio, or bounded), then `asinh` and a clamp at ±5. Nothing is fitted on later data.

| Group | Features | Learnable periods | New to the backbone? |
|---|---|---|---|
| EMA distance | (y − EMA) / (σ√P) at P = 5, 15, 30 | 3 spans | no (function of closes) |
| MACD | line, signal, histogram at (15,30,5), (45,60,30), (30,45,15) | 9 spans | no |
| MACD pairs | the same at (30,60,9), (15,30,9), (25,45,9) | 9 spans | no |
| RSI | (RSI − 50)/50 at 9, 21, 30 | 3 spans | no |
| Bollinger | %B and log band width at 10, 20, 50 | 3 spans | no |
| Momentum | (y_t − y_{t−L}) / (σ√L), fractional L from 1, 2, 3 | 3 lags | no |
| Trend geometry | slope and curvature at 1, 5, 15 bars (neural_trade's `EXTENDED_TREND_PERIODS`) | fixed | no |
| Volatility geometry | log EWMA vol at 5, 15, 60, 240, 1440; log variance ratios at 5, 15 | 6 spans | no |
| Path shape | efficiency ratios (15, 60), realised skew, log kurtosis, Timestral's scale frame | 3 spans | frame only |
| Range | Parkinson high-low vol, short/long range ratio | 1 span | **yes** (high and low) |
| Calendar | UTC and New York time of day (sin, cos, two harmonics), weekend | fixed | **yes** (clock) |

Periods start at neural_trade's `Config` values. The two neural_trade defects in this area are fixed by design:
there is no x20 forward multiplier on the period logits (D1) and no asinh write-back of momentum periods (D7). Every
EMA is exact: `last` mode uses a matmul with the EMA kernels, and `seq` mode (for site E) uses a blocked triangular
scan, so a feature at step t never reads a later bar, not even through floating-point roundoff.

### 2.3 Losses

The loss is computed on the averaged forecast, in log-return bps from the last close, at all 15 steps (weight 1/6 at
h = 1, 5, 15 and 1/24 at the other steps).

| Term | Formula (short) | Weight | Role |
|---|---|---|---|
| Pinball | 21-level pinball / (σ_t √k), divided by its zero-shot Train mean | 1 (fixed) | anchor; strictly consistent for the true quantiles |
| QLIKE | y²/v + ln v of the quantile-implied variance | governed, 0.025-1 | the headline variance score |
| Fan | soft hinge: widths must not shrink as k grows | governed | geometry |
| Term structure | log-log term structure of widths stays straight | governed | neural_trade's cross-horizon coherence, on widths |
| Tail | tail ratio W98/W50 must not rise with k | governed | aggregational Gaussianity |
| Median curvature | median path must not bend | governed | neural_trade's coherence penalty, on the median |
| Median random walk | (median / σ√k)² | governed | random-walk prior (no model has price skill) |
| Gates | 1e-3 x sum of gates (ramped in) | fixed | feature selection |

Governed means that the model sets the weight during training (Section 2.4).
- **Start values.** QLIKE 0.25 (range 0.025–1), fan 1.0, tail 0.2. Term structure, median curvature and median random walk start at 3% of the pinball loss on Train, so their start differs per set.
- **Limits.** Each geometry weight stays within 0.1x to 10x of its start.

neural_trade's trend and coherence losses used only the targets, so they had no gradient. Here every geometry term
acts on the predictions and has a gradient.

### 2.4 Who decides what

| Decided by the model | Decided by us, before Test |
|---|---|
| 40 indicator periods and lags | the anchor score (pinball, weight 1) and horizon weights |
| 11 group gates (identifiable: first layers have fixed column norms per group) | site scales, weight ranges, gate strength |
| QLIKE and geometry weights: every 25 steps, each term's gradient is compared with the gradient of pinball on V1 (held out); weight up when they agree, down when they conflict | which sites exist (the ladder) |
| the adapters themselves; early stopping can return step 0 | the pass rule, written before Test |
| which rung is best on V2 | which rung is the candidate (Occam rule), frozen in git before Test |

## 3. Protocol

- **Data.** For each set, Train / V1 (1 day) / V2 (3 days) / Test. Blocks are separated by 76 bars (16-bar label purge
  plus a 60-bar embargo). Test origins are the benchmark's (repo 2,895; clean 6,715). Each set is trained and tested on
  its own.
- **Tier T1 (cached head).** Sites H and G only, trained on cached backbone outputs. Batch 256 origins, up to 20
  epochs, early stopping on V2.
- **Tier T2 (backprop through the frozen stack).** Adds E and D (A3), then LoRA (A4). Warm start from the same seed's
  A2. Batch 8 origins, up to 1,200 steps, early stopping on V2 (step 0 allowed).
- **Ladder.** A0 zero-shot; A1 = G only; A2 = A1 + H with all 11 groups; A3 = A2 + E + D; A4 = A3 + LoRA.
- **Controls.**
  - A2-plc: placebo, each row gets another origin's features.
  - A2-g0: no geometry losses.
  - A2-close: no range and no calendar groups.
  - C-FHS / C-FHS-hi: licence-free; FHS quantiles with a conditional recalibration on the same features, no Timestral.
- **Seeds.** A1 and A2: 5 seeds. Controls: 3 seeds. T2: seed 0; more seeds only for a promoted T2 rung.
- **Selection (V2 only).**
  - Rung a beats rung b if its V2 score improves by at least max(0.1%, 2 x a's seed SD) and the paired 60-minute block-bootstrap CI excludes 0.
  - The best rung is the ladder rung (A0–A4) with the lowest V2 score. Controls do not take part.
  - The candidate is the lowest ladder rung that the best rung does not beat (Occam rule).
  - The common candidate is the lower of the two sets' candidates. `candidate.json` (with the SHA-256 of every
  checkpoint) and `preregistration.json` were committed to git (commit 46ce1ed) before any Test inference.
- **V2 score.** The mean over h = 1, 5, 15 of 0.5 x (QLIKE % change + pinball21 % change) against zero-shot.
- **Pass rule (P1-P7, Test, both sets; aggregate = mean over h = 1, 5, 15).**
  - P1, QLIKE: aggregate at most −0.5% with the CI below 0, and no h cell significantly worse.
  - P2, pinball: CI upper bound at most +0.10%, and no h cell point estimate above +0.30%.
  - P3, tails: the tail tests fail no more often than for zero-shot.
  - P4, seeds: every seed has a QLIKE gain (point estimate) and meets P2.
  - P5, attribution: on Test, the candidate's QLIKE is lower than A1's on both sets, with the CI below 0 on at least one set. On V2, A2 beats the placebo on both sets (CI below 0).
  - P6, licence: on Test, the candidate's QLIKE is lower than the licence-free control's, with the CI below 0 on both sets.
  - P7, price: the median's MAE skill against a random walk does not fall by more than 0.2 pp at any h.
- **Verdicts.**
  - PASS if P1–P7 all pass.
  - PASS-licence-free if P1–P5 pass and P6 fails.
  - PASS-recal if A1 meets P1–P4 and P5 fails.
  - Otherwise FAIL.

---

## 4. Results

### 4.1 Selection on V2 (both sets, before Test)

V2 score against A0 (negative = better) for the seed ensemble, with the 95% block-bootstrap CI and the range of
per-seed values. **Bold** = CI excludes 0.

| Rung | Repo V2 | Repo seeds | Clean V2 | Clean seeds |
|---|---|---|---|---|
| A1 recalibration | −0.52 [−1.18, +0.08] | −0.51 to −0.52 | −0.29 [−0.87, +0.22] | −0.28 to −0.30 |
| A2 indicators at the head | **−0.68 [−1.43, −0.02]** | −0.64 to −0.72 | **−0.99 [−2.47, −0.12]** | −0.87 to −1.06 |
| A3 (+ encoder, query) | −0.64 (= A2 seed 0, step 0) | 1 seed | −1.01 (= A2 seed 0, step 0) | 1 seed |
| A4 (+ LoRA) | −0.64 (step 0) | 1 seed | −1.01 (step 0) | 1 seed |
| A2-plc placebo | −0.54 [−1.20, +0.07] | | −0.28 [−0.99, +0.35] | |
| A2-g0 no geometry losses | −0.62 [−1.45, +0.07] | | **−1.33 [−3.38, −0.12]** | |
| A2-close no range, no calendar | −0.59 [−1.25, +0.02] | | −0.46 [−1.12, +0.08] | |
| C-FHS (licence-free) | +0.04 | | +0.47 | |
| C-FHS-hi (licence-free) | −0.86 [−2.64, +0.58] | −0.50 to −0.92 | −0.78 [−1.86, +0.20] | −0.44 to −0.90 |

The selection by the rule:

| Set | Best rung | Best vs A1 | Candidate |
|---|---|---|---|
| Repo | A2 | −0.17, CI [−0.51, +0.07]: not beaten | **A1** (Occam) |
| Clean | A3 (= A2 seed 0) | −0.72, CI [−1.65, −0.18]: beaten; vs A2 −0.02, CI [−0.09, +0.02]: not beaten | **A2** |

- **Why A3 is "best" on the clean set.** Its best checkpoint is step 0, which is A2 seed 0. That seed scores −1.01, against −0.99 for the A2 seed ensemble. The difference (−0.02) does not meet the rule.
- **Controls.** C-FHS-hi on repo and A2-g0 on clean have better V2 scores than the best ladder rung. Controls do not take part in the selection.
- **Common candidate:** A1, the lower of the two.
- **Licence-free control:** C-FHS-hi on both sets (stronger than C-FHS on V2).
- **A3 and A4:** never beat their A2 start on either set. So, under the spec's abort rule, we trained no further T2 seeds and no LoRA-only control (A4L).
- **Seal:** `candidate.json`, `preregistration.json` and the SHA-256 of all 27 checkpoints per set were committed (46ce1ed) before any Test inference. Export verified every hash.

### 4.2 Test: the pass rule

| Check | Result | Detail |
|---|---|---|
| P1 QLIKE ≤ −0.5%, CI < 0, both sets | **fail** | A1: repo +0.43 [+0.10, +0.77] (worse), clean −0.37 [−0.91, +0.09]; repo h = 1 cell +0.26 [+0.13, +0.39] is also significantly worse |
| P2 pinball CI upper ≤ +0.10% | **fail** | repo −0.34 [−0.66, −0.05]; clean −0.15 [−0.39, **+0.1006**], fails by 0.0006 |
| P3 tails no worse than A0 | pass | A1 fails 1 of 6 on each set, as A0 |
| P4 every seed | **fail** | QLIKE is worse than A0 for all 5 repo seeds |
| P5 attribution | **fail** | the candidate is A1 itself (A1 vs A1 = 0). Separately, A2 did not beat the placebo on repo V2 (−0.15, CI [−0.44, +0.07]); on clean V2 it did (−0.71, CI [−1.48, −0.26]) |
| P6 beats the licence-free control | **fail** | A1 vs C-FHS-hi QLIKE: repo +0.47, clean +0.92 (both CIs include 0) |
| P7 price skill not lower | pass | A1's MAE skill vs a random walk is −0.1 pp at h = 1 (A0: −0.6 pp) and 0.0 pp at h = 5 and 15 |
| PASS-recal (A1 meets P1-P4) | **fail** | |
| **Verdict** | **FAIL** | |

### 4.3 Test: every rung against zero-shot (descriptive)

% change of loss against A0 on identical origins; negative = better; **bold** = 95% CI excludes 0. Aggregate = mean
over h = 1, 5, 15. Tails = overlap-robust tail tests failed, of 6 (A0 fails 1 on each set).

| Rung | Repo QLIKE | Repo pinball21 | Repo tails | Clean QLIKE | Clean pinball21 | Clean tails |
|---|---|---|---|---|---|---|
| A1 | **+0.43** | **−0.34** | 1 | −0.37 | −0.15 | 1 |
| A2 | −0.18 | **−0.47** | 2 | −0.43 | **−0.28** | 3 |
| A3 = A4 = A2 seed 0 (step 0) | +0.00 | **−0.45** | 2 | −0.35 | **−0.28** | 3 |
| A2-plc | **+0.43** | **−0.36** | 2 | −0.41 | −0.15 | 2 |
| A2-g0 | −0.10 | **−0.37** | 3 | −0.36 | **−0.38** | 2 |
| A2-close | +0.20 | **−0.41** | 3 | −0.64 | **−0.24** | 2 |
| C-FHS-hi (no Timestral) | −0.03 | −0.37 | 0 | −1.27 | −0.10 | 4 |
| C-FHS (no Timestral) | +0.48 | −0.30 | 0 | −0.05 | −0.01 | 4 |

Per horizon, A2's QLIKE gain sits at h = 1 on the repo set (−0.60%) and at h = 5 and 15 on the clean set (−0.65% and
−1.12%). At h = 1 on the clean set it is worse (+0.47%). Only the repo h = 1 cell is significant (95% CI −1.14 to
−0.04).

### 4.4 What the indicators and the geometry add (descriptive, Test)

Paired on identical origins; negative = the first model is better.

| Comparison | Repo QLIKE | Repo pinball21 | Clean QLIKE | Clean pinball21 | Reading |
|---|---|---|---|---|---|
| A2 vs A1 (indicators vs recalibration only) | **−0.60** [−0.95, −0.21] | −0.12 | −0.07 | **−0.13** [−0.19, −0.07] | favours A2 on repo QLIKE and clean pinball (descriptive; P5 failed, so the indicators stay unproven) |
| A2 vs A2-plc (real vs placebo features) | **−0.60** [−0.95, −0.21] | −0.11 | −0.03 | **−0.13** [−0.19, −0.08] | real features beat placebo features on Test (descriptive); the pre-registered V2 placebo check failed on the repo set |
| A2 vs A2-close (with vs without range and clock) | **−0.38** [−0.67, −0.06] | −0.06 | +0.20 | −0.04 | range and clock help on repo only |
| A2 vs A2-g0 (with vs without geometry losses) | −0.07 | **−0.09** [−0.18, −0.02] | −0.07 | +0.10 | no consistent effect |
| A2 vs C-FHS-hi (Timestral + LIG vs licence-free) | −0.14 | −0.10 | +0.85 | −0.18 | no significant difference; wide CIs, so not shown to be equal |

### 4.5 Against the free baselines

QLIKE % change against FHS (metrics2.py), h = 1 / 5 / 15; **bold** = CI excludes 0.

| Model | Repo | Clean |
|---|---|---|
| A0 zero-shot Timestral-28M | −0.52 / **−1.23** / **−1.59** | **−2.77** / **−1.97** / −2.05 |
| A1 | −0.26 / −0.93 / −0.88 | **−2.80** / **−2.28** / −2.79 |
| A2 | −1.12 / −1.22 / −1.53 | **−2.31** / **−2.61** / −3.15 |
| C-FHS-hi (no Timestral) | −0.25 / −1.27 / −1.93 | **−2.44** / **−3.69** / **−4.39** |

A1 and A2 stay 0.26–3.15% better than FHS on QLIKE (point estimates), as zero-shot Timestral was (0.52–2.77%). The
licence-free control C-FHS-hi is 0.25–4.39% better. LIG does not widen the gap in a way we can detect.

- **Price.** The medians of A1 and A2 move 54–82% of the way to the last price at h = 1, and all the way from h = 3–4 (A2). Their MAE skill against a random walk is −0.0004 to −0.001 at h = 1 and 0.000 at h = 5 and 15, against −0.006 for A0 at h = 1.
- **Direction.** AUC stays at 0.49–0.53. LIG has no price or direction skill.

### 4.6 What the model learned

| Lever | Repo (A2, 5 seeds) | Clean (A2, 5 seeds) |
|---|---|---|
| Gates kept open | volatility 0.89, calendar 0.89, path shape 0.84, Bollinger 0.75 | calendar 0.94, volatility 0.93, path shape 0.89, Bollinger 0.82 |
| Gates half closed | MACD, RSI, momentum, MACD pairs, EMA distance: 0.49–0.54; range 0.58; trend geometry 0.66 | the same groups 0.54–0.57; range 0.71; trend geometry 0.67 |
| Periods | almost unchanged; largest moves 240 → 249 (volatility geometry), 240 → 245 (path shape: skew and kurtosis span), 50 → 52 (Bollinger) | 240 → 252 (volatility geometry), 240 → 257 (path shape) |
| Loss weights | fan 1.0 → 0.59, median random walk 6.6 → 5.0; others within 12% of start | QLIKE 0.25 → 0.33, fan 1.0 → 1.15; others within 8% of start |
| Mean cosine with held-out pinball | −0.17 (fan) to +0.03 (tail) | −0.02 to +0.08 |
| Recalibration G | median shrink 80–82% at h = 1, 100% from h = 3 (A1: 54–80%) | 76–80% at h = 1, 100% from h = 4 (A1: 71–79%) |
| Site H | increments change by 0.06–0.10 raw units RMS; the median by 0.001 | 0.04–0.07; 0.001 |
| Sites E, D, LoRA (A3, A4) | best checkpoint = step 0 (worse at steps 150–450) | step 0 (A3 at step 300 within 0.01 of it) |

- The gates start at 0.88. No gate closed below 0.1, so no group is fully switched off.
- Gradient alignment found one consistent conflict: on the repo set the fan term disagreed with held-out pinball in all five A2 seeds (mean cosine −0.16 to −0.18), and its weight fell by about 40%. The other cosines were near 0, and those weights moved less.
- The median shrink comes partly from our random-walk penalty: without the geometry losses (A2-g0) it is 49–51% (repo) and 69–74% (clean) at h = 1, and it does not reach 100%.

### 4.7 Cost

| Item | Value |
|---|---|
| Trainable parameters | A1 10; A2 24,568; A3 86,520; A4 123,384 (0.44% of the backbone) |
| Training time, 2 CPU threads | T1 run 3–9 s (A1) to 1.2–2.6 min (A2); T2 run 7.5–8 min (repo A4: 14.6 min, partly while paused) |
| Single-origin latency, 4 threads, ctx 2048 (p50 / p99) | A0 29.8 / 34.0 ms; A2 35.2 / 50.2 ms; A3 39.3 / 45.0 ms |

---

## 5. Why it did not work (our reading)

These points explain the results. No dedicated experiment tested them, and the comparisons behind them are descriptive.

1. **Most features repeat what the backbone reads.**
   - 51 of the 62 features are functions of the 2,048 closes in the context.
   - We could not detect what they add.
   - Feeding all features deeper (encoder tokens, query, LoRA) never improved V2 within the budget.
   - The gates do not separate redundant from new information. Three close-derived groups stayed near their start (volatility, path shape, Bollinger), and the range group fell to 0.58–0.71.
2. **New information helps only a little.**
   - The two groups with information that is not in the closes are the high-low range and the clock.
   - Removing both cost 0.38% QLIKE on the repo Test. On the clean Test, the point estimate went the other way (+0.20%, not significant).
3. **Recalibration alone helps pinball, not QLIKE.**
   - On V2, A1 scored −0.52% and −0.29%, and neither CI excludes 0.
   - On Test it improved pinball (repo −0.34%, significant) but made repo QLIKE worse (+0.43%, significant).
   - A1's tail tests stayed at A0's level.
   - A2's outer bands became too wide: 98% coverage rose from 0.973–0.985 (A0) to 0.978–0.988, and 2 and 3 of 6 tail tests failed.
4. **The sample is small for the effect size.**
   - V2 is 3 days per set and Test is 6 or 14 days.
   - Effects of 0.1–0.5% are near the detection limit (CIs about ±0.3–1.0%).
   - A2's V2 score (−0.68% and −0.99%) fell to −0.32% and −0.36% for the same score on Test. Its QLIKE alone went from −0.55% and −1.55% on V2 to −0.18% and −0.43% on Test.
5. **We could not show that the backbone adds anything beyond the features.**
   - The licence-free control with the same features is not significantly different from A2 on Test.
   - The CIs span about ±1–1.5% QLIKE.
6. **The geometry losses showed no consistent benefit.**
   - Without them (A2-g0), the clean V2 score was better (−1.33 vs −0.99).
   - On Test, A2 with them is 0.09% better on repo pinball, and its clean h = 1 QLIKE is 0.26% worse (descriptive).

---

## 6. Changes made during the work (all before Test scoring)

| When | Change | Effect |
|---|---|---|
| Before training | The `seq`-mode EMA used a zero-padded FFT. A check found that later bars changed earlier features by about 1e-8, and by 0.06 for realised skew at step 0. We replaced it with an exact blocked triangular scan. | Exact causality (change 0.0) |
| Before training | Parity signs of odd and even features were swapped; `sqrt` of tiny negative FFT values gave NaN | Fixed; parity check 0.0 |
| After an 8-agent review (4 lenses, each verified by a skeptic): 14 confirmed findings (10 distinct issues), none in the trained A1-A4 models | `select` compared each rung with the last promoted rung, not as the spec says. We implemented the spec rule: the candidate is the lowest rung the best rung does not beat. | Changed the repo candidate from A2 to A1, by the spec's own rule |
| same | The T2 threshold used the largest seed SD of any rung (0.24, from the C-FHS control at its original learning rate, now C-FHS-hi) | Now A2's SD |
| same | The pre-registration was not enforced | Final select hashes every checkpoint; prereg freezes both sets; export verifies hashes and refuses extra checkpoints; re-registration fails once Test files exist |
| same | Verdict logic: different candidates per set forced FAIL; PASS-licence-free could fire on a P7 failure; thresholds compared rounded values; one set's seed lists could be used for the other | Fixed |
| same | The placebo could take features from 1–60 bars after the origin, which contain the label path, for 0.05–1% of rows | Sampler fixed; A2-plc retrained on both sets |
| same | C-FHS's conditional offsets could move up to about 20x more per step than its shared parameters (about 3–4x more V2-score volatility, as measured by the verifier) | We lowered the lr, which made C-FHS weak on V2 (+0.04 and +0.47). A weak control would favour LIG, so we kept both lrs (C-FHS and C-FHS-hi) and registered the one that was stronger on V2. This is the conservative choice for LIG. |
| Not changed | The 1440-bar volatility span starts at its upper clamp and cannot move (judged not a defect by the verifier) | Measured effect < 0.003 pp |

---

## 7. Limits and risks

- **Validation reuse.**
  - V2 did early stopping for every run and chose the candidate, so V2 scores are optimistic.
  - V2 also chose between the two C-FHS learning rates.
  - Only Test, scored once, is confirmatory.
- **Possible under-training.** On the repo set, 4 of 5 A2 seeds were still improving at the 20-epoch cap.
- **Thin T2 evidence.** A3 and A4 used one seed each and stopped after 450 of 1,200 steps. V2 checks used a 720-origin subsample.
- **Many comparisons.** Only P1–P7 are confirmatory. The tables in 4.3–4.6 are descriptive.
- **One market, short windows.**
  - BTC/USDT 1-minute data, with 18–20 days of training per set.
  - The repo Test window spans the 2025-11-02 DST change.
  - The clean Test window covers one 14-day regime.
- **Not run.**
  - The LoRA-only control (A4L), because A4 was not promoted.
  - The transfer check (repo-trained model on the clean set).
  - The sealed forward window, which the spec requires only after a PASS.
  - A full fine-tune, rejected in the design (27.8M parameters on about 2,000 independent 15-minute windows).
- **Licence.** As in Section 1.

---

## 8. Recommendation

1. **Do not adopt Timestral-LIG** in any role. The plan in the zero-shot report stands:
   - fix neural_trade (D1–D8);
   - wire a cheap risk input (FHS or scale-refit EWMA-t);
   - use zero-shot Timestral only as an optional benchmark, after legal clears the licence.
2. **Reuse the indicator bank** (`research/timestral_lig/indicators.py`) in neural_trade in place of `LearnableIndicators`.
   - It is causal, tested for parity and gradient flow, and free of D1 and D7.
   - We could not detect a forecast gain from its close-derived groups.
3. **Optional, licence-free.** Test C-FHS-hi (FHS quantiles plus a small recalibration that uses the same 62 features) against FHS in the risk path.
   - It beat FHS by 2.4–4.4% QLIKE on the clean set (significant), but only by 0.25–1.9% on the repo set (not significant).
   - Its seeds vary the most of any model.
   - It fails 4 of 6 tail tests on the clean set.
   - Fix the seed spread and the tails before any use.
4. **For any learned median, shrink it toward the last price.**
   - Here the shrink removed most of Timestral's small negative price skill.
   - Its effect on pinball and QLIKE was not isolated.

---

## 9. Reproduce

Code and per-run logs are in `research/timestral_lig/` (see its README). Steps:
1. Run `train.py cache` and `checks.py` for each set.
2. Run `train.py t1` and `train.py t2` (the ladder).
3. Run `train.py select --final` for both sets, then `train.py prereg`.
4. Run `train.py export` for both sets, then `score.py`.

The full run took about 3 hours on 4 CPU threads. Adapter checkpoints and forecast files are not in git.
