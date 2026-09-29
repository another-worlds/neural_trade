
Commands: from D:/nt_research/wfp/C, run `CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python <script>`. Run the two re-prediction scripts from ./scratch. "Measured" means computed by a named script; "Estimate" means a stated assumption. Numbers from the test fold (fold -1) are used only for sizing, and are labelled.

## Answer
**Purge rule: (a+).** It is the label-overlap rule for unbounded memory, keeping D-005's gap for the finite window.
- **Gap between adjacent blocks** = max(2·max(H), W + max(H)). W is the longest finite window any consumer reads: 60 today (the network input, realized_vol, DIRECTION_SKIP, logreg_lags, HD). So the gap stays **80 bars**: anchors identical to today, zero extra cost.
- **Series state** reads every earlier bar, as it does live. It resets, with a masked burn-in of M(1e-3) + W, only at the data start and at data gaps, including flat zero-volume runs. Training, evaluation and the Predictor do this identically.
- **Judgement folds** come after every fold whose score makes a choice.
- **D-005 taken literally** (reset plus burn-in inside every gap) costs 12.5% / 43% / 246% / 1,711% of a 7-day training block at learned periods of 60 / 240 / 1,440 / 10,080 bars, after the maximal ±0.5-logit shift (Measured, calculation). With D-032's unlimited periods that is infeasible, and it closes no leak that (a+) leaves open.

**Pair counts.** Measured noise, 80% power, margin = 1/3 of arm A's edge over constant variance, primary on horizon-mean log CRPS, 5-day out-of-sample blocks, one seed per fold:
- **14–28 judgement folds** at the planning edge of 0.0195. That edge is the latest run's test fold, used for sizing only. The range covers noise inflation of ×1.0–1.5 for 7-day training (Estimate).
- **5–8 folds** if the study's dev folds show an edge near the 0.040 measured on dev fold −2. I recommend a floor of 10.
- With h1 alone as the primary: 24–48 folds, or 6–10.

**What drives the design** (both Measured):
- **Short evaluation blocks are costly.** With 1-day blocks the paired SD is ×2.8 the 5-day value, so about 8× the pairs.
- **Few folds do not work.** With 3 folds, a fold-clustered test has only 15–42% power. The naive i.i.d.-pairs test passes a null at the margin 9–20% of the time instead of 5%, because arm×fold effects exist. So spread the pairs over many folds, one seed each.

## Files (all in D:/nt_research/wfp/C/)
| Script | Output | Contents |
|---|---|---|
| q1_purge_costs.py | .json | M(eps) and rule costs |
| q1_folds_today.py | .json | Today's folds, via the repo's make_purged_splits imported read-only |
| q1_gap_census.py, q1_flat_runs.py | .json | Missing minutes and flat zero-volume runs in Bitcoin_BTCUSDT.csv |
| q2_noise_v1.py | .json, q2_v1_per_run.csv, .log | Variance components of the v1 grid |
| q2_notebook_runs.py | .json | 7 notebook runs |
| q2_block_scaling.py | .json, .log, q2_block_scaling_preds/*.npz (12) | CPU re-prediction of 12 v1 dev-fold runs |
| q2_dev_edge_checkpoint.py | .json | Dev-fold edge with checkpoint weights |
| q3_val_grouping_check.py | .json, .log | Val metrics against validation batch grouping |
| q3_probe_target.py, q3_probe_classification.py | .json | The probe's target rule on real curves, and its error rates |
| q3_power.py | .json, .log | Pair counts |
| q3_planning_counts.py | .json | Planning counts for the recommended designs |
| q4_comparator_sim.py | .json, .log | Comparator error rates; 4,000 reps, fixed seed |


---

## Q1 Purge rule

**Code facts**
- **Label of anchor i** = close[i+h−1] − close[i−1]: increments over bars i..i+H−1 (windowing.py:3-8, :88).
- **Today's gap** = LOOKBACK + max(H) = 80 (splits.py:36). The invariant is "no bar is both a training label and an evaluation input" (splits.py:26-29), pinned by test_data_processor.py:9-31.
- **Shift range.** meta_adjust = tanh(·) (gru_attention.py:56) × meta_scale 0.5 (learnable_indicators.py:33, :115), so the logit shift is ±0.5. The longest applied period uses logit − 0.5.
- **Today's folds (Measured):** TimeSeriesSplit expanding windows; every training block starts at sequence 0.
  - Fold −1's train [0, 30212] contains folds −5…−2's val and cal blocks, plus the first 1,264 anchors of fold −2's test block.
  - Fold −1's val [30293, 33158] and cal [33239, 36104] lie inside fold −2's test [28949, 36184].
  - Within each fold, the minimum distance between label increments of adjacent blocks is 62 bars. All 30 block pairs pass label-disjointness with a max(H) embargo, and D-005.
- **NT-041's plan:** 7-day training blocks at configured dates, held-out folds "in different months".
- **Fixed lambdas already exist:** train_and_evaluate(calibrate=False) plus LAMBDA_* overrides. The v1 grid used this (ablation.py:186-217).

**M(1e-3)** (Measured, calculation; the cascade closed form was checked by brute force, 454 = 454):

| Learned period | Period after max shift | Single EWMA | No shift | Cascade of 2 EWMAs |
|---|---|---|---|---|
| 60 | 98 | 340 | 208 | 454 |
| 240 | 395 | 1,365 | 829 | 1,824 |
| 1,440 | 2,374 | 8,198 | 4,974 | 10,958 |
| 10,080 | 16,618 | 57,399 (≈40 days) | 34,816 | 76,723 |

**Leakage argument**

Label uses within one fold:
- T (train) labels: gradients and the target scaler.
- V (val) labels: early stopping, the LR schedule, and the served epoch (D-011).
- C (cal) labels: temperature, conformal, delta beta, and the strategy thresholds.
- X (test) labels: the verdict only.

Rule (a): label overlap plus an embargo; the state reads all earlier bars.
- **Later labels into earlier fits.** A fit on block P sees bars only up to P's last label bar. A later block Q's increments start after that when gap ≥ H−1. So no later label enters any gradient, epoch choice or calibrator. This holds only if every input path is causal: the recurrence, the shift context, and the normalisers. Today the target scaler and the window normaliser are fit on train (processor.py:166, :176).
- **Earlier labels into later inputs.** Q's state reads P's labelled bars, but those bars are Q's past and are available live. Memorisation adds nothing the history lacks. What remains is serial dependence right after the boundary, which is exactly the live situation. The embargo of max(H) keeps the label increments apart, with a margin for short-lag autocorrelation and off-by-one errors.
- **Across folds.** A later fold training on an earlier fold's evaluation blocks is legitimate: they are its past. The real risk is **choices**. A configuration chosen on a dev fold whose training data or memory contains a judgement block has been weakly tuned on judgement data.
  - Today the judgement fold is the latest, so this is fine. Choices made on fold −2's test block do touch fold −1's val and cal data, but not its test block.
  - Under NT-041, a dev fold placed after a judgement fold reads it through memory. The read range is train start − M − W, which is 57k bars at a period of 10,080. Hence the fold-placement rule.
- **Paths that tests must pin:**
  - whole-file or whole-block normalisers (RevIN);
  - backward NaN propagation in the matrix form;
  - batch statistics that include val or test bars.

Rule (b), D-005 literally: reset at each evaluation block and spend the burn-in inside the gap.
- It removes no leak that (a) has.
- It changes what is measured: a cold state plus M bars instead of the served state, so test numbers describe a model that is only ε-close to the served one.
- Gap = max(H) + W + M.

Rule (c1), (b) only at the cal|test boundary:
- It gives the literal D-005 guarantee on the reported block, at one M per fold.
- It breaks serving parity on exactly that block, with no leakage benefit.

Rule (a+) = (a) with the gap kept at 80:
- It keeps D-005 exact for every finite 60-bar window.
- It keeps anchors identical, so A/B arms pair exactly and the golden run is untouched.
- Cost: 0.

**Cost per fold** (Measured, calculation). Layout: 7-day train plus 1-day val, cal and test (their lengths are not yet set in NT-041); 3 boundaries; single EWMA after the shift:

| Rule | p = 60 | p = 240 | p = 1,440 | p = 10,080 |
|---|---|---|---|---|
| (a), gap 40 | 120 bars, 1.2% of the 7-day block | same | same | same |
| (a+), gap 80 (today) | 240 bars, 2.4% | same | same | same |
| (b) | 1,260, 12.5% | 4,335, 43% | 24,834, 246% | 172,437, 1,711% |
| (b), cascade M | 15.9% | 56.7% | 329% | 2,286% |
| (c1) | 5.8% | 15.9% | 84% | 572% |

- As a share of the whole fold span, (b) costs 8 / 23 / 63 / 92%.
- **All rules: data-start and gap mask** of M + 60 bars: 400 / 1,425 / 8,258 / 57,459 bars, that is 4 / 14 / 82 / 570% of a 7-day block. It applies only where history is missing.
- **All rules: compute.** The per-step series pass covers the batch span plus M. The median span is 11,130 anchors, so the pass is 11.5k / 12.5k / 19.3k / 68.5k bars.

**Long file** (Measured). 4,598,198 bars, 2017-01-01 to 2025-09-29.
- There is **one** missing-minute gap: 1,160 minutes, in 2025. Resetting there masks 0.02–2.5% of bars, depending on the period.
- 3.2% of bars are flat zero-volume: 118,924 runs, of which 90 last ≥10 minutes, 15 last ≥60 minutes, and the longest lasts 287 minutes (2020-04-25). These are forward-filled outages, invisible to NT-041 (8)'s missing-bar check.

**Recommendation: adopt (a+).**
- Gap = max(2·max(H), W + max(H)).
- Reset plus a mask of M(1e-3) + W at the data start, at missing-bar gaps, and at flat zero-volume runs ≥30 minutes. M comes from the current learned periods after the maximal shift. Training, evaluation and the Predictor do this identically.
- Judgement folds come after every choice fold, or a recorded separation larger than the choice runs' read ranges.
- Normalisers are fit on the training block, or are trailing.

**The test that pins it**, as pytest pseudo-code (for example tests/test_purge_rule.py):
```python
CASES = [(60,[10,15,20]), (30,[5,10,40]), (0,[10,15,20])]   # W = finite window; 0 = per-bar model
@pytest.mark.parametrize("W,H", CASES)
def test_gap_label_disjoint_with_embargo_and_finite_windows_clean(W, H):
    Hm = max(H)
    for f in make_purged_splits(20_000, lookback=W, horizon_steps=H, n_folds=5):   # or NT-041's builder
        assert f.gap == max(2*Hm, W + Hm)
        for a, b in itertools.combinations([f.train, f.val, f.cal, f.test], 2):     # every pair
            last_label_a = a[-1] + Hm - 1
            assert b[0] - last_label_a > Hm          # disjoint labels plus a max(H) embargo
            assert b[0] - W > last_label_a           # D-005 for finite windows
def test_config_refuses_gap_below_rule(): ...
# Series mode (INDICATOR_MEMORY item), CPU, bitwise:
def test_training_blind_to_bars_after_last_training_label(): ...   # fixed epochs, early stopping off
def test_served_epoch_blind_to_bars_after_val_labels(): ...
def test_calibrators_blind_to_bars_after_cal_labels(): ...          # temperature, conformal, beta, thresholds
def test_burn_in_mask_identical_train_eval_serve(file_with_hole_and_flat_run): ...
def test_judgement_folds_after_every_choice_fold(scenario_spec): ...
```

## Q2 Measured noise

**Source:** the v1 grid, 84 runs: 14 conditions × 3 seeds × 2 folds. P1 = fold −2 (dev); P2 = fold −1 (test, sizing only).
- **Caveats:** commit 6dec27a; last-epoch weights (before D-011); lambdas frozen.
- **Model per fold:** y = μ_c + β_s + e.
  - s_e = seed noise, 26 df per fold.
  - s_β = seed effect shared across conditions, 2 df per fold.
  - s_cp = condition×fold interaction minus seed noise, 13 df.
- **Pair SDs:**
  - shared-seed = √(2 s_e²);
  - nominal = √(2(s_e² + s_β²));
  - adding the folds' interaction adds 2 s_cp².

| Metric | s_e | s_β | s_cp | Pair SD, shared | nominal | nominal + folds | Direct, fold −2 / −1 |
|---|---|---|---|---|---|---|---|
| log CRPS, horizon mean | .0050 | .0036 | .0023 | .0070 | .0087 | .0093 | .0075/.0065 |
| log CRPS h0 | .0044 | .0036 | .0004 | .0063 | .0081 | .0081 | .0073/.0050 |
| log CRPS h1 | .0056 | .0043 | .0025 | .0079 | .0100 | .0106 | .0090/.0065 |
| log CRPS h2 | .0061 | .0029 | .0033 | .0087 | .0096 | .0106 | .0075/.0097 |
| CRPSS h0/h1/h2 | .0043/.0054/.0060 | | | .0061/.0077/.0085 | .0078/.0097/.0094 | .0078/.0103/.0104 | |
| AUC h0/h1/h2 (= AUC − logreg_lags) | .011/.013/.014 | .013/.012/.021 | .0045/.0047/.0008 | .016/.018/.020 | .024/.025/.036 | .025/.026/.036 | .012–.021 |
| \|cov90 − 0.90\|, mean | .0014 | .0007 | 0 | .0020 | .0023 | .0023 | .0017/.0023 |
| \|cov90 − 0.90\| h0/h1/h2 | | | 0 | .0023/.0027/.0030 | .0025/.0030/.0031 | same | |
| Net Sharpe (annualised) | 13.5 | 5.4 | 6.2 | 19.1 | 20.6 | 22.4 | 17.4/20.6 |
| n_trades | 34 | 13 | 20 | 48 | 52 | 59 | |
| ln median epoch_s (epochs ≥1) | .153 | .028 | .101 | .216 | .220 | .262 | |

- **AUC − logreg_lags** has the same paired noise as AUC, because logreg_lags is constant within a fold.
- **Timing is dominated by contention.** The robust within-cell SD (1.4826·MAD) is **0.0021**. 20 of 84 runs sit more than 2% off their fold median, up to 6.7×. One run is 1.9× the fold median with no epoch flagged by the within-run 1.5× rule.

**Block-length scaling** (Measured, q2_block_scaling.py).
- **Method:** 12 dev-fold runs (4 conditions × 3 seeds) were re-predicted on the CPU from their checkpoint weights, with the calibration pipeline refitted as in train_and_evaluate.
- **Validation:** two runs whose evaluated weights were the checkpoint reproduce eval_report_test.json exactly (h1 CRPS 136.952 = 136.952; 134.473 = 134.473).
- **Result:** SD of the paired ln-CRPS ratio over null pairs (same condition, different seeds, same sub-block):

| Block | h1 | Horizon mean |
|---|---|---|
| 5 d | .0082 | .0062 |
| 2 d | .0155 | .0123 |
| 1 d | .0234 | .0174 |
| 12 h | .0253 | .0189 |
| 6 h | .0292 | .0224 |

- The ratio is ×2.8 from 5 days to 1 day, against ×2.24 for pure averaging. There is no large length-independent component. The 5-day estimate rests on only 12 null pairs.
- **Within-run bootstrap understates.** The 80-bar block-bootstrap SD of the same 5-day difference is 0.0042–0.0044, only 0.54–0.75 of the between-seed SD. Within-run SEs (a DM test per pair, or pooling anchors) understate the uncertainty.

**Arm A's edge over constant variance** (ln ratio; positive = better).
- **Dev fold −2, v1, 42 runs, last-epoch weights:** h0 .0402, h1 .0378 (CRPSS .0370), h2 .0369, mean .0383. All 42 runs beat const_var at every horizon.
- **Dev fold −2, 12 runs re-predicted with checkpoint weights:** h0 .0426, h1 .0402, h2 .0383, mean .0404. The today-like condition (without:LAMBDA_VAC) gives mean .0418 and h1 .0416.
- **Test fold −1, sizing only:**
  - v1: h1 .0138, mean .0143. 38 of 42 runs beat const_var at every horizon; the h2 minimum is −.006.
  - Latest run: h1 .0169 (CRPSS .0167), mean .0195.
- **The edge is fold-dependent, ×2.5–2.9.** Margins must therefore be anchored on the study's own dev folds.
- **Direction:** dev h1 AUC .489 against logreg_lags .492, both at chance. Test fold: .478–.504 against .523.
- **Coverage90:** dev .917–.930 in every run; test .900–.917.
- **Net Sharpe:** dev −128, every run negative; test −121.

**Timing and convergence** (Measured).
- **Latest two notebook runs:**
  - median epoch time 11.74 / 11.87 s;
  - within-run robust SD of ln epoch time .003 / .006;
  - epoch 0 takes 25 s;
  - 0.9–1.5 s between epochs.
- **Notebook runs, convergence:** best val_loss at epoch 18–19 of 20, so the cap binds. The smoothed val-CRPS minimum comes at epochs 10–19.
- **v1 dev fold, 42 curves at batch 256.** J = val_crps_loss, and J~ = its 3-epoch running mean.
  - In 41% of the 20-epoch runs the minimum falls in the last 3 epochs: the cap censors.
  - The epoch that reaches 5% of the improvement span has median 13 and p90 20.
  - The SD of ln(reach epoch) is **0.40**.
  - Late epoch-to-epoch |ΔJ| is about 19% of the span.

## Q3 Draft specifications

**Measured support**
- **Grouping** (q3_val_grouping_check.py): the same weights on the same val block (2,866 anchors), evaluated with batch 256, 1024 and the whole block.
  - Val CRPS and NLL for h0–h2 differ by less than 2e-7 relative.
  - val_loss differs by −8.2% at 1024 and −10.6% for the whole block: soft-ECE −50%, T-perp −58%, HD −23%, vol +3%.
  - Direction BCE differs by +0.04%.
  - So J = val_crps_loss is lambda-free and grouping-free with **no code change**: logged metrics are weighted by batch size (custom_model.py:244, :289). val_loss is neither.
- **Probe error rates** (q3_probe_classification.py): ln(seed-mean reach ratio) ~ N(ln ρ, 0.40² × 2/S).
  - With 4 seeds per arm: P(epoch-bound | ρ = 1) = .92, P(update-bound | ρ = 4) = .85, cross-misclassification under 0.1%.
  - With 3 seeds: .89 and .81.
- **Comparator simulation** (q4_comparator_sim.py): margin .0065, pair SD .0087, s_cp .0023 (×1/×2/×4).

| Design | Naive size | Clustered size | Power at Δ = 0, naive / clustered (×1; ×2) |
|---|---|---|---|
| 1 fold × 6 seeds | .096/.196/.313 | – | .48 |
| 3 × 5 | .090/.148/.204 | ≈.05 | .79/.42; .66/.26 |
| 6 × 3 | .068/.115/.151 | ≈.05 | .87/.74; .76/.53 |
| 10 × 2 | .065/.073/.111 | ≈.05 | .90/.86; .80/.69 |
| 16 × 1 | .044/.052/.053 | = naive | .85; .74 |
| 20 × 1 | .050/.053/.050 | = naive | .92; .83 |

- A pooled-anchor test (anchors treated as i.i.d.) passes a null at the margin **47%** of the time.
- **Speed** (v1 contention: 24% of runs contended; true ratio 1; margin ln 1.05). Share of false failures:

| Statistic | 20 pairs | 12 pairs |
|---|---|---|
| mean/t | .91 | .91 |
| mean/t with re-timing | .68 | .63 |
| Hodges-Lehmann/Wilcoxon (HL) | .58 | .71 |
| HL with re-timing | **.027** | .146 |

- **Intersection-union:** P(ADOPT | truly equivalent) = p^k. With k = 4 that is .41 at p = .8 and .81 at p = .95. Keep the judged criteria few.

### (0) PROBE window_free_probe_v0 (a measurement, not a verdict)
- **Question:** is today's default epoch-bound or update-bound? The answer gates stage 5.
- **Arms:**
  - A256: the defaults.
  - B1024: BATCH_SIZE 1024, with LR 2e-3 or 4e-3 picked on the LR fold (1 seed each; lower min J~ wins).
- **Lambdas:** one calibration at 256 on the judging fold's train block (seed 0), frozen, and reused in every run via calibrate=False plus overrides (the runner must expose this).
- **VAL_BATCH_SIZE = 256 in both arms is recommended** (an implementer key). Without it, B1024's ReduceLROnPlateau reacts to a val_loss that reads 8% lower for the same weights, and the REPORT must record that recipe difference.
- **Judging:** J = val_crps_loss, J~ = its centred 3-epoch mean.
  - Target from A256's 4 seeds: T = mean(min J~), span = mean(J~(1) − min J~), reach level R = T + 0.05 × span.
  - E = the first epoch with J~ ≤ R.
  - W = fit wall-clock to the end of that epoch (metrics.jsonl `time`).
- **Caps:**
  - A256: EPOCHS 40, early stopping off. If J~'s minimum lands in the last 3 epochs for 2 or more of the 4 seeds, re-run once at 80.
  - B1024: EPOCHS = ceil(3 × mean E_A) + 3.
  - A run that never reaches R counts as E = ∞. If 2 or more runs fail to reach it, ρ = ∞.
- **Folds, dev only:** the LR pick on fold −3; judging on fold −2 of the bundled file; no test evaluation needed. Alternatively, two 7-day dev folds of the long history if NT-041 has landed. The SPEC names them.
- **Seeds:** 0–3 in each arm.
- **Rule:** ρ = mean E_B / mean E_A.
  - Epoch-bound iff ρ ≤ 1.5 and mean W_B < mean W_A.
  - Update-bound iff ρ ≥ 3.
  - Otherwise inconclusive. Update-bound or inconclusive parks stage 5 until the owner decides.
- **Run first:** the 0a kit's GPU timings, including the A2 window-gather forward+backward case under TF_DETERMINISTIC_OPS=1. If it errors, that is recorded and blocks the A2 design, not the probe.
- **Budget (Estimate):** A256 4 × ~7.5 min; B1024 4 × 4–5 min; LR pick 2 × 3–4 min; kit ~10 min. That is **~1.1 GPU-h**, or ~2.1 h if A256 needs its re-run. SPEC limit: 2.2 h.
- **Prerequisites:** R1 exited; a runner (NT-026 or a named one-off) that supports fixed lambdas, early stopping off, seeds and the GPU-free check; VAL_BATCH_SIZE (recommended); the kit.

### (1) A/B-1 series_memory_v1
- **Hypothesis (non-inferiority):** series memory is not worse in probabilistic skill by more than 1/3 of A's edge over const_var, and does not slow epochs by more than 5%.
- **Arms: one change, one pinned commit.**
  - A: INDICATOR_MEMORY = window, as today (ceiling 60, per-window shift).
  - B: INDICATOR_MEMORY = series, ceiling 60 unchanged. Its per-bar shift context at bar t is the trailing 60-bar mean and max of (close − close_t)/scale, which equals today's context exactly at the anchor bar. Network context 60.
  - Identical in both: batch 256, LR 1e-3, EARLY 6, PATIENCE 3, EPOCHS 60, and one frozen lambda set per study (calibrated on the first dev fold).
  - Same variable creation order, so that a given seed gives identical initial weights in every shared layer (a test). This makes the pairing shared-seed: h1 pair SD .0079 instead of .0100.
  - Raising the ceiling is a separate, later scenario.
- **Blocks (after NT-041):** 7-day train, 1-day val, 1-day cal, **5-day out-of-sample**. Gap 80 under (a+). Anchor hashes (y, last_close, anchor_bar, mask) are identical and asserted. The burn-in mask is the same in both arms.
- **Dev folds:** at least 3, arm A only, 1 seed, placed before every judgement fold. They measure Ē_dev (A's horizon-mean log edge) and the run time, and fix **δ = Ē_dev / 3** (planning value .0065).
- **Judgement folds:** F different months after the dev folds, 1 seed each.
  - F = max(10, n80(δ, σ_plan)), with σ_plan = .0113 (the v1 nominal .0087 × 1.25 for 7-day training, an Estimate, combined with s_cp).
  - That gives F = 21 at the planning δ, and F = 10 if Ē_dev ≈ .040.
- **P1 (quality):** per pair, d = the mean over horizons of Δ ln CRPS, served predictions, on the out-of-sample block.
  - Non-inferior iff the one-sided 95% upper bound < δ.
  - Breach iff the lower bound > δ.
  - An h1-only primary would need 35 folds at the planning δ, or 10 at Ē_dev.
- **P2 (speed, D-018):** per pair, r = ln(median epoch_s_B / median epoch_s_A) over epochs ≥1. Hodges-Lehmann estimate, one-sided 95% Wilcoxon upper bound ≤ ln 1.05.
  - Contention rules: the GPU-free check before each run, recorded; a run slower than 1.10× its arm's study median is re-timed once (both timings recorded, the re-time used); epochs above 1.5× the run's median are flagged; runs are ordered ABBA.
  - A P2 failure means the owner decides under D-018. It is not a REJECT.
- **G1 (coverage, paired):** c = the mean over horizons of (|cov_B − .9| − |cov_A − .9|).
  - Non-inferior iff the upper bound < 0.01; breach iff the lower bound > 0.01.
  - Pair SD is .0023, so power ≈ 1.
  - The per-run band [0.88, 0.92] is a diagnostic only: v1 fold −2 had 0 of 42 runs inside it.
- **Integrity:**
  - each arm's pooled one-sided lower bound of the log edge over const_var is above 0 at every horizon;
  - 0 non-finite steps; weights_epoch recorded; identical anchors;
  - more than 25% of an arm's runs capped means the verdict is at most INCONCLUSIVE.
- **AUC guard-rail dropped.** A has no dev-fold direction skill to protect (.489 against .492), and any margin would sit below chance. Report AUC − logreg_lags per horizon with paired intervals instead (pair SD .018–.025).
- **Net Sharpe is a secondary.** Every A run loses money, and with a pair SD of ~20, even 20 pairs only detect a loss of about 13 annualised units.
- **Verdict (intersection-union):**
  - ADOPT iff P1 and G1 are non-inferior, integrity holds and P2 passes.
  - If everything except P2 holds, the owner decides, with the slowdown measured.
  - REJECT iff P1 or G1 breaches, or B fails integrity.
  - Otherwise INCONCLUSIVE: one re-run with twice the folds (new folds), then the owner.
- **Joint power when the arms are equivalent (Estimate):** ≈ .8 × 1.0 × .97 = .77 at 20 pairs; ≈ .68 at 12 pairs (P2 false-fail .146 at v1's contention rate).
- **Secondaries:**
  - per-horizon CRPS, CRPSS, NLL, PIT KS, AUC and coverage; net Sharpe; n_trades;
  - share of learned periods at the 60 ceiling per arm, and whether series slow periods still drift there;
  - correlation of A's capped MACD-slow line with the 59-bar return (the Challenge's item);
  - applied-period ranges (5–95% per bar for B, per window for A);
  - sec_per_step, peak memory, epochs to the served checkpoint;
  - B's memory-reset sensitivity: re-score with the state reset at the block start, burn-in anchors excluded.
- **GPU budget (Estimate):**
  - An epoch on a 7-day block ≈ 40 steps × .095 s + val = 4.1 s, plus 1.5 s between epochs.
  - 21–60 epochs plus ~40 s fixed ⇒ **2.6–6.3 min per run**.
  - Runs = 2F + 3 dev + 1 calibration: F = 10 ⇒ 1.0–2.5 GPU-h; F = 21 ⇒ 2.0–4.8 GPU-h.
  - **Rule:** after the dev runs, recompute the budget from their measured time. If 2F × measured > 3 GPU-h, the SPEC goes to the owner before any judged run.
- **Must exist first:**
  - NT-026 with the options listed under the proposed items (lambdas once per study, fold roles, contention records);
  - NT-032 closing the gaps below;
  - NT-041: 7-day folds, 5-day out-of-sample blocks, fold roles and dates, the fingerprint, a gap policy that includes flat runs;
  - the series engine and INDICATOR_MEMORY, with the purge-rule, same-init and end-to-end causality tests and state-relative tolerances;
  - the kit's GPU gather case, run: no segment-sum backward.

### (2) A/B-2 window_free_v1
- **Hypothesis:** the per-bar model B reaches the served quality of A (the A/B-1 winner) at least 2× faster in fit wall-clock (the owner may choose 1.5× or 3×), without breaching the skill margins.
- **Arms:** A and B each pinned. At most 3 B variants of (T, chunks per step, LR), chosen by min J~ on 2 dev folds × 1 seed and recorded before judging.
- **Quality level per judgement fold:** Q_f = min J~_A + 0.05 × (J~_A(1) − min J~_A).
  - J is computed at identical val anchors; B's state is warmed through the history before them, and anchor identity is asserted.
  - J is grouping- and lambda-free, so no VAL_BATCH_SIZE or lambda question arises.
- **Primary (speed):** ℓ = ln(t_A / t_B) per pair, where t = fit wall-clock (tracing included, evaluation excluded) to the end of the first epoch with J~ ≤ Q_f.
  - If B converges (early stopping fires) without reaching Q_f, t_B = ∞.
  - Any run capped without early stopping makes its pair INCONCLUSIVE (excluded and counted).
  - Statistic: the HL median of ℓ with a one-sided Wilcoxon lower bound ≥ ln 2, and at most 1/4 of pairs INCONCLUSIVE.
  - The measured reach noise gives a pair SD of ℓ ≈ .57 (Estimate). At 20 pairs a true 3.5× passes with probability ≈ 1, and a true 2.5× with probability ≈ .55.
- **Caps:** EPOCHS 80 for A; the same number of passes over the training anchors for B; EARLY 6 in both.
- **Guard-rails:**
  - G1: horizon-mean log CRPS with δ = Ē_dev/3, tested like P1 of A/B-1.
  - G2: coverage, as in A/B-1.
  - AUC dropped.
- **Integrity:**
  - each arm beats const_var at every horizon (pooled);
  - 0 non-finite steps;
  - B passes T1–T6 and assert_no_lookahead;
  - anchor identity holds;
  - capped runs are counted.
- **Pairs:** F = 20 judgement folds × 1 seed. G1 power is .92 at the measured interaction and .83 at twice it (simulated).
- **Verdict:**
  - ADOPT iff speed passes, G1 and G2 are non-inferior, and integrity holds.
  - REJECT iff the speed upper bound < ln 2, G1 or G2 breaches, or B fails integrity.
  - Otherwise INCONCLUSIVE: one re-run with more folds, then the owner.
- **Diagnostics:**
  - loss by chunk position (flat after W);
  - τ of B's per-position loss;
  - optimizer steps and supervised anchors to the served checkpoint;
  - seconds per supervised anchor (the cross-design D-018 metric);
  - t_A, t_B;
  - period ranges.
  - The A′ arm runs only after an ADOPT (+0.6 GPU-h).
- **GPU budget (Estimate):**
  - A 2.6–5.9 min per run; B 1.3–2.5 min.
  - 20 pairs plus the variant choice (~10 min) plus dev A runs (~12 min) ≈ **1.7–3.0 GPU-h**.
  - Same budget rule as A/B-1.
- **Must exist first:**
  - the A/B-1 verdict;
  - the probe says epoch-bound (or the owner overrides);
  - NT-041 and NT-042;
  - NT-032 with the gaps below, including robust bounds, ∞ values and INCONCLUSIVE pairs;
  - B's Models entry with T1–T6, and B's val metric computed at identical anchors.

## Q4 Comparator gaps

NT-032's acceptance today covers:
- (1) pairing by (seed, fold), refusal on differing blocks or fingerprints, the mean difference with a 95% interval, a "beats / inconclusive" verdict, named judgement folds, and at least 5 pairs;
- (2) the spec hash;
- (3) guard-rail breach = an interval beyond the allowed degradation;
- (4) a null simulation with 80-bar block noise and seed noise;
- (5) JSON and markdown output.

Gaps:
1. **Non-inferiority outcomes per criterion:** pass, breach, undecided, with the direction (higher or lower is better).
2. **One-sided 95% bounds,** with alpha per criterion set in the spec.
3. **Log-ratio metrics** (ln CRPS, ln t_A/t_B, ln epoch ratio), margins stated on that scale, and a back-transformed report.
4. **Derived metrics computed identically for both arms** from the run store, including metrics.jsonl: |cov − .9|, AUC − logreg_lags, the log edge over const_var, horizon means, time-to-quality (J~, Q_f).
5. **Margins as a fraction of A's dev-fold edge,** measured before the judged runs. Record the pre-measurement run ids, and refuse them if they sit on judgement folds.
6. **An intersection-union verdict table** (ADOPT, REJECT, INCONCLUSIVE, and the owner route for D-018), with the joint power reported.
7. **The arm×fold interaction:** a clustered test on fold means (F−1 df), or 1 seed per fold, plus a naive-versus-clustered agreement check. Put the fold term into the simulation: simulated naive size is 9–31% with 1–3 folds.
8. **Robust timing and time-to-quality:** HL with Wilcoxon bounds, ∞ values, INCONCLUSIVE pairs, and contention and re-time metadata. Simulated mean/t false-fail is .63–.91.
9. **Per-block anchor hashes,** burn-in masks and val anchors included.
10. **Fold placement:** refuse a judgement fold that precedes a choice fold, or that falls inside a choice run's recorded read range.
11. **Pre-registered pair count;** refuse a partial set (no peeking).
12. **A simulation calibrated to the measured components** (s_e, s_β, s_cp, and within-run dependence beyond 80 bars, since the bootstrap SD is only .54–.75 of the between-seed SD). Report the size at the NI boundary, the power at 0 and the joint ADOPT probability. Refuse pooled-anchor tests (size .47).
13. **Seed pairing marked** shared (init equality tested) or nominal in the spec, with the matching variance used in the power report.

## Proposed backlog items and amendments (acceptance checkable at QA)
1. **New P1 item: purge rule for unbounded memory, split test and config guard** (implementer, with or before INDICATOR_MEMORY).
   - tests/test_purge_rule.py as in Q1, passing for W ∈ {60, 30, 0} and two horizon sets.
   - The reference gap stays 80, and golden_run verify passes.
   - Config.validate refuses a gap below the rule, naming the field (test).
   - The formula is documented in splits.py.
2. **INDICATOR_MEMORY amendment.**
   - The three CPU bitwise blindness tests of Q1.
   - An identical burn-in mask (M(1e-3) + W from the current periods after the maximal shift) in train, eval and the Predictor, on a synthetic file with a missing-minute gap and a 90-minute flat run; counts recorded in meta.json.
   - A same-init test.
   - An anchor-identity test.
   - A context-mirror test: B's shift context equals A's at the anchor bar, within 1e-6.
3. **NT-041 amendment.**
   - A configurable out-of-sample length, with a documented A/B default of 5 days.
   - The gap policy flags flat zero-volume runs of 30 minutes or more (test).
   - Fold roles (dev or judgement), with judgement folds after dev folds; a spec that violates this is refused (test).
   - meta.json records the read range.
4. **NT-026 amendment.**
   - A `lambda_calibration: once` scenario option (test: identical LAMBDA_* values and calibrate=False in both arms).
   - Early stopping off, or EPOCHS set per arm.
   - A per-run contention record.
5. **NT-032 amendment.** Gaps 1–13 above as acceptance criteria, each with a test. The simulation must reproduce this report's sizes within Monte Carlo error: naive size > .07 at 1 fold × 6 seeds, and pooled-anchor size > .4.
6. **New P2 item: VAL_BATCH_SIZE.**
   - The default equals BATCH_SIZE, so golden_run passes.
   - At batch 1024 with VAL_BATCH_SIZE 256, val_loss equals the batch-256 value within 1e-6 relative for the same weights (test).
7. **Experimenter items for the probe, A/B-1 and A/B-2.**
   - The SPEC equals the draft above, or is stricter.
   - The dev pre-measurement and the recomputed budget are recorded before any judged run; above 3 GPU-h goes to the owner.
   - The REPORT carries the verdict table with every bound, the joint power, the contention log and the run ids.
   - No choice uses a test-block number.

## Risks
- **Noise may not transfer.** It was measured on a small change (physics weights) with 23–30k training anchors. 7-day blocks and a bigger change may be noisier. The ×1.25 inflation is an assumption, and the budget rule catches an overrun only after the dev runs.
- **Out-of-sample block length.** 1-day blocks would need about 8× the pairs.
- **Fold-dependent edge (×2.5–2.9).** A fixed δ is strict on low-edge folds and loose on high-edge ones. The alternative is a retention criterion E_B ≥ ⅔ E_A per pair, whose variance adds the fold spread of E_A/3.
- **Joint power decays as p^k.** Each judged criterion needs power ≥ .95.
- **Contention.** Even with re-timing, 12-pair speed tests false-fail 15% of the time at v1's contention rate. The GPU-free check should lower that, but it is unmeasured.
- **Probe noise.** Reach epochs have log SD .40, and a 20-epoch cap censors 41% of runs. Hence the 40-epoch cap and 4 seeds. A true ρ ≈ 2 is inconclusive ~77% of the time, by design.
- **7-day blocks give ~40 updates per epoch** against 119 today, so EPOCHS 20 may under-train. Hence 60, which sets the budget's upper end.
- **(a+) holds only if every input path is causal.** The end-to-end causality test and train-fit or trailing normalisers are essential.
- **Unlimited periods reach far back.** One learned period of 10,080 reads ~40 days back, so fold placement must be enforced, not assumed.
- **Per-step pass length grows with M** (68.5k bars at p = 10,080). Caching a warm state behind a stop-gradient would bias long periods; the plan must choose.
- **Forward-filled outages are invisible** to a missing-bar check.
- **Epoch-end costs.** The ~1.5 s between epochs is ~27% of a 7-day-block epoch (Estimate), so NT-054 matters for every budget above.