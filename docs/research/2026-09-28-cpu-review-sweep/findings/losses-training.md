# Area: losses-training (14 findings)

Back to the [index](README.md) and the [report](../README.md).

<a id="loss-1"></a>

## LOSS-1: Layer regularisers enter the objective at 0.1 x LAMBDA_INTER: every configured L2 (DIRECTION_SKIP_L2, REG_MOMENTUM_L2, INDICATOR_L2) is 10x weaker than documented, and direction_v1 reports the wrong L2 levels

- **Severity:** P3 (reported P2). **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-27.
- **Files:** `src/neural_trade/losses/functions.py`, `src/neural_trade/core/config.py`, `configs/default.yaml`, `src/neural_trade/models/gru_attention.py`, `src/neural_trade/models/layers/learnable_indicators.py`, `runs/experiments/direction_v1/REPORT.md`

**Description.** custom_loss computes reg_loss = add_n(model.losses) and inter_reg = LAMBDA_INTER * reg_loss (functions.py:690-691), then adds `0 * reg_loss + 0.1 * inter_reg` to total (functions.py:776-777; the comment calls it 'Indicator correlation'). Config documents LAMBDA_INTER as 'weight of model.losses (layer regularisers)' = 1.0 (config.py:118, default.yaml:52). model.losses holds every Keras L2: the direction skip (gru_attention.py:38, DIRECTION_SKIP_L2 'L2 on the direction skip weights', config.py:171), the towers (gru_attention.py:161,185,216,237, REG_MOMENTUM_L2) and the indicator logits (learnable_indicators.py:57-99, INDICATOR_L2). So each configured coefficient acts at one tenth of its value. The factor dates from Jan 2026 (git log -S shows 11a611b); direction_v1 (Sep 2026) chose 'DIRECTION_SKIP_L2 = 1e-4' and reported a tower L2 grid of 1e-3 and 1e-2 (REPORT.md:66-69, 99-105). The effective values were 1e-5 and 1e-4 / 1e-3. Lambda calibration does not touch this term.

**Failure scenario.** Config DIRECTION_SKIP_L2=1e-2, REG_MOMENTUM_L2=1e-2 (as in the direction_v1 grid): the objective penalises sum(model.losses) with weight 0.1, i.e. an effective L2 of 1e-3. The reported 'L2 1e-2 hurts EV' conclusion is about 1e-3. A reader who sets 1e-4 believing the report gets 1e-5.

**Evidence (finder, reproduced).** Ran agents/losses-training/probes/p5_reg_weight.py (real gru_attention model, B=16): 'model.losses entries: 25  sum = 1.423256'; 'LAMBDA_INTER=1: total=17.963539 reg_loss=1.423256 inter_reg=1.423256'; 'LAMBDA_INTER=0: total=17.821215'; 'share of sum(model.losses) that enters total: 0.1000 (Config doc: LAMBDA_INTER = weight of model.losses = 1.0)'.

**Verifier (reproduced): confirmed.** Code: functions.py:690-691 (reg_loss = add_n(model.losses); inter_reg = LAMBDA_INTER*reg_loss), :776-777 (`0 * reg_loss + 0.1 * inter_reg`, comment 'Indicator correlation'); config.py:118 / default.yaml:52 document LAMBDA_INTER as 'weight of model.losses'; model.losses come only from Keras L2 regularisers (gru_attention.py:38,161,185,216,237; learnable_indicators.py:57-99). Ran agents/verify-losses-training/probes/v1_reg.py (tiny 10-Dense base with L2(0.1), B=32): 'sum(model.losses) 5.531076 = 1e-1*sum w^2 5.531076'; 'total(LI=1) - total(LI=0) = 0.5527  share: 0.0999'. So every configured L2 acts at 0.1x. Lambda calibration does not touch it (not in _CALIB_LAMBDA_NAMES).

**Verifier corrections.** Severity P3, not P2: no computed number is wrong; the knob values in direction_v1 REPORT.md:66-69, 99-105 are the real Config values, only their meaning relative to Keras L2 is 10x off. Do NOT take the primary fix (divide the defaults by 10, or move the 0.1 into LAMBDA_INTER): every stored config (gates m1a..m6 meta.json, direction_v1, the 84 ablation v1 cells, the latest run) records DIRECTION_SKIP_L2=1e-4 and LAMBDA_INTER=1.0, so a semantics change makes an engine replay of any old config run a regulariser 10x off while the golden run (defaults) still passes. Fix = the finding's alternative: state the effective coefficient in the docs (LAMBDA_INTER: 'the objective adds 0.1 x LAMBDA_INTER x sum(model.losses)'; DIRECTION_SKIP_L2, REG_MOMENTUM_L2, INDICATOR_L2: 'Keras L2 coefficient; effective 0.1 x LAMBDA_INTER x value'), fix the misleading comment at functions.py:777, and add a one-line note to direction_v1 REPORT.md. Acceptance: (1) test pins total(LAMBDA_INTER=1) - total(LAMBDA_INTER=0) == 0.1 x sum(model.losses) within 1e-3 relative on a tiny regularised model; (2) the four descriptions state the effective coefficient (NT-029's generated config reference shows it); (3) direction_v1 REPORT.md carries the note; (4) golden run unchanged. MVP growth point 1, during NT-029 (config metadata/docs). cpu_cost instant.

**Backlog check (finder).** Not in NT-001..NT-025. NT-012 mentions that the dashboard 'repeats the 0.1 factors of custom_loss by hand' but not that the regulariser weight contradicts the documented L2 coefficients, and it does not mention the mislabelled direction_v1 levels. REMEDIATION_PLAN S14 fixed only the coherence double count.

**Fix sketch.** Behaviour-preserving: replace `0 * reg_loss + 0.1 * inter_reg` with `inter_reg`, and divide the three L2 defaults by 10 (DIRECTION_SKIP_L2 1e-5; REG_MOMENTUM_L2 and INDICATOR_L2 stay 0). The objective then equals today's to float precision, and every documented coefficient is the real one. Fix the comment. Add a correction note in runs/experiments/direction_v1/REPORT.md (skip L2 1e-4 -> effective 1e-5; tower grid 1e-3/1e-2 -> 1e-4/1e-3). Alternative: keep the code and change the Config/yaml descriptions to say 'effective weight 0.1 x LAMBDA_INTER'.

**Acceptance (proposed).** (1) New test: on a real gru_attention model, total(LAMBDA_INTER=1) - total(LAMBDA_INTER=0) == sum(model.losses) within 1e-6 relative. (2) On a fixed batch and seed, the default objective value before and after the change agrees within 1e-5 relative (the effective regularisation is unchanged). (3) The Config/yaml descriptions of LAMBDA_INTER, DIRECTION_SKIP_L2, REG_MOMENTUM_L2 and INDICATOR_L2 state the coefficient that acts. (4) direction_v1 REPORT.md carries the correction. (5) tests/test_custom_loss.py and tests/test_losses*.py pass.

<a id="loss-2"></a>

## LOSS-2: val_loss (the EarlyStopping / checkpoint / LR / served-epoch criterion) is about 30% batch-statistic terms evaluated on 256 consecutive, label-overlapping validation windows: they mostly measure within-batch regime autocorrelation, not model quality

- **Severity:** P2. **Status:** confirmed. **Type:** design-risk. **CPU cost:** instant. **Placement:** addition to NT-004.
- **Files:** `src/neural_trade/data/datasets.py`, `src/neural_trade/losses/functions.py`, `src/neural_trade/training/custom_model.py`, `src/neural_trade/training/callbacks.py`, `src/neural_trade/training/trainer.py`

**Description.** Several objective terms are batch statistics: soft_ece_loss (functions.py:229-259, per-batch soft bins), t_perp_calibration_loss (263-286, log batch-mean var vs log batch-mean residual^2), hyper_decoherence_coupling_loss (371-400, batch Pearson), information_flow_entropy_loss (405-444, batch correlation) and vol_loss (693-700, batch std). val_ds is built with shuffle=False (datasets.py:27-28), so every validation batch is 256 consecutive 1-minute windows whose 10-20-bar labels overlap. Training batches are quasi-shuffled (see LOSS-12). The same terms therefore measure different things on train and on val, and on val they carry a large floor that depends on local regime clustering rather than calibration. test_step feeds this val_loss (custom_model.py:555-580) to EarlyStopping, ModelCheckpoint and ReduceLROnPlateau (callbacks.py:342, 347, 359) and to the served epoch (trainer.py:162-198, D-011).

**Failure scenario.** Take a perfectly calibrated forecaster (p = the block's up rate; sigma^2 = the block's mean squared scaled move) on a validation-sized block of the real BTC data. Per-batch t_perp is 0.55-0.71 per horizon on consecutive batches vs 0.015-0.020 on random batches. Per-batch soft-ECE is 0.077-0.105 vs 0.035. m6's logged val_t_perp_loss at the end (0.2085 = 0.1 x sum over 3 horizons, i.e. 0.70 per horizon) sits exactly at this artefact floor. Its val soft_ece_loss is 0.711 vs 0.244 on train. At m6's best epoch these terms contribute about 1.8 of val_loss 5.52 (soft-ECE 2.02 x ~0.72, t_perp ~0.22, hd ~0.07, vol ~0.08). The served epoch is thus partly chosen by how well variance and probability levels track 4-hour regime runs inside a batch, which the training batches do not reward in the same way.

**Evidence (finder, reproduced).** Ran probes/p12_val_batch_ece.py: 'h=10: soft-ECE consecutive 0.077, random 0.035 | t_perp consecutive 0.554, random 0.015'; 'h=15: 0.090 / 0.035 | 0.581 / 0.020'; 'h=20: 0.105 / 0.034 | 0.712 / 0.016'. From runs/gates/m6/training_log.csv (epoch 19): val_soft_ece_loss 0.7112 vs soft_ece_loss 0.2441; val_t_perp_loss 0.2085 vs t_perp_loss 0.0298; val_hd_loss 0.0716 vs hd_loss 0.0267; calibrated lambda_soft_ece 2.02 (meta.json).

**Verifier (reproduced): confirmed.** Code: datasets.py:27-28 (val_ds shuffle=False, 256 consecutive windows); batch-statistic terms at functions.py:229-259 (soft-ECE), 263-286 (t_perp), 371-400 (HD), 405-444 (IFE), 693-700 (vol); callbacks.py:342/347/359 monitor val_loss; trainer.py:162-198 serves the best val_loss epoch. Re-ran probes/p12_val_batch_ece.py: identical output (h=10 soft-ECE 0.077 vs 0.035, t_perp 0.554 vs 0.015; h=15 0.090/0.035, 0.581/0.020; h=20 0.105/0.034, 0.712/0.016). New: decomposed runs/gates/m6/training_log.csv per epoch (batch-stat part = 2.02*val_soft_ece + val_t_perp + val_hd + 0.1*val_vol + val_ife): it falls 2.688 (ep0) -> 1.788 (ep19) while the rest is 3.939 -> min 3.647 at ep6 -> 3.884 at ep19; argmin(val_loss) = epoch 14 (the served one) but argmin(rest) = epoch 6. In m5 both argmins are epoch 16. At ep19 the batch-stat part is 1.79 of val_loss 5.67 (32%) vs about 0.64 of train loss 5.14 (12%).

**Verifier corrections.** Backlog check is wrong: NT-004 already questions the criterion ('Early stopping watches the total val loss, which keeps improving through the direction and variance terms') and lists 'early stopping / checkpoint on per-head val point loss' as a candidate variant; docs/research/2026-09-28-window-free/README.md:441-442 records val contiguity as deliberate (same meaning across window-free arms). Retitle 'Evidence for NT-004 and the engine scorer (NT-026): the served epoch is driven by batch-statistic terms evaluated on contiguous val batches' and add the m6 decomposition (served epoch 14 vs epoch 6 without those terms). The 'artefact floor' claim holds only for a constant (marginally calibrated) forecaster; an adaptive model can go below it, since on contiguous batches t_perp measures 4-hour conditional variance calibration plus label-overlap noise (n_eff about 256/h = 13-26 labels per batch). So 'not model quality' is overstated; 'a different and noisier quantity than on train batches' is what the evidence shows. Option (a) changes every val_loss (golden run changes; DECISIONS entry). Option (b) is a pre-registered criterion choice in the scorer. Acceptance (2) applies only to option (a). MVP growth point 2 (one scorer and served epoch feed the leaderboard), decided before NT-050's first real sweep and written into NT-004's SPEC. cpu_cost instant (code); judging the effect is GPU work.

**Backlog check (finder).** Not covered. NT-012 adds logging only. NT-003/NT-004 do not question the val criterion's composition. The REMEDIATION_PLAN made val_loss an epoch aggregate (not last-batch) but did not address batch composition.

**Fix sketch.** Give val the same batch composition as train. Option (a): val_ds = shuffle once with a fixed seed and reshuffle_each_iteration=False (deterministic, and epoch-level direction/PIT metrics are order-free). Option (b): monitor a criterion without the batch-statistic terms, e.g. a 'val_select_loss' = point + trend + dir + nll + crps, and point EarlyStopping / ModelCheckpoint / ReduceLROnPlateau / _serve_best_weights at it. Either is a one-file change. Its effect on served epochs is GPU work (NT-003/NT-004 variants).

**Acceptance (proposed).** (1) Test: val_ds yields the same order in two epochs and its batches are not contiguous (in-batch index span > 10 x BATCH_SIZE), or the monitored key excludes soft-ECE, t_perp, HD, IFE and vol (test on the callback monitors). (2) Test on synthetic autocorrelated data (AR regime blocks): a calibrated constant forecaster's val t_perp contribution is <= 2x its value on randomly drawn batches. (3) test_served_epoch.py and the training smoke tests pass. (4) RUNBOOK or DECISIONS states which quantity selects the served epoch.

<a id="loss-3"></a>

## LOSS-3: Hyper-decoherence term orders sigma by the std of the window's price LEVELS, not by realised volatility as documented (a weaker predictor of the future squared move: Spearman 0.31 vs 0.36)

- **Severity:** P2. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-19.
- **Files:** `src/neural_trade/losses/functions.py`, `src/neural_trade/calibration/conformal.py`

**Description.** hyper_decoherence_coupling_loss says 'predicted variance should be ORDERED like the window's realised volatility' (functions.py:372-384). But it computes local_vol = reduce_std(x_window, axis=1) (functions.py:388). x_window is the window_relative input (x - last_close)/scale, so this is the dispersion of price levels, i.e. trend or range, not the std of 1-bar changes. The project's own realised volatility (calibration/conformal.py:65-88, used by D-008's conformal scale) is np.diff(X).std(). A steady trend has large level-std and zero volatility; a choppy flat window has small level-std and large volatility. D-003 says to fix the physics maths before judging them. HD is the term closest to VALUE in the v1 ablation (leave_one_out corr_var_err2_spearman VALUE), so NT-006 would judge it with the wrong proxy.

**Failure scenario.** Two windows: a steady +2 $/bar trend with no noise (level-std 0.149 in scaled units, 1-bar-change std 0 $) and a flat window with 15 $ noise per bar (level-std 0.065, 1-bar-change std 21.1 $). HD pushes sigma higher for the calm trend than for the volatile chop. On all real 60-bar windows, the HD target predicts the future squared 10/15/20-bar move with Spearman 0.318/0.308/0.298, vs 0.364/0.348/0.342 for the realised-vol definition.

**Evidence (finder, reproduced).** Ran probes/p8_hd_vol.py on binance_btcusdt_1min_ccxt.csv: 'h=10 bars: Spearman(level-std, fut^2) = +0.318   Spearman(step-std, fut^2) = +0.364' (h=15: 0.308 vs 0.348; h=20: 0.298 vs 0.342); 'Spearman(level-std, step-std) = +0.814'; toy: 'steady trend +2$/bar level-std (HD input) = 0.149 1-bar-change std = 0.0 $', 'flat, 15$ noise per bar level-std = 0.065 1-bar-change std = 21.1 $'. A candidate fix (local_vol = reduce_std(x[:,1:]-x[:,:-1], axis=1)) in agents/losses-training/wt passes tests/test_physics_terms_bounded.py, test_qbox_losses.py, test_custom_loss.py, test_losses.py and test_losses_reference.py: '75 passed in 12.84s'.

**Verifier (reproduced): confirmed.** Code: functions.py:372-384 docstring 'ordered like the window's realised volatility'; :388 local_vol = reduce_std(x_window, axis=1), where x_window = (X - last_close)/target_scale (data/scaling.py:48-52), i.e. dispersion of price levels. The project's realised vol is np.diff(X).std() (calibration/conformal.py:84-88, D-008). Ran probes/v3_hd.py on binance_btcusdt_1min_ccxt.csv (43,420 windows): 'Spearman(level-std, step-std) = 0.814'; 'h=10: Spearman(level-std, fut^2) = +0.318  Spearman(step-std, fut^2) = +0.364'; h=15 +0.308 vs +0.348; h=20 +0.298 vs +0.342 (exactly the finding's numbers). Toy in $: 'trend +2/bar: level-std 34.64 $ step-std 0.00 $'; 'flat, 15$ noise: level-std 13.46 $ step-std 16.52 $'.

**Verifier corrections.** Partly known but not planned: docs/research/2026-09-28-window-free/README.md:248 records that HD reads 'std of the window positions as local volatility' and that the window-free design will turn it into a per-bar trailing-volatility input; no BACKLOG item says the statistic is level dispersion rather than volatility. The fix changes the objective for every run (golden run changes by design; record under D-003 'fix the maths' in DECISIONS). Numbers in the toy are in $ in my probe (34.64 / 13.46 level-std); the finding's scaled numbers have the same ordering. MVP growth point 4 (generality: NT-053 must re-express HD as trailing per-bar volatility, and the 1-bar-change std carries over unchanged); land before NT-006 (R4) and before NT-053's A/B specifications fix the loss. cpu_cost instant.

**Backlog check (finder).** Not covered. REMEDIATION_PLAN S10 rewrote HD as 1 - Pearson but kept reduce_std(x). NT-006 re-runs the ablation without touching the term's maths.

**Fix sketch.** local_vol = stop_gradient(reduce_std(x[:, 1:] - x[:, :-1], axis=1)) (scale-free under window_relative, like the conformal realized_vol), and update the docstring. Do this before NT-006's GPU grid.

**Acceptance (proposed).** (1) Unit test: for the steady-trend vs choppy-flat window pair above, the choppy window gets the larger local_vol, and a variance ordering that follows it gives the lower HD loss. (2) HD stays in [0, 2] and sends no gradient into local_vol (existing bound tests). (3) The docstring names the statistic. (4) The fast loss tests pass.

<a id="loss-4"></a>

## LOSS-4: vac_overflow term is unsatisfiable by construction and its gradient has the wrong sign (the term worsened monotonically 0.80 -> 0.985 over m6's 20 epochs)

- **Severity:** P1. **Status:** confirmed. **Type:** bug. **CPU cost:** short. **Placement:** new item CPU-07.
- **Files:** `src/neural_trade/losses/functions.py`, `src/neural_trade/models/gru_attention.py`, `src/neural_trade/models/layers/vacuum_saturation_noise.py`

**Description.** vacuum_overflow_t_perp_loss (functions.py:447-487) wants mean(overflow) == mean |y - p| (about 0.5 in scaled units). overflow = relu(mean_d h_sat^2 - E_max) (gru_attention.py:124-136), with h_perp = tanh(.) in (-1, 1) (gru_attention.py:110-111) and VACUUM_E_MAX = 1.0. VacuumSaturationNoise (vacuum_saturation_noise.py:34-55) adds noise whose std sqrt(E_max - batch energy) is stop_gradient-ed. Real signal can never exceed E_max (|h| < 1), so the overflow is noise-only. Its mean is at most about 0.14, at h = 0 (the chi-square fluctuation of 16 dims), and it falls toward 0 as |h| saturates. The target is therefore unreachable. Because the noise std is stop-gradient-ed, autodiff says 'increase |h|' at every scale, while the true derivative is positive for a >= 1: growing |h| shrinks the noise and the overflow. Gradient descent therefore drives the T-perp projection into tanh saturation. That projection conditions all three variance heads through t_perp_magnitude, and the term's own value gets worse. D-003 keeps the six terms but says to 'fix the maths'. NT-006 would spend its only:/without:VAC_OVERFLOW cells (12 of 84 GPU runs) judging this term.

**Failure scenario.** h = tanh(a z), B = 256, D = 16, residual magnitude 0.5. a=0.05: overflow 0.139, loss 0.521. a=1: overflow 0.135, loss 0.534. a=4: overflow 0.091, loss 0.670. The autodiff d loss/da is -0.60 at a=1 (descent increases a), but the finite difference with the same noise draws is +0.050. In m6 the logged train vac_overflow_loss (lambda 0.1) rose 0.080 -> 0.093 -> 0.097 -> 0.0985: the raw term went from 0.80 to 0.985, so the overflow fell to about 0.75% of the residual.

**Evidence (finder, reproduced).** Ran probes/p13_vac_overflow.py (real VacuumSaturationNoise layer and loss): rows 'a | mean overflow | loss | autodiff | finite diff': '0.05 | 0.1394 | 0.521 | -0.1751 | -0.0108', '0.30 | 0.1406 | 0.517 | -0.7494 | -0.0130', '1.00 | 0.1348 | 0.534 | -0.6010 | +0.0503', '2.00 | 0.1163 | 0.589 | -0.2522 | +0.0529', '4.00 | 0.0907 | 0.670 | -0.0767 | +0.0298'. runs/gates/m6/training_log.csv vac_overflow_loss at epochs 0/6/12/18/19: 0.08, 0.0932, 0.0967, 0.0985, 0.0985.

**Verifier (reproduced): confirmed.** Code: functions.py:447-487 (loss = (mean_ov - mean_res)^2/mean_res^2); gru_attention.py:110-111 (h_perp = tanh), :123-134 (overflow = relu(mean_d h_sat^2 - E_max)); vacuum_saturation_noise.py:43-55 (noise std sqrt(E_max - batch energy) under stop_gradient); config.py:138 VACUUM_E_MAX=1.0. Ran probes/v4_vac.py (real layer, the Lambda's formula, the real loss; B=256, D=16, residual 0.5; mean over 20 noise seeds): 'a=0.05: ov 0.1390 loss 0.522 autodiff -0.1770 FD -0.0079'; 'a=0.30: 0.1395 0.521 -0.7566 -0.0067'; 'a=1.00: 0.1344 0.535 -0.5959 +0.0469'; 'a=2.00: 0.1163 0.590 -0.2475 +0.0533'; 'a=4.00: 0.0902 0.672 -0.0742 +0.0307'. The loss never gets below about 0.52 (unreachable target), and for a >= 1 the autodiff and finite-difference signs disagree. m6 training_log vac_overflow_loss: 0.0800 (ep0), 0.0889 (3), 0.0932 (6), 0.0959 (9), 0.0967 (12), 0.0978 (15), 0.0985 (18, 19): the raw term goes to about 1, i.e. overflow about 0.

**Verifier corrections.** Add: the stop-gradient is intended by the layer ('the network learns to fill the vacuum with real signal', vacuum_saturation_noise.py:21-22), and filling the vacuum with real signal (|h| -> 1) is exactly what makes the overflow vanish. So the layer's intent and the loss's target contradict each other: a maths decision under D-003, possibly the owner's. At small a the autodiff and FD signs agree but the autodiff overstates by 20-100x. Not in BACKLOG or REMEDIATION_PLAN (S12 only added the residual stop_gradient). Severity P1 kept, because NT-006 (R4) cannot give a meaningful verdict on this term until it is fixed (12 of 84 v1 cells). MVP growth point 3 (gradient stability: 'autodiff sign equals finite-difference sign' is an invariant that NT-036/NT-038 should carry), before NT-006 and before NT-051's harness thresholds are fixed. Acceptance as proposed; the sign test must use the same noise seeds (tf.random.set_seed per evaluation, as in v4_vac.py). cpu_cost short.

**Backlog check (finder).** Not covered. REMEDIATION_PLAN S12 added stop_gradient on the residual and dropped the term from val. NT-006 plans a re-run of the grid with the term unchanged. No backlog item questions whether the term can be satisfied.

**Fix sketch.** This needs a maths decision before NT-006, and the owner may need to decide (D-003: keep the term, fix the maths). Options: (a) define the overflow in reachable units, e.g. E_max < 1, or let the overflow be the pre-tanh energy above E_max; (b) compare a normalised overflow with a normalised residual (both z-scored in the batch), which makes the term an ordering constraint like HD; (c) remove the stop_gradient on the noise std and accept that the optimum is then h -> 0. Whichever is chosen, the loss must be minimisable and its gradient must match its finite difference.

**Acceptance (proposed).** (1) Unit test with the real layer + Lambda + loss on h = tanh(a z): for a in {0.3, 1, 2, 4}, the sign of the autodiff d loss/da equals the sign of the finite difference taken with the same noise seeds. (2) Unit test: for a residual magnitude of 0.5 there exist inputs where the term is < 0.1 (the target is reachable). (3) The docstring states the reachable range. (4) The fast suite passes. (5) NT-006 lists this item as a dependency.

<a id="loss-5"></a>

## LOSS-5: Finite-gradient guard does not zero the update: Adam momentum still moves every weight on a guarded step (and the guard costs ~370 extra ops per step)

- **Severity:** P3. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-036, NT-054.
- **Files:** `src/neural_trade/training/custom_model.py`

**Description.** train_step replaces the gradients with zeros when any is non-finite and calls both apply_gradients anyway (custom_model.py:470-511); the comment says 'Zero the whole update instead, and count it'. With Adam, a zero gradient still applies lr * m_hat/(sqrt(v_hat)+eps) from the existing first moment, increments iterations and decays both moments (and AdamW would still decay weights). The guard also adds one IsFinite + All + SelectV2 per trainable variable (123 in gru_attention): about 370 small ops on the kernel-launch-bound GPU step (D-010, D-018). is_finite(grad_global_norm) is already computed and gives the same verdict with one op.

**Failure scenario.** 3 normal steps, then 1 step whose gradient is NaN: nonfinite_grad_steps = 1, yet the weights move by up to 8.2e-4 (about 0.8 x lr) on the 'zeroed' step, and optimizer.iterations advances to 4. The gate runs logged 0 non-finite steps, so this is latent until a non-finite step happens.

**Evidence (finder, reproduced).** Ran probes/p2_guard_adam.py on current code: 'nonfinite_grad_steps after poisoned step: 1.0', 'max |weight change| on the poisoned (guard-zeroed) step: 0.0008216649293899536', 'optimizer iterations: 4'. With a tf.cond around both apply_gradients (slots pre-created with _create_all_weights) in agents/losses-training/wt: 'max |weight change| ... 0.0', 'optimizer iterations: 3', and tests/test_train_smoke.py, test_training_wiring.py, test_served_epoch.py and test_physics_terms_bounded.py give '20 passed in 189.12s'. probes/p16_guard_ops.py (real model, B=256): 'trainable variables: 123 | total ops in the traced train_step: 8795; IsFinite 172, All 124, SelectV2 312'.

**Verifier (reproduced): confirmed.** Code: custom_model.py:470-485 (step_finite; grads replaced by tf.where(step_finite, g, 0)), :509-510 (both apply_gradients always run). Ran probes/v5_guard.py with the registry pair build_optimizers(Config()) (Keras 2.10 optimizer_v2 Adam) and the guard's exact logic, 3 normal steps then 1 step with a NaN gradient: 'step finite: False | main LR 0.001 indicator LR 0.005'; 'max |dw| main on the guard-zeroed step: 8.124e-04'; 'max |du| indicator on the guard-zeroed step: 3.680e-03'; 'main iterations before/after: 3 4'.

**Verifier corrections.** Only the momentum part is new. The op-count part is already NT-054 (acceptance (2): 'fused finite guards; the graph op count before and after is reported'; its why cites the same 312 SelectV2 and 172 IsFinite nodes). Retitle 'Correction to NT-036 (5) and NT-054 (2): a guard-zeroed step still moves every weight through Adam momentum'. NT-036 acceptance (5) only requires that the weights stay finite after a NaN gradient, which today's code passes although the step is not skipped. Amend it: after an injected non-finite gradient, all trainable variables, both optimizers' slots and iterations are bitwise unchanged, and nonfinite_grad_steps == 1. Implement inside NT-054's fused guard: tf.cond around both apply_gradients, with slots pre-created. MVP growth point 3, during NT-036/NT-054. cpu_cost instant.

**Backlog check (finder).** Not covered. REMEDIATION_PLAN S3 introduced the guard (done). NT-012 wants clip counts, not a correct skip.

**Fix sketch.** step_finite = is_finite(total) & is_finite(grad_global_norm). Pre-create the optimizer slots (optimizer._create_all_weights(vars) for both optimizers; Keras 2.10 API, D-001), then tf.cond(step_finite, apply both optimizers, no-op). Drop the per-variable tf.where. Check sec_per_step on the next GPU run (D-018).

**Acceptance (proposed).** (1) Test: after a forced non-finite step, every trainable variable and both optimizers' iterations are bitwise unchanged, and nonfinite_grad_steps == 1. (2) The op count of the traced train_step drops by >= 2 x n_trainable_variables (test or recorded measurement). (3) tests/test_train_smoke.py, test_training_wiring.py and test_served_epoch.py pass.

<a id="loss-6"></a>

## LOSS-6: LOSS_WEIGHT_SCHEDULE silently accepts names that cannot be scheduled (outer multipliers, typos, LAMBDA_* spelling); the compiled step keeps the old value

- **Severity:** P3. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-029.
- **Files:** `src/neural_trade/training/callbacks.py`, `src/neural_trade/core/config.py`, `src/neural_trade/training/custom_model.py`

**Description.** LambdaScheduleCallback does setattr(self.model, name, value) for any key (callbacks.py:291-307). Only the 14 lambda_<key> weights are tf.Variables (lambdas.py:15-16). lambda_trend_outer, lambda_dir_outer, lambda_dir_align_outer, lambda_coherence_outer and lambda_nll_outer are Python floats (custom_model.py:100-104), captured as constants when the step is traced (functions.py:675, 773-780). Config.validate checks ABLATE_LAMBDAS names (config.py:291-293) but not LOSS_WEIGHT_SCHEDULE keys (config.py:219-220). Separately, EarlyStopping, ModelCheckpoint and ReduceLROnPlateau compare val_loss across epochs even when the schedule changed the objective's weights in between.

**Failure scenario.** LOSS_WEIGHT_SCHEDULE={'lambda_dir_outer': {0: 1.0, 1: 0.0}}: Config accepts it, model.lambda_dir_outer reads 0.0 after epoch 1 begins, but the compiled step still computes the loss with 1.0. {'lambda_hdd': ...} or {'LAMBDA_HD': ...} are accepted too and do nothing.

**Evidence (finder, reproduced).** Ran probes/p7_schedule.py: 'Config accepted {lambda_dir_outer: ...}', 'Config accepted {lambda_hdd: ...}', 'Config accepted {LAMBDA_HD: ...}'; 'model.lambda_dir_outer after the schedule: 0.0'; 'val loss, compiled step: before 21170.62695, after scheduling lambda_dir_outer 1 -> 0: 21170.62695; freshly traced step: 21149.91016'; control 'scheduling lambda_hd (a tf.Variable) 0.1 -> 0 changes the compiled step's loss: 21170.62695 -> 21170.53125'.

**Verifier (reproduced): confirmed.** Code: callbacks.py:291-307 (setattr(self.model, name, value) for any key); lambdas.py:15-16 (only 14 lambda_<key> are tf.Variables); custom_model.py:100-104 (outer multipliers are Python floats); config.py:219-220 (LOSS_WEIGHT_SCHEDULE unvalidated), :291-293 (only ABLATE_LAMBDAS validated). Ran probes/v6_sched.py: 'Config accepted {lambda_dir_outer: ...}', 'Config accepted {lambda_hdd: ...}', 'Config accepted {LAMBDA_HD: ...}'; 'model.lambda_dir_outer after schedule: 0.0'; 'compiled step before / after: 7152.887 7152.887'; 'freshly traced step: 7135.625'; control 'scheduling lambda_dir (tf.Variable) -> compiled step 7135.625'.

**Verifier corrections.** Fits NT-029 acceptance (2) ('Config.validate enforces the declared ranges and choices'): declare the allowed LOSS_WEIGHT_SCHEDULE keys (lambda_<k> for k in lambdas._LAMBDA_VARIABLE_KEYS) as its choices, so the fix lands with NT-029 rather than as a separate item. MVP growth point 2 (the control panel and search spaces may set schedules), during NT-029. Acceptance as proposed. cpu_cost instant.

**Backlog check (finder).** Not covered by any NT item (LOSS_WEIGHT_SCHEDULE defaults to null; no item mentions it).

**Fix sketch.** Config.validate: every LOSS_WEIGHT_SCHEDULE key must be lambda_<key> for a key in lambdas._LAMBDA_VARIABLE_KEYS (clear error listing the allowed names); LambdaScheduleCallback raises on anything else. Optionally make the outer multipliers variables too. Document (or warn at fit time) that val_loss is not comparable across a schedule change, or reset EarlyStopping's best at each change.

**Acceptance (proposed).** (1) Config(LOSS_WEIGHT_SCHEDULE={'lambda_dir_outer': {0: 1}}) and {'lambda_hdd': ...} raise InvalidConfigurationError (test). (2) A valid schedule changes the compiled step's loss (existing test_lambda_schedule_is_piecewise_constant plus a compiled-step check). (3) The Config description of LOSS_WEIGHT_SCHEDULE lists the schedulable names and the val_loss caveat.

<a id="loss-7"></a>

## LOSS-7: At the early-stopping epoch, params_logger and metrics.jsonl record the restored best-epoch indicator periods under the last epoch

- **Severity:** P3. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-28.
- **Files:** `src/neural_trade/training/callbacks.py`, `src/neural_trade/training/trainer.py`, `configs/default.yaml`, `src/neural_trade/telemetry/epoch_logger.py`

**Description.** Default CALLBACKS order: csv_logger, early_stopping, model_checkpoint, tqdm, params_logger, and with a run context jsonl_epoch_logger (inserted before reduce_lr_on_plateau, trainer.py:215-226) (default.yaml:128). Keras 2.10 EarlyStopping calls set_weights(best_weights) inside on_epoch_end of the epoch where it stops. Both indicator loggers read get_learned_parameters() in their own on_epoch_end afterwards (callbacks.py:177-226, epoch_logger.py:141-145). So the last row of indicator_params_history.csv and metrics.jsonl shows the best epoch's periods labelled as the stopping epoch. The indicator-evolution figure then shows a spurious snap-back at the end of every early-stopped run.

**Failure scenario.** Epoch 0 ends with weight 10.0 (val_loss 1.0, best); epoch 1 ends with 20.0 (val_loss 2.0), and EARLY=1 stops. Both loggers record 10.0 for epoch 1.

**Evidence (finder, reproduced).** Ran probes/p10_es_logger_order.py (real build_callbacks and _with_epoch_logger, fake indicator layer reading a Dense kernel): 'callback order: [csv_logger, early_stopping, model_checkpoint, tqdm_progress, params_logger, jsonl_epoch_logger, reduce_lr_on_plateau]'; 'epoch-1 weights were 20.0 (epoch 0: 10.0)'; 'params_logger CSV ma_period_0 per epoch: [10.0, 10.0]'; 'metrics.jsonl periods per epoch: [{period/ma_period_0: 10.0}, {period/ma_period_0: 10.0}]'.

**Verifier (reproduced): confirmed.** Keras 2.10 EarlyStopping.on_epoch_end calls model.set_weights(best_weights) inside the stop branch (keras/callbacks.py:2030-2040 in the venv). The default order (default.yaml:128; trainer.py:215-226 inserts jsonl before reduce_lr_on_plateau) runs params_logger (callbacks.py:177-226) and JsonlEpochLogger._record (epoch_logger.py:122-124) after it. Ran probes/v7_es.py (real build_callbacks + _with_epoch_logger, driven through CallbackList without training): 'order: [CSVLogger, EarlyStopping, ParamsLogger, JsonlEpochLogger, ReduceLROnPlateau]'; 'epoch 1: weights set to 20.0, after callbacks layer holds 10.0, stop_training=True'; 'params_logger CSV: [10.0, 10.0]'; 'metrics.jsonl: [10.0, 10.0]'.

**Verifier corrections.** Latent on today's reference runs: EarlyStopping (EARLY=6, 20 epochs) did not fire in m5, m6 or the latest run (weights_epoch 19 of 20, research README). It becomes reachable in NT-030 quick-mode sweeps (few epochs, short patience) and matters for NT-048's discovered-indicators report, which reads the per-epoch periods. MVP growth point 5 (visuals: indicator evolution, NT-048), before NT-048. The simplest fix keeps behaviour: have _with_epoch_logger / build_callbacks put params_logger and jsonl_epoch_logger before early_stopping, and assert the order in a test. cpu_cost instant.

**Backlog check (finder).** Not covered. NT-019 covers indicator_evolution polish (a start label, empty panels) and the clipping of weights_epoch, not wrong logged periods.

**Fix sketch.** Run the loggers before the stopper: in train_and_evaluate, move 'early_stopping' (and 'model_checkpoint') after the logging callbacks, or insert the jsonl and params loggers at the front. Alternatively build EarlyStopping without the in-epoch restore and let _serve_best_weights always restore (Keras 3 behaviour, already implemented for the no-stop case).

**Acceptance (proposed).** (1) Test like the probe: with an early stop at epoch e, the last row of both indicator_params_history.csv and metrics.jsonl holds the weights the epoch ended with, and the served model still holds the best epoch's weights (test_served_epoch.py still passes). (2) The callback order is asserted in a test.

<a id="loss-8"></a>

## LOSS-8: Loss-weight calibration moves 'damping 0' weights into [CALIB_LAMBDA_MIN, CALIB_LAMBDA_MAX], contrary to its docstring (0.03 / 0.05 silently become 0.1), and measures vac_overflow at 0.1 while the others are measured at 1.0

- **Severity:** P2 (reported P3). **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-05.
- **Files:** `src/neural_trade/training/lambda_calibration.py`

**Description.** The module docstring says 'Components with damping 0 (the trend prior and the bounded physics regularisers, by default) keep their configured weight' (lambda_calibration.py:8-10). _rescale returns clip(orig * (ref/med)**0, lam_min, lam_max) = clip(orig, 0.1, 20) (lambda_calibration.py:206-209, 214, 220-223). Any damping-0 weight below 0.1 (or above 20) is changed. The shipped physics defaults (0.1) and LAMBDA_EXTENDED_TREND (0.1) sit exactly on the clip edge, so any half-strength experiment with calibrate=True is silently run at 0.1. Separately, lambda_vac_overflow is neither saved nor reset to 1.0 (lambda_calibration.py:26-30, 84-96), yet its median (measured at weight 0.1) enters ref_loss (192-194) next to medians measured at weight 1.0.

**Failure scenario.** Config LAMBDA_HD=0.03, LAMBDA_T_PERP=0.05, LAMBDA_CASIMIR=0.05, calibrate=True: the calibrated weights are 0.1, 0.1, 0.1 (a 2-3x stronger term than configured); calibration_lambdas in meta.json shows 0.1 without any warning.

**Evidence (finder, reproduced).** Ran probes/p11_calib_clip.py (real calibrate_loss_weights): '[calib] ... λ_t_perp med=7.438639 0.0500 → 0.1000', 'λ_casimir 0.0500 → 0.1000', 'λ_hd 0.0300 → 0.1000'; summary 'lambda_hd configured 0.030 -> after calibration 0.100 (damping 0)', 'lambda_t_perp configured 0.050 -> 0.100', 'lambda_casimir configured 0.050 -> 0.100'. (lambda_ife stays 0.01 only because its sampled median was 0.)

**Verifier (reproduced): confirmed.** Code: lambda_calibration.py:8-10 (docstring: damping-0 components keep their configured weight) vs :206-209 (clip(orig*(ref/med)**0, lam_min, lam_max) = clip(orig, 0.1, 20)); :214, :220-223 apply it to trend and physics; :26-30/:84-96 never reset lambda_vac_overflow to 1.0, yet its median enters ref_loss at :192-194. Ran probes/v8_calib.py (real calibrate_loss_weights, tiny model, LAMBDA_HD=0.03, T_PERP=0.05, CASIMIR=0.05, extended_trend 0.05): log 'λ_trend 0.0500 → 0.1000', 'λ_t_perp 0.0500 → 0.1000', 'λ_casimir 0.0500 → 0.1000', 'λ_hd 0.0300 → 0.1000'; 'lambda_vac_overflow: configured 0.100 -> after calibration 0.100' (not reset, measured at 0.1).

**Verifier corrections.** Raise to P2: calibrate=True is the default in train_and_evaluate (trainer.py:267, :326) and in the CLI (cli.py:66). Any NT-030/NT-050 sweep trial or engine variant that sets LAMBDA_EXTENDED_TREND or a physics weight below 0.1 (or above 20) therefore runs at the clip, while the trial records the requested value, and the objective is flat over that part of the search space. The shipped defaults (0.1) sit exactly on the clip edge. MVP growth point 2 (search spaces, NT-029/NT-030), before NT-050. Also give NT-029 the fact that the effective range of these weights under calibration is [CALIB_LAMBDA_MIN, CALIB_LAMBDA_MAX]. Acceptance as proposed, plus: a clip that binds logs a WARNING naming the weight. cpu_cost instant.

**Backlog check (finder).** Not covered. REMEDIATION_PLAN S5/S11 (restore on failure; CALIB_DAMPING_PHYSICS=0) are done, and neither addresses the clip.

**Fix sketch.** In _rescale: `if damping == 0: return orig`, or clip only the multiplicative factor, not the result. Measure vac_overflow at weight 1.0 like the others, or leave it out of ref_loss. Log a warning whenever a clip binds.

**Acceptance (proposed).** (1) Test: with LAMBDA_HD=0.03, LAMBDA_EXTENDED_TREND=0.05 and calibrate=True, the calibrated values equal the configured ones. (2) Test: ref_loss does not change when LAMBDA_VAC_OVERFLOW goes 0.1 -> 1.0 with its damping at 0 (or the median is taken at weight 1). (3) tests/test_calibration_pass.py passes.

<a id="loss-9"></a>

## LOSS-9: Coherence penalty: two of its three parts carry no gradient but sit in val_loss (a y-only constant and a non-trainable, prediction-dependent step function that EarlyStopping selects on)

- **Severity:** P3. **Status:** confirmed. **Type:** design-risk. **CPU cost:** instant. **Placement:** addition to NT-037.
- **Files:** `src/neural_trade/losses/functions.py`

**Description.** coherence_penalty = (dir_disagree + magnitude + target_smoothness)/3, weighted by LAMBDA_COHERENCE = 1 (functions.py:566-594, 779). dir_disagree uses tf.sign/tf.equal of the price heads: zero gradient everywhere. target_smoothness depends only on y_true: a data constant (0.253 on the real data, i.e. +0.084 in loss and val_loss). Only the magnitude-ordering part trains. The dir_disagree part (range 0..1, up to 0.33 of val_loss) changes whenever the price heads' signs flip. It cannot be optimised, yet it moves the EarlyStopping / ModelCheckpoint / ReduceLROnPlateau criterion.

**Failure scenario.** For random heads (B=256) the coherence contribution is 0.5328 = (dir_disagree 0.5137 + magnitude 0.6393 + target_smoothness 0.4453)/3. Its gradient w.r.t. the price heads is exactly the magnitude part's 1/(3B) = 0.0013 per element: two thirds of the term are dead weight in the objective. On real data the y-only part adds a constant 0.084.

**Evidence (finder, reproduced).** Ran probes/p15_coherence.py: 'coherence contribution to total: 0.5328 = (dir_disagree 0.5137 + magnitude 0.6393 + target_smoothness 0.4453)/3 = 0.5328'; 'its gradient w.r.t. the price heads, max |.|: 0.0013; expected from the magnitude part alone (1/3 per sample / B): 0.00130'; 'target_smoothness on the real data (a y-only constant in loss and val_loss): 0.253'.

**Verifier (reproduced): confirmed.** Code: functions.py:566-594 (dir_disagree from tf.sign/tf.equal; target_smoothness from y_true_raw signs only; penalty = mean of the three), :779 (x lambda_coherence_outer). Ran probes/v9_coh.py (custom_loss on B=256 random heads): 'coherence contribution (total diff) 0.5426; numpy dir_disagree 0.5176, magnitude 0.6649'; 'remaining part (target_smoothness) = 0.4453'; 'max |grad| wrt price heads 0.00130'; 'max |autodiff - magnitude-only analytic| = 3.14e-09'; same heads with a different y: 0.5674; 'target_smoothness on the real data: 0.253 -> 0.084 in loss and val_loss'.

**Verifier corrections.** Numbers differ slightly from the finding because the random heads differ; the claim is exact: the gradient equals the magnitude part's alone to 3e-9. Removing the parts changes loss and val_loss values (golden run changes), so record it under DECISIONS. MVP growth point 3, during NT-037, whose per-term logging should log dir_disagree and target_smoothness as diagnostics. cpu_cost instant.

**Backlog check (finder).** NT-012 only asks to log coherence_penalty. REMEDIATION_PLAN S14 removed the double count. Neither notes that 2/3 of the term has no gradient.

**Fix sketch.** Keep only the magnitude part in the objective (or give dir_disagree a differentiable form, e.g. soft signs tanh(p/tau) as in Casimir). Log dir_disagree and target_smoothness as diagnostics (fits NT-012). This changes the val_loss offset and noise, so record it in DECISIONS.

**Acceptance (proposed).** (1) Test: the coherence contribution to total equals LAMBDA_COHERENCE x (a differentiable function of the heads) and has no y-only part (value with shuffled y equals value with y). (2) dir_disagree and target_smoothness still appear in the logs. (3) test_custom_loss.py passes.

<a id="loss-10"></a>

## LOSS-10: vol_loss (std matching on h1) and the magnitude-ordering part of coherence reward spread in no-skill price heads (2.7x spread inflation without memorisation): a candidate NT-004 variant

- **Severity:** P3. **Status:** confirmed. **Type:** design-risk. **CPU cost:** instant. **Placement:** addition to NT-004.
- **Files:** `src/neural_trade/losses/functions.py`, `src/neural_trade/core/config.py`

**Description.** vol_loss = lambda_vol * min(|std(p_h1) - std(y_h1)|, 10), added as 0.1 x vol_loss (functions.py:693-700, 778; config.py:119 'prediction-spread vs target-spread penalty'). For a head with correlation rho to the target, the MSE-optimal spread is about rho x std(y), so this term pays a zero-skill head to add spread. magnitude_loss (functions.py:575-582) pushes |p2| >= |p1| >= |p0| and so carries the inflation from h1 over to h2. The m6 calibration set lambda_vol = 3.13 (meta.json). In a setting where memorisation is impossible, the objective's own equilibrium is a 2.7x wider h1 spread. The EV cost is small (about -0.03), so this is NOT the main cause of m6's raw EV h1 -0.66: the committed test analytics give EV = -r^2 almost exactly (m6 h1 r = 0.79, -r^2 = -0.62, EV -0.66), and that spread is mostly memorisation. Still, the term works against NT-004's goal and is not among its candidate variants.

**Failure scenario.** Linear price heads on features that carry no information about y, a fresh batch every step, m6's calibrated lambdas, 3000 Adam steps: pred std / target std for h1 is 0.165 with lambda_vol = 3.13 vs 0.061 with lambda_vol = 0 (h2 0.162 vs 0.121, pulled along by the magnitude ordering).

**Evidence (finder, reproduced).** Ran probes/p6b_vol_spread_ext_in_features.py: 'm6 lambdas (as trained) pred std / target std: h0 0.083 h1 0.165 h2 0.162 (EV ~ -0.03 on h1)'; 'lambda_vol = 0: h0 0.057 h1 0.061 h2 0.121'; 'lambda_vol = 0, lambda_extended_trend = 0: h0 0.054 h1 0.056 h2 0.110'. From runs/gates/*/analytics.json: m6 h1 pred_std_raw 186.15 / true 235.92 (r 0.79, EV -0.655); m5 h1 r 0.93, EV -0.845.

**Verifier (reproduced): confirmed.** Code: functions.py:693-700 (vol_loss = lambda_vol*min(|std(p_h1)-std(y_h1)|, 10)), :778 (0.1 x vol_loss), :575-582 (magnitude ordering); m6 meta lambda_vol 3.1275. Ran probes/v10_vol.py as a loss landscape, with no optimisation: custom_loss on 4096 real BTC targets, the three price heads = s*z with one z independent of y, m6 final lambdas, var=1, P(up)=0.5. Output: 'lambda_vol 3.13, lambda_ext 0.10: argmin s = 0.08 (pred std / target std)'; 'lambda_vol 0.00: argmin s = 0.00'; 'lambda_vol 0, lambda_ext 0: argmin s = 0.00'. So vol_loss alone moves a zero-skill head's optimum spread off 0. I did not re-run the finding's 3000-step Adam simulation (FAST CPU rules: no training).

**Verifier corrections.** Frame as 'Evidence for NT-004' (a candidate variant, LAMBDA_VOL=0 or a one-sided vol_loss), not as a separate defect. The size depends on the setup: 0.08 sigma_y (EV about -0.006) in my landscape with equal-spread heads, 0.165 (EV about -0.03) in the finding's independent-feature simulation. It is negligible against m6's raw EV h1 -0.66, but comparable to the latest run's raw EV h1 -0.061 cited in NT-004, which makes it worth a pre-registered variant. MVP growth point 2 (an engine scenario variant in NT-004's SPEC, R2), during R2. A code option behind a Config flag is instant; judging it is GPU work.

**Backlog check (finder).** NT-004 attributes the negative raw EV to overfitting and lists early stopping on per-head loss, tower regularisation or a robust beta as example variants. It does not name the objective terms that reward spread. Not a duplicate; this adds a variant.

**Fix sketch.** Add a pre-registered NT-004 variant with LAMBDA_VOL = 0, and/or make vol_loss one-sided (penalise only std(p) > std(y)), and/or drop magnitude ordering. Code: a one-sided option behind a Config flag (instant). Judging it is GPU work.

**Acceptance (proposed).** (1) Unit test on the no-memorisation simulation above: with the chosen option, the h1 spread ratio at equilibrium is <= 1.2x its value with vol_loss off. (2) If the default changes, the NT-004 SPEC records it as a variant and the gate runs report raw EV and spread ratio per horizon.

<a id="loss-11"></a>

## LOSS-11: Every epoch ends with a 5-window training step (30213 % 256 = 5): its gradient norm is ~6x a full step's, and its batch statistics are computed on 5 samples, right before validation

- **Severity:** P3. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-053.
- **Files:** `src/neural_trade/data/datasets.py`, `src/neural_trade/training/custom_model.py`

**Description.** make_tf_dataset batches without drop_remainder (datasets.py:14-23). Fold -1 has n_train = 30213 (runs/gates/*/meta.json), so the last of the 119 steps per epoch trains on 5 windows at batch 256 (also 5 at batch 64). That step gets full optimizer weight. Its batch-statistic terms (t_perp, HD Pearson, IFE correlation, soft-ECE on ~4 masked samples, vol std) are computed on 5 points. It is the step immediately before each validation pass that picks the served epoch.

**Failure scenario.** Real gru_attention model at init, real fold -1 windows, m6 lambdas: full batches have gradient global norm 28.1-32.7; the 5-window tail batch has 188.6 (clipped to GRAD_CLIP_NORM=20 per group). Its t_perp_total is 0.305 vs 0.048-0.059, and hd 0.183 vs 0.087-0.109.

**Evidence (finder, reproduced).** Ran probes/p14_tail_step.py: 'n_train 30213 n_val 2866 -> last train batch 5 ; last val batch 50'; 'batch 256: grad global norm 28.05 | ... t_perp_total 0.059 hd 0.094' (5 more full batches 29.6-32.7); 'batch   5: grad global norm 188.56 | soft_ece_h1 0.435 t_perp_total 0.305 hd 0.183'.

**Verifier (reproduced): confirmed.** Code: datasets.py:22 (.batch(batch_size) without drop_remainder); m6 meta.json train 30213, val 2866. Ran probes/v12_shuffle.py (create_datasets with the production Config): 'steps/epoch 119 last batch size 5'; arithmetic 30213 % 256 = 5, 30213 % 64 = 5, and the val tail batch 2866 % 256 = 50. I did not re-run the real-model gradient norm (188.6 vs 28-33), which needs the full gru_attention forward/backward (FAST CPU rules). The ratio matches noise-dominated scaling sqrt(256/5), about 7x.

**Verifier corrections.** Impact is smaller than implied. At init all steps exceed GRAD_CLIP_NORM=20 (full batches 28-33), so the tail step's post-clip norm equals a full step's. Its harm is one noisy Adam step (1 of 119, 0.8% of updates) with 5-sample batch statistics, right before validation. Logged epoch means are sample-weighted (custom_model.py:244, 289), so logs are not distorted. A fix must also update lambda_calibration.py:39 (ceil(n_train/BATCH_SIZE)). NT-053's chunk sampler replaces this path; fold the rule 'no partial training batch' into it, or land it with NT-036. MVP growth point 3. Acceptance as proposed. cpu_cost instant.

**Backlog check (finder).** Not covered.

**Fix sketch.** Train dataset: .batch(batch_size, drop_remainder=True) when n_train >= 2 x BATCH_SIZE (keep the remainder for tiny test datasets, which would otherwise give 0 steps). With reshuffling, a different 5 windows drop each epoch. val_ds unchanged. Check that _EpochTrainLogs and the calibration batch counts use the new cardinality.

**Acceptance (proposed).** (1) Test: for n_train = 30213 and BATCH_SIZE = 256, every training batch has 256 windows and there are 118 steps; for n_train < 2 x BATCH_SIZE the dataset still yields >= 1 step. (2) test_train_smoke.py and test_data_processor.py pass.

<a id="loss-12"></a>

## LOSS-12: Training shuffle buffer (2048) is ~7% of the 30,213 time-ordered windows: every epoch is a near-chronological sweep, and the lambda calibration samples only the oldest 17% of the block

- **Severity:** P2. **Status:** confirmed. **Type:** design-risk. **CPU cost:** instant. **Placement:** new item CPU-18.
- **Files:** `src/neural_trade/data/datasets.py`, `src/neural_trade/training/lambda_calibration.py`

**Description.** make_tf_dataset uses ds.shuffle(buffer_size=2048, seed, reshuffle_each_iteration=True) (datasets.py:18-22) over windows stored in time order. A 2048-element buffer only shuffles locally, so the batch 'centre of mass' walks from the oldest to the newest data within each epoch. Consequences: (1) SGD sees strongly non-iid, time-ordered batches; (2) the last steps before every validation pass train only on the newest half of the block, next to the validation block, so val_loss and the served epoch reflect recency; (3) calibrate_loss_weights samples its magnitudes from train_ds.take(n) (lambda_calibration.py:39-43, 102, 115), which covers only the oldest windows, so the calibrated lambdas reflect the oldest regime; (4) batch-level loss terms see time-local batches (see LOSS-2).

**Failure scenario.** Fold -1 (N = 30213, batch 256): batch median index 1078 at step 0 vs ~28,800 over the last 8 steps. The last 10 steps before validation use only windows 15547..30212 (the latest 48.5%). The 12 calibration sampling batches cover windows 0..5001 (median 1978, 90% below 3868). A full-size buffer gives a median in-batch span of 30,035.

**Evidence (finder, reproduced).** Ran probes/p9_shuffle.py (tf.data with the production arguments): 'steps/epoch 119; last batch size 5'; 'index span inside one batch: median 11130'; 'first batch median index 1078; last 8 batches median index [28844, 28978, 28820, 28728, 28724, 28864, 28766, 26346]'; 'the last 10 steps of the epoch use windows 15547..30212 only, i.e. the latest 48.5% of the block'; 'reference, full-buffer shuffle: median in-batch span 30035'. probes/p9b_calib_batches.py: 'calibration sampling batches: windows 0..5001, median 1978, 90% below 3868 of 30213'.

**Verifier (reproduced): confirmed.** Code: datasets.py:18-22 (shuffle(buffer_size=2048, reshuffle_each_iteration=True) over time-ordered windows); lambda_calibration.py:39-43, 102, 115 (take(n_warmup), then a fresh iteration take(n_sample)). Ran probes/v12_shuffle.py with the production arguments: 'Spearman(step position, batch median index) = 0.998'; 'first batch median 1078 | last 8 medians [28843, 28978, 28819, 28728, 28723, 28864, 28766, 26346]'; 'last 10 steps use windows 15547..30212'; 'calibration: warm-up 6 batches, sampling 12 batches -> windows 0..5017 (median 1990) of 30213'.

**Verifier corrections.** Partly known: docs/research/2026-09-28-window-free/README.md:298-299 already measured the locality (median in-batch span 11,130, the finding's number), :423-424 the design effect 1.35-1.52, and :249-250 notes that recalibration would be needed. No BACKLOG item proposes a change. The new parts are the within-epoch chronological sweep (Spearman 0.998), the recency of the pre-validation steps, and the calibration pass sampling only the oldest 17% of the block (windows 0..5,017). Split the fix. (a) The calibration sampling can be fixed without changing the training order (draw the calibration batches from a separately, fully shuffled copy); it changes the calibrated lambdas, so land it before NT-039, whose baseline is today's calibration. (b) The full-buffer shuffle changes every run: pre-register it and re-baseline the golden run, or fold it into NT-053's chunk sampler (random tiled offsets). MVP growth point 3 (NT-039, gradient/loss weighting), (a) before NT-039, (b) before NT-050 or inside NT-053. Acceptance as proposed. cpu_cost instant.

**Backlog check (finder).** Not covered. The REMEDIATION_PLAN only added an explicit shuffle seed. NT-003/NT-004 do not question batch composition.

**Fix sketch.** shuffle(buffer_size=len(Xseq), seed=seed, reshuffle_each_iteration=True). Memory is about 8 MB for fold -1 (60+7 float32 per window), and there is no per-step cost (D-018). Draw the calibration samples from the shuffled stream as well (automatic after the fix). Every subsequent gate run changes, so record the change in DECISIONS and re-baseline via NT-024 / NT-003.

**Acceptance (proposed).** (1) Test: for N = 30213, the Spearman correlation between batch position and batch median index is < 0.2 (today it is about 1), and the order is reproducible for a fixed SEED. (2) Test: the windows sampled by calibrate_loss_weights span > 80% of the training block. (3) test_reproducibility.py, test_train_smoke.py and test_calibration_pass.py pass.

<a id="loss-13"></a>

## LOSS-13: Correction to NT-012: acceptance (1)'s contribution list omits the six physics terms (and dir_align), so 'their sum equals loss within 1e-4' cannot pass with the default lambdas

- **Severity:** P3 (reported P2). **Status:** already in the backlog. **Type:** docs. **CPU cost:** instant. **Placement:** addition to NT-037.
- **Files:** `docs/BACKLOG.md`, `src/neural_trade/losses/functions.py`

**Description.** NT-012 acceptance (1) asks for contrib_point, contrib_trend, contrib_dir, contrib_nll, contrib_crps, contrib_soft_ece, contrib_vol, contrib_reg and coherence_penalty, and says 'Their sum equals loss / val_loss within 1e-4 relative (test)'. But total also includes total_t_perp, casimir_val, vac_val, hd_val, ife_val and vac_overflow_val (and dir_align when LAMBDA_DIR_ALIGN_OUTER > 0) (functions.py:771-789). All physics lambdas default to 0.1. contrib_reg must also carry the 0.1 factor of LOSS-1, and coherence_penalty must be multiplied by LAMBDA_COHERENCE.

**Failure scenario.** An implementer who follows NT-012 literally gets a gap equal to the physics sum. It is 1.8% of total on a random batch, and in m6's logs about 3% of train loss (t_perp 0.030 + hd 0.027 + ife 0.008 + vac_overflow 0.0985 of 5.14) and about 5% of val_loss (0.21 + 0.072 + 0.008 of 5.67). QA's 1e-4 check fails, or the implementer 'fixes' the sum by fudging a contribution.

**Evidence (finder, reproduced).** Ran probes/p18_nt012_sum.py: 'total 14.35958; listed contributions + coherence 14.09768; gap 0.26190 (1.82% relative)'; 'physics terms (t_perp + casimir + vac + hd + ife + vac_overflow) = 0.26190'.

**Verifier: already in the backlog.** BACKLOG NT-012 is 'dropped (2026-09-28): absorbed into NT-037'. NT-037 acceptance (3) reads: 'Per-term contributions for train and val_: contrib_* for every term of the total (losses/functions.py:770-788) with coherence_penalty among them; their sum equals loss / val_loss within 1e-4 relative (test)'. It already covers the six physics terms and dir_align, which the finding says are missing. The finding corrects the superseded NT-012 text.

**Verifier corrections.** Nothing to add beyond two details for NT-037's implementer: the total spans functions.py:771-789 (not 770-788); coherence enters as LAMBDA_COHERENCE x coherence_penalty and the regulariser as 0.1 x LAMBDA_INTER x reg_loss (see LOSS-1).

**Backlog check (finder).** This corrects NT-012's acceptance (1).

**Fix sketch.** Amend NT-012 acceptance (1): the sum of contrib_* plus the existing lambda-weighted physics keys (t_perp_loss, casimir_loss, vac_loss, hd_loss, ife_loss, and train-only vac_overflow_loss) plus contrib_dir_align plus LAMBDA_COHERENCE x coherence_penalty equals loss / val_loss within 1e-4. contrib_reg = the effective regulariser weight x reg_loss (see LOSS-1).

**Acceptance (proposed).** BACKLOG.md NT-012 acceptance (1) names every additive part of custom_loss's total. A CPU test that sums the named keys matches total within 1e-4 on a random batch with the default Config (all physics lambdas at 0.1).

<a id="loss-14"></a>

## LOSS-14: Loss docstrings, comments and config descriptions state wrong numbers: Casimir bound (18.4, not 9.2, also pinned in a test), vacuum-bandwidth 'weight via Λ_vac', LAMBDA_DIR '(focal + dice)', soft-ECE ≈ 1.24 x ECE

- **Severity:** P3. **Status:** confirmed. **Type:** docs. **CPU cost:** instant. **Placement:** new item CPU-27.
- **Files:** `src/neural_trade/losses/functions.py`, `tests/test_physics_terms_bounded.py`, `src/neural_trade/core/config.py`, `configs/default.yaml`

**Description.** (1) casimir_interference_loss says 'Bounded above by log(v_ref / VAR_FLOOR) ~ 9.2' (functions.py:308), and tests/test_physics_terms_bounded.py:93 asserts val <= log(1e4). The loss sums two pair hinges, each up to 9.21, so the bound is 18.42. The test passes only because its random inputs never reach the worst case. (2) custom_loss comments vac_val as 'always active, weight via Λ_vac' (functions.py:785). LAMBDA_VAC is a threshold (0 = off); the term enters at weight 1, and a larger Λ_vac weakens it. (3) LAMBDA_DIR is described as 'direction loss (focal + dice)' (config.py:117, default.yaml:51), but DIRECTION_LOSS defaults to bce (D-006). (4) soft_ece_loss is presented as a differentiable ECE (functions.py:229-259), but its kernel mass per sample is 1.235 for p in [0.1, 0.9] and 0.62 at p in {0, 1}. The logged soft_ece_* is therefore about 1.24 x ECE for interior probabilities (0.250 vs 0.200 true ECE), which matters when someone reads soft_ece as ECE.

**Failure scenario.** price heads (+5, -5, +5) with every variance at VAR_FLOOR: casimir = 18.42 > the documented and tested bound 9.21. Constant p = 0.30 with a 50% up rate (true ECE 0.200): soft_ece 0.2504.

**Evidence (finder, reproduced).** Ran probes/p17_casimir_bound.py: 'casimir = 18.420482635498047 ; docstring / test bound log(1/1e-4) = 9.210340371976182'. probes/p3_soft_ece.py: 'p=0.00: sum_b w_b = 0.618', 'p=0.50: sum_b w_b = 1.235'; 'constant p=0.30, 50% up (true ECE 0.200): soft 0.2504 hard 0.2027'. The gradient of soft_ece is correct: probes/p4b_softece_fd.py 'h=0.0001: max|ad-fd| = 1.38e-06'.

**Verifier (reproduced): confirmed.** Ran probes/v14_docs.py: 'VAR_FLOOR 0.0001  casimir(+5,-5,+5, var=VAR_FLOOR) = 18.420482635498047  doc/test bound log(1/VAR_FLOOR) = 9.210340371976182'; soft-ECE on 20,000 samples with a 50% up rate: 'constant p=0.3: true ECE 0.1967 soft_ece 0.2429 ratio 1.235', 'p=0.7: 0.2033 / 0.2512 ratio 1.235'. Code: functions.py:308 (bound ~9.2), tests/test_physics_terms_bounded.py:93 (asserts <= log(1e4)); functions.py:785 comment 'always active, weight via Λ_vac' vs :741-745 and :345-349 (LAMBDA_VAC is a threshold, default 0 = off; the term enters at weight 1); config.py:117 and default.yaml:51 'direction loss (focal + dice)' vs config.py:172 DIRECTION_LOSS 'bce' (D-006).

**Verifier corrections.** The wrong Casimir bound comes from REMEDIATION_PLAN S17 ('casimir ∈ [0, 9.2]'). The training dashboard's ECE chance level uses the hard val_dir_ece (training_dashboard.py:317-322, 1124), not soft_ece, so the 1.24 scale misleads only human readers. MVP growth point 1, during NT-029: its generated config reference would otherwise publish the wrong LAMBDA_DIR text. The docstring and test fixes can land anytime. Acceptance as proposed. cpu_cost instant.

**Backlog check (finder).** Not covered (NT-019..NT-023 are figure polish).

**Fix sketch.** Correct the Casimir docstring to 2·log(v_ref/VAR_FLOOR) and change the test to that bound, plus a worst-case input that exceeds 9.21. Fix the vac comment and the LAMBDA_DIR description. State the soft-ECE scale in its docstring, or normalise each sample's kernel weights to sum to 1.

**Acceptance (proposed).** (1) test_physics_terms_bounded.py asserts casimir <= 2·log(1/VAR_FLOOR) and includes the alternating-sign, variance-at-floor case (value > 9.21). (2) grep finds no 'focal + dice' in the LAMBDA_DIR description and no 'weight via Λ_vac'. (3) The soft_ece docstring states its scale relative to hard ECE (or a test shows soft_ece within 5% of hard ECE for constant p in [0.1, 0.9]).

