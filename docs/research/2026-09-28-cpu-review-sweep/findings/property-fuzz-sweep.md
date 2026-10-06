# Area: property-fuzz-sweep (6 findings)

Back to the [index](README.md) and the [report](../README.md).

<a id="fuzz-1"></a>

## FUZZ-1: Temperature scaling never reaches the NLL minimum: fixed-step gradient descent stalls near T = 1 on weak heads, so the 'calibrated' P(up) is barely calibrated

- **Severity:** P1. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-04.
- **Files:** `src/neural_trade/calibration/temperature_scaling.py`, `src/neural_trade/calibration/pipeline.py`, `src/neural_trade/strategy/signals.py`, `tests/test_calibration.py`

**Description.** _fit_temperature (temperature_scaling.py:62-106) fits T by plain gradient descent: it starts at T = 1 and runs 500 steps of lr 0.05 (:90-100). The NLL gradient in T scales with z^2, where z is the head's logit. On this project's heads (AUC about 0.50, P(up) within about ±0.05 of 0.5, logit sd about 0.08) each step moves T by about 1e-4, so the fit returns about 1.0 whatever the optimum is. A second defect: the loop measures the NLL at the pre-step T (:91, :102) but stores the post-step T as best_T (:99, :104), so the returned T is one step past the best one it saw. The fitted T feeds every calibrated number: CalibrationPipeline.fit_from_arrays (pipeline.py:209), the served P(up) (apply), the eval report's direction Brier and ECE, and the strategies' SignalFrame, which reads calibrated probabilities by default (signals.py:72-74). CalibrationPipeline.summary labels 0.95 < T < 1.05 as 'well-calibrated (T ≈ 1)' (pipeline.py:457-462). The gate temperatures of m3-m6 (0.95-1.03, runs/gates/m*/analytics.json) therefore say nothing about calibration. The existing test (tests/test_calibration.py:59-80) only checks that probabilities move toward 0.5 and toward p_true, so it passes with an unconverged T.

**Failure scenario.** Input: a weak, overconfident head (3,500 calibration samples, logit sd 0.4, true temperature 5). Output: _fit_temperature returns 1.41, while the NLL minimiser is 5.16. The 'calibrated' head keeps ECE 0.040, where 0.001 is reachable. On the real served P(up) saved in notebooks/02_backtest.ipynb (test block, 7,236 bars) the fit returns T = 1.07, while the NLL optimum is at the search bound (T → ∞, i.e. flatten to 0.5: no skill). Even on the repo's own test fixture (sharpen = 3) it returns 2.41 against an optimum of 2.99.

**Evidence (finder, reproduced).** Reproduced (scratch agents/property-fuzz-sweep/). temp_probe.py compares against scipy minimize_scalar over log T on the same NLL, n = 3,500: logit sd 0.4, T_true 5: T_exact 5.162, T_fit 1.411, ECE 0.0400 vs 0.0010. sd 0.3, x3: 2.752 vs 1.231. sd 0.5, x10: 10.798 vs 1.575. sd 1.5, x2: 2.130 vs 1.809. With lr = 2.0: T_fit 0.3491 vs exact 0.2948, NLL 0.11725 vs 0.11558 (the off-by-one). nb_series.py decodes the 'close' and 'weighted' traces of notebooks/02_backtest.ipynb cell 4 (7,236 bars): weighted P(up) sd 0.0194, logit sd 0.078. On 5,686 scored h1 bars: _fit_temperature T = 1.0737, exact at the bound; NLL 0.695345 vs 0.693149. Fitting on the first half and scoring the second: served ECE 0.0393 / Brier 0.25143; GD fit (T 1.067) 0.0388 / 0.25132; exact fit 0.0354 / 0.25000. The repo fixture (tests/test_calibration.py _synthetic, sharpen 3, n 3,000): fitted h0/h1/h2 2.412/2.396/2.412, exact 2.987/2.966/3.011. A candidate fix (bounded minimize_scalar over log T in [1e-2, 1e3], in agents/property-fuzz-sweep/wt/) matches the exact T on every probe case (NLL difference < 1e-15), and tests/test_calibration.py passes (10 passed), as do 180 related non-TF tests.

**Verifier (reproduced): confirmed.** Code: temperature_scaling.py:86-106. Gradient descent starts at T=1 and runs 500 steps at lr 0.05. The only caller is pipeline.py:209 through TemperatureScaler.fit:151, and nothing passes another lr or n_steps. My own probe (agents/verify-property-fuzz-sweep/v1_temp.py, n=3,500) compared the fit with a bounded minimize_scalar over log T in [1e-2, 1e3]:
- logit sd 0.4, true T 5: fit 1.448 vs exact 8.359, ECE 0.042 vs 0.006.
- sd 1.5, T 5: 2.299 vs 4.744, ECE 0.057 vs 0.017.
- sd 4, T 5: 3.126 vs 5.010.
- sd 0.08, T 2: 1.016 vs 1.742.
- No-skill head: fit 1.054 vs exact 1000 (the bound).
When the optimum is below 1, the fit converges (0.465 = 0.465, 0.461 = 0.461). The failure is one-sided: it cannot soften.
Analytic check: for a no-skill head with logit sd s, T moves at most about lr*n_steps*s^2/4. Measured 1.0103, 1.0423, 1.0494 and 1.1773 at s = 0.05, 0.08, 0.1 and 0.2. So on this project's heads (logit sd about 0.08) T always lands in about [0.96, 1.04], and summary() (pipeline.py:457-462) always prints 'well-calibrated (T ≈ 1)'. The gate files agree: m3-m6 temperatures are 0.94-1.03 (runs/gates/m*/analytics.json).
Repo fixture (tests/test_calibration.py _synthetic, SHARPEN 3): the pipeline fits 2.412 / 2.396 / 2.412, the exact optimum is 2.987 / 2.966 / 3.011. The fixture's own docstring (:23-24) says the fit 'must fit T ~ sharpen'.
The notebook-02 figures reproduce: 5,686 bars, fit T 1.0737, exact at the bound, NLL 0.695345 vs 0.693149.
The calibrated probabilities are the default everywhere: PredictionFrame.prob(calibrated=True) at frame.py:61-64, SignalFrame.build at signals.py:72-74, and the eval report.
Candidate fix (bounded minimize_scalar, in verify-property-fuzz-sweep/wt): the fixture gives 2.987 / 2.966 / 3.011. tests/test_calibration.py passes (10 passed), and all 57 non-TF tests that mention temperature or calibration pass (test_viz_tables, test_viz_misc, test_calibration). Not covered by any BACKLOG item (grep temperat: only DECISIONS:76 and the archive).

**Verifier corrections.** (1) The 'second defect' (NLL measured at the pre-step T, best_T stored post-step) is real but has no effect today. Only lr 0.05 is ever used, and from T=1 the path is monotone, so the post-step T is always at least as good. My lr=2.0 probe gave 0.3129 vs 0.3131. Mention it as a note, not as a defect with numbers.
(2) The notebook-02 evidence is a proxy. It refits on the 'weighted' trace, which is already temperature-scaled and a lambda-weighted mix of the three horizons, scored against h1 labels on the test block. It is not the pipeline's per-horizon cal-block input. The analytic bound above is the stronger proof for weak heads.
(3) The golden run does not cover this. scripts/golden_run.py:53-62 records the raw heads (res.predictions, trainer.py:404), lambdas, periods and conformal coverage, but no temperature and no calibrated P(up). So acceptance (5) 'the report lists which recorded arrays changed' is wrong: verify passes unchanged after the fix. Worse, a later NT-026/NT-027 move could break temperature scaling without the golden run noticing. The fix should add temperature/{h} and pred_cal/direction_prob/{h} (res.predictions_calibrated) to golden_run's record in the same change, and re-record the baseline.
(4) Acceptance (1) as written is circular if the implementation itself is scipy's bounded search. Instead check NLL(fit) <= min over a dense log-grid (for example 20,001 points on [1e-2, 1e3]) + 1e-9, and the fitted T is within 1% of the grid minimiser whenever that minimiser is not at a bound.
(5) Downstream effect to state in the item: after the fix, a no-skill head's calibrated P(up) collapses toward 0.5. The fixed-vote logic (VOTE_UP/DOWN 0.55/0.45, signals.py:44-45) and the enhanced_multi_horizon and liberal strategies then trade less or not at all. That creates more zero-trade leaderboard rows, so FUZZ-3's guard (NT-031) must land first or together.
(6) The OnlineTemperatureCalibrator (online_calibrator.py) uses the same per-sample gradient step; note it, but it is outside this item.
MVP growth point 1, before the NT-026/NT-027 calibration move, as its own numbers-change step. cpu_cost: instant.

**Backlog check (finder).** Not in BACKLOG NT-001..NT-054, ROADMAP, STATUS, DECISIONS or docs/archive/REMEDIATION_PLAN_2026-09.md: grep for temperat/_fit_temperature/best_T finds only D-008 (conformal) and NT-013 (baselines). NT-004 is about the delta shrinkage beta, not the temperature.

**Fix sketch.** Replace the gradient-descent loop with a bounded 1-D minimisation of _nll(z / exp(lt), y) over lt in [log 1e-2, log 1e3] (scipy.optimize.minimize_scalar, method='bounded', xatol 1e-6). Keep the n_steps/lr arguments for API compatibility (ignored or deprecated). Return 1.0 for an empty input or all-zero logits. Record in pipeline_meta.json whether T hit a bound, and change summary() so it never calls a bound-hitting or unconverged T 'well-calibrated'. MVP growth point 1 (the engine's one scorer inherits calibrated P(up), Brier and ECE, and the default strategy's weighted direction). Land it BEFORE the NT-026/NT-027 golden recording, as its own numbers-change step: a 'no number changed' module move cannot carry it. The calibrated direction outputs, ECE, Brier and the calibrated_quantile thresholds will differ from today's recordings.

**Acceptance (proposed).** (1) Test: synthetic heads with n = 3,500, logit sd in {0.1, 0.3, 1.5, 4} and true T in {0.2, 0.5, 2, 5, 10}: the fitted T is within 1% of the exact NLL minimiser over log T (scipy bounded search), and NLL(fit) - NLL(exact) < 1e-9. (2) Test: a head independent of its labels gets T >= 100 or the upper bound, and every calibrated probability is within 0.01 of 0.5. (3) tests/test_calibration.py asserts T within 5% of the fixture's SHARPEN on each horizon, not only the direction of movement. (4) CalibrationPipeline.summary does not print 'well-calibrated' for a T at a bound (test on the text). (5) The fast suite and ruff pass. scripts/golden_run.py is re-recorded, and the implementer's report lists which recorded arrays changed (only calibrated direction outputs and what reads them).

<a id="fuzz-2"></a>

## FUZZ-2: Evidence for NT-053 / NT-041: the period ceiling does not follow LOOKBACK: every override path keeps MOMENTUM_CLIP_MAX = 60

- **Severity:** P2. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-06.
- **Files:** `src/neural_trade/core/config.py`, `configs/default.yaml`, `src/neural_trade/training/custom_model.py`, `src/neural_trade/cli.py`, `src/neural_trade/visualization/indicator_evolution.py`, `tests/test_config.py`

**Description.** The window research (docs/research/2026-09-28-window-free/README.md:20) and NT-053 treat the indicator period ceiling as 'the window itself (config.py:150, :229-230)'. That holds only for a fresh Config(LOOKBACK=L). __post_init__ sets MOMENTUM_CLIP_MAX = LOOKBACK once (config.py:229-230), and configs/default.yaml:84 pins MOMENTUM_CLIP_MAX: 60. override() (:314), copy() (:329), from_yaml of default.yaml, the CLI's --set (cli.py:43-47) and a run's saved config.yaml never re-derive it. So a scenario that changes LOOKBACK keeps the 60-bar ceiling that train_step enforces (custom_model.py:517), and indicator_evolution.py:207 draws it. Examples: an NT-026 spec 'base config + overrides', an NT-041 wall-clock window converted to bars, or an NT-053 A/B arm. With LOOKBACK 240 the periods cannot grow past 60. With LOOKBACK 12 (a 60-minute window at 5-minute bars) they may reach 60 on a 12-bar window. D-032 removes the ceiling eventually, but until NT-053's stages land the window model stays the default (D-032) and this coupling is what every window-changing run gets.

**Failure scenario.** `neural-trade train --config configs/default.yaml --set LOOKBACK=120` (or Config().override(LOOKBACK=120), or Config().copy(LOOKBACK=120)) trains with a 60-bar period ceiling. Config(LOOKBACK=120) gives 120. Minimal counterexample found by hypothesis: Config().override(LOOKBACK=3).MOMENTUM_CLIP_MAX == 60, while Config(LOOKBACK=3).MOMENTUM_CLIP_MAX == 3.

**Evidence (finder, reproduced).** Reproduced. props/test_calib_config_props.py::test_override_lookback_rederives_clip_max fails with 'AssertionError: (60, 3)'. Through the CLI loader: `python -c "from neural_trade.cli import _load_config, _parse_sets; ..."` printed 'LOOKBACK=12: CLI --config default.yaml --set -> MOMENTUM_CLIP_MAX=60.0; CLI no config --set -> 60; Config(LOOKBACK=12) -> 12', and the same for 30, 120 and 240. Round trip Config -> to_yaml -> from_yaml also turns MOMENTUM_CLIP_MAX 60 (int) into 60.0 (float); nothing else changes (hypothesis round trip over strings, floats, lists, dicts: passed). A candidate fix in agents/property-fuzz-sweep/wt: no materialisation, custom_model uses cfg.momentum_clip_max, default.yaml null. With it, --set LOOKBACK=12/120 gives 12/120 and default.yaml gives 60. Only the two tests that pin the old behaviour fail (tests/test_config.py:24 and :84-86).

**Verifier (reproduced): confirmed.** Code: config.py:150 (field 'None -> LOOKBACK') and :229-230 (__post_init__ fills in LOOKBACK once). override (:314) and copy (:329) never re-derive it. configs/default.yaml:84 pins 60. custom_model.py:517 reads config.MOMENTUM_CLIP_MAX. indicator_evolution.py:207 uses `MOMENTUM_CLIP_MAX or LOOKBACK`. The momentum_clip_max property (:297-298) also returns 60, because the field is already filled in.
Reproduced for L = 12, 120 and 240. Config(LOOKBACK=L) gives L. Config().override(LOOKBACK=L) gives 60, and copy gives 60.0. cli._load_config('configs/default.yaml', --set LOOKBACK=L) gives 60.0, and cli._load_config(None, --set LOOKBACK=L) gives 60. A YAML that sets only LOOKBACK gives L (correct).
No code in src/neural_trade/experiments, scripts, notebook or serving overrides LOOKBACK today. The trigger is the CLI --set now, and the engine or NT-041 window specs later.
The research README:20 states the ceiling 'is the window itself (config.py:150, :229-230)', which holds only for a fresh Config. Not in NT-041, NT-053 or elsewhere.

**Verifier corrections.** Line references are correct. Title and framing are fine ('Evidence for NT-053 / NT-041').
Also name NT-029 and NT-030: a sweep axis over LOOKBACK, or a search space that includes it, inherits the 60-bar ceiling.
The fix sketch works. An equivalent, smaller alternative: override() re-derives MOMENTUM_CLIP_MAX when LOOKBACK changes and the ceiling was derived rather than set. But default.yaml:84 must become null either way, otherwise the default.yaml + --set path keeps 60.
Acceptance (2) needs cfg.momentum_clip_max passed at custom_model.py:517. tests/test_physics_terms_bounded.py:164 must switch to the property, or clip_learned_periods receives None.
MVP growth point 4, before NT-041 converts the window and before any NT-030 sweep or NT-053 A/B arm that changes LOOKBACK. The golden run is unchanged at LOOKBACK 60. cpu_cost: instant.

**Backlog check (finder).** Not in NT-001..NT-054. NT-053's why and the research README cite the ceiling as 'the window itself'; NT-041 plans 'the period ceiling in minutes' (research README:141) but not that today's default does not follow LOOKBACK. D-032 (no configured ceiling) is not re-litigated: this concerns the interim behaviour and the A/B arms.

**Fix sketch.** Keep MOMENTUM_CLIP_MAX None unless it is set explicitly. Resolve it lazily through Config.momentum_clip_max in custom_model.py:517, indicator_evolution.py:207 and tests/test_physics_terms_bounded.py:164. Set configs/default.yaml:84 to null. Update tests/test_config.py:24, :84-86 to the property. NT-053's A/B specs must then set the ceiling explicitly per arm (the research's 'B1 keeps ceiling 60'). MVP growth point 4 (generality: wall-clock window). Land it before NT-041 converts the window to bars, and before any sweep or A/B arm that changes LOOKBACK. The golden run is unchanged at LOOKBACK 60.

**Acceptance (proposed).** (1) Test: Config().override(LOOKBACK=120), Config().copy(LOOKBACK=120), Config.from_yaml('configs/default.yaml').override(LOOKBACK=120) and cli._load_config('configs/default.yaml', {'LOOKBACK': 120}) all give momentum_clip_max == 120. An explicit MOMENTUM_CLIP_MAX=20 stays 20 through the same paths. (2) Test: the value custom_model's train_step passes to clip_learned_periods equals cfg.momentum_clip_max. (3) configs/default.yaml loads unchanged in behaviour at LOOKBACK 60, and scripts/golden_run.py verify passes. (4) The fast suite passes.

<a id="fuzz-3"></a>

## FUZZ-3: Evidence for NT-031: a configuration that never trades scores net Sharpe exactly 0 and a NaN random-null percentile, so it outranks every losing configuration and slips past a '< threshold' guard-rail

- **Severity:** P1. **Status:** confirmed. **Type:** design-risk. **CPU cost:** instant. **Placement:** addition to NT-031.
- **Files:** `src/neural_trade/strategy/performance.py`, `src/neural_trade/strategy/backtest.py`

**Description.** The scorer the engine inherits returns sharpe_net = 0.0 for a run with zero trades, because sharpe() returns 0.0 when the std is 0 (performance.py:16). random_same_frequency returns percentile_total_return = NaN for zero trades (backtest.py:277-278). NT-005 records that no trading configuration is net-positive on the reference setup. So on a leaderboard ranked by dev-fold net Sharpe (D-020), every row that trades sits below 0, and a configuration whose thresholds never trigger ranks first. It also passes max drawdown (0) and beats buy-and-hold on any falling fold. Its NaN random percentile passes a guard written as 'disqualify if pct < 95', because NaN < 95 is False. NT-031's acceptance takes guard-rail thresholds 'from the scenario' and names 'the number of trades', but it sets no non-zero default minimum and says nothing about NaN guard values.

**Failure scenario.** Two rows on the same test block (the real closes saved in notebook 02). A calibrated_quantile variant that trades: 1,074 trades, sharpe_net -398.6. The same strategy with thresholds no bar reaches: 0 trades, sharpe_net 0.000, max_dd 0, random percentile NaN. Sorted by net Sharpe, the never-trading row is the winner. `pct < 95` evaluates to False for it, so a guard coded that way does not disqualify it.

**Evidence (finder, reproduced).** Reproduced: agents/property-fuzz-sweep/zero_trade_rank.py printed 'calibrated_quantile (trades) n_trades 1074 sharpe_net -398.636 ... random pct 45.0', 'calibrated_quantile, never triggers n_trades 0 sharpe_net 0.000 total_return +0.0000 max_dd 0.0000 | B&H return +0.0410 | random pct nan' and "guard 'disqualify if random percentile < 95': False | guard 'qualify if random percentile >= 95': False". Code: performance.py:16, backtest.py:277-278. Related check: on the same real closes, raising costs never raised the net Sharpe for 120 random-strategy cells (sharpe_costs.py). Hypothesis found only a synthetic counterexample (a constant-decline price, where a negative Sharpe becomes less negative as costs add variance), so cost monotonicity is not a practical problem.

**Verifier (reproduced): confirmed.** Code: performance.py:16 (sharpe returns 0.0 when sd = 0) and backtest.py:276-278 (random_same_frequency returns NaN percentiles when n_trades == 0).
My reproduction (agents/verify-property-fuzz-sweep/v3_zero.py) uses the real notebook-02 test-block closes (7,236 bars), the real served weighted P(up) on every horizon, and served deltas 0 (as in gate m6, beta_h1 = 0). Rows sorted by sharpe_net:
- enhanced_multi_horizon: n_trades 0, sharpe_net 0.000, max_dd 0, random percentile nan.
- always_flat: the same.
- liberal: 113 trades, sharpe_net -96.2.
- calibrated_quantile: 380 trades, sharpe_net -153.7.
For the zero-trade row, `pct < 95` evaluates to False, so a guard coded that way passes it.
This zero-trade row is not synthetic: BACKLOG NT-005 (line 147) and line 170 record enhanced_multi_horizon (tag 'default') making 0 trades in the latest notebook run. NT-031 lists 'the number of trades' as a guard-rail with thresholds from the scenario, but sets no default minimum and has no NaN rule. D-020 adds nothing.

**Verifier corrections.** (1) Stronger evidence than the finding's synthetic setup: a registered strategy (enhanced_multi_horizon) already yields a zero-trade row on today's reference run, and it tops the net-Sharpe order ahead of every trading strategy.
(2) Qualify 'beats buy-and-hold on any falling fold'. On this block buy-and-hold is +4.1%, so a total-return 'beat buy-and-hold' guard would disqualify the zero-trade row here. It escapes only on falling folds, or when the guard is coded on Sharpe.
(3) Note that a zero-trade configuration is indistinguishable from the always_flat baseline. The leaderboard could mark such rows 'equals always_flat'.
(4) FUZZ-1's fix makes zero-trade rows more common, so land this rule with NT-031 before NT-050's first sweep.
MVP growth point 2, during NT-031. cpu_cost: instant.

**Backlog check (finder).** NT-031 lists 'the number of trades' as a guard-rail, with thresholds from the scenario, but has no default minimum, no NaN rule and no zero-trade test. NT-005 establishes that trading rows are negative. Not covered elsewhere.

**Fix sketch.** In NT-031: (a) a default minimum trade count per dev fold (for example >= 1, better a scenario default such as 20), which the scenario may raise but not set to 0 without a stated reason; (b) any non-finite guard-rail value disqualifies the row, with the reason 'guard-rail not measurable (no trades)'; (c) optionally show sharpe_net as n/a for zero-trade rows. MVP growth point 2 (leaderboard). Land it during NT-031, before NT-050's first real sweep ranks anything.

**Acceptance (proposed).** (1) Test: a run index with a zero-trade row (sharpe_net 0.0, random percentile NaN) and a trading row (sharpe_net -3): the zero-trade row is disqualified with a named reason and the trading row is the eligible winner. (2) Test: a guard-rail value of NaN disqualifies under every guard-rail. (3) The default minimum trade count is visible in the leaderboard table's header or caption (test on the text).

<a id="fuzz-4"></a>

## FUZZ-4: Off-by-one in the window start: when max(EXTENDED_TREND_PERIODS) >= LOOKBACK, the first sequence's momentum feature reads bar -1 and gets a made-up 0.0

- **Severity:** P3. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** addition to NT-041, NT-053.
- **Files:** `src/neural_trade/data/windowing.py`, `src/neural_trade/core/config.py`

**Description.** make_sequences_with_extended_trends starts at start_idx = max(LOOKBACK, max(P)) (windowing.py:69), but the feature for anchor i reads close[i - 1 - p] (:89 with :52). At i = max(P) >= LOOKBACK the longest period reads index -1, and compute_extended_trend_features substitutes 0.0 (:60). make_inference_windows has the same start (:131, :138). sequence_anchor_bars mirrors the start (:110), so a fix must move all three together. Config.validate does not require max(EXTENDED_TREND_PERIODS) < LOOKBACK. The reference setup (60 vs 20) is unaffected. It happens when the window is shorter than the longest momentum lag, for example NT-041's wall-clock window with long horizons, or NT-053's 'network context' shorter than the horizons.

**Failure scenario.** Hypothesis minimal inputs. LOOKBACK=1, HORIZON_STEPS=[1,1,1], EXTENDED_TREND_PERIODS=[1,1,1], close=[1,1,1]: sequence 0 (anchor bar 0) gets ext = 0.0 for period 1, where no past bar exists. make_inference_windows(close, 1, extended_trend_periods=[1]): window 0 reads bar -1. In general exactly one sequence (the first) gets one fabricated 0.0 feature.

**Evidence (finder, reproduced).** Reproduced: props/test_data_props.py::test_targets_and_anchors_are_hand_computed failed with 'extended feature for period 1 at anchor 0 reads bar -1 < 0 (filled with 0.0)', and test_inference_windows_match_training_windows failed with 'inference window 0 (anchor 1) period 1 reads bar -1'. With that case allowed, the same properties pass over 200 examples each: targets equal close[a+h] - close[a], windows equal close[a-L+1:a+1], last_close = close[a]; sequence_anchor_bars agrees with the sequences; purged splits are disjoint, ordered and gapped.

**Verifier (reproduced): confirmed.** Code: windowing.py:68-69 (start = max(L, max P)) and :89 (the features read idx = i-1, ref_idx = i-1-p). When i = max(P), ref_idx = -1, and :60 substitutes 0.0. sequence_anchor_bars (:110) and make_inference_windows (:131) mirror that start.
Reproduced with the real windowing functions on a 200-bar random walk, HORIZON_STEPS and periods [10, 15, 20]:
- L = 60: 0 bad features.
- L = 20, 15 and 12: exactly 1 bad feature, at sequence 0, period 20 (ext[0] = [-4.51, -3.87, 0.0]).
Config.validate (config.py:249-262) has no check of max(P) against LOOKBACK. The default MAX_SEQUENCE_COUNT (53,280, config.py:69) exceeds the committed CSV's 43,421 sequences, so the first sequence is not dropped by the cap.

**Verifier corrections.** (1) Also serving/predictor.py:135 and :150 mirror the same start (the anchor rows and the timestamp index). A start fix must move them too, or the anchor rows and timestamps shift against the windows by one bar.
(2) The inference half has no effect today: the Predictor discards make_inference_windows' extended features (`X, lc, _` at predictor.py:133 and :148). Only the training feature is fabricated: one value out of N sequences, feeding extended_trend_loss (losses/functions.py:160).
(3) Prefer the start fix, max(L, max(P) + 1), over refusing max(P) >= LOOKBACK in validate. Refusing would block NT-041's long horizons with a short window, and NT-053's short network context.
MVP growth point 4, during NT-041 or folded into NT-053's input path. The golden run is unchanged at the defaults (60 vs 20). cpu_cost: instant.

**Backlog check (finder).** Not in NT-001..NT-054. The window research (README:227) describes 'starts at max(L, max ext period)' without noting the off-by-one.

**Fix sketch.** start = max(LOOKBACK, max(P) + 1) in make_sequences_with_extended_trends, sequence_anchor_bars and make_inference_windows, or refuse max(P) >= LOOKBACK in Config.validate. MVP growth point 4. Land it during NT-041, or fold it into NT-053's input-path redesign. The golden run is unchanged at the defaults.

**Acceptance (proposed).** (1) Property test (hypothesis or a parameter grid): for every LOOKBACK, periods and horizons, every extended feature equals close[a] - close[a - p] with a - p >= 0, and sequence_anchor_bars equals the anchors of the built sequences. (2) The same for make_inference_windows. (3) scripts/golden_run.py verify passes.

<a id="fuzz-5"></a>

## FUZZ-5: Evidence for NT-028: numpy_metrics.pit_uniformity has no caller and computes a wrong one-sided KS statistic

- **Severity:** P3. **Status:** confirmed. **Type:** hygiene. **CPU cost:** instant. **Placement:** addition to NT-028.
- **Files:** `src/neural_trade/metrics/numpy_metrics.py`

**Description.** pit_uniformity (numpy_metrics.py:130-163) returns max|i/n - u_(i)| (:162). That misses the D- side, max(u_(i) - (i-1)/n), so it understates the KS distance by up to 1/n. pit_ks (:277-290), the function the report and the Metrics registry use, is exact. A search of src, scripts, tests, notebooks and docs finds no caller of pit_uniformity. It is a D-029 removal candidate with evidence of both conditions, and a wrong helper someone might pick up.

**Failure scenario.** y = [1], mu = [0], sigma = [1]: pit_uniformity returns 0.1587, while scipy.stats.kstest(Phi(1), 'uniform') gives 0.8413 (and pit_ks gives 0.8413).

**Evidence (finder, reproduced).** Reproduced: props/test_metrics_props.py::test_pit_uniformity_vs_scipy failed on that minimal input. test_pit_ks_vs_scipy passes on 200 random examples. `grep -rn pit_uniformity` over the worktree (py, ipynb, md) matches only the definition.

**Verifier (reproduced): confirmed.** Code: numpy_metrics.py:130-163. Line 162 takes only max|i/n - u_(i)| and misses the D- side, max(u_(i) - (i-1)/n). pit_ks at :277-290 is exact.
Reproduced: for y = [1], mu = [0], sd = [1], pit_uniformity returns 0.1587, while pit_ks and scipy kstest both return 0.8413. Over 200 random normal draws at each of n = 5, 50 and 500, it understates KS in 48-49% of cases, by at most exactly 1/n (0.2, 0.02, 0.002).
The function is not decorated and not registered. grep over py, ipynb, md, yaml and json in the worktree finds only the definition. It is not in NT-028's candidate list.

**Verifier corrections.** Impact wording: the bias is at most 1/n, so it is negligible at report sizes, and no caller exists, so no number is wrong today. It is purely a D-029 removal candidate with evidence of both conditions (the grep shows no use; nothing re-creates it). Otherwise as reported: MVP growth point 1, during NT-028. cpu_cost: instant.

**Backlog check (finder).** NT-028 lists stale-removal candidates but not this function.

**Fix sketch.** Add it to NT-028's candidate list: remove it, or make it an alias of pit_ks, with the grep as evidence of no use. MVP growth point 1 (structure / stale removal). Land it during NT-028.

**Acceptance (proposed).** (1) numpy_metrics has no pit_uniformity, or it equals pit_ks on random inputs (test). (2) The NT-028 report shows the grep with no caller. (3) The fast suite passes.

<a id="fuzz-6"></a>

## FUZZ-6: build_strategy silently drops long_above / short_below / median for calibrated_quantile: accepted keys with no effect

- **Severity:** P3. **Status:** confirmed. **Type:** bug. **CPU cost:** instant. **Placement:** new item CPU-24.
- **Files:** `src/neural_trade/strategy/params.py`, `src/neural_trade/strategy/strategies.py`, `src/neural_trade/notebook/backtest_ui.py`

**Description.** params.py promises 'unknown knobs rejected (a typo must not silently fall back to a default)'. _check_keys (params.py:15) accepts long_above, short_below and median as known fields, and build_strategy then pops them without a warning (:34-35) before from_calibration sets them from the calibration table. The notebook hides them as derived (backtest_ui.py:26, :523), but a params YAML, the CLI (--params) or a sweep search space (NT-030, NT-033, NT-005's re-knobbing) can set them, and every such trial silently runs the calibration-derived thresholds.

**Failure scenario.** build_strategy('calibrated_quantile', {'long_above': 0.70, 'short_below': 0.30, 'median': 0.6}, calibration={0.1: 0.45, 0.5: 0.5, 0.9: 0.55}) returns a strategy with long_above 0.55, short_below 0.45 and median 0.5. No error, no warning, no log line (run with -W error::UserWarning and DEBUG logging).

**Evidence (finder, reproduced).** Reproduced: printed 'requested long_above 0.70 short_below 0.30 median 0.6 -> got 0.55 0.45 0.5'. The YAML round trip of every other registered strategy's params and of BacktestConfig is lossless (checked for all 7 strategies).

**Verifier (reproduced): confirmed.** Code: params.py:1-2 promises unknown knobs are rejected. _check_keys (:15-21) accepts every dataclass field, and long_above, short_below and median are fields of QuantileSignalStrategy (strategies.py:241-243). build_strategy then pops them silently (params.py:34-35) before from_calibration.
Reproduced with python -W error and DEBUG logging:
- dict params {long_above 0.70, short_below 0.30, median 0.6} with table {0.1: 0.45, 0.5: 0.5, 0.9: 0.55} gives 0.55 / 0.45 / 0.5, with no warning and no log line.
- The same through load_params from a YAML file (the CLI --params path, cli.py:116-118) gives the same result.
- A typo 'long_abve' is refused with "did you mean 'long_above'?". Following the hint gives a key that is silently ignored.
No src caller passes derived keys: the notebook widgets skip _DERIVED (backtest_ui.py:26, :523), and no params writer emits them. So raising breaks nothing. The existing test (tests/test_backtest.py:253-275) does not pass derived keys. Not in the backlog.

**Verifier corrections.** Add the typo-hint trap to the description. Otherwise as reported: MVP growth point 2, before NT-030's search spaces. cpu_cost: instant.

**Backlog check (finder).** Not in NT-001..NT-054. NT-029 covers Config fields, not strategy knobs.

**Fix sketch.** Raise InvalidConfigurationError (or warn and record the fact) when a derived key is passed to a from_calibration strategy. Expose the derived set as a class attribute (for example QuantileSignalStrategy.DERIVED), which backtest_ui._DERIVED and the NT-030 search-space builder read, so a search space cannot include them. MVP growth point 2 (sweeps and control panel). Land it before NT-030's search spaces.

**Acceptance (proposed).** (1) Test: build_strategy('calibrated_quantile', {'long_above': 0.7}, calibration=table) raises and names the key and the reason. (2) Test: a search space containing a derived key is refused before any trial starts (with NT-030). (3) Existing strategy tests and assert_no_lookahead pass.

