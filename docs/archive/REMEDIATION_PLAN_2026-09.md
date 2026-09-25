> **Archived (superseded).** The remediation plan of 2026-09-22, executed on branch `remediation/plan`. Its open items now live in [docs/BACKLOG.md](../BACKLOG.md) and its milestones in [docs/ROADMAP.md](../ROADMAP.md).

# neural_trade — Remediation & Restructure Plan

## Context

**Why.** The capability review found a working measurement stack around a model that has never learned. The committed training run (`indicator_params_history.csv`, 20 epochs) shows validation loss bit-identical every epoch, every direction head emitting one class (MCC 0, predicted-up 0/100%), all 18 learnable-indicator periods NaN, variance heads constant to 1e-6, and zero trades in every backtest. Independently, the codebase is a 3,665-line monolith with a 3,615-line stale fork, a byte-identical duplicate loss module, a planned 9-registry plugin architecture of which 1/9 exists and is never queried, a test suite where 40% of tests import no project code and two files fail to collect, and no packaging, CI, README or serving path.

**Root cause of the non-learning (verified by two independent readers).** `losses.py:259` scales a raw BTC price (~110,000) with the *delta* scaler (mean −4.16, std 261) → ≈421; `losses.py:278-280` `log(cosh(421))` overflows to inf, is clipped to 10.0 (the constant `1.333295` seen in every epoch), and TF's backward computes `0 × sinh(421) = NaN`; `model.py:2551-2560` `clip_by_global_norm` then rescales every gradient in the group to NaN on step one. Nothing guards gradients. One bug explains every observed symptom. A second, independent defect — the input is a globally z-scored price *level* (`model.py:528-533`) — would make the model learn "repeat last close" even after the NaN is fixed.

**Decisions already made by the owner.** Complete all 9 registries as specified in `REGISTRY_SPECIFICATIONS.md`; stay on TF 2.10 / Keras 2; keep the six physics-inspired loss terms — *fix the math and iterate until they demonstrably provide value*; leave git history alone.

**Outcome.** A model that provably trains; an installable `neural_trade` package with the nine registries actually wired into the training path; a typed config; a serving API; an evaluation protocol with baselines, purged splits and calibration on its own split; a run-tracked experiment loop and an ablation harness that answers the physics-term question with evidence; an honest backtest engine; a test suite where every test touches project code; CI.

**Assumptions stated.** Full-scale runs happen on the owner's Windows GPU box (TF 2.10 native GPU); CI runs on Linux CPU with `tensorflow-cpu==2.10`. Single flat `Config` dataclass (not nested) to preserve the hundreds of flat reads. No stub components are registered — spec'd components with no real implementation (`lstm_transformer`, `conv_net`, `binance_api`, `postgres`, `wavenet_causal`, `squeeze_excitation`) are listed as deferred in each registry's docstring.

---

## Phase 0 — Foundation (unblocks everything; 1 PR)

| # | Change | Files |
|---|---|---|
| 0.1 | `pyproject.toml` with `[tool.pytest.ini_options]`: `pythonpath=["src","."]`, `testpaths=["tests"]`, `addopts="--strict-markers -ra"`, markers `tf`, `slow`, `gpu`, `data`, `notebook` | new |
| 0.2 | `tests/__init__.py` (new) + keep `tests/registries/__init__.py` → becomes `tests.registries`, no longer shadows the real `registries` package. `tests/conftest.py`: `tf` fixture (lazy import, skip if absent), `tiny_config`, `synthetic_close` (seeded random walk, 3,000 bars @110,000), `synthetic_bars`, `real_slice` (`nrows=5000` of the committed CSV, marker `data`), `realistic_scales = (pred_scale=261.0, pred_mean=3.2, last_close=110_000)`; auto-add `tf` marker to modules importing tensorflow. Remove `sys.path.insert` at `tests/test_model_math_consistency.py:20`, `tests/test_loss_functions_exhaustive.py:677-679` | new/edit |
| 0.3 | Delete `registries/model.py` (stale fork; nothing imports it) and `losses_old_backup.py` | delete |
| 0.4 | `.gitattributes`: `* text=auto`, `*.py *.md *.yml *.yaml *.toml text eol=lf`, `*.ipynb filter=nbstripout`, `*.csv -text -diff`, `*.h5 *.joblib binary`; one `git add --renormalize .` commit (all `.py` are CRLF today). `.pre-commit-config.yaml`: ruff, nbstripout, `check-added-large-files --maxkb=5000` | new |
| 0.5 | `requirements.txt` = pip block of `nt.yml` verbatim; `environment.yml` = `nt.yml` minus `prefix:`; `requirements-ci.txt` (`tensorflow-cpu==2.10.1 numpy==1.23.5 protobuf==3.19.6 pandas==2.0.3 scikit-learn==1.3.2 scipy==1.10.1 pyyaml pytest pytest-cov pytest-timeout`) | new |
| 0.6 | `.github/workflows/ci.yml`: job `unit` (py3.10, `pytest -m "not gpu and not slow" --timeout=600`, coverage xml artifact); `nightly.yml` (`-m "slow or data"` + smoke ablation, added in Phase C) | new |
| 0.7 | Delete the stale local `nn_learnable_indicators_v3.weights.h5` if present (holds NaN weights; `force=True` still loads it at `model.py:1531-1535`) | local |

**Gate:** bare `pytest` from the repo root collects and runs the existing 5 files; CI green.

---

## Phase A — Correctness (edit `losses.py` / `model.py` in place, BEFORE any move)

All lambdas, gates and columns below refer to `indicator_params_history.csv` (the 263-channel oracle) and the `inference.ipynb` analytics printout.

### A1 — Milestone M1: a gradient step is possible

| Step | File:line | Change | Gate |
|---|---|---|---|
| S1 | `losses.py:217-218, 236-237, 253-254, 279-280, 295-296` | Add `_logcosh_safe(x) = x + softplus(-2x) - log 2` (gradient = tanh, bounded); replace every `log(cosh)`+`clip_by_value(-10,10)` pair | T1b: `_logcosh_safe(500.)` finite, grad == 1.0 |
| S2 | `losses.py:244-316`, call sites `:680-692` | Rewrite `extended_trend_loss` in consistent **scaled-delta** units: the price head is *already* a scaled delta → no re-scaling, no `last_close` subtraction; compare `y_pred_scaled` with `_to_scaled_static(extended_trends[:, k])` for horizon k (pass `horizon_idx=0/1/2`); drop the `start_of_window` "global" half (input-scaler value converted with the target scaler — meaningless) and the "multi-scale" sub-term (`:298-301`, identically 1.0, zero gradient). Delete the three `local_trend_loss` calls (algebraically the point loss; `log_local_h0 == log_global_h0` to 7 s.f. in the CSV). Fill the 6 local/global `LossComponents` slots with `tf.constant(0.0)` — **keep the 34-field contract** (positional unpacks at `model.py:2522-2534, 2866-2878, 1291-1322`) | `log_extended_h*` ∈ (0,3), ≠ 1.333295, train ≠ val |
| S3 | `model.py:2537` (after `tape.gradient`), dict at `:2689` | Finite-gradient guard: `finite = is_finite(total) & all(is_finite(g))`; `grads = [tf.where(finite, g, zeros_like(g))]` (**`tf.where`, not `g*mask`** — NaN×0 = NaN); `keras.metrics.Sum` counter `nonfinite_grad_steps` returned in logs; also return `grad_global_norm` (pre-clip) | new column `log_nonfinite_grad_steps == 0`; T2 |
| S4 | `model.py:123, 1886-1923, 2571-2575` | `MOMENTUM_CLIP_MIN = 2.0`; `clip_learned_periods` clips in **logit** space (`lo=_logit_from_period(max_p)`, `hi=_logit_from_period(min_p)`, `v.assign(clip(v, lo, hi))`); remove the `try/except: pass` around the call. Reason: period 1 → α=1 → logit 18.42 → float32 `sigmoid == 1.0` exactly → zero gradient forever | all 18 period columns finite; ≥12 change between epoch 0→1 |
| S5 | `model.py:1242-1268, 1487-1490` | Calibration pass: save the 13 originals in a dict; restore them in the `except` (today a throw after the reset leaves all 13 at 1.0 while printing "proceeding with default lambdas"); rename the misnamed "BN warmup" (no BatchNorm exists; keep the loop — it builds `model.losses`) | T3 |
| S6 | `tests/test_custom_loss.py`, `tests/test_train_smoke.py` | T1 `test_all_34_components_finite[scales]` at realistic scales (all fields finite; `extended_h0` not the clip constant; all grads w.r.t. `trainable_variables` finite); T2 3-step fit on `synthetic_close` → all weights finite, `nonfinite_grad_steps==0`, periods moved, `evaluate(val)` differs before/after; T3 calibration restores on failure. **All three fail on HEAD** | CI |

**M1 stop/go:** `train_and_evaluate(force=True, epochs=2)` twice — (a) physics lambdas overridden to 0, (b) defaults. GO when both show: `log_val_loss` changes between epochs; `log_nonfinite_grad_steps == 0`; periods finite and moving; `log_extended_h*` unpinned; `log_train_pred_up_rate_h1 ∉ {0,1}`; `log_val_dir_mcc_h1 ≠ 0.0` exactly; in (b) `log_casimir_loss > 0` (price heads non-zero).

### A2 — Milestone M2: every loss term bounded and pulling the right way

| Step | File:line | Change |
|---|---|---|
| S7 | `model.py:2313-2334`; `losses.py:820,831,840,848,861,867,873-874`; calibration writes `:1256-1268, 1397-1409` | The 14 per-term lambdas become `tf.Variable(trainable=False)` (realised as `LossWeights` in Phase B §B2; `CustomTrainModel.lambda_*` become properties whose setter calls `.assign`). In `losses.py` replace `tf.constant(float(getattr(model,'lambda_x')))` with `tf.cast(getattr(model,'lambda_x'), tf.float32)` (`float(Variable)` inside the traced step raises). Attach under `_setattr_tracking=False` so Keras does not add 14 variables to the H5. Reason: every lambda is frozen into the graph at trace time; scheduling/ablation is impossible today |
| S8 | `losses.py:383-405` t_perp | `r2 = stop_gradient((y-mu)²)`; `return (log(mean r2 + eps) − log(mean var + eps))²`. Today `(std r − mean √var)²` is zero only for *constant* σ (Jensen) — it rewards the collapse observed — and leaks gradient into μ |
| S9 | `losses.py:408-441` casimir | Soft sign `s(p)=tanh(p/0.5)`; `interf = stop_gradient(relu(−s(p_a)s(p_b)))` ∈ [0,1] (no gradient into price heads); penalty `relu(log v_ref − log avg_var)` with `v_ref=1.0` → bounded by `log(1/VAR_FLOOR)≈9.2`. Today `relu(−p_a p_b)/avg_var` scales as |p||p| ≤ 1e4 with unbounded 1/var, inflating σ and shrinking price heads |
| S10 | `losses.py:482-511` hd | `1 − Pearson(z(log-vol), z(log-σ²))` ∈ [0,2] with `stop_gradient` on the data statistic. Today `−mean(local_vol·σ)` is a linear, unbounded bounty on σ mixing input-scaler and target-scaler units |
| S11 | `losses.py:550-551`; `model.py:104-110, 1392-1395` | IFE: linear hinge (drop the squares). Add `CALIB_DAMPING_PHYSICS = 0.0` and `CALIB_DAMPING_TREND = 0.0`: hinge/constraint terms are excluded from magnitude equalisation (IFE is bounded ≤0.005 by construction, so the calibrator pins it at `CALIB_LAMBDA_MAX=20` every run) |
| S12 | `losses.py:587`; `model.py:2861-2863` | vac_overflow: `stop_gradient(residual_mag)` (today rewards worse predictions when mean overflow > mean residual); `test_step` passes `vacuum_overflow=None` (overflow ≡ 0 at eval, so the term is a constant λ in every `val_loss`) |
| S13 | `losses.py:761-764`; `model.py:2613, 2907, 1617-1619` | `var_c = maximum(var, VAR_FLOOR)` in the loss (drop the cap — `clip_by_value` zeroes every σ-gradient at `VAR_CAP=1e3`, which is where the heads froze); unify the `1e4` literals to `cfg.VAR_CAP` in the metric paths |
| S14 | `losses.py:734, 794-802, 889` | Remove the coherence double count (0.01× inside `trend_loss_val` and 1.0× outer); skip the dead `dir_align` compute unless `lambda_dir_align_outer > 0` (also buggy: BCE on 1-D args returns a scalar so its mask is a no-op) |
| S15 | `model.py:118, 164-168, 173` | Defaults while iterating: `LAMBDA_EXTENDED_TREND=0.1`, `LAMBDA_T_PERP/CASIMIR/HD/IFE/VAC_OVERFLOW = 0.1`, `LAMBDA_VAC=0.0`; keep `LAMBDA_VAR=LAMBDA_CRPS=1.0`. Net σ-force is then calibrating (two proper scoring rules at 1.0 vs four bounded regularisers at 0.1) |
| S16 | `model.py:2832-2833, 2661-2687, 3519, 2640-2648, 2936-2944` | ECE → positive-class convention (`bin_pos = Σ true_dir·in_bin`), matching `soft_ece_loss`; PIT-KS computed in-graph via sorted `Φ((y−μ)/σ)` vs ECDF in **both** train and test steps (today `.numpy()` inside the traced step is swallowed → always NaN); `convergence_score = nan if not isfinite(mean)` (today `min(1.0, nan)` → 1.0); delete the never-returned `trend_margin/agreement_rate/magnitude_bps` computations |
| S17 | `tests/test_physics_terms_bounded.py` | T4 `assign(0.0)` changes the next `train_on_batch` without retrace; T5 bounds/directions (hd ∈ [0,2]; casimir ∈ [0, 9.2]; t_perp minimised at correct heteroscedastic σ and `< constant σ`; `grad(vac_overflow, price_h*)` zero); T6 ECE on `p≡0.05`, 5% positives → `< 0.02` (HEAD ≈ 0.9); `clip_learned_periods(2,60)` leaves `sigmoid'(logit) > 0.1` |

**M2 stop/go (5 epochs):** all physics columns finite, ≥0, non-constant; `log_val_vac_overflow_loss == 0`; `val_nll` and `val_crps` improve over epoch 0; `log_val_pit_ks_h1 < 0.2`; in the analytics printout `corr(var_scaled, error_scaled²) > 0.10` for h1 (HEAD 0.032/−0.039/−0.002) and `std(var_h1)/mean(var_h1) > 0.05`. No skill requirement yet.

### A3 — Milestone M3: the input can carry the signal

| Step | File:line | Change |
|---|---|---|
| S18 | `model.py:529-537, 1547-1548` | Window-relative input in target-scaler units: `X_scaled = (X − last_close[:,None]) / target_scaler.scale_[0]`; `input_scaler=None` (parameter-free). Puts window, `last_close`, extended trends, price heads and σ in one unit system. Indicators are affine-covariant/invariant (EMA/MACD/BB covariant, RSI/%B invariant); only `tanh(macd_hist*10)` at `:1811` sees O(1) inputs (acceptable). Placed here, not M1, so M1/M2 gates are attributable |
| S19 | `model.py:217` | `DIR_DEADBAND_BPS = 5.0` (the comment block at `:205-217` documents 5 bps as the label-noise fix; the value is 0 → every ±$1 tick is a hard label) |

**M3 stop/go (20 epochs):** analytics `EV (delta)` for h1 > 0 (HEAD 0.0000); `ROC-AUC` h1 > 0.52 on 7,200 rows (≈3 SE); `Total Trades > 0`; CSV best-epoch `log_val_dir_mcc_h1 > 0.02` and `log_val_gauss_dir_mcc_h1 > 0`. If M3 fails while M2 passed, the problem is representation/capacity — do not go back to tuning physics lambdas.

### A4 — Milestone M4: metrics and calibration can be trusted

| Step | File:line | Change |
|---|---|---|
| S20 | `model.py:505-519, 1176-1202, 1537-1539, 1553-1619, 1641-1663`; `calibration/pipeline.py:115-142` | **Four-way purged split** (see §C2): `train / val / cal / test` with gap = `LOOKBACK + max(HORIZON_STEPS) = 80` sequences (exactly the 79-sequence leak + 1). `val_ds` from **val** (today `val_ds` = the test set — EarlyStopping/Checkpoint/LR-plateau select on test, `model.py:1199-1202`); `CalibrationPipeline.fit(result, split='cal')` — refuse `split='test'` (today fits on `result.y_test`, the same arrays that produce the reported metrics); extract `_predict_heads(model, X, ...)` from `:1553-1619`; apply the fitted pipeline to test predictions → `TrainResult.predictions_calibrated` (the pipeline finally has a consumer); print conformal coverage **on test** |
| S21 | `model.py:72-73, 1496, 1504, 1509, 1518` | `EARLY=6`, `PATIENCE=3` (today both `=EPOCHS` → early stopping, MCC stopping, LR decay all inert); read `cfg.*` not `Config.*` (the three class reads silently ignore `config_overrides`); drop the second racing `EarlyStopping` (`es_dir`, no restore) |
| S22 | `model.py:620-622, 701-711, 623` | `_compute_all_horizon_metrics` uses `compute_direction_labels_np` with the neutral mask (today a delta-space threshold with **no** mask, so notebook acc ≠ trained `val_dir_acc`); declare `DELTA_MAPE_MIN_ABS = 1.0` (read at `:623`, never declared) |
| S23 | `model.py:3573-3575` | Guard `df.to_csv` (a `PermissionError` with the CSV open in Excel aborts training) — superseded by the JSONL logger in §C3 |
| S24 | `model.py:1-6, 1164` | `os.environ.setdefault("TF_DETERMINISTIC_OPS","1")` **above** `import tensorflow`; do **not** enable `enable_op_determinism()` here (GPU `UnimplementedError` risk mid-training) |

**M4 stop/go:** test coverage at α=0.1 ∈ [0.87, 0.93] all horizons; early stopping fires on a plateaued run; bare `pytest` green.

**Phase A does not touch:** `registries/`, `core/`, `math_helpers.py`, `metrics_utils.py`, notebooks, package layout, the 54 broad excepts (except S4/S5), the `LossComponents` field list, `TimeSeriesSplit` itself beyond S20.

---

## Phase B — Package, config and the nine registries

### B1 — Layout decision: single installable package `src/neural_trade/`, thin root shims until B17

Reasons: `tests/registries/` + no `tests/__init__.py` made `from registries.losses import …` resolve to the test package; generic top-level names (`core`, `models`, `data`, `registries`, `training`) will keep colliding; nothing is installable so every test/notebook does `sys.path.insert`; `registries/model.py` exists because flat files invite copy-instead-of-import. Root `model.py` becomes `from neural_trade.compat import *` with an explicit `__all__` of the names `inference.ipynb`/`diagnostics.ipynb` use; `core/`, `registries/`, `calibration/`, `losses.py`, `metrics_utils.py`, `math_helpers.py` become `sys.modules`-aliasing shims.

```
pyproject.toml  requirements*.txt  environment.yml  README.md  LICENSE  .gitattributes  .pre-commit-config.yaml
configs/default.yaml  configs/ci.yaml  configs/ablation_physics.yaml  configs/ablation_criteria.yaml
src/neural_trade/
  __init__.py (no TF import)   compat.py   cli.py
  core/        registry.py  exceptions.py  config.py  outputs.py (PredictiveOutputs←model.py:1950, LossComponents←losses.py:23-39)  plugin_loader.py  logging.py
  registries/  __init__.py (explicit discovery, ALL_REGISTRIES, load_all, print_registry_summary, validate_config_components)
               losses.py models.py optimizers.py metrics.py callbacks.py data_loaders.py visualizations.py layers.py preprocessors.py
  losses/      functions.py ← losses.py:127-924 (git mv as ONE file so Phase A edits rebase cleanly)
  models/      gru_attention.py ← model.py:1965-2218;  layers/{learnable_indicators←1682-1922, positional_encoding←1924-1940, vacuum_saturation_noise←2240-2294, energy_gate←2004-2037}
  data/        loaders.py  preprocessors.py ← model.py:306-367  windowing.py ← 376-460 (+ make_inference_windows, target-free)
               scaling.py (TargetScaler; WindowNormalizer: per_lag_standard ← 529-534, window_relative from S18)  splits.py (§C2)  datasets.py ← 486-527, 2220-2235  processor.py (DataProcessor facade)
  metrics/     numpy_metrics.py ← metrics_utils.py (+mse/rmse/mae/ev/corr/r2/mcc/brier/ece_pos/pit_ks/coverage)  tf_direction.py ← model.py:2741-2853  direction_labels.py (np←metrics_utils:163, tf←losses.py:596; the 4 inline copies at model.py:621, 2591, 2885, calibration/pipeline.py:68 delegate here)  evaluate.py ← model.py:578-719
  training/    lambdas.py (LossWeights)  optim.py (OptimizerPair)  custom_model.py ← model.py:2297-3042  lambda_calibration.py ← 1206-1490  callbacks.py ← 3366-3616 + JsonlEpochLogger  trainer.py ← 1143-1205, 1492-1679 (Trainer, train_and_evaluate, TrainResult←548-567)  artifacts.py (ArtifactBundle)
  visualization/ aliases.py ← 3044-3068, 3106-3365  plotly_training.py (figure half of 722-1140)  qbox_dashboard.py ← 3070-3103  matplotlib_splits.py ← 462-484  plotly_trading.py, indicator_evolution.py ← inference.ipynb cells 9/12  eval_plots.py
  calibration/ ← calibration/ (git mv, unchanged)
  serving/     predictor.py  postprocess.py (sanitize_heads ← model.py:1605-1619)
  strategy/    signals.py (MultiHorizonSignal ← inference.ipynb ~5133-5387; helpers ← trade.ipynb)  trades.py (Trade ∪ EnhancedTrade)  strategies.py  backtest.py  performance.py  params.py
  evaluation/  report.py (PredictionFrame, EvalReport, evaluate)  baselines.py  walk_forward.py
  experiments/ run_context.py  compare.py  ablation.py  runner.py
  telemetry/   epoch_logger.py
  utils/       math.py ← math_helpers.py (the 2 in-class logit/period copies at model.py:1695-1708, 2369-2377 delegate here)  seeding.py
plugins/  __init__.py  README.md  templates/{model,loss,metric}_plugin_template.py (not imported)  examples/echo_metric.py (real; tested)
notebooks/ 01_train_and_monitor  02_backtest  03_signals_and_trades  04_diagnostics   (thin; no def/class)
scripts/  ablate.py  check_coverage.py
tests/    __init__.py conftest.py  test_*.py  registries/{__init__,test_contracts,test_<each of 9>}.py  reference/physics_np.py
```

### B2 — Config redesign (`core/config.py`)

One flat `@dataclass Config`; grouping via `field(metadata={"group": …})` (drives grouped YAML and docs), **not** nested sub-dataclasses (hundreds of `cfg.X` / `getattr(config,'LAMBDA_…')` reads exist).
- All 78 existing names and defaults preserved verbatim; `HOUR`/`DAY` become `ClassVar`; mutable lists via `default_factory` (today shared across instances); `PATIENCE/EARLY/MOMENTUM_CLIP_MAX: Optional = None` resolved in `__post_init__` (so any surviving `Config.EARLY` class read fails loudly).
- New registry keys: `MODEL_NAME="gru_attention"`, `OPTIMIZER_NAME="adam"`, `INDICATOR_OPTIMIZER_NAME="adam"`, `LOSS_NAME="custom_loss"`, `METRICS: List[str]` (numpy tier), `STEP_METRICS: List[str]` (tf tier, the 11 direction metrics), `CALLBACKS: List[str]` (= today's `model.py:1518` order), `DATA_LOADER="csv"`, `PREPROCESSORS: List[str]`, `VISUALIZATION="plotly_interactive"`, `LAYERS: Dict[str,str]` (role → component; the Layers key the spec never named), `WINDOW_NORMALIZER="window_relative"`.
- New real optimizer params the spec assumed: `ADAM_BETA1/2/EPSILON`, `WEIGHT_DECAY`, `SGD_MOMENTUM`, `SGD_NESTEROV`. New ops keys: `ARTIFACTS_DIR`, `PLUGINS_DIR`, `LAMBDA_SCHEDULE`, `LAMBDA_ABLATE`, `CALIB_DAMPING_PHYSICS`, `DELTA_MAPE_MIN_ABS`.
- Methods: `validate()` raising `InvalidConfigurationError(RegistryError, ValueError)` (existing `pytest.raises(ValueError)` tests stay green); `override(**kw)` (unknown key → error with difflib suggestion; re-validates; replaces `_apply_config_overrides` `model.py:570-575`); `to_dict/from_dict` with per-field type coercion (PyYAML parses `1e-3` as str); `from_yaml`/`to_yaml` with **flat keys = field names** (the spec's `_flatten_dict` produces `OPTIMIZER_LEARNING_RATE` etc. and cannot load its own `config.yaml`); `lambda_weights()`.
- `training/lambdas.py::LossWeights` — the 25 `LAMBDA_*` as `tf.Variable`s; `CustomTrainModel.lambda_*` properties (getter → Variable, setter → `assign`) so `losses/functions.py` reads and the calibration-pass writes are unchanged; applies `LAMBDA_ABLATE`; persisted in `meta.json`, not the H5.

### B3 — BaseRegistry hardening (`core/registry.py`, `core/exceptions.py`)

- `__init_subclass__` auto-creates `registry = {}` per subclass (today `registry: Dict = {}` at `:84` is shared unless redeclared; no test covers forgetting).
- Exceptions made live: `ComponentNotFoundError(RegistryError, KeyError)` from `get`/`get_metadata` (existing tests matching `KeyError, "not found"` pass); `InvalidConfigurationError(…, ValueError)`; `ComponentValidationError`/`DuplicateRegistrationError` raised in strict mode; `DependencyError` from `get()` when an entry's `dependencies` are not importable (`find_spec`); `RegistryNotInitializedError` when a declared discovery module fails to import.
- `strict: bool = False` on the base (the test `TestRegistry` keeps warn-and-register), `strict = True` on the nine real registries; env `NEURAL_TRADE_STRICT_REGISTRY=1` in CI. `validate_component` returns bool everywhere — resolves the spec's raise-vs-bool conflict once.
- `default: Optional[str]`, `get_default()`, `resolve(name)`; `build(name, *a, **kw)` (always calls; `get(name, **kw)` keeps its tested call-if-kwargs behaviour as legacy).
- `auto_discover()` made real: `discovery_modules: Tuple[str,...]` imported explicitly; **no glob** (the spec's glob would import the stray fork). `plugin_loader.load_plugins(dir, strict=False)` wired only through `registries.load_all(config)` at Trainer/Predictor/CLI startup — never at import time.

### B4 — The nine registries

| # | Registry / key | Enforced contract | Initial components (all real) | Where `.get/.build` is called |
|---|---|---|---|---|
| 1 | `Models` / `MODEL_NAME` | callable; first param `config`; built model validated by `ensure_predictive_outputs` (10 outputs named per `PredictiveOutputs._fields`) | `gru_attention` ← `build_model` (uses `Layers.build(config.LAYERS["indicators"], …)`) | `Trainer.run`, `Predictor.from_artifacts`, `CustomTrainModel.from_config` |
| 2 | `Optimizers` / `OPTIMIZER_NAME` + `INDICATOR_OPTIMIZER_NAME` | callable; first param `config`; accepts `learning_rate=` kw | `adam`, `adamw`, `sgd_momentum`, `rmsprop`, `nadam` | `training/optim.py::build_optimizers → OptimizerPair(main, indicator@LR·INDICATOR_LR_MULT)`; replaces `model.py:2340-2341, 1492`. Clipping stays in `train_step` (single site) |
| 3 | `Metrics` / `METRICS` (numpy) + `STEP_METRICS` (tf) | first two params `y_true, y_pred`; tf tier registered via `register_tf` (graph-safe; a source scan forbids `.numpy()`) | numpy 16: mse rmse mae explained_variance corr r2 safe_mape smape wape direction_accuracy direction_f1 mcc brier ece_pos pit_ks coverage; tf 11 ← `_compute_direction_metrics`. Groups `REGRESSION_METRICS`, `DIRECTION_METRICS`, `STEP_METRICS`. Sharpe/drawdown are **not** metrics (they take an equity curve) → `strategy/performance.py` | `metrics/evaluate.py` loops `cfg.METRICS`; `CustomTrainModel.__init__` resolves `STEP_METRICS` **once** (no registry access inside `tf.function`) |
| 4 | `Callbacks` / `CALLBACKS` | `(config, context: TrainContext) → Callback \| list` | `csv_logger`, `early_stopping`, `model_checkpoint`, `mcc_early_stopping`, `reduce_lr_on_plateau`, `tqdm_progress`, `params_logger`→`jsonl_epoch_logger`, `interactive_plot` (deps ipywidgets/IPython/plotly), `lambda_schedule`, `metric_threshold`, `tensorboard` | `training/callbacks.py::build_callbacks` replaces `model.py:1495-1520` |
| 5 | `DataLoaders` / `DATA_LOADER` | first param `config`; result validated by `validate_ohlcv_frame` | `csv`, `parquet` (dep pyarrow), `dataframe` (in-memory; tests + `Predictor.predict_frame`) | `Trainer.run`. Deferred: `binance_api`, `postgres`, `ccxt` (no code exists) |
| 6 | `Visualizations` / `VISUALIZATION` | first two params `data, config` | `plotly_interactive` (figure half of 722-1140), `qbox_dashboard_html`, `matplotlib_splits`, `plotly_trading`, `indicator_evolution`, `eval_report` | `interactive_plot` callback; notebooks; `cli backtest --plot` |
| 7 | `Layers` / `LAYERS` (role map) | `Layer` subclass or factory accepting `**kwargs`; instance check at build | `learnable_indicators`, `positional_encoding`, `vacuum_saturation_noise`, `energy_gate` | inside `build_gru_attention`; `Layers.as_custom_objects()` for loading. Deferred: `wavenet_causal`, `squeeze_excitation` |
| 8 | `Preprocessors` / `PREPROCESSORS` | first two params `df, config`; returns DataFrame | `standardize_ohlcv`, `sort_dedupe`, `resample_bars`, `drop_missing_close`, `strip_currency_symbols`, `log_returns`, `add_time_features`. df-level scalers **not** registered (scaling before the split leaks test statistics; it lives in `data/scaling.py` at the window stage) | `data/preprocessors.py::run_preprocessors` |
| 9 | `Losses` / `LOSS_NAME` | two tiers in one registry: **component** (first param `model`) and **objective** (`register_objective`: exact params `(model, x_window, y_true, y_pred, last_close, extended_trends, *, vacuum_overflow)` → `LossComponents`) | the 16 existing; `custom_loss` re-registered under the objective tier | `Trainer → Losses.get_objective(cfg.LOSS_NAME) → CustomTrainModel(objective=…)`; `train_step/test_step` call `self.objective(...)`; the 7 delegating methods at `model.py:2423-2508` become lookups bound once in `__init__`. **The registry stops being write-only** |

Honest component count at completion: ~77 (Models 1, Optimizers 5, Losses 16, Metrics 27, Callbacks 11, DataLoaders 3, Visualizations 6, Layers 4, Preprocessors 7) — ~55 extracted from existing code, ~22 small new ones each with a test. `tests/registries/test_contracts.py` monkeypatches `.get` and asserts every registry is hit during a 1-epoch smoke run.

### B5 — Serving API (`serving/predictor.py`, `training/artifacts.py`)

Persist at train time what is missing today: `pred_scale/pred_mean` (`model.py:1183-1184`, never saved), the window normalizer, a config snapshot, the fitted `CalibrationPipeline` (`save()` at `pipeline.py:321` is never called), calibrated lambdas, version + git sha. `ArtifactBundle.save/load(dir)`; `data/windowing.py::make_inference_windows` (target-free; today `make_sequences_with_extended_trends` requires future bars); `Predictor.from_artifacts(dir).predict(close_window) → Prediction` (delta/direction_prob/variance per horizon + conformal intervals via `CalibrationPipeline.apply` at `pipeline.py:218`, finally consumed); `predict_frame(df)`, `predict_last(close)`. There is no `joblib.load` anywhere in the repo today.

### B6 — Migration order (each row leaves `pytest` green)

| Step | PR | model.py lines left | Tests |
|---|---|---|---|
| B0 | = Phase 0 | 3665 | existing collect |
| B1 | `git mv core/*.py → src/neural_trade/core/`; root `core/` shim; `src/neural_trade/__init__.py`; full `pyproject` (§B8); `pip install -e .`; CI | 3665 | `test_registry_base.py` unchanged |
| B2 | BaseRegistry hardening (§B3) | 3665 | `registries/test_base_hardening.py` |
| B3 | `git mv losses.py → losses/functions.py`; registry class → `registries/losses.py`; root `losses.py` + `registries/losses.py` shims; `tests/test_losses.py:53` tripwire → `assert Losses.get('point_huber') is neural_trade.losses.point_huber` | 3665 | existing loss tests |
| B4 | Typed Config (§B2); `model.py` imports it; `cfg.override` | ~3420 | `test_config.py` (kwargs ctor, asdict, unknown-key error, yaml round-trip incl. `1e-3`, list isolation, derived PATIENCE/EARLY) |
| B5 | `utils/math.py`, `metrics/numpy_metrics.py`, `metrics/direction_labels.py`; delete the 2 logit/period copies and 4 inline label copies | ~3360 | `test_direction_labels.py` (np/tf/inline agree incl. deadband edge) |
| B6 | 3 layers + `core/outputs.py`; `registries/layers.py` | ~3040 | `registries/test_layers.py` |
| B7 | `models/gru_attention.py`; `registries/models.py`; `PricePredictor` facade | ~2760 | `registries/test_models.py` (10 outputs/names at LOOKBACK=32) |
| B8 | Metrics tiers; `metrics/evaluate.py` | ~2500 | golden test vs frozen `_compute_all_horizon_metrics` output |
| B9 | `data/*` incl. `splits.py` (§C2); two registries; `DataProcessor` facade | ~2250 | `test_windowing.py`, `test_data_processor.py`, `registries/test_{preprocessors,data_loaders}.py` |
| B10 | `training/custom_model.py`, `lambdas.py`, `optim.py`, `registries/optimizers.py`; then `lambda_calibration.py`; then `trainer.py`, `artifacts.py` | ~950 (<1000 reached) | `registries/test_optimizers.py`, `test_loss_weights.py`, `test_trainer_smoke.py` (`tf slow`) |
| B11 | `training/callbacks.py` + registry; `JsonlEpochLogger`/`RunContext` (§C3); delete `SimpleLoggingCallback` (never constructed) | ~700 | `registries/test_callbacks.py`, `test_telemetry.py` |
| B12 | Visualization split + registry; `make_interactive_plot_callback` = compat wrapper | <100 (shim) | `registries/test_visualizations.py` |
| B13 | `git mv calibration → src/…`; `serving/`; Trainer writes the full bundle | — | `test_artifacts_predictor.py` (train→save→load→predict identical) |
| B14 | `evaluation/` (§C2), `strategy/` + backtest engine (§C5) | — | `test_eval_metrics.py`, `test_backtest.py`, `test_strategy.py` |
| B15 | `experiments/` ablation harness (§C4); `ewma_sequence_matrix` (§C4) | — | `test_ablation_harness.py`, `test_ewma_matrix_equals_scan` |
| B16 | CLI + README; `print→logging` and `except: pass` sweeps (ruff `T201 E722 S110 BLE001`); LICENSE | — | `test_cli.py` |
| B17 | Notebooks rewritten thin; delete `cfg.ipynb` and the duplicate liberal-strategy cell; nbstripout | — | `test_notebooks_thin.py` (AST: no top-level def/class) |
| B18 | Remove root shims; tests import `neural_trade.*` directly | 0 | all |

Sweep rules: every function converts `print`→`logger` when it moves (no `print` survives in `src/`); each of the 14 `except …: pass` is handled when its block moves — essential ops (`load_weights` 1531-1535, `joblib.dump` 1545-1550, `clip_learned_periods` 2571-2575, indicator-var scan 2354-2363) re-raise or `logger.exception`; cosmetic ones lose the `try`; UI guards become `logger.warning(exc_info=True)`.

### B7 — Notebook policy
`inference.ipynb → notebooks/01_train_and_monitor` (+ its viz cells → `03_signals_and_trades`); `trade.ipynb → 02_backtest`; `diagnostics.ipynb → 04_diagnostics`; `cfg.ipynb` deleted. Notebooks define no `def`/`class` (enforced by test). Outputs stripped via the nbstripout filter.

### B8 — Packaging & tooling
`pyproject.toml`: `name="neural-trade"`, `requires-python=">=3.10,<3.11"`, deps `numpy>=1.23,<1.24 pandas scikit-learn scipy joblib pyyaml tqdm protobuf<3.20`; extras `tf` (`tensorflow==2.10.0`), `tf-cpu`, `viz`, `notebooks`, `dev`; script `neural-trade = neural_trade.cli:main`; ruff config. CLI: `train --config … [--set K=V]`, `predict --artifacts DIR --csv FILE`, `backtest --artifacts DIR --csv FILE --strategy …`, `registry list|info|search`, `env`. CI jobs: `lint`, `unit-no-tf` (`-m "not tf"`), `unit-tf` (`-m "tf and not slow"`), `smoke-train` (1 epoch on synthetic + `predict`); nightly: `-m "slow or data"` + smoke ablation.

---

## Phase C — Validation and the experiment loop

### C1 — Test suite triage (161 → every test touches project code)

| File | Decision |
|---|---|
| `tests/test_pipeline_integrity.py` (26) | **Delete**; move `create_synthetic_ohlcv_data` into `conftest.py`. Zero project imports — cannot fail from any project change |
| `tests/test_loss_functions_exhaustive.py:1-670` (38 numpy re-implementations) | **Rewrite** as `tests/test_losses_reference.py`: independent numpy reference vs the *production* TF function, 5 seeds, `rtol=1e-5`; delete classes testing math the code does not contain (`TestTrendLossExhaustive`, `TestInterconnectionLossExhaustive`, `TestVolatilityMatchingExhaustive`, `TestCompositeLossExhaustive`) |
| `:676-867` (15 T⊥ tests) | **Keep**; import `neural_trade.losses` (today `registries.losses`, the copy production never imports); update expectations to the Phase A forms (`test_hd_coupling_zero_volatility`, `test_casimir_*`) |
| `tests/test_model_math_consistency.py` (47 + 4 stubs) | **Keep 12** structural `validate()` negatives → `test_config.py`; **delete** the opinion tests on constants (`*_reasonable`, `_geometric`, `_balanced` …); **implement** the 4 skip stubs at `:531-557` as `test_custom_loss.py::test_total_equals_weighted_sum_of_components`, `test_nll_exact_with_var_floor`, `test_vacuum_contrib_zero_when_lambda_vac_zero`, `test_calibration_prepass_records_effective_lambdas` |
| `tests/test_registry_base.py` (30) | Keep as-is |
| `tests/test_losses.py` (5) | Keep 3; rewrite the `__module__` tripwire to identity; `test_custom_loss_smoke_runs` → parametrised over `(1,0,1)`, `(261,3.2,110000)`, `(0.01,0,5)` |

New test files (markers): `test_custom_loss.py` (tf), `test_losses_reference.py` (tf), `test_train_step.py` (tf slow; incl. `test_determinism_two_runs_identical`), `test_learnable_indicators.py` (tf; `test_output_shape [B,60,31]`, `test_gradient_reaches_all_18_logits`, `test_clip_never_saturates`, `test_ewma_matrix_equals_scan`), `test_data_processor.py` (data; `test_purged_split_no_overlap`, `test_leak_count_without_gap_is_79`, `test_scaler_fit_on_train_only`, `test_window_relative_normalisation`), `test_build_model.py` (tf), `test_direction_labels.py` (tf), `test_eval_metrics.py` (`test_ece_positive_class_calibrated_is_zero` — current formula gives ≈0.25; `test_pit_ks_uniform_when_calibrated`; `test_ev_price_trap_is_not_reported`; `test_effective_sample_size`), `test_calibration.py` (`test_fit_refuses_test_split`, coverage, T>1 on over-confident probs, save/load), `test_telemetry.py` (`test_convergence_score_nan_is_nan`, `test_jsonl_logger_never_raises`), `test_predictor_round_trip.py` (tf slow), `test_backtest.py` (hand-computed PnL after 13 bps costs, TP/SL on high/low with `sl_first`, `test_no_lookahead_all_strategies`, `test_trailing_features_are_causal`, EOW mark), `test_ablation_harness.py`, `test_notebooks_thin.py`. Coverage gates via `scripts/check_coverage.py`: `losses` ≥85, `evaluation` ≥85, `strategy` ≥85, `calibration` ≥80, `data` ≥75, `core` ≥95, `training`/`models` ≥50.

### C2 — Evaluation protocol (`data/splits.py`, `evaluation/`)

- `make_purged_splits(n_seq, lookback, horizon_steps, n_folds=5, val_len=2880, cal_len=2880, gap=80) → list[FoldIndices(train, val, cal, test)]` over `TimeSeriesSplit(n_splits, gap=gap)`. Per fold: `test` = TSS block; `cal` = the `cal_len` sequences before it (−gap); `val` = the `val_len` before that (−gap); `train` = the rest. On the committed CSV (43,421 sequences) fold 5: test 7,236 / cal 2,880 / val 2,880 / train ≈30,185. Scalers and `pred_scale` fit on train only; early stopping on val; temperature/conformal/`var_scale` on cal; test touched once.
- `walk_forward(config, folds=(3,4,5), seeds) → list[EvalReport]`; folds double as the ablation's "periods".
- **Baselines reported in every report** (fit on train): `zero_delta` (persistence: EV(delta)=0, EV(price)≈0.999 — makes the trap visible), `mean_delta`, `class_prior`, `logreg_lags` (logistic regression on trailing returns over {1,5,10,15,20,30,60} bars + 60-bar vol), `const_var` (marginal error variance → CRPS/PIT/coverage), and backtest `buy_and_hold`, `always_flat`, `random_same_freq`. `EvalReport.beats_baseline[name][metric]`.
- Metric definitions (neutral mask applied to every direction metric): direction `mcc auc brier ece_pos acc bal_acc pred_up_rate true_up_rate` + `gauss_*`; delta `rmse mae ev corr skill_vs_zero` (**no price-space EV**); variance `crps` (closed-form Gaussian, dollars) `crpss nll pit_ks corr_var_err2_spearman coverage90 width90`; coherence `mag_order_full unanimity delta_dir_align_all coherence_primary`; confidence gap with block-bootstrap CI (block=80, split at the **cal** median; verdict `WORKS` iff CI excludes 0 and gap ≥1 pp — the committed 0.2 pp "✓ Confidence works" becomes `NOISE`); `n_eff_h = N // h_steps` (482 for h1) in every header.
- `evaluate(frame: PredictionFrame, config, split, run_id, baselines, backtest_cfg) → EvalReport` (`to_json`, `to_markdown`, `flat()`) replaces `inference.ipynb` cell 4 and `_compute_all_horizon_metrics`.

### C3 — Run tracking and telemetry (`experiments/run_context.py`, `telemetry/epoch_logger.py`)

`RunContext.create(config, seed, tags)` → `runs/<UTC-ts>-<sha>[-dirty]-<cfghash8>/` holding `config.yaml`, `env.txt`, `metrics.jsonl` (append-only per epoch; replaces the O(n²) CSV rewrite and `CSVLogger('training_log.csv', append=True)`), `indicator_params.jsonl`, `calibrated_lambdas.json`, `weights.h5` (best on **val**), `scalers.joblib` (with `pred_scale/pred_mean`), `calibration/`, `predictions_{cal,test}.parquet`, `eval_report_{val,test}.{json,md}`, `backtest/`, `tb/`, `status.json` (incl. `sec_per_step`, `n_errors`). `JsonlEpochLogger.on_epoch_end` is fully try/excepted (log, count, continue); computes PIT-KS and `ece_pos` on fixed ≤2,048-sequence train/val probe sets via `model.predict` (no `.numpy()` in the traced step); `convergence_score` is NaN on NaN. TensorBoard callback for ~60 curated scalars. MLflow: **no** for now — `RunTracker` protocol with `NoopTracker`/`TensorBoardTracker`; an `MlflowTracker` is a 40-line addition later. `compare_runs(run_ids|glob, split, metrics) → DataFrame` refuses runs without an eval report.

### C4 — Ablation harness for the physics terms (`experiments/ablation.py`, `scripts/ablate.py`, `configs/ablation_*.yaml`)

- Grid: `modes: [all_on, all_off, leave_one_in, leave_one_out]` over the six `LAMBDA_*` (14 conditions) × `seeds [0,1,2]` × `periods {P1: fold 4, P2: fold 5}` = **84 runs**; `--scale smoke` (`MAX_SEQUENCE_COUNT=5000, EPOCHS=3`, 2 runs in minutes on CPU, runs nightly) and `--scale full`. `lambda_source: fixed` and `calibrate=False` inside the grid (the pre-pass rescales every lambda by a shared reference, so toggling one term would silently change the others); optionally calibrate once on `all_on`, freeze, then ablate. `resume: true` skips completed `(condition, seed, period)`; `--dry-run` projects hours from the last measured `sec_per_step`.
- **Pre-registered criteria** (`ablation_criteria.yaml`): per term a primary metric (t_perp/casimir/hd/vac_overflow → `crps_h1` ↓ and `corr_var_err2_spearman_h1` ↑; vac → `coherence_primary` ↑; ife → `mcc_h1`/`auc_h1` ↑; family → all four incl. `sharpe_net`) and guard-rails with tolerances; paired deltas over the 6 (seed, period) pairs; verdict `VALUE` iff mean Δ > max(σ_seed, MDE) with ≥5/6 pairs agreeing and no guard-rail breach; `HARMFUL` / `NEUTRAL` / `INCONCLUSIVE (add seeds)` otherwise. Term verdict = VALUE only if both modes agree (or VALUE + NEUTRAL). Outputs `results.csv`, `summary.csv`, `report.md` with run ids.
- **Compute:** today `ewma_sequence` (`math_helpers.py:227-258`, `tf.scan(parallel_iterations=1)`) is called 24× per forward pass ≈1,416 sequential steps per batch — the dominant cost (full grid ≈22-33 GPU-h). **In scope:** `ewma_sequence_matrix` — `ema = einsum('btk,bk->bt', M, x)` with `M[b,t,k] = α_b(1−α_b)^{t−k}` built as `exp(D·log1p(−α_b))`; exact to fp32, per-sample α supported, gradient flows; guarded by `test_ewma_matrix_equals_scan` (`atol=1e-5`, grad `rtol=1e-4`). Expected full grid ≈7-11 GPU-h after.

### C5 — Honest backtest engine (`strategy/`)

`Strategy` protocol (`warmup()`, `decide(signal: MultiHorizonSignal, bar, state) → Order(side, size_frac, tp, sl, reason)`); the four notebook strategies collapse into `ThresholdSpikeStrategy` (trade.ipynb), `EnhancedMultiHorizonStrategy` (inference cell 7), `LiberalStrategy` (cells 13/14, knobs as fields) + baselines `BuyAndHold`, `AlwaysFlat`, `RandomSignal(trade_rate)`. `BacktestConfig`: `fill="next_open"` (today fills at the signal bar's own close), `fee_bps=10, half_spread_bps=1, slippage_bps=2`, `tp_sl_on="high_low"` (OHLC is in the committed CSV; today only close reaches the engine), `same_bar_tiebreak="sl_first"`, `max_hold=30`, `mark_to_market_at_end=True`, `minutes_per_year=525_600`. `BacktestResult`: equity, trades, `summary {n_trades sharpe_net sharpe_gross sortino max_drawdown turnover hit_rate profit_factor exposure avg_hold_bars total_return fees_paid}`, baselines incl. `random_same_freq` percentile (100 seeds). `assert_no_lookahead(frame, strategy)` perturbs rows > t and asserts the decision at t is unchanged.

Bugs fixed on port: centred `np.convolve(mode='same')` / `rolling(center=True)` → trailing (`FeatureBuilder`); test-set-median `var_scale` → computed on cal, stored in `calibration/var_scale.json`; `thresh=0.67` (silently unanimity, direction-blind) → `min_agreeing_horizons=2` + consensus must match the trade side; cell-7 SL `curr_price − vol_est·1.5·curr_price` (scaled σ treated as a fraction → stop at −0.5×price) → `entry − sl_vol_mult·sigma_dollars`; `quality_ok` hard-coded → honours `require_*` knobs; trade.ipynb tp/sl computed-never-read → `Order` fields enforced by the engine.

### C6 — Reproducibility
`utils/seeding.py::seed_everything(seed)` (random/np/`set_random_seed`/`Dataset.shuffle(seed=)` — `model.py:2227` has none — /dropout seeds; `enable_op_determinism()` only under the experiment runner, never mid-notebook); `TF_DETERMINISTIC_OPS`, `TF_CUDNN_DETERMINISTIC`, `PYTHONHASHSEED` set before TF import (scripts re-exec if unset); `neural-trade env` fingerprint (versions, CUDA/cuDNN, devices, git sha+dirty, hostname *hash* — the repo already leaks three machine identities); every reported number carries a `run_id`; `TESTING_DOCUMENTATION.md` regenerated from CI output instead of hand-written claims.

---

## Execution sequence (cross-phase)

| Order | Work | Depends on |
|---|---|---|
| 1 | Phase 0 | — |
| 2 | A1 (M1) | 1 |
| 3 | A2 (M2), A3 (M3) | 2 |
| 4 | A4 (M4: purged 4-way split, calibration on cal, real early stopping) | 3 |
| 5 | B1–B4 (package skeleton, registry hardening, canonical losses, typed Config + LossWeights) | 4 — Phase A edits land on the old files, which are then **moved, not edited** |
| 6 | B5–B9 (helpers, layers, models, metrics, data + `splits.py`) | 5 |
| 7 | B10–B11 (training package, callbacks, `RunContext`/`JsonlEpochLogger`) | 6 |
| 8 | B12–B13 (viz, serving/Predictor) | 7 |
| 9 | B14 (evaluation `evaluate()`/baselines; strategy + backtest engine) | 8 |
| 10 | B15 (`ewma_sequence_matrix`; ablation harness; smoke grid in nightly CI) | 9 |
| 11 | B16–B18 (CLI, README, logging/except sweeps, thin notebooks, remove shims) | 10 |
| 12 | Full 84-run ablation grid on the GPU box; first `report.md` committed under `runs/ablations/ablate_physics_v1/` | 11 |

Parallelisable: C1 pure-numpy tests (`test_eval_metrics`, `test_calibration`, `test_backtest`) can be written any time after Phase 0; B5/B6 can proceed while A2–A4 are being run.

---

## Verification (end-to-end)

1. **Phase 0:** `pytest` from repo root collects all existing tests; CI `unit` green.
2. **M1:** T1/T2/T3 fail on HEAD and pass after S1–S5; 2-epoch smoke CSV shows changing `val_loss`, finite periods, `nonfinite_grad_steps == 0`, `extended_h*` unpinned.
3. **M2:** 5-epoch run: physics columns finite/bounded/non-constant; `corr(var, err²)_h1 > 0.10`; PIT-KS < 0.2; ECE ≈ soft-ECE.
4. **M3:** 20-epoch run: `EV(delta)_h1 > 0`, `AUC_h1 > 0.52`, trades > 0, `val_gauss_dir_mcc_h1 > 0`.
5. **M4:** test conformal coverage ∈ [0.87, 0.93]; early stopping fires; `test_purged_split_no_overlap` and `test_leak_count_without_gap_is_79` pass.
6. **Package:** after each migration step `python -m pytest` green; root `model.py` < 30 lines by B12; `import neural_trade` < 0.5 s (no TF); `registries/test_contracts.py` proves every registry is queried during a smoke run; `neural-trade train --config configs/ci.yaml --epochs 1` then `neural-trade predict` succeed in CI.
7. **Serving:** `test_predictor_round_trip` (train → save → load → identical 9 heads).
8. **Evaluation:** `eval_report_test.json` contains baseline rows and `beats_baseline`; `test_ev_price_trap_is_not_reported`.
9. **Backtest:** hand-computed PnL after costs to `rtol=1e-9`; look-ahead probe passes for all registered strategies.
10. **Ablation:** smoke grid < 10 min on CPU (nightly); full grid produces `report.md` with per-term verdicts and run ids; `sec_per_step` recorded for scan vs matrix EWMA.
11. **Hygiene:** ruff (`T201 E722 S110 BLE001`) clean in `src/`; `test_notebooks_thin` passes; coverage gates enforced.

## Definition of done
- Model demonstrably trains (M1–M4 gates recorded, with run ids).
- `src/neural_trade` installable; all 9 registries implemented, strict, tested, and reached by the training/serving path; ~77 real components; plugin loader works (`plugins/examples/echo_metric.py` appears in `registry list metrics`); typed `Config` with YAML round-trip.
- Four-way purged split; calibration fit on its own split and consumed by `Predictor`; baselines in every report; every reported number links to a run.
- Test suite: every test imports project code; two collection failures fixed; ≥ the coverage gates; CI + nightly green.
- Honest backtest engine with fees/spread/slippage/next-open fills/high-low TP-SL and look-ahead self-test; four notebook strategies collapsed to registered classes.
- First full physics-term ablation report committed — the artefact that answers "does it provide value?" with evidence, and the loop the owner asked for to keep iterating.

## Explicitly out of scope
Keras 3 / newer TF migration; git history rewrite; new data sources (exchange APIs, order book); new architectures beyond `gru_attention` (the Models registry is ready for them); multi-asset evaluation; MLflow (tracker protocol left open).
