# Backlog

All open work, prioritised. The lead picks the highest-priority `todo` item whose dependencies are
done and whose role is not `owner` (see [OPERATING_MODEL.md](OPERATING_MODEL.md)). IDs are stable:
never renumber; new items take the next free number. Done items stay (status `done`, with the
evidence) until the next milestone review, then move to the log at the bottom.

Priorities: **P0** wrong numbers or broken behaviour; **P1** the milestone path and owner decisions
that block it; **P2** useful features, logging, infra; **P3** polish (batched per module).

Status values: `todo`, `in-progress`, `blocked` (with the reason), `done` (with the evidence).

| ID | P | type | role | status | title |
|---|---|---|---|---|---|
| [NT-001](#nt-001) | P0 | bug | implementer | todo | CI unit job green again (red since fe4ba85) |
| [NT-002](#nt-002) | P0 | bug | implementer | todo | Engine random null ignores position size (sized strategies are ranked against a costlier null) |
| [NT-003](#nt-003) | P1 | research | experimenter | todo | Direction skill: pass the M3 direction clauses (AUC h1 > 0.52, val MCC h1 > 0.02, Gaussian readout MCC > 0) |
| [NT-004](#nt-004) | P1 | research | experimenter | todo | Price heads: a served delta with positive EV (M3 EV clause) |
| [NT-005](#nt-005) | P1 | research | experimenter | todo | Cost-aware trading: an edge per trade above the 26 bps round trip |
| [NT-006](#nt-006) | P1 | research | experimenter | todo | Physics-term ablation re-run on the current trainer (served-epoch fix) |
| [NT-007](#nt-007) | P1 | owner-decision | owner | todo | Owner decision: which delta the strategies read (served beta-shrunk vs raw heads) |
| [NT-008](#nt-008) | P1 | owner-decision | owner | todo | Owner decision: merge remediation/plan into master |
| [NT-009](#nt-009) | P1 | owner-decision | owner | todo | Owner: free space on C: (Docker WSL disk image) |
| [NT-010](#nt-010) | P2 | infra | implementer | todo | Every cited number links to a tracked run (run-tracking policy, clean git status) |
| [NT-011](#nt-011) | P2 | infra | implementer | done | Notebook output policy: remove nbstripout (it contradicts D-013) |
| [NT-012](#nt-012) | P2 | feature | implementer | todo | Training logging: per-term loss contributions, gradient max and clip counts, deadband sample counts |
| [NT-013](#nt-013) | P2 | feature | implementer | todo | Evaluation report: realised-vol variance baseline and honest baseline verdicts |
| [NT-014](#nt-014) | P2 | feature | implementer | todo | Evaluation report: stored bootstrap intervals, n/a for constant readouts, low-memory confidence gap |
| [NT-015](#nt-015) | P2 | feature | implementer | todo | One overlap-aware interval toolkit in stats.py (reliability bands, PIT band per bin) |
| [NT-016](#nt-016) | P2 | feature | implementer | todo | Backtest data: trade info columns, entry tiers, per-side summary statistics |
| [NT-017](#nt-017) | P2 | owner-decision | owner | todo | Owner decision: licence |
| [NT-018](#nt-018) | P3 | feature | implementer | todo | Notebook package UX: explorer window slider with presets, widget-state policy, stable saved outputs |
| [NT-019](#nt-019) | P3 | polish | implementer | todo | Training-record figures polish (training_dashboard.py, indicator_evolution.py) |
| [NT-020](#nt-020) | P3 | polish | implementer | todo | Head-analytics figures polish at notebook widths (analytics_direction/delta/confidence, analytics_tables) |
| [NT-021](#nt-021) | P3 | polish | implementer | todo | Trading figures polish (trading_dashboard.py, trade_analytics.py) |
| [NT-022](#nt-022) | P3 | polish | implementer | todo | Run-comparison and calibration-explorer figures polish (comparison.py, calibration_plots.py) |
| [NT-023](#nt-023) | P3 | polish | implementer | todo | Theme: one reference dash, a strategy palette, legend fixes (theme.py) |

## Items

### NT-001

**CI unit job green again (red since fe4ba85)**

- **status:** todo
- **priority / type / role:** P0 / bug / implementer
- **area:** .github/workflows/ci.yml, requirements-ci.txt, tests/, plotly/pandas-version-sensitive code in src/neural_trade/visualization and src/neural_trade/notebook
- **why:** The GitHub Actions 'ci' workflow fails on the last two pushes of remediation/plan: be93193 (run 36043324055, job 107780609497) and fe4ba85 (run 36032011043, job 107742810495). Both fail in the step 'Unit tests (CPU, no slow/gpu)'. The last green run was 609d19e. Lint passes. Locally, the fast non-TF suite passes (2026-09-25, pytest -m 'not tf and not slow and not gpu': 557 passed, 142 deselected). The main suspect is version drift: the review rounds were written and checked against plotly 6.7.0 / pandas 2.3.3 in the nt env, while requirements-ci.txt pins plotly==5.24.1 and pandas==2.0.3. The TF-marked tests are the other candidate. CI logs need an authenticated API or the gh CLI (not installed here); failure annotations are readable without auth. This breaks the plan's Definition of Done ('CI + nightly green') and every future QA gate.
- **acceptance:** (1) Root cause recorded in the item: the failing test names, taken from the CI log. Without auth, add a CI step that turns junit failures into '::error' annotations, then read them via the public check-runs annotations API. (2) The 'ci' workflow ends with conclusion=success (lint + unit, including 'Coverage gates' and 'CLI smoke') on the pushed branch head. (3) requirements-ci.txt and the local env agree on the plotly and pandas major versions, or the suite passes under both (documented in docs/RUNBOOK.md). (4) The local 'pytest -m "not slow"' still passes.
- **source:** GitHub Actions API (runs 36043324055, 36032011043); plan 'Definition of done' (CI + nightly green); requirements-ci.txt vs local plotly 6.7.0 / pandas 2.3.3

### NT-002

**Engine random null ignores position size (sized strategies are ranked against a costlier null)**

- **status:** todo
- **priority / type / role:** P0 / bug / implementer
- **area:** src/neural_trade/strategy/strategies.py (RandomSignal), src/neural_trade/strategy/backtest.py (random_same_frequency, backtest), src/neural_trade/notebook/backtest_ui.py (matched_random_null), src/neural_trade/cli.py cmd_backtest, scripts/backtest_gate.py
- **why:** random_same_frequency (backtest.py:271-293) runs RandomSignal, which hard-codes Order size 1.0 (strategies.py:326). enhanced_multi_horizon sizes each position between 0.1 and 1.0 (strategies.py:147-148). After costs, return is mostly cost x size x trade count. So the 'random percentile' printed by 'neural-trade backtest' (cli.py:125-141) and by scripts/backtest_gate.py puts a sized strategy against a null that pays more costs. That number is wrong. The notebooks use the size-matched matched_random_null (backtest_ui.py:196-236), so the CLI and notebook paths give different ranks for the same run.
- **acceptance:** (1) RandomSignal has a field size_frac: float = 1.0 and uses it in its Order. (2) random_same_frequency passes size_frac = the mean decision size. It returns size_frac, random_p05_total_return, random_p95_total_return, random_mean_gross_return and percentile_gross_return. (3) matched_random_null is a thin wrapper over it. (4) New test: for a strategy with sizes < 1, every null trade's size equals the mean size, and the engine and notebook paths give an identical percentile_total_return on the same seeds. (5) The existing backtest and strategy tests pass, including assert_no_lookahead for all strategies.
- **source:** integration_todo.md trade_analytics requests (RandomSignal size_frac, random_same_frequency size_frac: carried over); verified in code 2026-09-25

### NT-003

**Direction skill: pass the M3 direction clauses (AUC h1 > 0.52, val MCC h1 > 0.02, Gaussian readout MCC > 0)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** configs/default.yaml, src/neural_trade/models/gru_attention.py (direction heads and skip), src/neural_trade/losses/functions.py, scripts/direction_experiments.py, scripts/gate_run.py, scripts/check_gates.py, runs/experiments/direction_v2/, runs/gates/
- **why:** M3 fails on the final code (runs/gates/REPORT.md, m6): test AUC h1 0.5015 (0.5239 with the deadband mask), best-epoch val_dir_mcc_h1 0.0175 against the 0.02 threshold, and val_gauss_dir_mcc_h1 -0.0219. The latest default run (runs/20260924T182915Z-1aeff1c-dirty-af67ee43, batch 256, served epoch 19) scores test AUC h1 0.478 and MCC -0.034. The logreg_lags baseline scores 0.523 on the same block, and a logistic regression on trailing returns reaches about 0.56 on fold -1 (direction_v1 REPORT section 1). Identical GPU runs differ by 0.01-0.05 AUC, so no single run can decide. D-010's claim that batch 256 matches the batch-64 run m6 rests on one run.
- **acceptance:** (1) A pre-registered spec in runs/experiments/direction_v2/: one hypothesis and at most 3 variants, all within gru_attention (new architectures are out of scope). Batch 64 vs 256 is a candidate variant. (2) Variants are chosen on dev folds -3/-2 only, with at least 3 seeds each. (3) One judgement on fold -1: scripts/check_gates.py on the new gate runs under runs/gates/ shows, averaged over at least 3 seeds, test AUC h1 > 0.52 on all 7,236 test rows, best-epoch log_val_dir_mcc_h1 > 0.02 and best-epoch log_val_gauss_dir_mcc_h1 > 0. The mean AUC h1 is also >= the logreg_lags baseline on the same block. (4) REPORT.md lists per-seed, per-fold AUC with 80-bar block-bootstrap CIs. A negative verdict meets the criteria when the report holds all of this. (5) STATUS.md and README status are updated.
- **source:** runs/gates/REPORT.md (M3 FAIL); runs/experiments/direction_v1/REPORT.md sections 1, 4, 5; eval_report_test.json of 20260924T182915Z-1aeff1c-dirty-af67ee43; docs/DECISIONS.md D-010

### NT-004

**Price heads: a served delta with positive EV (M3 EV clause)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** src/neural_trade/calibration/pipeline.py (delta shrinkage fit), src/neural_trade/training/trainer.py / callbacks (checkpoint criterion), src/neural_trade/models/gru_attention.py (price towers), runs/experiments/price_heads_v1/
- **why:** The M3 clause EV(delta) h1 > 0 (served) cannot pass while delta shrinkage sets beta_h1 = 0: the served delta is then exactly 0. In m6, served EV was 0.0000 (beta 0) and raw EV -0.6555. The latest run has betas h0 0.123 / h1 0.000 / h2 0.069, with raw EV h1 -0.061. The raw heads overfit. Early stopping watches the total val loss, which keeps improving through the direction and variance terms, so it cannot catch this (direction_v1 section 6). The beta fit is plain OLS clipped to [0, 1] (pipeline.py:222), and a few high-leverage points drive it: on cal, h0 beta goes from 0.212 to 0.155 without the top 0.5% |pred|, and h1 from 0.023 to negative.
- **acceptance:** (1) A pre-registered spec with at most 3 variants, for example early stopping / checkpoint on per-head val point loss, stronger tower regularisation, or a robust (trimmed or Huber) beta fit. Variants are chosen on folds -3/-2 x at least 3 seeds. (2) Judged once on fold -1: served EV(delta) h1 > 0 with beta_h1 > 0, on the mean over at least 3 seeds and on every one of them, with the raw EV reported alongside. (3) CalibrationPipeline records both the OLS beta and any robust beta in pipeline_meta.json (unit test). (4) Report at runs/experiments/price_heads_v1/REPORT.md. A negative result meets the criteria when the report is complete.
- **source:** runs/gates/REPORT.md (M3, m6: EV served 0.0000, raw -0.6555); runs/experiments/direction_v1/REPORT.md section 6; integration_todo.md delta 'calibration/pipeline.py (finding 4)'

### NT-005

**Cost-aware trading: an edge per trade above the 26 bps round trip**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** src/neural_trade/strategy/strategies.py (new or re-knobbed strategy), configs/strategies/, runs/experiments/trading_costs_v1/
- **depends on:** NT-002, NT-007
- **why:** No strategy has made money after costs. In gate m6, calibrated_quantile made 120 trades: gross +4.1% (the same as buy-and-hold), net -23.4% after 26 bps per round trip. In the latest notebook run, calibrated_quantile made 147 trades, liberal 58 and enhanced_multi_horizon 0. Trades almost never move in their favour by more than the 0.26% round-trip cost, and a 15-minute move has a standard deviation of about 21 bps (direction_v1 REPORT section 5).
- **acceptance:** Either (a) a registered strategy whose knobs are fixed on the cal block or dev folds only, with all of these on test: net total return > 0 after the default costs (fee 10 + half-spread 1 + slippage 2 bps per side) on fold -1; size-matched random-null percentile (after costs) >= 95; and net > 0 on at least 2 of 3 walk-forward folds. Or (b) a negative-result report at runs/experiments/trading_costs_v1/REPORT.md with, per strategy and horizon, the mean gross edge per trade in bps and its 80-bar block-bootstrap 95% CI against the 26 bps cost, and the break-even round-trip cost. In both cases: a pre-registered spec with at most 3 variants (for example: enter only when |expected move| > cost + k x sigma; longer holds; fewer trades).
- **source:** runs/gates/REPORT.md 'Reading' (Trading); README status; notebooks/02_backtest.ipynb saved outputs; runs/experiments/direction_v1/REPORT.md section 5

### NT-006

**Physics-term ablation re-run on the current trainer (served-epoch fix)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** scripts/ablate.py, configs/ablation_physics.yaml, runs/ablations/ablate_physics_v2-full/, notebooks/05_compare_runs.ipynb (ABLATION_DIR), docs/DECISIONS.md D-003
- **depends on:** owner approval of the GPU time (84 runs at 5-7 min each, about 7-10 GPU-hours: over the 3-hour limit; see STATUS 'Waiting for the owner'); NT-009 (disk); soft: NT-012 (so the grid logs per-term contributions)
- **why:** ablate_physics_v1-full ran at commit 6dec27a, before 53c0df2 (D-011), so every cell was scored on its last epoch, not the best-validation one. It also ran before the batch-256 default. The family verdict is INCONCLUSIVE: its variance VALUE (CRPSS +0.0056, var/err^2 Spearman +0.0364) is withdrawn by the h1 direction-AUC guard-rail breach (-0.0119 against tolerance 0.01). No single term reaches VALUE. The pre-registered rule for INCONCLUSIVE is 'add seeds'.
- **acceptance:** (1) runs/ablations/ablate_physics_v2-full/ is committed with report.md, analysis.json, summary.csv, results.csv and spec.json. (2) Every cell completed, and each cell's status.json has weights_epoch (trained at a commit >= 53c0df2). (3) configs/ablation_criteria.yaml is byte-identical to the one used by v1. (4) report.md gives per-term and family verdicts, with the family BREACH row when one applies. (5) D-003's status in docs/DECISIONS.md cites the v2 verdicts. (6) Notebook 05 points ABLATION_DIR at v2 and is re-executed per D-013.
- **source:** runs/ablations/ablate_physics_v1-full/report.md; docs/DECISIONS.md D-011; computed task thread (3)

### NT-007

**Owner decision: which delta the strategies read (served beta-shrunk vs raw heads)**

- **status:** todo
- **priority / type / role:** P1 / owner-decision / owner
- **area:** src/neural_trade/strategy/signals.py (SignalFrame.build: magnitude_coherent, direction_aligned), src/neural_trade/strategy/strategies.py (EnhancedMultiHorizonStrategy entry d1 > 0, INCOH exit, TP sizing; LiberalStrategy TP sizing)
- **why:** SignalFrame.build (signals.py:90-92) computes magnitude_coherent and direction_aligned on the served deltas, where served = beta x raw. The served ordering therefore mostly reflects the beta ratio: magnitude_coherent is true on 7.2% of test bars on served deltas vs 56.2% on the raw heads, and it drove INCOH exits on 42 of 53 enhanced_multi_horizon trades. A verifier estimated 209 trades and 110 INCOH exits on the raw heads. With beta_h1 = 0 (the latest run), enhanced_multi_horizon needs served d1 > 0 to enter (strategies.py:147), so it cannot trade at all (notebook 02: 0 trades). direction_aligned then only means 'P(up) <= 0.5'. Changing this changes trading behaviour, which the operating model escalates to the owner.
- **acceptance:** (1) docs/DECISIONS.md has an entry with the choice (served, raw heads, or served / beta) and its scope: magnitude_coherent, direction_aligned, the enhanced_multi_horizon d1 entry rule and take-profit sizing, and liberal take-profit sizing. (2) If behaviour changes, an implementer item exists whose acceptance includes a test in tests/ pinning the new rule, and before/after trade and INCOH counts on the reference run.
- **source:** integration_todo.md (delta, confidence, tables: finding 23 / 23d owner decision); fix_results.json tables finding 5; final session message 'Decisions for you: Strategy exits'

### NT-008

**Owner decision: merge remediation/plan into master**

- **status:** todo
- **priority / type / role:** P1 / owner-decision / owner
- **area:** git (master), .github/workflows/nightly.yml
- **depends on:** NT-001
- **why:** master still holds the pre-remediation code. GitHub runs scheduled workflows only from the default branch, so nightly.yml (slow, data and notebook tests plus the smoke ablation) has never run: the Actions API lists 0 scheduled runs, and nightly is not even registered as a workflow. The Definition of Done's 'nightly green' therefore cannot be met. D-004: merging into master is the owner's decision.
- **acceptance:** (1) The owner's answer is recorded in docs/DECISIONS.md. (2) If yes: master contains the remediation/plan HEAD through a merge with no history rewrite, and the nightly workflow has at least 1 completed run on master. Its failures, if any, are filed as backlog items.
- **source:** plan 'Definition of done' (CI + nightly green); GitHub Actions API (workflows list, scheduled runs = 0); docs/DECISIONS.md D-004

### NT-009

**Owner: free space on C: (Docker WSL disk image)**

- **status:** todo
- **priority / type / role:** P1 / owner-decision / owner
- **area:** machine: C: drive, C:\Users\Step\AppData\Local\Docker\wsl\disk\docker_data.vhdx; docs/RUNBOOK.md
- **why:** C: has 6.0 GB free (98% used) on 2026-09-25. The Docker WSL image is about 116-119 GB. During the review, C: reached 0 bytes, and pytest, git index writes and figure renders all failed. GPU grids, notebook executions and renders need headroom. Agents may not touch Docker data or anything they did not create.
- **acceptance:** 'df -h /c' shows at least 20 GB free, or the owner names another drive for runs and scratch files and docs/RUNBOOK.md records it. The session-start step in STATUS.md records the free space.
- **source:** integration_todo.md (ENVIRONMENT notes in training, misc, tables); memory neural-trade-env; df on 2026-09-25

### NT-010

**Every cited number links to a tracked run (run-tracking policy, clean git status)**

- **status:** todo
- **priority / type / role:** P2 / infra / implementer
- **area:** .gitignore, runs/, README.md (status section), docs/RUNBOOK.md
- **why:** git status is never clean after a session. 7 notebook run dirs, the ablation's cells/, logs/ and runs/, and direction_v1's logs/ and runs/ are untracked, while the session protocol requires a clean tree. The saved notebook outputs and README cite runs, for example 20260924T182915Z-1aeff1c-dirty-af67ee43 (4.5 MB), whose directories exist only on this machine. The README status numbers carry no run ids, and its AUC range (0.50-0.55) does not cover the latest default run (h1 0.478). This is a Definition of Done gap: 'every reported number links to a run'.
- **acceptance:** (1) docs/RUNBOOK.md states which run files are tracked (config.yaml, meta.json, status.json, metrics.jsonl, eval_report_*.json/md, artifacts/meta.json and pipeline_meta.json) and which are ignored (weights, joblib, prediction npz, ablation cells/logs). (2) .gitignore implements it, so 'git status --porcelain' is empty after a notebook execution. (3) A script or test finds every run id cited in README.md, runs/**/REPORT.md, runs/**/report.md and the saved notebook outputs, and passes only when each has its light files tracked. (4) Each README status number cites a run id.
- **source:** plan 'Definition of done' (every reported number links to a run); git status at session start; docs/OPERATING_MODEL.md session protocol (clean tree)

### NT-011

**Notebook output policy: remove nbstripout (it contradicts D-013)**

- **status:** done (2026-09-25, lead): the nbstripout filter is gone from `.gitattributes` and the hook from `.pre-commit-config.yaml` (with a comment why); `tests/test_notebooks_thin.py` docstring updated. Evidence: `git grep nbstripout` only finds the explanatory comments.
- **priority / type / role:** P2 / infra / implementer
- **area:** .gitattributes, .pre-commit-config.yaml, docs/DECISIONS.md D-013
- **why:** .gitattributes marks *.ipynb with filter=nbstripout, and .pre-commit-config.yaml runs the nbstripout hook. Its header tells every clone to run 'nbstripout --install'. D-013 (owner) requires notebooks to be committed with real outputs, and tests check the saved outputs (tests/test_notebooks_thin.py::test_saved_training_dashboards_mark_each_chance_band_edge_in_its_horizon_colour, tests/test_viz_variance.py::test_saved_notebook_outputs_carry_the_current_encoding). A future session that follows the pre-commit instructions would strip every output on its next commit.
- **acceptance:** (1) 'git check-attr filter notebooks/01_train_and_monitor.ipynb' reports 'unspecified'. (2) .pre-commit-config.yaml has no nbstripout hook, or its hook excludes notebooks/. (3) The install comment in its header no longer mentions nbstripout. (4) D-013 cites the change.
- **source:** .gitattributes line '*.ipynb text eol=lf filter=nbstripout'; .pre-commit-config.yaml; docs/DECISIONS.md D-013; plan B7 (superseded by D-013)

### NT-012

**Training logging: per-term loss contributions, gradient max and clip counts, deadband sample counts**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/training/custom_model.py (_update_diagnostics, train_step), src/neural_trade/core/outputs.py (LossComponents), src/neural_trade/losses/functions.py, src/neural_trade/metrics/tf_direction.py, src/neural_trade/telemetry/epoch_logger.py, src/neural_trade/visualization/training_dashboard.py
- **why:** (1) coherence_penalty is computed (losses/functions.py:593) but not logged. The loss stack therefore draws an inferred 'other: coherence (not logged)' band (training_dashboard.py:98), and it repeats the 0.1 factors of custom_loss by hand. (2) Only the epoch mean of the pre-clip global gradient norm is logged (custom_model.py:475; tf_direction.py:92-94). There is no per-group maximum and no count of steps clipped at GRAD_CLIP_NORM. (3) The direction chance bands use n_val // steps, although 11-17% of validation samples (19-26% on test) sit inside the deadband and are not scored (training_dashboard.py:1133 says so).
- **acceptance:** A 1-epoch smoke run's metrics.jsonl rows contain: (1) contrib_point, contrib_trend, contrib_dir, contrib_nll, contrib_crps, contrib_soft_ece, contrib_vol, contrib_reg and coherence_penalty, for train and val_. Their sum equals loss / val_loss within 1e-4 relative (test). (2) Train-only grad_norm_max_main, grad_norm_max_indicator, grad_clip_steps_main and grad_clip_steps_indicator. (3) val_dir_n_h0/h1/h2, equal to DirectionStats.mask_sum. For such a run, the training dashboard has no 'other: coherence (not logged)' trace, the gradient panel draws the maximum with the clip count in the hover, and the direction chance bands use val_dir_n // steps (tests in tests/test_viz_training.py). Older runs without these keys still render. tests/test_custom_loss.py and tests/test_telemetry.py pass.
- **source:** integration_todo.md training requests (findings 38, 106b, NEW val_dir_n); fix_results.json training findings 2/3 declined; grep 'not logged' in src

### NT-013

**Evaluation report: realised-vol variance baseline and honest baseline verdicts**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/evaluation/baselines.py, src/neural_trade/evaluation/report.py (variance_block, beats_baseline, to_markdown _md_baselines), src/neural_trade/visualization/analytics_tables.py (baseline_table caption)
- **why:** (1) The only variance baseline is const_var. For a constant sigma its var/err^2 Spearman is stored as 0.0000 (report.py:188-189), so every report prints 'beats (boot z +6.96)' for it by construction (eval_report_test.md of 20260924T182915Z, line 133). (2) The variance figure already compares against a trailing realised-vol baseline that the report does not have (finding 41, report half). (3) The report tests 78 baseline cells at 5% each with no multiplicity note, so about 4 pass by chance. The only 'significantly worse' cell (logreg_lags Brier h2, z -2.07) depends on the lag choice.
- **acceptance:** (1) RELEVANT['realized_vol'] = ('variance',). BaselineSet.fit fits k_h = sqrt(mean(err^2 / u^2)) on train, with u from calibration.conformal.interval_scale('realized_vol'), and predict gives a per-sample sigma = k_h x u_h (pf accepts per-sample sigma arrays). Reports on a frame with X_raw carry realized_vol rows for crps, nll, pit_ks, coverage90 and the Spearman (test). (2) const_var's Spearman cell prints n/a and is left out of beats_baseline (test). (3) to_markdown and the baseline_table caption state how many cells were tested and how many false passes to expect at 5%, or add a Holm/BH-adjusted flag (test on the text).
- **source:** integration_todo.md variance requests (finding 41, report half) and tables verifier remaining (multiplicity); fix_results.json variance finding 41 partial

### NT-014

**Evaluation report: stored bootstrap intervals, n/a for constant readouts, low-memory confidence gap**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/evaluation/report.py (direction_block, variance_block, confidence_gap, gauss_direction at beta = 0), src/neural_trade/experiments/compare.py (_flat), src/neural_trade/visualization/comparison.py
- **why:** (1) eval_report_test.json stores no intervals (no *_ci keys), so runs_comparison_figure falls back to N/steps analytic intervals, which a verifier measured as about 1.6-1.75x wider than a block bootstrap. CRPSS has no interval at all. (2) At beta = 0 the JSON keeps gauss_direction auc 0.5 and pred_up_rate 0, and compare._flat reads gauss_direction rather than gauss_direction_raw. A run comparison of gauss_direction/auc therefore shows a constant as a measured 0.5 with an interval. comparison.py flags only delta metrics with '0 (β=0)'. (3) report.confidence_gap builds [n_boot, n] index arrays (report.py:199-211), about 1 GB at a month of 1-minute bars.
- **acceptance:** (1) direction_block stores auc_ci and mcc_ci, and variance_block stores coverage90_ci and crpss_ci, each as [lo, hi] from the moving-block bootstrap (BLOCK = 80). runs_comparison_figure draws them (test). (2) At beta = 0, gauss_direction metrics are null in the JSON, compare._flat exposes gauss_direction_raw, and the comparison figure shows n/a with the reason (test). (3) confidence_gap gives the same gaps and CIs as the current code on a fixed seed with O(n_boot x n / block) memory, for example by reusing analytics_confidence._boot_plan / _resample_sums (test).
- **source:** integration_todo.md misc (report.py block-bootstrap CIs, finding 16), confidence (confidence_gap memory), direction (auc_ci); fix2_results.json report_tables requests (gauss_direction at beta = 0)

### NT-015

**One overlap-aware interval toolkit in stats.py (reliability bands, PIT band per bin)**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/visualization/stats.py, analytics_direction.py, calibration_plots.py, analytics_variance.py, analytics_delta.py, analytics_confidence.py, analytics_common.py
- **why:** D-012 sets the convention, but the figures implement it differently. (1) Reliability bands are HAC with lag = horizon in the calibration explorer (calibration_plots.py:6) and 80-bar clusters in the direction figure; on the same notebook 04 h1 bins the half-widths differ by up to about 25%. (2) The direction figure's AUC CI, chance band and DeLong test use n/steps (1.4-1.9x wider than a block bootstrap), while its reliability bars use clustered SEs. (3) The variance figure's PIT noise band uses the mean overlap factor over all bins (analytics_variance.py:473). For the edge bins, where a U-shape is read, that is too narrow: the edge bins need about ±0.22-0.30, the drawn band is ±0.15-0.18. (4) The helpers (lrv_factor, circular block bootstrap, Newey-West variance of a mean, skill_half_width, design_effect, cluster_mean_ci) live in figure modules. (5) analytics_common._binned (i.i.d. SE) has no caller left.
- **acceptance:** (1) Those helpers live in stats.py with unit tests, and the figure modules import them from there. (2) calibration_plots.reliability_table and the direction figure's reliability rows call one function and give identical half-widths on the same input (test). (3) Variance row 2 draws a per-bin PIT band as step lines, and a test asserts that the edge-bin half-width is >= the centre-bin one on overlapping heavy-tailed data. (4) Each figure's subtitle names its interval method. (5) analytics_common._binned is removed. (6) Notebooks re-executed and changed figures rendered, per D-013.
- **source:** integration_todo.md direction (convention decision, stats.py helper), delta (promote helpers), variance (noise-band convention; verifier remaining: PIT per bin), misc (shared reliability-band helper); fix_results.json variance new_problems

### NT-016

**Backtest data: trade info columns, entry tiers, per-side summary statistics**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/strategy/backtest.py (BacktestResult.trades_frame), src/neural_trade/strategy/strategies.py (LiberalStrategy, EnhancedMultiHorizonStrategy), src/neural_trade/strategy/performance.py (summarize), src/neural_trade/notebook/backtest_ui.py (trade_stats)
- **why:** (1) trades_frame drops Trade.info (backtest.py:146), so tp1_offset and the entry confidence of each trade are lost to analysis. (2) Liberal's entry tier (high / standard / low quality, strategies.py:201-207) is not recorded, so there is no per-tier breakdown. (3) summarize() lacks avg_win, avg_loss, expectancy, n_long, n_short and gross/net long/short. backtest_ui.trade_stats recomputes them for the notebooks only, so CLI and ablation summaries do not have them.
- **acceptance:** (1) trades_frame has info_<key> columns (e.g. info_tp1_offset, info_confidence). An optional signals= argument joins the decision-bar context: conviction, strength, avg_confidence, agreement, delta_h1, sigma_h1. (2) Order.info['tier'] is one of 'high', 'standard', 'low' for liberal and enhanced_multi_horizon entries (test). (3) summarize() returns the 9 keys, equal to trade_stats on a reference backtest (test), and trade_stats becomes a thin wrapper. (4) The existing strategy and backtest tests pass, including assert_no_lookahead for every registered strategy.
- **source:** integration_todo.md trade_analytics requests (finding 25: trades_frame info_*, liberal tier, summarize keys); fix_results.json trade_analytics finding 25 partial

### NT-017

**Owner decision: licence**

- **status:** todo
- **priority / type / role:** P2 / owner-decision / owner
- **area:** LICENSE, README.md 'Licence' section
- **why:** There is no LICENSE file. README line 243 says the choice belongs to the repository owner, and the plan's B16 lists a LICENSE. Nothing is blocked by it, but the repository is public on GitHub without terms.
- **acceptance:** A LICENSE file exists at the repo root, or docs/DECISIONS.md records 'no licence (all rights reserved)'. The README 'Licence' section matches.
- **source:** README.md 'Licence'; plan B16 / layout (LICENSE)

### NT-018

**Notebook package UX: explorer window slider with presets, widget-state policy, stable saved outputs**

- **status:** todo
- **priority / type / role:** P3 / feature / implementer
- **area:** src/neural_trade/notebook/backtest_ui.py (widget, summary_frame), src/neural_trade/notebook/calibration_ui.py (refit table), src/neural_trade/notebook/_display.py, scripts/notebooks/execute.py, docs/RUNBOOK.md
- **why:** (1) The backtest widget has no window control: detail views exist only as static cells (notebook 02 cell 4, notebook 03). (2) Saved notebooks carry ipywidgets state (01: 157 kB, 02: 627 kB, 04: 96 kB), because execute.py stores it by default. The interactive controls do not work without a kernel, and old state once reopened as 20 px slivers. (3) summary_frame puts the random null's mean Sharpe next to the strategy's; with negative returns Sharpe rewards variance (liberal read as the 100th Sharpe percentile against the 24th return percentile). (4) calibration_ui.refit's 'ECE raw' / 'ECE calibrated' columns use different bins than the reliability plot. (5) _display.mime_bundle's text/plain for a Styler is repr() with a memory address (line 24), so saved outputs differ between runs.
- **acceptance:** (1) BacktestExplorer.widget() has an IntRangeSlider window plus a preset Dropdown ('whole block', 'steepest fall', 'worst trade', 'last 600') that calls trading_dashboard.detail_window, with no plotly rangeslider (headless test). (2) The widget-state policy is decided and written in docs/RUNBOOK.md, and execute.py implements it: either metadata.widgets is stripped, or a check confirms the saved views reopen correctly and hold no figure payload that is already a cell output. (3) summary_frame drops or explicitly labels the null's mean Sharpe (test). (4) The refit table's ECE columns name their binning, or equal-count ECE columns are added (test). (5) text/plain for a Styler contains no memory address (test).
- **source:** integration_todo.md trading_dashboard (#28 IntRangeSlider with presets, store_widget_state), trade_analytics verifier remaining (summary_frame Sharpe), direction requests (calibration_ui ECE columns); fix2_results.json trades requests (_display.py); computed task thread (8)

### NT-019

**Training-record figures polish (training_dashboard.py, indicator_evolution.py)**

- **status:** todo
- **priority / type / role:** P3 / polish / implementer
- **area:** src/neural_trade/visualization/training_dashboard.py, src/neural_trade/visualization/indicator_evolution.py
- **why:** Verifier leftovers the fix rounds did not address; each was confirmed in the current code. The static headings still say '(shade: …)' where no band is drawn (training_dashboard.py:73-75). _ECE95 = 2.47 is about 5-7% tight for 2.7-3.0 effective bins (line 138). The live overfitting flag uses an i.i.d. OLS t, so about 11% of pure-noise runs raise it at some epoch. HD and vac overflow are told apart by lightness only. A weights_epoch beyond the logged rows is clipped silently (line 283). indicator_evolution prints 'key = copy: start' even when no start is known (line 764), and empty panels keep plotly's default heading style.
- **acceptance:** (1) The Brier, ECE and PIT-KS headings drop '(shade: …)' when no horizon has a chance range (test with n_val = 90). (2) The ECE 95% level is derived from the effective bin count or the logged P(up) spread (test). (3) The overfitting flag gives <= 5% false alarms in a 300-run pure-noise simulation (test), for example by requiring 2 consecutive epochs or using a HAC standard error. (4) vac overflow and HD use different marker symbols. (5) A weights_epoch beyond the logged rows gives a warning tile instead of being clipped. (6) The indicator subtitle leaves out 'key = copy: start' without a start, and empty panels use the drawn panels' bold left-aligned heading. Tests in tests/test_viz_training.py and tests/test_viz_indicators.py.
- **source:** integration_todo.md training verifier remaining (minor items), indicators verifier remaining (optional polish); fix_results.json training/indicators new_problems

### NT-020

**Head-analytics figures polish at notebook widths (analytics_direction/delta/confidence, analytics_tables)**

- **status:** todo
- **priority / type / role:** P3 / polish / implementer
- **area:** src/neural_trade/visualization/analytics_direction.py, analytics_delta.py, analytics_confidence.py, analytics_tables.py, evaluation/frame.py (optional times)
- **why:** Verifier leftovers, confirmed in the current code. Direction figure: at 900 px the row-2/3 titles overlap and the scorecard's last row is cut, because _WRAP_PX = 40 is fixed (analytics_direction.py:53); the cal block's extreme y tick sits in the panel corner. Delta figure: the table's last row label clips below about 1,050 px (analytics_delta.py:804); ticks fall inside the shaded clip margins; the no-raw footer asserts β unconditionally (line 850). Confidence figure: the NEUTRAL reference ticks on MUTED bars have about 1.2:1 contrast (TICK, analytics_confidence.py:43); the 'same sign' swatch matches only the 'all 3' bar; the majority line is not labelled as this block's hindsight rate; at 1100 px subtitle line 3 and the row-3 heading clip. analytics_tables._raw_or ignores frame.meta['delta_raw'] (line 107), unlike report._resolve_raw. The rolling x axes are 'hours into the block' because PredictionFrame has no timestamps.
- **acceptance:** Rendered at 900 and 1000 px (scripts/notebooks/render.py): (1) direction: no overlapping titles, the scorecard's last row fully visible, no y tick in the row-2 corner on the cal block. (2) delta: the last table row label is not clipped; no tick or gridline inside a shaded margin; the footer is hedged ('if these are the served deltas…'). (3) confidence: reference ticks with contrast >= 3:1 against their bars; the 'same sign' swatch removed or relabelled; the majority line labelled 'this block's majority (hindsight)'; nothing clipped at 1100 px. (4) analytics_tables._raw_or falls back to frame.meta['delta_raw'] / delta_scale (test). (5) Optional: rolling rows show dates when the frame carries times. Notebooks re-executed per D-013.
- **source:** integration_todo.md direction/delta/confidence verifier remaining; tables verifier remaining (_raw_or); fix2_results.json confidence requests (1100 px clipping); delta requests (PredictionFrame timestamps)

### NT-021

**Trading figures polish (trading_dashboard.py, trade_analytics.py)**

- **status:** todo
- **priority / type / role:** P3 / polish / implementer
- **area:** src/neural_trade/visualization/trading_dashboard.py, src/neural_trade/visualization/trade_analytics.py
- **why:** Verifier leftovers, confirmed in the current code. The long-view P(up) band key is about 60 characters (trading_dashboard.py:455-456), which pushes the legend left and cuts 'decided short' at 1000 px. TP/stop text labels overlap each other and the markers when trades cluster (lines 384-399), and there is no collision rule. The P(up) panel does not say whether the values are calibrated. The entry hover has no TP1 (_levels_text, lines 139-142), although liberal and enhanced store info['tp1_offset']. 'avg confidence' is not named as mean exp(-var_h / var_scale) (line 485). The trade-analytics 'sign right' readout has no chance CI (line 524). The cumulative 'gross' line uses the reference-line dash (line 364). A single-trade figure draws an empty cumulative panel (mode='lines'). The comparison titles run together at 900 px (line 656).
- **acceptance:** (1) The full-block P(up) legend fits at 1000 px, checked by a legend-text-length test. (2) TP/stop labels never overlap another label or marker, with a test on the liberal bars 6850-6870 of the reference run. (3) The panel title reads 'Calibrated P(up)' when calibrated=True. (4) The entry hover shows TP1 (entry ± tp1_offset), and TP1 is drawn as a level. (5) The confidence trace is named 'mean over h of exp(-var_h / var_scale)', with var_scale in the hover. (6) 'sign right' carries a Wilson 95% CI on n_eff. (7) 'gross' uses a dash that no reference line uses. (8) With n == 1 the cumulative panel shows the point. (9) The strategy_comparison titles and subtitle are not cut at 900 px (render).
- **source:** integration_todo.md trading_dashboard verifier remaining (#123, #124, #125, #98), trade_analytics verifier remaining, confidence request (finding 24 label); fix2_results.json trades requests (single trade)

### NT-022

**Run-comparison and calibration-explorer figures polish (comparison.py, calibration_plots.py)**

- **status:** todo
- **priority / type / role:** P3 / polish / implementer
- **area:** src/neural_trade/visualization/comparison.py, src/neural_trade/visualization/calibration_plots.py
- **why:** Verifier leftovers, confirmed in the current code. With long run labels (the 12 VAC_OVERFLOW ablation runs), the panel sub-headings overlap at 900-990 px. In the no-report fallback, MIN_HALF_SPAN (AUC 0.03, MCC 0.05; comparison.py:53) is about half of one run's 95% interval, so noise fills the axis. The non-horizon legend entry is named 'value' (line 493), and headings read 'dashed: no skill 0' (line 436), including on empty panels. coverage_over_time_figure draws no line at the saved pipeline's target when it differs from the refit's.
- **acceptance:** (1) With the VAC_OVERFLOW glob, no headings overlap at 900 and 990 px (render). (2) The fallback axis spans at least a typical 95% half-width at about 380 n_eff (AUC >= 0.06, MCC >= 0.10), or a shaded 'typical noise' band is drawn (test). (3) The non-horizon legend entry carries the metric name. (4) Headings read 'dashed: 0 (no skill)', and an empty panel promises no dashed line (test). (5) When saved_target != target, a 'saved target' reference line is drawn (test).
- **source:** integration_todo.md misc verifier remaining; fix_results.json misc new_problems

### NT-023

**Theme: one reference dash, a strategy palette, legend fixes (theme.py)**

- **status:** todo
- **priority / type / role:** P3 / polish / implementer
- **area:** src/neural_trade/visualization/theme.py and the modules that define their own dash / strategy colours (analytics_confidence.py, analytics_direction.py, analytics_variance.py, calibration_plots.py, comparison.py, trade_analytics.py)
- **why:** (1) The reference-line dash has drifted: '6px,4px' in analytics_confidence.py:42 and trade_analytics.py:44, '5px,4px' in analytics_direction.py:55, analytics_variance.py:69 and calibration_plots.py:30, '6px,3px' in comparison.py:621. (2) trade_analytics._STRATEGY_COLORS = OTHER_SERIES[1, 3, 0] (line 38): pink (the same as the 'costs' line), violet (= LONG_COLOR) and amber (= SHORT_COLOR). OTHER_SERIES[0] == DOWN_COLOR and OTHER_SERIES[3] == UP_COLOR (theme.py:40-46). (3) panel_legend uses title side='left', whose width plotly.js ignores when it wraps keys, so the last keys are cut at 1000-1100 px (theme.py:169). (4) empty_panels does not treat a panel whose traces hold only non-finite y as empty.
- **acceptance:** (1) theme.T.REF_DASH is defined once, and a test scans visualization/ for any other dash literal used for reference lines. (2) T.STRATEGY_COLORS has at least 4 colours distinct from the horizon, status, LONG/SHORT and costs colours, and trade_analytics uses it (test). (3) No OTHER_SERIES slot equals UP_COLOR or DOWN_COLOR, or figures that show both are tested to avoid the overlap. (4) panel_legend puts its title on top (side='top'). (5) empty_panels treats an all-non-finite panel as empty (test).
- **source:** integration_todo.md trade_analytics requests (theme owner: strategy palette, legend bottom, REF_DASH), variance requests (panel_legend side-left), misc (empty_panels non-finite), confidence (OTHER_SERIES collisions); fix2_results.json variance requests (shared REF_DASH)

## Done log

Items closed at earlier milestone reviews: the remediation plan's phases 0, A (M1, M2, M4), B and C, and
the notebook review rounds (see [STATUS.md](STATUS.md) and `git log`).
