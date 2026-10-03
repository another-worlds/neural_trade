# Backlog

All open work, prioritised. The lead picks by OPERATING_MODEL "Picking the next item". IDs are stable:
never renumber; new items take the next free number. Done items stay (status `done`, with the
evidence) until the next milestone review, then move to the log at the bottom.

Priorities: **P0** wrong numbers or broken behaviour; **P1** the milestone path and owner decisions
that block it; **P2** useful features, logging, infra; **P3** polish (batched per module).

Status values: `todo`, `in-progress`, `blocked` (with the reason), `done` (with the evidence),
`dropped` (with the reason and date; lead only, never P0 or owner-decision items).

The pick order is in [OPERATING_MODEL.md](OPERATING_MODEL.md) "Picking the next item": priority, then
the earliest open milestone in the one total order of [ROADMAP.md](ROADMAP.md) "Order", then table
order; implementer and experimenter slots are picked separately by that order. ROADMAP says which
milestone each item belongs to (R1, MVP-1 to MVP-6, the research tracks R2-R4, R5). Every milestone
exit criterion is P1. NT-043 may be taken by a second implementer in parallel, next to NT-026 or NT-029
(not next to NT-027 or NT-028: shared visualization/ and build.py). NT-047 is built in window mode (D-037, 2026-09-29).
Acceptance criteria are checkable at QA time; the lead's STATUS / DECISIONS / ROADMAP updates are not
criteria.

Reference setup: numbers in the items are measured on the reference setup (BTC/USDT 1-minute bars, a
60-minute window, horizons of 10, 15 and 20 minutes; [VISION.md](VISION.md), D-019) unless an item
says otherwise. The project is not about that setup; it is the one the MVP tests.

Provenance: `source:` lines that name `integration_todo.md`, `fix_results.json`, `final_findings.json`,
'finding N' or 'verifier' refer to the review rounds of 2026-09-24 (artefacts not kept). 'owner Q&A
2026-09-28' is [docs/qa/2026-09-28-vision-mvp.md](qa/2026-09-28-vision-mvp.md); 'indicator Q&A
2026-09-28' is [docs/qa/2026-09-28-indicators.md](qa/2026-09-28-indicators.md); 'survey 2026-09-28' is
the lead's code survey for the MVP plan. Every item is self-contained; code references (file:line)
were verified on 2026-09-25 (NT-001 to NT-025) and 2026-09-28 (NT-026 to NT-054 and that day's
changes).

| ID | P | type | role | status | title |
|---|---|---|---|---|---|
| [NT-001](#nt-001) | P0 | bug | implementer | done | CI unit job green again (red since fe4ba85) |
| [NT-002](#nt-002) | P0 | bug | implementer | done | Engine random null ignores position size (sized strategies are ranked against a costlier null) |
| [NT-003](#nt-003) | P1 | research | experimenter | todo | Direction skill: pass the M3 direction clauses (AUC h1 > 0.52, val MCC h1 > 0.02, Gaussian readout MCC > 0) |
| [NT-004](#nt-004) | P1 | research | experimenter | todo | Price heads: a served delta with positive EV (M3 EV clause) |
| [NT-005](#nt-005) | P1 | research | experimenter | done | Cost-aware trading: an edge per trade above the 26 bps round trip |
| [NT-006](#nt-006) | P1 | research | experimenter | todo | Physics-term ablation re-run on the current trainer, pre-registered under D-025 |
| [NT-007](#nt-007) | P1 | owner-decision | owner | todo | Owner decision: which delta the strategies read (served beta-shrunk vs raw heads) |
| [NT-008](#nt-008) | P1 | owner-decision | owner | todo | Owner decision: merge remediation/plan into master |
| [NT-009](#nt-009) | P1 | owner-decision | owner | done | Disk: C: nearly full; resolved by moving the project to D: (D-030) |
| [NT-010](#nt-010) | P1 | infra | implementer | done | Every cited number links to a tracked run (run-tracking policy, clean git status) |
| [NT-011](#nt-011) | P2 | infra | implementer | done | Notebook output policy: remove nbstripout (it contradicts D-013) |
| [NT-012](#nt-012) | P2 | feature | implementer | dropped | Training logging: per-term loss contributions, gradient max and clip counts, deadband sample counts |
| [NT-013](#nt-013) | P2 | feature | implementer | todo | Evaluation report: realised-vol variance baseline and honest baseline verdicts |
| [NT-014](#nt-014) | P2 | feature | implementer | todo | Evaluation report: stored bootstrap intervals, n/a for constant readouts, low-memory confidence gap |
| [NT-015](#nt-015) | P2 | feature | implementer | todo | One overlap-aware interval toolkit in the shared statistics module (reliability bands, PIT band per bin) |
| [NT-016](#nt-016) | P2 | feature | implementer | todo | Backtest data: trade info columns, entry tiers, per-side summary statistics |
| [NT-017](#nt-017) | P3 | owner-decision | owner | todo | Owner decision: licence |
| [NT-018](#nt-018) | P3 | feature | implementer | todo | Notebook package UX: explorer window slider with presets, widget-state policy, stable saved outputs |
| [NT-019](#nt-019) | P3 | polish | implementer | todo | Training-record figures polish (training_dashboard.py, indicator_evolution.py) |
| [NT-020](#nt-020) | P3 | polish | implementer | todo | Head-analytics figures polish at notebook widths (analytics_direction/delta/confidence, analytics_tables) |
| [NT-021](#nt-021) | P3 | polish | implementer | todo | Trading figures polish (trading_dashboard.py, trade_analytics.py) |
| [NT-022](#nt-022) | P3 | polish | implementer | todo | Run-comparison and calibration-explorer figures polish (comparison.py, calibration_plots.py) |
| [NT-023](#nt-023) | P3 | polish | implementer | todo | Theme: one reference dash, a strategy palette, legend fixes (theme.py) |
| [NT-024](#nt-024) | P1 | infra | implementer | dropped | Multi-seed gate runs and judge: gate_run.py --seed and no silent overwrite; check_gates.py judges named runs averaged over seeds |
| [NT-025](#nt-025) | P1 | infra | implementer | done | check.py enforces the 5 MB per-notebook limit of D-013 |
| [NT-026](#nt-026) | P1 | infra | implementer | done | Experiment engine: one scenario and sweep spec, a resumable runner, one run store with an index, one scorer |
| [NT-027](#nt-027) | P1 | refactor | implementer | done | Layering: no circular subpackage imports, one metrics and statistics module, figures only draw |
| [NT-028](#nt-028) | P1 | infra | implementer | done | Stale removal under D-029: every deletion shows evidence of stale and of no effect |
| [NT-029](#nt-029) | P1 | infra | implementer | done | Config metadata for the control panel and search spaces, and a generated config reference |
| [NT-030](#nt-030) | P1 | feature | implementer | todo | Sweeps: quick mode (about 5 minutes) and Optuna mode (measured budget, resumable), `neural-trade sweep` |
| [NT-031](#nt-031) | P1 | feature | implementer | todo | Leaderboard ranked by dev-fold net Sharpe after costs, with guard-rails and test columns that never rank |
| [NT-032](#nt-032) | P1 | feature | implementer | done | Paired comparator for "A beats B" verdicts (D-025) |
| [NT-033](#nt-033) | P1 | feature | implementer | todo | Manual-search baselines: frozen-period twin and classic TA rules tuned by the same search |
| [NT-034](#nt-034) | P1 | feature | implementer | todo | Control-panel notebook 06 (ipywidgets + plotly) |
| [NT-035](#nt-035) | P1 | infra | experimenter | done | GPU measurements: concurrent-runs throughput and deterministic-mode speed |
| [NT-036](#nt-036) | P1 | feature | implementer | todo | Stability invariants in CI (strict mode, masks off) |
| [NT-037](#nt-037) | P1 | feature | implementer | todo | Per-run gradient health at most 2% of sec_per_step, per-term probe behind a flag (absorbs NT-012) |
| [NT-038](#nt-038) | P1 | feature | implementer | todo | Stability harness and config guard (refuse hyperparameter regions known to fail) |
| [NT-039](#nt-039) | P1 | research | experimenter | todo | Pre-registered A/B: gradient-based loss weighting against today's value calibration |
| [NT-040](#nt-040) | P1 | bug | implementer | todo | Annualisation ignores the bar size (Sharpe and Sortino overstated by sqrt(k) at k-minute bars) |
| [NT-041](#nt-041) | P1 | feature | implementer | todo | Dataset spec and wall-clock configuration (window, horizons, blocks, costs, fingerprint, gaps) |
| [NT-042](#nt-042) | P1 | feature | implementer | todo | Variable number of horizons |
| [NT-043](#nt-043) | P1 | feature | implementer | done | Learned indicators on price against the textbook defaults (notebook 07) |
| [NT-044](#nt-044) | P1 | docs | implementer | todo | Guides for the owner and reviewers, README landing page, ARCHITECTURE |
| [NT-045](#nt-045) | P1 | feature | implementer | todo | Notebook overlap: each figure gets one home |
| [NT-046](#nt-046) | P1 | feature | implementer | done | Indicators package and registry with today's four families |
| [NT-047](#nt-047) | P1 | feature | implementer | done | OHLCV input and the new indicator families, all learnable and on by default |
| [NT-048](#nt-048) | P1 | feature | implementer | todo | Discovered-indicators report: a self-contained interactive HTML report per run |
| [NT-049](#nt-049) | P2 | bug | implementer | todo | Training silently warm-starts from weights in the working directory |
| [NT-050](#nt-050) | P1 | research | experimenter | todo | First real Optuna sweep on the reference setup (learned, frozen twin, TA rules) and the paired verdicts |
| [NT-051](#nt-051) | P1 | research | experimenter | todo | First stability-harness run on the reference setup against its pre-registered thresholds |
| [NT-052](#nt-052) | P1 | research | experimenter | todo | Stability-harness runs for N = 2 and N = 4 horizons on the reference data |
| [NT-053](#nt-053) | P1 | research | lead | done | Window-free plan: a second research round that writes the path, gates and A/B specifications |
| [NT-054](#nt-054) | P2 | performance | implementer | todo | Per-run fixed costs and GPU launches (independent of the window) |
| [NT-055](#nt-055) | P2 | infra | implementer | todo | CI failure annotations: every failed test name readable, right paths for class tests and collection errors |
| [NT-056](#nt-056) | P3 | infra | implementer | todo | CI and packaging hygiene: CI lints tests, actions off Node 20, the viz extra and jinja2 |
| [NT-057](#nt-057) | P3 | polish | implementer | todo | Random-null follow-ups: one mean-size definition, labels with the size, the CLI prints the matched null |
| [NT-058](#nt-058) | P2 | polish | implementer | todo | Indicator views: which period is 'learned', RSI smoothing named, no private cross-module helpers |
| [NT-059](#nt-059) | P1 | infra | implementer | done | Window-free benchmark kit in scripts/bench/ (kernel V1, assembly D6b, the A2 and today's layers, op census, TF32 check) |
| [NT-060](#nt-060) | P1 | infra | experimenter | todo | GPU run of the window-free benchmark kit (G-A2) |
| [NT-061](#nt-061) | P1 | decision | lead | todo | TF32 decision for the indicator layer (plan stage 1b) |
| [NT-062](#nt-062) | P2 | feature | implementer | todo | VAL_BATCH_SIZE key (validation grouping independent of the training batch) |
| [NT-063](#nt-063) | P1 | feature | implementer | todo | Engine options for pre-registered studies: lambdas once per study, per-arm EPOCHS, cap-extension re-runs, contention records |
| [NT-064](#nt-064) | P1 | feature | implementer | todo | Series kernel V1 and assembly D6b in the indicators package (plan stage 3) |
| [NT-065](#nt-065) | P1 | feature | implementer | todo | INDICATOR_MEMORY switch: the series engine with per-bar adaptation, M_run, burn-in, history bound and pass budget (plan stage 4a) |
| [NT-066](#nt-066) | P1 | feature | implementer | todo | Purge-rule test and config guard (D-034) |
| [NT-067](#nt-067) | P1 | feature | implementer | todo | Predictor series mode and bundle metadata |
| [NT-068](#nt-068) | P1 | feature | implementer | todo | Reporting of per-bar periods and bound counts |
| [NT-069](#nt-069) | P1 | research | experimenter | todo | A/B-1: the series engine against the window engine (pre-registered) |
| [NT-070](#nt-070) | P1 | research | experimenter | todo | A/B-1b: removing the 60-bar clip in series mode (pre-registered) |
| [NT-071](#nt-071) | P2 | research | experimenter | todo | GPU probe: epoch- or update-bound on 7-day blocks with the adopted engine |
| [NT-072](#nt-072) | P2 | feature | implementer | todo | Per-bar causal model as a Models registry entry (option B) |
| [NT-073](#nt-073) | P2 | research | experimenter | todo | A/B-2: the per-bar model against the default (pre-registered) |
| [NT-074](#nt-074) | P1 | bug | implementer | todo | Same-seed runs differ at epoch 0 with op determinism on: find and fix the source |
| [NT-075](#nt-075) | P1 | performance | experimenter | todo | Did sec_per_step regress on the MVP-1 head? (0.1066 vs 0.0984, one run each) |
| [NT-076](#nt-076) | P1 | feature | implementer | done | Engine: store each cell's predictions; `scenario rescore` compares strategies on stored cells (CPU) |
| [NT-077](#nt-077) | P1 | feature | implementer | done | Target-exposure backtest mode and the shortlisted variance-driven strategies with EWMA twins |
| [NT-078](#nt-078) | P1 | research | implementer | todo | EWMA and HAR-RV variance baselines in the evaluation report, same block, with the DM test |
| [NT-079](#nt-079) | P3 | bug | implementer | todo | Strategy study spec: values of cal-fitted strategies are not checked at load |
| [NT-080](#nt-080) | P2 | feature | implementer | todo | Exposure-aware backtest views; explorer hides fitted knobs; YAML loading of cal-fitted strategies |
| [NT-081](#nt-081) | P2 | bug | implementer | todo | Timing null replay micro-rebalances after a capped fill; a full target leaves cash negative by the costs |
| [NT-082](#nt-082) | P1 | feature | implementer | done | Long-history run: SHUFFLE_BUFFER setting and notebook 08 (launch and live progress of an engine run) |
| [NT-083](#nt-083) | P1 | bug | implementer | done | Adding a Config field changes every engine cell's config_hash: finished cells re-run and rescore skips them |
| [NT-084](#nt-084) | P3 | polish | implementer | todo | Notebook 08 / longrun.py edge cases; the notebook kernel's PYTHONPATH in worktrees |
| [NT-085](#nt-085) | P1 | research | lead+experimenter | in-progress | The micro loop (D-041): minutes-long runs iterating toward predictive power and PnL |
| [NT-086](#nt-086) | P1 | bug | implementer | todo | The slow notebook test runs the main checkout's src/ from a worktree (false passes) |
| [NT-087](#nt-087) | P1 | feature | implementer | done | pnl_utility objective: net P&L after costs on the direction heads (P&L plan E2) |
| [NT-088](#nt-088) | P1 | feature | implementer | done | Screen mode: mass ultra-small runs (6-hour training block) for maths, hyperparameters and losses |
| [NT-089](#nt-089) | P2 | bug | implementer | todo | HD physics term: a +inf bar in x_window sends NaN gradients to the variance heads even at LAMBDA_HD 0 |
| [NT-090](#nt-090) | P2 | bug | implementer | todo | pnl_utility: sigma floor 1e-6 makes flat windows a 2600x cost; config guard; test pins |
| [NT-091](#nt-091) | P2 | bug | implementer | todo | DATA_END protection: floor for short files and outside screen mode; screen resume across shard counts; first-trial windowing of the whole file |
| [NT-092](#nt-092) | P1 | feature | implementer | done | Screen phase 2: reuse the traced graph across trials (trace is 73% of a 6-hour trial); clip rule skips the first epoch |
| [NT-093](#nt-093) | P2 | bug | implementer | todo | Identity follow-ups: notebook 08 launch guard trusts the recorded hash; screen trial keys moved once; config_identity docs |
| [NT-094](#nt-094) | P1 | feature | implementer | done | Trading costs default to 0 (D-044) |
| [NT-095](#nt-095) | P2 | feature | implementer | done | Notebook 09: the candidate run (training, fit, backtest) |
| [NT-096](#nt-096) | P1 | bug | implementer | todo | Loss hygiene: epsilon inside every batch std, coherence without its zero-gradient parts and logged, stale comments |
| [NT-097](#nt-097) | P2 | feature | implementer | todo | Indicator hygiene: bound the applied period, no meta bias, LR schedule for both optimizers, GRAD_MULT 1, applied-period report |
| [NT-098](#nt-098) | P1 | research | experimenter | todo | Per-term gradient shares measured on real trainings (the probe of NT-037) |
| [NT-099](#nt-099) | P1 | research | experimenter | in-progress | Pre-registered A/B: soft ECE off, and soft ECE plus the vol penalty off |
| [NT-100](#nt-100) | P2 | research | experimenter | todo | Pre-registered A/B: the NLL tail (lower variance weight, Student-t NLL) |
| [NT-101](#nt-101) | P1 | feature | implementer | todo | Gradient-norm loss-weight calibration mode (CALIB_MODE: gradient), default off |
| [NT-102](#nt-102) | P1 | research | experimenter | todo | Re-choose GRAD_CLIP_NORM and the max_clipped_share rule on the cleaned loss |
| [NT-103](#nt-103) | P1 | feature | implementer | todo | Epoch selection on proper scores (EPOCH_SELECT_METRIC); waits for the owner (D-011) |
| [NT-104](#nt-104) | P1 | feature | implementer | todo | Capacity variants through the Models registry, deep direction logit zero-initialised; then the capacity A/B |
| [NT-105](#nt-105) | P2 | feature | implementer | todo | Attention across the indicator channels and pooling instead of Flatten; then an A/B |
| [NT-106](#nt-106) | P3 | feature | implementer | todo | MACD parametrised as fast = r x slow; a fast leg may reach the price; then an A/B |
| [NT-107](#nt-107) | P2 | feature | implementer | todo | Scale-free inputs: each window normalised by its own sigma, the dollar target rescaled at the output; then an A/B |
| [NT-108](#nt-108) | P2 | bug | implementer | todo | Stochastic-layer reset seeds derived from model.submodules position: any new tf.Module attribute silently changes screen-mode numbers |
| [NT-109](#nt-109) | P2 | performance | implementer | todo | Shrink the six slowest fast-suite tests (28-55 s default-config trainings) |

## Items

### NT-001

**CI unit job green again (red since fe4ba85)**

- **status:** done
- **priority / type / role:** P0 / bug / implementer
- **area:** .github/workflows/ci.yml, requirements-ci.txt, tests/, plotly/pandas-version-sensitive code in src/neural_trade/visualization and src/neural_trade/notebook
- **why:** The GitHub Actions 'ci' workflow fails on be93193 (run 36043324055) and fe4ba85 (run 36032011043), in the step 'Unit tests (CPU, no slow/gpu)'; lint passes; the last green run was 609d19e. Reproduced locally (2026-09-25) by emulating CI's plotly 5.24.1, which writes figure arrays as JSON lists (RUNBOOK 'CI': the `plotly5_lists` pytest plugin): `tests/test_viz_delta.py::test_no_empty_panel_and_size_budget_on_a_full_size_block` (1,295,932 > 600,000 chars) and `tests/test_viz_trading.py::test_size_budget_x0_dx_and_float32` (760,043 over its budget) fail; both pass under the local plotly 6.7.0. Other pins drift too (pandas 2.0.3 vs 2.3.3, scikit-learn 1.3.2 vs 1.7.2) and may hide further failures. Every item's definition of done needs CI green.
- **acceptance:** (1) The unit step writes junit XML and a failure step turns failed tests into `::error` annotations, so failing test names are readable without auth (checks API). (2) `requirements-ci.txt` matches the tested local env for plotly, pandas, scikit-learn and scipy (preferred: CI tests what the owner runs; TF stays 2.10.x), or the size budgets are measured version-independently; RUNBOOK 'CI' says which. (3) The `ci` workflow is green (lint, unit incl. coverage gates and CLI smoke) on the pushed head of `nt-001` and, after the merge, of `remediation/plan`. (4) Local fast suite and ruff still pass.
- **source:** GitHub Actions API (runs 36043324055, 36032011043); plan 'Definition of done' (CI + nightly green); requirements-ci.txt vs local plotly 6.7.0 / pandas 2.3.3
- **evidence (done 2026-09-28):** implemented by a remote session (PR #14, e3fd63e; D-033), merged as 6d01d11. The cause was wider than the 'why' says: 11 failures at 7785ec9 (8 jinja2 missing for `DataFrame.style`, 1 plotly 5 unable to load the plotly-6 notebooks, 2 size budgets). requirements-ci.txt now pins plotly 6.7.0, narwhals 2.20.0, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.9.3 and jinja2 3.1.6 as the nt env (TF 2.10.1, numpy 1.23.5); the unit step writes junit.xml and failed tests become `::error` annotations; RUNBOOK 'CI' updated. QA PASS (all 4 criteria; annotation step simulated on a real junit file; local fast suite 697 passed, ruff clean). CI green: branch push run 36360621477, pull_request run 36362670048, and remediation/plan 6d01d11 run 36375717146 (lint, unit incl. coverage gates and CLI smoke). Follow-ups: NT-055, NT-056.

### NT-002

**Engine random null ignores position size (sized strategies are ranked against a costlier null)**

- **status:** done
- **priority / type / role:** P0 / bug / implementer
- **area:** src/neural_trade/strategy/strategies.py (RandomSignal), src/neural_trade/strategy/backtest.py (random_same_frequency, backtest), src/neural_trade/notebook/backtest_ui.py (matched_random_null), src/neural_trade/cli.py cmd_backtest, scripts/backtest_gate.py
- **why:** random_same_frequency (backtest.py:271-293) runs RandomSignal, which hard-codes Order size 1.0 (strategies.py:326). enhanced_multi_horizon sizes each position between 0.1 and 1.0 (strategies.py:147-148). After costs, return is mostly cost x size x trade count. So the 'random percentile' printed by 'neural-trade backtest' (cli.py:125-141) and by scripts/backtest_gate.py puts a sized strategy against a null that pays more costs. That number is wrong. The notebooks use the size-matched matched_random_null (backtest_ui.py:196-236), so the CLI and notebook paths give different ranks for the same run.
- **acceptance:** (1) RandomSignal has a field size_frac: float = 1.0 and uses it in its Order. (2) random_same_frequency passes size_frac = the mean decision size. It returns size_frac, random_p05_total_return, random_p95_total_return, random_mean_gross_return and percentile_gross_return. (3) matched_random_null is a thin wrapper over it. (4) New test: for a strategy with sizes < 1, every null trade's size equals the mean size, and the engine and notebook paths give an identical percentile_total_return on the same seeds. (5) The existing backtest and strategy tests pass, including assert_no_lookahead for all strategies.
- **source:** integration_todo.md trade_analytics requests (RandomSignal size_frac, random_same_frequency size_frac: carried over); verified in code 2026-09-25
- **evidence (done 2026-09-28):** branch nt-002, b0fd0cf, merged as a0fcc8e. `RandomSignal.size_frac`; `random_same_frequency` sizes the null by `_mean_fill_size` (mean of the clipped sizes of the opened positions) and returns size_frac, random_p05/p95_total_return, random_mean_gross_return, percentile_gross_return; `matched_random_null` is a one-call wrapper (`_Sized` removed: stale after the change, no effect: notebook outputs bit-identical on 147 probes and on notebook 02's run; D-029 evidence in QA's report); cli.py and scripts/backtest_gate.py unchanged (they call `backtest()`). tests/test_random_null.py (5 tests; 4 fail on ffb1b67). Synthetic no-edge fixture (800 bars, enhanced_multi_horizon, 207 trades, mean size 0.724): engine-path percentile 100 -> 23.3, equal to the notebook path. Full-size strategies' numbers unchanged (gate m6 test block: calibrated_quantile 120 trades, 95th percentile = runs/gates/m6/backtest.json). QA PASS (fast 702 passed, slow 13 passed, ruff clean). CI green on a0fcc8e (run 36377389290). Follow-ups: NT-057.

### NT-003

**Direction skill: pass the M3 direction clauses (AUC h1 > 0.52, val MCC h1 > 0.02, Gaussian readout MCC > 0)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** configs/default.yaml, src/neural_trade/models/gru_attention.py (direction heads and skip), src/neural_trade/losses/functions.py, an engine scenario spec (NT-026), runs/experiments/direction_v2/
- **depends on:** NT-026 (experiment engine), NT-031 (leaderboard), NT-032 (paired comparator)
- **why:** M3 fails on the final code (runs/gates/REPORT.md, m6): test AUC h1 0.5015 (0.5239 with the deadband mask), best-epoch val_dir_mcc_h1 0.0175 against the 0.02 threshold, and val_gauss_dir_mcc_h1 -0.0219. The latest default run (runs/20260924T182915Z-1aeff1c-dirty-af67ee43, batch 256, served epoch 19) scores test AUC h1 0.478 and MCC -0.034. The logreg_lags baseline scores 0.523 on the same block, and a logistic regression on trailing returns reaches about 0.56 on fold -1 (direction_v1 REPORT section 1). Identical GPU runs differ by 0.01-0.05 AUC, so no single run can decide. D-010's claim that batch 256 matches the batch-64 run m6 rests on one run. The research tracks run as scenarios through the engine (D-021), so this item waits for the engine, the leaderboard and the comparator instead of the frozen gate scripts (D-023).
- **acceptance:** (1) A pre-registered spec in runs/experiments/direction_v2/: one hypothesis and at most 3 variants, all within gru_attention (new architectures are out of scope), run as an engine scenario (NT-026). Batch 64 vs 256 is a candidate variant. (2) Variants are chosen on dev folds -3/-2 only, with at least 3 seeds each. (3) One judgement on fold -1 by the engine's scorer over the scenario's runs: averaged over the judgement seeds of (4) (at least 5), test AUC h1 > 0.52 on all 7,236 test rows, best-epoch log_val_dir_mcc_h1 > 0.02 and best-epoch log_val_gauss_dir_mcc_h1 > 0 (the M3 clauses). The mean AUC h1 is also >= the logreg_lags baseline on the same block (the M3 point-estimate clause). (4) Whether the model beats logreg_lags beyond the noise is the paired comparator's verdict (NT-032, D-025), with the minimum effect fixed in the spec. A verdict's pairs are (seed, fold) over judgement folds that no choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds today; more held-out folds from the long history once NT-041 exists); at least 5 pairs (lead's reading of D-025, OPERATING_MODEL "Sweeps and pre-registered studies"). (5) REPORT.md lists per-seed, per-fold AUC with 80-bar block-bootstrap CIs, and the variants appear on the leaderboard (NT-031). A negative verdict meets the criteria when the report holds all of this. The Gaussian-readout clause depends on the price heads (NT-004): the SPEC may spend one variant on it, or state that the clause is judged with NT-004.
- **source:** runs/gates/REPORT.md (M3 FAIL); runs/experiments/direction_v1/REPORT.md sections 1, 4, 5; eval_report_test.json of 20260924T182915Z-1aeff1c-dirty-af67ee43; docs/DECISIONS.md D-010, D-021, D-025; owner Q&A 2026-09-28 (round 2: R2-R4 become engine scenarios)

### NT-004

**Price heads: a served delta with positive EV (M3 EV clause)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** src/neural_trade/calibration/pipeline.py (delta shrinkage fit), src/neural_trade/training/trainer.py / callbacks (checkpoint criterion), src/neural_trade/models/gru_attention.py (price towers), an engine scenario spec (NT-026), runs/experiments/price_heads_v1/
- **depends on:** NT-026 (experiment engine), NT-031 (leaderboard), NT-032 (paired comparator)
- **why:** The M3 clause EV(delta) h1 > 0 (served) cannot pass while delta shrinkage sets beta_h1 = 0: the served delta is then exactly 0. In m6, served EV was 0.0000 (beta 0) and raw EV -0.6555. The latest run has betas h0 0.123 / h1 0.000 / h2 0.069, with raw EV h1 -0.061. The raw heads overfit. Early stopping watches the total val loss, which keeps improving through the direction and variance terms, so it cannot catch this (direction_v1 section 6). The beta fit is plain OLS clipped to [0, 1] (pipeline.py:222), and a few high-leverage points drive it: on cal, h0 beta goes from 0.212 to 0.155 without the top 0.5% |pred|, and h1 from 0.023 to negative.
- **acceptance:** (1) A pre-registered spec with at most 3 variants, run as an engine scenario (NT-026), for example early stopping / checkpoint on per-head val point loss, stronger tower regularisation, or a robust (trimmed or Huber) beta fit. Variants are chosen on folds -3/-2 x at least 3 seeds. (2) Judged once on fold -1: served EV(delta) h1 > 0 with beta_h1 > 0, on the mean over the judgement seeds (at least 5) and on every one of them, with the raw EV reported alongside. Any 'A beats B' claim in the report is the paired comparator's verdict (NT-032, D-025). A verdict's pairs are (seed, fold) over judgement folds that no choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds today; more held-out folds from the long history once NT-041 exists); at least 5 pairs (lead's reading of D-025, OPERATING_MODEL "Sweeps and pre-registered studies"). (3) CalibrationPipeline records both the OLS beta and any robust beta in pipeline_meta.json (unit test). (4) Report at runs/experiments/price_heads_v1/REPORT.md, and the variants appear on the leaderboard (NT-031). A negative result meets the criteria when the report is complete.
- **source:** runs/gates/REPORT.md (M3, m6: EV served 0.0000, raw -0.6555); runs/experiments/direction_v1/REPORT.md section 6; integration_todo.md delta 'calibration/pipeline.py (finding 4)'; owner Q&A 2026-09-28 (round 2)

### NT-005

**Cost-aware trading: an edge per trade above the 26 bps round trip**

- **status:** done (2026-09-29), acceptance (b), the negative result: runs/experiments/strategy_study_v1/REPORT.md (SPEC e2b0b74 before scoring; 9 reference-scenario cells at 82a848f, 0.56 GPU-hours; rescore runs/scenarios/reference_default/rescore/strategy_study_v1-20260929T095605Z at 6aa7aaf). 12 candidates (volatility targeting, regime stand-aside, net-edge Kelly, edge over cost, variance-gated TA, the incumbent) and 8 EWMA twins: none passes the guard-rails, all 12 lose on dev; winner always_flat. Every discrete candidate's gross edge per trade has its 80-bar block-bootstrap 95% CI below 26 bps (largest mean +4.4 bps); the incumbent calibrated_quantile is the worst of 20 (-62% per dev block, +0.47 bps gross per trade on 2,357 trades). The EWMA twin beats the model's sigma for vt and rs. QA PASS on 981cc1e (rescore bit-identical on rebuild, guard-rails recomputed independently). The default strategy is left to the owner (STATUS). Leaderboard and comparator: NT-076's rescore leaderboard stood in for NT-031; no A-beats-B claim is made, so NT-032 was not needed. Research: docs/research/2026-09-29-strategy-architectures/.
- **priority / type / role:** P1 / research / experimenter
- **area:** src/neural_trade/strategy/strategies.py (new or re-knobbed strategy), configs/strategies/, an engine scenario spec (NT-026), runs/experiments/trading_costs_v1/
- **depends on:** NT-002, NT-007, NT-026 (experiment engine), NT-031 (leaderboard), NT-032 (paired comparator)
- **why:** No strategy has made money after costs. In gate m6, calibrated_quantile made 120 trades: gross +4.1% (the same as buy-and-hold), net -23.4% after 26 bps per round trip (the reference setup's default costs, 13 bps per side). In the latest notebook run, calibrated_quantile made 147 trades, liberal 58 and enhanced_multi_horizon 0. Trades almost never move in their favour by more than the 0.26% round-trip cost, and on the reference setup a 15-minute move has a standard deviation of about 21 bps (direction_v1 REPORT section 5).
- **acceptance:** Either (a) a registered strategy whose knobs are fixed on the cal block or dev folds only, with all of these on test: net total return > 0 after the default costs (fee 10 + half-spread 1 + slippage 2 bps per side) on fold -1; size-matched random-null percentile (after costs) >= 95; and net > 0 on at least 2 of 3 walk-forward folds. Or (b) a negative-result report at runs/experiments/trading_costs_v1/REPORT.md with, per strategy and horizon, the mean gross edge per trade in bps and its 80-bar block-bootstrap 95% CI against the 26 bps cost, and the break-even round-trip cost. In both cases: a pre-registered spec with at most 3 variants (for example: enter only when |expected move| > cost + k x sigma; longer holds; fewer trades), run as an engine scenario (NT-026) and shown on the leaderboard (NT-031); any 'A beats B' claim in the report is the paired comparator's verdict (NT-032, D-025).
- **source:** runs/gates/REPORT.md 'Reading' (Trading); README status; notebooks/02_backtest.ipynb saved outputs; runs/experiments/direction_v1/REPORT.md section 5; owner Q&A 2026-09-28 (round 2)

### NT-006

**Physics-term ablation re-run on the current trainer, pre-registered under D-025**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** an engine scenario spec (NT-026), a new v2 criteria file under configs/, runs/ablations/ablate_physics_v2-full/, notebooks/05_compare_runs.ipynb (ABLATION_DIR)
- **depends on:** NT-026 (experiment engine), NT-032 (paired comparator); owner approval of the GPU time (v1 was 84 runs at 5-7 min each, about 7-10 GPU-hours, over the 3-hour limit that pre-registered A/B studies keep under D-024; asked on 2026-09-25 in STATUS 'Waiting for the owner', not asked again); NT-009 (disk); soft: NT-037 (so the grid logs per-term contributions and gradient health; it absorbed NT-012)
- **why:** ablate_physics_v1-full ran at commit 6dec27a, before 53c0df2 (D-011), so every cell was scored on its last epoch, not the best-validation one. It also ran before the batch-256 default. The family verdict is INCONCLUSIVE: its variance VALUE (CRPSS +0.0056, var/err^2 Spearman +0.0364) is withdrawn by the h1 direction-AUC guard-rail breach (-0.0119 against tolerance 0.01). No single term reaches VALUE. The pre-registered rule for INCONCLUSIVE is 'add seeds'. D-025 supersedes v1's verdict rule for new studies: a paired test over (seed, fold) pairs plus a pre-registered minimum effect, with guard-rails judged by the same test instead of a point tolerance. v1 stays the record under its own criteria (D-003).
- **acceptance:** (1) A SPEC committed before any GPU time: per term and for the family, the metrics, the minimum effects and the guard-rails, all judged by the paired comparator (NT-032); one condition per term plus the family (the ablation exception to 'at most three variants', OPERATING_MODEL "Sweeps and pre-registered studies"; D-003); the GPU-time estimate. A verdict's pairs are (seed, fold) over judgement folds that no choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds today; more held-out folds from the long history once NT-041 exists); at least 5 pairs (lead's reading of D-025). configs/ablation_criteria.yaml (v1's criteria) is unchanged; v2's criteria are a new file. (2) The grid runs as an engine scenario (NT-026); every cell completed, and each run's status.json has weights_epoch (trained at a commit >= 53c0df2). (3) runs/ablations/ablate_physics_v2-full/ is committed with report.md and the summary files the engine writes (per NT-010's policy). (4) report.md gives per-term and family verdicts from the paired comparator, the guard-rail results by the same test, and v1's verdicts beside them for reference. (5) Notebook 05 points ABLATION_DIR at v2 and is re-executed per D-013.
- **source:** runs/ablations/ablate_physics_v1-full/report.md; docs/DECISIONS.md D-003, D-011, D-024, D-025; computed task thread (3); owner Q&A 2026-09-28 (round 7)

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

**Disk: C: nearly full; resolved by moving the project to D: (D-030)**

- **status:** done (2026-09-29): moved to D:/neural_trade and verified (acceptance 1-4, 2026-09-28); the owner closed (5): "ignore c copy" (D-038): the C: copy is left as it is and not deleted. The local 2017-2025 data file stays (VISION "The reference setup").
- **priority / type / role:** P1 / owner-decision / owner
- **area:** machine: C: and D: drives; the repo, its untracked runs/, the sibling worktrees neural_trade_gates and neural_trade_ablation, the editable install in the nt env, the Claude memory folder; docs/RUNBOOK.md, CLAUDE.md
- **why:** C: had 6.0 GB free (98% used) on 2026-09-25. The Docker WSL image (C:\Users\Step\AppData\Local\Docker\wsl\disk\docker_data.vhdx) is about 116-119 GB and belongs to the owner's other project. During the review, C: reached 0 bytes, and pytest, git index writes and figure renders all failed. GPU grids, Optuna studies, notebook executions and renders need headroom. The owner's answer (owner Q&A 2026-09-28, rounds 5b and 6): move the whole project to D: (repo with its untracked runs, both sibling worktrees, the editable install re-pointed with `pip install --no-deps -e .`, the Claude memory carried); runs, the Optuna studies and the run index live in the repo on D:; after the owner confirms the D: copy works, the lead deletes the C: copy (repo and both worktrees) only on the owner's explicit go-ahead. The local 2017-2025 file stays: Bitcoin_BTCUSDT.csv (291 MB, git-ignored, 2017-01-01 to 2025-09-29), used for walk-forward folds over months (NT-041).
- **acceptance:** (1) The repo (with its untracked runs), the two sibling worktrees and the Claude memory folder are on D:, and `git worktree list` run in D:/neural_trade shows every worktree at an existing D: path. (2) In the nt env, `python -c "import neural_trade; print(neural_trade.__file__)"` prints a path under D:/neural_trade/src. (3) On D:, the fast suite passes and scripts/notebooks/check.py passes. (4) docs/RUNBOOK.md and CLAUDE.md give the D: paths. (5) The C: copy is kept: after the owner confirms the D: copy works, the lead deletes the C: copy (repo and both worktrees) only on the owner's explicit go-ahead.
- **source:** integration_todo.md (ENVIRONMENT notes in training, misc, tables); memory neural-trade-env; df on 2026-09-25; owner Q&A 2026-09-28 (rounds 5b, 6); docs/DECISIONS.md D-030

### NT-010

**Every cited number links to a tracked run (run-tracking policy, clean git status)**

- **status:** done
- **priority / type / role:** P1 / infra / implementer
- **area:** .gitignore, runs/, a checker script or test, docs/RUNBOOK.md
- **why:** git status is never clean after a session. 7 notebook run dirs, the ablation's cells/, logs/ and runs/, and direction_v1's logs/ and runs/ are untracked, while the session protocol requires a clean tree. The saved notebook outputs and docs/STATUS.md cite runs, for example 20260924T182915Z-1aeff1c-dirty-af67ee43 (4.5 MB), whose directories exist only on this machine. The status numbers now live in docs/STATUS.md (the README links to it, README.md:25), and nothing checks that each of them cites a run id. This is a Definition of Done gap: 'every reported number links to a run'.
- **acceptance** (tightened 2026-09-28 by the lead: every non-heavy run file is tracked, so one rule keeps git status clean; measured on the 2026-09-28 tree: 1,060 light files, 48 MB raw, about 15 MB compressed, one-off; all 91 cited run ids resolve to a directory): (1) docs/RUNBOOK.md states the run-tracking policy: in a run directory (a directory under runs/, at any depth, whose name starts with a run id `YYYYMMDDTHHMMSSZ-<sha7>[-dirty]-<hash8>`), every file is tracked except the ignored heavy artefacts (weights `*.h5`, `*.joblib`, `*.pkl`, `*.npz`, `*.parquet`, `tb/`) and the experiment log folders (`runs/ablations/*/cells/`, `runs/ablations/*/logs/`, `runs/experiments/*/logs/`); it names the tracked files (config.yaml, meta.json, status.json, env.json, metrics.jsonl, eval_report_*.json/md, training_log.csv, indicator_params_history.csv, period_init.json, artifacts/meta.json, artifacts/config.yaml, artifacts/calibration/*.json, result.json) and says how a new run's light files are staged (the checker's list mode, explicit paths; never `git add runs`). (2) .gitignore implements it: a test copies the repo's .gitignore into a tmp git repo with a fake run directory holding each heavy kind and each light kind, and `git status --porcelain --untracked-files=all` there lists exactly the light files. (3) `scripts/check_run_evidence.py` finds every run id cited in docs/**/*.md, README.md, runs/**/REPORT.md, runs/**/report.md, runs/**/summary.md and the saved notebook outputs (notebooks/*.ipynb), resolves each to its run directory, and exits 0 only when, for each: git tracks at least config.yaml and meta.json in that directory, and, if the directory exists on disk, no light file in it is untracked. Otherwise it exits 1 and names each failing run id, a file that cites it and the missing paths; a cited id with no directory in git or on disk fails. `--list-untracked` prints the untracked light files of the cited runs, one repo-relative path per line. (4) Tests with fixture git repos in tmp_path: a fixture docs/STATUS.md citing a run whose light files are untracked fails, one citing a tracked run passes; citations in a notebook output and in a runs/**/report.md are found; a cited id without a directory fails; heavy files never count as missing. (5) A fast-suite test runs the checker on the real repo, so CI enforces it; it fails on `nt-010` until the lead stages the light files after the merge (git refuses a merge that would overwrite the main checkout's untracked run files), and passes locally and in CI after that commit. (6) The fast suite (apart from (5) on the branch) and ruff pass. (That every STATUS number cites a run id is ROADMAP R1's exit, which the lead ensures in step 7.)
- **source:** plan 'Definition of done' (every reported number links to a run); git status at session start; docs/OPERATING_MODEL.md session protocol (clean tree)
- **evidence (done 2026-09-29):** branch nt-010, b0be1d1, merged as 4e679cf; the lead staged the light files in 349b168 (1,060 files, 48.3 MB raw, 14.9 MB on disk in git) and updated the docs in b59eafb. scripts/check_run_evidence.py: 112 cited run ids in 10 files (not 91: runs/experiments/direction_v1/summary.md cites 21 more), all tracked; 9 tests in tests/test_run_evidence.py, including the real-repo test (failed on b0be1d1 in CI run 36377609475, passes at b59eafb). Main checkout: git status empty; 1,612 files under runs/ = 1,122 tracked + 490 ignored (h5, joblib, npz, cells/, logs/). QA PASS (fast 714 passed, ruff clean; shallow-clone check passes). Follow-ups fixed by the lead: CLAUDE.md step 2 (uncited runs), RUNBOOK (result.json location). Not enforced (P3, for the backlog): partial citations such as '20260924T182915Z' alone.

### NT-011

**Notebook output policy: remove nbstripout (it contradicts D-013)**

- **status:** done (2026-09-25, lead edit in a0edba7; checked by the setup QA pass): `git check-attr filter notebooks/01_train_and_monitor.ipynb` reports unspecified; no nbstripout hook; the pre-commit header no longer mentions it; D-013 cites NT-011.
- **priority / type / role:** P2 / infra / implementer
- **area:** .gitattributes, .pre-commit-config.yaml, docs/DECISIONS.md D-013
- **why:** .gitattributes marks *.ipynb with filter=nbstripout, and .pre-commit-config.yaml runs the nbstripout hook. Its header tells every clone to run 'nbstripout --install'. D-013 (owner) requires notebooks to be committed with real outputs, and tests check the saved outputs (tests/test_notebooks_thin.py::test_saved_training_dashboards_mark_each_chance_band_edge_in_its_horizon_colour, tests/test_viz_variance.py::test_saved_notebook_outputs_carry_the_current_encoding). A future session that follows the pre-commit instructions would strip every output on its next commit.
- **acceptance:** (1) 'git check-attr filter notebooks/01_train_and_monitor.ipynb' reports 'unspecified'. (2) .pre-commit-config.yaml has no nbstripout hook, or its hook excludes notebooks/. (3) The install comment in its header no longer mentions nbstripout. (4) D-013 cites the change.
- **source:** .gitattributes line '*.ipynb text eol=lf filter=nbstripout'; .pre-commit-config.yaml; docs/DECISIONS.md D-013; plan B7 (superseded by D-013)

### NT-012

**Training logging: per-term loss contributions, gradient max and clip counts, deadband sample counts**

- **status:** dropped (2026-09-28): absorbed into NT-037 (per-run gradient health), whose acceptance carries these criteria.
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

**One overlap-aware interval toolkit in the shared statistics module (reliability bands, PIT band per bin)**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** the shared statistics module of NT-027 (today src/neural_trade/visualization/stats.py), analytics_direction.py, calibration_plots.py, analytics_variance.py, analytics_delta.py, analytics_confidence.py, analytics_common.py
- **depends on:** NT-027 (one metrics and statistics module)
- **why:** D-012 sets the convention, but the figures implement it differently. (1) Reliability bands are HAC with lag = horizon in the calibration explorer (calibration_plots.py:6) and 80-bar clusters in the direction figure; on the same notebook 04 h1 bins the half-widths differ by up to about 25%. (2) The direction figure's AUC CI, chance band and DeLong test use n/steps (1.4-1.9x wider than a block bootstrap), while its reliability bars use clustered SEs. (3) The variance figure's PIT noise band uses the mean overlap factor over all bins (analytics_variance.py:473). For the edge bins, where a U-shape is read, that is too narrow: the edge bins need about ±0.22-0.30, the drawn band is ±0.15-0.18. (4) The helpers (lrv_factor, circular block bootstrap, Newey-West variance of a mean, skill_half_width, design_effect, cluster_mean_ci) live in figure modules. (5) analytics_common._binned (i.i.d. SE) has no caller left. NT-027 creates the one statistics module these helpers move into.
- **acceptance:** (1) Those helpers live in the shared statistics module of NT-027 with unit tests, and the figure modules import them from there. (2) calibration_plots.reliability_table and the direction figure's reliability rows call one function and give identical half-widths on the same input (test). (3) Variance row 2 draws a per-bin PIT band as step lines, and a test asserts that the edge-bin half-width is >= the centre-bin one on overlapping heavy-tailed data. (4) Each figure's subtitle names its interval method. (5) analytics_common._binned is removed. (6) Notebooks re-executed and changed figures rendered, per D-013.
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
- **priority / type / role:** P3 / owner-decision / owner
- **area:** LICENSE, README.md 'Licence' section
- **why:** There is no LICENSE file. README 'Licence' says the choice belongs to the repository owner, and the plan's B16 lists a LICENSE. Nothing is blocked by it, but the repository is public on GitHub without terms. The owner left the licence open (owner Q&A 2026-09-28, round 9); the lead lowered it to P3 because the MVP audience, the owner and a few reviewers, does not need one (D-019).
- **acceptance:** A LICENSE file exists at the repo root, or docs/DECISIONS.md records 'no licence (all rights reserved)'. The README 'Licence' section matches.
- **source:** README.md 'Licence'; plan B16 / layout (LICENSE); owner Q&A 2026-09-28 (round 9)

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

### NT-024

**Multi-seed gate runs and judge: gate_run.py --seed and no silent overwrite; check_gates.py judges named runs averaged over seeds**

- **status:** dropped (2026-09-28): superseded by NT-026. The engine runs seeds and folds as one scenario, scores every run with one scorer and never overwrites a run directory; gate_run.py and check_gates.py are frozen as history (D-023).
- **priority / type / role:** P1 / infra / implementer
- **area:** scripts/gate_run.py, scripts/check_gates.py, scripts/backtest_gate.py, tests/test_gate_scripts.py (new)
- **why:** NT-003 and NT-004 must be judged on fold -1 averaged over at least 3 seeds, but `check_gates.py` only knows the fixed runs m1a..m6 and judges single runs, and `gate_run.py` deletes an existing run directory without asking (gate_run.py:157-158), which can destroy evidence. The experimenter may not write this code (it never edits src/ or tests/; scripts it may extend only via an implementer item).
- **acceptance:** (1) `gate_run.py` refuses an existing non-empty run directory unless `--overwrite` is given (test). (2) `gate_run.py --seed N` sets SEED and records it in meta.json (test). (3) `check_gates.py --runs NAME ...` (or `--glob`) judges the M3/M4 clauses per named run and prints the per-run values and their mean and sd; the mean decides the exit code; with no arguments its output is unchanged (test on small fixture run directories). (4) The logreg_lags baseline AUC h1 on the same test block is written to each run's analytics and shown next to the model's. (5) Fast suite and ruff pass.
- **source:** setup QA 2026-09-25 (cold-start research probe): NT-003's criteria could not be met with the existing scripts.

### NT-025

**check.py enforces the 5 MB per-notebook limit of D-013**

- **status:** done
- **priority / type / role:** P1 / infra / implementer
- **area:** scripts/notebooks/check.py, tests/test_notebook_tooling.py
- **why:** D-013 says each notebook stays under 5 MB (the repo's large-file limit), but nothing enforces it; a figure change could silently push a notebook over it (02 was 6.6 MB once).
- **acceptance:** (1) `check.py` exits 1 and names the notebook when a saved notebook exceeds 5 MB (test with a synthetic notebook in tmp_path). (2) The committed notebooks pass. (3) scripts/notebooks/README.md mentions the limit.
- **source:** setup QA 2026-09-25 (fact check of D-013).
- **evidence (done 2026-09-28):** branch nt-025, 66c57c8, merged as 7579e0b. `scripts/notebooks/check.py`: `MAX_BYTES = 5_000_000` (decimal, as check.py prints sizes); a notebook over it fails with `TOO LARGE` and its name, at or under passes; other problems are still reported. Three tests in tests/test_notebook_tooling.py (each shown necessary by a mutation check); the committed notebooks (0.14-1.9 MB) pass; scripts/notebooks/README.md and CLAUDE.md name the limit. Sizes are bytes on disk, equal to the blob size (`*.ipynb eol=lf`). QA PASS (fast 700 passed on the branch; 705 on the merged head 7579e0b, ruff clean). CI green on 330ba2f (run 36378359175).

### NT-026

**Experiment engine: one scenario and sweep spec, a resumable runner, one run store with an index, one scorer**

- **status:** done
- **priority / type / role:** P1 / infra / implementer
- **area:** src/neural_trade/experiments/ (new engine modules next to run_context.py and compare.py), src/neural_trade/cli.py, src/neural_trade/notebook/runs.py (pick_run), a folder of scenario specs (for example configs/scenarios/), the frozen set (D-023; a header note only), docs/RUNBOOK.md and scripts/notebooks/README.md (which run notebooks 02-05 read), tests/
- **depends on:** soft: NT-010 (which run files are tracked)
- **why:** Four experiment paths exist, each with its own run layout, scorer and resume logic: scripts/gate_run.py and its judge scripts/check_gates.py (the gates, fixed run names m1a..m6), scripts/direction_experiments.py (direction_v1), and scripts/ablate.py with experiments/ablation.py (the physics grid); CLI and notebook runs use RunContext. AUC alone is computed separately in scripts/gate_run.py:85, scripts/direction_experiments.py:55 and evaluation/report.py:116. There is no index across runs, no run records the data file it used, and gate_run.py deletes an existing run directory without asking (gate_run.py:157-158). The yardstick (VISION; D-020) needs every run scored the same way on the dev folds and the test fold, and the sweeps, leaderboard and comparator (NT-030 to NT-032) need one store to read. D-023: incremental restructure, engine first; the old paths are frozen as history (the frozen set, D-023). Supersedes NT-024. notebook/runs.py pick_run takes the newest run anywhere under runs/ (rglob, runs.py:15), so without a guard notebooks 02-05 would load an engine cell (a quick-mode or frozen-twin trial) as 'the latest run'.
- **acceptance:** (1) A scenario and sweep spec in YAML: a base config, overrides, variants, sweep axes, folds and seeds; unknown keys and invalid Config values are refused before any run starts (test). An example spec for the reference setup is committed. (2) A resumable runner: each (variant, fold, seed) cell is one run directory in one run store (default runs/), recorded in an sqlite index with at least the run id, scenario, cell key, status, commit, config hash, dataset fingerprint and scores. A test stops a tiny CPU scenario (2 folds x 2 seeds) after its first cell and resumes it: finished cells are not re-run, and the index equals that of an uninterrupted run. (3) The runner never overwrites or deletes a non-empty run directory (test). (4) One scorer: every run gets an eval report for the out-of-sample block of its fold, labelled dev (the earlier folds) or test (the last fold), with the metrics of evaluation/report.py plus the backtest numbers the leaderboard needs: net Sharpe after costs, max drawdown, trade count, buy-and-hold and the size-matched random null (test on the tiny scenario). The backtest uses the scenario's strategy (default Strategies.default = calibrated_quantile, D-009), its knobs fitted on the fold's cal block only, next-open fills and the default cost profile (test). (5) Each run's meta.json records the dataset fingerprint (file sha256, first and last timestamp, bar count) and the setup as configured today (bar minutes from RESAMPLE_MINUTES, LOOKBACK and HORIZON_STEPS; NT-041 adds the symbol and wall-clock units) (test). (6) The frozen set (D-023) carries a header note naming the engine as its replacement and is otherwise unchanged, and its existing tests pass. No new code imports it; experiments/ablation.py stays importable (visualization/comparison.py:653 reads its analysis for notebook 05) until the engine subsumes it. (7) The CLI runs and resumes a scenario (test through neural_trade.cli.main). (8) scripts/golden_run.py verify passes against a recording made on the base commit (train_and_evaluate unchanged). (9) Engine runs live in their own subtree (runs/scenarios/<scenario>/...) or are marked in the index, and notebook/runs.py pick_run's default ignores them (test: after a tiny scenario, pick_run still returns the notebook/CLI run).
- **source:** owner Q&A 2026-09-28 (rounds 2, 5b); docs/DECISIONS.md D-020, D-023; survey 2026-09-28
- **evidence (done 2026-09-29):** branch nt-026, 4c72070, merged as aa0753d (+ fd960cd: `runs/index.sqlite*` ignored, RUNBOOK). experiments/{scenario,runner,store,scorer,dataset}.py, `neural-trade scenario run|plan|reindex`, configs/scenarios/reference.yaml; 39 fast + 1 slow tests. QA PASS: 26 bad specs refused before any run; a stopped scenario resumes without retraining and its index equals an uninterrupted run's (also after a doc-only Config change, since identity hashes values only); never overwrites; dev/test scoring with cal-only knobs, next-open fills, default costs and the size-matched null; dataset fingerprint in meta.json; pick_run ignores engine runs; golden run equal (273 arrays); fast 741, slow 14, ruff clean; a tiny real CLI scenario run, stopped, resumed and reindexed by QA. Merged head fd960cd: fast 795 passed. CI green on c5b8a64 (run 36519610045), which contains it. Findings: NT-030 note (cell locking before --parallel), NT-040 note.

### NT-027

**Layering: no circular subpackage imports, one metrics and statistics module, figures only draw**

- **status:** done
- **priority / type / role:** P1 / refactor / implementer
- **area:** src/neural_trade/ (every subpackage), a layering test (for example tests/test_layering.py); scripts/golden_run.py is used, not changed
- **why:** Counting imports inside functions, the subpackages import each other in 10 mutual pairs (2026-09-28): data and registries, data and visualization, evaluation and experiments, evaluation and registries, evaluation and training, losses and registries, metrics and registries, models and registries, registries and training, serving and training; all 12 subpackages with package imports form one import cycle. evaluation/walk_forward.py launches training (it imports experiments.run_context at :47 and training.trainer at :48), and data/processor.py imports figure code (visualization.matplotlib_splits at :109). AUC is computed in five places (evaluation/report.py:116, visualization/analytics_common.py:28 roc_curve, the DeLong placements in visualization/analytics_direction.py:91, scripts/gate_run.py:85, scripts/direction_experiments.py:55). The statistics are split between visualization/stats.py (n_eff, Wilson, AUC CI) and evaluation/report.py (long-run variance, DM test, block bootstrap: report.py:331-419). D-023: module moves one at a time, each checked by `scripts/golden_run.py verify`, with the notebooks working at every step.
- **acceptance:** (1) A layering test builds the subpackage import graph from the source, including imports inside functions, and fails on any mutual import between two subpackages; it passes. (2) evaluation/ imports neither training/ nor experiments/, and data/ does not import visualization/ (covered by the same test). (3) One metrics and statistics module holds the AUC (with its DeLong variance), the effective-sample helpers, the block bootstrap and the long-run variance; evaluation/report.py and the figure modules call it, and a test finds no other AUC implementation in src/neural_trade (roc_auc_score or a rank-based AUC). The frozen set (D-023) is exempt. (4) Figures only draw: no module in visualization/ computes a score that the evaluation report provides; they read it from the report or the statistics module. The overlap-aware interval helpers still in figure modules move in NT-015. (5) `scripts/golden_run.py verify` passes at the item's head against a recording made on the base commit, and the implementer's report lists one verify run per module move. (6) scripts/notebooks/build.py regenerates the notebooks with no drift other than import lines, and the notebook tests pass (tests/test_notebooks_thin.py, tests/test_notebook_tooling.py).
- **source:** owner Q&A 2026-09-28 (round 5b: incremental restructure, golden-run check); docs/DECISIONS.md D-023; survey 2026-09-28 (import graph re-computed from the source on 2026-09-28)
- **evidence (done 2026-09-29; CI on the merged head pending at record time):** branch nt-027, 6a897e3 (10 commits), merged as 968ef51. tests/test_layering.py (fails on fd960cd with the 10 mutual pairs, passes at 6a897e3; QA's independent AST check agrees); src/neural_trade/metrics/statistics.py holds AUC with DeLong, effective-sample helpers, block bootstrap, long-run variance (QA: bit-identical to the old functions); evaluation/plots.py moved to visualization/eval_report.py; callbacks self-register (registry identical). Deletions (D-029 evidence in QA's report): DataProcessor.plot_splits, evaluation.walk_forward.walk_forward. Golden run equal after each of 8 moves. QA PASS (fast 803, slow 14, ruff clean; branch CI 36526346935 green). Merged head 968ef51: fast 807 passed, kit smoke G-A1 PASS. Findings (P3): analytics_direction._horizon_stats computes its own deadband MCC/ECE (NT-015); metrics/evaluate.py hardcodes h0-h2 (NT-042).

### NT-028

**Stale removal under D-029: every deletion shows evidence of stale and of no effect**

- **status:** done
- **priority / type / role:** P1 / infra / implementer
- **area:** src/neural_trade/compat.py, src/neural_trade/models/facade.py, src/neural_trade/training/trainer.py (train_model), src/neural_trade/training/callbacks.py, src/neural_trade/registries/callbacks.py, src/neural_trade/registries/{data_loaders,layers,models}.py (docstrings), src/neural_trade/core/config.py, src/neural_trade/visualization/, REGISTRY_SPECIFICATIONS.md, nt.yml, indicator_params_history.csv (repo root), docs/archive/, tests/
- **why:** The owner's rule (D-029): "only delete something which is both stale and doesn't affect current system in any sense". Candidates from the 2026-09-28 survey: compat.py (the pre-package model.py API; only the two tests that check it use it, tests/registries/test_visualizations.py:40 and tests/test_config.py:108); the PricePredictor facade (models/facade.py; re-exported by compat, and tests/test_train_smoke.py uses it as a helper); train_model() (training/trainer.py:530-578, a legacy wrapper with no caller in src, scripts or notebooks except the compat re-export); figure modules without a production caller; the interactive_plot callback (registries/callbacks.py:60, training/callbacks.py:394; not in the default CALLBACKS, configs/default.yaml:128); the root outputs (the weights file, the scaler and training_log.csv, git-ignored) and the defaults that re-create them: Config.MODEL_PATH and SCALER_PATH (core/config.py:182-183) and RunContext.path's working-directory fallback (training/callbacks.py:288), which the CSV logger uses (training/callbacks.py:331). A run without a RunContext also loads or warm-starts from MODEL_PATH when that file exists (training/trainer.py:365-377), so the root weights file still affects a run started at the repo root: that is a behaviour, not a deletion, and NT-049 handles it. Further candidates: the root indicator_params_history.csv (tracked; cited as evidence by docs/archive/REMEDIATION_PLAN_2026-09.md:7, 37; the per-run copy written by ParamsLogger stays); nt.yml (tracked; referenced only by the archived remediation plan, docs/archive/REMEDIATION_PLAN_2026-09.md:27 (history), and its prefix line 189 carries another machine's path, C:\Users\aegorshev); REGISTRY_SPECIFICATIONS.md (the pre-remediation registry design, cited by the docstrings of registries/data_loaders.py:6, registries/layers.py:8 and registries/models.py:8, and by docs/archive); legacy Config fields (for example DAMPING, core/config.py:88; LAMBDA_LOCAL_TREND, :110; USE_HUBER, :161).
- **acceptance:** (1) For every removed file, function, class, registry entry or config default, the implementer's report gives evidence of both D-029 conditions: a search of src, scripts, scripts/notebooks/build.py, notebooks, tests, docs, configs and pyproject.toml finds no remaining use, and no default re-creates it. A candidate with any effect stays, with the reason in the report. (2) Tests that exist only to check a removed item go with it; a test that uses a removed item as a helper (for example tests/test_train_smoke.py and PricePredictor) is first ported to the current API and passes. (3) With the default Config, a 1-epoch CPU train_and_evaluate without a RunContext creates no file in its working directory that nothing reads (for example training_log.csv) (test in a temporary directory). Files that something reads stay: the MODEL_PATH weights, which the warm start reads (NT-049). (4) REGISTRY_SPECIFICATIONS.md is moved to docs/archive/ with git mv, and the three registry docstrings point there. (5) Legacy Config fields with no effect warn (DeprecationWarning) when set to a non-default value (test); none is removed, so old config files still load. (6) Nothing untracked is deleted: the root weights, scaler and training_log.csv, and anything under runs/, are listed in the report for the owner (models and data need the owner, D-029). Remote branches are out of scope. (7) Fast and slow suites pass, ruff is clean, `scripts/golden_run.py verify` passes, and the notebooks build with no drift.
- **source:** owner Q&A 2026-09-28 (round 6: deletion rule); docs/DECISIONS.md D-029; survey 2026-09-28
- **evidence (done 2026-09-29):** branch nt-028, aab9da4 + repair round 1 (32ec09e), merged as f4dd5e7. Deleted with D-029 evidence (QA re-grepped): src/neural_trade/compat.py, models/facade.py (PricePredictor), trainer.train_model, nt.yml, the DAMPING fallback in lambda_calibration.py (and 3 tests that only checked removed items; test_train_smoke ported). REGISTRY_SPECIFICATIONS.md moved to docs/archive (git mv). Deprecated Config fields warn when set (8, HUBER_DELTA added: no effect). A default-Config run without a RunContext writes only the MODEL_PATH weights (no CSVs, no scaler; test asserts the full file list). Kept: the root indicator_params_history.csv (cited evidence; restored in the repair round), interactive_plot callback, registered figure modules, legacy fields. scripts/gate_run.py (frozen) passes a run_context so its CSV evidence is still written (now also metrics.jsonl, status.json, period_init.json). Also fixed: tests/test_viz_confidence reference made integer-exact (red on CI's BLAS). QA PASS (round 2): golden run equal (273 arrays), fast 806, slow 14, ruff clean, CI 36537956908 green. Findings (P3): deprecation warnings invisible outside pytest (config.py:572-578); SCALER_PATH doc wording; the inert calibration_scaler override in experiments/ablation.py:209; CustomTrainModel.huber dead.
- **note (2026-09-29, NT-029):** further candidates found by NT-029: `HUBER_DELTA` (no caller of `CustomTrainModel.huber` in src, tests or scripts) and the `DAMPING` fallback branch in `lambda_calibration.py:46-48` (dead for a real Config, which always has CALIB_DAMPING). Seven fields are now flagged `deprecated` in the Config metadata (DAMPING, LAMBDA_LOCAL_TREND, LAMBDA_GLOBAL_TREND, LAMBDA_QUANTILE, TANH_SCALE, SIGMOID_SCALE, USE_HUBER).

### NT-029

**Config metadata for the control panel and search spaces, and a generated config reference**

- **status:** done
- **priority / type / role:** P1 / infra / implementer
- **area:** src/neural_trade/core/config.py, a generator script (for example under scripts/), docs/guide/config-reference.md (new, generated), tests/test_config.py
- **why:** The control panel (NT-034) and the search spaces (NT-030, NT-038) need to know, for every Config field, its valid range or choices, its unit, whether a sweep may tune it and whether it is deprecated. Today a field carries only a group and a one-line doc (`_f`, core/config.py:36-41), the valid ranges live in code inside Config.validate (core/config.py:234 onward), and 26 of the 112 fields have no doc at all (2026-09-28): EPOCHS, CALIB_LAMBDA_MIN/MAX, the six CALIB_DAMPING_* fields, LAMBDA_TREND_OUTER, LAMBDA_DIR_OUTER, LAMBDA_COHERENCE, LAMBDA_NLL_OUTER, LAMBDA_CRPS, LAMBDA_SOFT_ECE, MA_SPANS, MACD_SETTINGS, RSI_PERIODS, BB_PERIODS, FOCAL_GAMMA, MODEL_PATH, SCALER_PATH, ADAM_BETA1, ADAM_BETA2, SGD_MOMENTUM, SGD_NESTEROV. D-022 moves the window and horizons to wall-clock time, so units must be explicit.
- **acceptance:** (1) Every Config field declares in its metadata a unit (the vocabulary includes bars and wall-clock minutes, so NT-041 can declare time-based fields), a range or a set of choices where one applies, a `tunable` flag and a `deprecated` flag. A test fails on a field without a doc or a unit, and on a default outside its own range or choices. RESAMPLE_MINUTES (core/config.py:68) is marked not tunable until NT-040 is done, because the annualisation ignores the bar size until then (test; NT-040 lifts it). (2) Config.validate enforces the declared ranges and choices: a test sets one out-of-range value per numeric type and one invalid choice, and each is refused with the field name in the message. (3) The 26 fields listed above have a doc (same test). (4) docs/guide/config-reference.md is generated from the metadata (group, name, default, unit, range or choices, tunable, deprecated, doc), and a test fails when the committed file differs from a fresh generation. (5) configs/default.yaml loads unchanged, and `scripts/golden_run.py verify` passes (no behaviour change).
- **source:** owner Q&A 2026-09-28 (rounds 3, 5, 8); docs/DECISIONS.md D-022, D-023, D-026; survey 2026-09-28 (field count re-checked with dataclasses.fields on 2026-09-28)
- **evidence (done 2026-09-29):** branch nt-029, 013772b, merged as 3845767. Config metadata on all 112 fields (unit, range or choices, tunable (35), deprecated (7)), `Config.field_specs()`, validate enforces the ranges with the field name, 26 docs added, docs/guide/config-reference.md generated by scripts/gen_config_reference.py (staleness test). No committed config refused. QA PASS (own golden recording: equal, 273 arrays; fast 730, slow 13, ruff clean). Merged head 3845767: fast 756 passed. CI green on c5b8a64 (run 36519610045). Findings: NT-028, NT-030, NT-034 notes.

### NT-030

**Sweeps: quick mode (about 5 minutes) and Optuna mode (measured budget, resumable), `neural-trade sweep`**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/experiments/ (sweep module), src/neural_trade/cli.py, pyproject.toml, requirements.txt, requirements-ci.txt, environment.yml, scenario specs, tests/
- **depends on:** NT-026 (experiment engine), NT-029 (config metadata)
- **why:** The yardstick is a search over configurations (VISION; D-020). The owner chose two modes (owner Q&A 2026-09-28, round 6; D-023): quick, the whole sweep in about 5 minutes, and Optuna, a Bayesian search whose GPU budget is measured and stated before it starts and which may run overnight while the owner's other project leaves the GPU idle (D-024). One seed per trial per dev fold; the top 5 are re-run with 3 seeds and ranked by the seed mean (D-020, D-024). The sweep limits (the one-night cap, parallel trials) are in OPERATING_MODEL "Sweeps and pre-registered studies"; this item implements them. Adding optuna to the nt env and the requirements is approved (D-023); the Q&A asks to check compatibility with Python 3.10 and numpy 1.23.5. The implementer never changes the nt env: the lead installs the pinned optuna once, as part of this item's integration, with a `pip install --dry-run` first, and stops and asks the owner if any installed package would change version (numpy stays at the env's 1.23.x, TF at 2.10.x; RUNBOOK "Environment", "Installing optuna").
- **acceptance:** (1) Quick mode: from a scenario and a search space, the sweep sizes the trials, epochs and data so that its estimate from the measured sec_per_step is at most 5 minutes of wall-clock for the whole sweep, prints the estimate before it starts, and labels every result quick (test with a stubbed runner and a given sec_per_step). (2) Optuna mode: a resumable Optuna study stored in sqlite in the run store; a test stops a CPU study after 2 trials and resumes it to 4 without repeating a finished trial. (3) Before an Optuna sweep starts, it prints and records its GPU budget (trials x dev folds x steps x the measured sec_per_step of the latest real run of the same setup, plus the top-5 x 3-seed re-run), and refuses to start when the budget exceeds --max-hours, which defaults to 12 (one night, the lead's reading of D-024; a larger budget goes to the owner) (test). (4) Parallel trials (`--parallel N`, default 1): trials launch in batches of N. The GPU-free check of RUNBOOK "GPU rules" runs only when none of the sweep's own trials is running, before each batch; when the GPU is busy the batch does not start, and the sweep waits or stops as configured. While a batch runs, the sweep watches the GPU utilisation and stops launching new batches if it exceeds the level NT-035 recorded for N own processes. `--parallel` above 1 is refused unless NT-035's recorded result allows that N (a small result file whose path the sweep config names; no file means N = 1). Tests with a stubbed check, a stubbed utilisation monitor and a stubbed record. (5) Trials rank on dev folds only (D-020). Failed or unstable trials (non-finite loss, and the stability guard of NT-036 and NT-038 once present) are pruned and recorded as failed, never dropped (test). (6) After the search, the top 5 are re-run with 3 seeds each and ranked by the seed mean, and the winner is recorded (test with a stubbed runner). (7) optuna is pinned in pyproject.toml, requirements.txt, requirements-ci.txt and environment.yml at a version that installs with Python 3.10, numpy 1.23.x (1.23.0 in the nt env, 1.23.5 in CI) and TF 2.10.x; the CI unit job installs it and passes. Tests that need optuna skip when it is not installed (pytest.importorskip), so the local suite passes before the lead's install. (8) `neural-trade sweep SCENARIO --mode quick|optuna [--resume]` works, and --help describes both modes (test through neural_trade.cli.main). (9) Until NT-040 is done, a sweep refuses a bar size other than 1 minute (RESAMPLE_MINUTES != 1 in the scenario or the search space) with a clear error (test; NT-040 lifts it).
- **source:** owner Q&A 2026-09-28 (rounds 3, 6, 7); docs/DECISIONS.md D-020, D-023, D-024
- **note (2026-09-29, NT-035):** `--parallel` may use N = 3 (runs/experiments/gpu_measurements_v1/parallel_n.json); never 4 (crash). The desktop alone showed a median sm of about 40% during a GPU-free check, so the watch level should rest on fb (memory) rather than sm.
- **note (2026-09-29, NT-026 QA):** the engine runs cells one at a time per process and has no locking around the index sync or the pending-cell choice, so `--parallel` above 1 needs a cell-claiming (lock) mechanism first, or two processes can train the same cell twice (no overwrite, but wasted GPU time); add it to this item's criteria when it is specified. `EARLY` (EarlyStopping patience) is not `tunable` while `PATIENCE` is (NT-029 QA): decide when the search space is written.

### NT-031

**Leaderboard ranked by dev-fold net Sharpe after costs, with guard-rails and test columns that never rank**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/experiments/ (leaderboard), src/neural_trade/visualization/ (a leaderboard table and figure, registered in Visualizations), src/neural_trade/cli.py, tests/
- **depends on:** NT-026 (experiment engine)
- **why:** D-020: the leaderboard ranks configurations by net Sharpe after costs on the dev folds only; guard-rails (max drawdown, the number of trades, beating buy-and-hold, beating the random null at the same frequency) sit beside it and can disqualify a row; every row also shows its test-fold numbers, which never rank. The owner accepted the pick-by-eye risk and asked that the UI make the ranking column explicit (owner Q&A 2026-09-28, round 3).
- **note (2026-09-29, QA of NT-076):** the rescore leaderboard ranks a 0-trade row (net Sharpe 0) above every losing row; the leaderboard's activity guard-rail must disqualify it.
- **acceptance:** (1) A leaderboard function reads the run index and returns one row per configuration: the dev-fold net Sharpe after costs (mean over the dev folds, and over seeds for re-runs, with the counts and the spread), the guard-rail columns and the test-fold columns (test). (2) Rows sort by the dev-fold net Sharpe only: a test builds an index whose test-fold order differs from its dev order and checks that the order follows dev. (3) A row that breaks a guard-rail (thresholds from the scenario) is marked disqualified, names the failing guard-rail and is not eligible as the winner (test). (4) Every row shows the dataset fingerprint, the bar size, the horizon lengths and the strategy name (test). (5) The ranking column is labelled as the ranking column, and the test-fold columns are labelled 'test, not used for ranking' in the table and the figure (test on the text). (6) Failed runs appear as failed rows (test). (7) `neural-trade leaderboard [SCENARIO]` prints the table (test through neural_trade.cli.main), and the figure follows theme.py and D-014, with each dev Sharpe drawn with its spread over folds and seeds (test: no empty panel).
- **source:** owner Q&A 2026-09-28 (rounds 2, 3, 7); docs/DECISIONS.md D-020, D-014

### NT-032

**Paired comparator for "A beats B" verdicts (D-025)**

- **status:** done (2026-09-30): 11296af, merged 89c9443; QA (Opus) FAIL on 2aef6ca (pair-level inference anti-conservative for one fold x seeds: false-'beats' 0.069-0.314; string-compared registration times; blocks not fingerprinted; configurations mixed) -> D-046 (fold is the unit of inference, >= 5 judgement folds) -> repair 1 (27da376, PASS on the criteria, one P1) -> repair 2 (285a64c) -> the lead's one-line fix (11296af: an uncommitted spec edit counts as registered at compare time) -> PASS. `neural-trade compare SPEC`, src/neural_trade/experiments/comparator.py, 59 tests; fast suite 1097 passed on 285a64c; RUNBOOK 'Paired comparator'. Deferred with reasons (module docstring): GPU-contention metadata, A/B-1's literal retention metric, infinite pairs kept in the ranks.
- **priority / type / role:** P1 / feature / implementer
- **area:** the statistics module (NT-027) or src/neural_trade/experiments/ (comparator), tests/
- **depends on:** NT-026 (experiment engine)
- **why:** D-025: "A beats B" (learned against frozen, a loss term on against off, any two scenarios) rests on a paired test over (seed, fold) pairs on the same blocks plus a minimum effect fixed before the runs, and guard-rails are judged by the same test, not by a point tolerance. The v1 ablation withdrew a family VALUE on a point tolerance (h1 AUC -0.0119 against 0.01, D-003), and identical GPU runs differ by 0.01-0.05 AUC (NT-003), so single-run comparisons cannot decide.
- **acceptance:** (1) A compare function pairs the runs of A and B by (seed, fold), refuses pairs whose blocks or dataset fingerprints differ (test), and returns the mean paired difference, its 95% interval, the test statistic, the number of pairs and the verdict (A beats B, B beats A, or inconclusive) against the minimum effect. A verdict's pairs are (seed, fold) over judgement folds that no choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds today; more held-out folds from the long history once NT-041 exists); at least 5 pairs (lead's reading of D-025). The comparator refuses fewer than 5 pairs and pairs on a fold the spec does not name as a judgement fold (test). (2) The minimum effect and the guard-rails come from a pre-registered spec; the output records the spec's hash, and the comparator refuses a spec that changed after the first compared run started (test). (3) Guard-rails are judged by the same paired test: a breach is a paired interval beyond the allowed degradation (test). (4) Error rates are checked by simulation, including 80-bar block noise within runs and seed noise between them: with a fixed seed and at least 1,000 simulated null comparisons, the false 'beats' rate is at most 5% plus its Monte Carlo error; the power at twice the minimum effect is reported (test). (5) The output is JSON plus a markdown paragraph that names the pairs, the metric, the minimum effect and the verdict.
- **source:** owner Q&A 2026-09-28 (round 7); docs/DECISIONS.md D-025, D-003
- **amendment (2026-09-29, D-037):** the comparator must support the plan's designs: per-fold retention r_f = d_f - E_A,f / 3 with one-sided bounds; non-inferiority with pass, breach or undecided per criterion; log-ratio metrics; paired coverage; Hodges-Lehmann with Wilcoxon bounds, infinite and inconclusive pairs, contention and re-time metadata; intersection-union verdicts with the owner route for D-018; Pocock two looks; anchor hashes per block; fold-placement refusal; a pre-registered pair count (no peeking); a simulation calibrated to the measured variance components that reports size and power (the gaps listed in C/ Q4 of the plan's evidence).

### NT-033

**Manual-search baselines: frozen-period twin and classic TA rules tuned by the same search**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/models/layers/learnable_indicators.py, src/neural_trade/models/gru_attention.py, src/neural_trade/training/optim.py, src/neural_trade/core/config.py, src/neural_trade/strategy/strategies.py, scenario specs, tests/
- **depends on:** NT-026 (experiment engine), NT-031 (leaderboard)
- **why:** VISION "The yardstick": the learned indicators must beat, under the same search budget and the same dev-fold net Sharpe, (a) the same network with the periods frozen at the textbook values and (b) classic technical-analysis rules whose parameters the same search tunes (D-020). No switch freezes the periods today: the period logits are trainable weights (learnable_indicators.py:49-101) with their own optimizer (training/optim.py:24), and every window shifts each logit by meta_adjust x meta_scale (learnable_indicators.py:115 in the batched path, :195-252 in the scan path; meta_scale 0.5 at :33; meta_adjust is a tanh Dense of the window's statistics, gru_attention.py:56). The Strategies registry holds only model-signal strategies and baselines (strategies.py:64-309).
- **acceptance:** (1) A Config switch freezes the period logits at their configured (textbook) values and removes the per-window meta_adjust shift (the shift's off switch is the one NT-046 defines; whichever item lands first creates it, the other reuses it): after a 1-epoch CPU run with it on, every applied period in every window equals its configured value to float32 precision (test). With it off, `scripts/golden_run.py verify` passes (the default is unchanged). (2) MA cross, RSI threshold and Bollinger breakout strategies are registered in Strategies with parameters (periods, thresholds) declared with ranges the sweep can read; they use only prices up to the decision bar and pass assert_no_lookahead (strategy/backtest.py:339); each has a test on a hand-made price series with known entries. (3) A TA-rule scenario runs through the engine without training a network. (4) Both baselines appear on the leaderboard (NT-031) with the same dev-fold net Sharpe column as the learned model (test on a tiny CPU scenario).
- **source:** owner Q&A 2026-09-28 (round 4: frozen twin + classic TA); docs/DECISIONS.md D-020; survey 2026-09-28

### NT-034

**Control-panel notebook 06 (ipywidgets + plotly)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** scripts/notebooks/build.py, scripts/notebooks/README.md, notebooks/06_control_panel.ipynb (new), src/neural_trade/notebook/ (a panel module), tests/test_notebooks_thin.py
- **depends on:** NT-030 (sweeps), NT-031 (leaderboard)
- **why:** D-023: the control panel is a Jupyter notebook (ipywidgets + plotly); the same engine is behind a CLI for long unattended runs. D-028: the notebooks persist and grow; new notebooks are added (06 control panel), generated by build.py and executed with outputs (D-013).
- **acceptance:** (1) notebooks/06_control_panel.ipynb is generated by scripts/notebooks/build.py, and the drift test covers it. (2) Its widgets choose a scenario, a search space and a mode (quick or Optuna), and launch or resume a sweep through the same engine code path as the CLI (a headless test drives the widget callbacks with a stubbed runner). (3) A leaderboard view refreshes while a sweep runs and labels the ranking and test columns as NT-031 does. (4) Comparison figures for two or more selected rows show every logged metric per horizon with its spread over folds and seeds (D-014, theme.py); when NT-032 is done, its paired verdict is shown. (5) A headless execution on a tiny CPU scenario runs without errors, and scripts/notebooks/check.py finds no error, stderr or empty panel in it. The real execution on the reference setup is the notebook routine of the definition of done. (6) Executing the notebook top to bottom (as the notebook routine does) launches nothing: it reads the run store and shows the latest real sweep's leaderboard and comparisons; only a widget callback launches a sweep (test: a top-to-bottom execution with a stubbed runner records no launch). (7) scripts/notebooks/README.md lists notebook 06.
- **source:** owner Q&A 2026-09-28 (rounds 3, 6; statement on notebooks); docs/DECISIONS.md D-013, D-014, D-023, D-028
- **note (2026-09-29, NT-029):** the panel reads ranges from `Config.field_specs()`; FOLD_INDEX's range is dynamic ([-N_FOLDS, N_FOLDS-1], kept in validate) and registry-key choices come from the registries (plugins make them dynamic); the cross-field rule CALIB_LAMBDA_MIN <= CALIB_LAMBDA_MAX is not checked anywhere yet.

### NT-035

**GPU measurements: concurrent-runs throughput and deterministic-mode speed**

- **status:** done (2026-09-29): runs/experiments/gpu_measurements_v1/REPORT.md (426de4f, text fixed 07d6e49); QA (Opus) reproduced every headline number and failed only the text; after the lead's fixes, re-QA (Sonnet) PASS on all 6 checks with the N = 4 throughput recomputed. Accepted deviations stated in the REPORT: GPU-free checks not recorded for Part A (one for Part B's 6 runs); N = 4 has 1 repeat (a crash is decisive for the refusal). Result: parallel N = 3 allowed, 4 refused; op determinism costs nothing; same-seed runs differ at epoch 0 (NT-074).
- **priority / type / role:** P1 / infra / experimenter
- **area:** runs/experiments/gpu_measurements_v1/ (SPEC.md, REPORT.md)
- **depends on:** NT-026 (experiment engine)
- **why:** D-024: several training processes at once only while the GPU is otherwise idle, with N set by one measured throughput test. D-025: a deterministic TF mode is opt-in for comparison studies after a measured speed test. Training is kernel-launch bound (D-010), so concurrent processes may raise the total throughput; nobody has measured it or the cost of determinism.
- **acceptance:** (1) A SPEC committed before any run: the reference setup's default config, N = 1, 2, 3 and 4 concurrent training processes with fixed epochs and steps, at least 3 repeats each, the metric (total training steps per second across processes, and sec_per_step per process), the pre-registered rule that picks N, and a GPU-time estimate within 3 hours. (2) Every measurement starts only when the RUNBOOK GPU-free check passes, and the check is recorded. (3) REPORT.md gives, per N, the total steps per second and the per-process sec_per_step with their spread, the peak GPU memory, the GPU utilisation that N own processes produce (the watch level of NT-030 (4)), and the N the rule picks; the allowed N and the per-N utilisation level are also written to the result file that NT-030's `--parallel` reads. (4) The deterministic mode of D-025 is `seed_everything(seed, deterministic=True)`, which calls `tf.config.experimental.enable_op_determinism()` (src/neural_trade/utils/seeding.py:31-32). `import neural_trade` already sets TF_DETERMINISTIC_OPS=1 for every run (src/neural_trade/__init__.py:26), and that does not make GPU runs reproducible (RUNBOOK "Traps on this machine"). With TF_DETERMINISTIC_OPS=1 in both arms: sec_per_step with and without enable_op_determinism (at least 3 repeats each); whether two deterministic GPU runs give identical val_loss per epoch; and a list of any op that raises for lack of a deterministic GPU kernel. Note (window research review, 2026-09-28): with only TF_DETERMINISTIC_OPS=1 (set by `import neural_trade`), `tensorflow.python.util._pywrap_determinism.is_enabled()` already reports True in TF 2.10, so the 'without' arm must unset TF_DETERMINISTIC_OPS before the import, and the report states which switch actually changes results (docs/research/2026-09-28-window-free/README.md, Challenge).
- **source:** owner Q&A 2026-09-28 (round 7); docs/DECISIONS.md D-010, D-024, D-025
- **where it stands (2026-09-29):** SPEC 8f35053 (pinned), results 7ee2916, REPORT 426de4f (runs/experiments/gpu_measurements_v1/). The SPEC's rule allows N = 3 (1.24x total steps/s; N = 2 1.33x; N = 4 0.68x and one of its processes crashed with 0xC00000FD, so N = 4 has 1 repeat, not 3); determinism costs nothing (det_on / det_off 0.996) but same-seed runs differ at epoch 0 and no op raised; TF_DETERMINISTIC_OPS=1 already enables op determinism. parallel_n.json written. 0.47 GPU-hours. **Open:** QA of the REPORT against the SPEC (and the SPEC itself was not QA-checked before GPU time, a deviation the experimenter reported); then done. Follow-ups: NT-074; NT-030 note; RUNBOOK traps.

### NT-036

**Stability invariants in CI (strict mode, masks off)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/losses/functions.py, src/neural_trade/training/custom_model.py, src/neural_trade/core/config.py, pyproject.toml (marker), .github/workflows/ci.yml, tests/ (new stability tests)
- **why:** D-026: hard invariants in CI, with a strict mode that turns the masks off. The loss replaces non-finite values by 0 in 38 places (`tf.where(tf.math.is_finite(...))` in losses/functions.py), including the total loss itself (losses/functions.py:790). The total-loss part of the finite-step guard (custom_model.py:476) can therefore never fire: a NaN in any term becomes a silent 0, the step proceeds, and nothing counts how often it happened. Only non-finite gradients are counted (nonfinite_grad_steps, custom_model.py:132, 482).
- **acceptance:** (1) A `stability` pytest marker, registered in pyproject.toml and run by the CI unit job. (2) A strict mode (a Config switch, on in the stability tests) turns the per-term masks off, so a non-finite term makes the total non-finite and the step guard fires (test: an injected non-finite term counts one non-finite step). With strict mode off, `scripts/golden_run.py verify` passes (default unchanged). (3) Per-loss-term mask counters: each masked term counts its masked steps into metrics.jsonl, and a test that injects a NaN into one term sees that term's counter rise and no other. (4) On a short CPU run of the default config on the bundled data, the stability tests assert 0 non-finite steps; finite weights, head outputs and learned periods after every epoch; and that no gradient group is clipped on every step, no variance head sits at VAR_FLOOR for every sample and no learned period sits at its bound, for N consecutive epochs (N and the levels stated in the test). (5) NaN injection proves the guard: a NaN in the input window, in one gradient and in one loss term each leave the weights finite and raise the right counter, and the per-group clip (custom_model.py:498-507) keeps each group's post-clip norm at or below GRAD_CLIP_NORM (test).
- **source:** owner Q&A 2026-09-28 (round 8); docs/DECISIONS.md D-026; survey 2026-09-28

### NT-037

**Per-run gradient health at most 2% of sec_per_step, per-term probe behind a flag (absorbs NT-012)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/training/custom_model.py (_update_diagnostics, train_step), src/neural_trade/core/outputs.py (LossComponents), src/neural_trade/losses/functions.py, src/neural_trade/metrics/tf_direction.py, src/neural_trade/telemetry/epoch_logger.py, src/neural_trade/evaluation/report.py (health section), src/neural_trade/visualization/training_dashboard.py, tests/
- **why:** D-026: health numbers in every run at no more than 2% of the training step's time, and a detailed per-loss-term probe (about 10%) behind a flag, so an unstable run can be attributed to its loss term. Only the epoch mean of the pre-clip global gradient norm is logged (custom_model.py:475, 286); there is no per-group maximum and no count of clipped steps (the clip is at custom_model.py:498-507). From NT-012 (absorbed): coherence_penalty is computed (losses/functions.py:593) but not logged, so the training dashboard draws an inferred 'other: coherence (not logged)' band (training_dashboard.py:98) and repeats the 0.1 factors of custom_loss by hand; the direction chance bands use n_val // steps although 11-17% of validation samples (19-26% on test) sit inside the deadband and are not scored (training_dashboard.py:1133).
- **acceptance:** (1) Per group (main, indicator), train only: grad_norm_max_main, grad_norm_max_indicator, grad_clip_steps_main and grad_clip_steps_indicator (steps whose pre-clip norm exceeded GRAD_CLIP_NORM) in metrics.jsonl (test on a 1-epoch CPU smoke run). (2) Dead-zone counters: val_dir_n_h0/h1/h2, equal to DirectionStats.mask_sum, and the counts of variance outputs at VAR_FLOOR and of learned periods at their bounds (test). (3) Per-term contributions for train and val_: contrib_* for every term of the total (losses/functions.py:770-788) with coherence_penalty among them; their sum equals loss / val_loss within 1e-4 relative (test). (4) The run's report shows the health numbers per group and per epoch: maximum norm against the clip, the share of clipped steps, non-finite steps and dead-zone counts (test on the text). (5) Cost: a micro-benchmark with the health numbers on and off (same seed, at least 5 repeats) shows at most 2% more time per step; the implementer reports the numbers, and the GPU check is the definition of done's sec_per_step comparison (D-018). (6) A probe behind a flag, off by default: per loss term, its share of the gradient norm and the cosine conflict between term gradients, per group, every K steps; with the flag on, a 1-epoch CPU smoke run has the probe keys and the shares sum to 1 within 1e-4; with it off, no probe key is written (test). (7) The training dashboard has no 'other: coherence (not logged)' trace for runs with these keys, the gradient panel draws the maximum with the clip count in the hover, and the direction chance bands use val_dir_n // steps; older runs without these keys still render (tests in tests/test_viz_training.py). (8) tests/test_custom_loss.py and tests/test_telemetry.py pass.
- **source:** owner Q&A 2026-09-28 (round 8); docs/DECISIONS.md D-018, D-026; NT-012 (integration_todo.md training requests: findings 38, 106b, NEW val_dir_n); survey 2026-09-28
- **amendment (2026-09-30, D-045):** the probe (6) also records, per term, the gradient norm on the shared trunk and the cosine with the total gradient, and the value share beside it (recommendation L7: every run reports both shares); the report shows the DIRECTION_SKIP logit's share of the direction logit variance (A4). Evidence: soft ECE 97% of the gradient direction in the CPU probe (A_losses.md section 7).

### NT-038

**Stability harness and config guard (refuse hyperparameter regions known to fail)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** a harness module in src/neural_trade/experiments/ whose cases run through the engine (NT-026), a pre-registered thresholds file under configs/, src/neural_trade/core/config.py (validate), the sweep search spaces (NT-030), tests/
- **depends on:** NT-026 (experiment engine), NT-029 (config metadata), NT-030 (sweeps and search spaces), NT-036 (invariants and strict mode), NT-037 (health numbers and the per-term probe)
- **why:** D-026: an on-demand stress harness (scale and volatility sweeps, extreme inputs, fault injection, three seeds) that every new setup must pass, with thresholds pre-registered, inside the experiment engine (owner Q&A 2026-09-28, round 1) and in strict mode in CI and in the harness (round 8). The owner added: "this should be configured to not allow a configuration of hyperparameters that would fail the system", so Config.validate and the sweep search spaces refuse regions the harness shows to fail, before a run starts. An unstable run is attributed to its loss term and fails loudly; Optuna prunes it and the leaderboard shows it as failed. The first real run on the reference setup is NT-051; the runs for N = 2 and N = 4 horizons are NT-052.
- **acceptance:** (1) The thresholds file is committed before the harness's first real run and not changed afterwards; its hash appears in every harness report. (2) The harness runs on demand: input scale and volatility x0.1 to x10, extreme-input fuzzing (constant windows, jumps, very large and very small prices), fault injection (a NaN in the input, in one loss term and in one gradient), 3 seeds each. The harness runs in strict mode (NT-036). Its cases run as engine scenarios, and their runs and verdicts are recorded in the run store and its index (test on a tiny CPU case). It writes a REPORT.md (for example runs/stability/<id>/REPORT.md) with pass or fail per case against the thresholds and, for a failure, the loss term the per-term probe blames. (3) A tiny CPU version runs under the `stability` marker. (4) An unstable run stops with an error that names the loss term (test by fault injection), and the sweep records it as failed (NT-030). (5) The harness writes the failing regions in a machine-readable file; Config.validate refuses a config inside one, naming the region and the report (test with a synthetic failing region), and the sweep search spaces exclude those regions (test). The first real run on the reference setup is NT-051 (experimenter).
- **note (2026-09-29, micro loop H4):** the config guard should refuse (or warn on) BATCH_SIZE x LOOKBACK^2 combinations that exceed GPU memory: at LOOKBACK 240, batch 2048 and 512 both OOM on the RTX 4070 Ti (attention softmax [B, 4, L, L]); measured in runs/scenarios/micro_lookback/ failed cells.
- **source:** owner Q&A 2026-09-28 (rounds 1, 8); docs/DECISIONS.md D-021, D-026
- **amendment (2026-09-29, D-037):** a named long-memory case: slow periods starting at 1,440 and 10,080 bars with INDICATOR_LR_MULT 5 and 1, plus the per-channel scale normalisation variant (B/ item 5 of the plan's evidence).
- **amendment (2026-09-30, D-045):** the memory guard counts both L^2 tensors: the batched EWMA weights [B, K, L, L] (K = 18 today; about 8.5 GB per float32 copy at B 2048, L 240, an estimate) and the attention scores [B, 8, L, L]; the threshold comes from a GPU memory profile the experimenter records (B_model_indicators.md 1.3).

### NT-039

**Pre-registered A/B: gradient-based loss weighting against today's value calibration**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** runs/experiments/loss_weighting_v1/ (SPEC.md, REPORT.md), an engine scenario spec (NT-026); code for the variants is an implementer item merged before the runs
- **depends on:** NT-032 (paired comparator), NT-037 (per-term gradient shares)
- **why:** D-026: measure first, then one pre-registered A/B of gradient-based loss weighting against today's value-based calibration of the loss weights (training/lambda_calibration.py). NT-037's per-term gradient shares and conflicts are the measurement.
- **acceptance:** (1) A SPEC committed before any GPU time: the hypothesis, at most 3 variants against today's calibration as the baseline, the dev folds and seeds for any choice, the judgement folds and seeds of (3), the metrics with their minimum effects, the guard-rails, and a GPU-time estimate within 3 hours (pre-registered A/B studies keep that limit, D-024) or the owner's approval. (2) The code for the variants is merged through an implementer item that the SPEC names. (3) One verdict by the paired comparator (NT-032). A verdict's pairs are (seed, fold) over judgement folds that no choice used; the SPEC names them before any GPU time (fold -1 x at least 5 seeds today; more held-out folds from the long history once NT-041 exists); at least 5 pairs (lead's reading of D-025, OPERATING_MODEL "Sweeps and pre-registered studies"). (4) REPORT.md gives the verdict, the numbers with their noise, every run id and NT-037's health numbers per variant. A negative or inconclusive verdict closes the item.
- **source:** owner Q&A 2026-09-28 (round 8: measure, then one pre-registered A/B); docs/DECISIONS.md D-024, D-025, D-026

### NT-040

**Annualisation ignores the bar size (Sharpe and Sortino overstated by sqrt(k) at k-minute bars)**

- **status:** todo
- **priority / type / role:** P1 / bug / implementer
- **area:** src/neural_trade/strategy/backtest.py (BacktestConfig), src/neural_trade/strategy/performance.py, src/neural_trade/strategy/params.py (build_backtest_config), src/neural_trade/cli.py, src/neural_trade/notebook/backtest_ui.py, the frozen set (D-023; a bar-size guard only), the engine's scorer (NT-026), the tunable flag of RESAMPLE_MINUTES (NT-029) and the sweep's bar-size refusal (NT-030), tests/
- **depends on:** NT-026 (experiment engine: the scorer)
- **why:** BacktestConfig.bar_minutes (strategy/backtest.py:45) sets periods_per_year (:66-67), but no caller sets it from the Config: scripts/backtest_gate.py:53, experiments/ablation.py:162 (no BacktestConfig passed, so the default), cli.py:119 and :125, and notebook/backtest_ui.py:330 and :500 all keep the default 1.0. With RESAMPLE_MINUTES = k (core/config.py:68), Sharpe and Sortino are annualised with k times too many periods per year and overstated by sqrt(k). MINUTES_PER_YEAR = 525,600 (strategy/performance.py:9) assumes a 24/7 market, which the MVP assumes (VISION "Not in the MVP"; D-022). No effect on the 1-minute reference today (k = 1); it becomes P0 before any other bar size runs.
- **acceptance:** (1) Every live path that builds a BacktestConfig for a run sets bar_minutes from that run's Config (RESAMPLE_MINUTES today, the dataset spec after NT-041): the CLI backtest, the notebook explorer and the engine's scorer. The frozen set (D-023), which includes experiments/ablation.py and scripts/backtest_gate.py, either gets the same fix or refuses a bar size other than 1 minute with a clear error. (2) A test with RESAMPLE_MINUTES = 5 checks, through each live path, that the annualised Sharpe equals the per-bar Sharpe x sqrt(525,600 / 5). (3) Periods per year come from one function of the bar size and a named calendar assumption (24/7 by default), so a market with sessions can set it later (test: the default gives 525,600 / bar_minutes). (4) 1-minute numbers are unchanged: the existing backtest tests pass, and a reference backtest's summary is identical before and after. (5) The two guards that wait for this item are lifted where their items are done: RESAMPLE_MINUTES becomes tunable in NT-029's metadata, and the sweep's 1-minute refusal of NT-030 (9) goes (test).
- **source:** survey 2026-09-28 (callers re-checked in code on 2026-09-28); docs/DECISIONS.md D-022
- **note (2026-09-29, NT-026):** the experiment engine's scorer already passes `bar_minutes` from RESAMPLE_MINUTES to the backtest; the frozen `scripts/backtest_gate.py` and `experiments/ablation.py` still use the default.

### NT-041

**Dataset spec and wall-clock configuration (window, horizons, blocks, costs, fingerprint, gaps)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/core/config.py, src/neural_trade/data/ (loaders, splits, windowing, processor), src/neural_trade/strategy/backtest.py (cost profile), src/neural_trade/evaluation/, src/neural_trade/visualization/ (labels), src/neural_trade/experiments/ (run meta), configs/, tests/
- **depends on:** NT-026 (experiment engine), NT-031 (leaderboard rows), NT-040 (annualisation)
- **why:** D-022: generality designed in now, only the reference setup tested in the MVP. Today the window and horizons are in bars (LOOKBACK, core/config.py:66; HORIZON_STEPS, :78), the val and cal blocks are fractions of the sequences (VAL_FRACTION and CAL_FRACTION, :70-71) inside N_FOLDS TimeSeriesSplit folds (:72), the data file is one CSV_PATH (:65), and costs are BacktestConfig defaults (fee 10, half-spread 1, slippage 2 bps per side; strategy/backtest.py:37-39). Figure labels are hard-coded: 'BTC' in visualization/matplotlib_splits.py:11, 26, 28, and '$' currency labels (in text and in d3 formats) in nine figure modules. Nothing checks for missing bars (the purge gap in data/splits.py is a gap between blocks, not a time gap in the data), and the local 2017-2025 file spans years of exchange history. The owner set the training block to 7 days, with the other blocks at their own configured lengths, and kept the long file for walk-forward folds over months (owner Q&A 2026-09-28, rounds 5, 5b).
- **acceptance:** (1) A dataset spec: symbol, quote currency, bar size and data file; the window and horizons in wall-clock minutes, converted to bars from the bar size; a length that does not divide exactly is refused (test). (2) The reference defaults give today's bars (a 60-minute window is 60 bars, horizons 10/15/20 minutes are 10/15/20 bars) and `scripts/golden_run.py verify` passes. (3) Every run's meta.json and every leaderboard row carry the dataset fingerprint (file sha256, first and last timestamp, bar count) and the setup (test). (4) Costs are a per-instrument profile (fee, half-spread, slippage per side) whose default equals today's 13 bps per side, and backtests read it from the spec (test). (5) Labels come from the spec: figure titles and axes name the symbol, the bar size and the quote currency, and a test finds no hard-coded 'BTC' label and no hard-coded '$' currency label (text or d3 format) left in src/neural_trade/visualization. (6) The training block is 7 days by default, and the val, cal and out-of-sample blocks have their own configured lengths in time; the purge gap stays (D-005) (test on the block lengths in bars). (7) Walk-forward folds over the long history: the data file is configurable, folds are placed at configured dates or spacing, and each fold records its dates in meta.json (test on a synthetic multi-month file: the folds fall in different months and their blocks never overlap). The bundled 30-day file stays the default for tests and CI. (8) Gap policy: missing bars are detected, no input window or target spans a gap, and the number of windows dropped is recorded in meta.json (test on a synthetic file with holes).
- **source:** owner Q&A 2026-09-28 (rounds 1, 5, 5b); VISION "The reference setup"; docs/DECISIONS.md D-022; survey 2026-09-28
- **amendment (2026-09-29, D-037):** from the window-free plan: detect forward-filled flat zero-volume runs; gaps and flat runs of 60 minutes or less are elapsed time, longer ones reset (series mode) and are counted in meta.json; do not drop anchors for short filled runs; fold roles (dev or judgement), a judgement fold's read range disjoint from every choice run's blocks (refused otherwise, test); a configurable out-of-sample length (5 days for the A/Bs); meta.json records each run's read range (A/ item 4 and C/ item 3 of the plan's evidence).
- **amendment (2026-09-30, D-045):** SKIP_LAGS (gru_attention.py:20, hard-coded bars) becomes a config field in wall-clock minutes, and learned periods are reported in minutes as well as bars (recommendation U4).

### NT-042

**Variable number of horizons**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/core/config.py (validate), src/neural_trade/core/outputs.py (LossComponents, PredictiveOutputs), src/neural_trade/models/gru_attention.py, src/neural_trade/losses/functions.py, src/neural_trade/training/custom_model.py, src/neural_trade/calibration/, src/neural_trade/serving/, src/neural_trade/evaluation/, src/neural_trade/strategy/signals.py, src/neural_trade/visualization/ (theme.py and every figure), tests/
- **depends on:** NT-038 (stability harness), NT-041 (dataset spec)
- **why:** D-022: a variable number of horizons now. Config.validate refuses anything but three ("the architecture has exactly three horizon towers", core/config.py:257-258). LossComponents is a fixed 34-element tuple with h0/h1/h2 fields (core/outputs.py:22-24), unpacked by position in train_step and test_step (custom_model.py:455-466, :564-575). The pairwise terms are written for two pairs: Casimir (losses/functions.py:328) and IFE (:442-443) sum the (h0, h1) and (h1, h2) terms, while the direction-disagreement part of the coherence penalty averages them (:572). Horizon colours are three fixed entries (theme.py:28-30).
- **acceptance:** (1) The model towers, the heads, LossComponents, calibration, serving, the evaluation report, the figures and the strategies' signal frame work for N horizons; tests run N = 2, 3 and 4 on CPU. (2) Pairwise physics terms run over every neighbouring pair (h_i, h_i+1) and are averaged, scaled so that N = 3 gives today's value exactly for each term (lead's reading of D-022, since Casimir and IFE sum their two pairs today; the golden run decides). (3) N = 3 reproduces today's numbers: `scripts/golden_run.py verify` passes against a recording made on the base commit. (4) Colours: h0 blue, h1 orange, h2 green; beyond three, theme.py extends in a fixed order with colours checked against the reserved status, LONG/SHORT and costs colours (test). (5) Saved runs with three horizons still load: serving bundles, metrics.jsonl and eval reports (test on fixtures). The stability-harness runs for N = 2 and N = 4 on the reference data are GPU work and belong to NT-052 (experimenter), not to this item.
- **source:** owner Q&A 2026-09-28 (rounds 5, 5b); docs/DECISIONS.md D-014, D-022; survey 2026-09-28

### NT-043

**Learned indicators on price against the textbook defaults (notebook 07)**

- **status:** done
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/visualization/ (a new figure module; indicator_evolution.py read, not rewritten), src/neural_trade/registries/visualizations.py, scripts/notebooks/build.py, scripts/notebooks/README.md, notebooks/07_discovered_indicators.ipynb (new), tests/test_viz_indicators.py
- **depends on:** none; may run in parallel with MVP-1 (files disjoint from NT-026 and NT-029; coordinate with NT-027 and NT-028 on visualization/)
- **why:** VISION "What every run delivers": the discovered indicators are the product, drawn on the price chart next to the textbook defaults. D-027: this is the first comprehension view, and D-014 applies to it in full (owner Q&A 2026-09-28, round 9: no simplified tier). Today indicator_evolution.py shows the periods over epochs and the periods applied per window (notebooks 01 and 04; build.py:228-229, :435-436), but no figure draws the indicators themselves on price. The configured starting periods are MA_SPANS, MACD_SETTINGS, RSI_PERIODS and BB_PERIODS (core/config.py:142-145).
- **acceptance:** (1) A figure draws, on the price of a chosen block and window, the learned moving averages and Bollinger lines, each next to the same indicator at its configured textbook period, with learned and textbook told apart by line style and named in the legend; horizon colours are not used for indicators (D-014). (2) RSI and MACD panels show learned against textbook on the same window. (3) Panels show how each period moved over training (from metrics.jsonl) and how it varies per window (the applied periods, including the meta_adjust shift), with a table of learned against textbook periods and their change. (4) Rich per D-014, theme.py throughout: every indicator family present, no empty panel (theme.empty_panels), and the figure within the size budget of the other figure tests (tests). (5) The figure is registered in Visualizations (D-002). (6) notebooks/07_discovered_indicators.ipynb is generated by build.py on a saved run and passes the notebook tests; its real execution is the notebook routine of the definition of done. scripts/notebooks/README.md lists notebook 07.
- **source:** owner Q&A 2026-09-28 (rounds 4, 9); VISION "What every run delivers"; docs/DECISIONS.md D-014, D-027, D-028
- **evidence (done 2026-09-29):** branch nt-043, c62d6f4, merged as 71fa02f; notebook 07 executed on run 20260924T182915Z-1aeff1c-dirty-af67ee43 (served epoch 19) in 31ac3ab: 1 figure, 199 traces, 0.3 MB, check clean; the lead looked at the render. visualization/discovered_indicators.py (registered as `discovered_indicators`): every family x 3 copies, learned solid against textbook dashed on price, RSI and MACD panels, periods over training with the per-window applied strip, and the table; the lines equal the served model's channels to float32 round-off (QA re-derived: MA <= 2.9e-6, MACD <= 4.3e-6, RSI <= 1.7e-5). 14 tests in tests/test_viz_indicators.py. QA PASS (fast 731 passed, slow 13 passed, ruff clean). Findings: NT-058.

### NT-044

**Guides for the owner and reviewers, README landing page, ARCHITECTURE**

- **status:** todo
- **priority / type / role:** P1 / docs / implementer
- **area:** docs/guide/concepts.md, docs/guide/reading-figures.md, docs/guide/experiments.md, docs/guide/own-data.md (all new), README.md, docs/ARCHITECTURE.md (new), a docs test in tests/
- **depends on:** NT-026, NT-030, NT-031 and NT-034 for experiments.md and the README's sweep quick start; NT-041 for own-data.md; NT-027 for the layering rules in ARCHITECTURE.md. The other parts can be written earlier.
- **why:** The audience is the owner and a few reviewers, and the docs are in English (D-019). The owner asked for short guides (concepts, reading the figures, sweeps and the panel, own data) and for the README as a landing page with a quick start (owner Q&A 2026-09-28, round 9). The README (250 lines) mixes install, performance, CLI, API, evaluation, backtests, the ablation and layout; there is no architecture document (the registry design docs are in docs/archive).
- **acceptance:** (1) concepts.md explains the learned indicators, the horizons and heads, costs, dev folds against the test fold, and the yardstick, linking to VISION and DECISIONS instead of repeating them. (2) reading-figures.md covers every figure the notebooks draw: what it shows, its noise band or reference, and how to read it; a test checks that every Visualizations key and every figure function called in scripts/notebooks/build.py is named in it. (3) experiments.md covers scenarios, quick and Optuna sweeps and their GPU budget, the panel, the leaderboard (the ranking column against the test columns) and verdicts by the paired comparator, with a worked example on the reference setup whose commands a test runs on a tiny CPU scenario. (4) own-data.md covers the CSV format, the dataset spec, the wall-clock window and horizons, the cost profile and the stability harness a new setup must pass (D-026). (5) README.md is a landing page: what the project is, a quick start whose commands the CLI smoke test exercises, links to the guides and to docs/ARCHITECTURE.md, and a link to docs/STATUS.md for the status (the README carries no status numbers; NT-010 checks those in STATUS). (6) docs/ARCHITECTURE.md gives a module map (one line per subpackage), the layering rules that NT-027's test enforces, and the registries; a test checks that every subpackage of src/neural_trade is listed. (7) Every relative link in README.md, docs/ARCHITECTURE.md and docs/guide/*.md resolves (test).
- **source:** owner Q&A 2026-09-28 (round 9); docs/DECISIONS.md D-019

### NT-045

**Notebook overlap: each figure gets one home**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** scripts/notebooks/build.py, notebooks/ (00-07), tests/test_notebooks_thin.py
- **depends on:** soft: NT-043 (notebook 07 may become the home of the indicator figures)
- **why:** D-028: notebooks 00-05 keep their numbers and roles, new notebooks are added, and each figure gets one home. Today 01 and 04 draw the same 10 figures: the training dashboard, direction detail and loss terms (build.py:138-140 and :401-403), the direction, delta, variance, confidence and coherence analytics (:180-207 and :420-425), and the indicator evolution and applied periods (:228-229 and :435-436); they also print the same tables. 02 and 03 share 2: the trading dashboard and the trade analytics (:276-279 and :341, :351).
- **acceptance:** (1) Each figure is drawn in exactly one notebook; a test over the cells that build.py generates fails when a Visualizations key or figure function is called in two notebooks (any exception is listed in the test with its reason). (2) The notebooks keep their numbers and roles (D-028), and a notebook that gives up a figure links to its home in a markdown cell. (3) No figure is lost: the set of figures across the notebooks after the change contains every figure drawn before (test against a list taken from the base commit's build.py). (4) The notebooks are regenerated by build.py, and the notebook tests pass; their execution is the notebook routine of the definition of done.
- **source:** owner Q&A 2026-09-28 (round 6: keep 00-05, add new ones, trim the overlap); docs/DECISIONS.md D-028; survey 2026-09-28 (overlap re-counted in build.py on 2026-09-28)

### NT-046

**Indicators package and registry with today's four families**

- **status:** done (2026-09-29): c919338, merged into remediation/plan; QA PASS on every criterion, the decisive one reproduced independently (QA's own golden record at f5aee70, verify at c919338: golden_equal true, 273 arrays, max|diff| 0 - bit-for-bit); layer timing equal within noise (D-018); fast 948 / slow 14 / ruff clean. QA P2/P3 findings: configured_periods cannot list an omitted dict param's textbook default (documented); dead momentum_raw_vars branch in learnable_indicators.py (D-029 candidate for a later item).
- **priority / type / role:** P1 / feature / implementer
- **area:** a new indicators package (for example src/neural_trade/indicators/), a new Indicators registry next to the nine of D-002 (for example src/neural_trade/registries/indicators.py), src/neural_trade/models/layers/learnable_indicators.py, src/neural_trade/models/gru_attention.py, src/neural_trade/core/config.py, src/neural_trade/registries/layers.py, tests/
- **depends on:** NT-026 (experiment engine)
- **why:** D-027, owner: "I want the list of indicators to be extendable and easily integrated via registry". D-031 (indicator Q&A 2026-09-28): one registry entry per family, declaring its inputs, its learnable parameters with textbook defaults and bounds, its output channels and how it is drawn (round B); the config lists the instances, 3 per family (round C); the adaptive per-window periods stay, with a switch that turns them off (round A). Today the families are fixed: LearnableIndicators builds MA, MACD, RSI and Bollinger logits (learnable_indicators.py:49-101) from four Config lists (core/config.py:142-145), the meta_adjust width is computed from the same lists (gru_attention.py:52-56), and the Layers registry selects the whole layer by role (gru_attention.py:59; registered in registries/layers.py:58 as '18 learnable EWMA periods -> 31 MA/MACD/RSI/Bollinger channels'). This item moves today's four families into the registry with no number changed; NT-047 adds the OHLCV input and the new families.
- **acceptance:** (1) An Indicators registry, strict like the nine of D-002 and covered by the registry contract tests (tests/registries/test_contracts.py), holds one entry per family for today's four (MA/EMA, MACD, RSI, Bollinger). Each entry declares its inputs (the close today), its learnable parameters with textbook defaults and bounds, its output channels and its drawing spec (on price or in its own panel) (test). (2) The config lists each family's instances. The default is 3 per family with today's starting periods (MA 5/10/30; MACD 12/26/9, 5/35/5, 8/17/9; RSI 9/14/21; Bollinger 10/20/25), in bars today and in wall-clock time once NT-041 exists. Config files that set MA_SPANS, MACD_SETTINGS, RSI_PERIODS or BB_PERIODS still load and give the same instances (test). (3) The model builds its indicator channels from the registry entries the config lists (18 learnable periods -> 31 channels by default, as today). A family is added by one registry entry and a config line, without editing the model or the training pipeline (test with a toy family registered in the test). (4) The adaptive per-window shift (meta_adjust) is kept, with one switch that turns it off: with it off, every applied period in every window equals the family's learned global value (test). NT-033's frozen twin uses the same switch (whichever item lands first creates it; the other reuses it). (5) No number changed: `scripts/golden_run.py verify` passes against a recording made on the base commit, saved serving bundles still load and predict identically (test on a fixture), and the notebooks build with no drift other than import lines.
- **source:** owner Q&A 2026-09-28 (round 4); indicator Q&A 2026-09-28 (rounds A-C); VISION "Also in the MVP"; docs/DECISIONS.md D-002, D-027, D-031
- **amendment (2026-09-29, D-037):** each family's registry entry also declares M(eps), the bars after which its state's dependence on the start is below eps (for a single EWMA ceil(ln eps / ln(1 - alpha_min)) after the maximal shift), with a per-family test that it is at least the empirical offset-invariance value (the plan's 'Warm-up' section).

### NT-047

**OHLCV input and the new indicator families, all learnable and on by default**

- **status:** done (2026-10-01): QA PASS on 72d3838 (after repair round 1); merged 7b1a8ae on the owner's answer to question 7 (D-047: OHLCV with all 14 families by default, the 1.6x step cost accepted). Merge resolution: build_windows(close, df) returns the model windows and prepare_datasets_from_windows takes them (NT-088's split), so the screen cache carries them; test_pnl_utility reads the close channel. Fast suite 1156 passed, slow suite 24 passed on the merge.
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/data/ (sequence building: OHLCV windows), the indicators package and registry (NT-046), src/neural_trade/models/gru_attention.py, src/neural_trade/core/config.py, src/neural_trade/serving/ (input shape), tests/
- **depends on:** NT-046 (indicators registry); NT-053 (the window-free plan: the input path and the indicator forms may change; D-032)
- **why:** D-031 (indicator Q&A 2026-09-28): the MVP catalogue has all four groups of families: today's four; range / volatility (ATR, Stochastic, Williams %R, Keltner); volume (OBV, VWAP, MFI); trend strength / channels (ADX/DMI, CCI, Donchian). The model input therefore becomes OHLCV. Owner: "All should be learnable", with smooth (differentiable) versions where needed. All families are on by default, 3 instances each. Combinations are left to the network, owner: "the combination are what the network does automatically, it's the core of it's mechanic. I don't think it needs any extra mechanism", so this item adds no gate or selection layer. Today the sequences are built from the close only (data/processor.py:44-45), although both data files carry open, high, low, close and volume, and the preprocessors keep all five (data/preprocessors.py:16-17). The owner also asked to research removing the fixed input window (round B: "I think we can get rid off it completely. Research this."); the input path may change with that decision, so this item waits for it.
- **acceptance:** (1) The model input carries open, high, low, close and volume; sequence building and serving read OHLCV (test on the bundled data: shapes, and the window ends at the decision bar). (2) The ten new families (ATR, Stochastic, Williams %R, Keltner, OBV, VWAP, MFI, ADX/DMI, CCI, Donchian) are registered in the Indicators registry (NT-046), each with learnable parameters that have textbook defaults and bounds. Where the textbook form is not differentiable (rolling max or min, for example), a smooth version is used, and at the textbook parameters it stays within a tolerance stated in the test of the textbook indicator on a hand-made series (test per family). Every channel at bar t depends only on bars up to t (test: changing later bars leaves it unchanged). (3) Gradients are finite with respect to every learnable parameter of every family, on the bundled data and on extreme inputs (constant windows, zero volume, jumps) (test). (4) All families are on by default with 3 instances each; more or fewer is a config change (test). (5) Speed: a CPU micro-benchmark of the train step before and after (same seed, at least 5 repeats) is in the implementer's report; the GPU check is the definition of done's sec_per_step comparison on a real run (D-018: a slower step needs the owner's acceptance, recorded in STATUS). (6) The `stability` tests (NT-036, once present), the fast and slow suites and ruff pass.
- **source:** indicator Q&A 2026-09-28 (rounds A-C); docs/DECISIONS.md D-018, D-027, D-031; survey 2026-09-28 (input path checked in code on 2026-09-28)
- **amendment (2026-09-29, D-037):** no longer waits for the window decision: built now in window mode against the Indicators registry interface (NT-046); each family declares its M(eps) in its registry entry (the plan's 'Warm-up' section); the series forms (exponential and leaky: decayed soft max/min, leaky OBV, VWAP as a ratio of EWMAs) come after A/B-1 (NT-069), drawn next to the textbook values. NT-065 runs after this item (shared files).

### NT-048

**Discovered-indicators report: a self-contained interactive HTML report per run**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/visualization/ (the report figures, reusing NT-043's figure module), src/neural_trade/evaluation/ (grouped permutation importance), src/neural_trade/registries/visualizations.py, src/neural_trade/cli.py, scripts/notebooks/build.py, notebooks/07_discovered_indicators.ipynb, tests/
- **depends on:** NT-043 (learned indicators on price, notebook 07), NT-046 (indicators registry)
- **why:** VISION "What every run delivers": the discovered indicators are the product. D-031 (indicator Q&A 2026-09-28): owner: "html report with extensive rich interactive visualizations", so each run gets a self-contained HTML report (plotly, D-014-rich), not a YAML or Pine export (round C); the importance read-out is grouped permutation importance after training, a read-out only and not a model mechanism (round B); the adaptive periods are reported as a global value plus a per-window range (round A).
- **acceptance:** (1) Grouped permutation importance: for each family and each instance, the loss of skill on the validation block when that group's channels are permuted (at least the validation loss and the per-horizon direction AUC), with a noise band from effective samples or a block bootstrap (D-012). Nothing in training reads it (test on a synthetic setup where one family carries the signal: it ranks first, and a pure-noise family's importance lies inside its band). (2) A self-contained HTML report per run (plotly, with the plotly library embedded, so it opens offline) shows: the learned indicators drawn on price against the textbook defaults; the learned parameters in wall-clock time with their global value and per-window range; how they moved during training; and the importance of (1) with its noise bands (test: the file loads no external script, holds every panel, and has no empty panel by theme.empty_panels). (3) Rich per D-014, theme.py throughout: every family and instance present, and horizon colours not used for indicators (test). (4) Every CLI or notebook training run writes the report into its run directory, and a CLI command writes it for an existing run (test through neural_trade.cli.main). Engine trials write it only when the scenario asks (lead's reading, to save disk). (5) The figures are registered in Visualizations (D-002), and notebook 07 (build.py) shows the same figures next to NT-043's view; the notebook tests pass, and the real execution is the notebook routine of the definition of done.
- **note (2026-09-30, QA of NT-047):** the learned-period figures (notebook 01 cell 25, 04 cell 11, 07 cell 4; visualization/indicator_evolution.py, discovered_indicators.py) draw only ma/macd/rsi/bb: 18 of the default 54 periods; the ten new families' 36 periods are logged but never drawn. This item's 'every family and instance present' covers them.
- **source:** indicator Q&A 2026-09-28 (rounds A-C); VISION "What every run delivers"; docs/DECISIONS.md D-012, D-014, D-027, D-031

### NT-049

**Training silently warm-starts from weights in the working directory**

- **status:** todo
- **priority / type / role:** P2 / bug / implementer
- **area:** src/neural_trade/training/trainer.py (train_and_evaluate, train_model), src/neural_trade/core/config.py (MODEL_PATH), tests/
- **why:** train_and_evaluate (training/trainer.py:259) loads cfg.MODEL_PATH whenever that file exists (trainer.py:365-377). The default MODEL_PATH is a bare file name (core/config.py:182), so a run without a RunContext reads it from its working directory. With force off (the default, trainer.py:266) it loads those weights and skips training; with force on it warm-starts from them. Both are logged at INFO only. The CLI (cli.py:65) and the notebook session (notebook/session.py:137) pass a RunContext, which points MODEL_PATH at a fresh run directory (experiments/run_context.py:53), so they are not affected; any other caller started at the repo root, where an old weights file exists, evaluates or continues stale weights. Under D-029 removing the load is not a deletion (it has an effect), so it is a behaviour fix with its own test; NT-028 leaves it alone. The lead notes the behaviour change in STATUS when it lands.
- **acceptance:** (1) In a temporary working directory that holds a planted weights file at the default MODEL_PATH, a 1-epoch CPU train_and_evaluate with the default Config and no RunContext trains from its seeded initial weights, with force off and with force on: its val_loss equals that of the same seeded run in an empty directory (test). (2) Loading or warm-starting from existing weights stays possible only through an explicit option that names the file (an argument or a Config field, off by default); when used, the path is logged at WARNING and recorded in the run's meta.json and status.json (test). (3) The implementer's report lists every caller that relied on the implicit load (a search of src, scripts, scripts/notebooks/build.py, notebooks and tests), and none is left relying on it. (4) The fast and slow suites pass, ruff is clean, and `scripts/golden_run.py verify` passes.
- **source:** doc review 2026-09-28 (NT-028 criterion (3) would have removed a behaviour under D-029); training/trainer.py:365-377 checked in code on 2026-09-28; docs/DECISIONS.md D-029

### NT-050

**First real Optuna sweep on the reference setup (learned, frozen twin, TA rules) and the paired verdicts**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** a sweep spec for the reference setup (NT-026, NT-030), the engine's run store, runs/experiments/yardstick_v1/ (SPEC.md, REPORT.md)
- **depends on:** NT-030 (sweeps), NT-031 (leaderboard), NT-032 (paired comparator), NT-033 (manual-search baselines), NT-034 (control-panel notebook 06)
- **why:** VISION "The yardstick" and MVP point 2 (D-020, D-025): the learned indicators must beat, under the same search budget and the same dev-fold net Sharpe, the frozen-period twin and the classic TA rules; "A beats B" is a paired verdict. NT-030 to NT-034 build the tools, but no item ran them on the reference setup (doc review 2026-09-28). It is GPU work, so it belongs to the experimenter. GPU rules: the sweep is an Optuna sweep under D-024 (budget stated before it starts, within the one-night cap of OPERATING_MODEL "Sweeps and pre-registered studies"); the verdict runs are a pre-registered study (3 GPU-hours, or the owner's approval).
- **acceptance:** (1) Before the first trial, a sweep spec is committed with its stated budget (NT-030's formula). The learned model, the frozen twin and the TA rules each get the same number of trials on the same dev folds, and the scenario states the budget (checked in the run index: equal trial counts and folds per arm). (2) The sweep runs through the engine in Optuna mode (NT-030); every trial, failed ones included, is in the run index; the top 5 of each arm are re-run with 3 seeds and ranked by the seed mean, and each arm's winner is recorded. (3) The leaderboard (NT-031) shows the three arms: REPORT.md includes the `neural-trade leaderboard` output. (Showing it in notebook 06 is the lead's notebook routine after this item; 06 reads the run store, NT-034 (6).) (4) Before any verdict run, a study SPEC is committed: the metric (net Sharpe after costs), the minimum effect, the guard-rails, the judgement folds and seeds under the verdict-fold rule of OPERATING_MODEL "Sweeps and pre-registered studies" (at least 5 (seed, fold) pairs on folds no choice used), and a GPU-time estimate within 3 hours or the owner's approval. (5) Two verdicts by the paired comparator (NT-032), each judged once: the learned winner against the frozen-twin winner, and the learned winner against the TA-rules winner. REPORT.md gives the verdicts, the numbers with their noise, the dataset fingerprint and every run id. A negative or inconclusive verdict is a valid outcome (VISION "The yardstick"): it is recorded and closes the item.
- **source:** doc review 2026-09-28 (no item owned the MVP-2 exit's GPU runs); VISION "The yardstick"; docs/DECISIONS.md D-020, D-024, D-025

### NT-051

**First stability-harness run on the reference setup against its pre-registered thresholds**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** the stability harness of NT-038 (run as engine scenarios), its reports (for example runs/stability/<id>/), a SPEC (for example runs/experiments/stability_ref_v1/SPEC.md)
- **depends on:** NT-038 (stability harness and config guard)
- **why:** D-026: every setup must pass the on-demand stress harness against pre-registered thresholds, and the MVP-3 exit needs that on the reference setup. NT-038 builds the harness and commits its thresholds file; its first real run is GPU work, so it belongs to the experimenter (doc review 2026-09-28). It is a pre-registered study: the 3-GPU-hour limit applies (OPERATING_MODEL "Sweeps and pre-registered studies").
- **acceptance:** (1) A SPEC committed before any GPU time: the harness cases and seeds (NT-038 (2)), the thresholds file by its hash (unchanged since NT-038 committed it), and a GPU-time estimate within 3 hours or the owner's approval. (2) Every GPU run starts only when the GPU-free check of RUNBOOK "GPU rules" passes, and the check is recorded. (3) The harness report gives pass or fail per case against the thresholds, the thresholds file's hash and every run id; for a failure, it names the loss term the per-term probe blames and the failing region written for Config.validate (NT-038 (5)). (4) A failure is a valid outcome: it is recorded in the report, each fix becomes a new backlog item, and the item closes with the report.
- **source:** doc review 2026-09-28 (no item owned the MVP-3 exit's GPU run); owner Q&A 2026-09-28 (round 8); docs/DECISIONS.md D-026

### NT-052

**Stability-harness runs for N = 2 and N = 4 horizons on the reference data**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** the stability harness of NT-038 (run as engine scenarios), its reports (for example runs/stability/<id>/), a SPEC (for example runs/experiments/stability_horizons_v1/SPEC.md)
- **depends on:** NT-042 (variable number of horizons), NT-051 (first harness run)
- **why:** D-022: any number of horizons; D-026: every new setup must pass the harness. NT-042 makes N horizons work in code, with tests on CPU; the harness runs for N = 2 and N = 4 on the reference data are GPU work, so they moved out of NT-042's acceptance into this experimenter item (doc review 2026-09-28). Same GPU rule as NT-051 (a pre-registered study).
- **acceptance:** (1) A SPEC committed before any GPU time: the horizon sets for N = 2 and N = 4 in wall-clock minutes, the harness cases and seeds, the same thresholds file as NT-051 (hash unchanged), and a GPU-time estimate within 3 hours or the owner's approval. (2) Every GPU run starts only when the GPU-free check of RUNBOOK "GPU rules" passes, and the check is recorded. (3) One harness report per N with pass or fail per case, the thresholds file's hash and every run id; for a failure, the blamed loss term and the failing region. (4) A failure is recorded; each fix becomes a new backlog item, and the item closes with the reports.
- **source:** doc review 2026-09-28 (NT-042's GPU criterion moved here); owner Q&A 2026-09-28 (rounds 5, 5b, 8); docs/DECISIONS.md D-022, D-026

### NT-053

**Window-free plan: a second research round that writes the path, gates and A/B specifications**

- **status:** done
- **priority / type / role:** P1 / research / lead
- **area:** docs/research/2026-09-28-window-free/ (input); a new docs/research/<date>-window-free-plan/ (output; CPU prototypes only); docs/DECISIONS.md (the purge rule); docs/BACKLOG.md and docs/ROADMAP.md (the items the plan creates)
- **depends on:** none (no GPU; the lead writes it, with subagents for the prototypes and an adversarial review)
- **why:** D-032, owner: "commit to this in another research and write it down in a plan." The first round (docs/research/2026-09-28-window-free/README.md) measured on CPU that the 60-bar window caps the learned periods (the slow MACD period reaches the ceiling in 3 of 7 local runs), cold-starts every indicator at the window's first bar, and grows quadratically in memory with the window; that the GPU step is launch-bound (about 4,200 ops per step, an estimate), so the window's speed cost is uncertain; and that a per-bar model could train much faster only if the model learns per epoch rather than per update. Its adversarial review found must-fix problems in the proposed specifications: A/B margins that would let an arm lose most of the model's edge (h1 CRPSS vs constant variance is 0.0167), a batch-size probe judged on losses recalibrated per arm, a window gather whose backward is a segment sum with no deterministic GPU kernel (op determinism is already on in every run), precision tolerances the prototype does not meet in the specified increment form, a speed primary that epoch caps could produce, and a coverage band that a block can fail whatever the arms do. The owner left the period ceiling unlimited, and the adaptive-period replacement and the purge rule to research.
- **acceptance:** (1) A plan document that fixes the stages and their order with an objective gate each (candidates: fixed-cost and launch cuts (NT-054); a GPU probe of epoch- vs update-boundness with a lambda-free, batch-size-free judging metric; indicators computed over the series feeding today's network; the per-bar model), the design of each stage in TF 2.10 (no window gather whose backward is a segment sum; a causal indicator recurrence with tolerances stated relative to the state's magnitude and met by a CPU prototype at 43k bars; state handling at data gaps and at the data start, identical in training and serving), and the corrected A/B specifications (margins as a stated fraction of arm A's measured edge with pair counts that power them; a speed primary that caps cannot produce; coverage judged paired; one change per arm). (2) The replacement for the per-window adaptive periods (D-031: reported as a range), with a CPU prototype, its cost measured, and how the discovered set reports it. (3) The purge rule for indicators with unbounded memory: the leakage argument, the cost of each candidate rule on a 7-day training block, the lead's decision, and the test that will pin it (the DECISIONS entry is the lead's step 7, not a criterion). (4) With no period ceiling (D-032): how the warm-up is derived from the learned periods at run time, what refuses a run whose warm-up exceeds the block, and what the Predictor needs. (5) Every claim labelled measurement (CPU, command given) or estimate; every GPU number an estimate until measured. (6) Proposed backlog items for each stage with acceptance criteria checkable at QA time and the owning role, and a ROADMAP placement that respects the pick order and the disjoint-files rule. (7) The plan is presented to the owner for approval (D-032); implementation items are picked only after it.
- **source:** owner Q&A 2026-09-28 (indicator Q&A Round D); docs/research/2026-09-28-window-free/README.md (synthesis and Challenge)
- **evidence (done 2026-09-29):** the plan docs/research/2026-09-29-window-free-plan/README.md (three CPU investigations A, B and C; an adversarial review, REVIEW.md, with 7 must-fix and 15 should-fix findings, all answered in the revision with the numbers of rev/); the lead's purge-rule decision D-034; approved by the owner as written (docs/qa/2026-09-29-window-free-plan.md, D-037). Items NT-059 to NT-073; amendments to NT-032, NT-038, NT-041, NT-046, NT-047.

### NT-054

**Per-run fixed costs and GPU launches (independent of the window)**

- **status:** todo
- **priority / type / role:** P2 / performance / implementer
- **area:** src/neural_trade/training/custom_model.py (train_step: per-group norms, finite guards), src/neural_trade/training/callbacks.py and src/neural_trade/telemetry/epoch_logger.py (epoch-end work), src/neural_trade/training/trainer.py (setup, tracing), tests/
- **depends on:** NT-027 (layering), NT-046 (the indicator logits move into per-family entries; a one-vector logit variable belongs there); coordinate with NT-037 (health diagnostics in the same train step)
- **why:** Measured on the newest run (runs/20260924T182915Z-1aeff1c-dirty-af67ee43, local only): about 59 of about 296 s per run are fixed or outside the timed epochs (about 13 s of tracing and warm-up in epoch 0, about 30 s between the timed epochs, about 17 s before and after training), and the step is launch-bound: the graph op census has 296 L2Loss (per-variable norms), 312 SelectV2 and 172 IsFinite nodes (docs/research/2026-09-28-window-free/README.md, compute investigation). Quick sweeps of about 5 minutes (D-023) and 7-day training blocks (about 4 s per epoch, an estimate) need both reduced, whatever happens to the window.
- **acceptance:** (1) A per-run timeline from a real run in the implementer's report: setup, tracing, each epoch, the time between epochs, evaluation. (2) One global-norm computation per optimizer group and fused finite guards; the graph op count before and after is reported. (3) `scripts/golden_run.py verify` passes; any change that cannot pass it is left out and listed. (4) At least 3 interleaved real GPU runs per side (before / after, same data, RUNBOOK GPU-free check): the median epoch time of epochs 1 and later and the total run wall-clock are both lower, reported with their spread (D-012, D-018). (5) Fast and slow suites and ruff pass.
- **source:** docs/research/2026-09-28-window-free/README.md (synthesis item P; the Challenge's correction on noise-aware timing)

### NT-055

**CI failure annotations: every failed test name readable, right paths for class tests and collection errors**

- **status:** todo
- **priority / type / role:** P2 / infra / implementer
- **area:** .github/workflows/ci.yml (the "Annotate failed tests" step)
- **why:** QA of NT-001 (2026-09-28, e3fd63e): GitHub documents a cap of about 10 error annotations per step (not verified here), and the red run at 7785ec9 had 11 failures, so some names would stay hidden. The `file=` path taken from the junit classname is wrong for class-based tests (`tests/<module>/<Class>.py`; 3 classes in tests/test_qbox_losses.py and tests/test_registry_base.py) and for collection errors (`file=.py`), whose cause is lost ("collection failure"). The step has never run end to end in CI (both runs since NT-001 were green).
- **acceptance:** (1) Besides the per-test annotations, the step writes every failed test id to the job summary (`$GITHUB_STEP_SUMMARY`), so all names are readable however many fail. (2) For class-based tests `file=` is the module path (tests/<module>.py); a collection error names its module path and the first line of its cause (a local simulation on a real pytest junit file with a class test, a parametrised test and a collection error shows it). (3) Proven in CI: on `nt-055`, a temporary commit adds at least 11 failing tests (including a class test and a collection error), its CI run's annotations API and job summary list them, and the next commit reverts it (history keeps both; nothing is deleted). (4) With no failures the step is skipped and the job is green.
- **source:** QA report for NT-001 (2026-09-28)

### NT-056

**CI and packaging hygiene: CI lints tests, actions off Node 20, the viz extra and jinja2**

- **status:** todo
- **priority / type / role:** P3 / infra / implementer
- **area:** .github/workflows/ci.yml, .github/workflows/nightly.yml, pyproject.toml, requirements-ci.txt
- **why:** QA of NT-001 (2026-09-28): CI lints `ruff check src scripts` (ruff 0.6.9) while the local rule lints `src tests scripts`, so CI never lints tests; the actions target Node 20 and are forced onto Node 24 (a deprecation warning on every job); pyproject.toml's `viz` extra allows `plotly>=5.24`, but plotly 5 cannot load the committed notebooks, and `DataFrame.style` (visualization/analytics_tables.py:815, notebook/backtest_ui.py:163) needs jinja2, which no extra declares.
- **acceptance:** (1) CI lints `src tests scripts` and passes. (2) No Node 20 deprecation annotation on the `ci` jobs (action versions bumped). (3) The `viz` extra requires a plotly the committed notebooks load (>= 6.7) and declares jinja2. (4) CI is green on the pushed head; nightly.yml installs the same pins (it runs only after NT-008).
- **source:** QA report for NT-001 (2026-09-28)

### NT-057

**Random-null follow-ups: one mean-size definition, labels with the size, the CLI prints the matched null**

- **status:** todo
- **priority / type / role:** P3 / polish / implementer
- **area:** src/neural_trade/visualization/trade_analytics.py (`_mean_size`, `_usable_null`, `_default_label`), src/neural_trade/cli.py (cmd_backtest output), src/neural_trade/strategy/backtest.py (the baselines' own null), tests/
- **why:** Found by NT-002's implementer and QA (2026-09-28, b0fd0cf): trade_analytics.py:595-598 `_mean_size` (an unclipped mean over all decisions) is a second definition of the null's size next to `backtest._mean_fill_size` (clipped, size-0 orders left out); `_default_label` describes a `random_signal` run by rate and hold but not by its new `size_frac` knob; `neural-trade backtest` prints only `random_percentile_return` (the matched size, the 5-95% band and the gross percentile are only in `--out backtest.json`); `backtest()` also computes a null for the baselines themselves (backtest.py:339), so on engine results `buy_and_hold` shows "beats 100% of random" (trade rate 1.0), visible in the trade-analytics comparison figure (not in the notebooks).
- **acceptance:** (1) trade_analytics uses `backtest._mean_fill_size` (or the null's own `size_frac`), with no second definition (test). (2) A `random_signal` run's default label names its size (test). (3) `neural-trade backtest` prints the matched size, the 5-95% band and the gross percentile next to the net percentile (test through cli.main). (4) The baselines' own null is dropped from the comparison figure or labelled as not meaningful for buy-and-hold; the figure is rendered on a real run and looked at (D-014). (5) Fast suite and ruff pass.
- **source:** NT-002 implementer and QA reports (2026-09-28)

### NT-058

**Indicator views: which period is 'learned', RSI smoothing named, no private cross-module helpers**

- **status:** todo
- **priority / type / role:** P2 / polish / implementer
- **area:** src/neural_trade/visualization/indicator_evolution.py, src/neural_trade/visualization/discovered_indicators.py, notebooks 01, 04, 07 (through build.py)
- **why:** Found by NT-043's implementer and QA (2026-09-29). (1) Some base periods are never applied: MACD #0 slow has base 32.3 but applied 22.2-29.8 (5-95%), and MACD #1 slow has base 58.9 (at the 60 ceiling) but applied median 73.0, with 81% of test windows applied above the 60-bar lookback (not warmed up); the "base period" framing of notebooks 01 and 04 can mislead, and NT-048's "global value" needs a choice. (2) The model's RSI smooths with an EWMA (alpha 2/(p+1)), not Wilder's 1/p, so "textbook RSI 14" is the model's definition at period 14; figures and NT-033's classic RSI rule must say which. (3) discovered_indicators.py imports 8+ underscore-private helpers of indicator_evolution.py.
- **acceptance:** (1) The period panels and tables of 01, 04 and 07 show the median applied period next to the base, labelled, and flag instances whose applied periods exceed the window. (2) The RSI smoothing is named in the figures' notes. (3) The shared helpers become public (no leading underscore) or move to a shared module; no module imports another's private names (test). (4) Notebooks rebuilt through build.py and executed; the changed figures looked at (D-014); fast suite and ruff pass.
- **source:** NT-043 implementer and QA reports (2026-09-29)

### NT-059

**Window-free benchmark kit in scripts/bench/ (kernel V1, assembly D6b, the A2 and today's layers, op census, TF32 check)**

- **status:** done
- **priority / type / role:** P1 / infra / implementer
- **area:** scripts/bench/ (new), tests/ (a CPU smoke test)
- **depends on:** none (plan stage 1)
- **why:** Every GPU number of the window-free plan is an estimate; the kit measures them (plan stages 1 and 1g).
- **acceptance:** (1) The kit measures forward and backward time (median of at least 5 interleaved repeats) and op counts for kernel V1, assembly D6b, the whole A2 indicator layer (meta-shift included, written elementwise) and today's LearnableIndicators, at the plan's sizes, and writes JSON with the device and TF version. (2) G-A1 (docs/research/2026-09-29-window-free-plan/README.md, stages table): the census of the whole A2 layer, forward and backward, contains no MatMul, BatchMatMul or Einsum and nothing from the determinism RAISE or host-round-trip lists; CPU precision within 1e-5 x max|state| and 3e-5 x RMS per channel at 43,008 bars. (3) A CPU smoke test in the fast suite; the kit runs on the CPU in 2 minutes or less. (4) No change under src/. (5) Fast suite and ruff pass.
- **source:** the window-free plan and D-037 (2026-09-29)
- **evidence (done 2026-09-29; CI green on bc42e9f, run 36525413777):** branch nt-059, eec1ce9 + 7fa0fe6 (MACD precision added on the lead's review), merged as 5ea8eb5. scripts/bench/{common,kernel_v1,assembly_d6b,a2_layer,window_free}.py, `window_free.py --device {cpu,gpu} --out JSON`; tests/test_bench_window_free.py. CPU: A2 layer 62 ms / 726 ops / 0.62 GB static at the 7-day block against today's layer 190-265 ms / 588 ops / 2.0 GB; worst precision 1.2e-6 of max|state| (every channel, MACD included). QA PASS (census proven to catch an injected MatMul and a densified gather; own float64 recomputation; full kit 27 s; fast 799 passed, ruff clean). Backlog notes: the census does not see a differentiable tf.gather whose IndexedSlices gradient is never densified (NT-064 note); `--reps` floor not enforced (P3).

### NT-060

**GPU run of the window-free benchmark kit (G-A2)**

- **status:** todo
- **priority / type / role:** P1 / infra / experimenter
- **area:** runs/experiments/window_free_kit_v1/
- **depends on:** NT-059
- **why:** Plan stage 1g: measure TF32's effect, launch cost and determinism on the RTX 4070 Ti.
- **acceptance:** (1) The kit on the GPU after the RUNBOOK GPU-free check, TF32 at its default and op determinism on. (2) G-A2: no error; precision within 1e-5 x max|state| and 3e-5 x RMS; two runs bitwise identical; the A2 layer's forward+backward time at most 1.10x today's layer's (median of at least 5 interleaved repeats), else reported as a failed gate. (3) REPORT.md with the JSON; GPU time under 0.5 hour.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-061

**TF32 decision for the indicator layer (plan stage 1b)**

- **status:** todo
- **priority / type / role:** P1 / decision / lead
- **area:** docs/DECISIONS.md, tests/ (the census rule if kept)
- **depends on:** NT-060
- **why:** TF32 is enabled for matmul in this env; the plan's layer avoids matmul, but the rest of the model uses it.
- **acceptance:** (1) A DECISIONS entry citing NT-060's numbers: keep TF32 with the indicator layer's no-matmul rule enforced by the census test, or turn it off. (2) If turning it off slows training (D-018), the owner decides first.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-062

**VAL_BATCH_SIZE key (validation grouping independent of the training batch)**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/core/config.py, src/neural_trade/data/datasets.py, tests/
- **depends on:** none
- **why:** val_loss moves 8-11% with the validation grouping (C/ Q3 of the plan's evidence), so arms with different BATCH_SIZE compare different functions.
- **acceptance:** (1) VAL_BATCH_SIZE, default = BATCH_SIZE, with Config metadata (NT-029). (2) At BATCH_SIZE 1024 with VAL_BATCH_SIZE 256, val_loss equals the batch-256 value within 1e-6 relative for the same weights (test). (3) golden_run verify passes; fast suite and ruff pass.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-063

**Engine options for pre-registered studies: lambdas once per study, per-arm EPOCHS, cap-extension re-runs, contention records**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/experiments/, tests/
- **depends on:** NT-026 (done)
- **why:** The plan's designs (docs/research/2026-09-29-window-free-plan/README.md, 'Pre-registered designs') need them.
- **acceptance:** (1) A spec option `lambda_calibration: once` calibrates on a named dev fold and passes the same frozen LAMBDA_* with calibrate=False to every cell (test: identical lambdas in both arms). (2) Per-arm EPOCHS and early stopping on or off. (3) A cell capped at EPOCHS is re-run with the same seed at 2 x EPOCHS when the spec asks, both recorded (test). (4) Each run records its GPU-free check, its epoch times and whether it was re-timed. (5) Fast suite and ruff pass.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-064

**Series kernel V1 and assembly D6b in the indicators package (plan stage 3)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** the indicators package (NT-046), tests/
- **depends on:** NT-046, NT-061
- **why:** docs/research/2026-09-29-window-free-plan/README.md, 'Kernel V1' and 'Window assembly D6b'.
- **acceptance:** Items 1-2 of A/ in the plan's evidence, with the plan's tolerances (1e-5 x max|state| and 3e-5 x RMS per channel, at 30,720 and 43,008 bars, periods 2 to 1e6, constant and per-bar alpha, C = 16 and 64): split invariance; causality bitwise with a zero future Jacobian; gradients within 1e-3 of float64 finite differences and finite at logits +-13.8, +-30 and -40; outputs after a reset independent of earlier inputs; non-finite input refused at data load; a clean op census; utils/math.py unchanged; fast suite and ruff pass.
- **source:** the window-free plan and D-037 (2026-09-29)
- **placement (2026-09-29, D-039):** research track R6, after the MVP.
- **note (2026-09-29, NT-059 QA):** the kit's op census flags a gather backward only once its IndexedSlices gradient is densified in the traced graph; the series engine's tests must census the full training step (variables, optimizer update), not a function returning IndexedSlices, and keep the D6b custom gradient.

### NT-065

**INDICATOR_MEMORY switch: the series engine with per-bar adaptation, M_run, burn-in, history bound and pass budget (plan stage 4a)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** the indicators package, src/neural_trade/models/, src/neural_trade/core/config.py, src/neural_trade/data/, tests/
- **depends on:** NT-064; NT-047 (runs after it: shared files)
- **why:** docs/research/2026-09-29-window-free-plan/README.md, 'Per-bar adaptive periods' and 'Warm-up, history bound and pass budget'; D-037.
- **acceptance:** (1) Window mode is the default and `golden_run.py verify` passes. (2) Series mode: G-B1 of B/ in the plan's evidence, including the context mirror at the anchors (<= 1e-6) and the off switch = V1 with a zero shift; M(eps) per family from the registry; M_run with a 3 x lr per-step margin, geometric growth and per-step projection; a burn-in of M_run + L - 1 fixed at run start; the history bound and the pass budget (default 4 training blocks) project logits and are counted per epoch (tests with a forced drift). (3) An end-to-end causality test from the raw frame through every normaliser. (4) The plan's D-018 check (3 interleaved GPU runs per mode, by the experimenter) within 5%, else the owner decides. (5) Fast and slow suites and ruff pass.
- **source:** the window-free plan and D-037 (2026-09-29)
- **placement (2026-09-29, D-039):** research track R6, after the MVP.

### NT-066

**Purge-rule test and config guard (D-034)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/data/splits.py, src/neural_trade/core/config.py, tests/test_purge_rule.py
- **depends on:** none
- **why:** D-034 records the rule; this item pins it.
- **acceptance:** Item 1 of C/ in the plan's evidence: tests/test_purge_rule.py for W in (60, 30, 0) and two horizon sets; the reference gap stays 80 (golden run passes); Config.validate refuses a gap below max(2 max(H), W + max(H)), naming the field; the formula documented in splits.py; fast suite and ruff pass.
- **source:** the window-free plan and D-037 (2026-09-29)
- **placement (2026-09-29, D-039):** research track R6, after the MVP.

### NT-067

**Predictor series mode and bundle metadata**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/serving/, tests/
- **depends on:** NT-065
- **why:** docs/research/2026-09-29-window-free-plan/README.md, 'Data gaps and the data start'.
- **acceptance:** G-B3 of B/ in the plan's evidence: the Predictor refuses a history since the last reset shorter than M_run + L - 1; its predictions equal the trainer's on the same block (bitwise, same pass-start rule); it takes timestamps; the bundle stores the kernel spec, eps, the M rule, M_run, the reset threshold, the bar size and the normalisation constants, with a format version bump (round-trip test); window-mode bundles still load and predict identically.
- **source:** the window-free plan and D-037 (2026-09-29)
- **placement (2026-09-29, D-039):** research track R6, after the MVP.

### NT-068

**Reporting of per-bar periods and bound counts**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/visualization/indicator_evolution.py, discovered_indicators.py, scripts/notebooks/build.py, tests/
- **depends on:** NT-065
- **why:** docs/research/2026-09-29-window-free-plan/README.md, 'Reporting'.
- **acceptance:** G-B4: base, instantaneous and effective p5/p50/p95 per instance; the effective period equals p for a fixed alpha (test); in series mode no 'lookback' line but a history-bound line and the bound counts; the D-014 figure tests pass; notebooks rebuilt through build.py and executed, every changed figure looked at.
- **source:** the window-free plan and D-037 (2026-09-29)
- **placement (2026-09-29, D-039):** research track R6, after the MVP.

### NT-069

**A/B-1: the series engine against the window engine (pre-registered)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** runs/experiments/series_engine_v1/
- **depends on:** NT-065, NT-066, NT-067, NT-041 (with its amendment), NT-032 (with its amendment), NT-063
- **why:** docs/research/2026-09-29-window-free-plan/README.md, spec (1); D-025, D-037.
- **acceptance:** (1) A SPEC equal to spec (1) or stricter, committed before any GPU time, with F and EPOCHS fixed from the dev runs before any judged run. (2) The comparator's verdict with every bound, the joint power, the contention log and every run id. (3) The recomputed worst-case budget within D-037's ceiling (about 6 GPU-hours), else the owner. (4) No choice uses a judgement or test fold.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-070

**A/B-1b: removing the 60-bar clip in series mode (pre-registered)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** the clip switch (an implementer part), runs/experiments/series_no_clip_v1/
- **depends on:** NT-069 ADOPT
- **why:** docs/research/2026-09-29-window-free-plan/README.md, spec (1b): the only study that delivers D-032's unlimited periods.
- **acceptance:** (1) An implementer part adds the switch that removes the clip in series mode (tests; golden run unchanged in window mode). (2) A SPEC as spec (1b); the verdict, the secondaries (drift of the slow periods, bound counts) and a budget within D-037's ceiling.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-071

**GPU probe: epoch- or update-bound on 7-day blocks with the adopted engine**

- **status:** todo
- **priority / type / role:** P2 / research / experimenter
- **area:** runs/experiments/window_free_probe_v0/
- **depends on:** NT-069 ADOPT, NT-041, NT-062, NT-063
- **why:** docs/research/2026-09-29-window-free-plan/README.md, spec (0).
- **acceptance:** A SPEC as spec (0) (reach at 50% of each run's own span, geometric mean, the quality guard, B1024 capped at 4x A's epochs, 4 seeds per arm); the classification; a budget within about 1.5 GPU-hours.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-072

**Per-bar causal model as a Models registry entry (option B)**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/models/, a chunk sampler, tests/
- **depends on:** NT-071 epoch-bound (or the owner overrides), NT-042
- **why:** docs/research/2026-09-29-window-free-plan/README.md, stage 7; the first round's item.
- **acceptance:** The first round's item criteria (gates T1-T6 on CPU; per-position towers for N horizons; strided physics terms; val_loss with A's function at identical anchors; status.json records fit_seconds and optimizer steps); one GPU sanity run at least 4x faster per epoch, else the item stops with a record.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-073

**A/B-2: the per-bar model against the default (pre-registered)**

- **status:** todo
- **priority / type / role:** P2 / research / experimenter
- **area:** runs/experiments/per_bar_model_v1/
- **depends on:** NT-072, NT-032
- **why:** docs/research/2026-09-29-window-free-plan/README.md, spec (2).
- **acceptance:** A SPEC as spec (2) (time to the served epoch, lower bound at ln 2, guard-rails as A/B-1, two looks of 12); the verdict; a budget within about 5.6 GPU-hours.
- **source:** the window-free plan and D-037 (2026-09-29)

### NT-074

**Same-seed runs differ at epoch 0 with op determinism on: find and fix the source**

- **status:** todo
- **priority / type / role:** P1 / bug / implementer
- **area:** src/neural_trade/utils/seeding.py, the data pipeline (tf.data shuffle and map), models/layers/vacuum_saturation_noise.py, training/trainer.py, tests/
- **why:** NT-035 (2026-09-29): three runs with seed 777 and op determinism on (TF_DETERMINISTIC_OPS=1 plus enable_op_determinism) gave val_loss 9.5673 / 9.6394 / 9.6603 at epoch 0 on the GPU; no op raised. D-025 assumes a deterministic mode makes comparison studies reproducible; it does not yet, so paired studies must use several seeds. Candidates: PYTHONHASHSEED unset on this path, the tf.data shuffle or parallel map order, the vacuum-noise layer's random numbers, CPU-pinned ops.
- **acceptance:** (1) The source is identified with evidence (a CPU test and, by the experimenter, a short GPU check). (2) Two same-seed runs in the deterministic mode give identical val_loss per epoch on the CPU (test) and, if the source is fixable on the GPU, on the GPU (3 runs, recorded). (3) If full GPU reproducibility is impossible in TF 2.10, the item records why and DECISIONS gets a corrected reading of D-025. (4) Speed unchanged (D-018); fast suite and ruff pass.
- **source:** NT-035 REPORT (2026-09-29)

### NT-075

**Did sec_per_step regress on the MVP-1 head? (0.1066 vs 0.0984, one run each)**

- **status:** todo
- **priority / type / role:** P1 / performance / experimenter
- **area:** runs/experiments/speed_check_mvp1/ (SPEC, REPORT)
- **why:** The notebook run on the MVP-1 head (runs/20260929T081632Z-426de4f-dirty-aba344d6) logged sec_per_step 0.1066 against 0.0984 for the previous notebook run (runs/20260924T182915Z-1aeff1c-dirty-af67ee43), +8%. The session touched callbacks (NT-028) and moved modules (NT-027) but not the per-step path; golden runs are equal. One run per side is not noise-aware (D-012), and the desktop alone showed about 40% GPU utilisation at times (NT-035). D-018: the per-step path must not get slower.
- **acceptance:** (1) At least 3 interleaved real runs per side (the MVP-1 head against 1aeff1c's commit or fd960cd, same data and config), each after the RUNBOOK GPU-free check; the median epoch time of epochs 1 and later and sec_per_step, with their spread. (2) A verdict: no regression within noise, or a regression with its size; in the latter case an implementer item that finds the cause. (3) GPU time under 1 hour.
- **source:** handoff 2026-09-29

### NT-076

**Engine: store each cell's predictions; `scenario rescore` compares strategies on stored cells (CPU)**

- **status:** done (2026-09-29): 82a848f, merged into nt-005-strategy-study as df9a202; QA PASS on every criterion (score_result byte-identical to f8b4253 on 12 settings x 358 scores; rescore of the scenario's own strategy 0 mismatches on 342 keys; fit on cal only checked by perturbing the OOS frame); fast suite 836 passed; ruff clean. QA findings: a 0-trade row ranks first (NT-031 note), spec values of cal-fitted strategies unchecked (NT-079).
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/experiments/ (scorer.py, runner.py, new rescore.py), src/neural_trade/evaluation/frame.py, src/neural_trade/cli.py, configs/strategy_studies/, tests/, docs/RUNBOOK.md
- **depends on:** NT-026
- **why:** The strategy study (NT-005, owner request 2026-09-29) compares many strategy configurations on the reference scenario's dev folds. The engine scores each cell with one strategy and keeps no predictions, so every comparison would retrain (about 5 GPU-minutes per cell).
- **acceptance:** (1) Each scored cell writes predictions_cal.npz and predictions_oos.npz (served and raw deltas, probabilities, variance, intervals, scales, OHLC at the anchor bars); a backtest from the loaded files equals the scorer's in-memory one on every summary key (test). (2) A strategy study spec (entries with optional grids; unknown keys, strategies and params refused). (3) `neural-trade scenario rescore SPEC --study STUDY` fits every configuration on each cell's cal block and backtests its OOS block with the scorer's rules and baselines, writing a new rescore directory with cells.csv, leaderboard.csv/.md ranked by the mean dev-cell net Sharpe (test columns shown, never ranking; test), study.yaml, meta.json. (4) Cells without predictions are skipped and listed; no dev cell with predictions exits non-zero. (5) Rescoring the scenario's own strategy reproduces each cell's result.json backtest scores (test). (6) Fast suite, ruff, TESTING_DOCUMENTATION.md, RUNBOOK.
- **source:** owner request 2026-09-29

### NT-077

**Target-exposure backtest mode and the shortlisted variance-driven strategies with EWMA twins**

- **status:** done (2026-09-29): 4b31c7a, merged into nt-005-strategy-study as 6aa7aaf; QA PASS on every criterion (independent exposure simulator matches the engine to 8.7e-19 on 8 configurations; causality bit-identical on 20 cases x 5 probes; existing strategies' numbers identical to 560fc8c on 843 values); fast suite 870 passed at 4b31c7a; ruff clean. Declared deviations accepted: horizon default -1 (h2 here, D-022), SignalFrame.horizon_bars, a 3-line exposure branch in tests/test_notebook_ui.py. Findings: NT-080, NT-081.
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/strategy/ (signals.py, backtest.py, strategies.py, performance.py), tests/
- **depends on:** none (runs beside NT-076 on disjoint files)
- **why:** The strategy research (docs/research/2026-09-29-strategy-architectures/) finds directional trading of this model cannot break even at 26 bps; its shortlist uses the variance forecast, and two architectures need a continuous target exposure with a rebalancing band, which the one-position engine cannot express. Each variance-driven strategy needs a model-free EWMA twin to show whether the model's sigma adds anything.
- **acceptance:** (1) SignalFrame gains sigma_ret, mu_gauss and a causal sigma_ewma (half-life 60, 240-bar warm-up). (2) ExposureStrategy and run_exposure_backtest (decide at close, band, fill at next open, drift, per-rebalance costs); hand-computed unit tests. (3) Discrete summaries add traded_notional, breakeven_cost_bps, gross_edge_per_trade_bps. (4) A circular-shift timing null for exposure strategies under the random null's keys. (5) Registered vol_target, net_edge_kelly, edge_over_cost, vol_regime_long, gated_ta, fitted on cal only, sigma_source model or ewma. (6) assert_no_lookahead covers every registered strategy, both sigma sources; fit reads cal only (test). (7) Existing numbers unchanged; fast suite, ruff, TESTING_DOCUMENTATION.md.
- **source:** owner request 2026-09-29; the research note sections 3 and 5

### NT-078

**EWMA and HAR-RV variance baselines in the evaluation report, same block, with the DM test**

- **status:** todo
- **priority / type / role:** P1 / research / implementer
- **area:** src/neural_trade/evaluation/ (baselines, report), tests/
- **why:** The strategy research (docs/research/2026-09-29-strategy-architectures/ section 2.0) measured an EWMA of squared 1-minute returns (half-life 60) at CRPSS +0.06 to +0.07 against a constant variance on the reference file before the test block, against the model's +0.009 to +0.020 on its test block: different blocks, so a warning, not a verdict. If a free volatility forecast beats the model's sigma, the model's only measured edge is not an edge; this bears on the indicator-learning purpose (VISION) too.
- **acceptance:** (1) The evaluation report's variance section carries EWMA (half-life 60, fitted scale on the train block) and HAR-RV (Corsi 2009, fitted on the train block) baselines on the same block, with CRPSS, the var/err^2 Spearman and the Diebold-Mariano test it already runs against const_var (noise-aware, D-012). (2) Tests on synthetic data with known variance. (3) The notebook that shows the variance baselines is updated through build.py (D-028).
- **source:** the strategy research note 2026-09-29, section 2.0 and follow-up 1

### NT-079

**Strategy study spec: values of cal-fitted strategies are not checked at load**

- **status:** todo
- **priority / type / role:** P3 / bug / implementer
- **area:** src/neural_trade/experiments/rescore.py (spec load, about lines 215-220), tests/test_rescore.py
- **why:** QA of NT-076: `grid: {entry_quantile: [0.9, 1.5]}` or a string value is accepted at load for a from_calibration strategy and fails only during the rescore, probably with a traceback (only InvalidConfigurationError and RescoreError are caught).
- **acceptance:** Such values are refused at load with exit 2 and nothing written (test).
- **source:** QA of NT-076, 2026-09-29

### NT-080

**Exposure-aware backtest views; explorer hides fitted knobs; YAML loading of cal-fitted strategies**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/visualization/ (trade_analytics, the trading dashboard), src/neural_trade/notebook/backtest_ui.py (BacktestExplorer), src/neural_trade/strategy/params.py (from_file), scripts/notebooks/build.py (notebook 05), tests/
- **why:** NT-077 (implementer report, 2026-09-29): the exposure strategies (vol_target, net_edge_kelly) have rebalances, not trades. The trade-analytics view then says "placed N orders but none became a trade", and the dashboard draws buys and sells as long and short markers. BacktestExplorer.widget shows fitted fields (sigma_star, in_below, out_above, gate) as knobs that do nothing. params.from_file passes no calibration, so it cannot build the five cal-fitted strategies. Notebook 05's compare_strategies() builds every registered strategy, so its next execution shows the new strategies through these views.
- **acceptance:** (1) An exposure result gets an exposure view (the target and held exposure paths, rebalances, cost drag, break-even cost) instead of the trade views (test, figure looked at, D-014). (2) The explorer skips fitted_fields (test). (3) from_file accepts a calibration (or a documented refusal) and configs/strategies/ gets an example for a cal-fitted strategy (test). (4) Notebook 05 re-executed through the routine (D-013, D-028).
- **source:** NT-077 implementer report, 2026-09-29

### NT-081

**Timing null replay micro-rebalances after a capped fill; a full target leaves cash negative by the costs**

- **status:** todo
- **priority / type / role:** P2 / bug / implementer
- **area:** src/neural_trade/strategy/backtest.py (_ReplayFills.target, about line 432; fill()), tests/test_exposure_backtest.py
- **why:** QA of NT-077: on bars without an event `_ReplayFills.target` returns `current` instead of NaN. After a fill to the 1.0 cap the costs leave cash slightly negative, the exposure drifts above 1, the engine clips `current` back to 1.0 and, with band 0, re-trades on following bars. On real data (vol_target, ewma) the unshifted replay's equity differs from the strategy's by $0.0078 on $10,000 and shifted copies make 14-16 rebalances against the strategy's 10. The effect on the null's returns is about 1e-6 of equity. P3 part: a target of 1.0 holds about 1.00015 exposure after an entry (implicit leverage on spot).
- **acceptance:** (1) The replay returns NaN (no decision) on bars without an event; the unshifted replay of a path that hits the cap reproduces the strategy's equity to 1e-9 and its rebalance count (test). (2) A full target never holds more than max_abs_exposure after costs, or the docstring states the tolerance (test).
- **source:** QA of NT-077, 2026-09-29

### NT-082

**Long-history run: SHUFFLE_BUFFER setting and notebook 08 (launch and live progress of an engine run)**

- **status:** done (2026-09-29): f029e39, merged into remediation/plan; QA PASS on every criterion (default shuffle order identical to 99891f8; SHUFFLE_BUFFER 0 permutes the whole block each epoch; fold -2 train 518,432 windows = 360.02 days, 2024-06-01 to 2025-05-28, dev OOS 2025-07-27 to 2025-08-28, fold -1 untouched; a detached dummy process outlived its parent; figures rendered and inspected); fast suite 928 passed; ruff clean. Findings: NT-083, NT-084.
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/core/config.py, src/neural_trade/data/datasets.py, a new src/neural_trade/notebook/longrun.py, scripts/notebooks/build.py (notebook 08), configs/scenarios/long_360d.yaml, tests/, docs (config reference, RUNBOOK)
- **why:** D-040: one 360-day training run on Bitcoin_BTCUSDT.csv, launched and tracked through a notebook. The training shuffle buffer is fixed at 2,048 windows (datasets.py:21), about 1.4 days of consecutive windows: on a 360-day block every batch would come from one narrow time slice.
- **acceptance:** see the implementer brief (2026-09-29); in short: (1) Config SHUFFLE_BUFFER (default 2048 = today's numbers unchanged; 0 = the whole training block), wired into create_datasets, with its metadata and the generated reference; (2) notebook 08 generated by build.py: a guarded launch cell (off unless set) that starts the scenario as a detached background process, and monitor cells reading status.json / metrics.jsonl / training_log.csv of the newest cell (progress, ETA, loss and metric curves per epoch in the theme, GPU utilisation) and the scored result once it exists; (3) the scenario spec for the run; (4) tests, fast suite, ruff.
- **source:** D-040, owner Q&A 2026-09-29 (long training)

### NT-083

**Adding a Config field changes every engine cell's config_hash: finished cells re-run and rescore skips them**

- **status:** done (2026-09-30): e293778, merged; QA (Opus) PASS: identity = non-default fields minus run-directory fields (no collision in 3,953 random Configs; injective by construction); every committed scenario's cells show done again (58 cells pending at base -> done); strategy_study_v1 rescore reproduces the committed cells.csv exactly (180 x 87); a stale recorded hash is ignored, a changed non-default field retrains; fast 1022 / ruff; training untouched. Findings: NT-093.
- **priority / type / role:** P1 / bug / implementer
- **area:** src/neural_trade/experiments/scenario.py (config_hash), src/neural_trade/experiments/runner.py, src/neural_trade/experiments/rescore.py (select_cells), tests/
- **why:** NT-082's implementer found that the new SHUFFLE_BUFFER field changes config_hash for every existing cell: `scenario plan configs/scenarios/reference.yaml` shows the 9 reference_default cells as pending again (a re-run would retrain them, about 0.6 GPU-hours), and `scenario rescore` skips them as an older spec, so the NT-005 study (runs/experiments/strategy_study_v1) can no longer be re-scored at the merged head. Every future Config field repeats this.
- **acceptance:** (1) A cell's resume/rescore identity ignores Config fields that did not exist when the cell ran and are at their default (for example: hash only the fields recorded in the cell's config.yaml, or hash the non-default values); a field added later with a non-default value still changes the identity (tests). (2) At the merged head, `scenario plan configs/scenarios/reference.yaml` lists the 9 reference_default cells as done and the NT-005 rescore reproduces its cells.csv (CPU). (3) Fast suite, ruff.
- **source:** NT-082 implementer report, 2026-09-29

### NT-084

**Notebook 08 / longrun.py edge cases; the notebook kernel's PYTHONPATH in worktrees**

- **status:** todo
- **priority / type / role:** P3 / polish / implementer
- **area:** src/neural_trade/notebook/longrun.py, tests/test_notebooks_thin.py, tests/test_longrun.py
- **why:** QA of NT-082 (2026-09-29): (1) tests/test_notebooks_thin.py:57 `_run`: the notebook kernel does not get <repo>/src on PYTHONPATH, so from a worktree it imports the main checkout and fails on new modules. (2) longrun.py:531: progress shows the newest cell directory whatever its config hash, while launch uses the runner's hash rule (DONE shown while launch would start a new run). (3) longrun.py:520-567: while a fresh launch is still loading data, progress shows an older incomplete cell as running. (4) longrun.py:232: psutil AccessDenied makes pid_alive False; a reused pid can block launch. (5) longrun.py:736: the subtitle's elapsed is wall clock while panel 5's is the sum of epoch times, unlabelled. (6) longrun.py:399: two launches in the same UTC second share log and pid names. (7) The training dashboard reused in notebook 08 says "~38 min left" on a run that early stopping ended (epoch 12 of 40).
- **acceptance:** each of (1)-(6) fixed or documented as accepted, with a test where behaviour changes.
- **source:** QA of NT-082, 2026-09-29

### NT-085

**The micro loop (D-041): minutes-long runs iterating toward predictive power and PnL**

- **status:** in-progress (lead, 2026-09-29)
- **priority / type / role:** P1 / research / lead+experimenter
- **area:** runs/experiments/micro_loop_v1/LOG.md (the journal), configs/scenarios/micro_*.yaml, configs/strategy_studies/
- **why:** D-041. The 360-day run showed data volume is not the limit; the zero-cost re-score showed a real but tiny timing signal. The loop tests one hypothesis at a time on micro setups (~10-day train, batch 2048, ~4 min per cell; CPU rescore where no retraining is needed) and records each in the journal.
- **acceptance:** Each hypothesis gets a journal row with its method, cost and result, and evidence committed (rescore dirs, scenario cells' light files). Quick sweeps stay within OPERATING_MODEL's sweep rules; a claimed improvement to the reference setup goes through D-025 before any default changes.
- **source:** owner /goal 2026-09-29 (D-041)

### NT-086

**The slow notebook test runs the main checkout's src/ from a worktree (false passes)**

- **status:** todo (folded into NT-047's repair round 1; closes with it if QA confirms)
- **priority / type / role:** P1 / bug / implementer
- **area:** tests/test_notebooks_thin.py (`_run`)
- **why:** QA of NT-047: the Jupyter kernel of `test_all_notebooks_execute` does not get the checkout's own src/ on PYTHONPATH, so in any worktree it imports D:/neural_trade/src and passes on the main checkout's code (NT-047's notebook crash was hidden this way). CI does not run the slow suite, so nothing else catches it. Same root as NT-084 item (1).
- **acceptance:** the kernel imports the checkout under test (as scripts/notebooks/execute.py:98 sets it); a test proves it from a worktree.
- **source:** QA of NT-047, 2026-09-29

### NT-087

**pnl_utility objective: net P&L after costs on the direction heads (P&L plan E2)**

- **status:** done (2026-09-30): 50c0b3e, merged into remediation/plan; re-QA (Opus) PASS after repair round 1: realized_vol sigma equals the raw-close value (max |ratio-1| 2.4e-7; median |r~| 0.60, c~ 1.2-1.7 on the intended scale), the cost and risk probes through pnl_utility reproduce their expected values, planted-edge / no-edge behaviour, finite pnl gradients on 36 extreme cases, golden bit-for-bit (LOSS_NAME custom_loss unchanged; LAMBDA_PNL defaults to 0), fast 977 / ruff. QA note for E2: in the full loss at LAMBDA_PNL 0.25/0.50 a realistic 60-100 bps edge in a toy acts mainly as cost shrinkage (confounded toy). Findings: NT-090 (sigma floor on flat windows), P3s listed there.
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/losses/functions.py, src/neural_trade/core/config.py, src/neural_trade/training/lambda_calibration.py, src/neural_trade/training/custom_model.py (only if logging requires), tests/
- **why:** The owner's point 3 ("absence of the PnL in the models' targets") and the /goal; docs/research/2026-09-29-pnl-target/README.md section 1.1 option A and section 3 "E2". E1 (cost-sensitive labels) failed its signal gate.
- **acceptance:** (1) synthetic data with a planted edge above cost: mean |2p - 1| grows and the utility rises; (2) synthetic data without an edge: positions go to 0 (the flat solution); (3) finite gradients on the stability cases (D-026); (4) sec_per_step not slower than custom_loss within noise (D-018); (5) `LOSS_NAME: custom_loss` runs bit-identical (golden verify); (6) fast suite, ruff, TESTING_DOCUMENTATION, config reference.
- **source:** the P&L research note (a9f296f), E2

### NT-088

**Screen mode: mass ultra-small runs (6-hour training block) for maths, hyperparameters and losses**

- **status:** done (2026-09-30): 6c156da, merged into remediation/plan (3e1528d; lead fix 11c2897: pnl_val joins the term shares); final re-QA (Opus) PASS after two repair rounds: term shares re-derived from custom_loss (residual 1.2e-6 with coherence off), calibrated and ablated lambdas read from the trained model, windows built once per data key (prep_s 0.007 s after the first trial), non-finite trials recorded as failed, D-020 pre-flight on the long file, 360-window example, per-shard result files, timings with epoch_s (trace ~13-14 s of a ~20 s CPU trial), golden bit-for-bit; fast suite on the merged head 1011 passed, ruff clean. Findings: NT-091.
- **priority / type / role:** P1 / feature / implementer
- **area:** a new src/neural_trade/experiments/screen.py, src/neural_trade/cli.py (`screen` command), src/neural_trade/core/config.py (DATA_END), src/neural_trade/data/processor.py (the slice), tests/, docs/RUNBOOK.md
- **why:** owner request 2026-09-29: plan runs on 6-hour training blocks of minute data (very small runs) to mass-test maths, hyperparameters and losses; approved plan docs/research/2026-09-29-screen-plan.md. Today a cell's cost is the harness, not training: on Bitcoin_BTCUSDT.csv every cell reads and windows the whole file twice (trainer.py:292, scorer.py:265), traces the graph, fits baselines, runs 100 null backtests and writes npz files (scoring alone 56-104 s).
- **acceptance:** (1) `neural-trade screen SPEC [--shard i/N] [--store runs]`: windows built once per data key per process and passed to training and scoring; no baselines, backtest, null, npz, checkpoints, bundle; calibrate switchable. (2) Config DATA_END (timestamp) selects the slice's end anywhere in the file; a slice overlapping the long file's dev/test period is refused (D-020) (test). (3) Trials from grid axes plus random/LHS sampling over Config.field_specs() with explicit bounds in the spec; x slices x seeds. (4) One row per trial in results.jsonl (health: finite checks, nonfinite_grad_steps, max grad norm, clipped-step share, train-loss drop, val loss, per-term shares, timings split into load / build+trace / train / score); resumable (finished trials are skipped) (test). (5) Pre-registered screen rules in the spec mark a trial pass/fail; a known-bad config (LR 1.0, a lambda of 1e6) fails (test). (6) Disjoint shards whose union is all trials (test). (7) scenario run and the golden run unchanged (bit-for-bit); fast suite, ruff, TESTING_DOCUMENTATION, config reference, RUNBOOK section.
- **source:** owner request 2026-09-29; docs/research/2026-09-29-screen-plan.md

### NT-089

**HD physics term: a +inf bar in x_window sends NaN gradients to the variance heads even at LAMBDA_HD 0**

- **scope note (2026-09-30, re-QA of NT-087):** -inf, NaN and 1e30 bars reproduce it too (at base 268379d under custom_loss); the fix covers all of them.

- **status:** todo
- **priority / type / role:** P2 / bug / implementer
- **area:** src/neural_trade/losses/functions.py (hyper_decoherence_coupling_loss: local_vol / log_vol via an unguarded tf.math.reduce_std, _z), tests/
- **why:** NT-087 repair round 1 (2026-09-30): with a +inf bar in x_window the gradient of out.total into the variance heads is NaN under plain custom_loss, even with LAMBDA_HD 0; the term's forward value is zeroed by its tf.where guard, but the chain rule multiplies the zeroed upstream gradient by an internally NaN local Jacobian (0 x NaN). The train step zeroes and counts such steps (custom_model.py:478-486), so a step is skipped, not corrupted. D-026 stability gap; relevant to NT-036/NT-038.
- **acceptance:** sanitise local_vol / log_vol before use (as NT-087 did for sigma); a test with a +inf bar at LAMBDA_HD 0 and > 0 gives finite gradients into every head; golden run bit-for-bit.
- **source:** NT-087 implementer report, 2026-09-30

### NT-090

**pnl_utility: sigma floor 1e-6 makes flat windows a 2600x cost; config guard; test pins**

- **status:** todo
- **priority / type / role:** P2 / bug / implementer
- **area:** src/neural_trade/losses/functions.py (pnl_utility _one_horizon), src/neural_trade/core/config.py (validate), tests/test_pnl_utility.py
- **why:** re-QA of NT-087 (2026-09-30): on Bitcoin_BTCUSDT.csv 0.017% of 60-bar windows are flat (sd 0) and 0.39% have c~ > 10; the floor sigma = max(sigma, 1e-6) makes c~ = 2600 there and the per-sample direction gradient ~80x typical (only GRAD_CLIP_NORM bounds it). Nothing refuses LOSS_NAME pnl_utility + realized_vol with WINDOW_NORMALIZER per_lag_standard (sigma silently wrong). realized_vol sanitising is inconsistent (a +-inf/NaN bar makes the term vanish; an overflowing sd hits the floor = maximum cost). The data test test_realized_vol_sigma_matches_raw_close_scale_on_the_bundled_csv never calls pnl_utility (passes on the buggy code).
- **acceptance:** a floor relative to the typical sigma (or flat windows masked), test on a flat window; Config.validate refuses the per_lag_standard combination; consistent sanitising (test); the data test evaluates pnl_utility (e.g. the cost probe) so it pins production.
- **source:** re-QA of NT-087, 2026-09-30

### NT-091

**DATA_END protection: floor for short files and outside screen mode; screen resume across shard counts; first-trial windowing of the whole file**

- **status:** todo
- **priority / type / role:** P2 / bug / implementer
- **area:** src/neural_trade/experiments/screen.py, src/neural_trade/data/processor.py, src/neural_trade/core/config.py, tests/
- **why:** re-QA of NT-088 (2026-09-30): (1) on files shorter than 64 days DATA_END_PROTECTED_DAYS can be lowered to 0, so a screen on the bundled CSV can train on its test block (the one notebooks 02-05 report); (2) outside screen mode any Config can set DATA_END_PROTECTED_DAYS 0 with a late DATA_END on the long file (only the screen pre-flight enforces the floor); (3) resume only sees shard files with the same N (a plain run after --shard 0/3 re-runs 22 duplicates); (4) the first trial of each data key windows the whole file up to DATA_END (10-20 s) and each slice reloads the CSV (6-7 s).
- **acceptance:** a floor for short files (e.g. the default fold's test start) and the same protection in Config.validate / processor for every path (tests); resume across shard counts (test); window only the needed tail before windowing (timing before/after).
- **source:** re-QA of NT-088, 2026-09-30

### NT-092

**Screen phase 2: reuse the traced graph across trials (trace is 73% of a 6-hour trial); clip rule skips the first epoch**

- **status:** done (2026-09-30): 57768cb, merged (7a1b8b0); re-QA (Opus) PASS after repair round 1: later reused trials equal independent fresh runs bit-for-bit (4- and 6-trial groups, calibrate on, ablation group), Keras RNG flag restored on exit and on error, a normal run after screen trials in one process equals the golden record, default path unchanged (SEEDED_STOCHASTIC_LAYERS off; CPU step ratio 0.984; checkpoints and pre-NT-092 bundles predict identically), per-trial clip_skip_epochs; fast 1018 / ruff. Speed-up ~2.0-2.6x on CPU after the first trial of a group; GPU per-trial time an estimate until the campaign measures it.
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/experiments/screen.py, src/neural_trade/training/ (only what trace reuse strictly needs), tests/test_screen.py, docs/RUNBOOK.md
- **why:** the approved screen plan (docs/research/2026-09-29-screen-plan.md, step 3 / "phase 2"): build phase 2 if tracing exceeds 50% of a trial. GPU measurement (runs/experiments/micro_loop_v1/LOG.md, 2026-09-30): trace 12.2 s of a 16.7 s trial (73%). Also: at LR 1e-4 every 2-epoch trial fails max_clipped_share because the initial pre-clip norm exceeds the clip.
- **acceptance:** (1) Trials that differ only in continuous values (LR, LAMBDA_*, ADAM betas, GRAD_CLIP_NORM, INDICATOR_LR_MULT / GRAD_MULT, loss weights) reuse one traced train/test step per structural key (model architecture, BATCH_SIZE, horizons, window, loss name, input series...); between trials weights, optimizer state and all lambda / LR variables are reset to the trial's seeded initial values. (2) A reused-graph trial equals a fresh-graph trial of the same config and seed (bit-for-bit, or within a stated float tolerance with the reason) - test. (3) Structural changes retrace (test). (4) Measured median trial wall on the GPU-free CPU path and a stated GPU estimate; target <= 5 s per trial after the first per structural key. (5) Rules: `clip_skip_epochs` (default 1) excludes the first epoch's logged steps from clipped_share; `min_epochs` guard. (6) scenario run and the golden run unchanged; fast suite, ruff, TESTING_DOCUMENTATION, RUNBOOK.
- **source:** screen plan phase 2; NT-088 GPU measurement 2026-09-30

### NT-093

**Identity follow-ups: notebook 08 launch guard trusts the recorded hash; screen trial keys moved once; config_identity docs**

- **status:** todo
- **priority / type / role:** P2 / bug / implementer
- **area:** src/neural_trade/notebook/longrun.py (_cells_of_spec, ~319-326), src/neural_trade/experiments/screen.py (resume keys), src/neural_trade/experiments/scenario.py (config_identity docstring), docs/RUNBOOK.md, tests/
- **why:** QA of NT-083 (2026-09-30): (1) longrun._cells_of_spec still compares the recorded meta.json config_hash, so notebook 08's "already done" guard no longer fires (micro_l2 shows 0 of 15 done) and a relaunch starts a runner that then skips every cell (no retraining, a wasted launch). (2) The new identity moved every recorded screen trial_key (0 of 960 match in runs/screens/l1_*): re-launching a finished screen would re-run all trials and append duplicates; one-time, later field additions no longer move keys. (3) By design a new field whose default changes behaviour leaves old cells marked done, and a removed or tightened field makes from_yaml fail so old cells rerun: undocumented.
- **acceptance:** (1) _cells_of_spec uses config_hash_of_dir (test: an old cell with a stale recorded hash counts as done); (2) screen resume recognises finished rows by recomputing the identity from each row's config diff (or a documented one-time note plus a refusal to append duplicates) (test); (3) config_identity docstring and RUNBOOK state both caveats.
- **source:** QA of NT-083, 2026-09-30

### NT-094

**Trading costs default to 0 (D-044)**

- **status:** done (2026-09-30): 69972e0, merged; QA (Opus) PASS: every default cost 0 at runtime (BacktestConfig, build_backtest_config, scorer / rescore / CLI / notebook UI, cost-aware strategies, PNL_COST_BPS), explicit 13 bps reproduces the base numbers byte-for-byte, arithmetic tests pass explicit costs, golden bit-for-bit, fast 1022 / ruff. Lead fixed the reference.yaml comment. Open: notebook 02's prose 'After costs the return is mostly cost x trade count' (scripts/notebooks/build.py:285) is rewritten at the next notebook routine; notebooks 02/03/05 show zero-cost numbers once executed.
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/strategy/ (BacktestConfig, cost-aware strategies), src/neural_trade/core/config.py (PNL_COST_BPS), configs/, README, RUNBOOK, docs/VISION.md (cost wording), tests/
- **why:** owner decision D-044 (2026-09-30): "издержки делай 0" (make the costs 0).
- **acceptance:** every default trading cost is 0 (fee, half-spread, slippage, the cost-aware strategies' cost, PNL_COST_BPS), the cost fields stay settable; tests pinning cost arithmetic pass explicit costs; docs and VISION wording updated; golden run bit-for-bit; fast suite, ruff, TESTING_DOCUMENTATION, config reference.
- **source:** D-044

### NT-095

**Notebook 09: the candidate run (training, fit, backtest)**

- **status:** done (2026-09-30): 228e52d, merged; QA (Sonnet, P2) PASS: build --check and check.py clean (09: 3.9 MB, 8 figures), notebooks 00-08 untouched; the backtest reproduces manifest C1 f-2/s0 exactly (+16.71%, 1,968 trades, 54.7% wins, max DD 4.66%, buy-and-hold -4.85%) and the key-numbers table matches eval_report_dev.json; `max_line_points=None` default byte-identical (02/03 unaffected); logic in notebook/run_report.py (11 tests); fast 1038 / ruff / TESTING_DOCUMENTATION.
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/notebook/run_report.py (new), notebook/backtest_ui.py, visualization/trading_dashboard.py (optional line thinning for the 5 MB limit), scripts/notebooks/build.py, notebooks/09_candidate_run.ipynb, tests/
- **why:** owner 2026-09-30: "Загрузи прогон с лучшей стратегиией в ноутбук. Я хочу суммаризацию с графиками ее трейна, оценки фита, бэктест" (load the run with the best strategy into a notebook: training, fit evaluation, backtest).
- **acceptance:** a generated, executed notebook 09 under 5 MB on candidate C1's run (long_360d_stab f-2/s0, calibrated_quantile 0.9, zero cost); its numbers match the manifest and the run's dev report; figure defaults of 02/03 unchanged; logic outside the notebook with tests; suites.
- **source:** the owner request above; configs/candidates/ (C1)

### NT-096

**Loss hygiene: epsilon inside every batch std, coherence without its zero-gradient parts and logged, stale comments**

- **status:** todo
- **priority / type / role:** P1 / bug / implementer
- **area:** src/neural_trade/losses/functions.py, src/neural_trade/models/gru_attention.py (comments), src/neural_trade/core/outputs.py, tests/
- **depends on:** none
- **why:** A_losses.md sections 3, 8, 10: `reduce_std` of a batch-constant head differentiates sqrt(0) and gives a NaN gradient in vol, HD, IFE and vacuum (hidden by the step guard, never observed); coherence has two parts with zero gradient (tf.sign, a label-only constant) and a live magnitude-ordering part with gradient norm about 0.9 that is not logged; gru_attention.py:183, 214, 235 call the towers 1-, 5- and 15-minute and :198-201 is a no-op clip.
- **acceptance:** (1) sqrt(var + eps) (or an equivalent guard) in every batch std of the loss; a test feeds a batch-constant head and gets finite gradients for vol, HD, IFE and vacuum. (2) Coherence keeps only the magnitude-ordering part; the logged contrib of coherence equals it (test); the dead parts' removal is shown not to change any gradient (test comparing gradients before and after on a fixed batch). (3) `scripts/golden_run.py verify` passes on the weights, or the implementer shows the only difference is the val_loss offset of the removed constant and the served epoch is unchanged on the golden config. (4) Comments name the configured horizons; the no-op clip is removed with D-029 evidence. (5) Fast suite, ruff, TESTING_DOCUMENTATION.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-097

**Indicator hygiene: bound the applied period, no meta bias, LR schedule for both optimizers, GRAD_MULT 1, applied-period report**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/models/layers/learnable_indicators.py, src/neural_trade/models/gru_attention.py (meta_adjust), src/neural_trade/training/callbacks.py, src/neural_trade/training/optim.py, src/neural_trade/core/config.py, src/neural_trade/evaluation/, tests/
- **depends on:** NT-032 (comparator, for the A/B)
- **why:** B_model_indicators.md 2.2, 4.2, 7 items 5, 6, 7, 9, 10: the per-window shift lets applied periods reach 1.6 and 74.7 bars outside [2, 60]; the base logit and the meta Dense bias are one unidentifiable direction trained by two optimizers; ReduceLROnPlateau never lowers the indicator LR (0.005 while the main one falls to 0.000125); INDICATOR_GRAD_MULT is a no-op under Adam apart from clipping; the logged period is the base value only.
- **acceptance:** (1) The applied period is bounded to [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX] (test on extreme meta inputs). (2) meta_adjust has no bias (config switch, new default after the A/B). (3) The plateau schedule scales both optimizers (config switch; test that both LRs fall). (4) INDICATOR_GRAD_MULT default 1 (config change, documented). (5) Each run reports p5 / p50 / p95 of every applied period on the evaluation block (test). (6) Each change is a Config switch; with all switches at today's values `scripts/golden_run.py verify` passes. (7) A pre-registered A/B (SPEC before GPU, at most 3 variants: today / all four fixes / fixes without the bias change) judged by NT-032 decides the new defaults; D-018 sec_per_step check.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-098

**Per-term gradient shares measured on real trainings (the probe of NT-037)**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** runs/experiments/grad_shares_v1/ (SPEC, REPORT), no code
- **depends on:** NT-037 (probe)
- **why:** A_losses.md section 7: soft ECE carried 97% of the gradient direction on three CPU batches of the served model; a sample, not a distribution over training. The A/Bs of D-045 phase 2 are gated on this being true during training.
- **acceptance:** (1) The probe of NT-037 on, on one 360-day run (fold -2, seed 0, the long_360d_stab config) and three micro-layout runs; GPU budget stated in the SPEC (estimate about 0.7 GPU-hours). (2) REPORT: per term, the gradient-norm share and the cosine with the total per epoch (median and range), and the value share beside it. (3) The gate for NT-099 is written in the SPEC before the runs: soft ECE's gradient share above 50% in most epochs.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045
- **amendment (2026-10-01, D-048):** run on the 6-hour screen layout first (at least 20 trials: 4 volatility slices x 5 seeds, seconds each) and three micro-layout runs; the 360-day run only if the two disagree.

### NT-099

**Pre-registered A/B: soft ECE off, and soft ECE plus the vol penalty off**

- **status:** in-progress (2026-10-03): both D-049 cells are in the verdicts. `ece0` vs control (fold −39 is the re-run `20261003T093120Z-2573116-d0074ed7-ece0__f-39__s0`; the suspect cell stays out of the verdict) and `ece0_vol0` vs control (fold −35 is `20261003T091952Z-2573116-063f015e-ece0_vol0__f-35__s0`) are each non-inferiority PASS, ordinary beats inconclusive, h0 and h2 guard-rails PASS. Defaults are not changed. SPEC adoption is a lead DECISIONS entry plus a follow-up implementer edit. QA has not passed, so the item is not done. Report: `runs/experiments/loss_prune_v1/REPORT.md`.
- **priority / type / role:** P1 / research / experimenter
- **area:** runs/experiments/loss_prune_v1/ (SPEC, REPORT), an engine scenario; no code beyond config
- **depends on:** NT-032 (comparator), NT-098 (the gate)
- **why:** A_losses.md sections 7-8 and recommendations 1, 3: soft ECE is improper and its |.| kink gives an O(1) gradient that does not vanish at calibration; the vol term demands std(mu) = std(y) against every proper score and is always active in the leader.
- **acceptance:** (1) SPEC committed before any GPU time: variants control / LAMBDA_SOFT_ECE 0 / LAMBDA_SOFT_ECE 0 and LAMBDA_VOL 0; the micro layout; judgement folds and at least 5 (seed, fold) pairs named; primary metric CRPSS non-inferiority (margin -0.005); secondary direction BCE and AUC against logreg_lags; guard-rails clipped_share, 90% coverage, 0 non-finite steps; minimum effects; GPU estimate within 3 hours (about 1.5, an estimate). (2) One verdict per variant by NT-032. (3) REPORT with every run id and NT-037 health numbers. (4) An adopted variant becomes the default through a DECISIONS entry citing the verdict.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-100

**Pre-registered A/B: the NLL tail (lower variance weight, Student-t NLL)**

- **status:** todo
- **priority / type / role:** P2 / research / experimenter
- **area:** runs/experiments/nll_tail_v1/; src/neural_trade/losses/functions.py (NLL_KIND option), src/neural_trade/core/config.py, tests/ (through an implementer sub-item)
- **depends on:** NT-099
- **why:** A_losses.md section 5 and recommendation 7: the NLL mean gradient e/v and variance gradient e^2/v are bounded only by the variance floor and the +-100 clip; NLL was the most batch-sensitive term (norm 0.63-4.61); LAMBDA_VAR correlates +0.58 with the screen's mean gradient norm.
- **acceptance:** (1) Code: NLL_KIND gaussian (default, golden run unchanged) or student_t with fixed dof (test: finite, bounded gradients on large residuals). (2) SPEC before GPU: control (NT-099's winner) / LAMBDA_VAR x 0.3 / student_t; metrics CRPSS and coverage, guard-rails as NT-099; within 3 GPU-hours. (3) Verdict by NT-032; adoption through DECISIONS.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-101

**Gradient-norm loss-weight calibration mode (CALIB_MODE: gradient), default off**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/training/lambda_calibration.py, src/neural_trade/core/config.py, tests/
- **depends on:** NT-037 (per-term gradients), NT-099 (the cleaned objective)
- **why:** A_losses.md recommendation 2: the calibration pass equalises loss values (lambda_calibration.py:207-210), so soft ECE went 1 -> 2.907 because its value is small while its gradient was already the largest.
- **acceptance:** (1) CALIB_MODE value (default, golden run unchanged) or gradient: weights set so each term's gradient norm on the shared trunk is equal (GradNorm-style, measured over the same calibration steps), clipped to [0.1, 20] (test on a synthetic two-term loss: equal gradient norms after calibration). (2) The chosen weights and each term's gradient share are written to meta.json (test). (3) Cost of the calibration pass reported. (4) NT-039 (the existing A/B) runs this mode against value calibration.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-102

**Re-choose GRAD_CLIP_NORM and the max_clipped_share rule on the cleaned loss**

- **status:** todo
- **priority / type / role:** P1 / research / experimenter
- **area:** configs/screens/ (a spec), runs/screens/, docs/DECISIONS.md (the lead records the choice)
- **depends on:** NT-101 and NT-039 (the objective the clip is chosen for)
- **why:** A_losses.md recommendation 10 and B_model_indicators.md 6: the mean gradient norm (about 20) sat at the clip (20); 44% of screen trials failed only the clipped-share rule; Adam bounds each step by 7.27 x lr regardless of the clip.
- **acceptance:** (1) A quick screen of GRAD_CLIP_NORM x LR on the cleaned objective (rules fixed in the spec before it runs; within 1 GPU-hour, an estimate). (2) The report gives the gradient-norm distribution and the pass rates. (3) The new GRAD_CLIP_NORM and max_clipped_share are chosen from the dev slices and recorded in DECISIONS with the evidence.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045
- **amendment (2026-10-01, D-047/D-048):** on the new default (OHLCV + 14 families, D-047) the NT-037 implementer reported a pre-clip norm of about 900; QA of NT-037 did not reproduce it: in the short-run test config (batch 64) the pre-clip maximum was 171 (main) / 235 (indicator) and the main group clipped on 17/30, 26/30 and 30/30 steps of epochs 1-3; at the default batch size 5-13% of steps clipped (identical on af07d11, so it is the new default, not a regression). Before the loss-pruning A/Bs, measure the norm distribution on the 6-hour screen layout (D-048) with the new default, and decide the clip (or accept clipping as the operating regime, since Adam bounds each step anyway: A_losses.md) with the evidence in DECISIONS.

### NT-103

**Epoch selection on proper scores (EPOCH_SELECT_METRIC); waits for the owner (D-011)**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/training/callbacks.py (EarlyStopping / best-epoch restore), src/neural_trade/training/trainer.py, src/neural_trade/core/config.py, tests/
- **depends on:** an owner decision (the rule of D-011 is on val_loss); NT-032
- **why:** A_losses.md solvability: epoch-to-epoch swings of the combined val_loss (0.1-0.3) are 100-200x the whole achievable direction gain (about 1.4e-3 weighted), so the served epoch is blind to direction.
- **acceptance:** (1) Not picked before the owner answers (STATUS question). (2) EPOCH_SELECT_METRIC: val_loss (default, golden run unchanged) or a pre-registered sum of proper scores (val direction BCE + val CRPS, each normalised by its epoch-1 value) (test: the restored epoch is the argmin of the chosen metric). (3) An A/B through NT-032 decides the default.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-104

**Capacity variants through the Models registry, deep direction logit zero-initialised; then the capacity A/B**

- **status:** todo
- **priority / type / role:** P1 / feature / implementer
- **area:** src/neural_trade/models/ (new registered variants), src/neural_trade/core/config.py, tests/; runs/experiments/capacity_v1/
- **depends on:** NT-099 or NT-101 (the objective), NT-032
- **why:** B_model_indicators.md 1.1 and 7 item 2: 296,591 parameters, 44.6% in one attention; the network is never above a 3-lag logistic regression and significantly below it at 1 h (LOG.md L2, z -3.3..-3.5). The direction head already holds a logistic skip (DIRECTION_SKIP).
- **acceptance:** (1) Registered model variants (D-002): gru_small (indicators -> GRU(32) -> heads) and linear_indicators (indicators -> linear heads), same heads and losses; default unchanged (golden run). (2) A config switch zero-initialises the deep direction logit, and the run reports the skip's share of the logit variance (test). (3) SPEC and A/B (at most 3 variants: today / gru_small / linear_indicators) through NT-032 on direction AUC, BCE and CRPSS; within 3 GPU-hours or the owner.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-105

**Attention across the indicator channels and pooling instead of Flatten; then an A/B**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/models/gru_attention.py, tests/
- **depends on:** NT-104
- **why:** B_model_indicators.md 1.4 and 7 item 1: the block commented 'attend across indicators' (gru_attention.py:69-73) attends across the 128 GRU units with the 60 time positions as features (513L + 384 parameters); Flatten -> Dense adds 512L + 32; both tie the model to the window length.
- **acceptance:** (1) A config switch: attention across the 31 indicator channels before the GRU (tokens = channels) or no block; pooling instead of Flatten. (2) Default unchanged (golden run); a test builds the model at L = 60 and L = 120 with the new switches and the parameter count does not depend on L. (3) An A/B through NT-032 decides.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-106

**MACD parametrised as fast = r x slow; a fast leg may reach the price; then an A/B**

- **status:** todo
- **priority / type / role:** P3 / feature / implementer
- **area:** src/neural_trade/indicators/families.py, src/neural_trade/models/layers/learnable_indicators.py, tests/
- **depends on:** NT-097
- **why:** B_model_indicators.md 4.1 and 7 item 8: macd_1_fast sits at the floor of 2 in 5 of 6 runs, ma_period_0 and macd_2_fast in 2 of 6; fast < slow is not enforced (a mirror symmetry).
- **acceptance:** (1) A MACD parametrisation switch: fast = r x slow with r in (0, 1) learned as a logit; the floor lets a fast leg reach p = 1 explicitly. (2) Default unchanged (golden run); a test shows fast < slow always holds under the switch. (3) An A/B through NT-032 on the identified periods and direction AUC.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-107

**Scale-free inputs: each window normalised by its own sigma, the dollar target rescaled at the output; then an A/B**

- **status:** todo
- **priority / type / role:** P2 / feature / implementer
- **area:** src/neural_trade/data/scaling.py, src/neural_trade/data/processor.py, src/neural_trade/models/gru_attention.py (a sigma input), src/neural_trade/training/trainer.py, tests/
- **depends on:** NT-041 (units), NT-032
- **why:** B_model_indicators.md 5.2 and 7 item 14: inputs are dollars / one scale per block, so a 16.0 -> 14.1 bps volatility reads as $105.6 -> $148.4 a year later; the variance head must learn the price level.
- **acceptance:** (1) An input-normalisation switch: each window divided by its own realised sigma, sigma fed as a separate feature; the target stays in dollars (D-022) and is rescaled at the output (test: served predictions in dollars match the old path's units). (2) Default unchanged (golden run). (3) An A/B through NT-032 on CRPSS and direction.
- **source:** docs/research/2026-09-30-math-report/ (A_losses.md, B_model_indicators.md); presentations/4_math_report.html; D-045

### NT-108

**Stochastic-layer reset seeds derived from model.submodules position: any new tf.Module attribute silently changes screen-mode numbers**

- **status:** todo
- **priority / type / role:** P2 / bug / implementer
- **area:** src/neural_trade/training/reset.py, tests/
- **depends on:** none
- **why:** found by the NT-037 implementer (2026-09-30): `reset_stateful_rngs` derives each stochastic layer's seed from its position in `model.submodules`; adding tf.keras Metric counters to CustomTrainModel (a Metric is a tf.Module) shifted every later layer's seed and changed SEEDED_STOCHASTIC_LAYERS screen numbers (caught by tests/test_screen.py, e.g. lambda_t_perp 1.29 vs 0.89). NT-037 at 541dee0 did not avoid it after all: its 18 contrib_* Mean metrics still shift the positions (QA: model.submodules 168 -> 186 entries); NT-037's repair round 1 must keep the positions unchanged. The mechanism stays fragile to any future Metric, Layer or nested Model attribute.
- **acceptance:** (1) Seeds derive from a stable identity (the layer's name or path), not its enumeration position (test: adding an unrelated tf.Module attribute to the model leaves every stochastic layer's derived seed unchanged). (2) A golden screen record made before the change is reproduced or the difference is documented and a new record committed (the derived seeds change once). (3) Fast and slow suites, ruff.
- **source:** NT-037 implementer report (nt-037 541dee0)

### NT-109

**Shrink the six slowest fast-suite tests (28-55 s default-config trainings)**

- **status:** todo
- **priority / type / role:** P2 / performance / implementer
- **area:** tests/test_served_epoch.py, tests/registries/test_contracts.py, tests/test_train_smoke.py, tests/conftest.py
- **depends on:** none
- **why:** D-048: with `-n 8` the fast suite takes 3 min 26 s and is bound by six tests that train the default config (OHLCV, 14 families since D-047) for 28-55 s each (test_served_epoch 55 s, test_training_and_reporting_query_all_ten_registries 49 s, three test_train_smoke tests 43-48 s, test_training_diagnostics_are_subsampled 29 s; `--durations` of 2026-10-01).
- **acceptance:** (1) Each of the six keeps what it asserts, on a smaller config (fewer windows or families, a short lookback, fewer steps) or a shared session fixture, and runs in under 10 s on CPU (report before/after). (2) At least one test still trains the full default config, marked slow. (3) The fast suite with `-n 8` runs in under 2 minutes on an idle machine (report). (4) Fast suite, ruff, TESTING_DOCUMENTATION.
- **source:** D-048

## Done log

Items closed at earlier milestone reviews: the remediation plan's phases 0, A (M1, M2, M4), B and C, and
the notebook review rounds (see [STATUS.md](STATUS.md) and `git log`).
