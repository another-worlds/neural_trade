# Phase A gate runs (M1-M4)

Produced by `scripts/gate_run.py` from a pinned worktree on the RTX 4070 Ti (2026-09-23/24) and judged by
`scripts/check_gates.py`, clause by clause as the plan words them. Heavy artefacts (weights, prediction
`.npz`) stay local; the per-epoch logs, analytics and backtest summaries are committed.

| Run | Commit | Settings |
|---|---|---|
| m1a | 6dec27a | 2 epochs, the six physics lambdas at 0 |
| m1b | 6dec27a | 2 epochs, defaults |
| m2 | 6dec27a | 5 epochs, defaults |
| m3 | 89593e0 | 20 epochs, first M3/M4 attempt (history, `@m3`) |
| m4 | 225c674 | + normalised conformal intervals (history, `@m4`) |
| m5 | d750892 | + BCE direction loss and direction skip (history, `@m5`) |
| m6 | 6dec27a | + delta shrinkage - the final code, judged for M3 and M4 |

"Total Trades" comes from `scripts/backtest_gate.py` (the backtest engine on the saved test predictions,
every registered strategy, default costs); see each run's `backtest.json`. Every run uses fold -1 (the last
walk-forward fold), which no modelling choice was tuned on (see `runs/experiments/direction_v1/REPORT.md`).

```
gate runs under: C:\Users\Step\Documents\neural_trade\runs\gates

M1a  (m1a, 2 epochs)  plan A1: 'train_and_evaluate(force=True, epochs=2) twice - (a) physics lambdas overridden to 0, (b) defaults'
  PASS     log_val_loss changes between epochs  -- val_loss=[7.25654, 6.82295]
  PASS     log_nonfinite_grad_steps == 0  -- max=0.0
  PASS     18 learned periods finite  -- 18 period columns
  PASS     >= 12 periods change between epoch 0 and 1 (S4)  -- 18 moved
  PASS     log_extended_h0 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0244, 0.0241] val=[0.0093, 0.0098]
  PASS     log_extended_h1 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0331, 0.0333] val=[0.0142, 0.0137]
  PASS     log_extended_h2 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0399, 0.0395] val=[0.0188, 0.0166]
  PASS     log_train_pred_up_rate_h1 not in {0,1}  -- [0.4896, 0.4869]
  PASS     log_val_dir_mcc_h1 != 0.0 exactly (final epoch)  -- per epoch [-0.0456, 0.0423]

M1b  (m1b, 2 epochs)  plan A1: 'train_and_evaluate(force=True, epochs=2) twice - (a) physics lambdas overridden to 0, (b) defaults'
  PASS     log_val_loss changes between epochs  -- val_loss=[6.65557, 6.61622]
  PASS     log_nonfinite_grad_steps == 0  -- max=0.0
  PASS     18 learned periods finite  -- 18 period columns
  PASS     >= 12 periods change between epoch 0 and 1 (S4)  -- 18 moved
  PASS     log_extended_h0 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0246, 0.0241] val=[0.0098, 0.0097]
  PASS     log_extended_h1 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0327, 0.0321] val=[0.014, 0.0134]
  PASS     log_extended_h2 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0395, 0.0392] val=[0.0191, 0.0169]
  PASS     log_train_pred_up_rate_h1 not in {0,1}  -- [0.4959, 0.4899]
  PASS     log_val_dir_mcc_h1 != 0.0 exactly (final epoch)  -- per epoch [-0.0011, 0.0517]
  PASS     (b) log_casimir_loss > 0 (price heads non-zero)  -- [0.000278, 0.000154]

M2  (m2, 5 epochs)  plan A2: 'M2 stop/go (5 epochs)'
  PASS     run is 5 epochs  -- 5 epochs
  PASS     physics columns finite
  PASS     physics columns >= 0
  PASS     physics columns non-constant
  PASS     log_val_vac_overflow_loss == 0
  PASS     val_nll_loss improves over epoch 0 (last < first)  -- [3.018, 2.8287, 2.7863, 2.711, 2.6757]
  PASS     val_crps_loss improves over epoch 0 (last < first)  -- [1.0354, 1.0021, 1.0004, 0.9945, 0.9878]
  PASS     log_val_pit_ks_h1 < 0.2 (last epoch)  -- [0.1392, 0.1786, 0.153, 0.1615, 0.1073]
  PASS     corr(var_scaled, error_scaled^2) h1 > 0.10  -- 0.2381  (HEAD 0.032)
  PASS     std(var_h1)/mean(var_h1) > 0.05  -- 0.4319

M3@m3  (m3, 15 epochs run)  plan A3: 'M3 stop/go (20 epochs)'
  PASS     run requested 20 epochs  -- 20 requested
  FAIL     EV(delta) h1 > 0  -- -0.0125  (HEAD 0.0000)
  FAIL     ROC-AUC h1 > 0.52 (all test rows)  -- 0.5116 on 7236 rows; masked 0.4948
  FAIL     Total Trades > 0  -- 0
  FAIL     best-epoch log_val_dir_mcc_h1 > 0.02  -- -0.0752 at epoch 8
  PASS     best-epoch log_val_gauss_dir_mcc_h1 > 0  -- 0.0249 at epoch 8

M4@m3  (m3)  plan A4: 'test coverage at alpha=0.1 in [0.87, 0.93] all horizons; early stopping fires on a plateaued run; bare pytest green'
  FAIL     test coverage@90 h0 in [0.87, 0.93]  -- 0.9655
  FAIL     test coverage@90 h1 in [0.87, 0.93]  -- 0.9688
  FAIL     test coverage@90 h2 in [0.87, 0.93]  -- 0.9746
  INFO     early stopping fires on a plateaued run  -- this run: 15/20 epochs, early_stopped=True; the plateau case is pinned by tests/test_train_smoke.py::test_early_stopping_fires_on_a_plateau
  INFO     bare pytest green  -- judged by the pytest run, not by this script

M3@m4  (m4, 20 epochs run)  plan A3: 'M3 stop/go (20 epochs)'
  PASS     run requested 20 epochs  -- 20 requested
  PASS     EV(delta) h1 > 0  -- 0.0055  (HEAD 0.0000)
  FAIL     ROC-AUC h1 > 0.52 (all test rows)  -- 0.4710 on 7236 rows; masked 0.5018
  FAIL     Total Trades > 0  -- 0
  PASS     best-epoch log_val_dir_mcc_h1 > 0.02  -- 0.0237 at epoch 17
  PASS     best-epoch log_val_gauss_dir_mcc_h1 > 0  -- 0.0625 at epoch 17

M4@m4  (m4)  plan A4: 'test coverage at alpha=0.1 in [0.87, 0.93] all horizons; early stopping fires on a plateaued run; bare pytest green'
  PASS     test coverage@90 h0 in [0.87, 0.93]  -- 0.9033
  PASS     test coverage@90 h1 in [0.87, 0.93]  -- 0.9057
  PASS     test coverage@90 h2 in [0.87, 0.93]  -- 0.9125
  INFO     early stopping fires on a plateaued run  -- this run: 20/20 epochs, early_stopped=False; the plateau case is pinned by tests/test_train_smoke.py::test_early_stopping_fires_on_a_plateau
  INFO     bare pytest green  -- judged by the pytest run, not by this script

M3@m5  (m5, 20 epochs run)  plan A3: 'M3 stop/go (20 epochs)'
  PASS     run requested 20 epochs  -- 20 requested
  FAIL     EV(delta) h1 > 0  -- -0.8454  (HEAD 0.0000)
  PASS     ROC-AUC h1 > 0.52 (all test rows)  -- 0.5364 on 7236 rows; masked 0.5509
  FAIL     Total Trades > 0  -- 0
  PASS     best-epoch log_val_dir_mcc_h1 > 0.02  -- 0.1251 at epoch 16
  FAIL     best-epoch log_val_gauss_dir_mcc_h1 > 0  -- -0.1152 at epoch 16

M4@m5  (m5)  plan A4: 'test coverage at alpha=0.1 in [0.87, 0.93] all horizons; early stopping fires on a plateaued run; bare pytest green'
  PASS     test coverage@90 h0 in [0.87, 0.93]  -- 0.9020
  PASS     test coverage@90 h1 in [0.87, 0.93]  -- 0.8930
  PASS     test coverage@90 h2 in [0.87, 0.93]  -- 0.8923
  INFO     early stopping fires on a plateaued run  -- this run: 20/20 epochs, early_stopped=False; the plateau case is pinned by tests/test_train_smoke.py::test_early_stopping_fires_on_a_plateau
  INFO     bare pytest green  -- judged by the pytest run, not by this script

M3  (m6, 20 epochs run)  plan A3: 'M3 stop/go (20 epochs)'
  PASS     run requested 20 epochs  -- 20 requested
  FAIL     EV(delta) h1 > 0 (served delta)  -- 0.0000 (beta 0.0000); raw head -0.6555  (HEAD 0.0000)
  FAIL     ROC-AUC h1 > 0.52 (all test rows)  -- 0.5015 on 7236 rows; masked 0.5239
  PASS     Total Trades > 0  -- 120
  FAIL     best-epoch log_val_dir_mcc_h1 > 0.02  -- 0.0175 at epoch 14
  FAIL     best-epoch log_val_gauss_dir_mcc_h1 > 0  -- -0.0219 at epoch 14

M4  (m6)  plan A4: 'test coverage at alpha=0.1 in [0.87, 0.93] all horizons; early stopping fires on a plateaued run; bare pytest green'
  PASS     test coverage@90 h0 in [0.87, 0.93]  -- 0.9023
  PASS     test coverage@90 h1 in [0.87, 0.93]  -- 0.9059
  PASS     test coverage@90 h2 in [0.87, 0.93]  -- 0.9124
  INFO     early stopping fires on a plateaued run  -- this run: 20/20 epochs, early_stopped=False; the plateau case is pinned by tests/test_train_smoke.py::test_early_stopping_fires_on_a_plateau
  INFO     bare pytest green  -- judged by the pytest run, not by this script

==============================================================================
  M1a   PASS       9 pass, 0 fail, 0 pending/not run
  M1b   PASS       10 pass, 0 fail, 0 pending/not run
  M2    PASS       10 pass, 0 fail, 0 pending/not run
  M3@m3 FAIL       2 pass, 4 fail, 0 pending/not run
  M4@m3 FAIL       0 pass, 3 fail, 0 pending/not run
  M3@m4 FAIL       4 pass, 2 fail, 0 pending/not run
  M4@m4 PASS       3 pass, 0 fail, 0 pending/not run
  M3@m5 FAIL       3 pass, 3 fail, 0 pending/not run
  M4@m5 PASS       3 pass, 0 fail, 0 pending/not run
  M3    FAIL       2 pass, 4 fail, 0 pending/not run
  M4    PASS       3 pass, 0 fail, 0 pending/not run
```

## Reading

* **M1a, M1b, M2 and M4 pass on the final code.** Gradients stay finite, the learned periods move, the
  variance heads are calibrated (PIT-KS 0.11, variance / squared-error correlation 0.24 at 5 epochs, 0.40 at
  20) and the conformal intervals, normalised by each window's realised volatility, cover 0.90-0.91 of the
  test block at a 0.90 target. M1a's direction clause now passes because the direction loss is a proper
  scoring rule (binary cross-entropy) rather than focal + dice, whose optimum is a constant extreme.
* **M3 fails.** Every one of its clauses passed in at least one attempt (AUC h1 0.536 in m5; EV > 0 in m4;
  trades in m6), never all together:
  * *Direction:* test AUC h1 0.50 (0.52 with the deadband mask) in m6 against 0.54 (0.55) in m5 with the same
    code apart from delta shrinkage; h2 0.55 in m6. The signal is weak (a logistic regression on trailing
    returns reaches about 0.56 on this fold) and identical GPU runs differ by 0.01-0.05 AUC, so single runs
    straddle the 0.52 line.
  * *Price heads:* the raw heads overfit in the batch-64, 20-epoch recipe (EV h1 -0.66 in m6, -0.85 in m5).
    Delta shrinkage, fit on the calibration block, set beta = 0 and serves a zero delta, so the served EV is
    0.000 - harmless, but not the "> 0" the clause asks for.
  * *Trading:* with calibrated heads the notebook strategies' fixed 0.55 / 0.45 lines are almost never
    crossed. The calibration-quantile strategy placed 120 trades on the test block: gross +4.1% (buy-and-hold
    +4.1%), net -23.4% after 26 bps per round trip, at the 95th percentile of random strategies with the same
    trade frequency. The edge per trade is far smaller than its cost at 10-20 minute horizons.
