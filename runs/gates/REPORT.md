# Phase A gate runs (M1-M4)

Produced by `scripts/gate_run.py` from a pinned worktree on the RTX 4070 Ti (2026-09-23) and judged by
`scripts/check_gates.py`, clause by clause as the plan words them. Heavy artefacts (weights, prediction
`.npz`) stay local; the per-epoch logs, analytics and backtest summaries are committed.

| Run | Commit | Settings |
|---|---|---|
| m1a | 89593e0 | 2 epochs, the six physics lambdas at 0 |
| m1b | 89593e0 | 2 epochs, defaults |
| m2 | 89593e0 | 5 epochs, defaults |
| m3 | 89593e0 | 20 epochs requested, defaults - first M3/M4 attempt, kept as history (`@m3`) |
| m4 | 225c674 | 20 epochs, defaults, after the normalised-conformal fix - judged for M3 and M4 |

"Total Trades" comes from `scripts/backtest_gate.py` (the C5 backtest engine on the saved test
predictions, every registered strategy, default costs); see each run's `backtest.json`.

```
gate runs under: C:\Users\Step\Documents\neural_trade\runs\gates

M1a  (m1a, 2 epochs)  plan A1: 'train_and_evaluate(force=True, epochs=2) twice - (a) physics lambdas overridden to 0, (b) defaults'
  PASS     log_val_loss changes between epochs  -- val_loss=[6.688, 6.79268]
  PASS     log_nonfinite_grad_steps == 0  -- max=0.0
  PASS     18 learned periods finite  -- 18 period columns
  PASS     >= 12 periods change between epoch 0 and 1 (S4)  -- 18 moved
  PASS     log_extended_h0 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0241, 0.0239] val=[0.0096, 0.0096]
  PASS     log_extended_h1 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0328, 0.0326] val=[0.0143, 0.0143]
  PASS     log_extended_h2 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0408, 0.0403] val=[0.0184, 0.018]
  PASS     log_train_pred_up_rate_h1 not in {0,1}  -- [0.5982, 0.5455]
  FAIL     log_val_dir_mcc_h1 != 0.0 exactly (final epoch)  -- per epoch [0.0, 0.0]

M1b  (m1b, 2 epochs)  plan A1: 'train_and_evaluate(force=True, epochs=2) twice - (a) physics lambdas overridden to 0, (b) defaults'
  PASS     log_val_loss changes between epochs  -- val_loss=[6.77907, 6.52426]
  PASS     log_nonfinite_grad_steps == 0  -- max=0.0
  PASS     18 learned periods finite  -- 18 period columns
  PASS     >= 12 periods change between epoch 0 and 1 (S4)  -- 18 moved
  PASS     log_extended_h0 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0241, 0.0239] val=[0.0097, 0.0096]
  PASS     log_extended_h1 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0329, 0.0327] val=[0.0143, 0.0144]
  PASS     log_extended_h2 in (0,3), != 1.333295, train != val (S2)  -- train=[0.0406, 0.0403] val=[0.0183, 0.0184]
  PASS     log_train_pred_up_rate_h1 not in {0,1}  -- [0.5249, 0.4262]
  PASS     log_val_dir_mcc_h1 != 0.0 exactly (final epoch)  -- per epoch [0.0, 0.0565]
  PASS     (b) log_casimir_loss > 0 (price heads non-zero)  -- [0.000519, 0.00032]

M2  (m2, 5 epochs)  plan A2: 'M2 stop/go (5 epochs)'
  PASS     run is 5 epochs  -- 5 epochs
  PASS     physics columns finite
  PASS     physics columns >= 0
  PASS     physics columns non-constant
  PASS     log_val_vac_overflow_loss == 0
  PASS     val_nll_loss improves over epoch 0 (last < first)  -- [3.0113, 2.8376, 2.8829, 2.8657, 2.8072]
  PASS     val_crps_loss improves over epoch 0 (last < first)  -- [1.0187, 0.9905, 0.995, 0.9934, 0.9864]
  PASS     log_val_pit_ks_h1 < 0.2 (last epoch)  -- [0.158, 0.1358, 0.1477, 0.1385, 0.1537]
  PASS     corr(var_scaled, error_scaled^2) h1 > 0.10  -- 0.2018  (HEAD 0.032)
  PASS     std(var_h1)/mean(var_h1) > 0.05  -- 0.4331

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

M3  (m4, 20 epochs run)  plan A3: 'M3 stop/go (20 epochs)'
  PASS     run requested 20 epochs  -- 20 requested
  PASS     EV(delta) h1 > 0  -- 0.0055  (HEAD 0.0000)
  FAIL     ROC-AUC h1 > 0.52 (all test rows)  -- 0.4710 on 7236 rows; masked 0.5018
  FAIL     Total Trades > 0  -- 0
  PASS     best-epoch log_val_dir_mcc_h1 > 0.02  -- 0.0237 at epoch 17
  PASS     best-epoch log_val_gauss_dir_mcc_h1 > 0  -- 0.0625 at epoch 17

M4  (m4)  plan A4: 'test coverage at alpha=0.1 in [0.87, 0.93] all horizons; early stopping fires on a plateaued run; bare pytest green'
  PASS     test coverage@90 h0 in [0.87, 0.93]  -- 0.9033
  PASS     test coverage@90 h1 in [0.87, 0.93]  -- 0.9057
  PASS     test coverage@90 h2 in [0.87, 0.93]  -- 0.9125
  INFO     early stopping fires on a plateaued run  -- this run: 20/20 epochs, early_stopped=False; the plateau case is pinned by tests/test_train_smoke.py::test_early_stopping_fires_on_a_plateau
  INFO     bare pytest green  -- judged by the pytest run, not by this script

==============================================================================
  M1a   FAIL       8 pass, 1 fail, 0 pending/not run
  M1b   PASS       10 pass, 0 fail, 0 pending/not run
  M2    PASS       10 pass, 0 fail, 0 pending/not run
  M3@m3 FAIL       2 pass, 4 fail, 0 pending/not run
  M4@m3 FAIL       0 pass, 3 fail, 0 pending/not run
  M3    FAIL       4 pass, 2 fail, 0 pending/not run
  M4    PASS       3 pass, 0 fail, 0 pending/not run
```

## Reading

* **M1b, M2 and M4 pass.** Gradients stay finite, the variance heads are calibrated (PIT-KS 0.15, variance-error
  correlation 0.20) and, with conformal scores normalised by each window's realised volatility, test coverage at a
  90% target is 0.903 / 0.906 / 0.913 (plain split conformal gave 0.966-0.975 because volatility fell ~35% from
  the calibration block to the test block). The scale was chosen on the calibration block alone.
* **M1a** fails one clause: with the physics terms off, the h1 direction head's validation MCC is exactly 0 after
  two epochs (it has not left a constant prediction yet).
* **M3 fails.** EV(delta) h1 is positive (0.0055) and the best-epoch MCC clauses pass, but test AUC h1 is 0.47 over
  all rows (0.50 with the deadband mask) and no strategy places a trade: the direction heads stay within
  about 0.49-0.51. A logistic regression on trailing returns from the same 60-bar window reaches out-of-sample
  AUC 0.524 / 0.523 / 0.532, so the window does carry direction signal that the network is not extracting.
  Per the plan, this is a representation / optimisation problem, not a physics-lambda one - see
  `runs/experiments/direction_v1/`.
