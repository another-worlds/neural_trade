# Leaderboard: `reference_default`

One row per configuration. **Ranking column: dev-fold net Sharpe after costs** (mean over the dev folds of each fold's seed mean, D-020, D-046). Guard-rails beside it can disqualify a row from the winner (VISION "The yardstick"). The **test-fold columns are test, not used for ranking** (D-020): shown for every row, never used to rank or choose.

Cost profile: the board ranks at 0 bps/side (D-044), the scenario's profile. Rows on this board use a different cost profile: 13 bps/side (fee 10 + half-spread 1 + slippage 2) (1 row). They are marked not comparable (guard-rail cost_profile) and cannot be the winner; they keep their place in the dev net Sharpe order.

Guard-rails: max drawdown not checked; trades >= 1 on the mean and on every dev fold; beat buy-and-hold; random-null percentile >= 50 [thresholds: reference.yaml: defaults]; a scored cell on every dev fold of the scenario; the board's cost profile.

| rank | configuration | status | ranking: dev net Sharpe (spread, counts) | cost profile of the stored net Sharpe | dev net return | dev max drawdown | dev trades | dev buy & hold | dev random-null percentile | guard-rails | test net Sharpe (test, not used for ranking) | test net return (test, not used for ranking) | test max drawdown (test, not used for ranking) | test trades (test, not used for ranking) | dataset fingerprint (sha256, first 12) | bar (min) | horizons (bars) | strategy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `default` | done | -155.975 (fold sd 26.495, seed sd 23.821, 2 folds x 3 seeds, 6 cells) | 13 bps/side (fee 10 + half-spread 1 + slippage 2) (not comparable) | -62.37% (fold sd 6.52%, seed sd 7.06%) | 62.37% (fold sd 6.51%, seed sd 7.06%) | 392.8 (fold sd 92.6, seed sd 77.5) | -4.58% (fold sd 4.09%, seed sd 0.00%) | 64 (fold sd 29, seed sd 22) | DISQUALIFIED: min_trades OK (392.8); beat_buy_and_hold FAIL (-62.37% vs -4.58%); beat_random_null OK (64); fold_coverage OK (2 of 2 dev folds); cost_profile FAIL (not comparable: 13 bps/side (fee 10 + half-spread 1 + slippage 2); the board ranks at 0 bps/side (D-044)) | -115.274 (seed sd 14.794, 1 fold x 3 seeds, 3 cells) | -42.87% (seed sd 5.05%) | 42.93% (seed sd 5.11%) | 224.3 (seed sd 43.7) | a57ee1e43981 | 1 | 10, 15, 20 | calibrated_quantile |

**Winner:** none (every row disqualified or not comparable, or no scored dev data).
