# Leaderboard: `micro_horizons`

One row per configuration. **Ranking column: dev-fold net Sharpe after costs** (mean over the dev folds and their seeds, D-020, D-046). Guard-rails beside it can disqualify a row from the winner (VISION "The yardstick"). The **test-fold columns are test, not used for ranking** (D-020): shown for every row, never used to rank or choose.

| rank | configuration | status | ranking: dev net Sharpe (spread, counts) | dev net return | dev max drawdown | dev trades | dev buy & hold | dev random-null percentile | guard-rails | test net Sharpe (test, not used for ranking) | test net return (test, not used for ranking) | test max drawdown (test, not used for ranking) | test trades (test, not used for ranking) | dataset fingerprint | bar (min) | horizons (bars) | strategy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `h_4h` | done | -127.252 (1 fold x 1 seed, 1 cell) | -96.70% | 96.70% | 1288.0 | -8.03% | 27 | DISQUALIFIED: min_trades OK (1288.0); beat_buy_and_hold FAIL (-96.70% vs -8.03%); beat_random_null FAIL (27) | n/a | n/a | n/a | n/a | 67966a49634f... | 1 | 160, 240, 320 | calibrated_quantile |
| 2 | `h_1h` | done | -145.108 (1 fold x 1 seed, 1 cell) | -98.28% | 98.28% | 1550.0 | -6.74% | 14 | DISQUALIFIED: min_trades OK (1550.0); beat_buy_and_hold FAIL (-98.28% vs -6.74%); beat_random_null FAIL (14) | n/a | n/a | n/a | n/a | 67966a49634f... | 1 | 40, 60, 80 | calibrated_quantile |
| 3 | `h_15m` | done | -181.507 (1 fold x 1 seed, 1 cell) | -99.65% | 99.65% | 2166.0 | -6.39% | 88 | DISQUALIFIED: min_trades OK (2166.0); beat_buy_and_hold FAIL (-99.65% vs -6.39%); beat_random_null OK (88) | n/a | n/a | n/a | n/a | 67966a49634f... | 1 | 10, 15, 20 | calibrated_quantile |

**Winner:** none (every row disqualified or no scored dev data).
