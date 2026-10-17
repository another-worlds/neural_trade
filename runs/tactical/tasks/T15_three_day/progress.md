# T15 progress (rewritten after every job)

wall clock since start: 0.04 h of 72 h; CPU only, below-normal priority, threads <= 6 (BLAS 4, GBM 4); GPU not used.
stop condition: none hit

| # | day | job | result file | wall s | cpu s | free RAM GB | free D: GB | status |
|---|---|---|---|---|---|---|---|---|
| 1 | day1 | build harmonic views K=4,6,8,12 | day1/build_views.json | 0.4 | 0.4 | 44.8 | 38.2 | ok |
| 2 | day1 | harness evaluate T15_h4 | day1/T15_h4/results.json | 6.3 | 6.3 | 44.8 | 38.2 | ok |
| 3 | day1 | harness evaluate T15_h6 | day1/T15_h6/results.json | 7.2 | 7.1 | 44.8 | 38.2 | ok |
| 4 | day1 | harness evaluate T15_h8 | day1/T15_h8/results.json | 7.1 | 7.0 | 44.8 | 38.2 | ok |
| 5 | day1 | harness evaluate T15_h12 | day1/T15_h12/results.json | 8.2 | 9.7 | 44.8 | 38.2 | ok |
| 6 | day1 | analyse: gains, sub-samples, windows, stability rule | day1/results_day1.json, day1/tables_day1.txt | 0.0 | 0.0 | 44.8 | 38.2 | FAILED: IndexError: list index out of range |
| 7 | day1 | harness evaluate T15_h4 | day1/T15_h4/results.json | 0.0 | 0.0 | 44.9 | 38.2 | ok |
| 8 | day1 | harness evaluate T15_h6 | day1/T15_h6/results.json | 0.0 | 0.0 | 44.9 | 38.2 | ok |
| 9 | day1 | harness evaluate T15_h8 | day1/T15_h8/results.json | 0.0 | 0.0 | 44.9 | 38.2 | ok |
| 10 | day1 | harness evaluate T15_h12 | day1/T15_h12/results.json | 0.0 | 0.0 | 44.9 | 38.2 | ok |
| 11 | day1 | analyse: gains, sub-samples, windows, stability rule | day1/results_day1.json, day1/tables_day1.txt | 0.0 | 0.0 | 44.9 | 38.2 | ok |

total job wall 29 s, total job CPU (process time, all threads) 30 s
