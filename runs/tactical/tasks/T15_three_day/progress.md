# T15 progress (rewritten after every job)

wall clock since start: 0.26 h of 72 h; CPU only, below-normal priority, threads <= 6 (BLAS 4, GBM 4); GPU not used.
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
| 12 | day2 | reproduction checks (3 modules + concatenated fwd pool) | day2/raw/repro.json | 29.3 | 57.0 | 44.8 | 38.2 | ok |
| 13 | day2 | M9h6 hour shuffle x10 | day2/raw/M9h6__CH.json | 67.2 | 66.6 | 44.7 | 38.2 | ok |
| 14 | day2 | M9h6 weekend shuffle x10 | day2/raw/M9h6__CW.json | 67.7 | 66.7 | 44.7 | 38.2 | ok |
| 15 | day2 | M9h6 pool sensitivity | day2/raw/M9h6__CP.json | 8.6 | 8.5 | 44.8 | 38.2 | ok |
| 16 | day2 | M9h6 start-date sensitivity | day2/raw/M9h6__CS.json | 8.9 | 8.8 | 44.7 | 38.2 | ok |
| 17 | day2 | M7c hour shuffle x10 | day2/raw/M7c__CH.json | 92.3 | 228.8 | 44.8 | 38.2 | ok |
| 18 | day2 | M7c weekend shuffle x10 | day2/raw/M7c__CW.json | 92.4 | 229.4 | 44.7 | 38.2 | ok |
| 19 | day2 | M7c pool sensitivity | day2/raw/M7c__CP.json | 10.5 | 36.3 | 44.7 | 38.2 | ok |
| 20 | day2 | M7c start-date sensitivity | day2/raw/M7c__CS.json | 6.2 | 16.0 | 44.7 | 38.2 | ok |
| 21 | day2 | M7_gbm hour shuffle x10 | day2/raw/M7_gbm__CH.json | 98.7 | 245.7 | 44.7 | 38.2 | ok |
| 22 | day2 | M7_gbm weekend shuffle x10 | day2/raw/M7_gbm__CW.json | 99.2 | 247.4 | 44.7 | 38.2 | ok |
| 23 | day2 | M7_gbm pool sensitivity | day2/raw/M7_gbm__CP.json | 11.4 | 38.8 | 44.7 | 38.2 | ok |
| 24 | day2 | M7_gbm start-date sensitivity | day2/raw/M7_gbm__CS.json | 6.4 | 16.1 | 44.7 | 38.2 | ok |
| 25 | day2 | analyse: survival rules | day2/results_day2.json, day2/tables_day2.txt | 0.1 | 0.0 | 44.7 | 38.2 | ok |

total job wall 628 s, total job CPU (process time, all threads) 1297 s
