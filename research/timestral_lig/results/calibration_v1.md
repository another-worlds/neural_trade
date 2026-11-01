
#### repo, P-A (calibrated before Test, evaluated on the full Test); rule W: max Wilson lower bound

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=5, agree3, pm top 30%, follow (calib acc 54.4%, n 1531) | 48.0% [42.1%, 53.9%] | 0.542 | 269 (44.6/day) | -18 (t -0.08) | -1094 | -2708 |
| A0 zero-shot (V1+V2 calib) | H=5, none, pm top 30%, follow (calib acc 58.8%, n 483) | 46.1% [38.9%, 53.4%] | 0.330 | 178 (29.5/day) | -111 (t -0.65) | -823 | -1891 |
| A1 | H=5, none, pm top 30%, follow (calib acc 57.0%, n 421) | 47.7% [39.4%, 56.2%] | 0.664 | 132 (21.9/day) | -135 (t -0.89) | -663 | -1455 |
| A2 | H=1, none, z top 100%, follow (calib acc 51.6%, n 4761) | 49.3% [47.3%, 51.2%] | 0.475 | 2545 (424.3/day) | -602 (t -1.62) | -10834 | -26182 |
| price-only baseline | H=1, none, z top 2%, fade (calib acc 60.0%, n 360) | 54.5% [40.1%, 68.3%] | 0.652 | 44 (7.3/day) | -8 (t -0.14) | -184 | -448 |

#### repo, P-A (calibrated before Test, evaluated on the full Test); rule R: max raw accuracy, n >= 100

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=1, path15, pm top 1%, fade (calib acc 59.1%, n 110) | 61.5% [35.5%, 82.3%] | 0.581 | 13 (2.2/day) | +19 (t 1.08) | -33 | -111 |
| A0 zero-shot (V1+V2 calib) | H=5, none, pm top 30%, follow (calib acc 58.8%, n 483) | 46.1% [38.9%, 53.4%] | 0.330 | 178 (29.5/day) | -111 (t -0.65) | -823 | -1891 |
| A1 | H=5, none, pm top 30%, follow (calib acc 57.0%, n 421) | 47.7% [39.4%, 56.2%] | 0.664 | 132 (21.9/day) | -135 (t -0.89) | -663 | -1455 |
| A2 | H=1, none, pm top 20%, follow (calib acc 52.6%, n 946) | 47.9% [41.4%, 54.5%] | 0.587 | 217 (36.3/day) | -18 (t -0.19) | -894 | -2208 |
| price-only baseline | H=15, none, z top 2%, fade (calib acc 60.8%, n 166) | 67.5% [52.0%, 79.9%] | 0.038 | 40 (6.6/day) | +248 (t 1.35) | +88 | -152 |

#### repo, P-A (calibrated before Test, evaluated on the full Test); descriptive only: best rule picked on the evaluation data itself

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=1, none, z top 5%, fade (calib acc 57.0%, n 142) | 57.0% [48.8%, 64.9%] | 0.111 | 142 (23.6/day) | +71 (t 0.93) | -497 | -1349 |
| A0 zero-shot (V1+V2 calib) | H=1, none, z top 5%, fade (calib acc 57.0%, n 142) | 57.0% [48.8%, 64.9%] | 0.111 | 142 (23.6/day) | +71 (t 0.93) | -497 | -1349 |
| A1 | H=1, none, z top 5%, fade (calib acc 58.5%, n 135) | 58.5% [50.1%, 66.5%] | 0.058 | 135 (22.4/day) | +29 (t 0.49) | -511 | -1321 |
| A2 | H=1, none, z top 5%, fade (calib acc 58.6%, n 128) | 58.6% [49.9%, 66.8%] | 0.063 | 128 (21.2/day) | +55 (t 0.75) | -457 | -1225 |
| price-only baseline | H=5, agree3, z top 10%, fade (calib acc 58.6%, n 116) | 58.6% [49.5%, 67.2%] | 0.077 | 116 (19.2/day) | +208 (t 1.37) | -256 | -952 |

#### repo, P-B (calibrated on the first half of Test, evaluated on the second half); rule W: max Wilson lower bound

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=15, path15, pm top 20%, follow (calib acc 62.7%, n 67) | 45.6% [37.7%, 53.6%] | 0.322 | 147 (48.8/day) | -337 (t -1.38) | -925 | -1807 |
| A1 | H=1, none, z top 10%, follow (calib acc 56.9%, n 137) | 46.0% [40.0%, 52.0%] | 0.216 | 261 (87.2/day) | -27 (t -0.33) | -1079 | -2657 |
| A2 | H=1, none, z top 30%, follow (calib acc 53.8%, n 392) | 46.6% [42.0%, 51.2%] | 0.159 | 453 (152.3/day) | -142 (t -1.14) | -1978 | -4732 |
| price-only baseline | H=5, agree3, z top 10%, fade (calib acc 65.5%, n 58) | 50.0% [38.4%, 61.6%] | 1.000 | 68 (22.6/day) | +46 (t 0.44) | -226 | -634 |
| neural_trade asis seed0 | H=1, none, z top 30%, fade (calib acc 55.1%, n 432) | 52.6% [48.4%, 56.7%] | 0.248 | 546 (184.4/day) | +60 (t 0.46) | -2164 | -5500 |
| neural_trade asis seed1 | H=5, none, dp top 5%, follow (calib acc 65.6%, n 61) | 44.4% [32.0%, 57.6%] | 0.497 | 54 (17.9/day) | -319 (t -1.92) | -535 | -859 |
| neural_trade asis seed2 | H=5, agree3, z top 100%, follow (calib acc 59.1%, n 225) | 49.1% [42.7%, 55.6%] | 0.843 | 228 (75.6/day) | +105 (t 0.54) | -807 | -2175 |
| neural_trade rolling1440m seed0 | H=15, agree3, dp top 30%, fade (calib acc 65.6%, n 64) | 66.7% [39.1%, 86.2%] | 0.388 | 12 (4.0/day) | +87 (t 1.5) | +39 | -33 |
| neural_trade rolling1440m seed1 | H=15, none, dp top 30%, fade (calib acc 60.7%, n 150) | 46.0% [38.0%, 54.3%] | 0.396 | 139 (46.1/day) | -316 (t -1.23) | -872 | -1706 |
| neural_trade rolling1440m seed2 | H=15, agree3, dp top 20%, follow (calib acc 70.3%, n 37) | 40.0% [11.8%, 76.9%] | 1.000 | 5 (1.7/day) | -37 (t -0.83) | -57 | -87 |

#### repo, P-B (calibrated on the first half of Test, evaluated on the second half); rule R: max raw accuracy, n >= 100

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=15, agree3, z top 30%, follow (calib acc 58.2%, n 110) | 43.0% [36.0%, 50.3%] | 0.072 | 179 (59.4/day) | -611 (t -2.02) | -1327 | -2401 |
| A1 | H=1, none, z top 10%, follow (calib acc 56.9%, n 137) | 46.0% [40.0%, 52.0%] | 0.216 | 261 (87.2/day) | -27 (t -0.33) | -1079 | -2657 |
| A2 | H=1, none, z top 10%, follow (calib acc 55.0%, n 131) | 44.1% [37.8%, 50.7%] | 0.093 | 222 (74.3/day) | -69 (t -0.88) | -965 | -2309 |
| price-only baseline | H=5, none, z top 10%, fade (calib acc 58.0%, n 119) | 50.4% [41.8%, 59.0%] | 1.000 | 125 (41.5/day) | -35 (t -0.24) | -535 | -1285 |
| neural_trade asis seed0 | H=15, none, dp top 20%, follow (calib acc 58.9%, n 107) | 53.3% [44.9%, 61.5%] | 0.491 | 135 (44.8/day) | +117 (t 0.54) | -423 | -1233 |
| neural_trade asis seed1 | H=5, none, dp top 10%, follow (calib acc 59.5%, n 116) | 41.8% [32.6%, 51.7%] | 0.129 | 98 (32.5/day) | -291 (t -2.11) | -683 | -1271 |
| neural_trade asis seed2 | H=5, agree3, dp top 50%, follow (calib acc 61.4%, n 114) | 50.6% [42.8%, 58.4%] | 0.936 | 154 (51.1/day) | +178 (t 1.16) | -438 | -1362 |
| neural_trade rolling1440m seed0 | H=5, agree3, dp top 30%, fade (calib acc 58.2%, n 103) | 56.4% [48.7%, 63.7%] | 0.119 | 165 (54.7/day) | +232 (t 1.61) | -428 | -1418 |
| neural_trade rolling1440m seed1 | H=15, none, dp top 30%, fade (calib acc 60.7%, n 150) | 46.0% [38.0%, 54.3%] | 0.396 | 139 (46.1/day) | -316 (t -1.23) | -872 | -1706 |
| neural_trade rolling1440m seed2 | H=1, agree3, z top 30%, fade (calib acc 60.7%, n 122) | 46.7% [36.9%, 56.9%] | 0.602 | 92 (30.9/day) | -36 (t -0.76) | -408 | -966 |

#### repo, P-B (calibrated on the first half of Test, evaluated on the second half); descriptive only: best rule picked on the evaluation data itself

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=1, agree3, z top 10%, fade (calib acc 61.4%, n 101) | 61.4% [51.6%, 70.3%] | 0.028 | 101 (33.5/day) | +37 (t 0.86) | -367 | -973 |
| A1 | H=1, none, z top 10%, fade (calib acc 56.7%, n 134) | 56.7% [48.3%, 64.8%] | 0.142 | 134 (44.5/day) | +19 (t 0.37) | -517 | -1321 |
| A2 | H=1, none, z top 10%, fade (calib acc 59.5%, n 126) | 59.5% [50.8%, 67.7%] | 0.040 | 126 (41.8/day) | +52 (t 1.1) | -452 | -1208 |
| price-only baseline | H=1, agree3, z top 20%, follow (calib acc 57.2%, n 131) | 57.2% [48.7%, 65.4%] | 0.116 | 131 (43.8/day) | +91 (t 0.99) | -437 | -1229 |
| neural_trade asis seed0 | H=15, none, z top 50%, follow (calib acc 60.0%, n 205) | 60.0% [53.2%, 66.5%] | 0.005 | 205 (68.0/day) | +844 (t 2.77) | +24 | -1206 |
| neural_trade asis seed1 | H=15, none, z top 20%, follow (calib acc 63.1%, n 103) | 63.1% [53.5%, 71.8%] | 0.010 | 103 (34.2/day) | +713 (t 3.2) | +301 | -317 |
| neural_trade asis seed2 | H=15, none, dp top 30%, fade (calib acc 56.0%, n 159) | 56.0% [48.2%, 63.5%] | 0.153 | 159 (52.7/day) | +286 (t 1.15) | -350 | -1304 |
| neural_trade rolling1440m seed0 | H=5, agree3, dp top 30%, fade (calib acc 58.0%, n 150) | 58.0% [50.0%, 65.6%] | 0.060 | 150 (49.8/day) | +300 (t 2.01) | -300 | -1200 |
| neural_trade rolling1440m seed1 | H=5, agree3, dp top 30%, fade (calib acc 62.7%, n 134) | 62.7% [54.2%, 70.4%] | 0.004 | 134 (44.5/day) | +364 (t 2.86) | -172 | -976 |
| neural_trade rolling1440m seed2 | H=15, none, dp top 30%, follow (calib acc 60.6%, n 137) | 60.6% [52.2%, 68.4%] | 0.016 | 137 (45.4/day) | +259 (t 1.23) | -289 | -1111 |

#### post, P-A (calibrated before Test, evaluated on the full Test); rule W: max Wilson lower bound

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=15, path15, pm top 10%, follow (calib acc 58.4%, n 274) | 56.8% [47.8%, 65.4%] | 0.167 | 118 (8.4/day) | +186 (t 1.08) | -286 | -994 |
| A0 zero-shot (V1+V2 calib) | H=1, agree3, z top 2%, follow (calib acc 65.8%, n 73) | 53.7% [44.9%, 62.2%] | 0.471 | 123 (9.2/day) | +19 (t 0.41) | -497 | -1271 |
| A1 | H=1, none, z top 2%, follow (calib acc 61.4%, n 101) | 52.0% [44.6%, 59.4%] | 0.646 | 171 (12.7/day) | +96 (t 1.44) | -612 | -1674 |
| A2 | H=1, none, pm top 2%, follow (calib acc 62.6%, n 99) | 52.9% [45.5%, 60.2%] | 0.493 | 172 (13.1/day) | -12 (t -0.22) | -744 | -1842 |
| price-only baseline | H=15, none, z top 10%, fade (calib acc 55.5%, n 659) | 56.1% [50.3%, 61.7%] | 0.045 | 287 (20.6/day) | +288 (t 0.88) | -864 | -2592 |

#### post, P-A (calibrated before Test, evaluated on the full Test); rule R: max raw accuracy, n >= 100

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=5, none, z top 1%, follow (calib acc 59.8%, n 112) | 61.1% [38.6%, 79.7%] | 0.481 | 18 (1.3/day) | +32 (t 0.44) | -40 | -148 |
| A0 zero-shot (V1+V2 calib) | H=1, none, z top 2%, follow (calib acc 61.6%, n 112) | 52.2% [44.9%, 59.5%] | 0.600 | 178 (13.2/day) | +99 (t 1.47) | -637 | -1741 |
| A1 | H=1, none, z top 2%, follow (calib acc 61.4%, n 101) | 52.0% [44.6%, 59.4%] | 0.646 | 171 (12.7/day) | +96 (t 1.44) | -612 | -1674 |
| A2 | H=1, none, z top 5%, follow (calib acc 55.7%, n 244) | 51.5% [46.6%, 56.3%] | 0.584 | 404 (30.3/day) | +109 (t 1.25) | -1587 | -4131 |
| price-only baseline | H=15, agree3, z top 5%, fade (calib acc 56.2%, n 217) | 56.2% [46.7%, 65.3%] | 0.241 | 105 (7.5/day) | +253 (t 1.17) | -167 | -797 |

#### post, P-A (calibrated before Test, evaluated on the full Test); descriptive only: best rule picked on the evaluation data itself

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=15, none, z top 5%, follow (calib acc 62.8%, n 121) | 62.8% [53.9%, 70.9%] | 0.006 | 121 (8.7/day) | +284 (t 1.53) | -200 | -926 |
| A0 zero-shot (V1+V2 calib) | H=15, none, z top 5%, follow (calib acc 62.8%, n 121) | 62.8% [53.9%, 70.9%] | 0.006 | 121 (8.7/day) | +284 (t 1.53) | -200 | -926 |
| A1 | H=1, none, z top 2%, follow (calib acc 54.9%, n 111) | 54.9% [45.7%, 63.9%] | 0.343 | 111 (8.3/day) | +58 (t 1.09) | -406 | -1102 |
| A2 | H=1, none, z top 5%, follow (calib acc 51.5%, n 270) | 51.5% [45.5%, 57.4%] | 0.670 | 270 (20.2/day) | +86 (t 1.29) | -1042 | -2734 |
| price-only baseline | H=5, agree3, z top 5%, fade (calib acc 58.4%, n 125) | 58.4% [49.6%, 66.7%] | 0.073 | 125 (9.0/day) | -187 (t -0.98) | -691 | -1447 |

#### post, P-B (calibrated on the first half of Test, evaluated on the second half); rule W: max Wilson lower bound

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=15, none, z top 20%, follow (calib acc 58.8%, n 216) | 48.6% [42.8%, 54.4%] | 0.677 | 282 (40.3/day) | -50 (t -0.15) | -1178 | -2870 |
| A1 | H=1, none, z top 100%, follow (calib acc 50.7%, n 2750) | 49.9% [48.1%, 51.7%] | 0.925 | 2854 (419.7/day) | -320 (t -1.26) | -12060 | -29670 |
| A2 | H=1, none, z top 100%, follow (calib acc 50.6%, n 2682) | 50.2% [48.4%, 52.1%] | 0.820 | 2783 (409.5/day) | -199 (t -0.79) | -11655 | -28839 |
| price-only baseline | H=15, agree3, z top 50%, fade (calib acc 60.1%, n 363) | 52.9% [47.9%, 57.9%] | 0.279 | 376 (53.8/day) | +248 (t 0.69) | -1256 | -3512 |
| neural_trade asis seed0 | H=5, agree3, z top 5%, follow (calib acc 71.4%, n 49) | 66.7% [35.4%, 87.9%] | 0.508 | 9 (1.3/day) | +20 (t 1.27) | -16 | -70 |
| neural_trade asis seed1 | H=5, none, z top 2%, fade (calib acc 62.3%, n 61) | 45.4% [35.2%, 55.8%] | 0.451 | 86 (12.3/day) | +152 (t 1.39) | -192 | -708 |
| neural_trade asis seed2 | H=15, none, z top 2%, fade (calib acc 66.7%, n 51) | 52.9% [39.5%, 66.0%] | 0.780 | 51 (7.3/day) | -94 (t -0.97) | -298 | -604 |
| neural_trade rolling1440m seed0 | H=1, agree3, dp top 30%, fade (calib acc 58.3%, n 396) | 58.8% [42.2%, 73.6%] | 0.392 | 34 (5.3/day) | +40 (t 1.49) | -108 | -330 |
| neural_trade rolling1440m seed1 | H=1, agree3, z top 20%, follow (calib acc 59.1%, n 193) | 52.8% [44.6%, 60.9%] | 0.557 | 142 (21.0/day) | +36 (t 0.72) | -552 | -1434 |
| neural_trade rolling1440m seed2 | H=5, none, dp top 10%, fade (calib acc 57.6%, n 229) | 43.9% [38.0%, 49.9%] | 0.054 | 260 (37.5/day) | -82 (t -0.62) | -1130 | -2702 |

#### post, P-B (calibrated on the first half of Test, evaluated on the second half); rule R: max raw accuracy, n >= 100

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=15, agree3, pm top 20%, follow (calib acc 59.6%, n 146) | 50.2% [43.4%, 57.1%] | 1.000 | 203 (29.0/day) | +62 (t 0.26) | -750 | -1968 |
| A1 | H=1, none, z top 5%, follow (calib acc 53.7%, n 136) | 50.7% [42.8%, 58.6%] | 0.935 | 150 (21.6/day) | +48 (t 0.69) | -556 | -1462 |
| A2 | H=1, none, pm top 5%, follow (calib acc 52.4%, n 126) | 50.3% [42.7%, 57.9%] | 1.000 | 163 (24.5/day) | -16 (t -0.28) | -700 | -1726 |
| price-only baseline | H=15, agree3, z top 50%, fade (calib acc 60.1%, n 363) | 52.9% [47.9%, 57.9%] | 0.279 | 376 (53.8/day) | +248 (t 0.69) | -1256 | -3512 |
| neural_trade asis seed0 | H=15, agree3, z top 20%, fade (calib acc 56.2%, n 160) | 57.5% [49.0%, 65.5%] | 0.100 | 134 (19.2/day) | +19 (t 0.09) | -517 | -1321 |
| neural_trade asis seed1 | H=1, agree3, dp top 20%, follow (calib acc 56.5%, n 161) | 52.9% [43.4%, 62.2%] | 0.624 | 104 (15.6/day) | +42 (t 1.14) | -394 | -1048 |
| neural_trade asis seed2 | H=5, none, z top 5%, follow (calib acc 61.3%, n 137) | 54.7% [43.5%, 65.4%] | 0.489 | 75 (10.7/day) | +20 (t 0.36) | -280 | -730 |
| neural_trade rolling1440m seed0 | H=1, agree3, dp top 20%, fade (calib acc 59.4%, n 266) | 58.8% [36.0%, 78.4%] | 0.629 | 17 (2.4/day) | +25 (t 1.88) | -43 | -145 |
| neural_trade rolling1440m seed1 | H=1, agree3, z top 20%, follow (calib acc 59.1%, n 193) | 52.8% [44.6%, 60.9%] | 0.557 | 142 (21.0/day) | +36 (t 0.72) | -552 | -1434 |
| neural_trade rolling1440m seed2 | H=15, none, dp top 10%, fade (calib acc 59.5%, n 131) | 45.6% [37.6%, 54.0%] | 0.349 | 138 (19.7/day) | -161 (t -1.06) | -713 | -1541 |

#### post, P-B (calibrated on the first half of Test, evaluated on the second half); descriptive only: best rule picked on the evaluation data itself

| Model | Chosen rule | Accuracy [95% CI] | p vs 50% | Trades (per day) | Gross PnL bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|
| A0 zero-shot | H=1, path15, z top 5%, follow (calib acc 55.3%, n 103) | 55.3% [45.7%, 64.6%] | 0.325 | 103 (14.9/day) | +54 (t 1.0) | -362 | -986 |
| A1 | H=1, none, pm top 5%, follow (calib acc 52.4%, n 143) | 52.4% [44.3%, 60.5%] | 0.616 | 143 (21.0/day) | +7 (t 0.09) | -581 | -1463 |
| A2 | H=1, none, z top 5%, follow (calib acc 52.9%, n 140) | 52.9% [44.6%, 60.9%] | 0.554 | 140 (20.6/day) | +55 (t 1.01) | -521 | -1385 |
| price-only baseline | H=15, none, z top 20%, fade (calib acc 55.8%, n 287) | 55.8% [50.0%, 61.4%] | 0.059 | 287 (41.0/day) | +401 (t 1.29) | -747 | -2469 |
| neural_trade asis seed0 | H=15, none, dp top 10%, fade (calib acc 59.0%, n 139) | 59.0% [50.7%, 66.8%] | 0.041 | 139 (19.9/day) | +457 (t 1.82) | -99 | -933 |
| neural_trade asis seed1 | H=1, agree3, dp top 20%, follow (calib acc 58.9%, n 163) | 58.9% [51.2%, 66.2%] | 0.028 | 163 (24.2/day) | +99 (t 1.7) | -577 | -1591 |
| neural_trade asis seed2 | H=5, none, dp top 10%, follow (calib acc 55.8%, n 231) | 55.8% [49.4%, 62.1%] | 0.087 | 231 (33.3/day) | -16 (t -0.12) | -948 | -2346 |
| neural_trade rolling1440m seed0 | H=5, agree3, dp top 20%, follow (calib acc 59.1%, n 154) | 59.1% [51.2%, 66.5%] | 0.029 | 154 (22.0/day) | +282 (t 3.39) | -334 | -1258 |
| neural_trade rolling1440m seed1 | H=5, none, z top 5%, follow (calib acc 61.0%, n 123) | 61.0% [52.1%, 69.1%] | 0.019 | 123 (17.6/day) | +141 (t 1.41) | -351 | -1089 |
| neural_trade rolling1440m seed2 | H=1, agree3, dp top 20%, follow (calib acc 59.6%, n 193) | 59.6% [52.5%, 66.3%] | 0.009 | 193 (27.9/day) | +201 (t 2.68) | -579 | -1749 |
