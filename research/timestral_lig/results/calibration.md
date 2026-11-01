
#### repo, P-A (calibrated on pre-Test forecasts, evaluated on the full Test)

| Model | Rule | Chosen on calibration | Calib acc (p_sel) | Decision | Test accuracy [95% CI] | p one-sided (Holm) | Trades / decided (per day) | Gross bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 zero-shot | W | H=5, agree3, pm top 1%, follow | 60.4% (0.025) | deploy | 45.7% [30.5%, 61.8%] | 0.750 (1.00) | 35 / 35 (5.8) | -16 (t -0.13) | -156 | -366 |
| A0 zero-shot | R | H=5, agree3, pm top 1%, follow | 60.4% (0.025) | deploy | 45.7% [30.5%, 61.8%] | 0.750 (1.00) | 35 / 35 (5.8) | -16 (t -0.13) | -156 | -366 |
| A0 zero-shot (V1+V2 calib) | W | H=5, agree3, z top 10%, follow | 59.0% (0.075) | **abstain** | (if deployed: 52.6% [46.2%, 59.0%]) | – | 228 / 228 | +216 (t 0.86) | -696 | -2064 |
| A0 zero-shot (V1+V2 calib) | R | H=15, none, z top 20%, follow | 59.8% (0.100) | **abstain** | (if deployed: 54.7% [47.8%, 61.4%]) | – | 203 / 203 | +125 (t 0.36) | -687 | -1905 |
| A1 (calib partly in-sample) | W | H=5, none, z top 2%, follow | 66.2% (0.025) | deploy | 49.0% [35.6%, 62.5%] | 0.612 (1.00) | 49 / 49 (8.1) | -54 (t -0.43) | -250 | -544 |
| A1 (calib partly in-sample) | R | H=5, none, pm top 10%, follow | 59.1% (0.125) | **abstain** | (if deployed: 48.4% [41.5%, 55.5%]) | – | 194 / 194 | -141 (t -0.57) | -917 | -2081 |
| A2 (calib partly in-sample) | W | H=1, none, no conf filter, follow | 51.6% (0.125) | **abstain** | (if deployed: 49.3% [47.3%, 51.2%]) | – | 2558 / 2545 | -602 (t -1.62) | -10834 | -26182 |
| A2 (calib partly in-sample) | R | H=1, none, z top 2%, fade | 56.9% (0.225) | **abstain** | (if deployed: 55.0% [44.1%, 65.4%]) | – | 80 / 80 | +12 (t 0.18) | -308 | -788 |
| price-only baseline | W | H=1, agree3, z top 2%, fade | 60.2% (0.025) | deploy | 52.1% [38.3%, 65.5%] | 0.443 (1.00) | 48 / 48 (8.0) | -57 (t -0.88) | -249 | -537 |
| price-only baseline | R | H=1, agree3, z top 2%, fade | 60.2% (0.075) | **abstain** | (if deployed: 52.1% [38.3%, 65.5%]) | – | 48 / 48 | -57 (t -0.88) | -249 | -537 |

Descriptive only, rule picked on the evaluation data itself (p_sel = permutation p of the best of the grid):

| Model | Picked rule | Accuracy | Decided trades | p_sel | Gross bps |
|---|---|---|---|---|---|
| A0 zero-shot | H=15, none, pm top 20%, follow | 56.2% | 203 | 0.725 | +142 |
| A0 zero-shot (V1+V2 calib) | H=15, none, pm top 20%, follow | 56.2% | 203 | 0.725 | +142 |
| A1 (calib partly in-sample) | H=5, none, z top 5%, follow | 53.5% | 101 | 0.925 | +19 |
| A2 (calib partly in-sample) | H=1, none, pm top 10%, fade | 51.5% | 324 | 0.975 | +85 |
| price-only baseline | H=5, agree3, z top 5%, fade | 58.9% | 112 | 0.275 | +189 |

#### repo, P-B (calibrated on the first half of Test, evaluated on the second half)

| Model | Rule | Chosen on calibration | Calib acc (p_sel) | Decision | Test accuracy [95% CI] | p one-sided (Holm) | Trades / decided (per day) | Gross bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 zero-shot | W | H=15, path15, z top 20%, follow | 60.0% (0.875) | **abstain** | (if deployed: 48.4% [38.8%, 58.3%]) | – | 97 / 97 | -159 (t -0.92) | -547 | -1129 |
| A0 zero-shot | R | H=15, none, pm top 20%, follow | 58.0% (0.550) | **abstain** | (if deployed: 55.3% [45.7%, 64.6%]) | – | 103 / 103 | -157 (t -0.9) | -569 | -1187 |
| A1 | W | H=1, none, no conf filter, fade | 51.1% (0.875) | **abstain** | (if deployed: 50.1% [47.4%, 52.8%]) | – | 1332 / 1318 | +266 (t 1.24) | -5062 | -13054 |
| A1 | R | H=5, none, z top 30%, follow | 53.7% (0.650) | **abstain** | (if deployed: 49.6% [43.8%, 55.5%]) | – | 276 / 276 | +1 (t 0.01) | -1103 | -2759 |
| A2 | W | H=1, none, z top 30%, follow | 53.6% (0.525) | **abstain** | (if deployed: 46.6% [42.1%, 51.1%]) | – | 475 / 468 | -172 (t -1.39) | -2072 | -4922 |
| A2 | R | H=1, none, z top 10%, follow | 54.3% (0.450) | **abstain** | (if deployed: 46.7% [39.8%, 53.7%]) | – | 197 / 195 | -12 (t -0.17) | -800 | -1982 |
| price-only baseline | W | H=5, none, z top 2%, fade | 75.0% (0.050) | deploy | 62.1% [44.0%, 77.3%] (too few trades) | 0.132 (0.26) | 29 / 29 (9.6) | +102 (t 1.23) | -14 | -188 |
| price-only baseline | R | H=1, agree3, z top 30%, fade | 57.9% (0.300) | **abstain** | (if deployed: 45.6% [39.5%, 51.8%]) | – | 253 / 250 | -140 (t -1.23) | -1152 | -2670 |
| neural_trade asis (3-seed ensemble) | W | H=15, none, z top 5%, follow | 64.7% (0.650) | **abstain** | (if deployed: 59.3% [46.0%, 71.3%]) | – | 54 / 54 | +336 (t 1.71) | +120 | -204 |
| neural_trade asis (3-seed ensemble) | R | H=15, none, z top 20%, follow | 57.4% (0.500) | **abstain** | (if deployed: 58.3% [49.4%, 66.8%]) | – | 120 / 120 | +417 (t 1.61) | -63 | -783 |
| neural_trade rolling1440m (3-seed ensemble) | W | H=1, none, z top 30%, fade | 55.1% (0.775) | **abstain** | (if deployed: 51.4% [46.7%, 56.1%]) | – | 439 / 432 | -45 (t -0.36) | -1801 | -4435 |
| neural_trade rolling1440m (3-seed ensemble) | R | H=5, agree3, z top 30%, fade | 58.7% (0.475) | **abstain** | (if deployed: 53.4% [46.8%, 59.9%]) | – | 221 / 221 | +44 (t 0.26) | -840 | -2166 |

Descriptive only, rule picked on the evaluation data itself (p_sel = permutation p of the best of the grid):

| Model | Picked rule | Accuracy | Decided trades | p_sel | Gross bps |
|---|---|---|---|---|---|
| A0 zero-shot | H=15, path15, no conf filter, fade | 57.4% | 251 | 0.525 | +710 |
| A1 | H=1, none, z top 30%, fade | 53.7% | 490 | 0.875 | +141 |
| A2 | H=1, none, pm top 5%, fade | 54.9% | 102 | 0.500 | -23 |
| price-only baseline | H=15, agree3, z top 30%, follow | 56.8% | 125 | 0.550 | +367 |
| neural_trade asis (3-seed ensemble) | H=15, none, z top 20%, follow | 58.3% | 120 | 0.375 | +417 |
| neural_trade rolling1440m (3-seed ensemble) | H=5, agree3, z top 10%, fade | 59.3% | 113 | 0.275 | +19 |

#### post, P-A (calibrated on pre-Test forecasts, evaluated on the full Test)

| Model | Rule | Chosen on calibration | Calib acc (p_sel) | Decision | Test accuracy [95% CI] | p one-sided (Holm) | Trades / decided (per day) | Gross bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 zero-shot | W | H=15, none, pm top 5%, follow | 57.8% (0.100) | **abstain** | (if deployed: 54.7% [47.4%, 61.8%]) | – | 181 / 181 | -67 (t -0.21) | -791 | -1877 |
| A0 zero-shot | R | H=15, none, pm top 2%, follow | 58.6% (0.125) | **abstain** | (if deployed: 61.3% [51.1%, 70.5%]) | – | 93 / 93 | +343 (t 1.55) | -29 | -587 |
| A0 zero-shot (V1+V2 calib) | W | H=15, none, pm top 5%, follow | 61.6% (0.600) | **abstain** | (if deployed: 54.7% [47.4%, 61.8%]) | – | 181 / 181 | -67 (t -0.21) | -791 | -1877 |
| A0 zero-shot (V1+V2 calib) | R | H=5, none, z top 5%, follow | 58.4% (0.375) | **abstain** | (if deployed: 43.9% [38.2%, 49.7%]) | – | 285 / 285 | -363 (t -1.83) | -1503 | -3213 |
| A1 (calib partly in-sample) | W | H=1, none, pm top 20%, fade | 53.8% (0.225) | **abstain** | (if deployed: 50.2% [47.3%, 53.0%]) | – | 1225 / 1180 | +180 (t 1.09) | -4720 | -12070 |
| A1 (calib partly in-sample) | R | H=1, none, pm top 20%, fade | 53.8% (0.500) | **abstain** | (if deployed: 50.2% [47.3%, 53.0%]) | – | 1225 / 1180 | +180 (t 1.09) | -4720 | -12070 |
| A2 (calib partly in-sample) | W | H=1, none, pm top 50%, fade | 52.0% (0.375) | **abstain** | (if deployed: 49.6% [47.7%, 51.5%]) | – | 2813 / 2696 | +199 (t 0.82) | -11053 | -27931 |
| A2 (calib partly in-sample) | R | H=1, none, z top 2%, fade | 54.9% (0.500) | **abstain** | (if deployed: 48.5% [40.4%, 56.8%]) | – | 145 / 138 | -15 (t -0.26) | -595 | -1465 |
| price-only baseline | W | H=5, agree3, z top 2%, fade | 58.9% (0.025) | deploy | 56.1% [46.6%, 65.1%] | 0.123 (0.61) | 108 / 107 (7.7) | -226 (t -1.2) | -658 | -1306 |
| price-only baseline | R | H=5, agree3, z top 2%, fade | 58.9% (0.150) | **abstain** | (if deployed: 56.1% [46.6%, 65.1%]) | – | 108 / 107 | -226 (t -1.2) | -658 | -1306 |

Descriptive only, rule picked on the evaluation data itself (p_sel = permutation p of the best of the grid):

| Model | Picked rule | Accuracy | Decided trades | p_sel | Gross bps |
|---|---|---|---|---|---|
| A0 zero-shot | H=15, none, z top 5%, follow | 57.5% | 181 | 0.300 | -3 |
| A0 zero-shot (V1+V2 calib) | H=15, none, z top 5%, follow | 57.5% | 181 | 0.300 | -3 |
| A1 (calib partly in-sample) | H=1, none, z top 2%, follow | 56.5% | 138 | 0.225 | +58 |
| A2 (calib partly in-sample) | H=1, none, pm top 5%, follow | 51.9% | 316 | 1.000 | +60 |
| price-only baseline | H=15, agree3, z top 30%, fade | 57.4% | 587 | 0.425 | +484 |

#### post, P-B (calibrated on the first half of Test, evaluated on the second half)

| Model | Rule | Chosen on calibration | Calib acc (p_sel) | Decision | Test accuracy [95% CI] | p one-sided (Holm) | Trades / decided (per day) | Gross bps (t) | Net 4 bps | Net 10 bps |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 zero-shot | W | H=15, agree3, z top 5%, follow | 63.1% (0.175) | **abstain** | (if deployed: 48.5% [37.0%, 60.2%]) | – | 68 / 68 | -178 (t -0.86) | -450 | -858 |
| A0 zero-shot | R | H=15, agree3, z top 5%, follow | 63.1% (0.050) | deploy | 48.5% [37.0%, 60.2%] | 0.642 (0.64) | 68 / 68 (9.7) | -178 (t -0.86) | -450 | -858 |
| A1 | W | H=1, none, no conf filter, follow | 50.7% (0.625) | **abstain** | (if deployed: 49.9% [48.1%, 51.7%]) | – | 2935 / 2854 | -320 (t -1.26) | -12060 | -29670 |
| A1 | R | H=1, none, z top 10%, follow | 51.8% (0.975) | **abstain** | (if deployed: 51.1% [45.6%, 56.7%]) | – | 311 / 307 | -4 (t -0.04) | -1248 | -3114 |
| A2 | W | H=1, none, no conf filter, follow | 50.6% (0.675) | **abstain** | (if deployed: 50.2% [48.4%, 52.1%]) | – | 2864 / 2783 | -199 (t -0.79) | -11655 | -28839 |
| A2 | R | H=1, none, pm top 5%, follow | 51.7% (0.975) | **abstain** | (if deployed: 52.1% [44.5%, 59.6%]) | – | 169 / 165 | +40 (t 0.55) | -636 | -1650 |
| price-only baseline | W | H=15, agree3, z top 30%, fade | 60.3% (0.150) | **abstain** | (if deployed: 54.5% [48.7%, 60.1%]) | – | 290 / 290 | +251 (t 0.8) | -909 | -2649 |
| price-only baseline | R | H=15, agree3, z top 30%, fade | 60.3% (0.125) | **abstain** | (if deployed: 54.5% [48.7%, 60.1%]) | – | 290 / 290 | +251 (t 0.8) | -909 | -2649 |
| neural_trade asis (3-seed ensemble) | W | H=15, none, z top 20%, fade | 56.3% (0.725) | **abstain** | (if deployed: 48.1% [42.0%, 54.1%]) | – | 258 / 258 | +215 (t 0.73) | -817 | -2365 |
| neural_trade asis (3-seed ensemble) | R | H=1, none, z top 5%, follow | 57.2% (0.475) | **abstain** | (if deployed: 51.5% [43.9%, 59.0%]) | – | 172 / 165 | +21 (t 0.43) | -667 | -1699 |
| neural_trade rolling1440m (3-seed ensemble) | W | H=15, none, z top 1%, follow | 68.8% (0.500) | **abstain** | (if deployed: 34.6% [19.4%, 53.8%] (too few trades)) | – | 26 / 26 | -53 (t -1.05) | -157 | -313 |
| neural_trade rolling1440m (3-seed ensemble) | R | H=5, agree3, dp top 10%, fade | 57.4% (0.475) | **abstain** | (if deployed: 45.1% [37.4%, 53.0%]) | – | 153 / 153 | -55 (t -0.49) | -667 | -1585 |

Descriptive only, rule picked on the evaluation data itself (p_sel = permutation p of the best of the grid):

| Model | Picked rule | Accuracy | Decided trades | p_sel | Gross bps |
|---|---|---|---|---|---|
| A0 zero-shot | H=15, none, pm top 10%, follow | 56.1% | 123 | 0.775 | +220 |
| A1 | H=1, none, pm top 5%, follow | 52.2% | 161 | 0.925 | +20 |
| A2 | H=1, none, pm top 5%, follow | 52.1% | 165 | 0.925 | +40 |
| price-only baseline | H=15, none, z top 30%, fade | 54.8% | 376 | 0.900 | +388 |
| neural_trade asis (3-seed ensemble) | H=1, agree3, z top 20%, fade | 57.8% | 225 | 0.650 | +108 |
| neural_trade rolling1440m (3-seed ensemble) | H=5, agree3, z top 10%, follow | 61.0% | 200 | 0.075 | +274 |
