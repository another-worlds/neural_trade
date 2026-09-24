# Ablation `ablate_physics_v1` - scale `full`

84 completed runs; terms LAMBDA_T_PERP, LAMBDA_CASIMIR, LAMBDA_HD, LAMBDA_IFE, LAMBDA_VAC_OVERFLOW, LAMBDA_VAC; seeds [0, 1, 2]; periods P1 (fold -2), P2 (fold -1); lambda calibration: once.

Deltas are paired over (seed, period) and oriented so that **positive = the term helps**. A verdict of VALUE needs mean delta > max(seed sigma, MDE) with at least 83% of pairs agreeing and no guard-rail breach (criteria pre-registered in `configs/ablation_criteria.yaml`).

## Verdicts

| term | leave-one-in | leave-one-out | verdict |
|---|---|---|---|
| `LAMBDA_T_PERP` | INCONCLUSIVE | NEUTRAL | **INCONCLUSIVE** |
| `LAMBDA_CASIMIR` | NEUTRAL | NEUTRAL | **NEUTRAL** |
| `LAMBDA_HD` | INCONCLUSIVE | VALUE | **INCONCLUSIVE** |
| `LAMBDA_IFE` | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** |
| `LAMBDA_VAC_OVERFLOW` | INCONCLUSIVE | NEUTRAL | **INCONCLUSIVE** |
| `LAMBDA_VAC` | INCONCLUSIVE | INCONCLUSIVE | **INCONCLUSIVE** |
| family (all_on vs all_off) | | | **INCONCLUSIVE** |

## Per-metric deltas

| term | mode | metric | pairs | mean delta | sd | seed sigma | MDE | +/- | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `LAMBDA_T_PERP` | leave_one_in | `h1/variance/crpss` | 6 | +0.0033 | +0.0085 | +0.0055 | 0.005 | 4/2 | NEUTRAL |
| `LAMBDA_T_PERP` | leave_one_in | `h1/variance/corr_var_err2_spearman` | 6 | +0.0197 | +0.0305 | +0.0274 | 0.01 | 4/2 | INCONCLUSIVE |
| `LAMBDA_T_PERP` | leave_one_out | `h1/variance/crpss` | 6 | +0.0015 | +0.0061 | +0.0060 | 0.005 | 4/2 | NEUTRAL |
| `LAMBDA_T_PERP` | leave_one_out | `h1/variance/corr_var_err2_spearman` | 6 | -0.0005 | +0.0077 | +0.0079 | 0.01 | 3/3 | NEUTRAL |
| `LAMBDA_CASIMIR` | leave_one_in | `h1/variance/crpss` | 6 | -0.0015 | +0.0047 | +0.0055 | 0.005 | 3/3 | NEUTRAL |
| `LAMBDA_CASIMIR` | leave_one_in | `h1/variance/corr_var_err2_spearman` | 6 | +0.0019 | +0.0118 | +0.0274 | 0.01 | 3/3 | NEUTRAL |
| `LAMBDA_CASIMIR` | leave_one_in | guard-rail `h1/direction/auc` | | -0.0117 | | | tol 0.01 | | BREACH |
| `LAMBDA_CASIMIR` | leave_one_out | `h1/variance/crpss` | 6 | +0.0020 | +0.0069 | +0.0059 | 0.005 | 4/2 | NEUTRAL |
| `LAMBDA_CASIMIR` | leave_one_out | `h1/variance/corr_var_err2_spearman` | 6 | +0.0068 | +0.0177 | +0.0120 | 0.01 | 4/2 | NEUTRAL |
| `LAMBDA_HD` | leave_one_in | `h1/variance/crpss` | 6 | +0.0019 | +0.0035 | +0.0055 | 0.005 | 4/2 | NEUTRAL |
| `LAMBDA_HD` | leave_one_in | `h1/variance/corr_var_err2_spearman` | 6 | +0.0264 | +0.0123 | +0.0274 | 0.01 | 6/0 | INCONCLUSIVE |
| `LAMBDA_HD` | leave_one_in | guard-rail `h1/direction/auc` | | -0.0126 | | | tol 0.01 | | BREACH |
| `LAMBDA_HD` | leave_one_out | `h1/variance/crpss` | 6 | +0.0033 | +0.0053 | +0.0024 | 0.005 | 5/1 | NEUTRAL |
| `LAMBDA_HD` | leave_one_out | `h1/variance/corr_var_err2_spearman` | 6 | +0.0207 | +0.0135 | +0.0182 | 0.01 | 6/0 | VALUE |
| `LAMBDA_IFE` | leave_one_in | `h1/direction/mcc` | 6 | -0.0193 | +0.0385 | +0.0228 | 0.005 | 2/4 | INCONCLUSIVE |
| `LAMBDA_IFE` | leave_one_in | `h1/direction/auc` | 6 | -0.0075 | +0.0250 | +0.0276 | 0.005 | 1/5 | INCONCLUSIVE |
| `LAMBDA_IFE` | leave_one_out | `h1/direction/mcc` | 6 | -0.0145 | +0.0230 | +0.0194 | 0.005 | 1/5 | INCONCLUSIVE |
| `LAMBDA_IFE` | leave_one_out | `h1/direction/auc` | 6 | -0.0086 | +0.0142 | +0.0137 | 0.005 | 2/4 | INCONCLUSIVE |
| `LAMBDA_VAC_OVERFLOW` | leave_one_in | `h1/variance/crpss` | 6 | +0.0035 | +0.0086 | +0.0055 | 0.005 | 5/1 | NEUTRAL |
| `LAMBDA_VAC_OVERFLOW` | leave_one_in | `h1/variance/corr_var_err2_spearman` | 6 | +0.0159 | +0.0301 | +0.0274 | 0.01 | 4/2 | INCONCLUSIVE |
| `LAMBDA_VAC_OVERFLOW` | leave_one_out | `h1/variance/crpss` | 6 | +0.0011 | +0.0092 | +0.0095 | 0.005 | 2/4 | NEUTRAL |
| `LAMBDA_VAC_OVERFLOW` | leave_one_out | `h1/variance/corr_var_err2_spearman` | 6 | +0.0015 | +0.0058 | +0.0064 | 0.01 | 4/2 | NEUTRAL |
| `LAMBDA_VAC_OVERFLOW` | leave_one_out | guard-rail `h1/direction/auc` | | -0.0118 | | | tol 0.01 | | BREACH |
| `LAMBDA_VAC` | leave_one_in | `coherence/coherence_primary` | 6 | -0.0116 | +0.2912 | +0.1766 | 0.01 | 2/4 | INCONCLUSIVE |
| `LAMBDA_VAC` | leave_one_out | `coherence/coherence_primary` | 6 | -0.0164 | +0.1075 | +0.2506 | 0.01 | 3/3 | INCONCLUSIVE |
| family | all_on vs all_off | `h1/variance/crpss` | 6 | +0.0056 | +0.0114 | +0.0055 | 0.005 | 5/1 | VALUE |
| family | all_on vs all_off | `h1/variance/corr_var_err2_spearman` | 6 | +0.0364 | +0.0329 | +0.0274 | 0.01 | 6/0 | VALUE |
| family | all_on vs all_off | `h1/direction/mcc` | 6 | -0.0168 | +0.0152 | +0.0228 | 0.005 | 1/5 | INCONCLUSIVE |
| family | all_on vs all_off | `backtest/sharpe_net` | 6 | -1.4660 | +18.7155 | +15.0856 | 0.5 | 3/3 | INCONCLUSIVE |

## Condition means

| condition | n | `backtest/sharpe_net` | `coherence/coherence_primary` | `h1/direction/auc` | `h1/direction/mcc` | `h1/variance/corr_var_err2_spearman` | `h1/variance/crpss` |
|---|---|---|---|---|---|---|---|
| `all_on` | 6 | -124.8453 | 0.5477 | 0.4862 | -0.0186 | 0.3336 | 0.0284 |
| `all_off` | 6 | -123.3793 | 0.4103 | 0.4981 | -0.0018 | 0.2972 | 0.0228 |
| `only:LAMBDA_T_PERP` | 6 | -124.5856 | 0.4913 | 0.4961 | 0.0025 | 0.3169 | 0.0261 |
| `only:LAMBDA_CASIMIR` | 6 | -117.4200 | 0.4903 | 0.4864 | -0.0213 | 0.2991 | 0.0213 |
| `only:LAMBDA_HD` | 6 | -124.9749 | 0.5950 | 0.4855 | -0.0307 | 0.3236 | 0.0247 |
| `only:LAMBDA_IFE` | 6 | -126.3551 | 0.4395 | 0.4906 | -0.0211 | 0.3114 | 0.0249 |
| `only:LAMBDA_VAC_OVERFLOW` | 6 | -117.1423 | 0.4434 | 0.4910 | -0.0120 | 0.3131 | 0.0263 |
| `only:LAMBDA_VAC` | 6 | -132.6779 | 0.3987 | 0.4902 | -0.0170 | 0.2986 | 0.0226 |
| `without:LAMBDA_T_PERP` | 6 | -121.7672 | 0.6521 | 0.4848 | -0.0224 | 0.3341 | 0.0269 |
| `without:LAMBDA_CASIMIR` | 6 | -115.4339 | 0.5228 | 0.4935 | -0.0112 | 0.3268 | 0.0264 |
| `without:LAMBDA_HD` | 6 | -134.1234 | 0.3973 | 0.4937 | 0.0005 | 0.3129 | 0.0251 |
| `without:LAMBDA_IFE` | 6 | -123.8206 | 0.6847 | 0.4948 | -0.0040 | 0.3297 | 0.0250 |
| `without:LAMBDA_VAC_OVERFLOW` | 6 | -124.7650 | 0.6550 | 0.4980 | -0.0026 | 0.3321 | 0.0273 |
| `without:LAMBDA_VAC` | 6 | -131.1389 | 0.5641 | 0.4866 | -0.0174 | 0.3284 | 0.0271 |

## Runs

| key | run id | epochs | minutes |
|---|---|---|---|
| all_off__s0__P1 | `20260923T203258Z-6dec27a-ddfc658c-all_off__s0__P1` | 16 | 4.4 |
| all_off__s0__P2 | `20260923T204821Z-6dec27a-db86c2eb-all_off__s0__P2` | 20 | 6.8 |
| all_off__s1__P1 | `20260923T203725Z-6dec27a-41ade6b7-all_off__s1__P1` | 20 | 5.4 |
| all_off__s1__P2 | `20260923T205517Z-6dec27a-1322b571-all_off__s1__P2` | 20 | 6.8 |
| all_off__s2__P1 | `20260923T204253Z-6dec27a-f485097f-all_off__s2__P1` | 20 | 5.4 |
| all_off__s2__P2 | `20260923T210213Z-6dec27a-60271b35-all_off__s2__P2` | 20 | 6.9 |
| all_on__s0__P1 | `20260923T195537Z-6dec27a-53cbe6af-all_on__s0__P1` | 20 | 5.4 |
| all_on__s0__P2 | `20260923T201204Z-6dec27a-f2f93af0-all_on__s0__P2` | 20 | 6.9 |
| all_on__s1__P1 | `20260923T200108Z-6dec27a-c41d9410-all_on__s1__P1` | 20 | 5.4 |
| all_on__s1__P2 | `20260923T201903Z-6dec27a-aa4aa01b-all_on__s1__P2` | 20 | 6.9 |
| all_on__s2__P1 | `20260923T200636Z-6dec27a-63ef2e2c-all_on__s2__P1` | 20 | 5.4 |
| all_on__s2__P2 | `20260923T202600Z-6dec27a-61cc47c8-all_on__s2__P2` | 20 | 6.9 |
| only-LAMBDA_CASIMIR__s0__P1 | `20260923T214544Z-6dec27a-9abeb709-only-LAMBDA_CASIMIR__s0__P1` | 15 | 4.1 |
| only-LAMBDA_CASIMIR__s0__P2 | `20260923T220047Z-6dec27a-5281435f-only-LAMBDA_CASIMIR__s0__P2` | 20 | 6.8 |
| only-LAMBDA_CASIMIR__s1__P1 | `20260923T214956Z-6dec27a-bffb56ab-only-LAMBDA_CASIMIR__s1__P1` | 20 | 5.3 |
| only-LAMBDA_CASIMIR__s1__P2 | `20260923T220743Z-6dec27a-37cda2bb-only-LAMBDA_CASIMIR__s1__P2` | 20 | 6.8 |
| only-LAMBDA_CASIMIR__s2__P1 | `20260923T215521Z-6dec27a-1573bc90-only-LAMBDA_CASIMIR__s2__P1` | 20 | 5.3 |
| only-LAMBDA_CASIMIR__s2__P2 | `20260923T221438Z-6dec27a-2b342340-only-LAMBDA_CASIMIR__s2__P2` | 20 | 6.8 |
| only-LAMBDA_HD__s0__P1 | `20260923T222133Z-6dec27a-be7548f9-only-LAMBDA_HD__s0__P1` | 13 | 3.6 |
| only-LAMBDA_HD__s0__P2 | `20260923T223607Z-6dec27a-b2525ccd-only-LAMBDA_HD__s0__P2` | 20 | 6.8 |
| only-LAMBDA_HD__s1__P1 | `20260923T222516Z-6dec27a-ebcdcc31-only-LAMBDA_HD__s1__P1` | 20 | 5.3 |
| only-LAMBDA_HD__s1__P2 | `20260923T224301Z-6dec27a-79b60dc2-only-LAMBDA_HD__s1__P2` | 20 | 6.8 |
| only-LAMBDA_HD__s2__P1 | `20260923T223041Z-6dec27a-3ab49427-only-LAMBDA_HD__s2__P1` | 20 | 5.3 |
| only-LAMBDA_HD__s2__P2 | `20260923T224957Z-6dec27a-d791e2d9-only-LAMBDA_HD__s2__P2` | 20 | 6.8 |
| only-LAMBDA_IFE__s0__P1 | `20260923T225652Z-6dec27a-716e2a67-only-LAMBDA_IFE__s0__P1` | 15 | 4.1 |
| only-LAMBDA_IFE__s0__P2 | `20260923T231203Z-6dec27a-99449444-only-LAMBDA_IFE__s0__P2` | 20 | 7.1 |
| only-LAMBDA_IFE__s1__P1 | `20260923T230105Z-6dec27a-b16eed59-only-LAMBDA_IFE__s1__P1` | 20 | 5.3 |
| only-LAMBDA_IFE__s1__P2 | `20260923T231918Z-6dec27a-609d77f0-only-LAMBDA_IFE__s1__P2` | 20 | 7.1 |
| only-LAMBDA_IFE__s2__P1 | `20260923T230630Z-6dec27a-e0c28a89-only-LAMBDA_IFE__s2__P1` | 20 | 5.4 |
| only-LAMBDA_IFE__s2__P2 | `20260923T232631Z-6dec27a-7f1f51eb-only-LAMBDA_IFE__s2__P2` | 20 | 7.1 |
| only-LAMBDA_T_PERP__s0__P1 | `20260923T210911Z-6dec27a-8c390f51-only-LAMBDA_T_PERP__s0__P1` | 20 | 5.4 |
| only-LAMBDA_T_PERP__s0__P2 | `20260923T212531Z-6dec27a-ae8a00d9-only-LAMBDA_T_PERP__s0__P2` | 20 | 6.9 |
| only-LAMBDA_T_PERP__s1__P1 | `20260923T211439Z-6dec27a-aa3f3cea-only-LAMBDA_T_PERP__s1__P1` | 20 | 5.3 |
| only-LAMBDA_T_PERP__s1__P2 | `20260923T213229Z-6dec27a-c948eae6-only-LAMBDA_T_PERP__s1__P2` | 18 | 6.2 |
| only-LAMBDA_T_PERP__s2__P1 | `20260923T212005Z-6dec27a-bc3f4dd8-only-LAMBDA_T_PERP__s2__P1` | 20 | 5.4 |
| only-LAMBDA_T_PERP__s2__P2 | `20260923T213847Z-6dec27a-65ddc87f-only-LAMBDA_T_PERP__s2__P2` | 20 | 6.9 |
| only-LAMBDA_VAC_OVERFLOW__s0__P1 | `20260923T233345Z-6dec27a-f4413078-only-LAMBDA_VAC_OVERFLOW__s0__P1` | 20 | 5.6 |
| only-LAMBDA_VAC_OVERFLOW__s0__P2 | `20260924T010620Z-6dec27a-67ad3323-only-LAMBDA_VAC_OVERFLOW__s0__P2` | 20 | 6.8 |
| only-LAMBDA_VAC_OVERFLOW__s1__P1 | `20260923T233930Z-6dec27a-7a7cf27c-only-LAMBDA_VAC_OVERFLOW__s1__P1` | 20 | 5.6 |
| only-LAMBDA_VAC_OVERFLOW__s1__P2 | `20260924T011315Z-6dec27a-6907527e-only-LAMBDA_VAC_OVERFLOW__s1__P2` | 20 | 6.8 |
| only-LAMBDA_VAC_OVERFLOW__s2__P1 | `20260923T234515Z-6dec27a-e94e22c9-only-LAMBDA_VAC_OVERFLOW__s2__P1` | 20 | 5.6 |
| only-LAMBDA_VAC_OVERFLOW__s2__P2 | `20260924T012011Z-6dec27a-64666da6-only-LAMBDA_VAC_OVERFLOW__s2__P2` | 20 | 6.8 |
| only-LAMBDA_VAC__s0__P1 | `20260924T012705Z-6dec27a-630ef563-only-LAMBDA_VAC__s0__P1` | 13 | 3.6 |
| only-LAMBDA_VAC__s0__P2 | `20260924T014145Z-6dec27a-68405b1d-only-LAMBDA_VAC__s0__P2` | 20 | 7.0 |
| only-LAMBDA_VAC__s1__P1 | `20260924T013047Z-6dec27a-0379d736-only-LAMBDA_VAC__s1__P1` | 20 | 5.3 |
| only-LAMBDA_VAC__s1__P2 | `20260924T014854Z-6dec27a-ba65ef4b-only-LAMBDA_VAC__s1__P2` | 19 | 6.7 |
| only-LAMBDA_VAC__s2__P1 | `20260924T013613Z-6dec27a-d35ba113-only-LAMBDA_VAC__s2__P1` | 20 | 5.4 |
| only-LAMBDA_VAC__s2__P2 | `20260924T015539Z-6dec27a-9269d5cd-only-LAMBDA_VAC__s2__P2` | 20 | 6.9 |
| without-LAMBDA_CASIMIR__s0__P1 | `20260924T023918Z-6dec27a-7052878d-without-LAMBDA_CASIMIR__s0__P1` | 20 | 5.3 |
| without-LAMBDA_CASIMIR__s0__P2 | `20260924T025600Z-6dec27a-576e0607-without-LAMBDA_CASIMIR__s0__P2` | 20 | 7.5 |
| without-LAMBDA_CASIMIR__s1__P1 | `20260924T024443Z-6dec27a-a05101d4-without-LAMBDA_CASIMIR__s1__P1` | 20 | 5.5 |
| without-LAMBDA_CASIMIR__s1__P2 | `20260924T030337Z-6dec27a-e499a12e-without-LAMBDA_CASIMIR__s1__P2` | 20 | 14.1 |
| without-LAMBDA_CASIMIR__s2__P1 | `20260924T025022Z-6dec27a-e98dc3b1-without-LAMBDA_CASIMIR__s2__P1` | 20 | 5.5 |
| without-LAMBDA_CASIMIR__s2__P2 | `20260924T111024Z-6dec27a-83f73f49-without-LAMBDA_CASIMIR__s2__P2` | 20 | 7.0 |
| without-LAMBDA_HD__s0__P1 | `20260924T111729Z-6dec27a-9a2a991a-without-LAMBDA_HD__s0__P1` | 20 | 8.1 |
| without-LAMBDA_HD__s0__P2 | `20260924T113748Z-6dec27a-e2883e6b-without-LAMBDA_HD__s0__P2` | 20 | 7.4 |
| without-LAMBDA_HD__s1__P1 | `20260924T112550Z-6dec27a-f3f1e2f1-without-LAMBDA_HD__s1__P1` | 20 | 6.0 |
| without-LAMBDA_HD__s1__P2 | `20260924T114518Z-6dec27a-e1ace559-without-LAMBDA_HD__s1__P2` | 20 | 12.9 |
| without-LAMBDA_HD__s2__P1 | `20260924T113200Z-6dec27a-e857c730-without-LAMBDA_HD__s2__P1` | 20 | 5.7 |
| without-LAMBDA_HD__s2__P2 | `20260924T115834Z-6dec27a-963be8a4-without-LAMBDA_HD__s2__P2` | 20 | 10.0 |
| without-LAMBDA_IFE__s0__P1 | `20260924T120924Z-6dec27a-5b851456-without-LAMBDA_IFE__s0__P1` | 20 | 7.0 |
| without-LAMBDA_IFE__s0__P2 | `20260924T122721Z-6dec27a-9bde89bf-without-LAMBDA_IFE__s0__P2` | 20 | 6.8 |
| without-LAMBDA_IFE__s1__P1 | `20260924T121628Z-6dec27a-b1ed200f-without-LAMBDA_IFE__s1__P1` | 20 | 5.4 |
| without-LAMBDA_IFE__s1__P2 | `20260924T123416Z-6dec27a-e5e02f0f-without-LAMBDA_IFE__s1__P2` | 20 | 6.8 |
| without-LAMBDA_IFE__s2__P1 | `20260924T122200Z-6dec27a-7f2f979a-without-LAMBDA_IFE__s2__P1` | 20 | 5.3 |
| without-LAMBDA_IFE__s2__P2 | `20260924T124107Z-6dec27a-ba0592e2-without-LAMBDA_IFE__s2__P2` | 20 | 6.9 |
| without-LAMBDA_T_PERP__s0__P1 | `20260924T020237Z-6dec27a-04b1da62-without-LAMBDA_T_PERP__s0__P1` | 20 | 5.3 |
| without-LAMBDA_T_PERP__s0__P2 | `20260924T021847Z-6dec27a-e88f709f-without-LAMBDA_T_PERP__s0__P2` | 20 | 6.8 |
| without-LAMBDA_T_PERP__s1__P1 | `20260924T020800Z-6dec27a-1d21ad46-without-LAMBDA_T_PERP__s1__P1` | 20 | 5.3 |
| without-LAMBDA_T_PERP__s1__P2 | `20260924T022538Z-6dec27a-5d19e662-without-LAMBDA_T_PERP__s1__P2` | 20 | 6.7 |
| without-LAMBDA_T_PERP__s2__P1 | `20260924T021321Z-6dec27a-87535633-without-LAMBDA_T_PERP__s2__P1` | 20 | 5.3 |
| without-LAMBDA_T_PERP__s2__P2 | `20260924T023227Z-6dec27a-e9b47e07-without-LAMBDA_T_PERP__s2__P2` | 20 | 6.8 |
| without-LAMBDA_VAC_OVERFLOW__s0__P1 | `20260924T124807Z-6dec27a-b67b786b-without-LAMBDA_VAC_OVERFLOW__s0__P1` | 20 | 6.2 |
| without-LAMBDA_VAC_OVERFLOW__s0__P2 | `20260924T131054Z-6dec27a-765ab374-without-LAMBDA_VAC_OVERFLOW__s0__P2` | 20 | 9.4 |
| without-LAMBDA_VAC_OVERFLOW__s1__P1 | `20260924T125508Z-6dec27a-cb1102b0-without-LAMBDA_VAC_OVERFLOW__s1__P1` | 20 | 7.9 |
| without-LAMBDA_VAC_OVERFLOW__s1__P2 | `20260924T132028Z-6dec27a-0091176b-without-LAMBDA_VAC_OVERFLOW__s1__P2` | 20 | 11.3 |
| without-LAMBDA_VAC_OVERFLOW__s2__P1 | `20260924T130411Z-6dec27a-10461644-without-LAMBDA_VAC_OVERFLOW__s2__P1` | 20 | 6.6 |
| without-LAMBDA_VAC_OVERFLOW__s2__P2 | `20260924T133154Z-6dec27a-7ebf0d14-without-LAMBDA_VAC_OVERFLOW__s2__P2` | 18 | 10.8 |
| without-LAMBDA_VAC__s0__P1 | `20260924T134305Z-6dec27a-23e1d47c-without-LAMBDA_VAC__s0__P1` | 20 | 34.9 |
| without-LAMBDA_VAC__s0__P2 | `20260924T144533Z-6dec27a-b365be62-without-LAMBDA_VAC__s0__P2` | 20 | 12.8 |
| without-LAMBDA_VAC__s1__P1 | `20260924T141821Z-6dec27a-e1c93970-without-LAMBDA_VAC__s1__P1` | 20 | 16.1 |
| without-LAMBDA_VAC__s1__P2 | `20260924T145858Z-6dec27a-82ba74cf-without-LAMBDA_VAC__s1__P2` | 20 | 12.9 |
| without-LAMBDA_VAC__s2__P1 | `20260924T143448Z-6dec27a-cdd868e4-without-LAMBDA_VAC__s2__P1` | 20 | 10.6 |
| without-LAMBDA_VAC__s2__P2 | `20260924T151208Z-6dec27a-2d526d5b-without-LAMBDA_VAC__s2__P2` | 20 | 12.1 |
