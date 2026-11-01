# Dev gates (dev OOS 2021-01-01..2025-10-31, engine E)

## Track 1

p_sel = 0.001 (K = 4; best by t: T1-LIN)

| candidate | ann. net C7-B (bps) | Sharpe | 95% LB daily (bps) | HAC t | G1 | G2 | G3 | G4 | G5 | pass |
|---|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| T1-FADEV | -7551 | -3.12 | -25.23 | -7.35 | n | n | Y | n | n | - |
| T1-LIN | 772 | 1.12 | 0.61 | 2.22 | Y | Y | Y | Y | n | - |
| T1-NET | 817 | 0.53 | -0.80 | 1.17 | Y | n | Y | Y | n | - |
| T1-NT | 713 | 0.35 | -2.77 | 0.63 | Y | n | Y | n | n | - |
| T1-NET-W (descriptive) | 817 | 0.53 | -0.80 | 1.17 | - | n | Y | Y | n | - |

Deployed: **CASH**; information-only: T1-LIN

## Track 2

p_sel = 0.141 (K = 3; best by t: T2-P)

| candidate | ann. net C7-B (bps) | Sharpe | 95% LB daily (bps) | HAC t | G1 | G2 | G3 | G4 | G5 | pass |
|---|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| T2-P | 1269 | 0.35 | -3.50 | 0.82 | n | n | n | Y | Y | - |
| T2-LIN | 926 | 0.16 | -7.17 | 0.43 | n | n | n | Y | n | - |
| T2-NET | 1229 | 0.28 | -6.12 | 0.58 | n | n | n | n | Y | - |
| T2-NET-W (descriptive) | -195 | -0.04 | -8.25 | -0.11 | - | n | n | n | n | - |

Deployed: **CASH**; information-only: T2-P

