# Pre-registration: sealed test of the PnL study

Code commit: `49eb0bad717376241e0ce48f51c2ff682d9207aa`. Seal: 2025-11-01T00:00:00+00:00 .. 2026-10-30 23:59 UTC (last bar in the store).

## Track 1

- Deployed: **CASH**; information only: T1-LIN
- p_sel (dev OOS): 0.001

| candidate | ann. net C7-B (bps) | Sharpe | 95% CI daily (bps) | gates passed |
|---|---:|---:|---|---|
| T1-FADEV | -7551 | -3.12 | [-26.08, -15.02] | G3 |
| T1-LIN | 772 | 1.12 | [0.37, 4.25] | G1, G2, G3, G4 |
| T1-NET | 817 | 0.53 | [-1.41, 5.98] | G1, G3, G4 |
| T1-NT | 713 | 0.35 | [-3.43, 8.94] | G1, G3 |
| T1-NET-W | 817 | 0.53 | [-1.41, 5.98] | G3, G4 |

## Track 2

- Deployed: **CASH**; information only: T2-P
- p_sel (dev OOS): 0.141

| candidate | ann. net C7-B (bps) | Sharpe | 95% CI daily (bps) | gates passed |
|---|---:|---:|---|---|
| T2-P | 1269 | 0.35 | [-4.76, 11.82] | G4, G5 |
| T2-LIN | 926 | 0.16 | [-9.00, 14.07] | G4 |
| T2-NET | 1229 | 0.28 | [-7.82, 15.73] | G5 |
| T2-NET-W | -195 | -0.04 | [-9.90, 8.43] | none |

## Verdict rules

- track1: PASS: one-sided stationary-bootstrap p < 0.05 for mean daily net > 0 at C7-B on the full seal, AND mean > 0 on the clean seal, AND mean > 0 in both seal halves. KILL: mean <= 0 on the full or the clean seal. Otherwise INCONCLUSIVE.
- track2: FALSIFIED: seal Sharpe < SR_dev - 1.645 sqrt(1/T_seal + 1/T_dev) on the full or the clean seal. CONSISTENT: not falsified and seal net PnL at C7-B > 0. WEAK: not falsified, PnL <= 0. CONFIRMED (flag): one-sided 95% lower bound of the pooled dev-OOS-plus-seal mean daily net > 0.
- meaning: PASS or CONSISTENT means only 'go to at least 12 months of paper trading', never capital. A deployed candidate that is KILLED or FALSIFIED is dropped for this venue and fee tier.
- cash_track: If a track deploys CASH, its information-only candidate is scored with the same rules, labelled information only.

## Disclosures

- Seen windows inside the seal: 2025-10-11..2025-11-10 and 2026-09-20..2026-10-30 were studied in earlier work in this repository (clean seal excludes them).
- Trend contamination: earlier checks computed daily TSMOM and vol-targeted long BTC on 2022-01..2026-11, covering most of dev OOS and the whole seal; trend-family results are not clean on either period.
- Earlier strat_probe and money-blocker critique looked at dev data.
- The designers (language models) may know the 2025-26 BTC price path from training data.
- 2026-10 funding prints are estimated from the premium index (MAE 0.02 bps per print).
- The 2026-10-28 kline file was missing (a 1-day gap in the seal).
- MLP band nets stayed at cash on Track 1 (also with the declared extended warm-up); LIN cells carry T1-NET. Reported as found, procedure unchanged.
- T1-NT's fold-S selection abstained (no horizon had positive validation net), so T1-NT is flat on the seal. On dev it is retired from trading use by K2 (T1-LIN beats it on C7-B net), although its OOS IC is small but positive (h = 1/5/15: 0.022/0.042/0.047).
- Two code reviews and one documented full dev rerun (C4 rows only) happened before this file; see CHANGES.log. The C7 gate scenarios did not change in the rerun.

Hashes of 36 data, feature and model files are in prereg.json.
