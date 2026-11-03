# Best result and outlook (descriptive, written after the seal; no result changed)

Written 2026-11-03. Sources: the scored files in this folder (gate.json, seal.json, seal_daily.npz, the dev daily
series), an independent check of every number, and the literature listed at the end. Nothing was re-scored or re-chosen.

## The best result

Two strategies made money both on dev (2021-01..2025-10) and in the sealed year (2025-11..2026-10): T2-P, the Track-2
information-only candidate (chosen on dev before the seal), and tsmom_unscaled, the plain daily trend rule it copies
(daily correlation 0.98). The plain rule is slightly better in every period; the network adds nothing. Both are long or
short BTC perpetuals with 37-44 trades a year. Net of fees and funding, C7-B, log returns:

| | dev %/yr | dev Sharpe | dev max DD | sealed year | sealed Sharpe | pooled %/yr | pooled Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| tsmom_unscaled | +17.2 | 0.39 | 51% | +34.4% | 1.11 | +20.2 | 0.48 |
| T2-P | +12.7 | 0.35 | 46.5% | +33.2% | 1.10 | +16.2 | 0.46 |
| buy-and-hold | +15.7 | 0.25 | 80.4% | -38.6% | -0.87 | +6.4 | 0.11 |

T2-P by year: 2021 +37.8, 2022 -4.6, 2023 +52.0, 2024 +5.3, 2025 (Jan-Oct) -29.2, sealed year +33.2.

## What holds and what does not

- Holds: sealed verdict CONSISTENT (positive, not falsified); both seal halves positive (+20.0, +13.2); 8 of 12 months
  positive; fees change little (C4 to C10: dev 10.8-13.5 %/yr).
- Not CONFIRMED: the pooled 95% lower bound of the mean daily net is -1.55 bps. Dev gates G1-G3 failed; the
  pre-registered decision was CASH. Deflated Sharpe with 129 trials: 0.03 (dev), 0.07 (pooled). The 88% share of
  positive bootstrap means is about 1 minus a one-sided p-value, not the probability of an edge.
- The seal test was easy to pass: with no edge, a strategy is CONSISTENT about half the time. One-year Sharpe has a
  standard error of about 1.
- The sealed-year profit came from the short side (+31.5 of +33.2) in a falling market. On dev the short side lost
  (-47%); the dev profit came from long trades in 2021 and 2023.
- It pays in large moves in either direction and loses in quiet months (pooled months with |B&H| < 10%: -2.5 to
  -4.8% on average). It lost 4.6% in 2022, when buy-and-hold fell 67%.
- It is still below its 2024-03-13 peak: 22% (simple) at the end of the seal, after 960 days. From 2024-01 to 2026-10
  it made +3.3 %/yr (Sharpe 0.10).
- Profit depends on a few days: in the sealed year the best 5 days gave +35.1%; without the best 10 the year is -23.6%.
  This is normal for trend following (and for buy-and-hold), but one year's result rests on 5-10 trend days.
- Funding is the main cost when the book is long in a bull market (dev: funding 23% of equity, fees 11%).
- The seal lies in the disclosed trend-contamination window, and the designers may know the 2025-26 price path.

## Outlook for the next 12 months

- Base case: a small positive edge, Sharpe about 0.3-0.5 (the typical single-market trend Sharpe), not the seal's 1.1.
- About a one-in-three chance of a losing year (simulation at Sharpe 0.35-0.46: 32-36%; past rolling 12-month
  windows: 29% negative, 5th to 95th percentile -27% to +49%).
- Drawdown along the way: median about 28-29%, 1 in 10 years above 43-45%; over 5 years a 50% drawdown is typical.
- It pays if BTC moves in strong, sustained trends, up or down; it bleeds in sideways markets.
- Evidence for BTC trend following out of sample is mixed: some studies find it kept working on a risk-adjusted
  basis, others find no out-of-sample predictability or losses in 2022-2023.

Next step under the study's own rule: at least 12 months of paper trading before any capital. Paper trading checks
execution, fees and funding; it cannot confirm the edge (at Sharpe 0.4, a t-statistic of 2 needs about 25 years).

## Literature

- Moskowitz, Ooi, Pedersen, Time Series Momentum, JFE 2012.
- Hurst, Ooi, Pedersen, A Century of Evidence on Trend-Following Investing, JPM 2017 (per-market Sharpe about 0.4).
- Lemperiere et al., Two Centuries of Trend Following, J. Investment Strategies 2014 (short-horizon trends withered).
- Huang, Li, Wang, Zhou, Time Series Momentum: Is It There?, JFE 2020 (asset-by-asset TSMOM weak out of sample).
- Liu, Tsyvinski, Risks and Returns of Cryptocurrency, RFS 2021 (BTC momentum, early era, weaker after 2013).
- Hudson, Urquhart, Technical Trading and Cryptocurrencies, Annals of OR 2021 (no BTC predictability out of sample).
- Deprez, Fruemmel, Are simple technical trading rules profitable in bitcoin markets?, IREF 2024 (risk-adjusted gains).
- Cambridge Associates, Does Trend-Following's Recent Struggle Signal That the Strategy Is Structurally Broken?, 2025.
