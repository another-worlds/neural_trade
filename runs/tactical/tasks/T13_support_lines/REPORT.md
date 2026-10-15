# T13 report: support and resistance lines as synthetic geometric indicators

Lead, from the implementer's hand-back (2026-10-15).
- **SPEC:** [SPEC.md](SPEC.md) at 1869df6, committed before any result.
- **Commits:** selection.json in 48e0c56, before the judgements; final 97d48ad.
- **Tests:** 20 passed. Every line feature is causal: replacing all bars from the anchor on leaves it bit-identical.

## Lines

| Line | What it is |
|---|---|
| S1 | upper and lower convex-hull lines |
| S2 | the nearest unbroken pivot levels (confirmed k = 3 bars later), with touch counts |
| S3 | an OLS regression channel at ±2 sd |

All distances and widths are in sigma units.

## Verdict (Holm over the three parts; each part needs both looks)

| Part | Target | J1 Bitstamp 2017-18 | J2 Binance 2025-26 | Holm p | Result |
|---|---|---|---|---|---|
| **C** | next-hour volatility change, over T11's V3 | **+0.0195 [+0.0138, +0.0251], 14/15** | **+0.0087 [+0.0033, +0.0141], 9/11** | **0.0072** | **PASS** |
| B | size of the next 60 min, over O5's L1 | +0.0004 [-0.0033, +0.0041] | +0.0008 [+0.0001, +0.0016] | 0.78 | below the +0.003 bar |
| A | P(±50 bps in 60 min), over H3 | -0.69% | -0.22% | 0.88 | no gain |

**Reading.** Support and resistance lines on 1-minute bars over 240 bars add real skill to the forecast of the next
hour's volatility change. This holds on two venues and on both ends of the history. The lines add nothing to the hit
probability, which H3 already prices from sigma, and next to nothing to the level of size.

**Caveat.** J1 and J2 are second looks: the V3 base saw both periods. The SR block is the new element.

## Selection (2019-01..2024-06), chosen pair per part

| Part | Bars | Lookback | Mean gain |
|---|---|---|---|
| A | 15 min | 240 | +0.0028 |
| B | 15 min | 60 | +0.0008 |
| C | 1 min | 240 | +0.0125 |

## Prospective freeze
- A1, B1 and C1 are frozen on Binance to 2026-09-30, sha256 `8fec9bf3667e48061b075aed6e4c64d70a6eae8d92fc8154ab913c95f689f553`.
- `score_B.py` scores once 3 complete months from 2026-10-02 exist.

## Page
A new section `#lines` in `report/positive_outlook.html` draws the hull, pivots and channel on three real Binance
windows, chosen at the 10th, 50th and 90th percentiles of H3's P(hit), next to P(hit) and sigma. It has a results card
per part.

## Deviations
- **Sigma for parts B and C:** a fixed blend of S3's log-variance columns. H3's HAR sd is used for part A.
- **Window filter:** rows need a clean SR window, so the bases trained on very slightly different rows per pair.
- **Clipping:** features are clipped at ±10 sigma.
- **Code copies:** the O1/O3/O5 code is copied from the outlook branch at ed87069; the outlook branch was not merged.
- **J2 end:** J2 stops at 2026-09-30 23:59.
