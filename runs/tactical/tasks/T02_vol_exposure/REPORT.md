# T02 report: volatility-timed exposure to BTC

The lead wrote this file from the implementer's hand-back (2026-10-13). The implementer's Write tool refused report files.
SPEC: [SPEC.md](SPEC.md), committed at 34b40f4. selection.json was committed at 8f949da, before either look. Code, data
tables and figures are at af23e25.

## Verdict: NO

The volatility switch does not beat plain de-leveraging on the forward data. Criterion 1 fails. Criteria 2 and 3 pass
only nominally (all three are listed below), in a falling market.

## Selection (2019-01..2024-06, score = Calmar(strategy) / Calmar(constant null) at 2 bps)

- **Selected: `b_q0.9`** (cash when the 24 h forecast is above the training 90th percentile). Its score is 1.234.
- It is the only one of the 12 configurations above 1. The next are a_band_k1.0 at 0.968 and a_k1.5 at 0.918.
- Rule (c), the vol-change switch from T01's model family, was the worst: -0.05 and 0.24, with 1,300-2,200 turns per year.
- Full table: `selection_tables.txt`.

## Look 1 (2024-07-01..2025-07-20, descriptive)

| series | return 0 / 2 / 5 bps | max DD 0 / 2 / 5 bps | Calmar 0 / 2 / 5 bps |
|---|---|---|---|
| b_q0.9 | +32.4% / +24.7% / +13.9% | 38.7% / 40.1% / 42.3% | 0.79 / 0.58 / 0.31 |
| constant null (w 0.906) | +80.8% | 28.3% | 2.67 |
| buy-and-hold | +89.8% | 31.0% | 2.70 |

- Paired bootstrap of the Calmar difference against the null at 2 bps: -2.38 [-7.76, -0.14]. The strategy is worse than the null.

## Look 2, the verdict (Binance forward 2025-10-08..2026-09-30, frozen model and thresholds)

| series | return 0 / 2 / 5 bps | max DD 0 / 2 / 5 bps | Calmar 0 / 2 / 5 bps |
|---|---|---|---|
| b_q0.9 | -28.8% / -31.2% / -34.6% | 52.9% / 54.4% / 56.6% | -0.553 / -0.582 / -0.620 |
| constant null (w 0.944) | -29.3% | 50.8% | -0.586 |
| buy-and-hold | -31.1% | 53.0% | -0.596 |

| Criterion (at 2 bps) | Result | Pass |
|---|---|---|
| 1. Max DD at most 0.8 x the null's | 0.544 against a limit of 0.406 | **fail** |
| 2. Calmar above the null's | -0.582 against -0.586 | nominal pass |
| 3. Calmar above buy-and-hold's | -0.582 against -0.596 | nominal pass |

- **Bootstrap:** the Calmar difference against the null is -0.026 [-0.957, +0.859], which is zero.
- **The market:** BTC fell 31% over the forward year, with a 53% drawdown. Every Calmar is negative, so criteria 2 and 3 carry little information.

## Exposure needed for DD < 5% (the owner's line)

| | Look 1 | Look 2 |
|---|---|---|
| Constant exposure to buy-and-hold | 0.150 | 0.075 |
| Scaled strategy: mean exposure | 0.095 | 0.066 |
| Scaled strategy: annual return | +3.1% | -2.1% |

- About 7-15% of capital in BTC is needed, whatever the timing rule.

## Why

- A good forecast of the volatility LEVEL does not tell you when the market falls. The drawdowns of 2024-25 and
  2025-26 built up over weeks, and the high-volatility hours the switch avoids are about as often up as down.
- Look 1: going to cash in the top-decile volatility hours missed rebounds and cost more than it saved.
- The vol-change forecast (T01) is accurate (H83), but as an hourly switch it trades far too often for its size.

## Deviations (SPEC silent or ambiguous)

- **Start of Look 2:** 2025-10-08, because the 7-day RV feature needs 10,080 bars inside the forward file. The null and buy-and-hold use the same hours.
- **Fills:** hour k's exposure is fixed from bars before T_k and trades at the open of T_k. The hour return is open(T_k+1) / open(T_k) - 1.
- **Cost:** bps x |w_k - w_(k-1)|. The first position and the constant null pay no cost.
- **Forecast:** sigma_hat = exp(ridge prediction / 2) on ln per-minute RV over 1 h / 4 h / 24 h / 7 d plus hour sin/cos. The quantiles and sigma* come from the in-sample training forecasts.
- **Rule (c):** a reimplementation of the gaps study 3 S1 model, not T01's artefact.
- **Missing minutes:** 1,160 in the main file, filled with the previous close.
- **Scope:** only the selected configuration was run on the looks.

## Tests and run times

- `test_t02.py`: 7 passed in 8.65 s on 8f949da (`tests.log`).
- Run times: selection 18 s, looks 11 s.
