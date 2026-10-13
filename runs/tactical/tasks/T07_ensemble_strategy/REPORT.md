# T07 report: an ensemble trading strategy with a backtest

The lead wrote this from the implementer's hand-back (2026-10-13).

| Item | Value |
|---|---|
| Owner request | "ансамблевую трейдинг стратегию с бэктестом" |
| SPEC | [SPEC.md](SPEC.md), commit 70be47f |
| selection.json | committed in 87e9b03, before any look |
| Results and figures | e4f41a3 |
| Tests | 22 passed |

## Verdict: FAIL on all four forward criteria at 2 bps

**Chosen on 2019-01..2024-06:** C1, equal risk across the 6 members. Calmar 0.35 at 2 bps; C2 had 0.34, so the
choice is essentially a coin flip.

**Forward, Binance 2025-11..2026-09** (models frozen at 2025-07-20):

| | Return | Max DD | Calmar | Monthly mean [95% CI] |
|---|---|---|---|---|
| C1 at 2 bps | -9.1% | 17.5% | -0.56 | -0.78% [-3.47, +1.90]; 6/11 months positive |
| Random-side null p95 | -8.35% | | | |
| Buy-and-hold (BTC -23%) | -23.2% | 47.3% | -0.53 | |

The criteria: the monthly lower bound is not above 0; C1 is not above the null p95 (close); its Calmar is not above
buy-and-hold's; its DD is not under 15%.

**Owner's goal, C1 at 2 bps:**

| Measure | Result | Goal |
|---|---|---|
| Trade hit | 54% (60% at 0 bps) | > 60% |
| Max DD | 17.5% | < 5% |
| Leverage for DD < 5% | 0.27x, return -2.4% | |

## Costs decide it

- **At 0 bps** (the project default, D-044) the forward returns are:
  - C1 +14.5%;
  - C2 +15.2%;
  - C5 +12.6%;
  - buy-and-hold -23.2%.
- **At 2 bps:** C1 is a 4,000-trades-per-year book. Its small gross edge (about 0.4 bps per trade) disappears at a 2 bps round trip.
- **Not tested:** a cost-aware version (turnover band, minimum hold) would be a new candidate under its own SPEC.

## Members, forward at 2 bps

| Member | Logic | Return | Max DD |
|---|---|---|---|
| M1 | minute reversal | -30% | 36% |
| M2 | H62 recipe | -16% | 24% |
| M3 | 1 h rich L1 | -17% | 26% |
| M4 | EMA 20/50 trend | +14% | 36% |
| M5 | RSI mean reversion | -12% | 26% |
| M6 | Bollinger breakout | +36% | 24% |

- **Only the textbook trend and breakout rules gained,** in a year when BTC fell 23%. In selection both were near
  zero, at +26% and +12% over 5.5 years. Nothing chose them, and one year of trend-following gains is weak evidence.
- **Diversity:** M4/M6 and M5 are strongly anti-correlated (-0.6 / -0.7). The model members M2 and M3 correlate +0.45.

## Combinations, return at 2 bps (and max DD for the forward period)

| Combination | Selection | Look 1 | Forward |
|---|---|---|---|
| C1 equal risk | +160% | -6.3% | -9.1% (DD 17.5%) |
| C2 equal weight | +118% | -4.7% | -2.1% (DD 15.0%) |
| C3 vote | -100% | -95% | -94% |
| C4 inverse loss | +156% | +1.9% | -9.1% (DD 20.5%) |
| C5 C1 + HAR | +34% | -9.3% | -9.4% (DD 15.5%) |
| C6 C3 + HAR | -100% | -92% | -92% |

- **Buy-and-hold in the same periods:** selection +1,792%, Look 1 +88.6%, forward -23.2%.
- **Why C3 and C6 are -100%:** the hourly vote includes the minute-reversal member, whose hourly sign is negatively
  correlated (-0.25) with the hour it votes on. A diagnostic minute-resolution vote also loses (-15% forward). It was
  not selectable.

## Deviations

These are gaps in the SPEC, filled before any look:
- each member's equal-risk scale is capped at 5;
- the reference sigma is the members' median;
- there is a 14-day warm-up fallback;
- C4 weights come from 2 bps net Sharpe;
- the cap is on the net position;
- M1 uses V_prev on the 15-minute head;
- per-minute cost accounting.

Full list in the hand-back.

**Figures:** `figs/{equity,members,corr,monthly}_{sel,look,fwd}.png` and `.json`.
