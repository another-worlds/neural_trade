# T08 report: trade on high confidence; a conservative consensus ensemble

The lead wrote this from the implementer's hand-back (2026-10-13).
- **Owner:** "Подними метрику уверенности и исходя из нее делай решение о сделке. Также сделай ансамбль, который будет
  консервативно торговать только на общем согласии".
- **SPEC:** [SPEC.md](SPEC.md) (47006ae, before results).
- **Commits:** selection.json in 392ee5c, before any look; results, figures and tests in 9b19fe0 and dd96334 (25 tests passed).

## Verdict: both finalists FAIL the forward test at 2 bps

| Finalist | Rule | Forward return at 2 bps | Hit | Mean bps/trade [95% week bootstrap] | Max DD | Against the random-sign null |
|---|---|---|---|---|---|---|
| A | M2 (H62 recipe) alone, top 10% confidence | -13.7% (435 trades) | 50.6% | -2.9 [-12.8, +6.7] | 21.9% | at its median |
| B | consensus: the 3 models agree, confidence >= 0.80 | -6.4% (222 trades) | 48.6% | -2.4 [-12.9, +8.8] | 13.1% | at its median |

- **The market:** BTC fell 23.2% over the same months, with a 47% drawdown.
- **At 0 bps** both finalists still lose: A -5.9%, B -2.2%.
- **Owner's goal:** hit above 60% was not reached. Drawdown below 5% needs 0.22-0.37x leverage, and the return there is negative.

## The key question: does confidence sort the trades?

Hit by confidence decile, from lowest to highest decile:

| Rule | Selection (2019-2024) | Forward |
|---|---|---|
| K1 consensus | 0.53 → **0.68** | 0.52 / 0.52 / 0.52 / 0.54 / 0.55 / 0.56 / 0.53 / 0.51 / 0.55 / **0.48** |
| M2 | 0.50 → 0.61 | 0.50 → 0.55, inside chance |

Figure: `figs/hit_by_decile.png`.

- **In selection, yes:** the higher the confidence, the higher the hit rate. Partly in-sample: M2's recipe was chosen on these years.
- **Forward, no:** the ordering disappears. The top-confidence decile is no better than the rest.

So raising the threshold cannot help: the most confident trades are not better today. Fewer trades only shrank the loss
by shrinking exposure.
- The strictest consensus lost more per trade: -34.6 bps over 21 trades.
- **Unanimity of all 6 never trades.** The trend and breakout rules point against the model members' reversion signals.

## Selection, 2019-01..2024-06, at 2 bps

| Family | Finalist | Calmar | Return | Hit | Trades/yr |
|---|---|---|---|---|---|
| A | M2, top 10% | 2.48 | +17.7x | 0.595 | 474 |
| B | K1, confidence >= 0.80 | 1.03 | +3.5x | 0.617 | 188 |

- **Buy-and-hold:** +17.9x, Calmar 0.92.
- **K1 at q 0.95:** hit 0.69, but only 13.6 trades a year (excluded by the 30/yr rule).

## Look 1 (2024-07..2025-07-20, descriptive, partly seen), at 2 bps

| Finalist | Return | Mean bps/trade [95%] |
|---|---|---|
| A | +37.2% | +7.4 [-1.0, +16.4] |
| B | +12.7% | +8.1 [-6.9, +26.1] |

Buy-and-hold: +88.6%.

## Not chosen, forward positive (one-year outcomes, not evidence)

| Configuration | Forward return |
|---|---|
| M1 (minute reversal) top 10% | +18.8% |
| M4 trend, top 2% | +17.0% (80 trades) |

## Deviations

1. The hourly M1 score is taken from the anchor at hh:00 (T07's cache, verified bit-identical on two months).
2. Confidence is undefined with fewer than 30 values in the previous month.
3. Costs are charged per trade, c/2 at entry and c/2 at exit, with no netting.
4. A q-free consensus confidence was used for the B deciles: min c for K1 and K3, the third-highest for K2.
5. A module was renamed after the selection commit (no logic change, tables identical).
6. A test once overwrote T02 result files in the T08 worktree. They were restored there and never committed; the merged
   tree shows no T02 change.
