# 360-day training run: result (D-040, 2026-09-29)

## The run

- **Cell:** `20260929T115241Z-1a9e068-95fe35a6-default__f-2__s0`, spec `configs/scenarios/long_360d.yaml`, code at 1a9e068.
- **Launch:** from notebook 08's `launch()` at 11:52 UTC.
- **Training:**
  - 360.02 days of Bitcoin_BTCUSDT.csv, 2024-06-01 to 2025-05-28 (518,432 windows);
  - full shuffle, batch 2048, LR 1e-3, up to 40 epochs;
  - 30-day val and cal blocks;
  - scored on the **dev** block 2025-07-27 to 2025-08-28 (46,544 bars, 32 days);
  - fold -1, the newest 32 days, stays untouched as the test fold.
- **Timing:** EarlyStopping ended the run at epoch 12 and served epoch 6 (val loss 5.959).
  - 82 s per epoch, 0.32 s per step;
  - 996 s of training and 19 min in all, against the estimate of about 1 h for 40 epochs;
  - GPU at 73% utilisation, 10.7 of 12.3 GB.
- **Learning curve:** training loss fell slowly (6.43 to 6.03), while validation loss was flat from epoch 4, with no
  widening gap. The model saturates within about 1,500 updates at this batch size; it is not undertrained.

## Skill on the dev block (h0 / h1 / h2 = 10 / 15 / 20 bars)

| metric | 360-day run | same-block baseline | 7-day runs, reference dev folds (6 cells, other blocks) |
|---|---|---|---|
| direction AUC | 0.522 / 0.505 / 0.528 | logistic on lagged returns: 0.524 / 0.527 / 0.531 (model does not beat it; boot z -0.24 / -1.93 / -0.46) | mean 0.505 / 0.511 / 0.509, range 0.48-0.54 |
| variance CRPSS against constant variance | 0.047 / 0.040 / 0.045 (DM z +21 / +14 / +16) | - | mean 0.042 / 0.037 / 0.036 |
| var / err^2 Spearman | 0.340 / 0.335 / 0.337 | - | h1 mean 0.322 |
| 90% interval coverage | 0.907 / 0.908 / 0.908 | - | h1 mean 0.898 |
| served delta (beta) | 0 on every horizon (the price heads shrink to zero) | - | same |

Direction AUC against chance (boot z against the class prior):

- h0: +2.95
- h1: +0.68
- h2: +3.20

These edges are real but tiny, and a linear model on lagged returns matches them.

## Trading (default calibrated_quantile, 13 bps per side)

- 1,838 trades, net -99.0%, gross +1.5% (+0.88 bps per trade).
- Buy-and-hold on the same block: -5.1%.
- The strategy beats its size-matched random null: percentile 100 on net return and net Sharpe, 79 on gross return.

The NT-005 conclusion is unchanged: the edge per trade is about 1/30 of the round-trip cost.

## Reading (one run, one seed; the blocks differ from the 7-day runs, so these are indications, not verdicts)

- **360 days of data against 7 days changes little.**
  - Direction stays at a linear baseline's level.
  - The variance edge is slightly larger, within the spread across blocks.
  - Training saturates within a few thousand updates.
- **Data volume is not the main limit of this model on this target.** The signal is: 10-20 minute BTC direction
  after costs. That points at the owner's other two factors: noise at this horizon, and a target without P&L.
- **To make "more data does not help" a verdict** needs a learning curve on the same dev blocks: 7 / 28 / 112 / 360
  days, at least 3 seeds each. It is not run.

## Figures

Notebook 08 was executed after the run, and all three figures were looked at:

- `D:/nb_png_08/08_c4_0.png` (progress);
- `D:/nb_png_08/08_c4_1.png` (training dashboard): validation direction MCC and balanced accuracy stay inside the
  95% chance band on all horizons;
- `D:/nb_png_08/08_c4_2.png` (loss terms): direction BCE sits at ln 2.
