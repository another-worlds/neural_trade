# Strategy study v1: REPORT

Item NT-005 (owner request 2026-09-29). Pre-registered in [SPEC.md](SPEC.md) (committed e2b0b74, before any
scoring); configurations in [configs/strategy_studies/strategy_study_v1.yaml](../../../configs/strategy_studies/strategy_study_v1.yaml)
(bf03dbc); design from [the research note](../../../docs/research/2026-09-29-strategy-architectures/README.md).

## Verdict (by the SPEC's rules)

**No candidate passes the guard-rails. The recorded winner is `always_flat`: do not trade this model.** It is a
clear negative in the SPEC's sense:

- All 12 candidates lose money on the dev cells: every mean dev net return is below 0, so guard-rail 1 fails.
- For every discrete candidate, the 95% CI of the gross edge per trade lies below the 26 bps round-trip cost.

More tuning of the strategy layer will not reverse this. The model's forecasts do not carry an edge that
survives costs on this setup.

Measured on the reference setup: BTC/USDT 1-minute, 60-bar window, horizons 10/15/20. Costs are 13 bps per side
with next-open fills.

## Evidence

- **Cells:** the reference scenario, 9 cells: dev folds -3 and -2, test fold -1, seeds 0-2. Code at 82a848f,
  0.56 GPU-hours. Run directories:
  `runs/scenarios/reference_default/20260929T09*-82a848f-*-default__f*__s*/`.
- **Rescore:** `runs/scenarios/reference_default/rescore/strategy_study_v1-20260929T095605Z/`, code at
  6aa7aaf. It contains cells.csv (20 configurations x 9 cells), leaderboard.csv/.md, and
  study_analysis.json/.md from [analyze.py](analyze.py), which applies the SPEC's guard-rails. It also has
  gross_edge_bootstrap.json from [bootstrap_edge.py](bootstrap_edge.py). Every threshold was fitted on each
  cell's calibration block only.
- **Commands:**
  - `neural-trade scenario rescore configs/scenarios/reference.yaml --study configs/strategy_studies/strategy_study_v1.yaml`
  - `python runs/experiments/strategy_study_v1/analyze.py <rescore dir>`
  - `python runs/experiments/strategy_study_v1/bootstrap_edge.py <rescore dir>`

## Leaderboard

Rows are ranked by mean dev net Sharpe. The test columns are shown and never used (D-020). Buy-and-hold is on
the same blocks: dev folds -1.7% and -7.5%, test fold +4.1%.

Guard-rail codes:

- **G1:** net return > 0.
- **G2:** beats buy-and-hold, paired per cell.
- **G3:** at the null's 95th percentile in each dev fold.
- **G4:** drawdown.
- **G5:** at least 20 trades per cell.

Sharpe is annualised from 1-minute bars (525,600 per year), so the magnitudes are large. The standard error of a
dev Sharpe is about 6 for the two dev folds pooled (research note 4.4).

| config | dev Sharpe (sd) | dev net return | vs B&H | trades/cell | fails | test Sharpe | test return |
|---|---|---|---|---|---|---|---|
| nk[f=0.25] | -5.4 (7.0) | -0.35% | +4.23% | 4.8 | G1 G3 G5 | 0.0 | 0.00% |
| nk[f=0.5] | -5.8 (6.8) | -0.38% | +4.20% | 4.8 | G1 G3 G5 | 0.0 | 0.00% |
| ec | -6.8 (10.1) | -0.91% | +3.66% | 3.3 | G1 G3 G5 | 0.7 | +0.05% |
| *rs_ewma[q_out=0.8]* (twin) | -8.3 (5.9) | -3.08% | +1.50% | 6.5 | G1 G3 G5 | 5.9 | +3.42% |
| *vt_ewma[band=0.25]* (twin) | -8.4 (6.6) | -4.15% | +0.43% | 9.5 | G1 G3 G5 | 6.2 | +3.49% |
| *vt_ewma[band=0.1]* (twin) | -9.4 (5.1) | -4.43% | +0.15% | 27.0 | G1 G3 | 6.5 | +3.63% |
| vt[band=0.25] | -10.3 (4.7) | -5.35% | -0.78% | 19.3 | G1 G2 G3 G5 | 6.7 | +3.92% |
| vt[band=0.1] | -11.3 (4.9) | -5.81% | -1.23% | 44.7 | G1 G2 G3 | 6.4 | +3.64% |
| *rs_ewma[q_out=0.95]* (twin) | -15.9 (3.8) | -7.41% | -2.84% | 4.0 | all | 7.3 | +4.44% |
| gt[ma_cross, q=0.8] | -18.2 (13.0) | -5.17% | -0.59% | 24.7 | G1 G2 G3 | -3.4 | -0.42% |
| rs[q_out=0.95] | -28.8 (4.9) | -12.68% | -8.10% | 33.7 | G1 G2 G3 G4 | 4.8 | +2.81% |
| *gt_ewma[ma_cross, q=0.8]* (twin) | -30.3 (13.7) | -8.73% | -4.16% | 35.0 | G1 G2 G3 | -4.9 | -0.51% |
| gt[ma_cross, q=0.5] | -31.8 (4.4) | -11.35% | -6.77% | 49.2 | G1 G2 G3 G4 | -33.3 | -9.99% |
| *gt_ewma[ma_cross, q=0.5]* (twin) | -37.3 (13.2) | -13.35% | -8.77% | 57.0 | G1 G2 G3 G4 | -10.1 | -2.58% |
| gt[bollinger, q=0.8] | -50.4 (23.8) | -21.12% | -16.54% | 94.3 | G1 G2 G4 | -31.1 | -6.14% |
| *gt_ewma[bollinger, q=0.8]* (twin) | -53.2 (24.4) | -22.07% | -17.50% | 98.5 | G1 G2 G4 | -29.0 | -2.14% |
| rs[q_out=0.8] | -64.4 (29.8) | -22.31% | -17.73% | 90.2 | G1 G2 G3 G4 | -10.5 | -5.89% |
| gt[bollinger, q=0.5] | -75.5 (23.3) | -33.37% | -28.79% | 158.3 | G1 G2 G4 | -76.5 | -26.19% |
| *gt_ewma[bollinger, q=0.5]* (twin) | -80.7 (37.4) | -34.75% | -30.17% | 169.5 | G1 G2 G3 G4 | -43.3 | -11.52% |
| **cq (incumbent default)** | **-156.0 (33.3)** | **-62.37%** | -57.79% | 392.8 | G1 G2 G3 G4 | -115.3 | -42.87% |

The top candidate by dev Sharpe is nk[f=0.25]. Its Deflated Sharpe Ratio (K = 12) is about 0, because its
Sharpe is negative. It is not a winner: it fails G1, G3 and G5, and it barely trades.

## Gross edge per trade against the 26 bps round trip (NT-005 (b))

The discrete candidates are pooled over the 6 dev cells. The 95% CI comes from a block bootstrap:
(cell, entry_bar // 80) blocks, 5,000 resamples.

| config | trades | mean gross edge (bps) | 95% CI | CI below 26 bps |
|---|---|---|---|---|
| cq (incumbent) | 2,357 | +0.47 | -0.24 .. +1.17 | yes |
| gt[ma_cross, q=0.8] | 148 | +4.35 | -1.95 .. +11.17 | yes |
| gt[ma_cross, q=0.5] | 295 | +1.56 | -2.56 .. +6.03 | yes |
| gt[bollinger, q=0.8] | 566 | +0.16 | -2.66 .. +3.31 | yes |
| gt[bollinger, q=0.5] | 950 | -0.09 | -2.00 .. +1.78 | yes |
| rs[q_out=0.8] | 541 | -2.41 | -4.72 .. -0.08 | yes |
| rs[q_out=0.95] | 202 | -14.04 | -22.08 .. -5.71 | yes |
| ec | 20 | -1.58 | -24.49 .. +25.06 | yes (20 trades) |

The exposure rows (vt, nk) lost gross before costs as well. Their per-cell break-even cost ranges from
-232 to +18 bps; see cells.csv `breakeven_cost_bps`. They are long-biased or near-flat, and the dev blocks fell.

**Break-even cost.** No discrete candidate reaches a positive break-even cost near 26 bps. The largest mean
gross edge, gt[ma_cross, q=0.8] at +4.4 bps, would break even only at a round trip of about 4 bps, about one
sixth of the default costs.

## Model sigma against the model-free EWMA twin

These comparisons are descriptive. They are not verdicts under D-025.

- **Volatility targeting (vt):** the EWMA twin is better at both bands. Dev Sharpe is -8.4 against -10.3 at
  band 0.25, and -9.4 against -11.3 at band 0.10. The model rows also rebalance about twice as often (19 and
  45 against 10 and 27 per cell).
- **Regime stand-aside (rs):** the EWMA twin is better. Dev Sharpe is -8.3 against -64.4 at q_out 0.8, and
  -15.9 against -28.8 at q_out 0.95. The model's sigma crosses its calibration quantiles far more often: 90
  against 6.5 trades per cell at q_out 0.8. So its variance forecast is much noisier bar to bar than a
  half-life-60 EWMA.
- **Variance-gated TA (gt):** the model gate is slightly better in all four pairs, by 2.8 to 12.1 Sharpe
  points. This is within about 2 standard errors, and every row loses.

This agrees with the research note's warning (section 2.0): a free EWMA volatility forecast is at least as useful
as the model's variance head for trading. NT-078 (EWMA and HAR baselines in the evaluation report) tests that
on the forecasts themselves.

## What this says about the strategy structure

1. **The incumbent default `calibrated_quantile` is the worst of 20 rows.** It lost -62% per dev block and
   -43% on the test fold, with about 390 trades per 5-day block. Its gross edge is +0.47 bps per trade against
   a 26 bps cost. It beats its size-matched random null in most dev cells (percentiles 71 to 100), but not
   by the SPEC's rule: G3 fails on fold -2, where the seed mean is 88. So the calibrated P(up) may carry a
   sliver of timing information; its gross edge is about 1/50 of the cost.
2. **Turnover is what separates the rows.** The ranking follows trades per cell almost monotonically. The
   near-flat rows (nk, ec, rs_ewma) cannot be told apart within the noise, about 6 Sharpe points.
3. **Cost-aware sizing (net_edge_kelly) works as designed.** It stays flat unless the predicted edge exceeds
   the cost. It traded about 5 times per dev block and 0 times on the test block.

## Not done or limited

- Two dev folds x three seeds on 30 days of data. Seeds share the market path, so the dev Sharpe's standard
  error is about 6. The long history (NT-041) is the only way to make the ranking among the near-flat rows
  meaningful.
- Meta-labelling and daily TSMOM were deferred by the research note.
- No served-delta or raw-delta strategy was tested. NT-007 is untouched.
- The timing null of the exposure rows carries NT-081's small replay defect, about 1e-6 of equity. It cannot
  change a guard-rail here: every exposure row already fails G1.

## Recommendation for the owner (not applied)

The SPEC keeps `Strategies.default` unchanged until the owner confirms (OPERATING_MODEL "Escalate to the owner").
The default is also the strategy the experiment engine scores every scenario with, so it drives the
leaderboard's net Sharpe.

- **Recommended: `net_edge_kelly` (f = 0.25) as the default.** The reason is structural, not its dev rank:
  - it converts a model's edge into exposure only when the edge exceeds the round-trip cost;
  - so a model with no edge scores near 0, instead of the incumbent's -150 Sharpe that is pure cost drag;
  - a future model with a real edge (for example after NT-047's indicator families) shows up as a positive
    Sharpe.
- **Alternatives:**
  - keep `calibrated_quantile`: it measures turnover more than skill;
  - `always_flat` as the default: every scenario would score exactly 0, and the leaderboard could not rank.

For live use of this model, the answer is: do not trade it.
