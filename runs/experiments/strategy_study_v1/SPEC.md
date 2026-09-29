# Strategy study v1: SPEC (pre-registered)

Item: NT-005 (owner request 2026-09-29: "Evaluate the trading strategy structure. Research best trading
strategy architectures targeting our model data. Test and integrate the strategies, pick the best one.").
Design source: [docs/research/2026-09-29-strategy-architectures/README.md](../../../docs/research/2026-09-29-strategy-architectures/README.md)
sections 3 and 4. Written and committed by the lead **before any dev fold is scored**; nothing below changes
after results.

## Kind

A **sweep** over strategy configurations (OPERATING_MODEL "Sweeps and pre-registered studies"): it picks a
winner; it is not an "A beats B" verdict (D-025). Model-against-EWMA comparisons are reported descriptively,
not as verdicts.

## Data

- The reference scenario `configs/scenarios/reference.yaml` (default model, folds -3 and -2 = **dev**, fold
  -1 = **test**, seeds 0-2: 9 cells), run once on the GPU with NT-076's stored predictions (about 45
  GPU-minutes, an estimate; under the 3-hour item limit).
- Strategies are scored on CPU by `neural-trade scenario rescore` (NT-076) from each cell's stored
  predictions: every knob and threshold is fitted on the cell's **calibration block** only; the backtest runs
  on the cell's out-of-sample block with the default costs (13 bps per side, next-open fills, stops on
  high/low), 100 random-null seeds.
- Test-fold numbers are shown in the leaderboard and this study's REPORT and **never** rank or choose (D-020).

## Candidates (K = 12 model-driven configurations)

Strategy names are NT-077's. `sigma_source: model` unless stated. h2 = the 20-bar horizon.

| # | id | strategy | knobs | engine mode |
|---|---|---|---|---|
| 1-2 | VT | vol_target | band in {0.10, 0.25} | exposure, decide every 60 bars |
| 3-4 | RS | vol_regime_long | q_out in {0.80, 0.95}, q_in = q_out - 0.10 | discrete, long only |
| 5-6 | NK | net_edge_kelly | f in {0.25, 0.5}; cost 0.0026; band 0.10 to the band's edge | exposure, decide every 20 bars |
| 7 | EC | edge_over_cost | cost 0.0026, hold 20 | discrete |
| 8-11 | GT | gated_ta | primary in {ma_cross, bollinger} x q in {0.5, 0.8} | discrete |
| 12 | CQ | calibrated_quantile | defaults (the incumbent, `Strategies.default`) | discrete |

**Twins (8, model-free baselines, not candidates for the winner):** VT, RS and GT with `sigma_source: ewma`
(EWMA of squared 1-bar log returns, half-life 60 bars). **Baselines:** buy_and_hold, always_flat (in every
backtest's baselines), and each row's null (size-matched random null for discrete rows, circular-shift timing
null for exposure rows).

No candidate reads the price-delta heads, so NT-007 (served against raw delta, the owner's decision) does not
affect this study.

## Ranking

Per configuration, the mean **net Sharpe after costs** over the 6 dev cells (2 dev folds x 3 seeds).

## Guard-rails (a row is disqualified if any fails; all on dev cells only)

1. **Beats always-flat:** mean dev net total return > 0.
2. **Beats buy-and-hold, paired per cell:** mean over dev cells of (row net return - buy-and-hold net return
   on the same cell) > 0.
3. **Beats its null:** in each dev fold, the seed-mean of `percentile_sharpe_net` (the row's percentile among
   its null's seeds) >= 95.
4. **Drawdown:** long-biased rows (VT, RS): mean dev max drawdown <= mean dev buy-and-hold max drawdown;
   other rows: mean dev max drawdown <= 10%.
5. **Activity:** mean dev `n_trades` (trades, or rebalances for exposure rows) >= 20 per cell.

## The winner

- The top row by mean dev net Sharpe **among the 12 candidates that pass every guard-rail**.
- It is reported with: its per-cell dev Sharpes, a Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014)
  with K = 12 and, conservatively, K = 20 (with the twins), T = the pooled dev bars, Gaussian per-bar returns
  assumed (an approximation, stated); its break-even round-trip cost; its EWMA twin where it has one; its test
  columns (shown once, not used). A winner with DSR < 0.95 is labelled "top of the leaderboard, not
  distinguishable from selection luck".
- **If no candidate passes**, the recorded winner is `always_flat` ("do not trade this model"); buy-and-hold is
  the benchmark, never the model's winner. The REPORT then takes NT-005's negative-result form (b): per row, the
  mean gross edge per trade (discrete) or the break-even cost (exposure) in bps against the 26 bps round trip,
  with the spread over the dev cells.
- A twin that passes while its model row does not is reported as a finding (a model-free rule works), not as
  the winner.

## Integration

The new strategies stay registered whatever the result. Changing `Strategies.default` (today calibrated_quantile,
D-009) changes default trading behaviour: the REPORT recommends and the owner confirms before the default
changes (OPERATING_MODEL "Escalate to the owner").

## Budget

GPU: one reference-scenario run, about 45 GPU-minutes (estimate; 9 cells at about 5 minutes). CPU: the rescore.
