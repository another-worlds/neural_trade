# Audit of the default network, for the rebuild (2026-10-09)

Read-only code audit (Plan agent, Opus; 42 tool calls, ~8 min; no code run). Owner's request: take the network apart and
rebuild it (PLAN 9). Paths relative to `src/neural_trade/` on branch nt-tactical.

## Bottom line

The 0.03-0.05 AUC gap to a logistic regression most likely comes from **feature representation and how the linear logit
is trained**, not from the GRU/attention trunk. H25: a jointly trained logistic readout that bypasses the trunk still
scores 0.527-0.534 against sklearn's 0.570 on the same slices. That readout never sees volatility-normalised last-bar
features, and it is trained unlike sklearn (random init, no standardisation, almost no L2, a short Adam run). The trunk's
gradient fight (direction gets ~10% of the gradient) and its per-window LayerNorm come next.

## 1. Forward graph (default: gru_attention, OHLCV [B,60,5], horizons 10/15/20)

- **Data:** windows `bars[i-60:i]`; targets `y_k = close[i+h_k-1] - last_close` (dollars); one StandardScaler over the
  pooled horizons. Window normaliser `window_relative`: OHLC `(x - last_close)/target_scale`, volume / train-mean volume.
  No per-window volatility scaling. Labels: `ret = y/last_close`, mask `|ret| > 5 bps`. Train shuffle buffer 2048, batch 256.
- **meta_adjust:** Dense(54, tanh) on avg+max pools of all 5 channels (594 params) shifts all 54 periods per window
  (unclipped, applied periods 1.6-74.7).
- **LearnableIndicators:** 54 logits, 14 families x 3 instances = 81 channels + raw close = 82 channels, **unnormalised**
  (per-channel std 0.015-358, max ~2.2e3). The trunk sees O/H/L/V only through the indicators.
- **Trunk:** BiGRU(64) 56,832; MHA(8x32) + LN 132,224; "time" MHA over the 60 steps + **LayerNorm over time per unit**
  31,284; 3 Conv1D(16, k 3/7/15) 51,248; EnergyGate + LN 41; positional encoding; 2 transformer blocks 10,880.
- **Side path:** context = avg pool; t_perp_proj Dense(16, tanh); VacuumSaturationNoise (training only); vacuum_overflow
  output (0 at inference); perp_magnitude, regime_gate -> variance heads only. 307 params.
- **Shared:** Flatten [960] -> Dense(32) 30,752; concat context -> [48].
- **Per horizon:** a tower Dense(16) **shared by price, direction and variance** (2,352); price Dense(1) clipped +-100;
  direction `sigmoid(Dense(1)(tower) + Dense(1, no bias, L2 1e-4)(skip8))`; variance softplus Dense(1) on tower +
  perp_magnitude + regime_gate.
- **DIRECTION_SKIP** (8 features): returns over 1/5/10/15/20/30 bars, last - first, log std; **not divided by the window's
  volatility**.
- **Total 316,751 parameters.** The direction head sees the indicators only through GRU -> LN -> LN over time -> conv ->
  gate -> LN -> transformer -> flatten -> Dense32 -> shared tower -> Dense1. The only linear route is the 8-return skip.

## 2. Loss terms (losses/functions.py:615-1048)

| # | term | weight / note | reads |
|---|---|---|---|
| 1 | point logcosh | calibrated | price |
| 2 | extended trend (pull to past delta) | 0.1, not rescaled | price |
| 3 | coherence | 1; two of its three parts have zero gradient but count in val_loss | price |
| 4 | direction masked BCE | calibrated | direction |
| 5 | dir-align | off | |
| 6 | Gaussian NLL | calibrated; floor 1e-4 has zero gradient below it | price + var |
| 7 | vol | effective 0.01 (0 lifted to 0.1 by CALIB_VOL_ZERO_TO_FLOOR, x0.1) | price h1 |
| 8 | CRPS | calibrated, clipped [0,100] | price + var |
| 9 | soft ECE | 0, computed every step | |
| 10 | t_perp | 0.1; gradient ~ 1/mean(v) | var |
| 11 | casimir | 0.1 | var |
| 12 | vac | dead (threshold 0) | |
| 13 | HD | 0.1; std of window LEVELS, not returns | var |
| 14 | IFE | 0.1; pushes horizons to decorrelate | price |
| 15 | vac overflow | 0.1, train only | trunk |
| 16 | inter_reg | ~1e-5 ||w_skip||^2 | skip |

Lambda calibration equalises loss **values**, not gradients (NLL and t_perp gradients scale like 1/v). The direction head's
Dense layers get only the BCE; the shared tower and the trunk get every term. H16 probe: trunk NLL 31%, t_perp 27%,
direction 11%; indicators NLL 30%, t_perp 28%, direction 10%; point, coherence, hd, trend, IFE anti-aligned (cos down to -0.98).

## 3. Training mechanics

Adam 1e-3 (indicator logits Adam 5e-3, never decays); a non-finite gradient zeroes the whole step; global-norm clip 20 per
group; ReduceLROnPlateau and EarlyStopping on **val_loss = the composite total** (direction is a small flat part, so the
served epoch is chosen by NLL, CRPS and point); post-hoc temperature, delta shrinkage, conformal (none changes AUC).
Warm-start hazard: an existing MODEL_PATH skips training unless forced. Screen trials have no ES/ReduceLR and score the last epoch.

## 4. Chaos list, ranked by how likely each explains the gap

1. **C1. No volatility-normalised last-bar feature reaches a linear readout.** The skip's returns are not divided by sd;
   the tactical INDICATOR_SKIP z-scores each channel by its own window mean and std, which removes oscillator levels and
   makes (c - EMA) unrepresentable.
2. **C2. Level-destroying normalisation in the trunk:** LayerNorm over time per GRU unit (gru_attention.py:237-240) plus per-step LNs.
3. **C3. Unnormalised indicator channels** (std 0.015-358; CCI up to ~+-1000) straight into the GRU: gates saturate.
4. **C4. The linear logit's training:** glorot init, no standardisation, L2 ~1e-5, 8 epochs x 65 Adam steps vs a convex
   optimum; regime-local batches (shuffle buffer 2048 = ~1.4 days).
5. **C5. Gradient budget and the shared tower:** direction ~10%; NLL + t_perp ~58%; the tower is shared by all heads.
6. **C6. Model selection on the composite val_loss**, not on direction.
7. **C7. Capacity vs data:** 316,751 parameters vs ~1-2.5k effective h1 labels; regularisation dropout 0.1 only.
8. **C8. Conflicting price-path objectives:** point vs vol vs extended trend vs IFE vs coherence; delta shrinkage undoes it.
9. **C9. Indicator numerics:** RSI 0/100 edge cases, BB %b blow-up early in the window, soft signs in target-std units, OBV tails.
10. **C10. Adaptive-period machinery:** meta_adjust shifts every period from pools that include volume spikes; learned = frozen anyway.
11. **C11. Variance-path oddities:** training-only noise, dead output at inference, 1/v gradients, zero-gradient floor and clip.
12. **C12. Dead computation active by default:** soft ECE, vac, gauss_p_up, retired trend terms, `0*reg_loss`, no-op clips,
    deprecated fields, calibration measuring terms it never rescales.

Measurement caveat: 0.570 vs 0.521 rests on 3 check slices (n_eff ~840 each, about +-0.02 per slice); judge rebuild steps
on 20-40 slices.

## 5. Rebuild ladder (each step must not lose AUC to the previous one; select on the head's own val BCE)

- **L0** sklearn logistic regression on the textbook features: the bar.
- **L1** the same regression in TF: standardised features, Dense(1) zero init, matched L2, full batch to convergence,
  masked BCE. Must equal L0 within 0.003; a failure blames training mechanics (C4).
- **L2** the features computed in-graph with fixed periods (parity test < 1e-4), then the soft/EWMA versions (expect +-0.005).
- **L3** three horizons, three independent logistic heads.
- **L4** learnable periods, direction loss only: must not lose (expected = L3, H25).
- **L5** a wider static vector of all families in scale-free form, stronger L2.
- **L6** a residual MLP on top, zero-init output: does nonlinearity add anything?
- **L7** a sequence encoder (GRU(32) on train-standardised, scale-free channels; no per-window z-score, no LN over time):
  if it does not beat L6, the window has no sequential direction information beyond L5.
- **L8** other heads on their own towers (variance with a stop-gradient into the encoder; price possibly dropped, H14),
  then physics terms one at a time, each improving its own metric and leaving direction AUC within +-0.003.
