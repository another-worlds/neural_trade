# PnL optimisation: design spec (judge synthesis of three independent designs)

Source of truth for research/pnl/. Produced by a design panel (protocol, network, strategy angles) and a judge.

## title

BTCUSDT USD-M perp PnL study: one exact 1-min fill/fee/funding simulator, a cost-aware no-trade-band PnL network, neural_trade fixed and measured, annual expanding walk-forward 2021-2025, 4 + 3 pre-registered candidates in two separate horizon tracks, and one frozen 12-month seal scored once

## data_and_features

STORE (exists, commit 576eaae)
- research/pnl/data.py reads $PNL_DATA/btcusdt_perp_1m.npz (3,591,360 one-minute bars, 2020-01-01..2026-10-30, UTC) and funding_full.npz (7,395 prints).
- Dev: 2020-01-01..2025-10-31, 3,068,640 bars. Seal: 2025-11-01..2026-10-30, 363 bar-days.
- Flags: 2026-10-28 is missing (gap_before). The 2026-10 funding prints are rebuilt from the premium index (funding_estimated; MAE 0.02 bps per print).
- SEEN_WINDOWS in data.py: 2025-10-11..2025-11-10 07:30, and 2026-09-20..2026-10-30. The first one also covers the last 3 weeks of dev.

SEAL GUARD (first change; nothing is modelled before it lands)
- Signature: load(minutes=1, seal=False).
- With seal=False, every array is truncated to t < SEAL_START.
- seal=True requires env PNL_UNSEAL == SHA-256 of research/pnl/prereg.json at HEAD, and appends a timestamped line to research/pnl/results/seal_access.log.
- Feature, fold and scaler code only ever receives truncated arrays.
- At unsealing, features are recomputed on the full arrays, and run_seal.py asserts that every dev row is bit-identical to the cached dev features. This is a second causality proof.

CONVENTIONS
- Bar t covers [open_t, open_t + 60 s).
- A decision at the close of bar t uses bars <= t. The earliest fill is in bar t+1.
- Funding: data.py maps each print to the first bar with open >= the print time. The position held at the end of the previous bar pays it; a long pays a positive rate.
- Hourly decisions fall at the hh:59 bar close; daily decisions at the 23:59 close.

BASE SERIES AND OPERATORS (float64)
- r_t = ln(c_t/c_{t-1}); rho_t = ln(h_t/l_t); taker imbalance iota_t = (2 tbb_t - v_t)/max(v_t, 1e-9).
- E_s[x]: adjusted causal EWMA with alpha = 2/(s+1). Numerator and denominator both go through scipy.signal.lfilter.
- sigma60_t = sqrt(E_60[r^2]); sigma_t = sqrt(E_1440[r^2]), floored at 0.5 bps per minute.
- Rolling max/min and every rank or percentile use trailing windows that exclude t+1.
- The first 31 days (2020-01) are warm-up and are never used as a training or evaluation row.

F-CORE: 48 columns, every minute, float32 (design 2's table)
- A. Returns (12): z_L = ln(c_t/c_{t-L})/(sigma_t sqrt L), L in {1, 3, 5, 15, 30, 60, 120, 240, 480, 1440, 4320, 10080}.
- B. Volatility (7):
  - ln sigma_s in bps per minute, s in {15, 60, 240, 1440, 10080}. These are absolute levels on purpose: cost is fixed in bps.
  - Parkinson/close ratio ln(sqrt(E_s[rho^2]/(4 ln 2))/sigma_s), s in {60, 1440}.
- C. Path shape (7):
  - efficiency ratio |E_s r|/E_s|r|, s in {60, 1440};
  - variance ratio ln(E_240[r15^2]/(15 E_240[r^2]));
  - distance to the 1-day and 7-day high and low, in sigma_t sqrt(L) units.
- D. Activity (5): ln(E_s v/E_43200 v) for s in {5, 60, 1440}; ln(E_60 n/E_43200 n); ln of average trade size against its 30-day value.
- E. Taker imbalance (5): E_s[2 tbb - v]/E_s[v], s in {5, 15, 60, 240, 1440}.
- F. Funding (3): last settled print (bps), mean of the last 3 prints, mean of the last 90 prints.
- G. Calendar (9): UTC hour sin/cos at harmonics 1, 2 and 3 (harmonic 3 is the 8-h funding cycle), day-of-week sin/cos, weekend flag.

F-DAILY: 7 columns, from completed UTC days, forward-filled (Track 2 only)
- z_L = ln(C_d/C_{d-L})/(sigma_d sqrt L) for L in {14, 28, 56} days, with sigma_d = sqrt(EWMA-30 of daily r^2).
- Drawdown from the 28-day and 56-day max close, in sigma_d sqrt L units.
- Distance to the 50-day SMA in sigma_d sqrt 50 units.
- ln(0.5/sigma_ann), with sigma_ann = sqrt(365) sigma_d.
- Warm-up: 56 days. Track 2 training rows start on 2020-02-26.

TRACK SUBSETS
- Track 1: 41 intraday columns. Drop z_4320, z_10080, ln sigma_10080, the 7-day high/low distances and the 90-print funding mean. Signal lookback is at most 1 day.
- Track 2 (cadence >= 60 min): 48 + 7 = 55 columns.

OPTIONAL (contingent only; see network)
- research/timestral_lig/indicators.py Bank, 'seq' mode, on 8,192-bar chunks with the first 2,048 bars of each chunk discarded, then normalise().
- Feature 50 ('frame') is dropped: it depends on window length.
- 61 features.

NORMALISATION
- Per fold and per feature: median/IQR fitted on that fold's training rows only, then clip to +-5. NaN becomes 0, and the count of NaNs after warm-up is asserted to be 0.
- Ridge uses the same scaler. Scalers are saved per fold.

TARGETS (gathered from cumulative arrays of log close and funding)
- R_k = c(t_k + h)/c(t_k) - 1, using the decision close.
- F_k = funding charged in (t_k, t_k + h].
- sigma_hat_k = sigma_t sqrt h, for winsorising and scaling.

STORAGE
- research/pnl/work/features_v1.npy: dev 3.07M x 55, float32, about 0.68 GB, memory-mapped.
- features_v1.json: column names, groups, the data SHA-256 and the code commit.
- work/ is gitignored.

CHECKS (checks.py; all must pass before any scoring, otherwise kill K0)
1. Causality: for 1,000 random t0, overwrite bars > t0 with noise and with a permutation. X[:t0+1] must be bit-identical.
2. Funding alignment on hand-built prints: a print at 08:00 is charged only to the position held over the 07:59 bar.
3. The fold manifest asserts that training, inner-validation and test index ranges are disjoint and purged.
4. Deliberate leak: shift the features by -1 bar. T1-LIN must then show a large positive PnL, which proves the harness detects leaks.
5. Labels shuffled within each day: gross PnL about 0 +- 2 SE.
6. Calling load() without seal=True can never return a bar with t >= SEAL_START.

## simulator

research/pnl/sim.py (numpy, float64) is the ONLY path to reported PnL. Every candidate, baseline and neural_trade goes through it. It always runs on 1-min bars, whatever the decision cadence.

ENGINE E: exact, event-driven; every reported number comes from it

State
- Equity E (starts at 1.0, compounding) and contracts q (BTC, signed).
- Exposure x = q c/E, which drifts exactly with price between fills.
- At most one working order.

Decision at the close of bar t
1. The candidate gives a target x* in [-1, 1]. For continuous policies x* = clip(x_drifted, tau - b, tau + b); discrete rules give it directly.
2. dq = (x* - x) E/c_t.
3. The change is split into a REDUCE part (toward 0, including the closing leg of a flip) and an INCREASE part.
4. A new decision cancels or replaces any working order.

Taker fill
- At o_{t+1}, fee 5 bps on filled notional.
- Base slippage 0 (the spread is 0.1 USDT, about 0.01 bps).

Maker fill
- Limit at L = c_t.
- It fills at L in the first bar j in [t+1, t+K] with l_j <= L - tick (buy) or h_j >= L + tick (sell). This is a strict trade-through; tick = 0.1 USDT. Fee 2 bps, all-or-none.
- K = 1 bar for decision cadence h <= 5, 3 bars for h = 15, 15 bars for every Track 2 order. A working order also ends at the next decision.
- If unfilled: a Track 1 entry is cancelled (the policy is told and generates no exit). A Track 2 order and every chased exit go as taker at o_{t+K+1}.

Accounting
- Between fills E_t = E_fill + q (c_t - P_fill).
- 1x cap: if |x| > 1.02 at a decision, or > 1.10 at any bar close, trim to 1.0 as taker at the next open. Every trim is logged.
- Funding: at a bar carrying rate f, E -= f q c_{prev}. Estimated 2026-10 prints are used as is. Sensitivity: replace them with the last real print.

Gaps
- No order is placed or filled inside a gap. The position is carried and marked at the first bar after it.
- Signals whose lookback spans the gap are suppressed for 60 bars.

Fold boundaries
- Positions are never force-closed at a fold switch.
- They are closed as taker only at the end of dev OOS and at the end of the seal.

Liquidity flag
- With $100k base equity, any order above 5% of that bar's quote volume is counted and reported.

SCENARIOS (per-side fee x traded notional)

Fee scenarios
- C7, primary: INCREASE as maker (fill model above); REDUCE as taker at the next open. A flip is a taker close plus a maker open.
- C4: both legs maker. An exit chases to taker after K bars (Track 1 time exits: posted at the exit decision, chased after K).
- C10: both legs taker at the next open.

Fill scenarios
- B, base: as above.
- S, stress: trade-through by 1 bps instead of 1 tick; +1 bar latency (decide at t, act from t+2); +1 bps on every taker fill.
- N, nominal: all fills at the decision close at nominal fees. Used for reconciliation only.

The primary scenario is C7-B. The decision is taken there, never at the cheapest scenario. C7-S is gate G5. All six C x {B, S} cells are reported for every candidate and baseline.

EXPECTED EFFECTIVE COSTS (measured on dev by strat_probe/fill.py, random-direction orders)
- Maker at 1 tick: fills 87% in 1 bar and 97% in 15 bars. 2.8-3.1 bps per side including fallback; fill markout -0.7 bps.
- At 1 bps trade-through: 4.0-4.3 bps per side.
- C7 round trip: about 7.9 bps (B) and 9.2 bps (S).
- No extra adverse-selection charge is applied. The trade-through rule already conditions on adverse moves; a separate charge would double-count.

ENGINE F: flat cost, vectorised; used for training, inner CV and nulls
- Costs at the decision close: c_in = 3 bps per unit of |x| increase (2 bps fee + 1 bps fill allowance), c_out = 5 bps per unit decrease.
- inc = (|x_k| + |dx| - |x_{k-1}|)/2 and red = (|x_{k-1}| + |dx| - |x_k|)/2, exact including flips.
- net_k = x_k (R_k - F_k) - (c_in inc_k + c_out red_k), on close-to-close returns.
- For nulls, the same formula is applied on the 1-min grid to the realised engine-E exposure.

OUTPUTS PER RUN
- Per-bar equity and exposure; daily (UTC) log equity returns.
- Trade log: decision time, fill time and type, price, fee, fill delay, gross / fees / funding / net bps, MAE.
- Maker fill rate, chase rate, implementation shortfall against the decision close, maker markouts at +1/+5/+15/+60 min.
- Attribution: market PnL, fees, funding.
- Static-versus-timing split: static = mean(x) x B&H daily PnL; timing = Σ(x_t - mean x) r_t.
- Beta and alpha t from a daily regression on BTC.
- Turnover sum|dx| per year, trades per year, mean and median hold, mean |x|, mean x, % time in market.
- Liquidity flags.

TESTS (research/pnl/tests/test_sim.py)
1. A zero position gives exactly 0, with no funding paid.
2. B&H equals Π(1+r) minus one entry fee minus funding, to 1e-9. B&H funding equals the sum of prints.
3. Fee totals equal turnover x fee, to 1e-9.
4. A maker buy never fills when every following low is >= L - tick.
5. Perturbing bars after t+K never changes a fill of an order placed at t; perturbing bars <= t never changes a fill.
6. Under C7-N a flip from +1 to -1 costs 5 + 2 bps.
7. A long with a positive funding rate loses money; a short gains.
8. A drifted short is trimmed when it breaches the cap.
9. Engine E under N equals engine F on the same positions to < 0.1 bps per day. This bounds the gap between training and simulation.
10. Random positions with matched turnover give about -turnover x cost (within 2 SE).
11. Perfect foresight gives a large positive result.
12. Reconciliation on 2021 for B&H, T2-P and the plain fade: |E(C10-N) - F(10 bps)| < 0.5% of equity per year.

## walk_forward

research/pnl/folds.py. One schedule for every candidate and baseline in both tracks.

FOLDS (expanding window; annual test blocks; the model is frozen inside each test block)
- Y2021: training from 2020-02-01 (Track 2: 2020-02-26) to 2020-12-31; test 2021-01-01..2021-12-31.
- Y2022: training to 2021-12-31; test 2022.
- Y2023: training to 2022-12-31; test 2023.
- Y2024: training to 2023-12-31; test 2024.
- Y2025: training to 2024-12-31; test 2025-01-01..2025-10-31.
- S (seal): training on all dev data to 2025-10-31; test 2025-11-01..2026-10-30, frozen for 12 months.
  - Every fold-S model is trained BEFORE pre-registration, and its weight hashes are pre-registered.
  - The seal run only does feature computation, inference, simulation and statistics.
  - S is exactly the dev procedure (one annual refit), so dev OOS predicts the seal in expectation.

PURGE
- A training row k needs t_k + h + 1 bar < the start of the next block (inner validation or test). This is >= 1 day for daily labels.
- No embargo is needed after a test block, because training never uses later data.
- Seal labels are unreachable (load truncates), so the last fold-S rows purge automatically.

INNER SELECTION (inside each fold only)
- Networks: inner validation is the last 20% of the training window (at least 120 days, at most 365 days), with a 1-day purge on both sides.
  - It is used for early stopping and for choosing the cell within the candidate.
  - There is no refit on training + validation.
  - Y2021 example: training 2020-02-01..09-01, validation 2020-09-03..12-30.
- Ridge: purged, blocked 5-fold CV on the training window (contiguous blocks, purge H+1 bars, 1-day embargo).
- neural_trade: trailing 12 months before the test start, minus a 1-day purge. Its last 15% is internal validation, purged 15 bars.
- Every percentile or rank threshold is trailing. Nothing is fitted on a test block.

DEV OOS
- One contiguous engine-E run per strategy, 2021-01-01..2025-10-31 (1,765 days, 4.83 years).
- The model switches at the first decision on or after 1 January. The position carries across, so switch turnover is charged exactly.
- These 1,765 days are the ONLY data used for gates and outer selection.
- Also reported (descriptive):
  - 2022-01..2025-10 (fold Y2021 has thin training data);
  - dev without the seen days 2025-10-11..10-31;
  - B&H and T2-P with the OOS start shifted month by month through 2021.

WHY ANNUAL
- A frozen 12-month seal fold then follows the same procedure as dev.
- Network compute is about 4x lower than with quarterly folds, which pays for 5-seed ensembles.
- Refit frequency matters less than ensemble size for a noisy PnL-trained policy.

DISCIPLINE
- Every dev run, including reruns after bug fixes, is appended to research/pnl/results/runs.jsonl (committed) with its config hash and OOS t. It enters N for the Deflated Sharpe.
- At most 2 full reruns after the first full results, and only for documented bugs (research/pnl/CHANGES.log).
- Nothing is tuned against dev OOS.
- A late candidate must be declared; it raises K, and p_sel is recomputed over all K.

## network

(A) PNL BAND NET (research/pnl/net.py and train.py; PyTorch 2.15 CPU via `uv run --offline --no-project --with torch --with numpy --with scipy`)
One model per (track, cadence h, config). Workers are single-thread (torch.set_num_threads(1)) and 4 run in parallel.

INPUT
- x_k = F-core intraday (41 columns, Track 1) or F-core + F-daily (55 columns, Track 2) at decision time t_k.
- There is no position input. The band carries the state.

ARCHITECTURES
- LIN (about 130-170 parameters): three linear heads on x.
- MLP (about 5-6k parameters): Linear(F, 64) - GELU - Dropout 0.1 - Linear(64, 32) - GELU, then three linear heads.
- There is deliberately no GRU, TCN or attention: per-step PnL signal-to-noise is about 0.01, and no direction skill has been found at these horizons.

HEADS
- tau_k = tanh(a_k), the target exposure.
- b_k = 0.02 + 0.98 sigmoid(beta_k), the no-trade half-width.
- aux_k: sigma_hat_k^2 = exp(aux_k) sigma_t^2 h. It starts at the EWMA forecast.

BAND LAYER (torch.autograd.Function)
- Forward: p_k = clip(p_{k-1}, tau_k - b_k, tau_k + b_k), then clip to [-1, 1]. It runs as a numpy loop over time, vectorised over the batch, and stores a branch code per step.
- Backward (reverse scan):
  - inside the band: the gradient carries to p_{k-1};
  - low edge: d tau += g, d b -= g;
  - high edge: d tau += g, d b += g.
- Checked by torch.autograd.gradcheck in float64 away from ties.
- Benchmarked (scratchpad pnl_design/bench.py): 0.5-0.9M rows/s single thread for the forward and backward pass.
- Learning b(x) is the only route by which the known volatility skill can reach PnL: the band widens when sigma sqrt(h) is small relative to cost.

LOSS (engine F, exact inc/red split)
- pnl_k = p_k (Rw_k - rbar h - F_k) - kappa (c_in inc_k + c_out red_k), with c_in = 3e-4 and c_out = 5e-4.
  - Rw_k = clip(R_k, +-6 sigma_hat_k).
  - rbar is the training-window mean 1-min return. DE-MEANING stops the net from earning by learning the in-fold drift (about +45%/yr at x = 1 in 2020-21). Static exposure is the B&H and vol-target baselines' job.
  - F_k is the realised funding: a cost, not an input.
- Daily sums Pi_d = Σ pnl_k over consecutive 24-h blocks of the loss region.
- L = -(1/s_d)[mean(Pi_d) - (gamma/2) mean(Pi_d^2)] + 0.05 QLIKE(aux) + weight decay.
  - gamma = 2 per unit of daily return, fixed (about half-Kelly; the optimum is p* = mu_d/(gamma sigma_d^2)).
  - s_d is the training SD of daily returns (conditioning only).
  - QLIKE = ln sigma_hat^2 + R^2/sigma_hat^2. It anchors the trunk to the one predictable quantity. Its validation value must beat EWMA, which is a pipeline check.
- With no edge the optimum is p = 0, so a collapse to cash is a valid result.
- Why not a differentiable Sharpe: its minibatch gradient is biased and degenerate near the flat policy.
- Why not accuracy: payoff asymmetry and cost (B6).

SEGMENTS
- Contiguous sequences with burn-in W (no loss), then T loss steps, starting at p_0 = 0.
- Random start; random phase in [0, h) for h > 1.
- Geometry (W, T, batch B):
  - h = 1: (240, 1440 = 1 day, 64)
  - h = 5: (96, 576 = 2 days, 64)
  - h = 15: (64, 480 = 5 days, 64)
  - h = 60: (48, 336 = 14 days, 64)
  - h = 240: (24, 180 = 30 days, 64)
  - h = 1440: (14, 60 = 60 days, 32)
- Epoch = min(500k, N_train) loss steps for h <= 15, and 100k for h >= 60.
- Maximum epochs 30 (h <= 15) or 40 (h >= 60); minimum 3.

OPTIMISER AND REGULARISATION
- AdamW: lr 1e-3 (MLP) or 3e-3 (LIN); weight decay 1e-3; cosine decay to 0.1x; gradient-norm clip 1.0.
- Dropout 0.1 (MLP); Gaussian input noise SD 0.05.
- Feature-group dropout: each group A-G or daily is zeroed per sample with p = 0.1.
- kappa in {1, 2}: cost inflation as a turnover regulariser. It is never used in evaluation.

INITIALISATION AND WARM-UP
- tau and b head weights ~ N(0, 1e-3); tau bias 0; b set to 0.05 at init; aux bias 0.
- Epochs 1-2 run with the band off (p = tau), so a flat start does not freeze the gradient.
- Pre-declared fallback: if in more than 1 of 5 seeds the band collapses (validation b > 0.9 with zero trades) while the warm-up epoch traded, switch to a soft-threshold band with temperature annealing.

EARLY STOPPING
- Every epoch, compute U = mean(Pi_d) - (gamma/2) mean(Pi_d^2) on the contiguous inner-validation chain.
- U uses kappa = 1, the hard band, and returns de-meaned by the validation-block mean, so U measures timing.
- Patience 5; restore the best epoch.

ENSEMBLE
- 5 seeds (initialisation, sampling, dropout).
- tau and b are averaged across seeds at each decision, then the hard band is applied once.
- The single-seed spread is reported.

IN-FOLD CELL CHOICE
- Take the cell with the highest ensemble inner-validation U.
- If the best cell is an MLP and the LIN at the same (h, kappa) reaches >= 0.9 x U_best, take the LIN.
- If U_best <= 0, the candidate is CASH for that test year (in-fold abstention).

EXECUTION
- The engine-E band with the C7 order mapping. Weights are rounded to 0.05.

UNIT TESTS
- Band gradcheck.
- Planted signal (IC 0.1 added to a copy of R at h = 15): positive held-out C7 PnL.
- Shuffled R: mean |p| < 0.05.
- The gradient of the cost term is non-zero.
- Engine E under N equals engine F (simulator test 9).

(B) LEARNABLE-BANK VARIANT (contingent)
- Runs only if T1-NET or T2-NET passes all dev gates in its track. It is then candidate K+1, and p_sel is recomputed over K+1.
- Bank 'seq' mode with trainable periods, loss term + 1e-3 x prior() (distance to theta0), feature 50 dropped.
- Only at the passing net's selected (h, arch, kappa).
- About 1-2 h.
- Otherwise it is not run: LIG found no direction skill, and a fixed span grid under a linear layer already spans nearly the same EMA function class.

(C) NEURAL_TRADE: fixed in place (see model_py_fixes). It is a forecaster, not a policy, and is traded only through the common hysteresis layer as T1-NT.

## objective

DECISION OBJECTIVE
- Net PnL of equity at C7-B on dev OOS (1,765 days).
- Constraints: |x| <= 1, shorts allowed, funding charged, realistic fills.
- C4 and C10 (B and S) are always reported. The decision is taken only at C7-B and is never moved to a cheaper scenario. Choosing C4 because it looks better would itself be selection.
- Accuracy is never a selection criterion.

GATE STATISTIC (both tracks)
- S = mean daily net log equity return at C7-B (%/yr, additive).
- p_sel uses its studentised form, t = mean/(HAC SE) = Sharpe x sqrt(T): a max over candidates with different volatility needs studentising.
- Track 2 is compared with B&H, TSMOM and vol-target baselines on Sharpe, because their 1x beta dominates raw PnL. Net PnL is printed alongside.

HYPOTHESES (per track, one-sided, alpha = 0.05)
- H0_skill: the selected candidate's timing is worth nothing beyond its own exposure and turnover (circular-shift null, max over K).
- H0_profit: mean daily net return at C7-B <= 0.
- Beating B&H on PnL: reported with a paired bootstrap, never tested as a verdict. It is not detectable at any available length: the minimum detectable excess is about 95%/yr on dev OOS.

TRAINING OBJECTIVES
- PnL net: de-meaned daily-sum mean-variance utility with the exact increase/reduce cost and funding (see network).
- Ridge: MSE on r_{t,H}/sigma_{t,H}, with hysteresis on top.
- neural_trade: its own likelihood losses after the fixes.
- Rules: no training.

INFERENCE (stats.py)
(i) Stationary block bootstrap (Politis-Romano) on daily net returns, B = 10,000, mean block 10 days (Track 1) or 20 days (Track 2). Gives percentile CIs for the mean, Sharpe and max drawdown, and PAIRED differences against each baseline on the same resampled days.
(ii) Newey-West HAC t, Bartlett kernel, lag 10 or 20 days. Reported; the bootstrap decides.
(iii) Per-trade CIs from blocks of 20 consecutive trades.
(iv) Deflated Sharpe with N = all configurations tried in the track plus logged reruns. Reported, not a gate.
(v) p_sel (see selection_and_preregistration).

POWER (dev-only scale proxies; one-sided 5%, 80% power)
Track 1
- Per-trade SD 25-40 bps; design effect 1.3.
- Minimum detectable net edge per trade: 2.2-3.6 bps at 1,000 trades, 1.3-2.1 at 3,000, 0.7-1.1 at 10,000.
- Dev OOS detects about 1 bps; the seal about 2 bps (2,000-4,000 fade trades).
- If the true net is -3 bps per trade, the chance of rejecting net >= 0 is 0.67 at 1,000 trades and 0.97 at 3,000.
Track 2
- SE of an annual Sharpe is about 1/sqrt(years).
- Minimum detectable Sharpe: 2.49 on the seal, 2.68 on the clean seal (0.86 yr), 1.13 on dev OOS, 1.03 on dev OOS plus seal.
- For a true Sharpe of 0.8, the power to show Sharpe > 0 is 0.20 on the seal and about 0.55 on dev.
- The seal can falsify Track 2; it cannot confirm it.

## strategy_layer

research/pnl/policy.py. Every candidate is reduced to a target exposure in [-1, 1] at each decision, plus a band or hysteresis. Differences between candidates therefore come from the signals alone.

1. COST-AWARE HYSTERESIS (forecasters: T1-LIN, T2-LIN, T1-NT)
- mu_H in bps = forecast x sigma_{t,H}, with sigma_{t,H} = sigma60 sqrt(H) for H <= 60 and sigma_t sqrt(H) beyond.
- State s in {-1, 0, +1}; size |x| = 1. c_rt = 8 bps, the effective C7-B round trip.
- From flat: enter sign(mu) if |mu| > theta c_rt.
- In a position, after a minimum hold of H:
  - exit if s mu <= 0;
  - flip if s mu < -theta c_rt;
  - otherwise hold, at no cost.
- theta in {0.5, 1, 2}.

2. EVENT RULES (T1-FADEV and the fade baselines)
- Maker entry at the signal close, K bars, then cancel.
- Taker exit exactly H after the fill. No pyramiding.
- FADEV only: a new same-side signal during a hold resets the exit clock (no new round trip), and an opposite signal is ignored.

3. BAND POLICY (PnL nets)
- Trade to the nearest band edge, including drift; rounded to 0.05; up to |x| = 1.

4. DAILY RULES (T2-P and the trend baselines)
- Decide at the 23:59 close; maker at that close, chase to taker after 15 bars.
- The old position pays the 00:00 print.
- Rebalance only when |dx| >= 0.2.

5. ORDER TYPES
- Increase |x| as maker; reduce as taker (C7). A flip is a taker close plus a maker open.
- Track 1 unfilled entries are cancelled. Track 2 orders chase after 15 bars.

6. SIZING
- Track 1: |x| = 1 when in a position. Sizing by 1/sigma made the fade worse before.
- Track 2: T2-P uses a 50% vol target; nets size through tau.
- The 1x cap and drift trim are in the simulator.

7. FUNDING
- Nets: funding features as inputs, and funding in the loss.
- FADEV: skip an entry if its planned hold crosses a print whose expected payment (side x last print) is >= 3 bps.
- T2-P: no carry overlay. Design 3's overlay tau -= clip(3 fbar/(gamma sigma_d^2), 0, 0.5) is reported as a descriptive variant only.
- No funding-timing trades: only 4.2% of prints exceed a maker/maker round trip.

8. RISK RULES
- No stop-loss, take-profit, confidence sizing or circuit-breaker overlays. Untrained overlays add degrees of freedom the gate does not count, and earlier brackets cut gross PnL. MAE is reported instead.
- Gap rule: no orders during a gap; signals whose lookback spans the gap are suppressed for 60 bars.
- No information later than the decision-bar close.
- Every free parameter is listed per candidate and chosen in-fold.

INTERPRETATION RULE (fixed now)
- If a Track 1 candidate's median hold exceeds 1 day, it is reported as 'the cost-optimal policy on the SRS grid is a slow policy', not as intraday skill.
- Its static-versus-timing split must be shown.

## track1

SCOPE
- Horizons and decision cadences 1, 5 and 15 min only.
- Intraday features (41 columns; lookback at most 1 day plus funding).
- C7 execution: maker entries with K = 1 (h <= 5) or 3 (h = 15), then cancel; taker exits.

PRE-REGISTERED CANDIDATES (K1 = 4)
1. T1-FADEV: volatility-gated fade, no free parameters.
   - z15 = ln(c_t/c_{t-15})/(sigma60 sqrt 15).
   - Signal when the rank of |z15| among the trailing 1,440 one-minute values is >= 0.98 AND the trailing-30-day percentile of sigma60 is >= 2/3.
   - Fade the move; maker at c_t, K = 3, cancel; taker exit 15 bars after the fill.
   - Same-side extension and the funding skip, as in strategy_layer.
2. T1-LIN: ridge on the 41 columns predicting r_{t,H}/sigma_{t,H}, with hysteresis, deciding every minute.
   - In-fold grid: H in {1, 5, 15} x theta in {0.5, 1, 2} x lambda in {1, 10, 1e2, 1e3, 1e4} = 45 configurations.
   - Chosen by purged CV on engine F at C7 costs. Cash for the year if the best CV net is <= 0.
3. T1-NET: PnL band net.
   - In-fold cell from h in {1, 5, 15} x {LIN, MLP} x kappa in {1, 2} = 12 cells, 5 seeds each.
   - Inner-validation choice with LIN preference and abstention (see network).
4. T1-NT: fixed neural_trade, mu_h from the tau = 0.5 head, through the hysteresis layer with theta = 1 (|mu| > 8 bps).
   - h in {1, 5, 15} chosen on neural_trade's own validation block under engine F; cash if every h is <= 0.

N1 (Deflated Sharpe) = 4 + 45 + 12 + 3 = 64, plus logged reruns. Prior looks are disclosed separately: up to 270 rules per model in plans/trading_calibration.md, and the strat_probe fades.

BASELINES (same simulator and scenarios)
- Cash.
- B&H perp: information only, because Track 1 strategies are close to market-neutral.
- PLAIN FADE: the rule pre-registered in plans/trading_calibration.md section 6. On the 1-min grid with a trailing rank over 1,440 minutes (the same 24 h that 480 three-minute origins covered); top 2%; fade; hold 15; no overlap. Same order types as FADEV.
- INTRADAY TREND: sign(EMA15 - EMA60) of 1-min closes, re-decided every 15 min, |x| = 1.
- Static-exposure-matched: x = the candidate's mean x, held constant.
- Random-timing null: circular shift.

GATE COMPARATORS
- Cash, for profit (G2).
- Plain fade and intraday trend on mean daily net (G3).

TRACK-SPECIFIC REPORTING
- Maker fill rate and post-fill markout.
- Gross, net, trade count and SD per trade.
- Median hold and the interpretation rule.
- A pass that holds only at C4 is not deployable on bar data. It would need tick or L2 (aggTrades/bookTicker) validation first.

COST GEOMETRY (dev)
- Mean |r| is 5.4, 11.9 and 20.6 bps at 1, 5 and 15 min.
- Break-even accuracy per round trip at 7 bps: above 100% at 1 min, 79-85% at 5 min, 67-70% at 15 min.
- Maker-entry fades gross only +0.2 to +1.6 bps per trade (strat_probe), about 0.8 bps of adverse selection below taker.

## track2

SCOPE
- Open search from 1 minute to 1 day. Decision cadences 1, 5, 15, 60, 240 and 1440 min.
- Execution: every order is maker at the decision close, chased to taker after 15 bars. Funding is charged throughout.
- These are separate statistics from Track 1, even where trained Track 1 models are reused as options.

PRE-REGISTERED CANDIDATES (K2 = 3)
1. T2-P: fixed prior trend, no free parameters.
   - Daily at the 23:59 close: tau = mean over L in {7, 14, 28, 56} days of sign(ln c_t - ln c_{t-L}), long/short.
   - x* = tau x min(1, 0.50/sigma_ann), with sigma_ann = sqrt(365 x EWMA-30-day of squared daily log returns); rounded to 0.1.
   - Trade only if |x* - x_held| >= 0.2.
   - Fixed from the literature-standard lookback ensemble, not chosen by any dev selection. The probes saw this family in-sample (disclosed).
2. T2-LIN: ridge on the 55 columns at 5-minute origins, with hysteresis and a minimum hold of H.
   - In-fold grid: H in {1, 5, 15, 60, 240, 1440} x theta (3) x lambda (5) = 90 configurations.
   - Purged CV picks H in every fold, so the open search adds no outer multiplicity. Cash if the best CV net is <= 0.
3. T2-NET: PnL band net. In-fold cell from 24 cells:
   - h in {60, 240, 1440} trained on 55 columns, x {LIN, MLP} x kappa {1, 2};
   - PLUS the 12 Track 1 cells (h in {1, 5, 15}), reused with no extra compute.
   - Same choice and abstention rules as Track 1.
   - Training uses random phases; evaluation uses phase 0 (hh:59 or 23:59 closes). A phase-12h evaluation is reported for sensitivity.

N2 = 3 + 90 + 24 = 117, plus reruns.

BASELINES
- Cash.
- B&H perp, funding paid; spot-equivalent B&H without funding as information.
- TSMOM-blend unscaled: tau of T2-P with no vol target, x = tau, rebalanced when it changes.
- Vol-target long: x = min(1, 0.5/sigma_ann), band 0.2, daily.
- 60-min fade: top 2% of |z60| by trailing 30-day rank, hold 60, maker entry K = 5 then cancel.
- Static-exposure-matched.
- Carry hurdle: delta-neutral cash-and-carry at 1x = sum of funding prints - 30 bps/yr. A reference hurdle only; the spot leg is not in the data.
- Random-timing null.

GATE COMPARATORS
- Cash, on mean net (G2).
- B&H, TSMOM-blend unscaled and vol-target long, on Sharpe (G3).

CONTAMINATION (printed next to every Track 2 number)
- Earlier checks computed daily TSMOM (14-56-day lookbacks, net Sharpe 0.67-0.82 against 0.32 for B&H) and vol-targeted long BTC on 2022-01..2026-11. That covers about 80% of dev OOS and the whole seal.
- The strat_probe trend.py looked at 2020-2025 in-sample.
- The design models may know the 2025-26 BTC path from training data.
- Trend-family results are therefore not clean on either period. The only clean evidence is forward data from 2026-11 onward.

COST GEOMETRY
- Break-even accuracy at 7 bps: 60% at 1 h, 55% at 4 h, 52% at 1 d.
- At 4 h to 1 day, gross per unit of turnover is 100-190 bps against about 4 bps. Cost hardly matters there; skill, regime and power bind.

## baselines

Every baseline goes through the same engine E, the same dev OOS chain (2021-01-01..2025-10-31) and seal, the same six C x {B, S} scenarios, and the same metrics and bootstraps.

B0 CASH: x = 0, PnL 0. A Binance USDT futures wallet earns nothing. The T-bill rate is given as a footnote.

B1 BUY-AND-HOLD PERP: x = 1 from the first OOS bar, maker entry (chase after 15 bars), funding paid on every print, taker exit at the end.
- Funding paid by a 1x long: 17.2% (2020), 30.6% (2021), 4.2% (2022), 7.9% (2023), 12.0% (2024), 4.3% (2025 Jan-Oct).
- A spot-equivalent B&H without funding is shown as information.

B2 TREND
- Track 1: sign(EMA15 - EMA60) of 1-min closes, re-decided every 15 min.
- Track 2: TSMOM-blend unscaled (lookbacks 7/14/28/56 days).

B3 FADES
- Track 1: the section-6 plain fade (see track1).
- Track 2: the 60-min fade (see track2).

B4 VOL-TARGET LONG (Track 2): x = min(1, 0.5/sigma_ann), band 0.2. It isolates what volatility scaling adds without any direction.

B5 STATIC-EXPOSURE-MATCHED: x = the candidate's mean x, held constant. Timing PnL is measured against it.

B6 CARRY HURDLE: sum of funding prints - 30 bps/yr, per year. A reference only, not a simulated strategy.

B7 RANDOM-TIMING NULL: the candidate's own realised positions, circularly shifted against returns by s in [30 days, n - 30 days]. This keeps exposure, turnover and autocorrelation. It decides H0_skill.

RULE
- A candidate must beat every gate comparator of its track on the point estimate (G3).
- Significance is required only against cash (G2) and the timing null (G1); the data cannot power the other comparisons.
- Every candidate is also reported as an excess over cash, over B&H and over B5, with paired-bootstrap CIs.

## selection_and_preregistration

P_SEL (selection-adjusted skill test; one per track; stats.py)
- Inputs: each candidate's realised dev-OOS exposure x_k(t) on the 1-min grid (engine E, C7-B), and 1-min returns plus funding.
- For b = 1..999:
  - draw s_b uniformly from [30 days, n - 30 days] (in minutes);
  - shift the RETURN and funding series circularly by s_b against every candidate's fixed positions (the same s_b for all K);
  - recompute the daily net PnL under engine F (c_in = 3, c_out = 5 bps; costs are unchanged by the shift);
  - compute each candidate's t-statistic; M_b = max_k t_k^b.
- The observed max_k t_k is computed the same way on unshifted returns.
- p_sel = (1 + #{b: M_b >= observed max}) / 1000.
- About 7 candidates x 999 x 2.5M multiply-adds per track: minutes.
- In-fold selection never touched test data, so the max over K is the only outer selection to correct for.

DEV GATE (per candidate; all five must hold on dev OOS)
- G1: p_sel <= 0.05.
- G2: the one-sided 95% stationary-bootstrap lower bound of the mean daily net return at C7-B is > 0.
- G3: point estimates beat every gate comparator.
  - Track 1: mean daily net > plain fade and intraday trend.
  - Track 2: Sharpe > B&H, TSMOM-blend unscaled and vol-target long.
  - The paired-bootstrap P(candidate > baseline) is reported but does not gate.
- G4: net PnL at C7-B > 0 in at least 3 of the 5 test years, and in both halves (2021-01-01..2023-06-01 and 2023-06-02..2025-10-31).
- G5: net PnL at C7-S > 0 (1-bps trade-through, +1 bar latency, +1 bps taker slippage).
- The Deflated Sharpe, HAC t and per-year tables are reported only.

SELECTION PER TRACK
- Among candidates that pass G1-G5, deploy the one with the highest dev mean daily net at C7-B.
- If none passes, the track deploys CASH. The candidate with the highest dev t is pre-registered as INFORMATION-ONLY.
- neural_trade is always measured and reported.

PRE-REGISTRATION (before unsealing)
- research/pnl/prereg.json plus prereg.md, committed and pushed.
- Contents:
  - git SHA of research/pnl and model.py; SHA-256 of both npz files, features_v1.json, every fold-S weight bundle and scaler;
  - per track: the deployed candidate (or cash) and the information-only candidate;
  - the frozen procedure: feature version, folds, inner grids, seeds, K, chase policy, fees, scenarios;
  - the primary statistics and the verdict rules below;
  - dev results with CIs; p_sel; N for the Deflated Sharpe; the power statement;
  - disclosures: seen windows; trend contamination; strat_probe and critique looks at dev; designer knowledge of the 2025-26 price path; estimated 2026-10 funding; missing 2026-10-28.

ENFORCEMENT (run_seal.py)
- It refuses to run unless:
  - the working tree is clean;
  - HEAD contains prereg.json and PNL_UNSEAL == its SHA-256;
  - research/pnl/work/seal_lock.json does not exist (it is written first);
  - the weight hashes match.
- It recomputes features on the full data and asserts the dev rows are bit-identical.
- It scores every candidate, baseline and neural_trade ONCE, in all six scenarios, on the full seal and the clean seal, and writes results/seal.json.
- It prints nothing until the end.
- No rerun, except one logged retry after a crash that happened before any output was written.
- Clean seal: excludes 2025-11-01..11-10 07:30 and 2026-09-20..10-30 (about 313 days). Daily PnL of overlapping days is dropped; positions evolve through them.

SEAL VERDICTS (only the deployed candidate gets a verdict; everything else is descriptive)
Track 1
- PASS: one-sided bootstrap p < 0.05 for mean daily net > 0 at C7-B on the full seal, AND mean > 0 on the clean seal, AND mean > 0 in both seal halves.
- KILL: the mean is <= 0 on the full or the clean seal.
- Otherwise INCONCLUSIVE: no deployment; forward paper trading with a pre-registered sequential test.
Track 2
- FALSIFIED (kill): seal Sharpe < SR_dev - 1.645 sqrt(1/T_seal + 1/T_dev) on the full or the clean seal. With SR_dev = 0.8 the bound is about -1.0.
- CONSISTENT: not falsified AND seal net PnL at C7-B > 0.
- WEAK: not falsified, PnL <= 0. No deployment change.
- CONFIRMED (a flag on CONSISTENT): the one-sided 95% lower bound of the pooled dev-OOS-plus-seal mean daily net is > 0.
- A PASS or CONSISTENT verdict means only 'go to at least 12 months of paper trading'. It never means capital.

KILL CRITERIA
- K0, harness: any check, reconciliation or sanity test fails. Stop, fix, rerun from scratch (logged).
- K1, Track 1 closure label: the upper 95% bound of net per round trip at C7-B is < 0 for every Track 1 candidate. Track 1 is closed at this fee tier; the seal is still scored once for the record.
- K2, neural_trade: the OOS IC CI includes 0 at every h, or T1-LIN beats it on C7-B net. Retire it from trading use.
- K3, nets: OOS mean |x| < 0.05 is reported as 'the net chose cash', a valid result. Failing the planted-signal test is a bug.
- K4, Track 2: no candidate passes. No active strategy is recommended; B&H versus cash is a risk choice, not a finding.
- K5, seal: a deployed candidate that is KILLED or FALSIFIED is dropped permanently for this venue and fee tier.
- K6, budget: apply the compute-plan cuts in their pre-declared order.

## model_py_fixes

In-place edits to /home/user/neural_trade/model.py (TF/Keras). The run environment is tensorflow-cpu 2.20.0 with keras 3.11.3. This combination is verified in the scratchpad venv nt_h2h/venv; recreate it as research/pnl/.venv-tf (gitignored) from the uv cache. Each fix gets a test in test_model.py, and the existing tests must keep passing (input-shape tests are updated under M4).

M1 (D1/B10): dead indicators
- At lines 1073, 1085-1087, 1110-1112, 1135 and 1147, replace `logit + stop_gradient(logit)*(grad_multiplier-1)` with `logit`.
- Set INDICATOR_GRAD_MULT = 1.0 (line 109). The multiply in train_step (1903-1905) then becomes a no-op.
- ParamsLogger logs the effective periods through the same mapping as the forward pass.

M2 (D7): momentum periods
- At init (about line 1051) and write-back (1928-1932), use raw = log(expm1(max(clip(p, min_p, max_p) - 1, 1e-3))). This is the exact inverse of softplus(raw) + 1 (1160-1162) and replaces asinh((p-1)/2).
- The clip is kept, so deleting the write-back (design 3) is rejected.

M3: add LayerNormalization on ind_seq before the BiGRU (about line 1239), so %B-like channels cannot saturate the GRU gates.

M4 (D8/B12): unit-free inputs
- In make_sequences_with_extended_trends (298-310), channel 0 = 1e4 ln(close/last_close), i.e. the window in bps relative to its last close.
- Targets = 1e4 ln(c_{t+h}/c_t) bps.
- The harness passes last_close = 1e4, so every internal ret = delta/last_close conversion (labels, deadband, alignment) stays correct without touching the loss code.
- Add 5 causal channels: log range in bps, ln(v/E_1440 v), taker imbalance, hour sin, hour cos.
  - Indicators and meta_adjust read channel 0 only; the extras are concatenated after the LayerNorm. Input shape (60, 6); Config.EXTRA_CHANNELS = True.
- Input scaler: one global StandardScaler(with_mean=False) fitted on training rows (386-393).
- Target scaler with_mean=False (379); pred_mean = 0 (712-713), so P(up) is evaluated at a raw zero.

M5 (D6): the loader accepts a `datetime` or `timestamp` column, keeps taker_buy_base and trades, and accepts an in-memory frame from research/pnl/data.py (198-204).

M6 (B6/D3): biased losses
- FOCAL_ALPHA = 0.5 (113); TAU_PRICE = 0.5 (78).
- DIR_DEADBAND_BPS = 7.0 (119), the cost level as plans/money_blockers.md recommends. Direction loss and metrics are masked when |r| < 7. Unmasked AUC is also reported.

M7 (D2/B11): loss scale
- Remove the zero-gradient trend terms from the total (1644-1676).
- Delete the lambda calibration and its 1000 clamp (734-813).
- Fixed lambda_point = lambda_dir = lambda_var = 1 on unit-free targets.
- LAMBDA_VOL = 0 (the SD-matching term rewards noisy forecasts); LAMBDA_INTER = 0 (L2 is otherwise counted twice).

M8 (D4/B8): validation and stopping
- Replace TimeSeriesSplit (360-372) with train_and_evaluate(train_range): validation = the last 15% of the range, purged 15 bars. The test fold exists only in the harness and is never passed to callbacks (728-731).
- Assert that both direction classes are present in validation.
- EPOCHS = 10; EarlyStopping on val_loss with patience 3 and restore_best_weights (819).
- Remove the MCC stopper (821-826).
- ReduceLROnPlateau: factor 0.5, patience 1 (827-828).
- Config.SEED replaces set_random_seed(42) (693).

M9 (B3): MAX_SEQUENCE_COUNT = None (48, 348-357). The harness sets WINDOW_STEP = 20, about 26k sequences per 12-month window. This is sized to the measured 4.6-5.6 ms per sequence-epoch: about 20 min per fit, whereas design 1's stride 5 would take about 90 min.

M10 (B11): artifacts
- force=True never loads existing weights, and force=False never silently loads stale ones (843-853).
- MODEL_PATH and SCALER_PATH are keyed by fold id plus config hash.
- One bundle holds the weights, both scalers and the config.

M11 (B9/B11): predict_at(bundle, frame, idx) returns, per h in {1, 5, 15}: mu_h in bps (tau = 0.5 head), P(up)_h and the Student-t scale in bps. It is causal and batched.

M12 (B9/D5): add a banner to the inference.ipynb backtest cell: superseded by research/pnl/sim.py. No number is taken from it.

Unchanged: layer sizes and LOOKBACK = 60 (SRS). Known limit: EMA-60 keeps 14% of its weight on the first bar of a window.

TESTS
- T1: the forward EMA alpha equals sigmoid(logit) (that is, 2/(P+1)) at init.
- T2: momentum round-trip error < 1e-6 for p in [1.5, 60].
- T3: every active loss term has a non-zero gradient; the trend term has zero weight.
- T4: with alpha = 0.5 on balanced labels, the loss-optimal P(up) is 0.50 +- 0.01; the tau = 0.5 optimum is the median.
- T5: scale invariance: prices x10 give identical inputs and outputs.
- T6: the scaler sees training indices only; the split has a purge and no overlap.
- T7: perturbing bars after an origin leaves its prediction bit-identical.
- T8: a planted AR(1) signal (0.1 at 5 min) gives OOS IC > 0.05.
- T9: on a synthetic random walk, validation P(up) is within +-0.05 of the base rate.
- T10: reloading the bundle in a fresh process reproduces predictions to 1e-5.

MEASUREMENT (nt_adapter.py; 1 seed; 1 thread per process)
- Fits: folds Y2021-Y2025 plus S, each on the trailing 12 months.
- Predictions at every test minute: 2.54M dev OOS plus 0.52M seal, at 1.2 ms per sample, split across processes.
- Reported per h: Spearman IC, AUC (masked and unmasked), pinball, and QLIKE against EWMA-t, all with block-bootstrap CIs; P(up) centring; the share of origins with |mu| > 8 bps.
- Traded only as T1-NT. Judged by K2.

## compute_plan

MACHINE: 4 threads, 15 GB RAM, no GPU.
- Torch 2.15 runs via `uv run --offline --no-project --with torch --with numpy --with scipy` (verified).
- TF 2.20 and keras 3.11.3 are verified in the scratchpad venv; recreate the venv under research/pnl/.venv-tf.
- Workers: 4 single-thread processes (torch.set_num_threads(1); TF intra-op and inter-op threads = 1) pulling from one job queue. Single-thread workers measured faster in aggregate than one 4-thread process.
- Memory: a 0.68 GB memory-mapped feature table plus under 1.5 GB per worker.

CODE (research/pnl/)
- data.py (seal guard), features.py, folds.py, sim.py, metrics.py, stats.py (bootstrap, HAC, p_sel, DSR, Holm), policy.py;
- rules.py (FADEV, T2-P, baselines), ridge.py, net.py, train.py, nt_adapter.py;
- checks.py, tests/;
- run_dev.py, gate.py, prereg.py, run_seal.py, report.py.
- Outputs: results/{dev.json, gate.json, seal.json, report.md, runs.jsonl, seal_access.log}. work/ is gitignored.

STEPS (CPU-min; wall time assumes 4 workers)
0. Environments: about 10 min wall.
1. Seal guard, simulator, metrics, checks and tests: about 10 CPU-min.
2. Dev features (lfilter, 55 columns): about 15 CPU-min.
3. Rules and baselines in 7 scenario cells (event loop, 1-3 s per run): about 15 CPU-min.
4. Ridge T1-LIN (45 configurations) and T2-LIN (90) x 5 inner folds x 5 folds. Normal equations on blocked sufficient statistics; the hysteresis inner CV loops in engine F. About 90 CPU-min.
5. Nets:
   - Track 1: 3 h x 4 configurations x 5 folds x 5 seeds = 300 runs, about 25 s average (h = 1 about 50 s, h = 5 about 15 s, h = 15 about 8 s), about 125 CPU-min.
   - Track 2: 300 runs at about 15 s, about 75 CPU-min.
   - Ensemble validation and test inference: about 10 CPU-min.
6. neural_trade: 5 dev fits x about 20 min, plus 51 CPU-min of inference. About 150 CPU-min, interleaved with step 5 on the queue.
7. Engine-E dev OOS runs of 7 candidates x 7 scenario cells: about 20 CPU-min.
8. Gates: p_sel 999 x 2 tracks, bootstraps (10k), Deflated Sharpe: about 20 CPU-min.
9. Fold-S training before pre-registration: nets 120 runs at about 35 s, ridge 10, neural_trade 20. About 100 CPU-min.
10. Pre-registration commit and push: minutes.
11. Seal: features plus the bit-identity assertion, neural_trade seal inference (10 CPU-min), simulation and statistics. About 30 CPU-min.

TOTAL CORE: about 11 CPU-h, about 3-3.5 h wall. With debugging, serial steps and up to 2 logged reruns of cheap stages, about 5-6 h wall.
Contingency: the learnable-bank variant, 1-2 h, only if triggered. Ceiling about 8 h, under the 10-h limit.

BUDGET STOP RULES (fixed before any result; never chosen by looking at results)
- Time the Y2021 fold of steps 5 and 6 first and project the rest.
- If the projection exceeds 10 h, cut in this order:
  1. drop the learnable-bank contingency;
  2. nets with 3 seeds instead of 5;
  3. neural_trade WINDOW_STEP 30 and EPOCHS 6;
  4. drop the kappa = 2 cells at h = 1 (recorded as 'not run'; they leave the grid);
  5. drop T1-NT from the candidates (K1 = 3, recorded before the gates). neural_trade is then measured on Y2025 and S only.
- The nulls, the bootstraps and the seal are never cut.

## risks

1. POWER
- The seal can only falsify Track 2 (minimum detectable Sharpe 2.5); dev OOS detects about 1.1.
- A real Sharpe 0.5-1 trend edge will most likely not be confirmed. That is a property of one asset and about 6 years, not a failure of the method.
- Track 1 has real power, which makes a negative verdict solid.

2. CONTAMINATION AND SELECTION
- Earlier TSMOM and vol-target checks covered 2022-01..2026-11, including the whole seal.
- strat_probe and the critique looked at dev in-sample; up to 270 rules were tried per model in the calibration study.
- Two seen windows lie inside the seal (and 3 weeks of dev).
- The design models may know the 2025-26 BTC path.
- Mitigations: fixed T2-P; mechanical in-fold selection; K = 4 and 3; p_sel; runs.jsonl and N in the Deflated Sharpe; clean-seal reporting; disclosure in prereg.md.
- The residual inflation cannot be removed.

3. DRIFT OR BETA MASQUERADING AS SKILL
- Mitigations: de-meaned net training and validation; the exposure-preserving circular-shift null; the static-matched baseline; the static-versus-timing split; beta and alpha in every report.

4. FILL MODEL
- 1-min bars hide the order of events inside a bar, latency, queue position and partial fills.
- A 1-tick trade-through may still miss in fast markets.
- C7-S is a gate (G5).
- Any pass that depends on maker fills needs aggTrades or bookTicker validation before real money.
- Training assumes 3 bps fills at the close; simulator test 9 bounds that mismatch, and implementation shortfall is reported.

5. REGIME SHIFT
- Volatility in 2025 is about half of 2021; funding fell from 17-31%/yr to about 4%.
- 10 of 12 unbanded daily trend variants lost in 2025.
- The expanding window overweights 2020-21.
- Per-year tables and the 2022+ subperiod are reported.

6. NET OPTIMISATION
- Noisy validation PnL picks lucky epochs.
- The band can collapse.
- Mitigations: 5-seed ensembles, kappa, LIN preference, warm-up, the soft-band fallback.

7. SEAL LEAKAGE THROUGH CODE
- data.load() currently returns seal bars. The guard lands first.
- The dev-row bit-identity check runs at unsealing.
- Fold-S weights are hashed before unsealing.

8. DATA
- 2026-10 funding is estimated (sensitivity: the last real print).
- 2026-10-28 is missing, inside a seen window.
- The close is used as a proxy for the mark price (< 0.1 bps per print).
- Thin liquidity in 2020 is flagged against bar volume.
- Flash wicks can inflate maker fills; fill rates are reported.

9. NEURAL_TRADE FIDELITY
- Stride 20 and 1 seed may under-train it. This is reported as a limit, not as the model's ceiling.

10. RESEARCHER DEGREES OF FREEDOM
- Every post-result change is logged and enlarges K or N.
- Choosing among scenarios after the fact (the 'C4 looks better' trap) is excluded: the decision is at C7-B.

11. OUT OF SCOPE
- Exchange and counterparty risk, ADL, liquidation (a 1x short is liquidated only near +100%; the drift trim keeps |x| <= 1.1), fee-tier or BNB discounts, maker rebates, tax.

12. COMPUTE
- neural_trade is the most uncertain cost. The stop rules are pre-declared.

## expected_outcome

HONEST PRIOR: no strategy beats cash or buy-and-hold after costs with statistical confidence. At most one slow trend rule shows a positive, consistent but unconfirmed expectation.

TRACK 1 (1, 5 and 15 min)
- P(any candidate passes the dev gate at C7-B) is below 5%.
- Expected:
  - T1-FADEV and the plain fade: maker-entry gross +0.3 to +2 bps per trade; net -5.5 to -7.5 bps at C7-B, -3 to -5 at C4, -8 to -10 at C10. That is -4% to -35%/yr at 700-4,700 trades per year.
  - T1-LIN: abstains (cash) in most folds, or picks H = 15 with theta = 2; slightly negative.
  - T1-NET: chooses cash or near-cash (mean |x| < 0.1) in most folds.
  - T1-NT: after the fixes P(up) is centred and QLIKE is close to EWMA, but IC is indistinguishable from 0, it rarely trades, and net is about 0 to slightly negative. Retired under K2.
- Deployment: CASH.
- The seal, with about 2,000-4,000 fade trades and kill power of at least 0.9 if the true net is <= -3 bps, should confirm the negative net of the information-only candidate.
- Verdict: the 1-15 min route is closed at retail perp fees. Only a rebate tier with tick-validated execution could reopen it.

TRACK 2 (1 min to 1 day)
- P(dev gate pass) is about 10-20%. G2 (bootstrap lower bound > 0) is the likely failure: a true Sharpe of 0.6 gives t of about 1.3 over 4.83 years.
- Expected:
  - T2-P: dev OOS net Sharpe 0.3-0.8, CI including 0, PnL below B&H in bull years, smaller drawdowns, about 13-21 units of turnover per year.
  - T2-LIN: picks H = 1440 or abstains.
  - T2-NET: near cash or slow-trend-like, unlikely (< 25%) to beat T2-P.
  - The carry variant adds +0.5-2%/yr at 2025 funding.
- Seal (information-only or deployed): Sharpe draw about N(0.3, 1.0); P(net > 0) about 60-65%; P(beats B&H) 40-55%. Most likely verdict CONSISTENT or WEAK, not FALSIFIED, never CONFIRMED on its own. It is contaminated for the trend family.
- 4, 7 and 10 bps change Track 2 by only about 1-2%/yr.

LIKELY HEADLINE
'After a 7 bps round trip (about 7.9 bps effective) and funding, no network and no intraday rule makes money on BTCUSDT perp. The cost-optimal intraday policy is cash. A daily vol-managed trend rule is the only candidate with a plausible positive expectation; it is unconfirmed and would need about 10 single-asset years or pooling across 10-30 perps. Cash or buy-and-hold remains a risk choice.'

DELIVERABLES REGARDLESS
- A leak-tested, exact fill, fee, funding and 1x simulator.
- Measured maker economics on perp bars.
- neural_trade fixed and measured honestly.
- A reusable PnL band-policy trainer.
- One pre-registered seal result per track, full and clean, with its stated power.
- Next steps that would actually add power: multi-asset trend pooling, forward paper trading from 2026-11, or a maker-rebate tier for Track 1. Not a bigger network.

## Judge scores

SCORES (1-10 per criterion: faithfulness / statistical validity / execution and cost realism / CPU feasibility in 6-10 h / risk control -> mean)

Design 1, "protocol first": 9 / 9 / 8 / 6 / 8 -> 8.0
- Strengths:
  - most faithful to the user's decisions (both tracks, both networks, C7 primary with C4/C10 reported, funding, shorts, 1x cap, seal guard, clean-seal report);
  - best statistics: few candidates (K1 = 4, K2 = 3) with in-fold selection, a circular-shift p_sel over K, stationary bootstrap, HAC, Deflated Sharpe, per-track power, and Track 2 "not falsified / unconfirmed" language;
  - the strongest harness checks (deliberate leak, shuffled labels, reconciliation, perfect foresight);
  - an explicit kill list.
- Defects:
  - The PnL-net is trained with a flat cost and no band, then executed with a band. Training and execution do not match, and no state-dependent no-trade region is learned.
  - The 61 Bank features enter the base feature set, which doubles the inputs for a tiny signal.
  - 19 quarterly folds multiply the cost of the networks.
  - The neural_trade budget (about 6 min per fit) is wrong. At the measured 4.6-5.6 ms per sequence-epoch, 105k sequences x 10 epochs is about 80-100 min per fit, so the primary plan does not fit.
  - Track 2 gates on Sharpe, not the user's PnL objective.
  - The adverse-selection charge is double-counted on top of the trade-through rule.

Design 2, "band policy net": 7 / 7 / 7 / 9 / 7 -> 7.4
- Strengths:
  - the best network: target exposure plus a learned no-trade band with an exact maker/taker increase/reduce cost split, a custom autograd band with gradcheck, warm-up, LIN and MLP, a QLIKE auxiliary head and seed ensembles;
  - measured, realistic compute, including a correct neural_trade stride (20, about 20 min per fit);
  - minimal model.py input fix (bps relative to the last close, so the loss code is untouched);
  - frozen 12-month seal fold consistent with annual refits; static-versus-timing split.
- Defects:
  - Dev OOS starts in 2022, which wastes the 2021 test year.
  - Selection is an argmax over 12 or 24 cells on dev OOS (SPA-adjusted) rather than in-fold.
  - There are no rule candidates, so the known fade and trend evidence is only used as comparators.
  - Four tranches without netting add simulator complexity.
  - Additive accounting has no 1x drift trim.
  - The tick is misquoted (0.1 USDT is about 0.01 bps, not 0.001).
  - The kill rules are thin.

Design 3, "strategy and execution": 8 / 7 / 9 / 7 / 8 -> 7.8
- Strengths:
  - the best execution realism: measured maker economics on perp bars (1-tick trade-through fills 87% in 1 bar and 97% in 15; 2.8-3.1 bps per side including fallback; effective C7 round trip 7.9 bps base and 9.2 stress), base and stress fill scenarios, a liquidity flag, implementation shortfall and markouts;
  - de-meaned training so the net cannot earn by learning in-fold drift;
  - daily-sum utility;
  - a procedure-level p_sel for rule grids;
  - fixed prior T2-P; disclosure of designer contamination.
- Defects:
  - Many candidate configurations (16 + 6 + 4 + 144).
  - Networks are gated only by Bonferroni plus 5 shifted-label retrains.
  - Annual dev folds but quarterly refits inside the seal, so the seal tests a different procedure than dev.
  - Track 2 gate at p 0.10.
  - The neural_trade estimate (105k sequences x 5 epochs in 15-20 min) does not match the measured speed.
  - Deleting the momentum write-back would also drop the period clip.
  - The 10-sigma circuit breaker is an untrained overlay.

FINAL SPEC: WHAT IT TAKES FROM EACH DESIGN
- From design 1: protocol, seal guard, conventions, harness checks, few candidates with in-fold selection, circular-shift p_sel, stationary bootstrap, Deflated Sharpe, gates G1-G5, Track 2 seal verdicts, kill list, FADEV and its baselines.
- From design 2: band-policy network and its exact cost split, feature table, minimal neural_trade input fix and stride, frozen seal fold, compute measurements.
- From design 3: fill model and measured effective costs, base/stress scenarios, de-meaned daily-sum loss, fixed prior trend rule, designer-contamination disclosure, attribution outputs.

DISAGREEMENTS RESOLVED (each is restated in the relevant field)
1. Refit cadence. Annual expanding folds with test years 2021, 2022, 2023, 2024 and 2025-01..10. The seal fold is trained once on all dev data and frozen for 12 months. This uses the same procedure in dev and seal. Quarterly refits (designs 1 and 3) were rejected because they multiply network cost and design 3's version is inconsistent. Starting dev OOS in 2022 (design 2) was rejected because it loses a year.
2. Selection. In-fold selection per candidate, then a max over K = 4 (Track 1) or K = 3 (Track 2) with a circular-shift p_sel. Design 2's SPA over cells and design 3's large grids were rejected.
3. Network. Design 2's tau plus learned band b(x), with the exact increase/reduce cost (3 bps in, 5 bps out). Combined with design 3's de-meaned returns and daily-sum utility, gamma = 2 fixed, LIN or MLP x kappa {1, 2}, 5 seeds. Design 1's train-without-band approach was rejected.
4. Bank features. Not in the base set. The learnable-bank variant runs only if a net passes its dev gate, and then becomes candidate K+1 (design 1's rule).
5. Gate statistic. Mean daily net log return at C7-B in both tracks (the user's PnL objective). p_sel uses its t-statistic, because max-type tests need studentising. For Track 2, comparisons with B&H and trend baselines are made on Sharpe.
6. Gate level. alpha = 0.05 in both tracks. Design 3's 0.10 for Track 2 was rejected because the seal cannot confirm Track 2, so the dev gate is the only significance filter.
7. Costs. Trade-through by 1 tick in the base case. Stress uses 1 bps trade-through plus 1 bar of latency plus 1 bps taker slippage. No separate adverse-selection charge, because the trade-through rule already prices it.
8. Accounting. Compounding equity, constant contracts between fills, and a cap trim at 1.10 (designs 1 and 3). Design 2's additive accounting and tranches were rejected.
9. Unfilled maker entries. Track 1 cancels after K bars. Track 2 chases with a taker order after 15 bars.
10. neural_trade inputs. Design 2's "bps relative to the last close" channel plus 5 extra channels.
11. neural_trade sampling. Stride 20, about 26k sequences per fit, sized to the measured speed.
12. neural_trade deadband. 7 bps, the cost level, as plans/money_blockers.md recommends. Design 1 used 3.5.
13. Momentum write-back. Use the exact inverse and keep the clip; do not delete it.
14. Overlays. No circuit breaker and no stops. The gap rule is kept.
15. Seal models. Every fold-S model is trained before pre-registration, and its weight hashes are pre-registered.
16. Fixed trend lookbacks. 7/14/28/56 days, long/short, 50% vol target, band 0.2. The carry overlay is descriptive only.
