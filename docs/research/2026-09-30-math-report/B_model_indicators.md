# B. The model, the learnable indicators, the search space, identifiability and generality

Research note, 2026-09-30. Read-only on the code; every number below either comes from a command
run for this note (scripts in `D:/nt_math_scratch/`, outside the repo) or from a committed run file,
and says which. Derivations are marked **(derivation)**; estimates are marked **(estimate)**. Nothing
here uses a test-fold number (D-020): the six runs are dev folds -3 and -2; the per-window periods
are measured on each run's dev (out-of-sample) block.

Sources: `src/neural_trade/models/gru_attention.py`, `models/layers/*.py`, `indicators/*.py`,
`utils/math.py`, `training/custom_model.py`, `training/optim.py`, `training/callbacks.py`,
`data/scaling.py`, `core/config.py`; branch `nt-047` (72d3838) for the OHLCV families; runs
`runs/scenarios/long_360d_stab/2026*` (six runs: folds -3, -2 x seeds 0-2, 360-day training block,
batch 2048); `runs/experiments/micro_loop_v1/LOG.md` (H4, Z5) and `interpret_c1.json`;
`runs/screens/l1_*`.

## 1. Architecture

### 1.1 The data path

Reference setup: window $L = 60$ one-minute closes, horizons $H = (10, 15, 20)$ bars, batch
$B = 2048$ (the long_360d_stab setting). The model input is one vector per window,

$$x_k = \frac{c_{t-L+1+k} - c_t}{s}, \qquad k = 0..L-1,$$

with $c_t$ the last close and $s$ the standard deviation of the pooled training targets (dollar price
changes of all three horizons; `data/scaling.py:22-25, 43, 52`; the loss uses the same scale,
`training/trainer.py:301`). So $x_{L-1} = 0$ and every input is in "target standard deviations".
For fold -2, $s = 223.7$ dollars; fold -3, $226.1$ (`scaler.joblib` of the runs).

The layers, with the shapes and parameter counts printed by `model.summary()` of the model built on
CPU from the f-2/s0 run's `config.yaml` (script `build_summary.py`; total **296,591** trainable
parameters, all trainable):

| # | Block (code) | Output shape | Params | Share |
|---|---|---|---:|---:|
| 1 | meta input: global avg and max pool of $x$, concatenated (`gru_attention.py:48-52`) | [B, 2] | 0 | |
| 2 | meta_adjust: Dense(18, tanh) (`:53-54`) | [B, 18] | 54 | 0.02% |
| 3 | LearnableIndicators: 18 learnable periods -> 30 indicator channels + raw close (`:57`; `layers/learnable_indicators.py`) | [B, 60, 31] | 18 | 0.006% |
| 4 | Bidirectional GRU(64) + Dropout 0.1 (`:60-61`) | [B, 60, 128] | 37,248 | 12.6% |
| 5 | "temporal" MHA, 8 heads x key 32, self-attention over the 60 positions, + residual + LayerNorm (`:64-67`) | [B, 60, 128] | 131,968 + 256 | 44.6% |
| 6 | "cross-indicator" MHA, 4 heads x key 32, on the transposed tensor (`:70-73`) | [B, 128, 60] -> [B, 60, 128] | 31,164 + 120 | 10.5% |
| 7 | Conv1D 16 filters, kernels 3 / 7 / 15, GELU (`:76-78`) | 3 x [B, 60, 16] | 6,160 + 14,352 + 30,736 | 17.3% |
| 8 | EnergyGate: softmax(Dense([var(x), max(x)])) blend of the 3 conv branches (`:82`; `layers/energy_gate.py:24-36`) + LayerNorm | [B, 60, 16] | 9 + 32 | |
| 9 | sinusoidal positional encoding, added (`:86`; `layers/positional_encoding.py`) | [B, 60, 16] | 0 | |
| 10 | 2 transformer blocks: MHA 4 x 16 (dropout 0.1) + LN + FF 32 GELU -> 16 + LN (`:89-97`) | [B, 60, 16] | 2 x 5,440 | 3.7% |
| 11 | context = global avg pool (`:100`) | [B, 16] | 0 | |
| 12 | T-perp projection Dense(16, tanh) -> VacuumSaturationNoise (train only) -> overflow Lambda, perp_magnitude Dense(1, softplus) (`:107-139`) | [B, 16], [B, 1] | 272 + 17 | 0.1% |
| 13 | regime_gate: Dense(1, sigmoid) on [mean abs deviation of $x$, context] (`:147-154`) | [B, 1] | 18 | |
| 14 | Flatten [B, 960] -> Dense(32, GELU), concatenated with context (`:157-162`) | [B, 48] | 30,752 | 10.4% |
| 15 | three towers Dense(16, GELU) (`:184, 215, 236`) | 3 x [B, 16] | 3 x 784 | 0.8% |
| 16 | heads per horizon: price Dense(1) clipped to $\pm 100$; direction logit Dense(1) + skip Dense(1, no bias) -> sigmoid; variance Dense(1, softplus) on [tower, perp_magnitude, regime_gate] (`:186-254`) | 9 x [B, 1] | 3 x (17 + 17 + 8 + 19) | 0.06% |
| 17 | outputs: 9 heads + vacuum_overflow (`:262-270`) | 10 x [B, 1] | | |

The 18 indicator periods are 0.006% of the parameters. About 72% of the parameters sit in
blocks 5, 7 and 14.

### 1.2 The heads and their parametrisations

- **Price (delta):** $\hat d_h = \mathrm{clip}(w^\top z_h + b, -100, 100)$ in scaled units (non-finite
  -> 0), $z_h$ the tower output (`gru_attention.py:186-194`). Served in dollars as $\hat d_h s + \mu$;
  outside $\pm 100$ the gradient is 0.
- **Direction:** $P(\uparrow) = \sigma(w_h^\top z_h + b_h + v_h^\top f)$, where $f \in \mathbb{R}^8$ is the
  DIRECTION_SKIP feature vector (`gru_attention.py:23-40, 179-181`): $f = (x_{L-1} - x_{L-1-k})_{k \in
  \{1,5,10,15,20,30\}}$, $x_{L-1} - x_0$, and $\log(\mathrm{std}(\Delta x) + 10^{-6})$. $v_h$ has no bias and an
  L2 of $10^{-4}$ (DIRECTION_SKIP_L2). This is a logistic regression on trailing returns and log
  volatility, added to the deep logit, so the head can always represent the linear baseline. The
  following clip to [0, 1] (`:198-201`) is a no-op after a sigmoid.
- **Variance:** $\hat v_h = \mathrm{softplus}(u_h^\top [z_h, m, g] + 1.0)$ at init (bias 1.0: $\hat v \approx 1.31$),
  in scaled units squared; $m$ = perp_magnitude (softplus), $g$ = regime_gate (sigmoid)
  (`:205-212`). Not exp and not log-variance: softplus grows linearly for large arguments and has
  gradient $\sigma(a)$, which vanishes only for very negative $a$ (tiny variances; the loss clips with
  VAR_FLOOR $10^{-4}$).

### 1.3 Attention cost against the window length

Two groups of tensors grow as $L^2$:

1. **The batched EWMA of the indicator layer** (`utils/math.py:221-241`): one weight tensor
   $[B, K, L, L]$ per stage, $K = 18$ rows in stage 1 (MA 3, MACD fast 3, slow 3, BB mean 3, RSI gains 3,
   losses 3) and $K = 6$ in stage 2 (MACD signal 3, BB variance 3). The function materialises
   `decay`, `weights` and two `tf.where` results of that shape.
2. **The attention score tensors:** block 5 $[B, 8, L, L]$, each block-10 MHA $[B, 4, L, L]$. Block 6
   is $[B, 4, 128, 128]$ and does not depend on $L$.

Float32 sizes of ONE such tensor **(estimate from the shapes; not measured on the GPU)**:

| Tensor | $B$ = 2048, $L$ = 60 | $B$ = 2048, $L$ = 240 | $B$ = 512, $L$ = 240 |
|---|---:|---:|---:|
| EWMA stage 1, $[B, 18, L, L]$ | 0.53 GB | 8.49 GB | 2.12 GB |
| block 5 scores, $[B, 8, L, L]$ | 0.24 GB | 3.77 GB | 0.94 GB |
| block 10 scores, $[B, 4, L, L]$ (x2 blocks) | 0.12 GB | 1.89 GB | 0.47 GB |

The RTX 4070 Ti has 12 GB. LOG.md H4 records the OOM at batch 2048 and 512 for LOOKBACK 240 and
attributes it to "the attention softmax". By these shapes the EWMA matrix is at least as large a
contributor as the attention (18 rows against 8 heads), and several copies of each live at once for
the backward pass. The same reading applies to the NT-047 duel's OOM at batch 2048 (LOG.md I2-duel
says "attention over 128 tokens"; block 6 is 0.54 GB and independent of the input width, while the
EWMA row count grows with the 14 families): **(estimate; a GPU memory profile would settle it)**.

Two blocks also tie the parameter count to $L$ **(derivation, checked against the summary at
$L = 60$)**: block 6 has $513L + 384$ parameters (31,164 at 60; 123,504 at 240) because the 60 time
positions are its feature dimension, and block 14 has $512L + 32$ (30,752 at 60; 122,912 at 240).
The model at $L = 240$ has about 481,091 parameters.

### 1.4 What block 6 actually attends over

The code comment says "attend across indicators" (`gru_attention.py:69`), but block 6 runs after
the Bi-GRU: its tokens are the **128 GRU hidden units**, and each token's features are its 60 time
positions (`Permute((2, 1))` of `[B, 60, 128]`). No layer attends across the 31 indicator channels;
they are mixed only by the GRU's input matrix. With position-specific weights over the time axis,
block 6 cannot run on a variable-length or window-free input.

## 2. Learnable indicators

### 2.1 From a period to a differentiable EWMA

Each learnable parameter $i$ is one scalar logit $\ell_i$ (`learnable_indicators.py:70-77`),
initialised from the configured period $p^{(0)}_i$ by $\ell = \mathrm{logit}(\alpha)$, $\alpha = 2/(p+1)$
(`utils/math.py:46-57`). In the forward pass, per sample $b$:

$$\alpha_{b,i} = \sigma\big(\ell_i + 0.5\,\tanh(W z_b + c)_i\big), \qquad z_b = \big(\overline{x}_b,\ \max_k x_{b,k}\big),$$

(`learnable_indicators.py:98-103`; `gru_attention.py:48-54`; META_SCALE 0.5, `indicators/base.py:29`).
The reported "learned period" is $p_i = 2/\sigma(\ell_i) - 1$ (`utils/math.py:60-72`,
`learnable_indicators.py:206-214`), i.e. **without** the per-window shift.

Every indicator is built from EWMAs with the recurrence $e_0 = x_0$, $e_t = \alpha x_t + (1-\alpha) e_{t-1}$,
computed as one matrix product (`utils/math.py:181-241`):

$$e_t = (1-\alpha)^t x_0 + \sum_{k=1}^{t} \alpha (1-\alpha)^{t-k} x_k,$$

with $\alpha$ clamped to $[10^{-6}, 1 - 10^{-6}]$. The weights sum to 1 for every $\alpha$. So a period is
not a hard window length: it is the half-life parameter of an exponential kernel, and the kernel is
truncated at the window start, where the first bar absorbs all the older weight.

The four families (`indicators/families.py`), per instance:

| Family | Params (textbook init) | Channels | Formula |
|---|---|---|---|
| ma | period (20) | 1 | $e(x; \alpha)$ |
| macd | fast (12), slow (26), signal (9) | 4 | line $= e(x;\alpha_f) - e(x;\alpha_s)$, signal $= e(\text{line};\alpha_g)$, hist, $\tanh(10\,\text{hist})$ |
| rsi | period (14) | 1 | $100 - 100/(1 + e(g;\alpha)/(e(\ell;\alpha)+10^{-8}))$, $g, \ell$ = one-bar gains / losses |
| bb | period (20) | 4 | mean $e(x;\alpha)$, $\sigma = \sqrt{e((x-\text{mean})^2;\alpha) + 10^{-8}}$, mean $\pm 2\sigma$, %B |

The reference config uses instances MA (5, 10, 30), MACD (12/26/9, 5/35/5, 8/17/9), RSI (9, 14, 21),
BB (10, 20, 25): 18 periods, 3 + 12 + 3 + 12 = 30 channels, plus the raw close = 31
(`period_init.json` of every run: `matches_config: true`).

The NT-047 catalogue (branch `nt-047`, `indicators/families_ohlcv.py`) adds ATR (1 param), Stochastic
(k, d: 2), Williams %R (1), Keltner (period, atr_period: 2), OBV (1), VWAP (1), MFI (1), ADX/DMI (1),
CCI (1), Donchian (1). Stochastic, Williams %R and Donchian use a Boltzmann soft rolling extremum over
an exact window of $p$ bars whose oldest bar enters with the fractional weight $p - \lfloor p \rfloor$
(`nt-047:indicators/base.py:283-322`); OBV, MFI and ADX use $\tanh(10\,\Delta x)$ / $\sigma(10\,\Delta x)$
soft signs (SOFT_SIGN_SHARPNESS, `:59`).

### 2.2 Bounds, projection, optimiser

- **Bounds:** after every optimizer step the logit is clipped so that the global period lies in
  [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX] = [2, 60] (`custom_model.py:547-550`,
  `learnable_indicators.py:224-238`, `core/config.py:411-412, 522-523`). This is a projection in
  logit space; the floor 2 ($\alpha = 2/3$) keeps $\sigma'$ away from 0.
- **The per-window shift is not bounded.** The applied $\alpha_{b,i}$ carries the shift
  $0.5 \tanh(\cdot) \in [-0.5, 0.5]$ after the clip, so the applied period ranges over about
  $p\,e^{\mp 0.5}$, i.e. $\times 0.61$ to $\times 1.65$ for large $p$ **(derivation)**. Measured on the
  dev blocks (section 4.2): applied periods of 1.6-1.8 bars below the floor of 2, and up to 74.7 bars
  above the ceiling of 60.
- **Optimiser:** the logits have their own Adam at $\mathrm{LR} \times \mathrm{INDICATOR\_LR\_MULT} = 0.001 \times 5 = 0.005$
  (`training/optim.py:21-25`; `custom_model.py:139, 540-542`); gradients are clipped by global norm
  per group (GRAD_CLIP_NORM 20, `custom_model.py:524-538`). ReduceLROnPlateau acts only on the main
  optimizer (`training/callbacks.py:377-380`, a Keras callback on `model.optimizer`): in the six runs
  the main LR fell to 0.000125-0.0005 while the indicator LR stayed 0.005 in every epoch
  (`log_lr_used`, `log_lr_indicator_used` in `indicator_params_history.csv`). The indicator-to-main
  ratio grows from 5 to 10-40 late in training.
- **INDICATOR_GRAD_MULT = 5** is a straight-through scale: forward uses $\ell$, backward $5\,\partial L/\partial \ell$
  (`learnable_indicators.py:100-103`). Adam's update is invariant to a constant gradient scale (up
  to $\epsilon = 10^{-7}$), so the multiplier changes nothing except how often the indicator group hits
  its clip norm **(derivation; `training/optim.py:3-5` says the same)**.
- INDICATOR_L2 = 0: no pull towards the textbook values.

### 2.3 The gradient with respect to a period (derivation)

Differentiate the recurrence: with $g_t = \partial e_t / \partial \alpha$,

$$g_0 = 0, \quad g_t = (x_t - e_{t-1}) + (1-\alpha)\,g_{t-1} \;\Rightarrow\; g_t = \sum_{j=0}^{t-1} (1-\alpha)^j (x_{t-j} - e_{t-j-1}) = \frac{1}{\alpha}\sum_{j=0}^{t-1}(1-\alpha)^j \Delta e_{t-j}.$$

$g_t$ is a discounted sum of the EWMA's own increments: how far the average moved over its memory.
With $\partial\alpha/\partial\ell = \alpha(1-\alpha)$ (times 5 in the backward pass) and
$\partial\alpha/\partial p = -\alpha^2/2$:

$$\frac{\partial e_t}{\partial \ell} = (1-\alpha)\sum_{j}(1-\alpha)^j \Delta e_{t-j}, \qquad \frac{\partial e_t}{\partial p} = -\frac{\alpha}{2}\sum_{j}(1-\alpha)^j\Delta e_{t-j} \approx -\frac{1}{p+1}\sum_j (1-\alpha)^j \Delta e_{t-j}.$$

Consequences:

1. **The logit is almost the log-period.** $\ell = \ln 2 - \ln(p-1)$, so $d\ell = -dp/(p-1)$: a logit
   step $\delta$ moves the period by the relative amount $\approx \delta$. The parametrisation is scale-free,
   which is right for periods.
2. **Where the gradient vanishes.** (a) $p \to 1$ ($\alpha \to 1$): $\alpha(1-\alpha) \to 0$ and the EWMA
   equals the raw close, a channel the layer appends anyway (`learnable_indicators.py:192`); the floor
   2 stops this. (b) $p \gg L$ ($\alpha \to 0$): to first order $e_t \approx x_0 + \alpha \sum_{k \le t}(x_k - x_0)$,
   so $\partial e_t/\partial \ell \approx \alpha \sum_k (x_k - x_0) \to 0$ like $1/p$, and only the product of
   $\alpha$ and the downstream weight is identified (a scale non-identifiability). The ceiling 60 keeps
   $p$ out of that regime, but not far from it: at $p = 60$ the first bar still carries weight
   $(1 - 2/61)^{59} = 0.14$ at the last position, at $p = 35$ it carries 0.034.
3. **Warm-up is inside the input.** All 60 positions of every channel go to the GRU. At position $t$ the
   start weight is $(1-\alpha)^t$, so the early positions of a slow EWMA are "the window's first close
   plus a ramp". The network also sees $x_0$ directly (raw close channel, DIRECTION_SKIP's
   $x_{L-1} - x_0$). A slow period therefore competes with a feature the network already has, which
   predicts weak identifiability of the slow legs (confirmed in section 4).
4. **Adam turns the gradient's sign consistency into speed.** A step moves $\ell$ by at most about
   LR = 0.005. With about 253 steps per epoch (360 days x 1,440 windows / 2,048, **estimate**), the
   ceiling is about 1.27 logit units per epoch. Measured after epoch 0, the median per-epoch logit
   move is 0.014-0.040 (script `snr.py`), 1-3% of the ceiling: the per-step gradients of the periods
   point in inconsistent directions, and the periods drift slowly (epoch 0 moves more: 0.04-0.37).
5. **RSI, BB, soft extrema.** $\partial\,\mathrm{RSI}/\partial\alpha = 100\,(\bar\ell\,\partial\bar g - \bar g\,\partial\bar\ell)/(\bar g+\bar\ell)^2$
   vanishes when the window has no losses (RSI saturates at 100). %B divides by $4\sigma$: in a flat
   window its gradient grows like $1/\sigma^2$. For the soft extremum, $p$ enters only through the fractional
   edge bar, so $\partial(\text{ext})/\partial p$ reads one bar per position, a high-variance gradient
   (NT-047 families; not trained in the six runs).

## 3. The search space the model replaces

### 3.1 Counting

Manual search over integer periods **(arithmetic)**:

| Search | Parameters | Grid | Combinations |
|---|---:|---|---:|
| reference 18 periods | 18 | 2..60 (59 values each) | $59^{18} \approx 7.5 \times 10^{31}$ |
| same, MACD fast < slow enforced | 18 | 2..60 | $59^{9}\,(\binom{59}{2}\cdot 59)^3 \approx 8.9 \times 10^{30}$ |
| reference 18 periods | 18 | 2..200 | $199^{18} \approx 2.4 \times 10^{41}$ |
| D-031 catalogue, 14 families x 3 instances | 54 | 2..60 | $59^{54} \approx 10^{95.6}$ |
| same | 54 | 2..200 | $199^{54} \approx 10^{124.1}$ |
| plus which families are on | | $2^{14}$ subsets, or 0-3 instances each | $\times 16{,}384$ or $\times 4^{14} \approx 2.7 \times 10^8$ |

Parameters per catalogue instance: ma 1, macd 3, rsi 1, bb 1, atr 1, stoch 2, willr 1, keltner 2, obv 1,
vwap 1, mfi 1, adx 1, cci 1, donchian 1 = 18; x 3 instances = 54 (`indicators/families.py`,
`nt-047:indicators/families_ohlcv.py`).

### 3.2 Cost

One 360-day training (long_360d_stab, `status.json`): 816-1,792 s, mean 1,355 s (22.6 min), at
0.299-0.309 s per step; 10-25 epochs with early stopping. (The "14 min" often quoted is the shortest
run, f-2/s0.)

| Strategy | Trainings | GPU time at 22.6 min each |
|---|---:|---:|
| full grid, 18 periods, 2..60 | $7.5 \times 10^{31}$ | $\approx 3 \times 10^{27}$ years |
| one period at a time (coordinate search), 18 x 59 | 1,062 | about 400 GPU-hours, and it ignores interactions |
| random / Optuna search, 100 trials | 100 | about 38 GPU-hours |
| gradient descent (today) | 1 | 22.6 min: all 18 periods move jointly in the same run |

The gradient path adds 18 scalar gradients to a 296,591-parameter backward pass: its extra cost is
negligible **(estimate; not measured separately)**, and a fixed-period layer runs the same EWMA
forward.

### 3.3 What "combinations" means here

A manual search enumerates indicator settings and combination rules (MA-cross of MA $a$ and $b$, RSI
threshold, ...). Here the combination is not enumerated at all: the 31 channels enter the Bi-GRU's
input matrix and everything after it, so any nonlinear function of all channels at all 60 positions
is a candidate. The owner's reading (D-031: "the combination are what the network does
automatically, it's the core of it's mechanic") is exactly this. The comparison with a manual
search therefore has two parts: the periods (a continuous 18-dimensional search, section 3.1) and the
combination rule (a function space with 296k parameters, not countable).

Two limits of the gradient path, both visible in the runs:

- It optimises the training loss (point, direction BCE, NLL, CRPS, soft ECE, physics terms), not
  the yardstick (dev-fold net Sharpe). Nothing guarantees that a better loss means a better Sharpe.
- It is a local search around the textbook initialisation. Across the six runs the largest relative
  distance any period reached from its init was 0.26-1.19 (the `maxrel` column of `ident.py`): the
  runs refine the textbook values; they do not explore the grid.

Whether learning the periods helps at all is the frozen-twin comparison (VISION "The yardstick";
NT-033, todo). No run has made it yet.

## 4. Identifiability

### 4.1 The 18 periods across the six runs

Data: `indicator_params_history.csv` of each run. The served weights are the best-validation epoch
(`status.json` `weights_epoch`, 1-based: 19, 12, 16, 4, 11, 11 for f-3 s0-s2, f-2 s0-s2); the last row
of each history is the restored best epoch (checked: it equals the row `epoch == weights_epoch - 1`
in all six runs). "Served" below is the global period $2/\sigma(\ell)-1$. $t$ = (mean - init) / (sd / $\sqrt 6$).
The six runs are **not independent**: the two folds' 360-day blocks overlap for about 91% of their
bars (**estimate**: the folds are shifted by one 46,544-window dev block, about 32 days), and the seeds are shared across folds, so $t$ overstates the evidence. (NT-074: same-seed runs
already differ at epoch 0 even with op determinism on.)

| Period | Init | Served f-3 s0 / s1 / s2 / f-2 s0 / s1 / s2 | Mean | SD | CV | Move | Sign agree | $t$ | Applied median (dev), mean of runs | Reading |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| ma_period_0 | 5 | 2.86 / 2.00 / 2.80 / 3.44 / 2.00 / 2.37 | 2.58 | 0.56 | 0.22 | -48% | 6/6 down | -10.6 | 2.48 | faster; floor binds in 2 runs |
| ma_period_1 | 10 | 7.07 / 4.91 / 10.01 / 7.54 / 4.97 / 5.53 | 6.67 | 1.97 | 0.30 | -33% | 5/6 down | -4.1 | 6.97 | direction likely, value not |
| ma_period_2 | 30 | 23.0 / 20.4 / 22.5 / 27.3 / 31.5 / 18.9 | 23.9 | 4.7 | 0.20 | -20% | 5/6 down | -3.2 | 25.7 | weak |
| macd_0_fast | 12 | 4.97 / 7.05 / 6.18 / 9.21 / 8.46 / 6.12 | 7.00 | 1.59 | 0.23 | -42% | 6/6 down | -7.7 | 6.60 | faster, identified in direction |
| macd_0_slow | 26 | 32.6 / 50.2 / 52.5 / 29.5 / 33.0 / 42.2 | 40.0 | 9.8 | 0.25 | +54% | 6/6 up | +3.5 | 38.5 | slower; value spread 29-52 |
| macd_0_signal | 9 | 6.37 / 7.27 / 7.79 / 9.69 / 7.24 / 7.22 | 7.60 | 1.12 | 0.15 | -16% | 5/6 down | -3.1 | 6.72 | weak |
| macd_1_fast | 5 | 2.00 / 2.00 / 2.00 / 2.66 / 2.00 / 2.00 | 2.11 | 0.27 | 0.13 | -58% | 6/6 down | -26.4 | 1.85 | at the floor in 5/6 runs |
| macd_1_slow | 35 | 51.9 / 25.1 / 29.2 / 42.9 / 28.7 / 38.7 | 36.1 | 10.3 | 0.28 | +3% | 3/6 | +0.3 | 33.7 | not identified |
| macd_1_signal | 5 | 4.56 / 4.25 / 3.66 / 4.28 / 4.11 / 3.84 | 4.12 | 0.33 | 0.08 | -18% | 6/6 down | -6.6 | 4.05 | identified: about 4 |
| macd_2_fast | 8 | 4.51 / 3.18 / 2.00 / 6.03 / 3.97 / 2.00 | 3.62 | 1.56 | 0.43 | -55% | 6/6 down | -6.9 | 3.43 | faster; value not (2 at floor) |
| macd_2_slow | 17 | 20.8 / 34.7 / 19.7 / 19.1 / 22.2 / 21.9 | 23.1 | 5.8 | 0.25 | +36% | 6/6 up | +2.6 | 24.3 | slower, weak (one outlier) |
| macd_2_signal | 9 | 6.26 / 11.48 / 7.61 / 7.28 / 9.80 / 8.90 | 8.56 | 1.90 | 0.22 | -5% | 4/6 | -0.6 | 8.93 | not identified |
| rsi_period_0 | 9 | 10.02 / 7.22 / 8.51 / 10.07 / 9.60 / 8.68 | 9.02 | 1.10 | 0.12 | 0% | 3/6 | 0.0 | 9.35 | stays at init |
| rsi_period_1 | 14 | 10.78 / 14.55 / 12.29 / 11.70 / 11.77 / 13.16 | 12.38 | 1.32 | 0.11 | -12% | 5/6 down | -3.0 | 11.30 | weak |
| rsi_period_2 | 21 | 26.5 / 22.0 / 23.8 / 23.5 / 23.4 / 26.0 | 24.2 | 1.7 | 0.07 | +15% | 6/6 up | +4.5 | 26.6 | identified: small move up |
| bb_period_0 | 10 | 7.08 / 6.56 / 7.10 / 7.03 / 5.80 / 6.86 | 6.74 | 0.50 | 0.07 | -33% | 6/6 down | -15.9 | 6.60 | identified: about 6.7 |
| bb_period_1 | 20 | 17.1 / 11.7 / 11.7 / 20.2 / 8.4 / 14.2 | 13.9 | 4.2 | 0.31 | -31% | 5/6 down | -3.6 | 14.5 | direction likely, value not |
| bb_period_2 | 25 | 18.3 / 16.1 / 22.8 / 19.6 / 14.6 / 28.3 | 20.0 | 5.0 | 0.25 | -20% | 5/6 down | -2.5 | 19.0 | weak |

(Script `ident.py`; applied medians from `applied.py`, section 4.2.) The reading column uses a rule
fixed for this table: *identified* = 6/6 same sign, $|t| > 3$ and CV < 0.10; *direction* = 6/6 same
sign and $|t| > 3$; *weak* = 5/6 or $|t| < 3$; *not identified* = at most 4/6.

The move vectors (relative move of all 18 periods) correlate 0.58-0.90 between runs; the same seed in
the two folds correlates 0.83-0.90, different seeds 0.58-0.84 (`ident.py`). So the pattern is shared,
and it is also partly set by the seed.

### 4.2 Applied periods and the per-window shift

`applied.py` recomputes, from each run's `weights.h5` (the served weights), the per-window applied
period $2/\sigma(\ell + 0.5\tanh(Wz+c)) - 1$ on every 5th window of the run's dev block (9,309 windows;
the windows are rebuilt from `Bitcoin_BTCUSDT.csv` and match the stored `last_close` exactly). Findings:

- The mean shift per period is small (-0.09 to +0.29 logit units), so the median applied period is
  close to the global one, but not equal: macd_1_fast is 2.0 globally and 1.7-1.8 applied (below the
  floor) in five runs; macd_2_slow in f-3/s1 is 34.7 globally and 42.0 applied.
- The 5-95% range across windows is wide for the slow legs: macd_0_slow f-3/s2 34.1-69.8,
  macd_1_slow f-3/s0 32.8-74.7 (above the 60-bar ceiling and the 60-bar window), rsi_period_2 f-2/s2
  21.9-41.4.
- The global logit $\ell_i$ and the meta Dense bias $c_i$ enter only as $\ell_i + 0.5\tanh(\ldots + c_i)$:
  one direction of the parameter space is not identifiable, and the two halves are trained by
  different optimizers at different learning rates. The logged "learned period" is therefore a
  partial description of what the model applies.

### 4.3 What the six runs say, with the surrogate analysis

1. **Periods are drifting when training stops.** ma_period_0 and bb_period_0 fall almost
   monotonically epoch after epoch (for example f-3/s0 ma_period_0: 4.73, 4.27, ..., 2.73), and the
   median per-epoch logit move is 1-3% of Adam's ceiling (section 2.3). The served value is partly set
   by the early-stopping epoch (served epochs 4-19).
2. **A consistent qualitative pattern:** the fast legs get faster (MA 5 -> 2.6, MACD fasts 12 -> 7,
   5 -> 2.1, 8 -> 3.6), the MACD slow legs get slower (26 -> 40, 17 -> 23), the short Bollinger shortens
   (10 -> 6.7). Pushed to the floor, a fast leg approaches the raw close, and the MACD line becomes
   "price minus a slow average". This agrees with the surrogate analysis (LOG.md Z5,
   `interpret_c1.json`): every model's signal falls with the position in the last hour's range, with
   RSI and with the last minute's return (short-term mean reversion; distance-to-SMA features).
3. **Volatility-type periods are the best identified.** bb_period_0 (CV 0.07) and rsi_period_2
   (CV 0.07) are the tightest. Z5 finds the sigma head reproducible across all six models (log sigma
   ~ 0.32 z(vol_60) + 0.10 z(vol_15), coefficient correlation 0.88-0.99), while the direction pattern
   beyond mean reversion is set by the seed (same seed across folds 0.78-0.92, across seeds
   -0.74..+0.42). The periods behave the same way: seed-shared, partly seed-specific.
4. **The slow MACD legs and two signal periods are not identified** (macd_1_slow 3/6, macd_2_signal
   4/6, CV 0.22-0.28), as section 2.3 point 3 predicts: within a 60-bar window a 30-50-bar EWMA is
   close to the window-start anchor that the network already reads.

**Conclusion.** The data identify a direction of movement for about half of the periods (faster
fast legs, slower slow legs, a shorter Bollinger), a value only for bb_period_0 (about 6.7), macd_1_signal
(about 4) and rsi_period_2 (about 24), and nothing for macd_1_slow, macd_2_signal and rsi_period_0.
Three fast legs sit at the floor of 2, so their optimum is below the allowed range. None of this shows
that the learned periods predict better than the textbook ones: that is NT-033's question.

## 5. Generality

### 5.1 Units

| Field or constant | Unit today | Where |
|---|---|---|
| LOOKBACK | bars | `core/config.py:226` |
| HORIZON_STEPS, EXTENDED_TREND_PERIODS | bars | `:255, 258` |
| indicator periods, MOMENTUM_CLIP_MIN / MAX | bars (MAX defaults to LOOKBACK) | `:411-412, 522-523` |
| SKIP_LAGS (1, 5, 10, 15, 20, 30) | bars, hard-coded | `gru_attention.py:20` |
| VAL_FRACTION, CAL_FRACTION | fractions of the sequences | config |
| number of horizons | exactly 3 | `core/config.py:577` |
| annualisation | 525,600 / bar minutes; the engine's scorer passes the bar size, the frozen set does not | BACKLOG NT-040 |

NT-040 (annualisation), NT-041 (dataset spec, wall-clock window and horizons, blocks in time) and
NT-042 (N horizons) are all `todo` (BACKLOG). Until NT-041, a learned period means "bars of this
bar size"; a 1-minute result transfers to 5-minute bars only by dividing by 5, and only if the dynamics
are self-similar across time scales, which nothing has tested.

### 5.2 Scale

The inputs and targets are in units of one global dollar scale $s$ per training block (section 1.1).
Over the 14 months before the fold -2 dev block, the monthly standard deviation of 10-minute changes
ranged 105.6-246.4 dollars, while in basis points it ranged 13.7-29.4 (script `scale.py`; for example
June 2024 16.0 bps = 105.6 dollars at a price of 66k, June 2025 14.1 bps = 148.4 dollars at 106k).
So the input magnitudes carry the price level as well as the volatility: the same relative move is
1.4x larger in input units a year later. The variance head must learn this level dependence, and
the meta_adjust input (the window's mean and max in these units, `gru_attention.py:48-52`) shifts the
periods partly by price level. Scale-dependent constants: the MACD soft cross $\tanh(10\,\text{hist})$
(`families.py`, MACDFamily.outputs) and NT-047's SOFT_SIGN_SHARPNESS 10 on $\Delta x$ are in these units;
the soft extremum was made scale-free in NT-047's repair (BETA relative to the window's range).

### 5.3 What would break for another ticker or bar size

- **Nothing refuses it**, but: block 6 and block 14 fix the window length; three towers are
  hard-coded; SKIP_LAGS are bars; the period bounds are bars; the $L^2$ memory limits the window
  (section 1.3); the soft-sign constants assume this scale; Sharpe annualisation is right in the
  engine only.
- For a lower-priced or less volatile instrument, $s$ adapts globally, but any within-block trend in
  price level has the effect described in 5.2.

### 5.4 The window-free plan (docs/research/2026-09-29-window-free-plan/README.md)

Key conclusions: a series engine (option A2) computes the indicators in one causal pass over the
series, so there is no cold start per window and no $L^2$ memory, while the network still reads the
last 60 bars; estimated at -3 to +3 ms on a 98 ms step. Removing the 60-bar clip is a separate A/B
(A/B-1b). "Unlimited" periods remain bounded by the data (about 360 bars for a 2,048-bar burn-in).
The per-window shift becomes per bar ($\alpha_{i,t} = \sigma(\ell_i + 0.5\tanh(\mathrm{ctx}_t W + b)_i)$),
and reporting gives the per-bar p5/p50/p95 and the effective period. A/B designs need many folds with
one seed each. The build-up is research track R6, after the MVP (D-039).

## 6. Screen campaign: what mattered for stability

Level-1 screen, block A (`runs/screens/l1_A_hyper`, 264 trials: LHS over LR $10^{-4}$-$10^{-2}$ log,
ADAM_BETA1 0.80-0.95, ADAM_BETA2 0.990-0.9999, INDICATOR_LR_MULT 1-20 log, plus a GRAD_CLIP_NORM grid
5 / 20 / 100; batch 64, 8 epochs, 4 slices x 2 seeds; rules in the spec). Script `screenA.py`:

- **No trial was non-finite** (0 non-finite gradient steps in 264). Every one of the 62 failures is
  `clipped_share > 0.5`: more than half of the steps after epoch 1 had a gradient norm above the clip.
  So these failures are about the clip threshold relative to the gradient norm, not divergence.
- **GRAD_CLIP_NORM decides it directly:** pass rate 0/8 at 5, 5/8 at 20 (the default), 8/8 at 100,
  at a median gradient norm of 19-21 in all three.
- **LR matters, in the opposite direction to intuition:** pass rate 61% at LR <= 3e-4, 69% at 3e-4..1e-3,
  84% at 1e-3..3e-3, 100% at 3e-3..1e-2; median gradient norm 20 -> 5.8 in the top bin (Spearman
  -0.60 between log LR and the mean gradient norm). Why a higher LR gives smaller gradients was not
  investigated (**unverified**).
- **INDICATOR_LR_MULT has no measurable effect** (pass rate 73-84% across its bins, Spearman -0.17
  with the clipped share); ADAM_BETA1 0.90-0.95 passes more (90% against 73%), ADAM_BETA2 no effect.
- **The seed matters more than any hyperparameter:** seed 0 passes 97% of its trials, seed 1 56%
  (median gradient norm 16.2 against 21.7).
- The screen's direction AUCs rest on n_eff 20-37 and say nothing.

Other blocks, for reference only (another report covers the losses): B loss weights 61/328 pass
(245 clipped_share, 22 also max_term_share; median gradient norm 52), C physics 212/288, D loss choice
50/64, E maths 12/16; all failures are clipped_share or term share.

## 7. Problems and recommendations

Each item: the problem, the evidence, the recommendation. None is a decision; they are candidates for
the backlog.

### Architecture

1. **Block 6 is not a cross-indicator attention and ties the model to the window length.**
   Evidence: `gru_attention.py:69-73` (tokens = 128 GRU units, features = 60 time positions);
   31,164 parameters, $513L + 384$. Recommend: either attend across the 31 indicator channels before
   the GRU (tokens = channels, a shared per-channel time embedding), or remove the block in an A/B;
   replace Flatten -> Dense (`:157-161`, $512L + 32$ parameters) by pooling or the last state. Both are
   prerequisites for window-free or variable-window runs.
2. **Capacity is out of proportion to the signal.** Evidence: 296,591 parameters, 44.6% in one
   8x32 attention over a 128-wide sequence; the network is significantly below a 3-lag logistic
   regression at 1 h (LOG.md L2, pooled z -3.3..-3.5) and never above it (STATUS). Recommend a
   pre-registered capacity study: indicators -> small GRU (or linear) -> heads against today's stack,
   judged by the paired comparator (D-025).
3. **The OOM at longer windows has two quadratic sources.** Evidence: section 1.3 shapes; the EWMA
   matrix `utils/math.py:221-241` is $[B, 18, L, L]$ (8.5 GB per copy at $B = 2048, L = 240$,
   **estimate**). Recommend: record the EWMA matrix next to the attention in LOG.md H4's reading and
   in the NT-038 config guard (refuse $B \cdot K \cdot L^2$ above a measured bound); the series engine of
   the window-free plan removes it.
4. **Stale comments.** `gru_attention.py:183, 214, 235` label the towers 1-, 5- and 15-minute; the
   reference horizons are 10, 15 and 20 bars. The direction clip `:198-201` is a no-op.

### Indicator parametrisation

5. **Bounds apply to the global logit only.** Evidence: `learnable_indicators.py:98-103, 224-238`;
   applied periods 1.6-1.8 below the floor of 2 and up to 74.7 above the ceiling of 60 on the dev blocks
   (section 4.2). Recommend: bound the applied value (clip $\ell + \text{shift}$ into the logit range, or
   scale the shift so the applied period stays inside [MIN, MAX]), and state in the telemetry that the
   logged period is the base value.
6. **The base logit and the meta bias are redundant.** Evidence: $\alpha = \sigma(\ell + 0.5\tanh(Wz + c))$,
   `gru_attention.py:53-54`, `learnable_indicators.py:103`; the two are trained by different
   optimizers (LR 0.005 and 0.001-0.000125). Recommend: `use_bias=False` on the meta Dense (or centre its
   input), so that $\ell$ alone carries the mean period and the reported period means what it says. A
   change of the default path: needs a golden-run record and the D-018 check.
7. **Report the applied periods.** Evidence: `get_learned_parameters` (`learnable_indicators.py:206-214`)
   and `indicator_params_history.csv` hold only $2/\sigma(\ell) - 1$. Recommend: per instance, the p5 / p50 /
   p95 applied period on the evaluation block, as the window-free plan's reporting section already
   specifies for series mode (B/ Q3); `applied.py` shows it takes a CPU pass over the stored windows.
8. **The fast legs want to be the raw price.** Evidence: macd_1_fast at the floor in 5/6 runs,
   ma_period_0 and macd_2_fast in 2/6 each (section 4.1); $p \to 1$ reproduces the appended raw close.
   Recommend: parametrise MACD as a slow EWMA plus a ratio $r \in (0, 1)$ with fast $= r \cdot$ slow (this also
   removes the unenforced fast < slow ordering, a mirror symmetry up to the sign of the downstream
   weights), and let a fast leg reach $p = 1$ explicitly instead of clipping at 2.
9. **INDICATOR_GRAD_MULT does nothing useful under Adam.** Evidence: `learnable_indicators.py:100-103`,
   Adam's scale invariance (`training/optim.py:3-5`); it only raises the indicator group's clip rate.
   Recommend: set it to 1 or remove it (a config change; D-029 evidence needed for a removal), and use
   INDICATOR_LR_MULT as the one control. The screen found no stability effect of INDICATOR_LR_MULT in
   1-20.
10. **The indicator LR never decays.** Evidence: `training/callbacks.py:380` acts on the main
    optimizer; `log_lr_indicator_used` = 0.005 in every epoch of all six runs while the main LR fell to
    0.000125-0.0005. Recommend: apply the plateau schedule to both optimizers (or a fixed ratio), so the
    periods settle instead of drifting until early stopping cuts them.
11. **The period gradients are mostly noise.** Evidence: per-epoch logit moves 1-3% of Adam's
    ceiling (section 2.3, `snr.py`); monotone drifts truncated at the served epoch (4.3 point 1).
    Recommend: log the per-parameter gradient signal-to-noise ratio (mean / SD of the per-step
    gradient over an epoch) in telemetry; then decide between a larger effective batch for the
    periods, a period warm-up then freeze, or keeping them fixed. The frozen-twin A/B (NT-033) decides
    whether learning them is worth anything.
12. **The warm-up is inside the input.** Evidence: section 2.3 point 3 (start weight 0.14 at $p = 60$,
    0.034 at $p = 35$ at the LAST position, larger earlier); slow legs not identified (4.1). Recommend:
    until the series engine exists, mask or drop the first $M(\epsilon)$ positions per channel (M(eps) is
    already computed per family, `indicators/base.py:101-110, 184-191`), or give the GRU a per-channel
    "warm" flag.

### Universalisation

13. **Wall-clock units** (NT-041): SKIP_LAGS (`gru_attention.py:20`) is not even a config field;
    make it one, in wall-clock minutes or as fractions of the horizon, together with the NT-041
    fields. Report learned periods in minutes as well as bars.
14. **Scale-free inputs.** Evidence: section 5.2 (dollar scale moves with price level: 105.6 against
    148.4 dollars at 16.0 against 14.1 bps). Recommend a pre-registered A/B of a per-window volatility
    normaliser (divide the window by its own realised sigma, feed the sigma as a separate feature) against
    `window_relative`, keeping the dollar target (D-022) by rescaling at the output. Make the soft-sign
    sharpness relative to local volatility, as NT-047's repair did for the soft extremum.
15. **N horizons** (NT-042) and **annualisation** (NT-040) are prerequisites for any second setup;
    both `todo`.

## Scripts (scratch, outside the repo)

`D:/nt_math_scratch/`: `build_summary.py` (model summary on CPU), `ident.py` (section 4.1),
`applied.py` (4.2), `snr.py` (2.3 point 4), `scale.py` (5.2 and the LR schedule), `screenA.py` (6).
Each reads only committed run files, `Bitcoin_BTCUSDT.csv` and the code.
