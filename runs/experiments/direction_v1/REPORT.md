# Why the direction heads did not learn, and what changed

Milestone M3 failed: after the Phase A fixes the model trained, but its three P(up) heads stayed
within about 0.49-0.51 (test AUC 0.47-0.50) and no strategy ever placed a trade. This note records
the diagnosis and the experiments behind the two changes made: a proper direction loss and a
linear direction skip. Scripts: `scripts/direction_experiments.py`. Per-run numbers:
[`summary.md`](summary.md). Run ids are listed there (run directories are kept locally).

## 1. Is there any direction signal to learn?

Labels are up/down beyond a 5 bps deadband, 10/15/20 bars ahead. A logistic regression on
trailing returns (1-60 bars) and 1-bar volatility is fitted on each walk-forward fold's train
block and scored on that fold's test block. The 95% intervals come from a block bootstrap with
80-bar blocks:

| fold | h0 | h1 | h2 |
|---|---|---|---|
| -4 | 0.565 [0.535, 0.601] | 0.563 [0.531, 0.601] | 0.560 [0.520, 0.606] |
| -3 | 0.497 [0.458, 0.535] | 0.533 [0.497, 0.575] | 0.557 [0.516, 0.608] |
| -2 | 0.506 [0.469, 0.553] | 0.513 [0.469, 0.561] | 0.524 [0.477, 0.573] |
| -1 | 0.554 [0.522, 0.587] | 0.562 [0.520, 0.600] | 0.567 [0.528, 0.608] |

(A second logistic-regression run on fold −3 got 0.535 / 0.544 / 0.521. That is the reference
used in `summary.md`.) The signal is weak and depends on the period: fold −2 is at chance. It is
a mix of mean reversion over 5-20 bars and momentum over 30-60 bars. All of it is a *linear*
function of the 60-bar input window. A gradient-boosted model does no better (0.48-0.52).
Pooling the folds into one AUC gives 0.49, because each fold's model has its own score scale;
that pooled number is not evidence of "no signal". The calibration block, about 190 independent
h1 outcomes, is too small to rank models by AUC. Model choices below were therefore made on
folds −3 and −2 only. Fold −1, the block every gate reports, was not used for any choice.

## 2. The direction loss was improper

The heads were trained on `0.5 focal(gamma=2) + 0.5 dice`.

* **Dice** on probabilities has per-sample losses (1-p)/(p+2) for an up label and p/(p+1) for a
  down label. Its expected value under y ~ Bernoulli(q) has an interior *maximum*, so the
  optimum is always p = 0 or p = 1. When samples look alike, dice pushes the head to a constant
  extreme. This matches the 0% / 100% predicted up-rates seen since the very first broken run.
* **Focal** loss is BCE minus an entropy bonus, which rewards predictions near 0.5.

Replaced by binary cross-entropy (`Config.DIRECTION_LOSS = "bce"`), a strictly proper scoring
rule. `tests/test_losses_reference.py` proves both properties: BCE's optimum is p = q, dice's is
an extreme.

## 3. The deep path does not carry the signal to the heads

With BCE alone the heads stop sitting at an extreme but stay nearly constant (P(up) s.d. about
0.002-0.010, AUC about 0.50). That is the correct BCE answer when the head's inputs carry no usable
signal. The window reaches the heads only through learnable indicators, a Bi-GRU, two attention
stages, convolutions and two transformer blocks, and the weak linear signal does not survive.
Removing soft-ECE lets the heads vary (s.d. about 0.02-0.03) but they fit noise (AUC ≤ 0.5).

`Config.DIRECTION_SKIP` adds a regularised linear logit to each direction head. It reads the
window's trailing changes over 1-30 bars and over the whole window, plus the log of the 1-bar
volatility.

## 4. Results on the development folds (batch 256, 20 epochs max)

Mean over folds −3 and −2 × three horizons:

| variant | mean AUC | logistic regression | AUC > 0.5 | mean MCC |
|---|---|---|---|---|
| legacy focal + dice (fold −3 only) | 0.490 | 0.533 | 0/3 | -0.006 |
| BCE | 0.496 | 0.515 | 2/6 | -0.004 |
| **BCE + skip (L2 1e-4)** | **0.514** | 0.515 | **5/6** | **+0.023** |
| BCE + skip (L2 1e-2) | 0.509 | 0.515 | 4/6 | +0.013 |

Chosen defaults: `DIRECTION_LOSS = "bce"`, `DIRECTION_SKIP = true`, `DIRECTION_SKIP_L2 = 1e-4`.
The network now matches the linear baseline on average. It beats the baseline on fold −2 h1/h2
and on fold −3 h0, and falls short on fold −3 h1. It does not go beyond it.

## 5. What this does and does not establish

* The heads now carry real direction information at about the level of the best simple
  baseline. The judgement on the untouched fold −1 is the M3 gate run (`runs/gates/REPORT.md`).
* AUC about 0.51-0.52 on 10-20 minute horizons is a weak edge. Each backtest trade pays 26 bps
  round trip, and a 15-minute move has a standard deviation of about 21 bps, so a positive
  net-of-cost backtest should not be expected from these heads alone.
* Batch 256 was used for speed (the step is kernel-launch bound). The gate re-run uses the
  default batch of 64.
