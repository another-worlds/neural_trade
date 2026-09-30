# A. The training loss: terms, gradients, bounds, stability, solvability

Research note, 2026-09-30. Read-only analysis of `remediation/plan` at 3ef6047. Every formula below was
read from the code at the cited line. Derivations are mine; the ones marked **(derived, not checked
against a test)** have not been checked by a test or a numeric probe. Numbers come from committed runs
or from the scratch scripts in `D:/nt_math_scratch/` (`screens.py`, `leader.py`, `probe.py`, `solv.py`;
their outputs are `*_out*.txt` and `probe_*.txt` there). The scratch folder is not in git.

Notation: $B$ is the batch size. $y$ is the scaled target delta, $(\text{raw delta}-\bar m)/s$, where $s$ is
`pred_scale` and $\bar m$ is `pred_mean` (`core/config`, `data/scaling.py:22-30`). $\mu$ is a price head
(scaled delta). $v$ is a variance head. $p$ is a direction head. $z$ is the direction logit. $e = y-\mu$
is the residual. $m_i \in\{0,1\}$ is the deadband mask. $h\in\{0,1,2\}$ is the horizon
(10/15/20 bars in the reference setup).

## 0. Heads and where the loss lives

- **Objective.** The objective is `custom_loss` (`losses/functions.py:490-810`), or `pnl_utility`
  (`:813-919`), which adds a P&L term to it. `Config.LOSS_NAME` picks the objective, and its default is
  `custom_loss` (`core/config.py:477`). `CustomTrainModel.train_step` calls the objective
  (`training/custom_model.py:468-567`).
- **Price heads.** Each is `Dense(1)`, linear. A Lambda clips it to $[-100,100]$ and replaces a
  non-finite value with 0 (`models/gru_attention.py:186-194`, `:217-221`, `:238-242`). `clip_by_value`
  has zero gradient outside the range.
- **Direction heads.** The head computes $p=\sigma(z)$ from a tower logit plus an optional linear skip
  logit. The skip takes trailing-return features and carries an L2 penalty of `DIRECTION_SKIP_L2` = 1e-4
  (`gru_attention.py:33-40`). A Lambda clips $p$ to $[0,1]$ (`:196-201`). The loss receives
  **probabilities**, not logits.
- **Variance heads.** Each is `Dense(1, softplus)`, $v=\mathrm{softplus}(a)$, with bias initialised at
  1.0, so $v_0\approx1.31$. The dense layer also sees `perp_magnitude` and `regime_gate`. A non-finite
  value becomes 1 (`gru_attention.py:204-212`). In the loss, $v_c=\max(v, \texttt{VAR\_FLOOR}=10^{-4})$
  (`functions.py:636-643`). There is no upper cap in the loss: `VAR_CAP` is read and unused
  (`:637`). **This is a softplus parametrisation, not log-variance.** That matters for the NLL bound
  (section 5).
- **Sanitising.** `custom_loss` replaces every non-finite head with 0 / 0.5 / 1 (`:519-530`). Nearly
  every term is wrapped in `tf.where(is_finite(term), term, 0)`, and the total is wrapped too (`:790`).
- **Direction labels.** $y^{dir}=1[r>d]$ and $m=1[|r|>d]$, where $r$ is the raw return and $d$ =
  `DIR_DEADBAND_BPS` = 5 bps (`metrics/tf_direction.py:17-35`).

The total is (`functions.py:771-789`):

$$L=\underbrace{\textstyle\sum_h\lambda^{pt}_h\,\ell^{pt}_h}_{\text{point}}+\lambda_{tr,out}\textstyle\sum_h\lambda_{ext}\ell^{ext}_h
+\lambda_{dir,out}\lambda_{dir}\textstyle\sum_h \ell^{bce}_h+\lambda_{al,out}\ell^{align}+0\cdot R+0.1\,\lambda_{inter}R
+0.1\,\lambda_{vol}\ell^{vol}+\lambda_{coh}\ell^{coh}+\lambda_{nll,out}\lambda_{var}\textstyle\sum_h\ell^{nll}_h
+\lambda_{crps}\textstyle\sum_h\ell^{crps}_h+\lambda_{ece}\textstyle\sum_h\ell^{ece}_h+\lambda_{tp}\textstyle\sum_h\ell^{tp}_h+\lambda_{cas}\ell^{cas}+\ell^{vac}+\lambda_{hd}\ell^{hd}+\lambda_{ife}\ell^{ife}+\lambda_{vo}\ell^{vo}$$

**How the weights are set.** The weights are live `tf.Variable`s (`training/lambdas.py:15-49`). By
default the pre-training calibration pass runs (`training/lambda_calibration.py`). It sets 13 weights to
1, measures the median of each component over 10% of an epoch, and applies
$\lambda\leftarrow\mathrm{clip}(\lambda_0(\bar L/\mathrm{med})^{d},0.1,20)$ (`:207-210`). Here
$\bar L$ is the mean of the active medians and $d$ is the damping.

- The default damping is 1 (`CALIB_DAMPING`).
- The trend weight and the physics weights have damping 0, so they keep their configured values
  (`config.py:292,304`).
- The outer multipliers are rescaled only when `CALIB_OUTER` is set (default off).
- **The pass equalises loss values, not gradients.** Section 7 returns to this.

**The leader run.** The leader is `runs/scenarios/long_360d_stab/20260930T094257Z-dce15ed-e3669618-default__f-2__s0`:
fold −2, seed 0, batch 2048, `SHUFFLE_BUFFER` 0, served epoch 4 (1-based). Its calibrated weights, from
`artifacts/meta.json` `lambda_values_final`:

| Weight | Value |
|---|---|
| $\lambda_{short}$ | 0.644 |
| $\lambda_{point}$ | 1.703 |
| $\lambda_{long}$ | 1.542 |
| $\lambda_{dir}$ | 0.749 |
| $\lambda_{var}$ | 0.382 |
| $\lambda_{vol}$ | 1.817 |
| $\lambda_{crps}$ | 0.896 |
| $\lambda_{ece}$ | **2.907** |
| $\lambda_{ext}$ | 0.1 |
| physics weights | 0.1 each |
| outer multipliers | 1 each |

## Per-term shares in the leader run

The table below gives each term's weighted value and its share of the total. It comes from
`training_log.csv` with the lambdas above (`D:/nt_math_scratch/leader.py`).

- The logged `dir_loss`, `nll_loss`, `crps_loss` and `soft_ece_loss` are unweighted, so the script
  multiplies them by their weights. `vol_loss` and `inter_reg` get the extra 0.1.
- The coherence penalty is not logged. It is the residual: total minus the sum of the others.
- Validation values are exact. Training term values are sampled every 10 steps; the training total is
  exact.

| Term | val ep 1 | share | val ep 4 (served) | share | train ep 1 share | train ep 4 share |
|---|---|---|---|---|---|---|
| point (log-cosh) | 1.022 | 16.4% | 1.021 | 17.2% | 18.1% | 19.5% |
| extended trend | 0.074 | 1.2% | 0.074 | 1.3% | 1.4% | 1.4% |
| direction BCE | 1.577 | 25.4% | 1.559 | 26.3% | 24.1% | 25.2% |
| Gaussian NLL | 1.311 | 21.1% | 1.272 | 21.5% | 21.0% | 21.1% |
| CRPS | 1.153 | 18.6% | 1.145 | 19.3% | 18.9% | 20.0% |
| soft ECE | 0.631 | 10.2% | 0.467 | 7.9% | 11.1% | 3.6% |
| vol (0.1·λ·\|Δstd\|) | 0.133 | 2.1% | 0.136 | 2.3% | 2.2% | 2.3% |
| inter_reg | 0.0000 | 0% | 0.0000 | 0% | 0% | 0% |
| T-perp | 0.024 | 0.38% | 0.016 | 0.27% | 0.25% | 0.11% |
| Casimir | 0.0007 | 0.01% | 0.0001 | 0.00% | 0.01% | 0.01% |
| vacuum bandwidth (off) | 0 | 0 | 0 | 0 | 0 | 0 |
| HD | 0.011 | 0.17% | 0.015 | 0.25% | 0.27% | 0.14% |
| IFE | 0 | 0 | 0 | 0 | 0 | 0 |
| vacuum overflow | 0 (val: forced 0) | 0 | 0 | 0 | 1.1% | 1.5% |
| coherence (residual) | 0.282 | 4.5% | 0.224 | 3.8% | 1.6% | 5.1% |
| **total** | **6.216** | | **5.927** | | 7.121 | 6.207 |

The loss **values** are spread over five terms: BCE, NLL, CRPS, point and soft ECE. The **gradient** is
not spread; section 7 shows it.

---

## 1. Point loss: log-cosh

- **Formula** (`functions.py:121-138`, helper `:22-31`): $\ell^{pt}_h=\frac1B\sum_i \log\cosh(y_i-\mu_i)$.
  It is implemented as $x+\mathrm{softplus}(-2x)-\log 2$, which is finite for every float32 value.
  Weights $\lambda_{short},\lambda_{point},\lambda_{long}$ apply to h0, h1, h2 (`:532-535`).
- **Gradient:** $\partial\ell/\partial\mu_i=-\tanh(e_i)/B$, so $|\cdot|\le 1/B$. This is a **provable
  bound**, and a test covers it (`tests/test_custom_loss.py:48-60`). Above $|\mu|=100$ the head clip
  zeroes the gradient.
- **Proper?** It is a point loss, not a distributional score. Its Bayes act is the
  $\arg\min_c E[\log\cosh(Y-c)\mid x]$ location functional. That functional sits between the mean
  (small residuals) and the median (large residuals), and equals both for a symmetric conditional law.
- **Default weight** 1 per horizon, calibrated to 0.644 / 1.703 / 1.542 in the leader. Share 16-20%.

## 2. Trend terms

- **Local and global trend** are retired. They are the constant 0 in the objective (`:541-557`) because
  they reduced algebraically to the point loss.
- **Extended trend** (`:160-197`): $\ell^{ext}_h=\frac1B\sum_i\log\cosh\big(\mu_{i,h}-\tfrac{\Delta^{past}_{i,h}-\bar m}{s}\big)$.
  $\Delta^{past}$ is the realised price change over `EXTENDED_TREND_PERIODS[h]` bars (10/15/20).
- **Gradient:** $\tanh(\cdot)/B$, so $|\cdot|\le\lambda_{ext}/B$. Provable.
- **Proper?** No. It is a **momentum prior**: it pulls $\mu$ toward the last 10-20-bar move. Together
  with the point loss, the minimiser is shrunk from the log-cosh location toward
  $\Delta^{past}$. It is biased unless the future move's location equals the past move.
- **Weight** 0.1, with calibration damping 0. Share about 1.2%.

## 3. Coherence penalty (not logged)

- **Formula** (`:566-594`): $\ell^{coh}=\frac13[\text{dir\_disagree}+\text{magnitude}+\text{target\_smoothness}]$,
  weight $\lambda_{coh}$ = `LAMBDA_COHERENCE` = 1 (not small).
  - `dir_disagree` uses `tf.sign` of the price heads, so its **gradient is 0 almost everywhere**.
  - `target_smoothness` depends only on the labels. It is a **data constant with zero gradient**.
  - Only $\text{magnitude}=\frac1B\sum_i[\mathrm{relu}(|\mu_{i0}|-|\mu_{i1}|)+\mathrm{relu}(|\mu_{i1}|-|\mu_{i2}|)]$
    has a gradient. Per example it is $\pm1/(3B)$ on the price heads where the ordering is violated.
- **Bound:** at most $2/(3B)$ per head per example. Provable.
- **Proper?** No. It imposes $|\mu_{h0}|\le|\mu_{h1}|\le|\mu_{h2}|$. That holds for the conditional
  mean under a drift model. It does not hold in general, for example under mean reversion.
- **Measured:** the probe (section 7) gives it a gradient norm of 0.83-0.93, about as large as the
  point loss's 0.54-1.02. It is not logged, so the screens' `loss_term_shares` omit it; `screen.py:731`
  calls it "small", which the numbers here contradict.

## 4. Direction BCE (default) and focal+dice (legacy)

- **BCE formula** (`:83-97`, used `:622-634`):
  $\ell^{bce}_h=\sum_i m_i\,[-y_i\log\tilde p_i-(1-y_i)\log(1-\tilde p_i)]/(\sum_i m_i+10^{-8})$,
  with $\tilde p=\mathrm{clip}(p,10^{-7},1-10^{-7})$.
- **Gradient with respect to the logit.** When $p$ is inside the clip, $\partial\ell/\partial z_i=m_i(p_i-y_i)/\sum m$
  and $|\cdot|\le 1/\sum m$. **Provable** (the sigmoid-then-log chain gives exactly $p-y$).
- **Saturation.** If $\sigma(z)$ rounds past $1-10^{-7}$ (float32, about $|z|>16$), the clip zeroes
  the gradient. Saturated wrong predictions then get **no** gradient. This is a vanishing-gradient
  risk, not an exploding one. In the probe batches $p\in[0.09,0.89]$, far from saturation.
- **Proper?** Yes, strictly proper. The population minimiser is
  $p^*(x)=P(r>d\mid |r|>d,x)$: the deadband mask conditions on a move. Tests cover it:
  `tests/test_losses_reference.py:180` (BCE is proper) and `:186` (dice is improper).
- **Weight:** $\lambda_{dir}=1$, calibrated to 0.749; outer 1. Share about 25%.
- **focal_dice** (`DIRECTION_LOSS: focal_dice`, `:34-80`, `:100-105`, `:612-621`). The focal and dice
  terms operate on probabilities clipped to $[10^{-7},1-10^{-7}]$, so the gradient is finite. The
  combination is **improper**: the docstring at `:87-91` states that dice's expected loss has an interior
  maximum, so its optimum is an extreme; the test at `:186` confirms it. Block D of the screen: focal_dice
  passes 28 of 32 trials, bce passes 22 of 32. The difference comes from `clipped_share` (section 8), not
  from finiteness.

## 5. Gaussian NLL

- **Formula** (`:645-657`):
  $\ell^{nll}_h=\frac1B\sum_i\big[\tfrac12\log 2\pi+\tfrac12\log(v_{c,i}+10^{-8})+\tfrac{e_i^2}{2(v_{c,i}+10^{-8})}\big]$,
  with $v_c=\max(\mathrm{softplus}(a),10^{-4})$.
- **Gradients (derived; the softplus facts are elementary):**
  - With respect to the mean: $\partial/\partial\mu_i=-e_i/(v_iB)$, so $|\cdot|\le|e_i|\cdot10^4/B$.
  - With respect to the variance: $\partial/\partial v_i=\frac{1}{2v_iB}(1-e_i^2/v_i)$.
  - With respect to the pre-activation, using $dv/da=\sigma(a)$ and the inequality
    $\sigma(a)\le\mathrm{softplus}(a)$ (that is, $x/(1+x)\le\log(1+x)$):
    $$\Big|\frac{\partial\ell}{\partial a_i}\Big|=\frac{\sigma(a_i)}{2v_iB}\,|1-e_i^2/v_i|\le\frac{1}{2B}\Big(1+\frac{e_i^2}{v_i}\Big)\le\frac{1}{2B}(1+10^4e_i^2).$$
  - Below the floor, `tf.maximum` sends no gradient to $a$.
  - With $|\mu|\le100$, $e$ is bounded by $100+\max|y|$, so the gradient is **bounded but not usefully
    bounded**: of order $10^8/B$ for $|e|\sim100$.
- **Contrast with a log-variance head.** With $v=e^{s}$ the gradient is $\tfrac12(1-e^2e^{-s})$, which
  is unbounded as $s\to-\infty$. Softplus plus the floor removes that. The remaining risk is
  $e^2/v$ for a large residual on a small-variance example. In the probe batches $\min v = 0.07$, so the
  floor was never active.
- **Proper?** Yes, the log score. Its population minimiser over $(\mu,v)$ is
  $\mu=E[Y|x]$, $v=\mathrm{Var}[Y|x]$ for any conditional law (derived: moment matching). It is
  **not robust**: the $\mu$-gradient grows linearly in $e$ and is inverse-weighted by $v$.
- **Weight:** $\lambda_{var}=1$, calibrated to 0.382; outer 1. Share about 21% of the value.
- **Measured:** gradient norm 0.63 / 2.84 / 4.61 in the three probe batches (section 7). It is the most
  batch-sensitive term, as the $e^2/v$ tail predicts.

## 6. CRPS (Gaussian)

- **Formula** (`:200-226`): with $\sigma=\sqrt{\max(v_c,10^{-8})}$ and $\omega=e/(\sigma+10^{-8})$,
  $\mathrm{CRPS}=\sigma[\omega(2\Phi(\omega)-1)+2\varphi(\omega)-1/\sqrt\pi]$, clipped to $[0,100]$ (the
  clip zeroes the gradient above 100, which is unreachable at these scales).
- **Gradients** (standard, derived; `tests/test_losses_reference.py:148` checks properness, not the
  derivatives):
  - $\partial/\partial\mu=-(2\Phi(\omega)-1)\in[-1,1]$.
  - $\partial/\partial\sigma=2\varphi(\omega)-1/\sqrt\pi\in[-0.564,\,0.234]$.
  - Chaining through $\sigma=\sqrt v$ and softplus:
    $|\partial/\partial a|\le0.564\cdot\sigma(a)/(2\sqrt v)\le0.282\min(\sqrt v,1/\sqrt v)\le0.282$.
    This uses $\sigma(a)\le v$ and $\sigma(a)\le1$.
  - **All bounded by constants independent of the data. Provable.** This is the best-behaved term.
- **Proper?** Yes, strictly proper for distributions. Within the Gaussian family the optimum is the
  CRPS-projection. For a symmetric law $\mu$ is the median, which equals the mean.
- **Weight** 1, calibrated to 0.896. Share about 19%.

## 7. Soft ECE (the dominant gradient)

- **Formula** (`:229-259`): bins $b=1..10$ with centres $c_b$ and bandwidth $h_{bw}=0.05$ (so
  $2h_{bw}^2=0.005$). The weights are $w_{ib}=m_i\exp(-(p_i-c_b)^2/2h_{bw}^2)$ and $W_b=\sum_iw_{ib}$.
  Then $a_b=\sum w y/W_b$, $f_b=\sum w p/W_b$, and
  $\ell^{ece}=\sum_b\frac{W_b}{N}|a_b-f_b|$ with $N=\sum m$.
- **Gradient (derived, not checked against a test).** Let $D_b=a_b-f_b$ and
  $w'=\partial w/\partial p=-w(p-c_b)/h_{bw}^2$. Then
  $$\frac{\partial\ell}{\partial p_i}=\frac1N\sum_b\Big[w'_{ib}|D_b|+\mathrm{sign}(D_b)\big(w'_{ib}(y_i-a_b-p_i+f_b)-w_{ib}\big)\Big].$$
- **Bound.** Summed over bins, $\sum_b|w'_{ib}|\approx 2/0.1=20$ (the Gaussian first absolute moment
  over bin spacing 0.1) and $\sum_b w_{ib}\approx1.25$. So $|\partial\ell/\partial p_i|\lesssim 61/N$.
  BCE's is $|p-y|/(p(1-p)N)\approx2/N$ near 0.5. The per-example sensitivity is up to about 30x BCE's.
  The bound is finite; it holds provided the softmax-free normalisation $W_b+10^{-8}$ does not
  degenerate, and bins with tiny $W_b$ enter weighted by $W_b/N$.
- **The mechanism.** The term $-\mathrm{sign}(D_b)w_{ib}/N$ has the **same sign for every example in the
  bin**, and its size does not shrink as $D_b\to0$. That is the $|\cdot|$ kink: an L1 penalty on a noisy
  batch statistic. Summed over the batch it gives an $O(1)$ push on the direction heads' bias and on the
  trunk, not an $O(1/\sqrt N)$ one. BCE's bias gradient, $\overline{p-y}$, vanishes at calibration.
  Soft ECE's does not: it chatters around the batch's up-rate. That up-rate has sampling noise of about
  $0.5/\sqrt{N}$, and on the unshuffled validation batches it drifts with the regime.
- **Proper?** No. It measures calibration only, with no sharpness incentive. A constant $p$ at the
  batch base rate scores about 0, and so does every calibrated predictor. Adding it to BCE does not
  move the population minimiser in principle, since a calibrated $p^*$ has ECE 0 in the population. With
  finite batches and the kink, though, it adds a persistent noisy gradient.
- **Weight:** 1, calibrated to **2.907**. The calibration pass multiplies it up **because its value is
  small**. Share: 3.6-11% of the value.
- **Measured.** A CPU probe used the leader's weights and calibrated lambdas, `training=True`, on
  batches of the bundled Oct-Nov 2025 file (`D:/nt_math_scratch/probe.py`, outputs `probe_*.txt`):

  | batch (B, seed) | total $\|g\|$ | soft-ECE $\|g\|$ | cos(soft-ECE, total) | BCE $\|g\|$ | NLL $\|g\|$ | CRPS $\|g\|$ | point $\|g\|$ |
  |---|---|---|---|---|---|---|---|
  | 2048, 0 | 10.19 | **10.15** | **0.973** | 0.44 | 0.63 | 0.89 | 0.54 |
  | 2048, 1 | 17.27 | **15.25** | **0.986** | 0.58 | 2.84 | 0.85 | 0.98 |
  | 512, 2 | 20.87 | **17.35** | **0.977** | 1.03 | 4.61 | 0.73 | 1.02 |

  - At the direction-head outputs, soft ECE's gradient norm is 0.32-1.13. BCE's is 0.067-0.13.
  - **Soft ECE is 3.6-8% of the served loss and about 97% of the gradient's direction.** Its norm alone
    is at or near the clip of 20.

## 8. Volatility penalty

- **Formula** (`:693-700`): $\ell^{vol}=\lambda_{vol}\min(|\mathrm{std}_B(\mu_{h1})-\mathrm{std}_B(y_{h1})|,10)$,
  entering the total with weight 0.1 (`:778`). It covers h1 only.
- **Gradient (derived):** $\partial/\partial\mu_i=0.1\lambda_{vol}\,\mathrm{sign}(\cdot)\,(\mu_i-\bar\mu)/(B\,\mathrm{std}(\mu))$.
  - The norm over the batch is $0.1\lambda_{vol}/\sqrt B$. It is bounded while $\mathrm{std}(\mu)>0$.
  - At $\mathrm{std}(\mu)=0$, `reduce_std` differentiates $\sqrt{0}$ and gives NaN. The guard would zero
    the step. This edge case is **not observed** (0 non-finite steps anywhere).
- **Proper?** No, and it **opposes the proper scores.** It is minimised at
  $\mathrm{std}(\mu)=\mathrm{std}(y)$. The conditional mean has
  $\mathrm{std}(E[Y|X])=\sqrt{R^2}\,\mathrm{std}(Y)$, which is much smaller than $\mathrm{std}(Y)$ at
  $R^2\approx0$. So it pushes the price head to spread its predictions to the noise level. That works
  against log-cosh, NLL and CRPS, which all want a shrunk $\mu$.
- **Measured:** h1 prediction std 0.16-0.21 against a target std of about 1.1-1.17. The validation vol
  loss of 1.358 means $|\Delta\mathrm{std}|=0.75$. The gradient norm is 0.92-0.99, about the point
  loss's size, from a 2.3% value share.
- **Weight:** 1, calibrated to 1.817.

## 9. inter_reg and reg

- `model.losses` are the layer L2 penalties. Only the direction-skip L2 of 1e-4 is on by default;
  `REG_MOMENTUM_L2` and `INDICATOR_L2` are 0.
- `reg_loss` enters $\times0$. `inter_reg` enters $\times0.1\,\lambda_{inter}$ (`:690-691`, `:776-777`).
- Gradient $0.2\cdot10^{-4}\,\lambda_{inter}w$. Negligible; logged as 0.0000.

## 10. Physics terms (D-003)

All five active terms have weight 0.1 and calibration damping 0. Vacuum bandwidth is off:
`LAMBDA_VAC` = 0 routes a `tf.cond` to 0.

### T-perp (`:262-286`)

- **Formula:** $\ell^{tp}_h=(\log(R+\epsilon)-\log(\bar V+\epsilon))^2$ with
  $R=\mathrm{sg}(\overline{e^2})$ (stop-gradient) and $\bar V=\bar v_c$.
- **Gradient:** $\partial/\partial v_i=-2\log(R/\bar V)/(\bar V B)$, the same sign for every example.
  Through softplus: $|\partial/\partial a_i|\le2|\log(R/\bar V)|/B\cdot v_i/\bar V$.
  - Bounded, because $\bar V\ge10^{-4}$. It is scale-free in the level.
  - It is coherent across the batch, so its gradient on shared variance-head parameters is not
    averaged away.
- **Proper?** No; it is a batch moment condition. Its zero set $E[v]=E[e^2]$ **contains** the NLL
  optimum, so it is compatible with NLL but redundant.
- **Measured:** gradient norm 0.16-0.99 against a value of 0.0005-0.0034. Its $\|g\|/\lambda$ is 1.6-9.9.

### Casimir (`:289-328`)

- **Formula:** $\ell^{cas}=\overline{\,\mathrm{sg}(\mathrm{relu}(-s_0s_1))\,\mathrm{relu}(-\log(\tfrac{v_0+v_1}2+\epsilon))+(\ldots)_{12}}$
  with $s=\tanh(\mu/0.5)$.
- **Gradient:** only into $v$. Through softplus, $|\partial/\partial a_j|\le1/B$ (derived, using
  $\sigma(a_j)\le v_j\le v_0+v_1$).
- **Bound:** the value is at most $\log(1/10^{-4})\approx9.2$. `tests/test_physics_terms_bounded.py:87`
  checks the bound and the absence of gradient into the price heads.
- **Proper?** No. It pushes the variance to at least 1 (scaled unit variance) where adjacent horizons
  disagree in sign. The h0 conditional variance is below 1: the mean $e^2$ at h0 is 0.81-0.97 in the
  probe batches. So where it binds, it biases the variance upward.
- **Measured:** value 1e-4, gradient norm 0.004. Inert.

### Vacuum bandwidth (`:331-366`)

- Off by default. When on, it penalises $\mathrm{relu}(\mathrm{std}(\mu_0,\mu_1,\mu_2)-\Lambda)$.
- Edge case: when all three heads are equal, `reduce_std` at 0 gives a NaN gradient (**derived, not
  checked against a test**).

### Hyper-decoherence (HD) (`:369-400`)

- **Formula:** $1-\mathrm{Pearson}_B(z(\log\mathrm{sg}(\mathrm{std}\,x)),\,z(\log\bar v))\in[0,2]$.
- **Gradient (derived):**
  $\partial\rho/\partial u_i=(z^{v}_i-\rho z^u_i)/(B(\mathrm{std}(u)+10^{-3}))$ with $u=\log\bar v$.
  It is bounded by about $2\max|z|/(B\cdot10^{-3})$. It **grows like $1/\mathrm{std}(\log\bar v)$** as
  the variance head becomes constant across the batch: scale-free in value, not in gradient.
- **Proper?** No. It is an ordering prior: variance should be ordered like the window's realised
  volatility. It is compatible with NLL only if the conditional variance is log-linearly monotone in
  window volatility.
- **Measured:** value 0.009-0.015, so $\rho\approx0.85$-$0.91$. Gradient norm about 0.19, with cosine
  −0.15 to −0.44 to the total: it opposes the net update direction.
  `tests/test_physics_terms_bounded.py:71` covers it.

### IFE (`:403-444`)

- **Formula:** $\sum_{(a,b)\in\{01,12\}}\mathrm{relu}(|\mathrm{corr}_B(\mu_a,\mu_b)|-\rho_{max})$ with
  $\rho_{max}=0.95$.
- **Gradient:** zero while the correlations are below 0.95, which holds in every batch here (value 0).
  When active, the Pearson gradient is about $1/(B\,\mathrm{std}(\mu))$. It explodes as a price head's
  batch std goes to 0; the $\epsilon=10^{-8}$ is added after the square root. The h0 head's batch std is
  0.029-0.034 here.
- **Proper?** No. The h0-h2 targets overlap heavily, so their conditional means are legitimately highly
  correlated; the penalty is a prior against that.
- **Measured:** inert (0 in every logged epoch and every probe).

### Vacuum overflow (`:447-487`)

- **Formula:** $(\overline{ov}-\bar r)^2/(\bar r^2+\epsilon)$ with $\bar r=\mathrm{sg}(\overline{|e|})$
  and $ov=\mathrm{relu}(\overline{(h_\perp+n)^2}-E_{max})$. Here $h_\perp=\tanh(\cdot)$ and $n$ is the
  training-only noise (`models/layers/vacuum_saturation_noise.py`).
- **Gradient:** only into the T-perp projection. At inference $ov\equiv0$, because $\tanh^2<1=E_{max}$,
  and `test_step` passes `None` (`custom_model.py:590-591`).
- **Measured:** gradient norm 0.006, about 1.5% of the training value. It is effectively a constant.
  `tests/test_physics_terms_bounded.py:127` covers it.

## 11. Off-by-default terms

### Direction alignment

- **Formula** (`:675-688`): BCE of $p$ against the Gaussian readout $P(\mathrm{up}\mid\mu,v)$, masked.
- Off: `LAMBDA_DIR_ALIGN_OUTER` = 0 skips it in Python.
- A soft-target BCE is minimised at $p=p_{gauss}$ (consistency), not at the truth.

### pnl_utility (NT-087; `:813-919`)

- **Formula:** $\ell=-\sum_h\overline{a\tilde r-\tilde c|a|-\tfrac\gamma2a^2\tilde r^2}$, weighted by
  $\lambda_{pnl}$.
  - $a=2p-1$.
  - $\tilde r=\mathrm{clip}(r/\sigma,\pm5)$ minus its stop-gradient batch mean, so $|\tilde r|\le10$.
  - $\tilde c=c/\sigma$ with $\sigma\ge10^{-6}$.
- **Gradient:** $\partial/\partial p_i=-2(\tilde r_i-\tilde c\,\mathrm{sign}(a_i)-\gamma a_i\tilde r_i^2)/B$,
  so $|\cdot|\le2(10+\tilde c+100\gamma)/B$.
  - $\tilde c$ is **not** bounded in practice: $c/\sigma$ reaches $2.6\times10^{3}$ at a 26 bps cost and
    $\sigma=10^{-6}$ (derived).
  - The cost is 0 by default (D-044). The leader config carries `PNL_COST_BPS: 26`, but with
    $\lambda_{pnl}=0$.
- **Population optimum:** $a^*=\mathrm{clip}\big((E[\tilde r|x]-\tilde c\,\mathrm{sgn})/(\gamma E[\tilde r^2|x]),\pm1\big)$.
  That is an aim, not a probability. Combined with BCE on the same head, the minimiser is a compromise:
  neither calibrated $P(\mathrm{up})$ nor $a^*$.
- **Tests:** `tests/test_pnl_utility.py:151` checks that the gradients are finite.

---

## Global stability argument

### What is provable under the implemented clamps

These statements assume finite inputs; the finite-gradient guard covers non-finite ones.

1. **Head-level gradient bounds.** The derivative of every active term with respect to the head outputs
   and pre-activations is bounded:
   - point, trend and CRPS by data-independent constants of order $1/B$;
   - BCE by $1/\sum m$ at the logit;
   - Casimir by $1/B$ at the variance pre-activation;
   - soft ECE by about $61/N$ (derived);
   - NLL by $\tfrac1{2B}(1+10^4e^2)$ at the variance pre-activation and $10^4|e|/B$ at the mean. This
     bound is finite only because of the variance floor and the $\pm100$ price clip, and it is huge.
   - T-perp, HD and IFE are bounded only through the floors and $\epsilon$s. HD and IFE scale like
     $1/\text{batch std}$ of a head.
   - The chain rule through the network multiplies these by the network Jacobian. **Nothing in the code
     bounds the Jacobian.** So there is **no provable bound on $\|\nabla_\theta L\|$.**
2. **Clipping.** `train_step` computes the gradient. It zeroes the whole step if the total or any
   gradient is non-finite and counts it in `nonfinite_grad_steps` (`custom_model.py:498-512`). It then
   applies `clip_by_global_norm(·, 20)` **separately** to the network group and the indicator-logit
   group (`:529-538`). After clipping, each group's gradient norm is at most 20.
   - The logged `grad_global_norm` is the combined **pre-clip** norm (`:503`).
   - The guard still calls `apply_gradients` with zeros. Adam's momentum therefore keeps moving the
     weights on a skipped step. That motion is bounded by the next point.
3. **Adam's per-coordinate step is bounded regardless of gradient size.** Adam here uses β1 0.9,
   β2 0.999, ε 1e-7. Cauchy-Schwarz on the exponential sums (derived, `D:/nt_math_scratch`) gives
   $$\Big|\frac{\hat m_t}{\sqrt{\hat v_t}}\Big|\le\frac{1-\beta_1}{1-\beta_1^t}\sqrt{\frac{1-r^t}{1-r}}\sqrt{\frac{1-\beta_2^t}{1-\beta_2}},\quad r=\beta_1^2/\beta_2 .$$
   - This equals 1 at $t=1$, 2.24 at $t=100$, 5.78 at $t=1000$, and tends to **7.27** (the supremum).
   - So $|\Delta\theta_j|\le7.27\,\mathrm{lr}$ per step: $7.3\times10^{-3}$ at lr 1e-3, and
     $3.6\times10^{-2}$ in logit units for the indicator group (lr × 5).
   - The bound needs no clipping. It holds for any finite gradient sequence. It is loose: the typical
     step is about lr.
   - Consequence: **clipping cannot be what prevents a blow-up of the weights under Adam.** Its real
     effects are two. It stops one spike from inflating $v$ (and so damping every later step) for about
     $1/(1-\beta_2)=1000$ steps. And it rescales the **relative** contribution of the terms inside each
     group.
   - A related consequence: `INDICATOR_GRAD_MULT`, a straight-through ×5
     (`models/layers/learnable_indicators.py:100-102`), is almost a no-op under Adam, because Adam is
     invariant to a constant gradient scale. It only interacts with the indicator group's clip and with
     ε. `optim.py:3-6` already says the equivalent for learning rates.
4. **Finite forward pass.** Heads are sanitised, terms are sanitised, and the total is sanitised to 0
   when non-finite. The weights stay finite as long as every applied update is finite. That is
   guaranteed by point 2, because non-finite steps are zeroed. The learned periods are projected to
   [2, 60] after every step (`custom_model.py:546-557`).

### What is not guaranteed

- Convergence (the problem is non-convex and the terms conflict).
- Absence of NaN inside the forward pass. The guard hides it by zeroing the step; nothing prevents it.
- Absence of a NaN gradient at `reduce_std` of a constant (vol, HD, IFE, vacuum).
- Any bound on $\|\nabla_\theta L\|$ before the clip.
- That the served model is a minimiser of any proper score, because the objective mixes improper terms
  (sections 3, 7, 8 and 10).

### Empirical evidence

**Tests.**

- `tests/test_custom_loss.py:63`: all 35 components are finite, with finite head gradients, at three
  scale regimes.
- `tests/test_physics_terms_bounded.py`: bounds and absence of gradient leaks for HD, Casimir, T-perp,
  IFE and vacuum overflow.
- `tests/test_losses_reference.py:148,180,186`: properness of CRPS and BCE; impropriety of dice.
- `tests/test_pnl_utility.py:151`.
- `tests/test_train_smoke.py:50`: three steps with finite weights and 0 non-finite steps.
- There is **no** `stability` pytest marker: NT-036 is `todo`
  (`pyproject.toml:51-57`, `docs/BACKLOG.md:71`).
- There is **no per-term gradient probe** in the code: NT-037 is `todo`, and no GRAD_PROBE flag exists.
  The probe here is a scratch script.

**The 960-trial level-1 screen.** Source: `runs/screens/l1_*/results.shard-*.jsonl`, analysed with
`D:/nt_math_scratch/screens.py`. Settings: batch 64, 8 epochs, `calibrate: false` (config lambdas),
first epoch excluded from `clipped_share`.

| block | n | passed | finite | non-finite steps | clipped_share p50 | ‖g‖max p50 / max | ‖g‖mean p50 | fail reasons |
|---|---|---|---|---|---|---|---|---|
| A optimiser | 264 | 202 (76.5%) | 264 | 0 | 0.26 | 38 / 235 | 17.9 | clipped_share 62 |
| B loss weights | 328 | 61 (18.6%) | 328 | 0 | 1.00 | 112 / 3,820 | 52 | clipped_share 267, max_term_share 22 |
| C physics | 288 | 212 (73.6%) | 288 | 0 | 0.41 | 47 / 159 | 21.1 | clipped_share 76 |
| D loss choice | 64 | 50 (78.1%) | 64 | 0 | 0.35 | 41 / 191 | 19.4 | clipped_share 14 |
| E maths | 16 | 12 (75%) | 16 | 0 | 0.33 | 44 / 171 | 20.0 | clipped_share 4 |
| **all** | **960** | **537 (55.9%)** | **960** | **0** | 0.46 | 52 / 3,820 | 22.0 | clipped_share 423, term share 22 |

- **No trial produced a non-finite value or a non-finite gradient step. Every failure is a
  threshold rule.**
  - `clipped_share` above 0.5 means the pre-clip norm was at or above 20 on more than half the logged
    steps.
  - `max_term_share` above 0.9 means one term held more than 90% of the loss value.
- **The mean gradient norm at the default weights (about 18-22) sits at the clip (20).** Clipping is a
  routine operating regime, not an alarm. Block A shows it: GRAD_CLIP_NORM 5 passes 0 of 8, 20 passes
  5 of 8, 100 passes 8 of 8. Only the rule threshold moved.
- **Knob drivers in block B** (Spearman ρ of the knob with `clipped_share` / `passed`):

  | Knob | ρ with clipped_share | ρ with passed |
  |---|---|---|
  | LAMBDA_SOFT_ECE | **+0.59** | **−0.45** |
  | LAMBDA_DIR_OUTER | +0.45 | −0.34 |
  | LAMBDA_NLL_OUTER | +0.38 | −0.22 |
  | LAMBDA_VAR | +0.33 (ρ with ‖g‖mean +0.58) | −0.23 |
  | LAMBDA_CRPS | +0.22 | −0.26 |
  | LAMBDA_COHERENCE, LAMBDA_TREND_OUTER, LAMBDA_POINT | about 0 | about 0 |

  Soft ECE and NLL are the two terms the probe identifies as gradient-heavy. Note that
  LAMBDA_DIR_OUTER multiplies BCE only, but its range [0.1, 10] reaches 13x the default.
- **Physics (block C):** no physics weight has $|\rho|>0.17$ with the gradient norm. The one-at-a-time
  ablations pass 5-8 of 8 each, with the gradient-norm median at 20.9-22.6. The physics terms do not
  drive the stability numbers.
- **Loss choice (block D):** `LAMBDA_PNL` raises `clipped_share`. The median goes 0.32 → 0.55 for bce
  and 0.27 → 0.44 for focal_dice as $\lambda_{pnl}$ goes 0 → 1. Passes: bce 7/6/5/4, focal_dice
  8/8/7/5.
- **The screen's `loss_term_shares` differ from the gradient picture.** NLL has the largest value share
  in 743 of 960 trials (median 0.41). That share is of the value, not of the gradient.

**The 360-day runs.** Source: `runs/scenarios/long_360d_stab/*/training_log.csv`, six runs (folds −3 and
−2, seeds 0-2), 10-25 epochs each.

- `nonfinite_grad_steps` = 0 in every epoch of every run.
- The per-epoch mean `grad_global_norm` falls monotonically-ish as the LR halves: 15.2 → 8.0 for
  f−3 s0, and 7.7-9.4 → 4.8-5.6 for the other seeds.
- The leader (f−2 s0) runs **hotter**: 18.9, 21.3, 19.0, 17.5, 14.7 … 11.8. Its epochs 1-3 have a mean
  norm at or above the clip. Its val loss is non-monotone (6.22, 6.08, 6.13, **5.93**, 6.03, …), and the
  best epoch is 4 of 10.
- Batch 2048 halves the norm relative to batch 64: about 18-22 in the screens, about 5-21 here.

---

## Solvability

**Direction.** Model: binormal equal-variance scores. AUC is $\Phi(d/\sqrt2)$ and the Bayes posterior
is $\sigma(d\,s)$. The Bayes-optimal BCE is then $\ln2-I(Y;S)$ (`D:/nt_math_scratch/solv.py`).

| AUC | Bayes BCE (nats) | reduction below ln 2 | Brier reduction below 0.25 | sd of the Bayes P(up) |
|---|---|---|---|---|
| 0.51 | 0.692990 | 1.6e-4 (0.02%) | 7.9e-5 | 0.009 |
| 0.52 | 0.692519 | **6.3e-4 (0.09%)** | 3.1e-4 | 0.018 |
| 0.53 | 0.691733 | 1.4e-3 (0.20%) | 7.1e-4 | 0.027 |
| 0.55 | 0.689215 | 3.9e-3 (0.57%) | 2.0e-3 | 0.044 |
| 0.60 | 0.677353 | 1.6e-2 (2.3%) | 7.8e-3 | 0.088 |

- The leader's dev AUC is 0.520 / 0.518 / 0.525 (`eval_report_dev.md`). Its logreg_lags baseline is
  0.524 / 0.527 / 0.531.
- So the **whole achievable direction gain is about 6e-4 nats per horizon, about 0.09% of the BCE**.
  The weighted gain over three horizons is $0.749\times3\times6.3\times10^{-4}\approx1.4\times10^{-3}$,
  about 0.02% of a total loss near 6.
- The leader's served validation BCE is 0.6937 / 0.6932 / 0.6937. That is **above** $\ln2=0.6931$ by
  $5.5\times10^{-4}$ / $5\times10^{-5}$ / $5.5\times10^{-4}$. The head loses more to miscalibration
  than its AUC can earn. The probe batches show direction-head std of 0.017-0.046, while the Bayes sd at
  AUC 0.52 is 0.018.
- Consequences:
  - Early stopping and epoch selection use the composite `val_loss`. They are **blind to direction
    skill**: epoch-to-epoch swings in val_loss are about 0.1-0.3, which is 100-200x the entire
    achievable BCE gain.
  - Any term whose gradient noise is O(1) swamps the direction signal. Soft ECE does exactly that
    (section 7).
- **Detectability** (derived). A per-example BCE difference of $6\times10^{-4}$ needs about
  $(2\cdot0.04/6\times10^{-4})^2\approx1.8\times10^4$ effective samples to reach $2\sigma$. The labels
  overlap, so $n_{eff}\approx N/h$. That means about $2\times10^5$ labelled bars per horizon, far more
  than any single minibatch. The direction gradient per step is noise-dominated by construction.

**Variance.** This is the one real edge. The leader's dev CRPSS against a constant variance is
0.046 / 0.044 / 0.044. The fold −3 seeds reach 0.073-0.081 (`result.json`, `scores.h*/variance/crpss`).
The NLL, CRPS and T-perp minimisers agree on $v=\mathrm{Var}[Y|x]$, so the variance task is well posed
and consistent across its proper terms.

**Conflicts in the combined objective** (derived, not checked against a test):

1. **Price head $\mu$.** Four supervisors act on $\mu$:
   - log-cosh (location functional);
   - NLL (mean, inverse-variance weighted, outlier-sensitive);
   - CRPS (median-like within the Gaussian family);
   - extended trend (momentum prior), magnitude-ordering coherence, and vol, which **opposes shrinkage**
     by demanding std(μ) = std(y).

   The first three agree on a symmetric conditional law. The last three move the minimiser away from
   $E[Y|x]$. The measured h1 prediction std of 0.16-0.21 against a target std of about 1.1 means the vol
   term is always active (|Δstd| = 0.75).
2. **Direction head $p$.** BCE is proper, with minimiser $P(\mathrm{up}\mid\text{move},x)$. Soft ECE
   is improper and does not change the population minimiser, but its finite-batch gradient is O(1)
   noise with a kink. pnl_utility, when on, has a different optimum ($a^*$).
3. **Variance head $v$.** NLL, CRPS and T-perp are consistent. Casimir biases $v$ up to at least 1 where
   horizons disagree. HD adds an ordering prior. Both are tiny in value and gradient.
4. **Net result.** The combined minimiser is not the minimiser of any single proper score, even in the
   population. The biases come from trend, coherence, vol and Casimir. At the served weights their
   combined value share is about 7.4% (validation); vol's and coherence's gradient norms are each about
   the point loss's.

---

## Problems and recommendations

1. **Remove soft ECE from the training objective, or cut it hard.**
   - It is 3.6-8% of the served loss value but carries **about 97% of the gradient's direction**:
     cosine 0.973 / 0.986 / 0.977 with the total, and norm 10.2 / 15.2 / 17.4 against totals of
     10.2 / 17.3 / 20.9 (the probe, section 7).
   - It is the strongest screen driver of clipping and failure (ρ +0.59 / −0.45).
   - It is improper, and its $|\cdot|$ kink gives an O(1) gradient that does not vanish at calibration.
     Temperature calibration on the cal block already exists (`calibration_report.temperature`
     1.01-1.14).
   - Proposed test: a pre-registered A/B of `LAMBDA_SOFT_ECE` 0 against the default. Primary metrics:
     direction AUC and BCE, CRPSS, `clipped_share`. The direction signal is 6e-4 nats, so the prediction
     is that removing a 15-norm noise source helps it. That prediction is an **untested hypothesis**.
2. **Calibrate weights by gradient norm, not by loss value.** `lambda_calibration.py:207-210`
   multiplies a term's weight by $\bar L/\mathrm{med}$. A small-valued term with a steep gradient (soft
   ECE: 1 → 2.907) is amplified. The alternatives are to equalise per-term $\|\nabla_\theta L_i\|$ on
   the shared trunk (GradNorm-style; it would need NT-037's per-term probe) or to fix the weights. Soft
   ECE is the evidence.
3. **Drop the vol term** (0.1·λ_vol·|std(μ)−std(y)|). It opposes the proper scores' shrinkage (section
   8), it is always active in the leader (|Δstd| = 0.75), and its gradient norm (about 0.95) equals the
   point loss's. Serving uses delta shrinkage (D-007) anyway. Check its removal with the A/B of item 1.
4. **Make the coherence penalty honest.** Two of its three parts have zero gradient: `tf.sign`, and a
   label-only constant that inflates `val_loss` by a data-dependent offset. The third (magnitude
   ordering) has gradient norm about 0.9 and is not logged. Log it, remove the two dead parts, and treat
   the ordering as an A/B-tested prior. `screen.py:731`'s claim that it is small is wrong at
   `LAMBDA_COHERENCE` = 1.
5. **Select the epoch on the proper scores, not the composite.** The composite val_loss has swings
   100-200x the achievable BCE gain. Use `val_dir_loss` (BCE) for direction and CRPS/NLL for variance,
   or a pre-registered sum of proper scores. Requires a decision, because D-011 fixes "best-validation
   epoch" on `val_loss`.
6. **Guard the `reduce_std` edges** in vol, HD, IFE and vacuum. Add an $\epsilon$ inside the square root
   (`sqrt(var+eps)`). Today a batch-constant head gives $\sqrt{0}'=\infty$ and a NaN gradient, which
   the guard hides by zeroing the step. This has not been observed (0 non-finite steps in 960 trials and
   6 long runs): it is a latent defect, P3.
7. **NLL's residual tail.** The $\mu$-gradient $e/v$ and the variance gradient $e^2/v$ are bounded only
   through the floor and the ±100 clip. NLL was the most batch-sensitive term in the probe
   (0.63-4.61). Options: rely on CRPS, whose gradients are bounded by 1 and 0.282, and lower
   $\lambda_{var}$; or use a Student-t / winsorised NLL. The ρ of LAMBDA_VAR with ‖g‖mean is +0.58.
8. **State the stability claim as follows.**
   - Provable: finite updates (the guard) and per-coordinate steps of at most $7.27\cdot$lr (Adam).
   - Provable: bounded head-level gradients for point, trend, CRPS, BCE and Casimir.
   - Only floor-bounded: NLL and HD.
   - Empirical: 0 non-finite steps in 960 screen trials and in every epoch of the six 360-day runs.
   - Rule failures (44% of screen trials) are `clipped_share` and term-share thresholds at a clip
     placed at the median gradient norm, not divergences.
   - Not proven: any bound on $\|\nabla_\theta L\|$, or convergence.
9. **Physics terms.** At weight 0.1 they are inert in value: 0.3-0.5% of the validation loss in total,
   with IFE, Casimir and vacuum overflow about 0. T-perp and HD have gradient norms of 0.16-0.99 and
   about 0.19, with HD opposing the net direction. No screen evidence ties them to stability
   (|ρ| ≤ 0.17). This agrees with D-003's v1 record. NT-006 is the pre-registered re-run; nothing here
   changes it.
10. **Clipping under Adam.** Clipping at 20 with a mean norm of about 20 mainly rescales the soft-ECE
    direction. With item 1 applied, re-measure the norm distribution before re-choosing `GRAD_CLIP_NORM`
    or the `max_clipped_share` rule. The clip is not what keeps the weights bounded (see "Global
    stability argument", point 3).

### Reproduction

```bash
PY=C:/Users/Step/miniforge3/envs/nt/python
$PY D:/nt_math_scratch/screens.py            # 960-trial statistics
$PY D:/nt_math_scratch/leader.py             # per-term shares, 360-day grad norms
CUDA_VISIBLE_DEVICES=-1 $PY D:/nt_math_scratch/probe.py 2048 0   # per-term gradient norms, ~3.5 min CPU
$PY D:/nt_math_scratch/solv.py               # AUC -> achievable BCE
```

The probe uses the bundled `binance_btcusdt_1min_ccxt.csv` (Oct-Nov 2025) with the leader's
`pred_scale` and `pred_mean`, in `training=True` mode (dropout and vacuum noise on). Its three batches
are one sample each. The soft-ECE dominance held in all three, but it is not a distribution over
training.
