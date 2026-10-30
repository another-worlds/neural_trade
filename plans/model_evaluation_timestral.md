# Evaluation: Mistral Timestral and AdityaLab Samay for neural_trade

| | |
|---|---|
| Date | 2026-10-30 |
| Scope | Fit of two time-series foundation-model options for the neural_trade BTC/USDT 1-minute system |
| Method | Code review, literature review, hands-on zero-shot benchmark on two data sets, head-to-head with neural_trade, internal adversarial review |
| Prepared by | Claude Code (AI). All numbers come from the scripts and result files in `research/tsfm_eval/`. |

---

## Terms used in this report

| Term | Meaning |
|---|---|
| bps | Basis points. 1 bps = 0.01%. All returns are log returns in bps. |
| Origin | A time t at which a model makes a forecast. It uses only bars up to and including t. We forecast every 3 minutes. |
| h | Forecast horizon in minutes: 1, 5 or 15 (the neural_trade horizons). |
| ctx | Context: the number of past 1-minute bars a model reads. ctx 2048 is about 34 hours. |
| Repo set / clean set | Repo set: the last 20% of the repo CSV (6 days, 2025-11). Clean set: 14 days after Timestral's release (2026-10-16 to 10-29), which no model can have seen in training. |
| Random walk (RW) | The forecast "the price does not change". |
| EWMA-t | Our reference baseline. Its variance is an exponentially weighted moving average (EWMA) of past squared returns. Its tails follow a Student-t distribution. It has 2 parameters. |
| FHS | Filtered historical simulation: the same EWMA variance, with tail shapes taken from past data instead of a formula. |
| HAR | A linear regression that forecasts volatility from its averages over short, medium and long past windows. |
| Quantile, q0.01 … q0.99 | q0.4 is the value the return falls below with probability 40%. q0.01 and q0.99 mark the 1% tails. |
| Pinball loss | The error of a quantile forecast. Lower is better. Averaged over many quantile levels, it scores the whole forecast distribution. |
| QLIKE | A loss for a variance forecast. Lower is better. It penalises a forecast that is too low more than one that is too high. |
| MAE skill vs RW | 1 − (mean absolute error of the model) / (mean absolute error of RW). Above 0 means better than RW. |
| AUC | The probability that the model gives a higher P(up) to a real up-move than to a real down-move. 0.5 is chance. |
| MCC | The correlation between the forecast sign and the real sign. 0 is chance. |
| Brier skill | 1 − (Brier score of P(up)) / (Brier score of a constant forecast equal to the share of up-moves). 0 or less means no skill. |
| Spearman | Rank correlation. Here: between the forecast interval width and the size of the real move. |
| Tail test | Checks whether a tail quantile is hit at its target rate (for example 1% for q0.01). "Pass" means p ≥ 0.05. "Too wide" means it is hit less often than the target, so the forecast is too cautious. |
| 95% CI, significant | 95% confidence interval from a block bootstrap. "Significant" means the CI excludes 0 (0.5 for AUC). |
| T-A … T-K | The result tables in Appendix A. |

---

## 1. Verdict

**Mistral Timestral: do not adopt it now.** Its only measured value is a small gain in variance forecasts. Free EWMA baselines get most of that value, and the licence blocks any use until legal clears it. First fix neural_trade and wire a cheap risk input. Later, if legal clears the licence, test Timestral-28M in the risk path only. Do not use it for price or direction.
- **No price or direction skill.** It does not beat a random walk on price at 1, 5 or 15 minutes. Its direction calls have no stable skill. Every other model we tested has the same result, and that includes our own model.
- **A small variance gain.** We compared it with three cheap EWMA baselines, all fitted only on data before the test. The strongest is FHS (an EWMA with tail shapes taken from past data). Against FHS, Timestral's QLIKE is 0.6–3.0% lower. The gain is significant in 7 of 12 cells (two model sizes × two data sets × three horizons). Its pinball loss equals that of FHS.
- **No clear tail advantage.** Timestral-28M fails 1 of 6 tail tests on each data set. FHS fails 2 of 6 on the repo set and 1 of 6 on the clean set. The plain EWMA-t baseline fails 5 of 6 on the clean set, because its tails were already too thin before the test.
- **Volatility forecaster.** As a direct forecaster of the mean 1-minute high-low range over the next h minutes, Timestral-28M has 5–17% lower QLIKE than an EWMA of that range. Against a bias-corrected HAR regression it is equal at 5–15 minutes on the repo set and 4–14% better on the clean set. At 1 minute it beats both.
- **The free alternative did not match it on returns.** We gave Chronos-2 (Apache-2.0 weights) the same return input as Timestral. Its pinball loss was equal or up to 1.2% higher, its QLIKE was 2–9% higher than Timestral's best setting, and it did not beat FHS on QLIKE. On the volatility (range) series it matched Timestral (Sections 7.3 and 7.5).
- **Before any use:**
  - **Licence (blocking).** The weights give no rights at all if the global consolidated monthly revenue of the user's company, or of the user's employer, exceeded $20M in the preceding month. This also covers derivatives, combined works and benchmarks. Legal must confirm that it does not apply. If it applies, we must get a commercial licence from Mistral or not use Timestral.
  - **Training data (known risk, not blocking).** Mistral does not name the real post-training data. Our clean post-release test controls for this risk (Section 10.2).
  - **Adaptation (known limit, not blocking).** The released code is inference-only. To fine-tune it, we would have to write our own training loop or use Mistral Forge.

**AdityaLab Samay: do not adopt it.**
- It adds no forecasting ability. For Chronos-2 and Chronos-Bolt (ctx 512, 483 repo origins), its outputs equal the official package to 0.0008 bp. We did not compare its TimesFM-2.5 path with the upstream package.
- It is unsafe as a dependency:
  - a normal install gave silently wrong Chronos-2 forecasts, and Chronos-Bolt failed to load with a misleading error;
  - it worked for us only in its exact lock file (Python 3.11, torch 2.5.1, transformers 4.47.1), which conflicts with our stack;
  - it has no tests, and `pip install samay` installs an unrelated package.
- We tested three of the models it wraps: Chronos-2, Chronos-Bolt and TimesFM-2.5. With price-level input, their pinball loss was 1.5–11% worse than EWMA-t. With return input, Chronos-2 improved a lot (pinball loss 3–8% lower at 1 minute), but its QLIKE stayed above Timestral's and FHS's (Section 7.3). If one of these models is wanted, use its official package, not Samay.

**neural_trade itself is the larger problem.**
- Retrained as-is on the same data and then frozen for the test window:
  - its direction head says DOWN at every origin in 13 of 18 seed-horizon runs;
  - its pinball loss is 1.6–10.7% worse than the EWMA-t baseline, and its QLIKE is 10–21% worse;
  - its volatility ranking is weak (Spearman 0.05–0.16 against 0.31–0.40 for the EWMA baselines).
- Retraining it every day did not help. It was worse than the frozen model on the repo set and mixed on the clean set. So stale weights do not explain its poor scores. Its inputs are raw price levels, so they often leave the training range even with daily retraining (D8).
- Defects D1–D3, D7 and D8 (Section 3.3) are plausible causes. Short training (40 optimizer steps) may also contribute. We did not test a fixed model. Fix D1–D3, D7, D8 and the evaluation set-up (D4–D6) before any model swap. Then measure again.

### Where each candidate fits

| Role in neural_trade | Timestral | Samay | Evidence |
|---|---|---|---|
| Replace the whole model | **No** | No | Price and direction outputs carry no information (T-A). |
| Price head (pinball τ = 0.4) | **No** | No | MAE is worse than random walk on both sets. A point forecast has no trading value here. |
| Direction head (focal BCE) | **No** | No | AUC 0.45–0.53 across the candidate models. Brier skill is at most +0.0024, which is no skill. |
| Variance input to a risk path (not yet wired) | **Optional, after the licence check** | No | QLIKE 0.6–3.0% lower than FHS, a free baseline. Tails about as good as FHS. 5–17% lower QLIKE than an EWMA of the range. |
| Extra input feature for the Keras model | No | No | ΔAUC is between −0.012 and +0.012. No value is significantly positive; two are significantly negative. |
| Standing benchmark baseline | **Yes, after the licence check** | No | Cheap. Gives an honest bar for any new model. The licence covers benchmarks too. |

---

## 2. Task and scope

**Request.** Evaluate the recent Mistral time-series model and "samaya sterling" for neural_trade. Run a QA step first to confirm the task.

**QA outcome (confirmed with the requester on 2026-10-30):**

| Name in request | What it is | In scope? |
|---|---|---|
| Mistral time-series model | **Timestral**, released 2026-10-15. Four zero-shot forecasters: 5M, 28M, 123M and 209M parameters. | Yes |
| "Samaya Sterling" | Samaya AI's **Sterling**, published 2026-10-28. It is a 30B-parameter (3B active) LLM research agent for finance questions. It does not forecast prices. | No |
| AdityaLab **Samay** | An open-source library that wraps other time-series foundation models (Chronos, TimesFM, Moirai and others). | **Yes. The requester chose this one.** |

The requester also asked us to decide *where* Timestral fits, from the loss math and the architecture. They did not prescribe a role.

**Method.**
1. Read the neural_trade code line by line (`model.py`, `math_helpers.py`, `helper_functions.py`, `inference.ipynb`, `training_log.csv`).
2. Read the Timestral source, model cards, licence and technical report (arXiv:2610.37585).
3. Read the Samay source, install it, and run it.
4. Survey published evidence on foundation models in finance (2021–2026).
5. Run a hands-on benchmark on BTC/USDT 1-minute data:
   - the repo CSV;
   - a clean post-release set (targets after 2026-10-15);
   - neural_trade as-is, retrained on the same data, as a head-to-head;
   - CPU latency.
6. Run an independent critic and an adversarial review over the findings and this report.

---

## 3. neural_trade today: what a forecaster must plug into

All facts below were checked in the code. `M:` = `model.py`, `MH:` = `math_helpers.py`, `HF:` = `helper_functions.py`.

### 3.1 Inputs, targets, outputs

| Item | Current behaviour |
|---|---|
| Input | Close price only, 60 bars (`M:1219`, `M:299`). No volume and no OHLC. SRS §3.5.1 specifies 60-minute OHLCV input, and §3.2.2 lists volume statistics. |
| Targets | USD deltas Δh = c[t+h] − c[t] for h = 1, 5 and 15 min. One `StandardScaler` is shared by all three horizons (`M:303-304`, `M:379-383`). |
| Training window | The 2,880 most recent sequences (`M:48`). That is about 2 days of 1-min data. |
| Split | `TimeSeriesSplit(5)`, last fold only: 2,400 train and 480 validation samples. **The validation fold is also the test fold.** It selects the final weights (best validation epoch, through `restore_best_weights` and the checkpoint) and drives the learning-rate schedule (`M:361-365`, `M:728-731`, `M:819-828`). Early stopping never fires, because patience = epochs = 8 (`M:45-47`). |
| Outputs | 3 towers × (price, direction probability, variance) = 9 outputs (`M:1289-1319`). |
| Risk path | `helper_functions.py` has confidence = 1/(1+v) (`HF:5-7`), a 2–6% stop (`HF:61-75`) and size multipliers 1.2/1.0/0.6 at confidence 0.7/0.5 (`HF:77-86`). **No code imports it.** The notebook backtest copies only the confidence formula. It has no stop and no position sizing; its exits are spike, reversal and time only. |

### 3.2 Loss math (`M:1790-1847`)

| Term | Formula (scaled units unless stated) | What the head learns |
|---|---|---|
| Point | Pinball loss, τ = 0.4 (`MH:141-162`). The code calls it "Huber"; it is not. Weights: `lambda_point` for the 5-min head; `lambda_short` and `lambda_long` (fixed at 1) for the 1- and 15-min heads (`M:1633-1635`, `M:74-75`). | 5-min head: a mix of the **40th percentile** (pinball × 1000) and the Student-t location (NLL × 1000). 1- and 15-min heads: the pinball weight stays at 1, so the NLL (× 1000) sets them; they estimate the Student-t location, not q0.4. |
| Direction | Focal BCE with α = 0.7 on DOWN and 0.3 on UP, γ = 1 (`MH:96-138`; values from `M:113-114`). | With ~50/50 labels, the best constant output is P(up) ≈ 0.38. **The head almost always outputs DOWN.** |
| Variance | Student-t NLL, ν = 3, centred on the price head (`M:1700-1733`). | v is the squared *scale*, in scaled units. The predictive variance is 3·v·s² USD². |
| Trend | Squared error, in raw USD², between *true* deltas and past trends (`M:1644-1676`). | **No prediction enters this term, so its gradient is zero.** |
| Align, vol, L2 | Small auxiliary terms. | Below 1e-3 of the total after calibration. |

**Loss-weight calibration (`M:734-813`).** The trend term is in raw USD², so its median (about 1.2–1.7 × 10⁵) is far above the other terms (about 0.5–5). It therefore dominates the reference loss, and the loss weights `lambda_point`, `lambda_dir` and `lambda_var` all hit the clamp at 1000 on every run we made (3 seeds × 2 data sets). That matches the logged notebook run. `lambda_point` scales only the 5-min point loss.

### 3.3 Defects that limit any comparison

| # | Defect | Evidence | Effect |
|---|---|---|---|
| D1 | The same multiplier (× 20) is used twice. In `LearnableIndicators.call` it multiplies the forward logit by 20: `logit + stop_gradient(logit)*(m−1)`. In `train_step` it also multiplies the indicator gradients by 20. The forward use is the defect. | Forward: `M:1073`, `M:1085-1087`, `M:1110-1112`, `M:1135`, `M:1147`. Gradient: `M:1903-1905`. α = sigmoid(logit). Our check: for period 5, α goes from 0.33 to 9.5e-7; for period ≥ 9, below 1e-12. | The EMA-based indicators (MA, MACD, RSI, Bollinger) stay at the first window value, so they do not learn from data. The momentum indicators fail for a different reason (D7). |
| D2 | The trend term has zero gradient but sets the loss-weight calibration. | `M:1644-1676`, `M:734-813`. | The loss weights clamp to 1000. The 5-min price head is pulled toward q0.4, but the 1- and 15-min heads are not (Section 3.2). |
| D3 | Focal α = 0.7 favours DOWN. | `MH:130-134`, `M:113`. | The direction head almost always outputs DOWN. Head-to-head: P(up) < 0.5 at every origin in 13 of 18 seed-horizon runs; the worst run says UP at 14% of origins (repo, seed 0, h = 5). |
| D4 | Validation = test (480 samples, 8 h). | `M:728-731`. | All logged metrics carry selection bias. |
| D5 | The backtest has no fees and uses a centred rolling mean (look-ahead) for its spike entries and exits. | `inference.ipynb`, the cell headed "TRADE DATACLASS AND COMPLETE MULTI-HEAD STRATEGY PIPELINE" (execution count 5): centred mean at line 196, profit without fees at lines 310-315. | The logged backtest result is not usable. |
| D6 | The loader needs a `timestamp` column; the repo CSV has `datetime`. | `M:201`, `dw_ccxt.py:114-116`. | The current code cannot read the repo CSV without a rename. |
| D7 | The learnable momentum period is written back with `asinh((p−1)/2)` after each step, but the forward pass reads `softplus(raw)+1`. These are not inverses. | `M:1160-1162`, `M:1928-1932`. | All three momentum periods go to about 1.95 min within the first epoch, whatever the data (configured: 1, 2, 3). |
| D8 | The inputs are raw close-price levels, scaled per lag with a `StandardScaler` fitted on the 2,400 training windows (about 40 hours). | `M:386-390`. | When the price drifts, the inputs leave the training range. Frozen for the test window: 88% of repo origins were more than 3 training standard deviations out. Even with daily retraining, 51–71% were out on 3 of 14 clean-set days. |

These defects are outside the scope of the request. They matter here because they set the bar a new model must beat. Section 7.8 measures that bar.

---

## 4. Timestral: facts that matter for us

Sources: the source code (`github.com/mistralai/timestral`, commit a534bdb), the Hugging Face model cards and LICENSE, the blog (`mistral.ai/news/timestral`) and the technical report (arXiv:2610.37585, 2026-10-27).

| Topic | Fact | Consequence for neural_trade |
|---|---|---|
| Model | Causal patch transformer, patch 32, "Decoupled Horizon Forecasting". 21 non-crossing quantiles (0.01 … 0.99) per step. | One call gives all three horizons (steps 1, 5, 15). Each step has its own output. |
| Inputs | Univariate or multivariate targets. Past-only and known-future covariates. Context up to 32,768 steps. | Close, plus volume and range as covariates, is possible. |
| Normalisation | Expanding (cumulative) mean and standard deviation over the context, then asinh (`scaler.py:14-101`). | On price levels, a typical 1-min move is about 0.04 × the standard deviation of the price over a 2k context, and 0.013 × over a 32k context (median over our origins). Context length matters (Section 7.3). |
| Default test-time augmentation | `symmetric_averaging=True`: it forecasts x and −x and averages the mirrored results (`forecaster.py:221-228`). That is 2 forward passes. | The forecast for −x is the mirror image of the forecast for x. This removes any fixed up/down bias of the network. Skew that comes from the price history still shows: on the repo set, the ratio of upper to lower tail width of Timestral-28M was 0.82–1.19 at h = 1 (1st–99th percentile). |
| Training data | Pretraining is synthetic only. Post-training uses "real and synthetic" data; **the real datasets are not named**. | Our Oct–Nov 2025 CSV may be in post-training. That is why we added a clean post-release test. |
| Fine-tuning | The released package is **inference-only**: no loss and no training code. Mistral offers fine-tuning only through "Mistral Forge". | We cannot adapt it to our data without re-building the training pipeline. |
| Benchmarks | #1 zero-shot on GIFT-Eval, TIME and fev-bench (vendor-run, reproduced from public CSVs). **No benchmark has 1-step-ahead minute-level finance.** The only crypto task (TIME, daily) is at random-walk level. | Leaderboard rank does not transfer to our task. Finance rank correlates only r = 0.43 with GIFT-Eval rank (FinVerse, arXiv 2608.03259). |
| Latency (vendor) | 17 ms (5M) to 113 ms (209M) on 8 server threads, with bf16 and `torch.compile`. | The package default is fp32 without compile. We measured it on 4 CPUs (Section 7.7). |
| **Licence (weights)** | "Modified MIT". Clause 2: **no rights at all if the global consolidated monthly revenue of your company, or of your employer, exceeded $20M** in the preceding month. It covers derivatives and "combined works". A commercial licence is at Mistral's discretion. The code is Apache-2.0. | Legal must confirm eligibility before any use, including benchmarks. |
| Support | The repo accepts no PRs. `CONTRIBUTING.md` promises best-effort maintenance only. Open issues had no maintainer reply as of 2026-10-30: #3 asks which real data sets were used in post-training; #7 reports that upper quantiles blow up with sign-fixed, piecewise-constant known-future covariates such as prices, and symmetric averaging is one trigger. | Treat it as a research artefact, not a supported product. |

---

## 5. Samay: facts that matter for us

Sources: `github.com/AdityaLab/Samay` (HEAD d23c90f, 2026-09-19), and our own install and runs.

| Topic | Fact |
|---|---|
| What it is | An academic wrapper around TimesFM 1.0/2.0/2.5, Chronos (T5, Bolt, Chronos-2), Moirai 1.1/MoE/2.0, MOMENT, TinyTimeMixer, Time-MoE and LPTM. It vendors copies of their code (about 40k lines). |
| Coverage | It does **not** wrap Timestral, TimesFM-3, TTM-r3, Toto-2, TiRex or Sundial. The last model was added on 2025-10-31. |
| PyPI name clash | `pip install samay` installs an **unrelated 2023 package**. AdityaLab Samay installs only from git. This is a dependency-confusion risk. |
| API | It is built around CSV datasets with fixed train/test fractions. There is no public "forecast this numpy window" method, except a thin TimesFM pass-through. |
| **Install, tested by us** | **`pip install` from git on Python 3.13 resolves, but the models break.** With transformers 5.22, its vendored Chronos-2 *silently* returned NaN or values 1.1% off the official package *from the same weights*. Its vendored Chronos-Bolt failed to load (`T5Stack.__init__` signature); a bare `except` re-raised this as a misleading "Repository not found" error. **It worked for us only in the exact `uv.lock` environment** (Python 3.11, torch 2.5.1, transformers 4.47.1). That environment conflicts with neural_trade's pins (torch 2.8, TF 2.20, Python 3.13). |
| Correctness | Its Chronos and Chronos-Bolt `evaluate()` reshape (B·C, H, Q) quantiles to (Q, B, C, H) without a transpose (`model.py:572-577`, `792-797`). That scrambles quantile metrics. |
| Fine-tuning | Hard-coded loops: fixed LR and epochs for most models, no validation, no early stopping, `save()` not implemented. In its own notebooks, fine-tuning made test MSE **worse** for 4 of 5 models (single runs, so weak evidence). |
| Licences | Samay code is Apache-2.0. **All Moirai weights it supports are CC-BY-NC-4.0 (non-commercial).** LPTM weights declare no licence. |
| Maintenance | No tests and no test CI. Package import was broken on main from 2026-02-24 to 2026-06-22. 36 stars, 9 commits in 2026. |
| Leaderboard in repo | Non-standard metrics, fixed 96/512 horizon/context, duplicate rows with values like 1e67. No intraday price task. Not usable. |

---

## 6. Benchmark design

### 6.1 Data

| Set | Source | Forecast origins | Purpose |
|---|---|---|---|
| **Repo holdout** | `binance_btcusdt_1min_ccxt.csv` (43,500 bars, 2025-10-11 → 2025-11-10, no gaps) | Last 20% of rows, every 3 min: **2,895 origins** (2025-11-04 → 11-10) | Model and config selection |
| **Post-release (clean)** | Binance public market-data mirror, BTCUSDT 1-min, 2026-09-20 → 2026-10-29 (57,600 bars, no gaps) | Every 3 min from 2026-10-16 00:00 UTC: **6,715 origins** | Out-of-sample check. Targets are after Timestral's release, so they cannot be in any model's training data. |

### 6.2 Target and scoring

- Target: log return y_h = ln c[t+h] − ln c[t], in basis points, for h = 1, 5, 15 (the neural_trade horizons).
- Every model forecasts from the bars up to and including t. No model sees data after t.
- Baselines are fitted only on data before the first scored origin.
- **Reference baseline: EWMA-Student-t (EWMA-t).** The variance is an exponentially weighted moving average of past squared 1-min returns: σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ. The quantiles are σ·√h·t_ν (Student-t, unit variance). We chose λ and ν on pre-holdout data by 21-level pinball loss, from {0.90 … 0.995} × {2.5 … ∞}. The choice was λ = 0.90–0.97 and ν = 6 (Appendix A, window facts). It has 2 parameters and runs in microseconds.
- **Two stronger cheap baselines** (added after our internal review). Both use only pre-holdout data and have no extra free parameters:
  - **EWMA-t, scale refit:** the same EWMA-t with its variance multiplied by c_h = mean(y² / forecast variance) on pre-holdout data. On pre-holdout data the realized squared returns were 20–45% above the variance of the pinball-tuned EWMA-t (c_h = 1.20–1.45), because pinball loss does not penalise a too-low variance strongly.
  - **FHS (filtered historical simulation):** the same EWMA σ, with quantiles taken from the empirical distribution of pre-holdout returns divided by σ. It needs no assumption about the tail shape.
- Other baselines:
  - random walk (zero return);
  - EWMA-Gaussian (λ = 0.94);
  - trailing 1-day empirical quantiles;
  - momentum and reversal signs;
  - "revert to the context mean" sign;
  - logistic regression on past returns (fitted pre-holdout).

| Question | Metric |
|---|---|
| Point forecast | MAE skill vs random walk = 1 − MAE / MAE_rw (> 0 is better) |
| Direction | MCC of sign(median), and AUC of P(up) from the quantiles. For neural_trade, also AUC of its direction head. |
| Distribution | Pinball loss on the 9 deciles (fair to decile-only models) and on 21 levels. Coverage of the 80% and 98% intervals. Tail hit rates, with a Kupiec test and a Newey-West test (60 min of lags) that stays valid when the h-step windows of nearby origins overlap. We use the Newey-West result for pass/fail (p < 0.05 = fail). |
| Variance | QLIKE of the quantile-implied variance v against y², in the unit-free form y²/v − ln(y²/v) − 1. It is lowest when v is right, and it penalises a v that is too low more than one that is too high. Origins with y = 0 (2–4%) are left out of QLIKE, because the loss is undefined there. Spearman rank correlation of interval width with \|y\|. |
| Economics | Non-overlapping trades when \|forecast\| > cost, at round-trip costs of 4, 10 and 20 bps (futures maker, futures taker, spot). Trades fill at the forecast bar's close, with no slippage and no delay, so real results would be worse. |

- **Uncertainty.** 95% moving-block bootstrap CIs with 1,000 resamples. Blocks are 60 min of origins on the full sets, 90 min on the every-6th-origin subsets and 63 min on the every-3rd subset. CIs barely change with 1-day blocks. For model-vs-baseline comparisons the CI is on the *paired* per-origin loss difference. "Significant" means the 95% CI excludes 0 (0.5 for AUC).
- **Many tests.** The tables hold several hundred CIs. About 1 in 20 excludes 0 by chance alone. We therefore treat a single significant cell as weak evidence, and we base conclusions on patterns that repeat across horizons, sizes or sets.
- **Scope.** Zero-shot only. We did not fine-tune any foundation model.

---

## 7. Results

The full tables are in Appendix A. They are generated from `research/tsfm_eval/results/*`. In the summary tables below, **bold** means the difference is significant: its 95% CI excludes 0 (0.5 for AUC). Bold does not mean better. In loss tables a positive value is worse and a negative value is better.

### 7.1 Price and direction: no model has usable skill

| Model (best config) | MAE skill vs RW, repo (1/5/15) | MAE skill vs RW, clean (1/5/15) | AUC of P(up), clean (1/5/15) |
|---|---|---|---|
| Timestral-209M, ctx 2048 | −0.30 / +0.06 / −0.04 % | −0.35 / −0.12 / −0.11 % | 0.513 / 0.500 / 0.509 |
| Timestral-28M, ctx 2048 | −0.61 / −0.17 / −0.29 % | −0.64 / −0.15 / −0.06 % | 0.502 / 0.504 / 0.515 |
| Chronos-2, ctx 512 | −3.42 / −2.49 / −3.50 % | −6.17 / −2.52 / −3.67 % | 0.493 / 0.499 / 0.497 |
| TimesFM-2.5 via Samay, ctx 512 | −0.83 / −2.52 / −4.51 % | −3.07 / −2.01 / −3.35 % | 0.491 / 0.511 / **0.534** |
| neural_trade as-is (3 seeds) | −4.4 … +0.2 % | −9.6 … −0.6 % | 0.487 … 0.509 (head 0.490 … 0.513) |
| Logistic regression, 5 features (1/5/15/60-min returns, 60-min volatility), fitted once before the test | 0 (no point forecast) | 0 | 0.500 / **0.516** / 0.512 |

- No model beats a random walk on price at any horizon on either set. The best MAE skill is +0.2% (neural_trade seed 1, repo, h = 15), and its CI includes 0.
- **Direction is at chance.** One pattern needs care. On the clean set at h = 15, Timestral-5M, -28M and -209M at ctx 512 and TimesFM-2.5 show MCC 0.04–0.06 and AUC 0.52–0.53. Their AUC CIs exclude 0.5. We did not run 123M on this set. The same configurations show nothing on the repo set. In that clean window a plain 15-minute reversal rule scores MCC 0.09. On both sets, Timestral's h = 15 calls agree with that rule 55–65% of the time, so its median has a built-in mild reversal tilt. **We treat the clean-set result as a window effect, not a signal.** The trades it implies lose money after costs (7.7).
- **Chronos-2 with price input is anti-predictive at 15 minutes on the repo set** (AUC 0.452), and close to chance on the clean set.

### 7.2 Distribution and variance: a small gain over strong cheap baselines

We compare each model with three cheap baselines (Section 6.2). EWMA-t is the reference in Appendix A. FHS is the strongest of the three on pinball loss and tails. Values are % change in loss; negative is better. The CI is paired, per origin.

| Model | Set | Pinball vs FHS, h = 1 / 5 / 15 | QLIKE vs EWMA-t | QLIKE vs FHS |
|---|---|---|---|---|
| Timestral-28M, ctx 2048 | repo | **+0.54** / +0.31 / +0.19 | −0.68 / −1.26 / −1.65 | −0.56 / **−1.23 / −1.59** |
| Timestral-28M, ctx 2048 | clean | +0.21 / −0.07 / −0.03 | **−5.56 / −3.41 / −4.49** | **−3.01 / −1.98** / −2.05 |
| Timestral-209M, ctx 2048 | repo | +0.17 / −0.02 / −0.17 | −0.71 / **−1.29** / −1.70 | −0.59 / **−1.27** / −1.65 |
| Timestral-209M, ctx 2048 | clean | −0.05 / −0.13 / −0.03 | **−5.39 / −3.98 / −4.12** | **−2.84 / −2.56** / −1.67 |
| Chronos-2, ctx 512, price input | clean | **+5.30 / +2.56 / +3.61** | −1.95 / +3.22 / **+5.75** | +0.69 / **+4.75 / +8.45** |
| neural_trade as-is, seed 0 | clean | **+10.59 / +5.83 / +6.83** | **+10.30 / +17.38 / +17.15** | **+13.28 / +19.12 / +20.14** |

- **Against EWMA-t the clean-set gain looks large (3–6%).** Most of that comes from a known flaw of EWMA-t: we tuned it on pinball loss, and before the test the realized squared returns were already 20–45% above its variance. FHS and the scale-refit EWMA-t fix that for free.
- **Against FHS the gain is 0.6–3.0%.** It is significant at h = 5 on both sets, at h = 1 on the clean set and at h = 15 on the repo set (28M). It is not significant at h = 1 on the repo set or at h = 15 on the clean set. Against the scale-refit EWMA-t the gain is 1.0–2.4% (T-B2).
- **Pinball loss equals FHS.** The only significant cell is 28M at h = 1 on the repo set (+0.54%). 209M has 0–0.4% lower pinball loss than 28M.
- QLIKE here is the unit-free form (Section 6.2). Its percentages are larger than in a bps-based QLIKE, but the paired tests are the same.

Tail calibration. Each cell shows how often the real return fell below q0.01 and above q0.99. The target is 1% for each. The test is the Newey-West test (Section 6.2). "Pass" means p ≥ 0.05. "Too wide" means the tail was hit less often than 1%, so the forecast was too cautious.

| Model | Set | h = 1 | h = 5 | h = 15 | Fails |
|---|---|---|---|---|---|
| EWMA-t (reference) | repo | 1.14% / 0.97% | 0.93% / 1.04% | 0.83% / 1.07% | 0 of 6 |
| FHS | repo | **0.59% (too wide)** / 0.86% | **0.66% (too wide)** / 0.83% | 0.59% / 0.66% | 2 of 6 |
| Timestral-28M, ctx 2048 | repo | **0.59% (too wide)** / 0.90% | 0.86% / 1.17% | 0.73% / 1.42% | 1 of 6 |
| Timestral-209M, ctx 2048 | repo | 0.73% / 0.93% | 0.90% / 1.11% | 0.69% / 1.38% | 0 of 6 |
| EWMA-t (reference) | clean | **1.59% / 1.56%** | **1.49% / 1.38%** | **1.82%** / 1.36% | 5 of 6 |
| FHS | clean | 0.98% / **0.73% (too wide)** | 0.94% / 0.83% | 1.38% / 0.82% | 1 of 6 |
| Timestral-28M, ctx 2048 | clean | 0.88% / 0.95% | 1.16% / 1.13% | **1.59%** / 1.07% | 1 of 6 |
| Timestral-209M, ctx 2048 | clean | 0.91% / 0.88% | 1.12% / 1.15% | **1.55%** / 1.16% | 1 of 6 |

**Bold** = fail (p < 0.05).

What this means:
- Timestral's quantiles are about as good as those of a well-built cheap baseline. Their variance scaling is slightly better: QLIKE is 0.6–3.0% lower than FHS.
- Its tails are about as well calibrated as FHS. The plain EWMA-t looked bad on the clean set only because its tails were too thin from the start.
- The gain is real but small. It is larger on the clean set, which was quieter and falling.

### 7.3 Input framing, and a fair test of the free alternatives (T-B2, T-J)

Timestral reads raw price levels by default. Price-level input may be unfair to other models (Section 4), so we also gave Timestral and Chronos-2 return series:
- **1-minute log returns.** The model forecasts per-minute returns, so only h = 1 has a forecast distribution.
- **h-minute returns.** For each horizon h, the input is the series of non-overlapping h-minute returns that end at the origin, and the model forecasts one step. This gives a distribution for every horizon.

| Comparison on the same origins (QLIKE, % change; negative = first model better) | Repo, h = 1 / 5 / 15 | Clean, h = 1 / 5 / 15 |
|---|---|---|
| Chronos-2 with 1-min log returns vs with price levels (ctx 2048) | **−4.5** / – / – | +4.1 / – / – |
| Chronos-2 vs Timestral-28M, both with 1-min log returns (ctx 2048) ¹ | +1.0 / – / – | **+6.6** / – / – |
| Chronos-2 vs Timestral-28M, both with h-min returns (512 bars) | +0.4 / +0.4 / **+2.4** | **+3.5 / +3.5** / −0.0 |
| Chronos-2 vs Timestral-28M, both with h-min returns (2048 bars, every 6th origin) | +3.4 / +0.0 / +2.1 | **+2.4** / +3.0 / +4.8 |
| Timestral-28M: h-min returns (2048 bars) vs price levels (ctx 2048) | +0.2 / **+2.4 / +3.6** | +0.7 / +1.1 / +3.0 |
| Chronos-2, h-min returns (512 bars) vs Timestral-28M, price levels (ctx 2048) | **+2.4 / +4.7 / +9.1** | **+5.8 / +6.0** / +4.0 |

¹ On the clean set the comparison is with Timestral-28M on price levels, because its log-return run there used every 6th origin only. At h = 1 the two Timestral inputs give the same result (T-J).

- **Return input helps Chronos-2 a lot.** At h = 1 its pinball loss is 3–8% lower than with price levels, on both sets. Price levels are a poor input for it.
- **On equal input, Timestral still has a small edge.** Pinball loss is equal (no significant cell; T-J). QLIKE is 0–7% higher for Chronos-2, significant in 5 of 14 cells.
- **For Timestral, price levels are the best input at 5–15 minutes.** Its h-minute return input gives about the same pinball loss but 0–4% higher QLIKE, and with it Timestral no longer beats FHS at 5–15 minutes (T-B2).
- **Chronos-2 with its best input does not beat FHS on QLIKE** (+0.6% to +7.4% on the repo set, +1.9% to +3.9% on the clean set; T-B2). So it is not a substitute for Timestral in the variance role. On the volatility series (7.5) it is.
- **TimesFM-2.5** (through Samay): log-return input lowers its pinball loss by about 2% at h = 1. It gives only deciles, so we cannot score its tails or QLIKE fairly.

### 7.4 Context length, size and options (T-G, T-J)

T-J compares each variant with its base configuration on identical origins. T-J holds about 50 paired cells per set, so about 2–3 cells would be significant by chance alone. We therefore treat a single significant cell as weak evidence.
- **Context of at least 512 is needed; 2048 is the best trade-off.** At ctx 64 (our 60-bar lookback), pinball loss is 1.4–2.5% worse than EWMA-t. At ctx 512 it is 0.1–1.5% worse. Going from 512 to 2048 (28M) lowers pinball loss by up to 0.8% and QLIKE by up to 2.3%. Two of the twelve cells are significant, both on the clean set.
- **Context 8192 adds little.** Against ctx 2048 on the same origins, 209M lowers QLIKE by 1.6% at h = 1 on the repo set (significant). No other cell is significant, on either set or for 28M. Latency is 2.6× higher.
- **Context 32768 hurts or does nothing.** For 28M on the repo set, pinball loss is 3.6% worse and QLIKE 8.0% worse at h = 1 than at ctx 2048 (significant). On the clean set no cell is significant. Latency is about 11× higher.
- **Size.** 209M has 0–0.4% lower pinball loss than 28M (significant at h = 1 on both sets and at h = 5 on the repo set). QLIKE is the same. 5M is 0.1–0.6 points worse than 28M (unpaired).
- **Covariates** (log volume, high-low range and return as past covariates) give no reliable gain:
  - 28M, repo: QLIKE −1.1% at h = 1 (significant); no other cell is significant.
  - 209M, repo (every 6th origin): no cell is significant.
  - 28M, clean (every 3rd origin): pinball loss is 0.9% *worse* at h = 5 and 15 (significant).
  - They are also about 10× slower in our batched runs.
- **Symmetric averaging off** halves latency. Quality changes by −0.2% to +0.8% (one significant cell: QLIKE +0.8% at h = 1, clean, ctx 512).

### 7.5 Volatility role: forecast the mean high-low range of the next h minutes (T-E)

The target is the mean 1-minute high-low range (bps) over the next h minutes. Timestral forecasts this series directly at ctx 2048. We use the mean of its forecast quantiles. Its median would be much worse, because the range is skewed and the median sits below the mean.

| Timestral-28M against | Repo (2,895 origins), h = 1 / 5 / 15 | Clean (6,715 origins), h = 1 / 5 / 15 |
|---|---|---|
| EWMA of the range (λ 0.94), QLIKE | **−5.1% / −6.9% / −9.5%** | **−7.5% / −14.3% / −17.4%** |
| HAR, bias-corrected, QLIKE | **−28%** / −0.2% / +1.5% | **−52% / −14% / −4.3%** |
| HAR, plain log-OLS, mean abs log error | **−6.7%** / +1.0% / +2.0% | **−10.1% / −1.6%** / −0.2% |

- HAR uses 6 log-range regressors (1, 5, 15, 60, 240 and 1,440 min). It is fitted on all data before the first origin (23–25 days). Plain log-OLS is biased low in levels; the bias-corrected ("smeared") version fixes that for QLIKE.
- At 1 minute both HAR variants do poorly, because the 1-minute range is very skewed.
- Chronos-2 (Apache-2.0 weights) is about as good as Timestral here: QLIKE 4.9–15.0% lower than the EWMA, against 5.1–17.4% for Timestral-28M. 5M, 28M and 209M are about the same.
- **Conclusion:** Timestral is at HAR level at 5–15 minutes on the repo set and somewhat better on the clean set. At 1 minute it beats both HAR variants. This matches the published evidence: zero-shot foundation models reach HAR level on realized volatility (Goel et al. 2025; Brini 2026).

### 7.6 Feature role: no added information (T-F)

We used walk-forward logistic regression with a 15-minute purge. The base model has 6 features: the same 5 as in 7.1 plus the last 1-minute range. We added Timestral or Chronos-2 outputs (both sets) or neural_trade outputs (clean set only). ΔAUC was between −0.012 and +0.012. No value is significantly positive, and two are significantly negative. The base model alone reaches AUC 0.517–0.541 on the last ~80% of clean-set origins (walk-forward). On the clean set it is the best direction model in this study, and it is still weak.

### 7.7 Economics (T-D)

- We traded only when |forecast| > round-trip cost, with one position at a time. Trades fill at the forecast bar's close, with no slippage and no delay, so real results would be worse.
- Timestral at ctx 2048 almost never forecasts a move above 4 bps, so it makes 0–73 trades per window (6 days on the repo set, 14 days on the clean set).
- Configurations that trade more lose money: Chronos-2, Chronos-Bolt, TimesFM-2.5, Timestral at ctx 512 or with h-minute return input, and neural_trade. With 100 or more trades on the full origin sets, the net result at 4 bps cost is −1.1 to −7.9 bps per trade, with t down to −8.2 (Chronos-Bolt, clean set, h = 1). At 10–20 bps costs almost every configuration loses money.
- A few small samples are positive (n ≤ 47). One is nominally significant: Timestral-28M, ctx 2048, symmetric averaging off, repo set, h = 5 (n = 16, +11.9 bps per trade, t = 2.8). We ran about 300 trade tests, so one such result is weak evidence. With default settings the same model loses on the same origins (n = 11, −3.0 bps).
- At the spot fee (20 bps round trip) almost no forecast clears the cost.
- These results agree with published BTC results: accuracy peaks at 51–54% at 1–15 minutes, and naive sign trading loses after costs (Jaquart et al. 2021).

### 7.8 Latency against NFR-1.1 (< 1 s per prediction) (T-H)

Single-series calls on 4 CPU threads, fp32, package defaults (no `torch.compile`), on an idle machine:

| Model | Context | p50 / p99 ms, symmetric averaging on | p50 / p99 ms, off | Cold start | Peak RSS |
|---|---|---|---|---|---|
| Timestral-28M | 2048 | 29 / 36 | 15 / 21 | 0.15 s | 0.6 GB |
| Timestral-209M | 2048 | 162 / 208 | 80 / 101 | 0.95 s | 2.5 GB |
| Timestral-209M | 8192 | 420 / 493 | 210 / 262 | | |
| Timestral-209M | 32768 | **1,759 / 1,870** | 887 / 981 | | |
| Chronos-2 (official) | 2048 | 77 / 107 | – | 1.2 s | 1.0 GB |
| neural_trade as-is (`Model.predict`, as in `model.py`) | 60 | 59 / 73 | – | | |

- Every Timestral size meets NFR-1.1 by a wide margin at ctx ≤ 8192. Only ctx 32768 with 123M or 209M fails (p50 1.13 s and 1.76 s).
- The recommended setting (28M, ctx 2048) takes about 30 ms, half of neural_trade's current `Model.predict` path. One forecast per minute per symbol is a negligible load.
- Cold start is load plus the first call. Peak RSS is for the whole process and is an upper bound.
- neural_trade's latency is the same at 1 thread. A compiled `tf.function` call of the same model takes 19 ms.

### 7.9 Head-to-head: neural_trade

- **Setup:**
  - We retrained it with its own pipeline on the same data, with 3 seeds. Only the IO paths and the seed were changed; `model.py` was imported unmodified.
  - It uses its own 2,880-sequence window, loss-weight calibration and early stopping. Each training run takes about 2–3 minutes on 1 CPU thread, with 40 optimizer steps (8 epochs × 5 batches).
  - **As-is:** trained once on data up to the first origin, then frozen for the whole test window (6 days repo, 14 days clean). Its inputs are price levels, scaled with a fit on about 40 hours of training windows. On the repo set, 88% of origins had inputs more than 3 training standard deviations from the mean. On the clean set only 0.5% did.
  - **Retrained daily:** a new model every 1,440 minutes, trained on all data up to the start of its day, as a fairer test. Its inputs are still raw price levels, so they still left the training range on some days: on 3 of 14 clean-set days, 51–71% of origins were more than 3 training standard deviations out.
- **Direction head:** it says DOWN at every origin in 13 of 18 seed-horizon runs. The worst run says UP at 14% of origins (repo, seed 0, h = 5). The loss-weight calibration clamped to 1000 on every run.
- **Point forecasts** are about 2.5–17× smaller than real moves (mean absolute size, by seed and horizon). Their correlation with realized deltas is −0.02 to +0.09.
- **Distribution (as-is):** pinball loss 1.6–10.7% worse and QLIKE 10–21% worse than EWMA-t (17 of 18 QLIKE cells significant). Volatility ranking is 0.05–0.16 against 0.31–0.40 for the EWMA baselines. The 21-level grid keeps only about 79% of its t(3) variance, so its QLIKE here is, if anything, slightly flattering.
- **Tails (as-is):** mixed. On the clean set at h = 1, its 98% coverage (0.985–0.991) is closer to nominal than EWMA-t's (0.968), and seed 0's 99% tail passes where EWMA-t fails. At 5 and 15 minutes on the clean set, its intervals are far too wide.
- **Retrained daily: no better.** Against the frozen model on the same origins, pinball loss changed by −0.4% to +3.0% and QLIKE was 3–8% worse on the repo set. On the clean set the result was mixed (pinball −3% to +0.3%, QLIKE −8% to +8%). Against EWMA-t it is still 2.6–8.2% worse on pinball loss and 7.5–27% worse on QLIKE (16 of 18 QLIKE cells significant). Its volatility ranking is even weaker (Spearman −0.10 to 0.12). Stale weights do not explain its poor scores.
- **The only cell with an apparent direction signal** is repo h = 15: MCC 0.073 and 0.065 for seeds 0 and 1, but 0.024 for seed 2. It is absent on the clean set. **It is not robust.**
- **Bottom line:** in its present state, frozen or retrained daily, neural_trade loses to a 2-parameter EWMA-t on pinball loss and volatility ranking in every cell, and on QLIKE in every cell by point estimate. Its tail calibration is mixed. In the variance and risk role, FHS, EWMA-t, HAR and Timestral are all upgrades over the current head.

---

## 8. Where Timestral fits, judged by the loss math

Timestral emits 21 quantiles per step, trained with pinball loss on all levels. Each neural_trade head is matched against that output below.

| neural_trade head and loss | What Timestral can supply | Mathematical fit | Measured value | Decision |
|---|---|---|---|---|
| **Price**: pinball τ = 0.4. The 5-min head is pulled toward q0.4; the 1- and 15-min heads are set mainly by the Student-t NLL (Section 3.2). | q0.4 and the median are among its 21 levels | Exact. It also removes the pooled-scaler mismatch across horizons. | None. No model beats random walk. The 5-min head is biased low by design: q0.4 sits about 0.15–0.20 standard deviations below the median on our data (0.25 for a Gaussian), so it loses to random walk by construction. | Do not replace. **Drop the point head**, or set its loss weight to 0 (Section 9, step 2). |
| **Direction**: focal BCE, α = 0.7 | P(up) = 1 − F(last close), interpolated from the quantiles | Good. It is a proper probability without the α bias, and the Gaussian "align" term becomes redundant. | None. AUC 0.48–0.53 on the full origin sets. Brier skill at most +0.0024. The current head is no better: it almost always outputs DOWN. | Do not replace. **Fix α** (or use plain BCE) and treat direction as unproven. |
| **Variance**: Student-t NLL, ν = 3, centred on the price head | Full predictive distribution in bps; variance by quantile integration; tail quantiles directly | **Best fit.** The pinball loss over 21 levels scores each of these quantiles properly. Together they approximate CRPS, a score for the whole forecast distribution. It needs no assumed distribution shape between q0.01 and q0.99. Beyond q0.01 and q0.99 the model gives no information. Its units are bps, not scaled units² that depend on the scaler. | Real but small. QLIKE 0.6–3.0% lower than FHS, a free baseline. Tails about as well calibrated as FHS. As a range forecaster, 5–17% lower QLIKE than an EWMA of the range. The current head is 10–21% *worse* than EWMA-t on QLIKE. | **This is the slot.** But a free FHS gives most of the gain, so start there. |
| **`helper_functions`** risk helpers: confidence = 1/(1+v), 2–6% stop, size multipliers. **Not wired today** (Section 3.1). | Interval width in bps; q0.05/q0.01 for stops | Good. It removes the unit dependence of `1/(1+v)` and the 0.7/0.5 thresholds. | If the stop were wired, it would sit 2–6% away (200–600 bps). The standard deviation of 15-minute returns is about 17 bps (clean set) to 26 bps (repo set), so that is 8–35 standard deviations away. It would almost never trigger. Quantile stops scale with the market. | **Use quantile-based, bps-denominated risk inputs.** Start with FHS. Then, after the licence check, A/B test Timestral. |
| **Trend, align, vol, L2 terms** | – | – | The trend term has zero gradient and distorts the loss-weight calibration. | Remove the trend term from the loss and from the calibration. |

**Why only the variance slot?** One-minute BTC returns are close to a martingale (lag-1 autocorrelation −0.011 on the repo file). They have strong volatility clustering and heavy tails (excess kurtosis 21). No model has anything to learn about the conditional mean. It can learn the conditional scale and shape, and that is where all the measured value sits. This matches the literature review (Section 10.3).

---

## 9. Recommendation and plan

Ordered by value per effort. Steps 1–3 are needed whatever the model choice. Each step names its owner role.

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | **Fix the evaluation set-up** (ML engineer). Do these five tasks. (a) Split the data into walk-forward folds, with a gap of at least 15 minutes between training and test data (a "purge"). (b) Keep a test set that early stopping and checkpoints never see. (c) Add fees: 20 bps round trip for spot, 4–10 bps for futures. Add a slippage estimate. (d) In the backtest, replace the centred rolling mean with a trailing one. (e) Make the column names match: read `datetime` in the loader (`model.py:201`), or write `timestamp` in the CSV writer (`dw_ccxt.py:114-116`). | Without this, no model change can be judged (D4–D6). | Small–medium |
| 2 | **Fix the model defects** (ML engineer). (a) Remove the × 20 forward scaling in `LearnableIndicators.call` (`M:1073`, `M:1085-1087`, `M:1110-1112`, `M:1135`, `M:1147`). Keep or retune the gradient scaling in `train_step` (`M:1903-1905`). (b) Drop the zero-gradient trend term from the loss and from the loss-weight calibration. (c) Set focal α = 0.5, or use plain BCE. (d) For the momentum periods (D7), write back and initialise with `log(expm1(max(p − 1, 1e-3)))`, the inverse of `softplus(raw) + 1`. (e) Drop the price (point) head, or set its loss weight to 0, and centre the Student-t head on zero. (f) Feed returns, or prices normalised within each window, instead of raw price levels with a fixed scaler (D8). Then retrain and measure again with the harness in `research/tsfm_eval/`. | The current model is worse than a 2-parameter baseline on pinball loss, QLIKE and volatility ranking. | Small |
| 3 | **Wire a risk module into the backtest** (ML engineer). None is wired today. Use **FHS** as its default variance input: an EWMA variance with tail shapes from past data, refitted daily. Use quantile stops and width-based confidence in bps. Add EWMA-t, FHS and random walk as **standing baselines** in the harness. Add Timestral-28M (ctx 2048) as a further baseline only after the licence check in step 4. | FHS gets most of the measured variance and tail benefit. It costs nothing and has no licence risk. | Small |
| 4 | **Optional model trial in the risk path.** **Precondition:** legal confirms that the $20M revenue clause does not apply. If it applies, the owner gets a commercial licence from Mistral first. Use Timestral-28M at ctx 2048 with price-level input. It has the same QLIKE as 209M at about 30 ms per call, but 0.3–0.4% higher pinball loss (Section 7.4). Chronos-2 is not a substitute here: it did not beat FHS on QLIKE (Section 7.3). Run an A/B test against FHS in the fixed backtest from step 1. Use at least 3 months of post-release data (Section 10.4, question 1). Write the pass rule down before the test starts. Suggested rule: adopt the model only if net PnL after fees is higher, the block-bootstrap 95% CI of the per-trade PnL difference excludes 0, and maximum drawdown is not worse. Otherwise keep FHS. | Expected gain: 0.6–3.0% QLIKE against FHS. No clear tail gain. | Medium |
| 5 | **Treat direction as a research question, not a model swap** (researcher). The only direction signal seen (a 15-min reversal, MCC 0.04–0.09) depends on the window and does not survive costs. Explore it with the walk-forward logistic model from Section 7.6, and with selective, cost-aware entry, before building more deep models. | Published accuracy at 1–15 min peaks at 51–54% (Jaquart et al. 2021). Our best measured direction AUC is 0.54. | Medium |
| — | **Do not** adopt Samay. **Do not** use Timestral for price or direction. **Do not** feed a foundation model a 60-bar context. **Do not** use a 32k context. | Sections 5, 7.1, 7.4 and 7.8. | — |

---

## 10. Limits, risks and open questions

### 10.1 Limits of this study
- **Two windows only.** Repo set: 6 days, 2025-11. Clean set: 14 days, 2026-10, quieter and falling (mean |1-min return| 3.0 bps against 5.0). More windows are needed before any production decision.
- **The clean set is not fully untouched.** It also informed the choice of context length and model size, so it is not a pure test for the recommended configuration.
- **Zero-shot only.** We did not fine-tune any foundation model. Timestral cannot be fine-tuned with the released code, and Samay's loops are not fit for use.
- **Subsampled runs.** The 8k/32k runs use every 6th origin (483 on repo, 1,120 on clean). Two covariate runs use every 6th (repo, 209M) or every 3rd origin (clean). Their CIs are wider.
- **neural_trade head-to-head** uses seeds 0/1/2, not the repo default seed 42. The system today scores only the 480 bars right after its training window. We tested it frozen for the whole window and retrained daily.
- **Baselines added after review.** FHS and the scale-refit EWMA-t were added after an internal review showed that the pinball-tuned EWMA-t was a weak variance reference. Both use only pre-test data and have no extra free parameters.
- **Batched timing.** Throughput figures in T-G were measured while other jobs ran. Only the latency in 7.8 was measured on an idle machine.
- **Costs.** We checked spot fees (10 bps per side) against Binance's fee page. We assumed futures fees of 2 bps (maker) and 5 bps (taker) per side. We did not model slippage or delay, so real trading results would be worse.

### 10.2 Risks
| Risk | Severity | Note |
|---|---|---|
| Timestral licence: $20M global consolidated monthly revenue of the user's company **or employer**; covers derivatives, combined works and benchmarks | High (legal) | A commercial licence is at Mistral's discretion. No hosted-API price was found. |
| Undisclosed Timestral post-training data | Medium | The repo-window results may be contaminated. The clean set avoids this. Its gains over FHS are similar to or larger than those on the repo set, so contamination does not explain the gain. |
| Two ML stacks (TensorFlow + PyTorch) in one process | Medium | Thread over-subscription and a larger image. Run any PyTorch forecaster in a separate process or service. |
| Vendor support | Medium | The Timestral repo accepts no PRs and promises best-effort maintenance only. Open issues had no maintainer reply as of 2026-10-30. |
| Samay dependency confusion (`pip install samay` installs another package) | High if adopted | Avoided by not adopting it. |

### 10.3 Literature in one paragraph
Across 2024–2026 studies, off-the-shelf zero-shot foundation models do not beat simple baselines on financial **returns**:
- out-of-sample R² of −1.4% to −2.8% on daily returns, below CatBoost (Rahimikia et al., arXiv 2511.18578);
- crypto return information coefficient (the correlation of forecast and outcome) within ±0.011 (Kronos, arXiv 2508.02739);
- MAE skill vs random walk of about 10⁻³ (arXiv 2606.27100).

On **volatility** they reach HAR level:
- zero-shot TimesFM with a long context matches HAR on daily realized variance built from 5-minute returns, and beats it on QLIKE at the 10% level; short-context zero-shot runs are weaker (arXiv 2505.11163). That study does not test minute-level horizons;
- among 9 models, only TTM beats log-HAR, by about 2% (arXiv 2607.05291).

Their **raw tails** tend to be too thin (arXiv 2410.11773; Jheng et al. 2026). Supervised BTC direction accuracy peaks at 51.9% / 52.9% / 53.8% at 1 / 5 / 15 min, and every strategy is negative at 30 bps costs (Jaquart et al. 2021). Our results agree with this literature on returns, volatility and direction. One result differs: Timestral's tails were better calibrated than the literature led us to expect.

### 10.4 Open questions
1. Does the variance gain over FHS hold over more windows and regimes, for example 3–6 months of rolling post-release data?
2. Does a better risk input change backtest outcomes (stop hit-rate, drawdown, net PnL), once steps 1–3 are done?
3. Does a fine-tuned model beat zero-shot Timestral on variance? An example is Chronos-2 with LoRA (low-rank adaptation, a cheap fine-tuning method) through the upstream package. Upstream supports this; Samay does not.
4. Does the Timestral licence apply to the operating entity or its group?

---

## 11. Reproduce

Everything is in `research/tsfm_eval/`. See its `README.md` for environments and commands. In short:
1. Main environment: Python 3.13, torch (CPU), `timestral` from git, `chronos-forecasting`, pandas, scipy, scikit-learn.
2. Samay environment: `uv sync --frozen` in a clone of AdityaLab/Samay at commit d23c90f (Python 3.11).
3. neural_trade environment: tensorflow-cpu 2.20 and keras 3.11.3.
4. Run order:
   - `bench/fetch_post.py`;
   - the `run_*.py` scripts, and `neural_trade_h2h/run_nt.py` and `run_nt_rolling.py`;
   - `metrics2.py` (with and without `BENCH_DATA=post`), `metrics_vol.py`, `incremental2.py` and `matched.py`;
   - `latency.py` and `neural_trade_h2h/nt_latency.py`;
   - `make_tables.py`.
5. Result summaries (JSON and markdown) are committed under `research/tsfm_eval/results/`. The forecast arrays are not committed because of their size; the scripts regenerate them.

---

## Appendix A. Full result tables

Generated by `research/tsfm_eval/bench/make_tables.py` from the committed result files. Do not edit by hand. Per data set: T-A point and direction, T-B distribution vs EWMA-t, T-B2 vs the two stronger baselines, T-C tails, T-D trades, T-E volatility role, T-F feature role. Then T-G ablations, T-J paired ablations, T-H latency and T-I Samay equivalence.

Notes for all tables. **Bold** = the 95% CI excludes 0 (0.5 for AUC). Bold does not mean better: in loss tables a positive value is worse and a negative value is better. QLIKE is the unit-free form y²/v − ln(y²/v) − 1 on origins with y ≠ 0; v is the variance implied by the 21 quantiles. Chronos-2 and Chronos-Bolt rows come from the official `chronos-forecasting` package; for these two models Samay gives the same numbers to 0.0008 bp (T-I). ‡ Log-return input: only h = 1 has a forecast distribution; h = 5 and 15 use the summed per-step medians as a point forecast. § h-min return input: for each h, the input is the series of non-overlapping h-minute returns ending at the origin (512 or 2048 bars), and the model forecasts one step. † Decile-only model (q0.1 … q0.9): its q0.01, q0.05, q0.95 and q0.99 are clamped to q0.1 / q0.9, so its 98% coverage, tail rates and QLIKE are biased against it. Compare it on 9-decile pinball only. neural_trade quantiles come from its Student-t head (ν = 3) centred on its price head; the 21-level grid keeps about 79% of a t(3) variance (97% for the t(6) EWMA-t), so its QLIKE here is slightly flattering to it. Trades fill at the forecast bar's close with no slippage and no delay.


### T-A (repo). Point and direction — Repo holdout (2,895 origins, 2025-11-04 → 11-10)

| Model | MAE skill vs RW (1 / 5 / 15 min) | MCC of sign (1 / 5 / 15) | AUC of P(up) (1 / 5 / 15) | Brier skill ×1000 (1 / 5 / 15) |
|---|---|---|---|---|
| Timestral-5M, ctx 512 | -0.85% / -1.02% / -1.20% | 0.002 / -0.016 / -0.003 | 0.500 / 0.491 / 0.500 | -11.1 / -9.7 / -13.7 |
| Timestral-28M, ctx 512 | -0.41% / -0.60% / -0.85% | -0.018 / -0.024 / -0.012 | 0.493 / 0.483 / 0.490 | -6.2 / -7.0 / -13.4 |
| Timestral-28M, ctx 2048 | -0.61% / -0.17% / -0.29% | -0.011 / 0.004 / -0.021 | 0.491 / 0.502 / 0.490 | -8.8 / -3.7 / -8.2 |
| Timestral-209M, ctx 512 | -0.33% / -0.48% / -0.49% | 0.006 / -0.004 / 0.009 | 0.501 / 0.500 / 0.501 | -3.6 / -4.1 / -7.7 |
| Timestral-209M, ctx 2048 | -0.30% / +0.06% / -0.04% | -0.013 / 0.031 / -0.016 | 0.492 / 0.513 / 0.493 | -3.7 / +0.1 / -2.6 |
| Chronos-2 (Samay family), ctx 512 | -3.42% / -2.49% / -3.50% | 0.014 / -0.049 / -0.071 | 0.504 / **0.469** / **0.452** | -45.4 / -36.9 / -47.1 |
| Chronos-2 (Samay family), ctx 2048 | -8.15% / -3.00% / -2.39% | -0.008 / -0.032 / -0.030 | 0.485 / 0.475 / 0.467 | -78.2 / -45.1 / -34.0 |
| Chronos-Bolt-S (Samay family), ctx 512 † | -8.66% / -4.79% / -4.64% | 0.019 / 0.018 / -0.003 | 0.509 / 0.498 / 0.501 | -66.7 / -48.5 / -57.7 |
| TimesFM-2.5 via Samay, ctx 512 † | -0.83% / -2.52% / -4.51% | 0.001 / -0.041 / -0.034 | 0.511 / 0.478 / 0.486 | -9.1 / -24.9 / -41.1 |
| TimesFM-2.5 via Samay, ctx 2048 † | -1.32% / -1.14% / -1.40% | -0.001 / 0.015 / -0.016 | 0.501 / 0.503 / 0.488 | -12.2 / -12.0 / -17.8 |
| neural_trade as-is, seed 0 | -2.57% / -2.10% / +0.07% | -0.026 / -0.010 / 0.073 | **0.478** (head 0.494) / 0.516 (head 0.510) / **0.538** (head 0.525) | -45.9 / -44.9 / -0.2 |
| neural_trade as-is, seed 1 | -0.63% / -2.08% / +0.20% | 0.021 / – / 0.065 | 0.510 (head 0.505) / 0.519 (head 0.511) / **0.533** (head 0.482) | -9.0 / -42.8 / +1.8 |
| neural_trade as-is, seed 2 | -4.41% / -0.04% / -0.13% | -0.042 / 0.009 / 0.024 | 0.484 (head 0.515) / 0.519 (head 0.518) / 0.524 (head 0.525) | -61.5 / +0.2 / -5.5 |
| neural_trade retrained daily, seed 0 | -5.93% / -2.18% / -1.32% | -0.006 / -0.008 / -0.014 | 0.494 (head 0.493) / 0.503 (head 0.470) / 0.483 (head 0.511) | -68.2 / -23.4 / -24.9 |
| neural_trade retrained daily, seed 1 | -2.87% / -2.84% / -1.21% | -0.037 / -0.015 / -0.001 | **0.477** (head 0.493) / 0.502 (head 0.465) / 0.507 (head 0.459) | -35.7 / -36.6 / -9.9 |
| neural_trade retrained daily, seed 2 | -4.66% / -1.96% / -0.22% | -0.039 / 0.009 / 0.066 | **0.476** (head 0.502) / 0.498 (head 0.489) / **0.539** (head 0.538) | -62.9 / -24.2 / +1.6 |
| Chronos-2, ctx 512, log-return input ‡ | -0.55% / -1.02% / -2.87% | -0.008 / 0.018 / -0.016 | 0.491 / – / – | -5.4 / – / – |
| Chronos-2, ctx 2048, log-return input ‡ | -0.11% / -0.18% / -0.51% | 0.009 / 0.018 / 0.028 | 0.497 / – / – | -1.5 / – / – |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | -1.08% / -2.26% / -5.59% | 0.011 / 0.016 / -0.011 | 0.496 / – / – | -13.9 / – / – |
| Timestral-28M, h-min return input, 512 bars § | -0.57% / -0.36% / -0.51% | 0.005 / 0.010 / 0.024 | 0.502 / 0.516 / 0.509 | -5.8 / -3.0 / -5.9 |
| Timestral-28M, h-min return input, 2048 bars § | -0.22% / -0.20% / -0.15% | 0.005 / 0.007 / 0.012 | 0.500 / 0.514 / 0.508 | -2.0 / -1.4 / -1.7 |
| Chronos-2, h-min return input, 512 bars § | -0.55% / -0.28% / -0.15% | -0.008 / -0.001 / 0.008 | 0.491 / 0.505 / 0.493 | -5.4 / -3.2 / -2.9 |
| EWMA-Gaussian (λ 0.94) | +0.00% / +0.00% / +0.00% | – / – / – | 0.500 / 0.500 / 0.500 | -0.2 / -0.1 / -0.1 |
| Trailing 1-day empirical quantiles | -0.07% / -0.20% / -0.31% | -0.030 / 0.009 / 0.022 | 0.488 / 0.507 / 0.520 | -1.0 / -1.6 / -1.9 |
| Logistic regression on past returns | +0.00% / +0.00% / +0.00% | 0.011 / 0.010 / 0.045 | 0.503 / 0.508 / 0.522 | – / – / – |
| 15/5/1-min reversal sign | +0.00% / -0.00% / +0.00% | 0.037 / -0.003 / 0.038 | – / – / – | – / – / – |


### T-B (repo). Distribution and variance vs EWMA-Student-t — Repo holdout (2,895 origins, 2025-11-04 → 11-10)

| Model | h | Pinball (9 deciles) vs REF, % [95% CI] | QLIKE vs REF, % [95% CI] | Cover 80% | Cover 98% | Spearman(width, \|y\|) |
|---|---|---|---|---|---|---|
| REF: EWMA-Student-t (λ 0.94, ν 6) | 1 | 0 (reference) | 0 (reference) | 0.795 | 0.979 | 0.395 |
| REF: EWMA-Student-t (λ 0.97, ν 6) | 5 | 0 (reference) | 0 (reference) | 0.796 | 0.980 | 0.359 |
| REF: EWMA-Student-t (λ 0.97, ν 6) | 15 | 0 (reference) | 0 (reference) | 0.785 | 0.981 | 0.310 |
| Timestral-5M, ctx 512 | 1 | **+1.08** [+0.53, +1.54] | -0.23 [-1.85, +1.30] | 0.828 | 0.982 | 0.390 |
| Timestral-5M, ctx 512 | 5 | **+1.12** [+0.57, +1.64] | -0.29 [-1.70, +1.11] | 0.819 | 0.977 | 0.355 |
| Timestral-5M, ctx 512 | 15 | **+1.45** [+0.46, +2.56] | +0.49 [-1.50, +2.48] | 0.803 | 0.978 | 0.293 |
| Timestral-28M, ctx 512 | 1 | **+0.45** [+0.00, +0.89] | -0.56 [-2.25, +0.98] | 0.835 | 0.986 | 0.399 |
| Timestral-28M, ctx 512 | 5 | **+0.67** [+0.22, +1.14] | -0.79 [-2.23, +0.58] | 0.825 | 0.980 | 0.356 |
| Timestral-28M, ctx 512 | 15 | **+1.02** [+0.19, +1.89] | -0.77 [-3.34, +1.59] | 0.820 | 0.979 | 0.300 |
| Timestral-28M, ctx 2048 | 1 | **+0.57** [+0.21, +0.92] | -0.68 [-2.26, +0.71] | 0.832 | 0.985 | 0.399 |
| Timestral-28M, ctx 2048 | 5 | +0.33 [-0.02, +0.68] | -1.26 [-2.76, +0.02] | 0.814 | 0.980 | 0.360 |
| Timestral-28M, ctx 2048 | 15 | +0.21 [-0.41, +0.87] | -1.65 [-4.03, +0.53] | 0.815 | 0.979 | 0.316 |
| Timestral-209M, ctx 512 | 1 | **+0.47** [+0.07, +0.88] | -0.19 [-1.76, +1.38] | 0.824 | 0.984 | 0.390 |
| Timestral-209M, ctx 512 | 5 | **+0.53** [+0.08, +0.99] | -1.00 [-2.42, +0.34] | 0.820 | 0.980 | 0.355 |
| Timestral-209M, ctx 512 | 15 | +0.56 [-0.33, +1.52] | -1.41 [-3.43, +0.47] | 0.815 | 0.979 | 0.304 |
| Timestral-209M, ctx 2048 | 1 | +0.19 [-0.16, +0.48] | -0.71 [-1.98, +0.53] | 0.822 | 0.983 | 0.399 |
| Timestral-209M, ctx 2048 | 5 | +0.00 [-0.30, +0.25] | **-1.29** [-2.70, -0.03] | 0.815 | 0.980 | 0.359 |
| Timestral-209M, ctx 2048 | 15 | -0.16 [-0.57, +0.28] | -1.70 [-3.82, +0.36] | 0.819 | 0.979 | 0.310 |
| Chronos-2 (Samay family), ctx 512 | 1 | **+4.63** [+3.17, +5.99] | +2.44 [-0.48, +4.88] | 0.823 | 0.989 | 0.385 |
| Chronos-2 (Samay family), ctx 512 | 5 | **+2.91** [+2.14, +3.80] | +0.61 [-1.22, +2.34] | 0.787 | 0.974 | 0.342 |
| Chronos-2 (Samay family), ctx 512 | 15 | **+3.17** [+2.02, +4.49] | +1.36 [-1.17, +3.90] | 0.773 | 0.965 | 0.294 |
| Chronos-2 (Samay family), ctx 2048 | 1 | **+9.38** [+7.35, +11.30] | **+5.19** [+2.10, +7.72] | 0.834 | 0.985 | 0.377 |
| Chronos-2 (Samay family), ctx 2048 | 5 | **+3.17** [+2.11, +4.15] | +0.76 [-1.55, +2.63] | 0.768 | 0.965 | 0.358 |
| Chronos-2 (Samay family), ctx 2048 | 15 | **+2.08** [+0.98, +3.08] | +0.99 [-2.00, +3.65] | 0.745 | 0.970 | 0.313 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 1 | **+8.91** [+7.12, +10.64] | **+3.33** [+0.93, +6.14] | 0.840 | 0.840 | 0.380 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 5 | **+4.96** [+3.63, +6.18] | **+13.30** [+9.07, +18.20] | 0.785 | 0.785 | 0.335 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 15 | **+4.85** [+2.81, +7.04] | **+19.02** [+12.22, +26.55] | 0.773 | 0.773 | 0.271 |
| TimesFM-2.5 via Samay, ctx 512 † | 1 | **+3.72** [+2.80, +4.64] | +1.27 [-1.08, +3.30] | 0.904 | 0.904 | 0.378 |
| TimesFM-2.5 via Samay, ctx 512 † | 5 | **+2.79** [+1.76, +3.79] | **+4.15** [+1.78, +6.94] | 0.829 | 0.829 | 0.355 |
| TimesFM-2.5 via Samay, ctx 512 † | 15 | **+4.75** [+2.37, +7.51] | **+12.30** [+6.62, +18.14] | 0.792 | 0.792 | 0.283 |
| TimesFM-2.5 via Samay, ctx 2048 † | 1 | **+2.47** [+1.72, +3.12] | **+3.29** [+1.04, +5.81] | 0.867 | 0.867 | 0.387 |
| TimesFM-2.5 via Samay, ctx 2048 † | 5 | **+1.66** [+0.93, +2.24] | **+7.78** [+4.63, +11.33] | 0.820 | 0.820 | 0.350 |
| TimesFM-2.5 via Samay, ctx 2048 † | 15 | **+1.55** [+0.37, +2.62] | **+12.98** [+5.94, +19.56] | 0.799 | 0.799 | 0.301 |
| neural_trade as-is, seed 0 | 1 | **+5.03** [+4.02, +6.10] | **+19.68** [+15.22, +25.05] | 0.778 | 0.980 | 0.157 |
| neural_trade as-is, seed 0 | 5 | **+4.57** [+3.34, +5.92] | **+12.82** [+8.12, +18.26] | 0.758 | 0.986 | 0.093 |
| neural_trade as-is, seed 0 | 15 | **+1.77** [+0.43, +3.30] | **+11.36** [+4.42, +18.85] | 0.801 | 0.992 | 0.095 |
| neural_trade as-is, seed 1 | 1 | **+3.52** [+2.55, +4.60] | **+20.00** [+15.44, +25.17] | 0.794 | 0.982 | 0.157 |
| neural_trade as-is, seed 1 | 5 | **+4.49** [+3.24, +5.74] | **+12.97** [+8.06, +18.50] | 0.760 | 0.985 | 0.108 |
| neural_trade as-is, seed 1 | 15 | **+1.62** [+0.58, +2.98] | **+11.19** [+3.71, +20.63] | 0.775 | 0.986 | 0.140 |
| neural_trade as-is, seed 2 | 1 | **+6.83** [+5.30, +8.46] | **+21.24** [+16.69, +26.38] | 0.796 | 0.985 | 0.115 |
| neural_trade as-is, seed 2 | 5 | **+2.60** [+1.85, +3.50] | **+14.63** [+9.91, +19.67] | 0.811 | 0.993 | 0.092 |
| neural_trade as-is, seed 2 | 15 | **+3.05** [+1.78, +4.61] | **+19.11** [+7.50, +35.62] | 0.696 | 0.971 | 0.062 |
| neural_trade retrained daily, seed 0 | 1 | **+8.21** [+6.81, +9.62] | **+25.44** [+19.77, +31.64] | 0.789 | 0.982 | 0.034 |
| neural_trade retrained daily, seed 0 | 5 | **+4.85** [+3.71, +6.00] | **+21.49** [+15.96, +27.55] | 0.811 | 0.989 | 0.035 |
| neural_trade retrained daily, seed 0 | 15 | **+3.75** [+2.38, +5.34] | **+18.70** [+11.26, +25.97] | 0.819 | 0.989 | 0.090 |
| neural_trade retrained daily, seed 1 | 1 | **+5.93** [+4.90, +7.15] | **+27.28** [+21.91, +33.49] | 0.803 | 0.981 | 0.104 |
| neural_trade retrained daily, seed 1 | 5 | **+6.01** [+4.70, +7.37] | **+22.14** [+16.74, +28.76] | 0.805 | 0.990 | -0.035 |
| neural_trade retrained daily, seed 1 | 15 | **+3.12** [+1.78, +4.70] | **+18.71** [+10.46, +28.36] | 0.822 | 0.988 | 0.026 |
| neural_trade retrained daily, seed 2 | 1 | **+7.27** [+5.96, +8.69] | **+25.52** [+20.59, +30.95] | 0.814 | 0.986 | 0.121 |
| neural_trade retrained daily, seed 2 | 5 | **+4.24** [+3.22, +5.35] | **+17.66** [+12.99, +22.85] | 0.810 | 0.990 | 0.109 |
| neural_trade retrained daily, seed 2 | 15 | **+2.63** [+1.37, +4.27] | **+22.40** [+11.16, +38.08] | 0.783 | 0.981 | -0.012 |
| Chronos-2, ctx 512, log-return input ‡ | 1 | **+1.02** [+0.59, +1.54] | **+1.69** [+0.50, +2.96] | 0.810 | 0.975 | 0.380 |
| Chronos-2, ctx 2048, log-return input ‡ | 1 | **+0.27** [+0.05, +0.49] | +0.45 [-0.72, +1.64] | 0.816 | 0.976 | 0.391 |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 1 | **+1.69** [+1.07, +2.30] | **+14.84** [+10.46, +20.50] | 0.803 | 0.803 | 0.365 |
| Timestral-28M, h-min return input, 512 bars § | 1 | **+1.05** [+0.51, +1.59] | +1.27 [-0.28, +2.86] | 0.823 | 0.980 | 0.379 |
| Timestral-28M, h-min return input, 512 bars § | 5 | **+1.10** [+0.56, +1.70] | **+2.98** [+0.60, +5.86] | 0.815 | 0.978 | 0.329 |
| Timestral-28M, h-min return input, 512 bars § | 15 | **+1.38** [+0.54, +2.34] | **+4.77** [+0.47, +8.72] | 0.817 | 0.984 | 0.238 |
| Timestral-28M, h-min return input, 2048 bars § | 1 | +0.20 [-0.08, +0.47] | -0.51 [-1.69, +0.60] | 0.824 | 0.979 | 0.398 |
| Timestral-28M, h-min return input, 2048 bars § | 5 | **+0.55** [+0.15, +0.98] | +1.10 [-0.88, +3.23] | 0.813 | 0.977 | 0.344 |
| Timestral-28M, h-min return input, 2048 bars § | 15 | +0.45 [-0.03, +1.04] | +1.86 [-1.23, +5.02] | 0.799 | 0.981 | 0.269 |
| Chronos-2, h-min return input, 512 bars § | 1 | **+1.02** [+0.59, +1.54] | **+1.69** [+0.50, +2.96] | 0.810 | 0.975 | 0.380 |
| Chronos-2, h-min return input, 512 bars § | 5 | **+0.86** [+0.43, +1.30] | **+3.39** [+0.91, +6.64] | 0.799 | 0.976 | 0.334 |
| Chronos-2, h-min return input, 512 bars § | 15 | **+1.38** [+0.79, +2.06] | **+7.29** [+3.77, +11.23] | 0.807 | 0.979 | 0.218 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 1 | +0.09 [-0.06, +0.23] | +0.58 [-0.46, +1.49] | 0.836 | 0.987 | 0.395 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 5 | **+0.16** [+0.00, +0.30] | +0.42 [-0.61, +1.33] | 0.835 | 0.986 | 0.359 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 15 | +0.25 [-0.05, +0.53] | +0.59 [-1.28, +2.39] | 0.830 | 0.987 | 0.310 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 1 | **+0.02** [+0.01, +0.04] | -0.12 [-0.56, +0.26] | 0.800 | 0.986 | 0.395 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 5 | +0.02 [-0.05, +0.09] | -0.02 [-0.63, +0.51] | 0.805 | 0.985 | 0.359 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 15 | +0.02 [-0.13, +0.15] | -0.06 [-1.18, +1.02] | 0.796 | 0.988 | 0.310 |
| EWMA-Gaussian (λ 0.94) | 1 | +0.10 [-0.06, +0.27] | -0.11 [-0.30, +0.06] | 0.828 | 0.969 | 0.395 |
| EWMA-Gaussian (λ 0.94) | 5 | **+0.31** [+0.09, +0.52] | +0.75 [-0.46, +1.82] | 0.823 | 0.966 | 0.356 |
| EWMA-Gaussian (λ 0.94) | 15 | **+0.37** [+0.07, +0.68] | +0.92 [-0.41, +2.24] | 0.809 | 0.970 | 0.306 |
| Trailing 1-day empirical quantiles | 1 | **+2.63** [+1.98, +3.36] | **+19.08** [+13.78, +24.97] | 0.811 | 0.976 | 0.217 |
| Trailing 1-day empirical quantiles | 5 | **+2.22** [+1.49, +2.93] | **+12.97** [+8.35, +18.80] | 0.802 | 0.977 | 0.217 |
| Trailing 1-day empirical quantiles | 15 | **+1.86** [+0.81, +3.16] | **+10.99** [+4.25, +17.21] | 0.800 | 0.970 | 0.196 |


### T-B2 (repo). Against two stronger cheap baselines (% change of loss; negative = model better)

| Model | h | Pinball (9) vs EWMA-t scale-refit, % | QLIKE vs EWMA-t scale-refit, % | Pinball (9) vs FHS, % | QLIKE vs FHS, % |
|---|---|---|---|---|---|
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 1 | -0.07 [-0.21, +0.07] | **-0.69** [-1.22, -0.10] | – | – |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 5 | -0.13 [-0.29, +0.03] | **-0.44** [-0.81, -0.02] | – | – |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 15 | -0.23 [-0.50, +0.03] | -0.64 [-1.36, +0.10] | – | – |
| Timestral-28M, ctx 512 | 1 | +0.36 [-0.09, +0.80] | **-1.13** [-2.14, -0.12] | +0.43 [-0.02, +0.86] | -0.44 [-1.83, +0.88] |
| Timestral-28M, ctx 512 | 5 | **+0.52** [+0.11, +0.97] | **-1.20** [-2.27, -0.12] | **+0.65** [+0.19, +1.12] | -0.77 [-1.88, +0.33] |
| Timestral-28M, ctx 512 | 15 | +0.78 [-0.04, +1.67] | -1.34 [-2.88, +0.36] | **+1.01** [+0.15, +1.92] | -0.71 [-2.50, +0.98] |
| Timestral-28M, ctx 2048 | 1 | **+0.47** [+0.11, +0.85] | **-1.25** [-2.17, -0.39] | **+0.54** [+0.19, +0.89] | -0.56 [-1.81, +0.55] |
| Timestral-28M, ctx 2048 | 5 | +0.18 [-0.18, +0.56] | **-1.67** [-2.92, -0.52] | +0.31 [-0.05, +0.66] | **-1.23** [-2.44, -0.12] |
| Timestral-28M, ctx 2048 | 15 | -0.04 [-0.73, +0.68] | **-2.23** [-3.76, -0.70] | +0.19 [-0.44, +0.86] | **-1.59** [-3.26, -0.03] |
| Timestral-209M, ctx 2048 | 1 | +0.10 [-0.22, +0.40] | **-1.28** [-2.15, -0.41] | +0.17 [-0.18, +0.46] | -0.59 [-1.63, +0.40] |
| Timestral-209M, ctx 2048 | 5 | -0.16 [-0.42, +0.12] | **-1.70** [-2.88, -0.39] | -0.02 [-0.34, +0.24] | **-1.27** [-2.43, -0.13] |
| Timestral-209M, ctx 2048 | 15 | -0.40 [-0.89, +0.12] | **-2.28** [-3.87, -0.60] | -0.17 [-0.61, +0.28] | -1.65 [-3.18, +0.01] |
| Timestral-28M, h-min return input, 512 bars § | 1 | **+0.95** [+0.45, +1.48] | +0.69 [-0.51, +2.02] | **+1.02** [+0.49, +1.56] | **+1.39** [+0.08, +2.77] |
| Timestral-28M, h-min return input, 512 bars § | 5 | **+0.94** [+0.43, +1.54] | **+2.55** [+0.41, +5.71] | **+1.08** [+0.56, +1.65] | **+3.01** [+0.75, +5.98] |
| Timestral-28M, h-min return input, 512 bars § | 15 | **+1.13** [+0.27, +2.10] | **+4.16** [+0.77, +7.45] | **+1.37** [+0.56, +2.28] | **+4.83** [+1.11, +8.33] |
| Timestral-28M, h-min return input, 2048 bars § | 1 | +0.11 [-0.16, +0.37] | **-1.08** [-1.93, -0.11] | +0.18 [-0.10, +0.45] | -0.39 [-1.34, +0.55] |
| Timestral-28M, h-min return input, 2048 bars § | 5 | **+0.40** [+0.01, +0.84] | +0.68 [-1.12, +2.93] | **+0.53** [+0.15, +0.93] | +1.12 [-0.65, +3.28] |
| Timestral-28M, h-min return input, 2048 bars § | 15 | +0.21 [-0.41, +0.90] | +1.26 [-1.53, +4.43] | +0.43 [-0.06, +1.03] | +1.92 [-0.90, +4.83] |
| Chronos-2 (Samay family), ctx 512 | 1 | **+4.53** [+3.05, +5.95] | +1.85 [-0.31, +3.62] | **+4.61** [+3.15, +5.96] | +2.56 [-0.04, +4.67] |
| Chronos-2 (Samay family), ctx 512 | 5 | **+2.75** [+1.94, +3.65] | +0.19 [-1.59, +2.10] | **+2.89** [+2.12, +3.76] | +0.63 [-1.15, +2.43] |
| Chronos-2 (Samay family), ctx 512 | 15 | **+2.92** [+1.74, +4.27] | +0.77 [-2.23, +3.90] | **+3.15** [+1.97, +4.50] | +1.42 [-1.17, +4.33] |
| Chronos-2, ctx 2048, log-return input ‡ | 1 | +0.17 [-0.04, +0.42] | -0.13 [-1.36, +1.42] | **+0.24** [+0.04, +0.46] | +0.57 [-0.53, +1.83] |
| Chronos-2, h-min return input, 512 bars § | 1 | **+0.93** [+0.50, +1.42] | +1.11 [-0.12, +2.60] | **+1.00** [+0.57, +1.52] | **+1.81** [+0.70, +3.13] |
| Chronos-2, h-min return input, 512 bars § | 5 | **+0.70** [+0.29, +1.14] | **+2.96** [+0.46, +6.56] | **+0.83** [+0.42, +1.28] | **+3.41** [+0.97, +6.86] |
| Chronos-2, h-min return input, 512 bars § | 15 | **+1.13** [+0.47, +1.90] | **+6.67** [+3.26, +10.66] | **+1.36** [+0.78, +2.04] | **+7.36** [+4.01, +11.37] |
| neural_trade as-is, seed 0 | 1 | **+4.93** [+3.84, +6.06] | **+18.99** [+14.69, +24.25] | **+5.00** [+4.00, +6.07] | **+19.82** [+15.43, +25.04] |
| neural_trade as-is, seed 0 | 5 | **+4.41** [+3.17, +5.82] | **+12.36** [+7.94, +17.66] | **+4.55** [+3.35, +5.84] | **+12.85** [+8.34, +18.21] |
| neural_trade as-is, seed 0 | 15 | **+1.52** [+0.19, +3.12] | **+10.71** [+4.77, +17.53] | **+1.76** [+0.40, +3.34] | **+11.42** [+5.12, +18.41] |
| neural_trade retrained daily, seed 0 | 1 | **+8.11** [+6.67, +9.56] | **+24.72** [+19.31, +30.73] | **+8.19** [+6.78, +9.59] | **+25.59** [+19.97, +31.71] |
| neural_trade retrained daily, seed 0 | 5 | **+4.69** [+3.56, +5.86] | **+20.99** [+15.80, +27.18] | **+4.83** [+3.71, +5.99] | **+21.52** [+16.23, +27.75] |
| neural_trade retrained daily, seed 0 | 15 | **+3.50** [+2.04, +5.21] | **+18.00** [+11.22, +24.69] | **+3.73** [+2.30, +5.38] | **+18.77** [+11.70, +25.54] |


### T-C (repo). Tail calibration (target 1% / 5% / 5% / 1%; p < 0.05 = fail)

| Model | h | P(y < q0.01) | P(y < q0.05) | P(y > q0.95) | P(y > q0.99) | Kupiec p (q0.01 / q0.99) | Newey-West p (q0.01 / q0.99) |
|---|---|---|---|---|---|---|---|
| REF: EWMA-Student-t | 1 | 0.0114 | 0.0549 | 0.0542 | 0.0097 | 0.459 / 0.858 | 0.479 / 0.855 |
| REF: EWMA-Student-t | 5 | 0.0093 | 0.0522 | 0.0511 | 0.0104 | 0.713 / 0.845 | 0.722 / 0.865 |
| REF: EWMA-Student-t | 15 | 0.0083 | 0.0525 | 0.0546 | 0.0107 | 0.341 / 0.705 | 0.552 / 0.809 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 1 | 0.0059 | 0.0390 | 0.0383 | 0.0073 | 0.016 / 0.118 | 0.003 / 0.085 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 5 | 0.0069 | 0.0401 | 0.0425 | 0.0076 | 0.077 / 0.175 | 0.062 / 0.162 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 15 | 0.0059 | 0.0363 | 0.0421 | 0.0069 | 0.016 / 0.077 | 0.090 / 0.189 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 1 | 0.0059 | 0.0473 | 0.0542 | 0.0086 | 0.016 / 0.450 | 0.003 / 0.421 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 5 | 0.0066 | 0.0453 | 0.0459 | 0.0083 | 0.047 / 0.341 | 0.035 / 0.367 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 15 | 0.0059 | 0.0453 | 0.0535 | 0.0066 | 0.016 / 0.047 | 0.090 / 0.118 |
| Timestral-28M, ctx 2048 | 1 | 0.0059 | 0.0349 | 0.0428 | 0.0090 | 0.016 / 0.575 | 0.003 / 0.586 |
| Timestral-28M, ctx 2048 | 5 | 0.0086 | 0.0453 | 0.0491 | 0.0117 | 0.450 / 0.359 | 0.450 / 0.412 |
| Timestral-28M, ctx 2048 | 15 | 0.0073 | 0.0401 | 0.0477 | 0.0142 | 0.118 / 0.034 | 0.274 / 0.265 |
| Timestral-209M, ctx 2048 | 1 | 0.0073 | 0.0425 | 0.0487 | 0.0093 | 0.118 / 0.713 | 0.076 / 0.734 |
| Timestral-209M, ctx 2048 | 5 | 0.0090 | 0.0428 | 0.0480 | 0.0111 | 0.575 / 0.575 | 0.591 / 0.620 |
| Timestral-209M, ctx 2048 | 15 | 0.0069 | 0.0428 | 0.0487 | 0.0138 | 0.077 / 0.051 | 0.215 / 0.288 |
| Timestral-28M, h-min return input, 2048 bars § | 1 | 0.0097 | 0.0435 | 0.0463 | 0.0111 | 0.858 / 0.575 | 0.850 / 0.595 |
| Timestral-28M, h-min return input, 2048 bars § | 5 | 0.0114 | 0.0449 | 0.0459 | 0.0117 | 0.459 / 0.359 | 0.524 / 0.438 |
| Timestral-28M, h-min return input, 2048 bars § | 15 | 0.0073 | 0.0497 | 0.0504 | 0.0114 | 0.118 / 0.459 | 0.225 / 0.655 |
| Chronos-2 (Samay family), ctx 512 | 1 | 0.0041 | 0.0377 | 0.0491 | 0.0073 | 0.000 / 0.118 | 0.000 / 0.123 |
| Chronos-2 (Samay family), ctx 512 | 5 | 0.0128 | 0.0560 | 0.0584 | 0.0135 | 0.149 / 0.075 | 0.196 / 0.129 |
| Chronos-2 (Samay family), ctx 512 | 15 | 0.0152 | 0.0601 | 0.0591 | 0.0197 | 0.009 / 0.000 | 0.170 / 0.011 |
| Chronos-2, h-min return input, 512 bars § | 1 | 0.0111 | 0.0470 | 0.0508 | 0.0142 | 0.575 / 0.034 | 0.602 / 0.077 |
| Chronos-2, h-min return input, 512 bars § | 5 | 0.0117 | 0.0487 | 0.0480 | 0.0128 | 0.359 / 0.149 | 0.414 / 0.250 |
| Chronos-2, h-min return input, 512 bars § | 15 | 0.0066 | 0.0432 | 0.0463 | 0.0149 | 0.047 / 0.014 | 0.126 / 0.205 |
| neural_trade as-is, seed 0 | 1 | 0.0114 | 0.0604 | 0.0446 | 0.0083 | 0.459 / 0.341 | 0.573 / 0.448 |
| neural_trade as-is, seed 0 | 5 | 0.0038 | 0.0477 | 0.0794 | 0.0107 | 0.000 / 0.705 | 0.000 / 0.776 |
| neural_trade as-is, seed 0 | 15 | 0.0017 | 0.0466 | 0.0435 | 0.0066 | 0.000 / 0.047 | 0.000 / 0.163 |
| neural_trade retrained daily, seed 0 | 1 | 0.0079 | 0.0421 | 0.0563 | 0.0100 | 0.249 / 0.993 | 0.361 / 0.994 |
| neural_trade retrained daily, seed 0 | 5 | 0.0031 | 0.0383 | 0.0487 | 0.0076 | 0.000 / 0.175 | 0.000 / 0.250 |
| neural_trade retrained daily, seed 0 | 15 | 0.0017 | 0.0366 | 0.0522 | 0.0097 | 0.000 / 0.858 | 0.000 / 0.921 |


### T-D (repo). Non-overlapping trades when |forecast| > cost

| Model | h | 4 bps: n / net bps (t) | 10 bps: n / net (t) | 20 bps: n / net (t) |
|---|---|---|---|---|
| Timestral-5M, ctx 512 | 1 | 22 / -7.8 (-2.3) | 0 | 0 |
| Timestral-5M, ctx 512 | 5 | 83 / -6.7 (-2.3) | 2 / -17.3 (-707.2) | 0 |
| Timestral-5M, ctx 512 | 15 | 169 / -5.1 (-1.9) | 35 / -0.2 (-0.0) | 2 / -6.8 (-0.3) |
| Timestral-28M, ctx 512 | 1 | 19 / -1.4 (-0.4) | 0 | 0 |
| Timestral-28M, ctx 512 | 5 | 51 / -2.8 (-0.8) | 1 | 0 |
| Timestral-28M, ctx 512 | 15 | 123 / -6.9 (-2.1) | 27 / -7.8 (-1.2) | 0 |
| Timestral-28M, ctx 2048 | 1 | 0 | 0 | 0 |
| Timestral-28M, ctx 2048 | 5 | 11 / -3.0 (-0.6) | 0 | 0 |
| Timestral-28M, ctx 2048 | 15 | 46 / -6.6 (-1.7) | 5 / -19.8 (-1.1) | 0 |
| Timestral-209M, ctx 512 | 1 | 8 / -0.5 (-0.1) | 1 | 0 |
| Timestral-209M, ctx 512 | 5 | 51 / -10.8 (-3.3) | 1 | 0 |
| Timestral-209M, ctx 512 | 15 | 132 / -3.7 (-1.2) | 27 / -22.4 (-2.3) | 0 |
| Timestral-209M, ctx 2048 | 1 | 0 | 0 | 0 |
| Timestral-209M, ctx 2048 | 5 | 0 | 0 | 0 |
| Timestral-209M, ctx 2048 | 15 | 16 / +11.7 (+0.8) | 0 | 0 |
| Chronos-2 (Samay family), ctx 512 | 1 | 149 / -3.7 (-3.4) | 9 / -13.3 (-1.6) | 2 / -39.8 (-1.5) |
| Chronos-2 (Samay family), ctx 512 | 5 | 220 / -5.5 (-4.3) | 21 / -13.5 (-2.7) | 3 / -44.0 (-10.8) |
| Chronos-2 (Samay family), ctx 512 | 15 | 268 / -7.1 (-3.7) | 69 / -12.0 (-3.2) | 10 / -22.3 (-1.6) |
| Chronos-2 (Samay family), ctx 2048 | 1 | 295 / -5.1 (-7.7) | 17 / -11.9 (-3.1) | 3 / -26.5 (-3.5) |
| Chronos-2 (Samay family), ctx 2048 | 5 | 251 / -7.2 (-5.6) | 27 / -13.9 (-2.6) | 4 / -38.5 (-3.5) |
| Chronos-2 (Samay family), ctx 2048 | 15 | 215 / -4.8 (-2.1) | 44 / -13.4 (-2.2) | 12 / -27.0 (-2.5) |
| Chronos-Bolt-S (Samay family), ctx 512 † | 1 | 394 / -4.2 (-8.0) | 22 / -5.6 (-1.6) | 1 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 5 | 490 / -5.2 (-6.5) | 60 / -14.3 (-5.7) | 3 / -21.1 (-1.8) |
| Chronos-Bolt-S (Samay family), ctx 512 † | 15 | 420 / -4.3 (-3.2) | 172 / -9.5 (-3.9) | 19 / -13.7 (-1.7) |
| TimesFM-2.5 via Samay, ctx 512 † | 1 | 31 / -0.4 (-0.2) | 0 | 0 |
| TimesFM-2.5 via Samay, ctx 512 † | 5 | 228 / -4.1 (-2.9) | 21 / -13.5 (-3.1) | 1 |
| TimesFM-2.5 via Samay, ctx 512 † | 15 | 301 / -6.1 (-3.5) | 92 / -11.6 (-3.3) | 22 / -28.4 (-5.3) |
| TimesFM-2.5 via Samay, ctx 2048 † | 1 | 19 / -5.6 (-1.3) | 0 | 0 |
| TimesFM-2.5 via Samay, ctx 2048 † | 5 | 136 / -5.5 (-3.1) | 8 / -25.0 (-3.3) | 0 |
| TimesFM-2.5 via Samay, ctx 2048 † | 15 | 192 / -2.3 (-0.9) | 34 / -7.2 (-1.2) | 2 / +0.1 (+0.0) |
| neural_trade as-is, seed 0 | 1 | 0 | 0 | 0 |
| neural_trade as-is, seed 0 | 5 | 138 / -4.4 (-3.2) | 0 | 0 |
| neural_trade as-is, seed 0 | 15 | 254 / -1.6 (-0.8) | 0 | 0 |
| neural_trade as-is, seed 1 | 1 | 0 | 0 | 0 |
| neural_trade as-is, seed 1 | 5 | 311 / -4.6 (-4.9) | 0 | 0 |
| neural_trade as-is, seed 1 | 15 | 47 / +8.4 (+1.4) | 0 | 0 |
| neural_trade as-is, seed 2 | 1 | 170 / -3.5 (-4.2) | 0 | 0 |
| neural_trade as-is, seed 2 | 5 | 0 | 0 | 0 |
| neural_trade as-is, seed 2 | 15 | 90 / -3.8 (-1.6) | 0 | 0 |
| neural_trade retrained daily, seed 0 | 1 | 81 / -3.9 (-6.1) | 0 | 0 |
| neural_trade retrained daily, seed 0 | 5 | 196 / -4.5 (-4.5) | 3 / -12.2 (-3.4) | 0 |
| neural_trade retrained daily, seed 0 | 15 | 187 / -4.5 (-1.9) | 10 / -19.2 (-2.1) | 0 |
| neural_trade retrained daily, seed 1 | 1 | 7 / -0.6 (-0.3) | 0 | 0 |
| neural_trade retrained daily, seed 1 | 5 | 324 / -4.0 (-5.0) | 1 | 0 |
| neural_trade retrained daily, seed 1 | 15 | 194 / -1.1 (-0.5) | 8 / -14.8 (-2.0) | 0 |
| neural_trade retrained daily, seed 2 | 1 | 94 / -4.7 (-3.5) | 0 | 0 |
| neural_trade retrained daily, seed 2 | 5 | 169 / -3.7 (-3.4) | 0 | 0 |
| neural_trade retrained daily, seed 2 | 15 | 176 / -2.6 (-1.7) | 0 | 0 |
| Chronos-2, ctx 512, log-return input ‡ | 1 | 4 / -15.8 (-4.5) | 2 / -17.1 (-46.4) | 0 |
| Chronos-2, ctx 512, log-return input ‡ | 5 | 60 / -3.0 (-0.9) | 4 / -25.5 (-2.3) | 2 / -18.5 (-2.2) |
| Chronos-2, ctx 512, log-return input ‡ | 15 | 262 / -5.9 (-3.1) | 46 / -14.2 (-2.4) | 6 / -42.1 (-2.4) |
| Chronos-2, ctx 2048, log-return input ‡ | 1 | 0 | 0 | 0 |
| Chronos-2, ctx 2048, log-return input ‡ | 5 | 1 | 0 | 0 |
| Chronos-2, ctx 2048, log-return input ‡ | 15 | 80 / -10.1 (-2.5) | 2 / -18.6 (-0.4) | 1 |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 1 | 18 / -5.9 (-1.3) | 2 / -24.8 (-3.4) | 0 |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 5 | 246 / -4.3 (-3.3) | 25 / -9.8 (-2.4) | 4 / -24.6 (-2.8) |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 15 | 401 / -4.7 (-3.3) | 158 / -15.7 (-5.7) | 43 / -18.0 (-3.4) |
| Timestral-28M, h-min return input, 512 bars § | 1 | 18 / -3.2 (-0.8) | 2 / -23.9 (-2.9) | 0 |
| Timestral-28M, h-min return input, 512 bars § | 5 | 60 / -8.7 (-2.5) | 3 / -20.5 (-1.5) | 0 |
| Timestral-28M, h-min return input, 512 bars § | 15 | 154 / -7.9 (-2.7) | 21 / -6.9 (-0.7) | 1 |
| Timestral-28M, h-min return input, 2048 bars § | 1 | 7 / -3.5 (-0.4) | 0 | 0 |
| Timestral-28M, h-min return input, 2048 bars § | 5 | 26 / -6.7 (-1.3) | 1 | 0 |
| Timestral-28M, h-min return input, 2048 bars § | 15 | 49 / -7.3 (-1.5) | 2 / -50.9 (-1.0) | 0 |
| Chronos-2, h-min return input, 512 bars § | 1 | 4 / -15.8 (-4.5) | 2 / -17.1 (-46.4) | 0 |
| Chronos-2, h-min return input, 512 bars § | 5 | 5 / -24.7 (-3.4) | 0 | 0 |
| Chronos-2, h-min return input, 512 bars § | 15 | 0 | 0 | 0 |


### T-E (repo). Volatility role: mean 1-min high-low range over next h min

| Model | h | Mean abs log error | Spearman | QLIKE | vs EWMA: QLIKE %, [CI] | vs HAR (smeared): QLIKE %, [CI] | vs HAR (plain, log-OLS): log-error %, [CI] |
|---|---|---|---|---|---|---|---|
| timestral-209M_ctx2048 | 1 | 0.4911 | 0.708 | 0.6800 | **-5.44** [-8.54, -2.61] | **-28.66** [-35.89, -22.21] | **-6.68** [-9.25, -4.12] |
| timestral-28M_ctx2048 | 1 | 0.4911 | 0.706 | 0.6823 | **-5.12** [-8.07, -2.70] | **-28.42** [-34.65, -21.75] | **-6.68** [-9.37, -4.28] |
| chronos2-x_ctx2048 | 1 | 0.4928 | 0.706 | 0.6841 | **-4.88** [-8.04, -2.24] | **-28.24** [-34.71, -22.14] | **-6.35** [-9.03, -3.97] |
| timestral-5M_ctx2048 | 1 | 0.4933 | 0.708 | 0.6725 | **-6.49** [-9.85, -3.57] | **-29.45** [-35.57, -23.46] | **-6.26** [-8.82, -3.55] |
| ewma_0.94 | 1 | 0.5003 | 0.691 | 0.7191 | – | – | – |
| har_ols_train | 1 | 0.5262 | 0.702 | 1.3706 | – | – | – |
| har_ols_train_smeared | 1 | 0.7329 | 0.702 | 0.9532 | – | – | – |
| har_ols_train | 5 | 0.2576 | 0.824 | 0.2641 | – | – | – |
| timestral-28M_ctx2048 | 5 | 0.2600 | 0.825 | 0.2300 | **-6.86** [-10.29, -3.32] | -0.18 [-2.30, +1.87] | +0.96 [-0.54, +2.33] |
| timestral-209M_ctx2048 | 5 | 0.2602 | 0.825 | 0.2300 | **-6.86** [-10.81, -3.24] | -0.19 [-2.65, +2.26] | +1.00 [-0.58, +2.37] |
| timestral-5M_ctx2048 | 5 | 0.2614 | 0.825 | 0.2274 | **-7.90** [-12.35, -3.73] | -1.30 [-4.17, +1.26] | +1.49 [-0.19, +3.11] |
| chronos2-x_ctx2048 | 5 | 0.2622 | 0.822 | 0.2339 | **-5.26** [-9.60, -1.30] | +1.53 [-1.39, +4.47] | +1.78 [-0.08, +3.38] |
| ewma_0.94 | 5 | 0.2659 | 0.818 | 0.2469 | – | – | – |
| har_ols_train_smeared | 5 | 0.2711 | 0.824 | 0.2304 | – | – | – |
| har_ols_train | 15 | 0.2093 | 0.859 | 0.1812 | – | – | – |
| timestral-209M_ctx2048 | 15 | 0.2128 | 0.858 | 0.1660 | **-9.31** [-15.30, -4.04] | +1.73 [-1.78, +5.27] | +1.66 [-0.24, +3.63] |
| timestral-28M_ctx2048 | 15 | 0.2134 | 0.859 | 0.1656 | **-9.51** [-14.97, -4.59] | +1.51 [-1.78, +5.15] | +1.95 [-0.24, +4.01] |
| timestral-5M_ctx2048 | 15 | 0.2150 | 0.858 | 0.1632 | **-10.82** [-18.58, -3.99] | +0.04 [-4.47, +4.60] | **+2.74** [+0.38, +5.02] |
| chronos2-x_ctx2048 | 15 | 0.2165 | 0.856 | 0.1690 | **-7.64** [-14.15, -1.97] | +3.61 [-1.72, +8.89] | **+3.44** [+0.76, +6.02] |
| ewma_0.94 | 15 | 0.2173 | 0.853 | 0.1830 | – | – | – |
| har_ols_train_smeared | 15 | 0.2182 | 0.859 | 0.1631 | – | – | – |


### T-F (repo). Feature role: walk-forward logistic regression

| Forecast features from | h | AUC base | AUC model-only | AUC base + model | ΔAUC [95% CI] |
|---|---|---|---|---|---|
| Timestral-209M, ctx 2048 | 1 | 0.523 | 0.505 | 0.514 | -0.0084 [-0.0220, +0.0041] |
| Timestral-209M, ctx 2048 | 5 | 0.511 | 0.508 | 0.500 | -0.0106 [-0.0290, +0.0085] |
| Timestral-209M, ctx 2048 | 15 | 0.516 | 0.519 | 0.507 | -0.0089 [-0.0448, +0.0227] |
| Chronos-2 (Samay family), ctx 512 | 1 | 0.523 | 0.491 | 0.512 | -0.0106 [-0.0239, +0.0025] |
| Chronos-2 (Samay family), ctx 512 | 5 | 0.511 | 0.525 | 0.516 | +0.0053 [-0.0150, +0.0257] |
| Chronos-2 (Samay family), ctx 512 | 15 | 0.516 | 0.536 | 0.528 | +0.0117 [-0.0151, +0.0386] |


Window facts (repo): `{"full": {"n_origins": 2895, "stride_min": 3, "first": "2025-11-04 06:30:00+00:00", "last": "2025-11-10 07:12:00+00:00", "mean_abs_ret_bps": {"1": 4.967, "5": 10.984, "15": 18.884}, "up_frac": {"1": 0.5036, "5": 0.5033, "15": 0.496}, "window_return_pct": 1.51, "r1_std_bps": 7.179}, "n483_step18": {"n_origins": 483, "stride_min": 18, "first": "2025-11-04 06:30:00+00:00", "last": "2025-11-10 07:06:00+00:00", "mean_abs_ret_bps": {"1": 4.887, "5": 10.865, "15": 19.403}, "up_frac": {"1": 0.4969, "5": 0.5217, "15": 0.4928}, "window_return_pct": 1.36, "r1_std_bps": 7.18}}`; baseline fit: `{"1": {"lambda": 0.94, "nu": 6.0, "pre_pinball": 1.391, "n_pre": 11095, "var_scale_c": 1.2375, "fhs_z": [-2.85336, -1.65396, -1.1983, -0.95503, -0.7589, -0.58678, -0.44568, -0.32132, -0.19599, -0.07155, 0.0, 0.0713, 0.19581, 0.3256, 0.45161, 0.58296, 0.74223, 0.92782, 1.181, 1.59708, 2.70496], "pre_hit_q01": 0.014, "pre_hit_q99": 0.0123}, "5": {"lambda": 0.97, "nu": 6.0, "pre_pinball": 3.1945, "n_pre": 11095, "var_scale_c": 1.1947, "fhs_z": [-2.84394, -1.65429, -1.20532, -0.95424, -0.75176, -0.60006, -0.46741, -0.33686, -0.22421, -0.1216, -0.00596, 0.09346, 0.19377, 0.31829, 0.43614, 0.57478, 0.7169, 0.92139, 1.18785, 1.64354, 2.77511], "pre_hit_q01": 0.0139, "pre_hit_q99": 0.0132}, "15": {"lambda": 0.97, "nu": 6.0, "pre_pinball": 5.4971, "n_pre": 11095, "var_scale_c": 1.2511, "fhs_z": [-2.82251, -1.64331, -1.22912, -0.96547, -0.77183, -0.60794, -0.47045, -0.35435, -0.24273, -0.1178, -0.01128, 0.10474, 0.2163, 0.3288, 0.45004, 0.58807, 0.73403, 0.92692, 1.17758, 1.59151, 2.96778], "pre_hit_q01": 0.0137, "pre_hit_q99": 0.0147}}`


### T-A (post). Point and direction — Post-release clean set (6,715 origins, 2026-10-16 → 10-29)

| Model | MAE skill vs RW (1 / 5 / 15 min) | MCC of sign (1 / 5 / 15) | AUC of P(up) (1 / 5 / 15) | Brier skill ×1000 (1 / 5 / 15) |
|---|---|---|---|---|
| Timestral-5M, ctx 512 | -1.53% / -0.50% / -0.42% | -0.006 / 0.012 / 0.043 | 0.493 / 0.510 / **0.524** | -16.8 / -8.4 / -8.6 |
| Timestral-28M, ctx 512 | -0.98% / -0.20% / -0.24% | -0.010 / 0.026 / 0.057 | 0.491 / 0.515 / **0.528** | -10.5 / -4.0 / -5.4 |
| Timestral-28M, ctx 2048 | -0.64% / -0.15% / -0.06% | 0.005 / 0.004 / 0.020 | 0.502 / 0.504 / 0.515 | -5.6 / -3.2 / -3.3 |
| Timestral-209M, ctx 512 | -0.75% / -0.25% / -0.21% | 0.007 / 0.023 / 0.052 | 0.500 / 0.515 / **0.531** | -6.8 / -3.8 / -4.8 |
| Timestral-209M, ctx 2048 | -0.35% / -0.12% / -0.11% | 0.014 / 0.011 / 0.016 | 0.513 / 0.500 / 0.509 | -1.3 / -3.0 / -3.8 |
| Chronos-2 (Samay family), ctx 512 | -6.17% / -2.52% / -3.67% | -0.023 / -0.009 / 0.002 | 0.493 / 0.499 / 0.497 | -57.3 / -32.3 / -39.6 |
| Chronos-2 (Samay family), ctx 2048 | -7.86% / -2.79% / -2.32% | -0.004 / -0.011 / -0.007 | 0.499 / 0.485 / 0.485 | -63.7 / -41.0 / -31.4 |
| Chronos-Bolt-S (Samay family), ctx 512 † | -12.35% / -5.11% / -4.82% | 0.004 / 0.012 / 0.027 | 0.503 / 0.505 / 0.521 | -93.4 / -55.5 / -52.9 |
| TimesFM-2.5 via Samay, ctx 512 † | -3.07% / -2.01% / -3.35% | -0.009 / 0.026 / 0.052 | 0.491 / 0.511 / **0.534** | -24.2 / -22.5 / -24.8 |
| neural_trade as-is, seed 0 | -9.64% / -2.08% / -1.65% | -0.004 / -0.000 / -0.019 | 0.504 (head 0.490) / 0.503 (head 0.505) / 0.487 (head 0.506) | -87.2 / -14.2 / -14.5 |
| neural_trade as-is, seed 1 | -3.90% / -0.58% / -0.77% | -0.007 / -0.005 / 0.003 | 0.495 (head 0.507) / 0.496 (head 0.491) / 0.509 (head 0.511) | -29.3 / -5.2 / -7.9 |
| neural_trade as-is, seed 2 | -3.99% / -0.84% / -2.08% | 0.001 / -0.020 / -0.011 | 0.502 (head 0.494) / 0.499 (head 0.513) / 0.489 (head 0.498) | -24.1 / -6.8 / -18.0 |
| neural_trade retrained daily, seed 0 | -5.41% / -1.85% / -0.64% | 0.006 / -0.010 / 0.011 | 0.503 (head 0.504) / 0.491 (head 0.499) / 0.510 (head 0.500) | -55.4 / -22.3 / -8.6 |
| neural_trade retrained daily, seed 1 | -2.56% / -1.70% / -0.73% | -0.007 / 0.015 / -0.035 | 0.496 (head 0.501) / 0.507 (head 0.504) / 0.493 (head 0.517) | -27.2 / -15.5 / -9.6 |
| neural_trade retrained daily, seed 2 | -4.02% / -0.86% / -1.22% | 0.028 / 0.008 / -0.011 | 0.513 (head 0.508) / 0.508 (head 0.502) / 0.485 (head 0.491) | -41.2 / -9.8 / -19.0 |
| Chronos-2, ctx 512, log-return input ‡ | -0.24% / -0.79% / -1.71% | 0.026 / -0.011 / 0.002 | 0.511 / – / – | -2.3 / – / – |
| Chronos-2, ctx 2048, log-return input ‡ | -0.13% / -0.29% / -0.63% | 0.010 / -0.019 / -0.038 | 0.501 / – / – | -1.9 / – / – |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | -1.41% / -3.08% / -7.86% | 0.018 / -0.016 / -0.053 | 0.512 / – / – | -13.8 / – / – |
| Timestral-28M, h-min return input, 512 bars § | -0.69% / -0.53% / -0.12% | 0.016 / -0.014 / 0.025 | 0.500 / 0.487 / 0.518 | -8.3 / -7.9 / -2.0 |
| Timestral-28M, h-min return input, 2048 bars § | -0.19% / -0.15% / +0.15% | 0.004 / 0.002 / 0.036 | 0.506 / 0.501 / **0.525** | -2.3 / -2.2 / +0.6 |
| Chronos-2, h-min return input, 512 bars § | -0.24% / -0.24% / -0.07% | 0.026 / -0.015 / -0.007 | 0.511 / **0.485** / 0.496 | -2.3 / -4.7 / -3.3 |
| EWMA-Gaussian (λ 0.94) | +0.00% / +0.00% / +0.00% | – / – / – | 0.500 / 0.500 / 0.500 | -0.6 / -0.8 / -1.5 |
| Trailing 1-day empirical quantiles | -0.03% / -0.13% / -0.43% | -0.009 / -0.010 / -0.030 | 0.493 / 0.490 / 0.479 | -1.4 / -2.9 / -8.7 |
| Logistic regression on past returns | -0.00% / +0.00% / -0.00% | 0.011 / 0.025 / 0.004 | 0.500 / **0.516** / 0.512 | – / – / – |
| 15/5/1-min reversal sign | -0.00% / +0.00% / +0.00% | -0.037 / 0.017 / 0.090 | – / – / – | – / – / – |


### T-B (post). Distribution and variance vs EWMA-Student-t — Post-release clean set (6,715 origins, 2026-10-16 → 10-29)

| Model | h | Pinball (9 deciles) vs REF, % [95% CI] | QLIKE vs REF, % [95% CI] | Cover 80% | Cover 98% | Spearman(width, \|y\|) |
|---|---|---|---|---|---|---|
| REF: EWMA-Student-t (λ 0.9, ν 6) | 1 | 0 (reference) | 0 (reference) | 0.776 | 0.968 | 0.395 |
| REF: EWMA-Student-t (λ 0.97, ν 6) | 5 | 0 (reference) | 0 (reference) | 0.797 | 0.971 | 0.353 |
| REF: EWMA-Student-t (λ 0.97, ν 6) | 15 | 0 (reference) | 0 (reference) | 0.797 | 0.968 | 0.317 |
| Timestral-5M, ctx 512 | 1 | **+0.97** [+0.52, +1.35] | **-5.28** [-9.29, -2.29] | 0.814 | 0.980 | 0.401 |
| Timestral-5M, ctx 512 | 5 | **+0.47** [+0.07, +0.90] | **-2.55** [-6.09, -0.18] | 0.809 | 0.974 | 0.354 |
| Timestral-5M, ctx 512 | 15 | **+0.98** [+0.23, +1.90] | -1.81 [-5.87, +1.79] | 0.813 | 0.973 | 0.298 |
| Timestral-28M, ctx 512 | 1 | **+0.62** [+0.31, +0.94] | **-5.10** [-8.50, -2.52] | 0.811 | 0.982 | 0.403 |
| Timestral-28M, ctx 512 | 5 | +0.13 [-0.21, +0.52] | **-2.70** [-5.93, -0.55] | 0.810 | 0.977 | 0.353 |
| Timestral-28M, ctx 512 | 15 | +0.45 [-0.16, +1.16] | -2.27 [-6.54, +1.25] | 0.814 | 0.972 | 0.303 |
| Timestral-28M, ctx 2048 | 1 | **+0.32** [+0.04, +0.57] | **-5.56** [-9.35, -2.70] | 0.809 | 0.982 | 0.403 |
| Timestral-28M, ctx 2048 | 5 | -0.01 [-0.30, +0.27] | **-3.41** [-6.86, -0.89] | 0.807 | 0.977 | 0.360 |
| Timestral-28M, ctx 2048 | 15 | +0.05 [-0.43, +0.55] | **-4.49** [-9.44, -0.71] | 0.814 | 0.973 | 0.318 |
| Timestral-209M, ctx 512 | 1 | **+0.54** [+0.21, +0.89] | **-4.45** [-7.18, -2.52] | 0.797 | 0.984 | 0.399 |
| Timestral-209M, ctx 512 | 5 | +0.13 [-0.19, +0.46] | **-2.98** [-6.08, -0.84] | 0.804 | 0.978 | 0.355 |
| Timestral-209M, ctx 512 | 15 | +0.41 [-0.20, +1.09] | -2.87 [-6.79, +0.43] | 0.813 | 0.973 | 0.308 |
| Timestral-209M, ctx 2048 | 1 | +0.05 [-0.20, +0.29] | **-5.39** [-8.70, -2.90] | 0.789 | 0.982 | 0.403 |
| Timestral-209M, ctx 2048 | 5 | -0.08 [-0.31, +0.17] | **-3.98** [-7.61, -1.39] | 0.802 | 0.977 | 0.363 |
| Timestral-209M, ctx 2048 | 15 | +0.06 [-0.28, +0.43] | **-4.12** [-8.55, -0.51] | 0.813 | 0.973 | 0.322 |
| Chronos-2 (Samay family), ctx 512 | 1 | **+5.41** [+4.27, +6.49] | -1.95 [-6.22, +1.00] | 0.792 | 0.984 | 0.387 |
| Chronos-2 (Samay family), ctx 512 | 5 | **+2.62** [+2.00, +3.28] | +3.22 [-0.51, +6.79] | 0.770 | 0.965 | 0.349 |
| Chronos-2 (Samay family), ctx 512 | 15 | **+3.70** [+2.76, +4.79] | **+5.75** [+1.24, +10.69] | 0.768 | 0.958 | 0.300 |
| Chronos-2 (Samay family), ctx 2048 | 1 | **+6.80** [+5.68, +7.88] | -3.27 [-9.10, +0.67] | 0.813 | 0.982 | 0.370 |
| Chronos-2 (Samay family), ctx 2048 | 5 | **+2.60** [+1.95, +3.19] | +2.78 [-0.24, +5.37] | 0.754 | 0.959 | 0.355 |
| Chronos-2 (Samay family), ctx 2048 | 15 | **+2.50** [+1.69, +3.24] | **+6.16** [+1.67, +10.35] | 0.750 | 0.952 | 0.299 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 1 | **+11.31** [+9.74, +12.62] | +0.99 [-3.13, +3.96] | 0.824 | 0.824 | 0.364 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 5 | **+4.67** [+3.76, +5.72] | **+16.89** [+12.04, +21.76] | 0.786 | 0.786 | 0.329 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 15 | **+4.75** [+3.39, +6.42] | **+28.45** [+18.62, +40.86] | 0.763 | 0.763 | 0.285 |
| TimesFM-2.5 via Samay, ctx 512 † | 1 | **+4.07** [+3.46, +4.70] | -2.05 [-6.08, +0.94] | 0.875 | 0.875 | 0.382 |
| TimesFM-2.5 via Samay, ctx 512 † | 5 | **+2.31** [+1.54, +3.17] | **+10.76** [+7.61, +14.05] | 0.814 | 0.814 | 0.352 |
| TimesFM-2.5 via Samay, ctx 512 † | 15 | **+3.87** [+2.24, +5.63] | **+30.75** [+17.87, +46.84] | 0.785 | 0.785 | 0.289 |
| neural_trade as-is, seed 0 | 1 | **+10.71** [+9.62, +11.89] | **+10.30** [+1.92, +17.56] | 0.807 | 0.985 | 0.048 |
| neural_trade as-is, seed 0 | 5 | **+5.89** [+5.05, +6.74] | **+17.38** [+9.30, +23.94] | 0.880 | 0.993 | 0.084 |
| neural_trade as-is, seed 0 | 15 | **+6.92** [+5.86, +7.98] | **+17.15** [+1.98, +28.30] | 0.910 | 0.996 | 0.053 |
| neural_trade as-is, seed 1 | 1 | **+6.04** [+5.30, +6.91] | **+10.69** [+2.02, +17.55] | 0.845 | 0.987 | 0.073 |
| neural_trade as-is, seed 1 | 5 | **+4.66** [+3.96, +5.39] | **+16.84** [+8.92, +23.72] | 0.876 | 0.992 | 0.099 |
| neural_trade as-is, seed 1 | 15 | **+5.68** [+4.65, +6.64] | **+15.29** [+0.27, +26.42] | 0.904 | 0.995 | 0.051 |
| neural_trade as-is, seed 2 | 1 | **+6.49** [+5.69, +7.34] | **+11.72** [+2.73, +18.28] | 0.864 | 0.991 | 0.102 |
| neural_trade as-is, seed 2 | 5 | **+4.36** [+3.64, +5.06] | **+14.52** [+6.75, +20.84] | 0.868 | 0.992 | 0.105 |
| neural_trade as-is, seed 2 | 15 | **+5.53** [+4.55, +6.46] | +11.69 [-3.16, +22.73] | 0.889 | 0.994 | 0.086 |
| neural_trade retrained daily, seed 0 | 1 | **+7.65** [+6.73, +8.57] | **+14.21** [+6.25, +22.26] | 0.781 | 0.978 | 0.015 |
| neural_trade retrained daily, seed 0 | 5 | **+4.72** [+3.91, +5.62] | **+11.86** [+3.91, +18.05] | 0.796 | 0.984 | 0.014 |
| neural_trade retrained daily, seed 0 | 15 | **+3.58** [+2.58, +4.62] | +7.54 [-6.78, +18.72] | 0.795 | 0.980 | -0.015 |
| neural_trade retrained daily, seed 1 | 1 | **+5.65** [+4.68, +6.68] | **+20.08** [+10.99, +29.36] | 0.759 | 0.969 | -0.050 |
| neural_trade retrained daily, seed 1 | 5 | **+4.92** [+3.98, +5.88] | **+16.51** [+7.71, +25.10] | 0.781 | 0.976 | -0.059 |
| neural_trade retrained daily, seed 1 | 15 | **+4.44** [+3.31, +5.54] | +14.33 [-1.05, +26.92] | 0.790 | 0.973 | -0.101 |
| neural_trade retrained daily, seed 2 | 1 | **+6.83** [+5.83, +7.94] | **+18.06** [+9.52, +27.54] | 0.773 | 0.973 | -0.011 |
| neural_trade retrained daily, seed 2 | 5 | **+4.07** [+3.25, +4.99] | **+15.75** [+7.20, +24.70] | 0.779 | 0.977 | -0.018 |
| neural_trade retrained daily, seed 2 | 15 | **+4.54** [+3.46, +5.67] | **+16.72** [+0.42, +30.72] | 0.761 | 0.967 | -0.081 |
| Chronos-2, ctx 512, log-return input ‡ | 1 | **+0.57** [+0.30, +0.84] | -0.13 [-2.65, +1.88] | 0.781 | 0.976 | 0.380 |
| Chronos-2, ctx 2048, log-return input ‡ | 1 | **+0.29** [+0.07, +0.50] | +0.68 [-1.05, +2.96] | 0.780 | 0.974 | 0.388 |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 1 | **+1.79** [+1.17, +2.35] | **+28.65** [+21.96, +35.80] | 0.759 | 0.759 | 0.368 |
| Timestral-28M, h-min return input, 512 bars § | 1 | **+0.79** [+0.45, +1.10] | **-3.48** [-7.05, -0.92] | 0.792 | 0.978 | 0.388 |
| Timestral-28M, h-min return input, 512 bars § | 5 | **+0.97** [+0.61, +1.40] | -1.05 [-5.63, +2.73] | 0.810 | 0.976 | 0.332 |
| Timestral-28M, h-min return input, 512 bars § | 15 | **+1.29** [+0.75, +1.92] | -0.60 [-9.26, +6.49] | 0.803 | 0.975 | 0.217 |
| Timestral-28M, h-min return input, 2048 bars § | 1 | +0.07 [-0.15, +0.28] | **-4.90** [-8.50, -2.10] | 0.795 | 0.979 | 0.399 |
| Timestral-28M, h-min return input, 2048 bars § | 5 | **+0.32** [+0.07, +0.59] | -2.33 [-6.31, +1.25] | 0.805 | 0.974 | 0.341 |
| Timestral-28M, h-min return input, 2048 bars § | 15 | **+0.48** [+0.03, +0.92] | -1.58 [-8.57, +4.46] | 0.801 | 0.974 | 0.267 |
| Chronos-2, h-min return input, 512 bars § | 1 | **+0.57** [+0.30, +0.84] | -0.13 [-2.65, +1.88] | 0.781 | 0.976 | 0.380 |
| Chronos-2, h-min return input, 512 bars § | 5 | **+0.89** [+0.54, +1.27] | +2.38 [-0.92, +5.39] | 0.797 | 0.974 | 0.317 |
| Chronos-2, h-min return input, 512 bars § | 15 | **+1.18** [+0.69, +1.73] | -0.65 [-8.01, +5.88] | 0.795 | 0.976 | 0.228 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 1 | **+0.65** [+0.48, +0.83] | **-3.24** [-6.25, -0.93] | 0.837 | 0.984 | 0.395 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 5 | **+0.40** [+0.27, +0.55] | -1.58 [-3.57, +0.09] | 0.838 | 0.982 | 0.353 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 15 | **+0.57** [+0.36, +0.79] | **-3.12** [-6.80, -0.16] | 0.846 | 0.979 | 0.317 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 1 | **+0.10** [+0.04, +0.16] | **-2.63** [-4.33, -1.33] | 0.794 | 0.983 | 0.395 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 5 | **+0.05** [+0.02, +0.09] | **-1.46** [-2.89, -0.26] | 0.809 | 0.982 | 0.353 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 15 | +0.08 [-0.01, +0.17] | **-2.49** [-4.86, -0.59] | 0.808 | 0.978 | 0.317 |
| EWMA-Gaussian (λ 0.94) | 1 | **+0.44** [+0.28, +0.61] | **-2.27** [-5.09, -0.43] | 0.814 | 0.960 | 0.396 |
| EWMA-Gaussian (λ 0.94) | 5 | **+0.33** [+0.16, +0.51] | -0.15 [-1.83, +1.06] | 0.813 | 0.958 | 0.355 |
| EWMA-Gaussian (λ 0.94) | 15 | **+0.52** [+0.32, +0.75] | +0.97 [-0.55, +2.44] | 0.811 | 0.953 | 0.320 |
| Trailing 1-day empirical quantiles | 1 | **+2.69** [+2.03, +3.42] | **+12.72** [+4.00, +20.56] | 0.791 | 0.976 | 0.150 |
| Trailing 1-day empirical quantiles | 5 | **+2.68** [+2.01, +3.40] | **+13.84** [+8.12, +20.08] | 0.791 | 0.973 | 0.126 |
| Trailing 1-day empirical quantiles | 15 | **+2.46** [+1.75, +3.26] | +7.90 [-0.99, +16.08] | 0.786 | 0.969 | 0.094 |


### T-B2 (post). Against two stronger cheap baselines (% change of loss; negative = model better)

| Model | h | Pinball (9) vs EWMA-t scale-refit, % | QLIKE vs EWMA-t scale-refit, % | Pinball (9) vs FHS, % | QLIKE vs FHS, % |
|---|---|---|---|---|---|
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 1 | **-0.55** [-0.73, -0.37] | +0.63 [-0.41, +1.99] | – | – |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 5 | **-0.35** [-0.48, -0.24] | +0.12 [-0.35, +0.69] | – | – |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 15 | **-0.48** [-0.70, -0.30] | +0.65 [-0.44, +2.01] | – | – |
| Timestral-28M, ctx 512 | 1 | -0.03 [-0.38, +0.30] | **-1.92** [-3.50, -0.38] | **+0.52** [+0.21, +0.83] | **-2.54** [-4.74, -1.02] |
| Timestral-28M, ctx 512 | 5 | -0.27 [-0.64, +0.13] | -1.13 [-3.02, +0.37] | +0.07 [-0.26, +0.47] | -1.26 [-3.45, +0.25] |
| Timestral-28M, ctx 512 | 15 | -0.11 [-0.74, +0.63] | +0.87 [-1.77, +3.69] | +0.37 [-0.23, +1.08] | +0.22 [-2.65, +3.10] |
| Timestral-28M, ctx 2048 | 1 | **-0.33** [-0.67, -0.04] | **-2.40** [-3.89, -1.30] | +0.21 [-0.06, +0.46] | **-3.01** [-5.30, -1.27] |
| Timestral-28M, ctx 2048 | 5 | **-0.41** [-0.72, -0.12] | **-1.86** [-3.99, -0.11] | -0.07 [-0.34, +0.21] | **-1.98** [-4.42, -0.14] |
| Timestral-28M, ctx 2048 | 15 | -0.51 [-1.07, +0.03] | -1.42 [-3.79, +1.19] | -0.03 [-0.51, +0.47] | -2.05 [-5.00, +0.73] |
| Timestral-209M, ctx 2048 | 1 | **-0.59** [-0.95, -0.28] | **-2.23** [-3.72, -0.88] | -0.05 [-0.30, +0.18] | **-2.84** [-4.90, -1.33] |
| Timestral-209M, ctx 2048 | 5 | **-0.48** [-0.77, -0.20] | **-2.44** [-4.56, -0.65] | -0.13 [-0.36, +0.12] | **-2.56** [-5.08, -0.65] |
| Timestral-209M, ctx 2048 | 15 | **-0.51** [-0.94, -0.06] | -1.04 [-3.94, +1.82] | -0.03 [-0.38, +0.37] | -1.67 [-4.86, +1.33] |
| Timestral-28M, h-min return input, 512 bars § | 1 | +0.13 [-0.28, +0.53] | -0.24 [-2.44, +2.09] | **+0.68** [+0.36, +1.00] | -0.87 [-3.49, +1.09] |
| Timestral-28M, h-min return input, 512 bars § | 5 | **+0.57** [+0.15, +1.00] | +0.54 [-2.86, +3.27] | **+0.92** [+0.56, +1.35] | +0.42 [-3.37, +3.29] |
| Timestral-28M, h-min return input, 512 bars § | 15 | **+0.72** [+0.06, +1.42] | +2.59 [-3.35, +8.44] | **+1.21** [+0.62, +1.87] | +1.93 [-4.98, +8.21] |
| Timestral-28M, h-min return input, 2048 bars § | 1 | **-0.58** [-0.88, -0.28] | **-1.72** [-3.19, -0.32] | -0.03 [-0.24, +0.17] | **-2.34** [-4.59, -0.52] |
| Timestral-28M, h-min return input, 2048 bars § | 5 | -0.08 [-0.39, +0.22] | -0.76 [-3.73, +1.86] | +0.27 [-0.00, +0.54] | -0.88 [-4.08, +1.88] |
| Timestral-28M, h-min return input, 2048 bars § | 15 | -0.09 [-0.63, +0.45] | +1.58 [-3.29, +6.85] | +0.39 [-0.08, +0.87] | +0.93 [-4.80, +6.35] |
| Chronos-2 (Samay family), ctx 512 | 1 | **+4.73** [+3.61, +5.82] | +1.33 [-1.40, +3.78] | **+5.30** [+4.19, +6.38] | +0.69 [-2.63, +2.97] |
| Chronos-2 (Samay family), ctx 512 | 5 | **+2.21** [+1.56, +2.91] | **+4.88** [+1.09, +9.04] | **+2.56** [+1.95, +3.22] | **+4.75** [+0.91, +8.87] |
| Chronos-2 (Samay family), ctx 512 | 15 | **+3.11** [+2.13, +4.24] | **+9.15** [+4.02, +15.12] | **+3.61** [+2.66, +4.72] | **+8.45** [+3.77, +13.73] |
| Chronos-2, ctx 2048, log-return input ‡ | 1 | **-0.36** [-0.68, -0.07] | **+4.05** [+0.53, +8.70] | +0.18 [-0.02, +0.40] | **+3.39** [+0.79, +6.87] |
| Chronos-2, h-min return input, 512 bars § | 1 | -0.08 [-0.42, +0.27] | **+3.21** [+0.50, +7.08] | **+0.47** [+0.21, +0.74] | **+2.56** [+0.27, +5.35] |
| Chronos-2, h-min return input, 512 bars § | 5 | **+0.49** [+0.09, +0.89] | **+4.03** [+1.40, +6.72] | **+0.84** [+0.47, +1.22] | **+3.90** [+1.19, +6.53] |
| Chronos-2, h-min return input, 512 bars § | 15 | **+0.61** [+0.02, +1.18] | +2.55 [-3.25, +7.93] | **+1.09** [+0.59, +1.66] | +1.89 [-4.33, +7.58] |
| neural_trade as-is, seed 0 | 1 | **+9.99** [+8.85, +11.21] | **+13.99** [+7.95, +19.82] | **+10.59** [+9.50, +11.80] | **+13.28** [+6.09, +19.63] |
| neural_trade as-is, seed 0 | 5 | **+5.47** [+4.66, +6.31] | **+19.27** [+12.94, +24.90] | **+5.83** [+5.01, +6.69] | **+19.12** [+12.44, +25.08] |
| neural_trade as-is, seed 0 | 15 | **+6.32** [+5.26, +7.35] | **+20.92** [+8.88, +29.83] | **+6.83** [+5.78, +7.87] | **+20.14** [+6.90, +29.97] |
| neural_trade retrained daily, seed 0 | 1 | **+6.95** [+5.97, +7.93] | **+18.03** [+11.84, +24.56] | **+7.54** [+6.63, +8.46] | **+17.29** [+10.39, +24.45] |
| neural_trade retrained daily, seed 0 | 5 | **+4.30** [+3.47, +5.21] | **+13.66** [+6.90, +19.13] | **+4.67** [+3.87, +5.54] | **+13.52** [+6.46, +19.27] |
| neural_trade retrained daily, seed 0 | 15 | **+3.00** [+1.92, +4.04] | +11.00 [-0.11, +20.80] | **+3.50** [+2.46, +4.58] | +10.28 [-2.18, +20.68] |


### T-C (post). Tail calibration (target 1% / 5% / 5% / 1%; p < 0.05 = fail)

| Model | h | P(y < q0.01) | P(y < q0.05) | P(y > q0.95) | P(y > q0.99) | Kupiec p (q0.01 / q0.99) | Newey-West p (q0.01 / q0.99) |
|---|---|---|---|---|---|---|---|
| REF: EWMA-Student-t | 1 | 0.0159 | 0.0625 | 0.0645 | 0.0156 | 0.000 / 0.000 | 0.000 / 0.000 |
| REF: EWMA-Student-t | 5 | 0.0149 | 0.0561 | 0.0542 | 0.0138 | 0.000 / 0.003 | 0.012 / 0.019 |
| REF: EWMA-Student-t | 15 | 0.0182 | 0.0560 | 0.0512 | 0.0136 | 0.000 / 0.005 | 0.007 / 0.155 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 1 | 0.0086 | 0.0420 | 0.0413 | 0.0070 | 0.251 / 0.009 | 0.217 / 0.004 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 5 | 0.0092 | 0.0413 | 0.0424 | 0.0091 | 0.522 / 0.444 | 0.634 / 0.509 |
| EWMA-t, variance scale refit for QLIKE (pre-test) | 15 | 0.0125 | 0.0429 | 0.0371 | 0.0085 | 0.047 / 0.201 | 0.307 / 0.462 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 1 | 0.0098 | 0.0509 | 0.0527 | 0.0073 | 0.887 / 0.019 | 0.883 / 0.011 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 5 | 0.0094 | 0.0517 | 0.0463 | 0.0083 | 0.607 / 0.159 | 0.708 / 0.210 |
| FHS: EWMA σ × empirical residual quantiles (pre-test) | 15 | 0.0138 | 0.0523 | 0.0423 | 0.0082 | 0.003 / 0.124 | 0.130 / 0.376 |
| Timestral-28M, ctx 2048 | 1 | 0.0088 | 0.0441 | 0.0500 | 0.0095 | 0.307 / 0.697 | 0.280 / 0.693 |
| Timestral-28M, ctx 2048 | 5 | 0.0116 | 0.0496 | 0.0508 | 0.0113 | 0.195 / 0.288 | 0.331 / 0.376 |
| Timestral-28M, ctx 2048 | 15 | 0.0159 | 0.0506 | 0.0462 | 0.0107 | 0.000 / 0.557 | 0.029 / 0.755 |
| Timestral-209M, ctx 2048 | 1 | 0.0091 | 0.0557 | 0.0576 | 0.0088 | 0.444 / 0.307 | 0.418 / 0.301 |
| Timestral-209M, ctx 2048 | 5 | 0.0112 | 0.0496 | 0.0511 | 0.0115 | 0.345 / 0.238 | 0.476 / 0.345 |
| Timestral-209M, ctx 2048 | 15 | 0.0155 | 0.0514 | 0.0505 | 0.0116 | 0.000 / 0.195 | 0.042 / 0.491 |
| Timestral-28M, h-min return input, 2048 bars § | 1 | 0.0112 | 0.0527 | 0.0541 | 0.0095 | 0.345 / 0.697 | 0.359 / 0.696 |
| Timestral-28M, h-min return input, 2048 bars § | 5 | 0.0141 | 0.0500 | 0.0509 | 0.0122 | 0.001 / 0.078 | 0.028 / 0.174 |
| Timestral-28M, h-min return input, 2048 bars § | 15 | 0.0150 | 0.0535 | 0.0487 | 0.0115 | 0.000 / 0.238 | 0.058 / 0.518 |
| Chronos-2 (Samay family), ctx 512 | 1 | 0.0074 | 0.0490 | 0.0561 | 0.0091 | 0.028 / 0.444 | 0.016 / 0.455 |
| Chronos-2 (Samay family), ctx 512 | 5 | 0.0158 | 0.0654 | 0.0649 | 0.0189 | 0.000 / 0.000 | 0.003 / 0.000 |
| Chronos-2 (Samay family), ctx 512 | 15 | 0.0231 | 0.0706 | 0.0624 | 0.0192 | 0.000 / 0.000 | 0.000 / 0.002 |
| Chronos-2, h-min return input, 512 bars § | 1 | 0.0113 | 0.0558 | 0.0575 | 0.0127 | 0.288 / 0.035 | 0.320 / 0.054 |
| Chronos-2, h-min return input, 512 bars § | 5 | 0.0125 | 0.0491 | 0.0520 | 0.0137 | 0.047 / 0.004 | 0.165 / 0.030 |
| Chronos-2, h-min return input, 512 bars § | 15 | 0.0113 | 0.0484 | 0.0466 | 0.0125 | 0.288 / 0.047 | 0.565 / 0.269 |
| neural_trade as-is, seed 0 | 1 | 0.0061 | 0.0341 | 0.0605 | 0.0092 | 0.001 / 0.522 | 0.001 / 0.579 |
| neural_trade as-is, seed 0 | 5 | 0.0036 | 0.0256 | 0.0256 | 0.0034 | 0.000 / 0.000 | 0.000 / 0.000 |
| neural_trade as-is, seed 0 | 15 | 0.0025 | 0.0220 | 0.0156 | 0.0015 | 0.000 / 0.000 | 0.000 / 0.000 |
| neural_trade retrained daily, seed 0 | 1 | 0.0095 | 0.0560 | 0.0558 | 0.0128 | 0.697 / 0.027 | 0.744 / 0.115 |
| neural_trade retrained daily, seed 0 | 5 | 0.0070 | 0.0468 | 0.0527 | 0.0094 | 0.009 / 0.607 | 0.037 / 0.712 |
| neural_trade retrained daily, seed 0 | 15 | 0.0112 | 0.0497 | 0.0557 | 0.0086 | 0.345 / 0.251 | 0.625 / 0.481 |


### T-D (post). Non-overlapping trades when |forecast| > cost

| Model | h | 4 bps: n / net bps (t) | 10 bps: n / net (t) | 20 bps: n / net (t) |
|---|---|---|---|---|
| Timestral-5M, ctx 512 | 1 | 7 / +8.9 (+1.3) | 0 | 0 |
| Timestral-5M, ctx 512 | 5 | 68 / -1.0 (-0.4) | 2 / -16.5 (-0.6) | 0 |
| Timestral-5M, ctx 512 | 15 | 222 / -3.2 (-2.0) | 30 / -14.8 (-3.2) | 1 |
| Timestral-28M, ctx 512 | 1 | 3 / -12.3 (-1.9) | 0 | 0 |
| Timestral-28M, ctx 512 | 5 | 33 / -2.8 (-0.8) | 1 | 0 |
| Timestral-28M, ctx 512 | 15 | 168 / -3.3 (-1.9) | 18 / -8.2 (-1.2) | 0 |
| Timestral-28M, ctx 2048 | 1 | 2 / +2.3 (+0.3) | 0 | 0 |
| Timestral-28M, ctx 2048 | 5 | 23 / +1.7 (+0.4) | 1 | 0 |
| Timestral-28M, ctx 2048 | 15 | 73 / -1.9 (-0.7) | 12 / -7.5 (-0.8) | 1 |
| Timestral-209M, ctx 512 | 1 | 6 / -8.6 (-1.6) | 0 | 0 |
| Timestral-209M, ctx 512 | 5 | 31 / -5.2 (-1.4) | 2 / -18.5 (-232.5) | 0 |
| Timestral-209M, ctx 512 | 15 | 147 / -4.1 (-2.3) | 14 / -8.2 (-1.0) | 2 / -9.3 (-0.2) |
| Timestral-209M, ctx 2048 | 1 | 0 | 0 | 0 |
| Timestral-209M, ctx 2048 | 5 | 11 / -1.3 (-0.2) | 0 | 0 |
| Timestral-209M, ctx 2048 | 15 | 31 / +3.0 (+0.6) | 4 / -2.4 (-0.3) | 0 |
| Chronos-2 (Samay family), ctx 512 | 1 | 143 / -4.2 (-4.3) | 20 / -8.0 (-2.2) | 2 / -19.9 (-1.3) |
| Chronos-2 (Samay family), ctx 512 | 5 | 242 / -4.3 (-4.0) | 21 / -22.6 (-4.3) | 8 / -32.8 (-3.2) |
| Chronos-2 (Samay family), ctx 512 | 15 | 414 / -5.3 (-5.0) | 77 / -15.2 (-4.3) | 12 / -33.2 (-4.0) |
| Chronos-2 (Samay family), ctx 2048 | 1 | 132 / -5.3 (-6.4) | 8 / -3.2 (-0.6) | 2 / -19.9 (-1.3) |
| Chronos-2 (Samay family), ctx 2048 | 5 | 147 / -4.5 (-3.2) | 12 / -21.9 (-3.4) | 2 / -65.9 (-5.7) |
| Chronos-2 (Samay family), ctx 2048 | 15 | 238 / -7.0 (-4.4) | 39 / -17.3 (-3.3) | 7 / -24.7 (-2.7) |
| Chronos-Bolt-S (Samay family), ctx 512 † | 1 | 339 / -4.0 (-8.2) | 22 / -12.3 (-3.1) | 1 |
| Chronos-Bolt-S (Samay family), ctx 512 † | 5 | 487 / -3.8 (-5.7) | 45 / -12.8 (-3.4) | 4 / -13.7 (-3.2) |
| Chronos-Bolt-S (Samay family), ctx 512 † | 15 | 667 / -4.2 (-5.4) | 113 / -12.1 (-4.2) | 18 / -26.4 (-3.0) |
| TimesFM-2.5 via Samay, ctx 512 † | 1 | 29 / -3.3 (-1.6) | 1 | 1 |
| TimesFM-2.5 via Samay, ctx 512 † | 5 | 295 / -4.3 (-4.3) | 35 / -12.0 (-2.6) | 3 / -6.2 (-0.5) |
| TimesFM-2.5 via Samay, ctx 512 † | 15 | 524 / -3.2 (-3.4) | 141 / -10.7 (-5.0) | 31 / -17.5 (-4.0) |
| neural_trade as-is, seed 0 | 1 | 29 / -3.1 (-4.5) | 0 | 0 |
| neural_trade as-is, seed 0 | 5 | 11 / -4.7 (-1.2) | 0 | 0 |
| neural_trade as-is, seed 0 | 15 | 229 / -6.3 (-5.5) | 0 | 0 |
| neural_trade as-is, seed 1 | 1 | 0 | 0 | 0 |
| neural_trade as-is, seed 1 | 5 | 0 | 0 | 0 |
| neural_trade as-is, seed 1 | 15 | 3 / -1.4 (-1.3) | 0 | 0 |
| neural_trade as-is, seed 2 | 1 | 0 | 0 | 0 |
| neural_trade as-is, seed 2 | 5 | 0 | 0 | 0 |
| neural_trade as-is, seed 2 | 15 | 303 / -4.3 (-5.3) | 0 | 0 |
| neural_trade retrained daily, seed 0 | 1 | 1 | 0 | 0 |
| neural_trade retrained daily, seed 0 | 5 | 30 / -3.6 (-3.8) | 0 | 0 |
| neural_trade retrained daily, seed 0 | 15 | 103 / -3.4 (-2.4) | 1 | 0 |
| neural_trade retrained daily, seed 1 | 1 | 0 | 0 | 0 |
| neural_trade retrained daily, seed 1 | 5 | 43 / -0.6 (-0.5) | 0 | 0 |
| neural_trade retrained daily, seed 1 | 15 | 43 / -2.3 (-1.2) | 0 | 0 |
| neural_trade retrained daily, seed 2 | 1 | 1 | 0 | 0 |
| neural_trade retrained daily, seed 2 | 5 | 10 / -7.2 (-3.1) | 0 | 0 |
| neural_trade retrained daily, seed 2 | 15 | 97 / -7.3 (-4.0) | 0 | 0 |
| Chronos-2, ctx 512, log-return input ‡ | 1 | 2 / -11.6 (-1.5) | 0 | 0 |
| Chronos-2, ctx 512, log-return input ‡ | 5 | 11 / -7.8 (-1.3) | 5 / -24.4 (-3.5) | 2 / -44.0 (-4.3) |
| Chronos-2, ctx 512, log-return input ‡ | 15 | 159 / -5.0 (-3.0) | 12 / -15.9 (-1.9) | 4 / -32.9 (-2.9) |
| Chronos-2, ctx 2048, log-return input ‡ | 1 | 0 | 0 | 0 |
| Chronos-2, ctx 2048, log-return input ‡ | 5 | 3 / -9.8 (-0.5) | 1 | 0 |
| Chronos-2, ctx 2048, log-return input ‡ | 15 | 8 / -31.8 (-4.5) | 3 / -33.1 (-3.6) | 1 |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 1 | 22 / -4.4 (-1.3) | 0 | 0 |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 5 | 274 / -4.7 (-4.5) | 28 / -11.8 (-2.1) | 5 / -5.5 (-0.2) |
| TimesFM-2.5 via Samay, ctx 512, log-return input † ‡ | 15 | 674 / -5.4 (-7.0) | 142 / -12.4 (-6.1) | 29 / -25.3 (-4.4) |
| Timestral-28M, h-min return input, 512 bars § | 1 | 7 / -11.5 (-1.8) | 1 | 0 |
| Timestral-28M, h-min return input, 512 bars § | 5 | 24 / -7.1 (-1.7) | 3 / +1.8 (+0.3) | 0 |
| Timestral-28M, h-min return input, 512 bars § | 15 | 22 / -3.5 (-0.8) | 3 / -21.0 (-2.1) | 1 |
| Timestral-28M, h-min return input, 2048 bars § | 1 | 0 | 0 | 0 |
| Timestral-28M, h-min return input, 2048 bars § | 5 | 3 / -1.9 (-0.2) | 1 | 0 |
| Timestral-28M, h-min return input, 2048 bars § | 15 | 7 / +4.6 (+0.7) | 0 | 0 |
| Chronos-2, h-min return input, 512 bars § | 1 | 2 / -11.6 (-1.5) | 0 | 0 |
| Chronos-2, h-min return input, 512 bars § | 5 | 5 / -3.5 (-0.5) | 1 | 1 |
| Chronos-2, h-min return input, 512 bars § | 15 | 2 / -11.8 (-0.7) | 0 | 0 |


### T-E (post). Volatility role: mean 1-min high-low range over next h min

| Model | h | Mean abs log error | Spearman | QLIKE | vs EWMA: QLIKE %, [CI] | vs HAR (smeared): QLIKE %, [CI] | vs HAR (plain, log-OLS): log-error %, [CI] |
|---|---|---|---|---|---|---|---|
| timestral-28M_ctx2048 | 1 | 0.9569 | 0.658 | 1.9616 | **-7.46** [-12.94, -3.84] | **-51.84** [-86.24, -27.88] | **-10.06** [-11.87, -7.94] |
| chronos2-x_ctx2048 | 1 | 0.9582 | 0.659 | 1.9645 | **-7.33** [-12.19, -3.41] | **-51.77** [-84.10, -28.61] | **-9.93** [-11.70, -7.92] |
| timestral-5M_ctx2048 | 1 | 0.9622 | 0.659 | 1.9282 | **-9.04** [-15.28, -4.73] | **-52.66** [-83.40, -29.71] | **-9.56** [-11.39, -7.69] |
| ewma_0.94 | 1 | 0.9737 | 0.635 | 2.1198 | – | – | – |
| har_ols_train | 1 | 1.0639 | 0.646 | 33.3047 | – | – | – |
| har_ols_train_smeared | 1 | 1.3949 | 0.646 | 4.0732 | – | – | – |
| timestral-28M_ctx2048 | 5 | 0.3842 | 0.780 | 0.5435 | **-14.34** [-19.23, -10.24] | **-13.74** [-19.39, -9.05] | **-1.62** [-2.89, -0.26] |
| chronos2-x_ctx2048 | 5 | 0.3856 | 0.779 | 0.5566 | **-12.28** [-17.62, -7.96] | **-11.67** [-17.00, -7.01] | -1.25 [-2.66, +0.31] |
| timestral-5M_ctx2048 | 5 | 0.3880 | 0.778 | 0.5429 | **-14.43** [-19.27, -10.15] | **-13.83** [-19.46, -8.93] | -0.64 [-1.97, +0.77] |
| har_ols_train | 5 | 0.3905 | 0.770 | 0.8655 | – | – | – |
| ewma_0.94 | 5 | 0.3979 | 0.764 | 0.6345 | – | – | – |
| har_ols_train_smeared | 5 | 0.4182 | 0.770 | 0.6301 | – | – | – |
| timestral-28M_ctx2048 | 15 | 0.3023 | 0.803 | 0.4187 | **-17.42** [-24.79, -11.06] | **-4.30** [-8.14, -1.05] | -0.21 [-1.72, +1.42] |
| har_ols_train | 15 | 0.3029 | 0.796 | 0.5204 | – | – | – |
| timestral-5M_ctx2048 | 15 | 0.3056 | 0.802 | 0.4121 | **-18.73** [-27.30, -11.32] | **-5.81** [-9.92, -1.83] | +0.87 [-0.86, +2.77] |
| chronos2-x_ctx2048 | 15 | 0.3061 | 0.800 | 0.4308 | **-15.03** [-22.60, -8.05] | -1.53 [-6.10, +2.67] | +1.05 [-0.73, +3.04] |
| ewma_0.94 | 15 | 0.3132 | 0.798 | 0.5070 | – | – | – |
| har_ols_train_smeared | 15 | 0.3225 | 0.796 | 0.4375 | – | – | – |


### T-F (post). Feature role: walk-forward logistic regression

| Forecast features from | h | AUC base | AUC model-only | AUC base + model | ΔAUC [95% CI] |
|---|---|---|---|---|---|
| Timestral-28M, ctx 512 | 1 | 0.517 | 0.509 | 0.514 | -0.0029 [-0.0128, +0.0069] |
| Timestral-28M, ctx 512 | 5 | 0.521 | 0.512 | 0.515 | -0.0060 [-0.0134, +0.0015] |
| Timestral-28M, ctx 512 | 15 | 0.541 | 0.527 | 0.540 | -0.0004 [-0.0122, +0.0113] |
| Timestral-28M, ctx 2048 | 1 | 0.517 | 0.497 | 0.509 | -0.0082 [-0.0190, +0.0031] |
| Timestral-28M, ctx 2048 | 5 | 0.521 | 0.496 | 0.509 | **-0.0120** [-0.0189, -0.0044] |
| Timestral-28M, ctx 2048 | 15 | 0.541 | 0.508 | 0.535 | -0.0054 [-0.0139, +0.0047] |
| Timestral-209M, ctx 2048 | 1 | 0.517 | 0.512 | 0.517 | +0.0003 [-0.0112, +0.0123] |
| Timestral-209M, ctx 2048 | 5 | 0.521 | 0.497 | 0.515 | -0.0062 [-0.0120, +0.0009] |
| Timestral-209M, ctx 2048 | 15 | 0.541 | 0.508 | 0.530 | **-0.0104** [-0.0177, -0.0022] |
| Chronos-2 (Samay family), ctx 512 | 1 | 0.517 | 0.511 | 0.515 | -0.0016 [-0.0085, +0.0059] |
| Chronos-2 (Samay family), ctx 512 | 5 | 0.521 | 0.496 | 0.516 | -0.0056 [-0.0121, +0.0001] |
| Chronos-2 (Samay family), ctx 512 | 15 | 0.541 | 0.518 | 0.540 | -0.0011 [-0.0073, +0.0051] |
| neural_trade as-is, seed 0 | 1 | 0.517 | 0.506 | 0.514 | -0.0030 [-0.0120, +0.0055] |
| neural_trade as-is, seed 0 | 5 | 0.521 | 0.498 | 0.515 | -0.0060 [-0.0139, +0.0016] |
| neural_trade as-is, seed 0 | 15 | 0.541 | 0.505 | 0.541 | -0.0002 [-0.0072, +0.0062] |


Window facts (post): `{"full": {"n_origins": 6715, "stride_min": 3, "first": "2026-10-16 00:00:00+00:00", "last": "2026-10-29 23:42:00+00:00", "mean_abs_ret_bps": {"1": 2.955, "5": 6.71, "15": 11.66}, "up_frac": {"1": 0.4703, "5": 0.4844, "15": 0.4806}, "window_return_pct": -4.65, "r1_std_bps": 4.504}, "n1120_step18": {"n_origins": 1120, "stride_min": 18, "first": "2026-10-16 00:00:00+00:00", "last": "2026-10-29 23:42:00+00:00", "mean_abs_ret_bps": {"1": 2.951, "5": 6.482, "15": 11.696}, "up_frac": {"1": 0.4777, "5": 0.4973, "15": 0.5036}, "window_return_pct": -4.65, "r1_std_bps": 4.504}, "n2239_step9": {"n_origins": 2239, "stride_min": 9, "first": "2026-10-16 00:00:00+00:00", "last": "2026-10-29 23:42:00+00:00", "mean_abs_ret_bps": {"1": 2.974, "5": 6.656, "15": 11.603}, "up_frac": {"1": 0.4761, "5": 0.5047, "15": 0.4989}, "window_return_pct": -4.65, "r1_std_bps": 4.504}}`; baseline fit: `{"1": {"lambda": 0.9, "nu": 6.0, "pre_pinball": 1.0696, "n_pre": 11975, "var_scale_c": 1.4451, "fhs_z": [-2.96523, -1.73641, -1.24321, -0.94153, -0.73166, -0.54903, -0.40599, -0.26281, -0.12187, -0.00099, 0.0, 0.00794, 0.14776, 0.28312, 0.41821, 0.58002, 0.76598, 0.9768, 1.25586, 1.74247, 3.0395], "pre_hit_q01": 0.0162, "pre_hit_q99": 0.0178}, "5": {"lambda": 0.97, "nu": 6.0, "pre_pinball": 2.4664, "n_pre": 11975, "var_scale_c": 1.2658, "fhs_z": [-2.86069, -1.63953, -1.19486, -0.92863, -0.73573, -0.57695, -0.45005, -0.31951, -0.20336, -0.08334, 0.0, 0.10087, 0.21345, 0.33476, 0.45672, 0.59216, 0.7661, 0.96487, 1.24224, 1.71764, 3.0575], "pre_hit_q01": 0.0144, "pre_hit_q99": 0.017}, "15": {"lambda": 0.97, "nu": 6.0, "pre_pinball": 4.2641, "n_pre": 11975, "var_scale_c": 1.3168, "fhs_z": [-2.84873, -1.65106, -1.18574, -0.94191, -0.73858, -0.57979, -0.43947, -0.31544, -0.20556, -0.09558, 0.00899, 0.12055, 0.23024, 0.35097, 0.47907, 0.61546, 0.77643, 0.97035, 1.24358, 1.73465, 3.04487], "pre_hit_q01": 0.0143, "pre_hit_q99": 0.0175}}`


### T-G. Context, size and framing ablations (all sets)

| Set | Config | h=1 pinball / QLIKE vs REF, % | h=5 pinball / QLIKE | h=15 pinball / QLIKE | ms/series (batched) |
|---|---|---|---|---|---|
| repo/full | official-chronos2_ctx2048_logret | **+0.27** [+0.05, +0.49] / +0.45 [-0.72, +1.64] | MAE skill -0.18% | MAE skill -0.51% | 225.23 |
| repo/full | official-chronos2_ctx2048_price | **+9.38** [+7.35, +11.30] / **+5.19** [+2.10, +7.72] | **+3.17** [+2.11, +4.15] / +0.76 [-1.55, +2.63] | **+2.08** [+0.98, +3.08] / +0.99 [-2.00, +3.65] | 76.87 |
| repo/full | official-chronos2_ctx512_aggret | **+1.02** [+0.59, +1.54] / **+1.69** [+0.50, +2.96] | **+0.86** [+0.43, +1.30] / **+3.39** [+0.91, +6.64] | **+1.38** [+0.79, +2.06] / **+7.29** [+3.77, +11.23] | 155.03 |
| repo/full | official-chronos2_ctx512_logret | **+1.02** [+0.59, +1.54] / **+1.69** [+0.50, +2.96] | MAE skill -1.02% | MAE skill -2.87% | 50.17 |
| repo/full | official-chronos2_ctx512_price | **+4.63** [+3.17, +5.99] / +2.44 [-0.48, +4.88] | **+2.91** [+2.14, +3.80] / +0.61 [-1.22, +2.34] | **+3.17** [+2.02, +4.49] / +1.36 [-1.17, +3.90] | 22.71 |
| repo/full | timestral-123M_ctx2048_price | +0.30 [-0.06, +0.67] / -0.85 [-2.00, +0.37] | +0.11 [-0.18, +0.36] / -1.03 [-2.20, +0.12] | -0.07 [-0.49, +0.38] / -1.68 [-3.52, +0.17] | 82.34 |
| repo/full | timestral-123M_ctx512_price | +0.34 [-0.11, +0.76] / -0.60 [-1.93, +0.82] | **+0.59** [+0.11, +1.07] / -0.89 [-2.17, +0.33] | +0.84 [-0.15, +1.92] / -1.24 [-3.21, +0.64] | 20.95 |
| repo/full | timestral-209M_ctx2048_logret | +0.09 [-0.10, +0.27] / -0.88 [-1.95, +0.22] | MAE skill -0.14% | MAE skill -0.43% | 507.44 |
| repo/full | timestral-209M_ctx2048_price | +0.19 [-0.16, +0.48] / -0.71 [-1.98, +0.53] | +0.00 [-0.30, +0.25] / **-1.29** [-2.70, -0.03] | -0.16 [-0.57, +0.28] / -1.70 [-3.82, +0.36] | 128.43 |
| repo/full | timestral-209M_ctx512_price | **+0.47** [+0.07, +0.88] / -0.19 [-1.76, +1.38] | **+0.53** [+0.08, +0.99] / -1.00 [-2.42, +0.34] | +0.56 [-0.33, +1.52] / -1.41 [-3.43, +0.47] | 64.25 |
| repo/full | timestral-209M_ctx64_price | **+1.41** [+0.88, +2.00] / +0.50 [-1.66, +2.29] | **+1.77** [+1.06, +2.54] / +1.69 [-0.25, +3.41] | **+1.88** [+0.75, +3.13] / **+8.73** [+4.93, +12.80] | 7.74 |
| repo/full | timestral-28M_ctx2048_aggret | +0.20 [-0.08, +0.47] / -0.51 [-1.69, +0.60] | **+0.55** [+0.15, +0.98] / +1.10 [-0.88, +3.23] | +0.45 [-0.03, +1.04] / +1.86 [-1.23, +5.02] | 198.73 |
| repo/full | timestral-28M_ctx2048_logret | +0.20 [-0.08, +0.47] / -0.51 [-1.69, +0.60] | MAE skill -0.12% | MAE skill -1.01% | 21.87 |
| repo/full | timestral-28M_ctx2048_price | **+0.57** [+0.21, +0.92] / -0.68 [-2.26, +0.71] | +0.33 [-0.02, +0.68] / -1.26 [-2.76, +0.02] | +0.21 [-0.41, +0.87] / -1.65 [-4.03, +0.53] | 22.93 |
| repo/full | timestral-28M_ctx2048_price_cov | **+0.67** [+0.04, +1.24] / **-1.75** [-3.09, -0.48] | +0.63 [-0.01, +1.20] / **-1.51** [-3.06, -0.11] | +0.64 [-0.56, +1.64] / -1.60 [-4.38, +0.70] | 238.25 |
| repo/full | timestral-28M_ctx2048_price_nosym | **+0.83** [+0.40, +1.20] / -0.46 [-1.99, +0.88] | +0.36 [-0.01, +0.69] / -1.04 [-2.50, +0.28] | -0.03 [-0.67, +0.56] / -1.50 [-3.95, +0.68] | 10.31 |
| repo/full | timestral-28M_ctx512_aggret | **+1.05** [+0.51, +1.59] / +1.27 [-0.28, +2.86] | **+1.10** [+0.56, +1.70] / **+2.98** [+0.60, +5.86] | **+1.38** [+0.54, +2.34] / **+4.77** [+0.47, +8.72] | 45.55 |
| repo/full | timestral-28M_ctx512_price | **+0.45** [+0.00, +0.89] / -0.56 [-2.25, +0.98] | **+0.67** [+0.22, +1.14] / -0.79 [-2.23, +0.58] | **+1.02** [+0.19, +1.89] / -0.77 [-3.34, +1.59] | 18.36 |
| repo/full | timestral-28M_ctx64_price | **+1.56** [+0.95, +2.15] / -0.66 [-2.80, +1.08] | **+2.52** [+1.67, +3.36] / **+2.46** [+0.50, +4.29] | **+2.51** [+1.24, +3.87] / **+11.25** [+6.75, +16.20] | 1.34 |
| repo/full | timestral-5M_ctx2048_price | **+0.71** [+0.25, +1.15] / +0.50 [-1.07, +2.02] | **+0.53** [+0.12, +0.95] / -0.37 [-1.84, +1.08] | +0.33 [-0.25, +1.00] / -0.79 [-3.17, +1.46] | 33.28 |
| repo/full | timestral-5M_ctx512_price | **+1.08** [+0.53, +1.54] / -0.23 [-1.85, +1.30] | **+1.12** [+0.57, +1.64] / -0.29 [-1.70, +1.11] | **+1.45** [+0.46, +2.56] / +0.49 [-1.50, +2.48] | 1.61 |
| repo/n483_step18 | official-chronos2_ctx2048_aggret_sub6 | +0.34 [-0.22, +0.85] / +1.42 [-1.24, +4.22] | +0.58 [-0.20, +1.52] / +1.59 [-3.00, +7.34] | +0.61 [-0.28, +1.52] / +4.50 [-3.34, +12.42] | 672.57 |
| repo/n483_step18 | timestral-209M_ctx2048_price_cov_sub6 | +0.18 [-1.68, +1.98] / **-5.62** [-10.83, -1.06] | +0.75 [-1.03, +2.44] / -1.45 [-5.13, +1.63] | +1.11 [-1.30, +3.17] / -1.33 [-6.68, +2.24] | 998.25 |
| repo/n483_step18 | timestral-209M_ctx8192_price_sub6 | -0.04 [-1.33, +1.05] / **-4.38** [-8.44, -1.26] | -0.11 [-0.71, +0.51] / **-2.36** [-4.78, -0.11] | -0.32 [-0.95, +0.27] / -2.66 [-6.00, +0.05] | 602.41 |
| repo/n483_step18 | timestral-28M_ctx32768_price_sub6 | **+4.60** [+3.05, +6.33] / +5.05 [-2.51, +11.60] | **+1.31** [+0.08, +2.59] / +0.08 [-3.40, +3.28] | +0.13 [-0.71, +0.99] / -1.43 [-5.86, +2.27] | 516.88 |
| repo/n483_step18 | timestral-28M_ctx8192_price_sub6 | **+1.32** [+0.18, +2.37] / -2.84 [-8.55, +1.53] | **+0.94** [+0.05, +1.82] / -1.22 [-4.28, +1.78] | +0.07 [-0.74, +0.86] / -1.97 [-6.62, +1.72] | 92.92 |
| post/full | official-chronos2_ctx2048_logret | **+0.29** [+0.07, +0.50] / +0.68 [-1.05, +2.96] | MAE skill -0.29% | MAE skill -0.63% | 220.77 |
| post/full | official-chronos2_ctx2048_price | **+6.80** [+5.68, +7.88] / -3.27 [-9.10, +0.67] | **+2.60** [+1.95, +3.19] / +2.78 [-0.24, +5.37] | **+2.50** [+1.69, +3.24] / **+6.16** [+1.67, +10.35] | 69.5 |
| post/full | official-chronos2_ctx512_aggret | **+0.57** [+0.30, +0.84] / -0.13 [-2.65, +1.88] | **+0.89** [+0.54, +1.27] / +2.38 [-0.92, +5.39] | **+1.18** [+0.69, +1.73] / -0.65 [-8.01, +5.88] | 155.69 |
| post/full | official-chronos2_ctx512_logret | **+0.57** [+0.30, +0.84] / -0.13 [-2.65, +1.88] | MAE skill -0.79% | MAE skill -1.71% | 51.74 |
| post/full | official-chronos2_ctx512_price | **+5.41** [+4.27, +6.49] / -1.95 [-6.22, +1.00] | **+2.62** [+2.00, +3.28] / +3.22 [-0.51, +6.79] | **+3.70** [+2.76, +4.79] / **+5.75** [+1.24, +10.69] | 14.38 |
| post/full | timestral-209M_ctx2048_price | +0.05 [-0.20, +0.29] / **-5.39** [-8.70, -2.90] | -0.08 [-0.31, +0.17] / **-3.98** [-7.61, -1.39] | +0.06 [-0.28, +0.43] / **-4.12** [-8.55, -0.51] | 131.4 |
| post/full | timestral-209M_ctx512_price | **+0.54** [+0.21, +0.89] / **-4.45** [-7.18, -2.52] | +0.13 [-0.19, +0.46] / **-2.98** [-6.08, -0.84] | +0.41 [-0.20, +1.09] / -2.87 [-6.79, +0.43] | 28.74 |
| post/full | timestral-28M_ctx2048_aggret | +0.07 [-0.15, +0.28] / **-4.90** [-8.50, -2.10] | **+0.32** [+0.07, +0.59] / -2.33 [-6.31, +1.25] | **+0.48** [+0.03, +0.92] / -1.58 [-8.57, +4.46] | 221.99 |
| post/full | timestral-28M_ctx2048_price | **+0.32** [+0.04, +0.57] / **-5.56** [-9.35, -2.70] | -0.01 [-0.30, +0.27] / **-3.41** [-6.86, -0.89] | +0.05 [-0.43, +0.55] / **-4.49** [-9.44, -0.71] | 19.45 |
| post/full | timestral-28M_ctx512_aggret | **+0.79** [+0.45, +1.10] / **-3.48** [-7.05, -0.92] | **+0.97** [+0.61, +1.40] / -1.05 [-5.63, +2.73] | **+1.29** [+0.75, +1.92] / -0.60 [-9.26, +6.49] | 46.83 |
| post/full | timestral-28M_ctx512_price | **+0.62** [+0.31, +0.94] / **-5.10** [-8.50, -2.52] | +0.13 [-0.21, +0.52] / **-2.70** [-5.93, -0.55] | +0.45 [-0.16, +1.16] / -2.27 [-6.54, +1.25] | 5.27 |
| post/full | timestral-5M_ctx512_price | **+0.97** [+0.52, +1.35] / **-5.28** [-9.29, -2.29] | **+0.47** [+0.07, +0.90] / **-2.55** [-6.09, -0.18] | **+0.98** [+0.23, +1.90] / -1.81 [-5.87, +1.79] | 1.63 |
| post/n1120_step18 | official-chronos2_ctx2048_aggret_sub6 | +0.45 [-0.18, +1.12] / -3.24 [-6.98, +0.20] | +0.06 [-0.49, +0.69] / +1.17 [-1.93, +4.46] | +0.62 [-0.27, +1.63] / -0.23 [-9.80, +9.25] | 791.97 |
| post/n1120_step18 | official-chronos2_ctx8192_price_sub6 | **+12.64** [+9.16, +15.85] / +0.15 [-6.04, +5.97] | **+3.73** [+1.93, +5.76] / -0.17 [-4.05, +3.73] | **+3.39** [+2.07, +4.86] / +3.91 [-1.73, +9.96] | 333.77 |
| post/n1120_step18 | timestral-209M_ctx8192_price_sub6 | -0.10 [-0.73, +0.47] / **-5.29** [-8.15, -2.79] | -0.37 [-0.90, +0.13] / **-2.92** [-5.76, -0.50] | -0.12 [-0.66, +0.39] / **-5.87** [-11.52, -1.27] | 592.49 |
| post/n1120_step18 | timestral-28M_ctx2048_logret_sub6 | +0.14 [-0.47, +0.78] / **-5.55** [-9.35, -2.42] | MAE skill -0.83% | MAE skill -1.31% | 20.07 |
| post/n1120_step18 | timestral-28M_ctx32768_price_sub6 | **+1.26** [+0.29, +2.06] / **-5.03** [-9.23, -1.40] | +0.00 [-0.69, +0.65] / -2.31 [-5.70, +0.70] | -0.06 [-0.74, +0.56] / **-7.94** [-17.60, -0.82] | 514.63 |
| post/n1120_step18 | timestral-28M_ctx512_price_nosym_sub6 | +0.97 [-0.14, +2.00] / **-4.53** [-8.14, -1.33] | +0.12 [-0.72, +1.00] / -1.40 [-4.34, +1.19] | +0.46 [-0.40, +1.45] / -3.93 [-10.09, +1.17] | 2.17 |
| post/n1120_step18 | timestral-28M_ctx8192_price_sub6 | **+0.88** [+0.10, +1.62] / **-4.78** [-8.27, -1.97] | -0.05 [-0.63, +0.54] / -1.88 [-5.22, +1.09] | +0.35 [-0.28, +0.98] / **-7.97** [-17.52, -0.94] | 95.08 |
| post/n2239_step9 | timestral-28M_ctx2048_price_cov_sub3 | **+0.91** [+0.15, +1.67] / **-2.02** [-4.08, -0.12] | +0.73 [-0.04, +1.56] / -1.68 [-4.57, +0.86] | +0.96 [-0.07, +2.05] / -2.02 [-5.02, +0.75] | 186.52 |


### T-J (repo). Paired ablations on identical origins (variant vs base, % change of loss; negative = variant better)

| Variant | Base | n | Pinball (9 deciles) h=1 / 5 / 15 | QLIKE h=1 / 5 / 15 |
|---|---|---|---|---|
| timestral-209M_ctx8192_price_sub6 | timestral-209M_ctx2048_price | 483 | -0.99 [-2.17, +0.03] / -0.11 [-0.67, +0.45] / -0.43 [-1.01, +0.16] | **-1.61 [-2.78, -0.40]** / -0.71 [-1.82, +0.43] / -0.67 [-1.86, +0.45] |
| timestral-209M_ctx2048_price_cov_sub6 | timestral-209M_ctx2048_price | 483 | -0.76 [-2.83, +1.12] / +0.75 [-0.89, +2.21] / +1.01 [-0.89, +2.77] | -2.89 [-6.62, +0.36] / +0.22 [-1.43, +1.77] / +0.69 [-1.53, +2.63] |
| timestral-28M_ctx8192_price_sub6 | timestral-28M_ctx2048_price | 483 | +0.38 [-0.60, +1.43] / +0.23 [-0.52, +0.94] / -0.55 [-1.33, +0.21] | -0.12 [-1.44, +1.11] / +0.10 [-1.01, +1.13] / +0.41 [-0.62, +1.57] |
| timestral-28M_ctx32768_price_sub6 | timestral-28M_ctx2048_price | 483 | **+3.62 [+2.13, +5.28]** / +0.60 [-0.61, +1.79] / -0.49 [-1.46, +0.46] | **+8.00 [+4.63, +11.67]** / +1.41 [-0.28, +3.09] / +0.96 [-0.33, +2.56] |
| timestral-28M_ctx2048_price_cov | timestral-28M_ctx2048_price | 2895 | +0.11 [-0.36, +0.58] / +0.30 [-0.19, +0.74] / +0.42 [-0.52, +1.27] | **-1.08 [-1.81, -0.20]** / -0.26 [-0.87, +0.37] / +0.05 [-1.05, +1.21] |
| timestral-28M_ctx2048_price_nosym | timestral-28M_ctx2048_price | 2895 | +0.26 [-0.01, +0.52] / +0.02 [-0.20, +0.24] / -0.24 [-0.63, +0.12] | +0.23 [-0.12, +0.66] / +0.22 [-0.20, +0.67] / +0.15 [-0.27, +0.66] |
| timestral-28M_ctx2048_logret | timestral-28M_ctx2048_price | 2895 | **-0.36 [-0.73, -0.01]** / – / – | +0.17 [-0.67, +1.19] / – / – |
| timestral-209M_ctx2048_logret | timestral-209M_ctx2048_price | 2895 | -0.10 [-0.36, +0.20] / – / – | -0.17 [-0.85, +0.56] / – / – |
| timestral-209M_ctx2048_price | timestral-28M_ctx2048_price | 2895 | **-0.37 [-0.69, -0.05]** / **-0.33 [-0.58, -0.08]** / -0.37 [-0.89, +0.07] | -0.03 [-0.65, +0.68] / -0.03 [-0.66, +0.59] / -0.05 [-0.66, +0.69] |
| timestral-28M_ctx2048_price | timestral-28M_ctx512_price | 2895 | +0.12 [-0.32, +0.47] / -0.34 [-0.73, +0.02] / -0.81 [-1.87, +0.07] | -0.12 [-0.81, +0.44] / -0.47 [-1.26, +0.28] / -0.89 [-2.12, +0.42] |
| official-chronos2_ctx2048_logret | official-chronos2_ctx2048_price | 2895 | **-8.33 [-9.94, -6.47]** / – / – | **-4.51 [-6.95, -1.50]** / – / – |
| official-chronos2_ctx512_logret | official-chronos2_ctx512_price | 2895 | **-3.45 [-4.74, -2.02]** / – / – | -0.73 [-3.01, +2.21] / – / – |
| official-chronos2_ctx2048_logret | timestral-28M_ctx2048_logret | 2895 | +0.07 [-0.24, +0.38] / – / – | +0.96 [-0.37, +2.53] / – / – |
| samay-timesfm25_ctx512_logret | samay-timesfm25_ctx512_price | 2895 | **-1.96 [-2.99, -0.79]** / – / – | **+13.40 [+7.90, +19.98]** / – / – |
| neural_trade-rolling1440m-seed0_ctx60_price | neural_trade-asis-seed0_ctx60_price | 2895 | **+3.03 [+1.75, +4.39]** / +0.27 [-0.81, +1.39] / **+1.94 [+0.69, +3.18]** | **+4.81 [+2.66, +6.83]** / **+7.68 [+4.58, +10.76]** / **+6.59 [+3.06, +10.63]** |
| neural_trade-rolling1440m-seed1_ctx60_price | neural_trade-asis-seed1_ctx60_price | 2895 | **+2.33 [+1.50, +3.11]** / **+1.46 [+0.49, +2.55]** / **+1.48 [+0.50, +2.64]** | **+6.06 [+3.15, +9.47]** / **+8.12 [+5.45, +10.99]** / **+6.76 [+4.15, +9.16]** |
| neural_trade-rolling1440m-seed2_ctx60_price | neural_trade-asis-seed2_ctx60_price | 2895 | +0.41 [-0.74, +1.68] / **+1.60 [+0.86, +2.44]** / -0.41 [-1.39, +0.51] | **+3.53 [+1.34, +6.12]** / **+2.63 [+0.23, +5.50]** / +2.76 [-1.42, +6.72] |
| official-chronos2_ctx512_aggret | timestral-28M_ctx512_aggret | 2895 | -0.02 [-0.50, +0.53] / -0.24 [-0.69, +0.15] / -0.00 [-0.73, +0.74] | +0.42 [-0.67, +1.68] / +0.39 [-1.13, +1.98] / **+2.40 [+0.20, +4.93]** |
| official-chronos2_ctx512_aggret | timestral-28M_ctx2048_price | 2895 | +0.46 [-0.08, +1.13] / **+0.52 [+0.08, +1.03]** / **+1.17 [+0.37, +1.98]** | **+2.39 [+0.91, +4.32]** / **+4.70 [+2.18, +7.76]** / **+9.09 [+6.38, +12.39]** |
| timestral-28M_ctx512_aggret | timestral-28M_ctx2048_price | 2895 | +0.48 [-0.08, +1.10] / **+0.77 [+0.18, +1.40]** / **+1.17 [+0.15, +2.14]** | **+1.96 [+0.82, +3.53]** / **+4.29 [+2.09, +7.04]** / **+6.53 [+3.85, +9.33]** |
| timestral-28M_ctx2048_aggret | timestral-28M_ctx2048_price | 2895 | -0.36 [-0.72, +0.03] / +0.22 [-0.20, +0.68] / +0.24 [-0.43, +0.90] | +0.17 [-0.70, +1.17] / **+2.39 [+0.67, +4.26]** / **+3.56 [+1.71, +5.86]** |
| official-chronos2_ctx2048_aggret_sub6 | timestral-28M_ctx2048_aggret | 483 | +0.48 [-0.51, +1.81] / +0.16 [-0.62, +0.93] / -0.40 [-1.26, +0.44] | +3.38 [-0.43, +8.46] / +0.01 [-4.23, +3.82] / +2.06 [-2.80, +7.63] |
| official-chronos2_ctx2048_aggret_sub6 | timestral-28M_ctx2048_price | 483 | -0.59 [-1.54, +0.42] / -0.13 [-0.98, +0.66] / -0.01 [-1.16, +1.20] | +4.26 [-0.88, +11.87] / +2.94 [-1.47, +8.36] / **+7.04 [+0.96, +15.11]** |
| official-chronos2_ctx2048_aggret_sub6 | official-chronos2_ctx512_aggret | 483 | -0.50 [-2.46, +0.91] / +0.09 [-0.57, +0.77] / **-0.97 [-1.93, -0.01]** | +0.45 [-3.81, +4.53] / -1.01 [-5.33, +3.15] / -3.15 [-8.91, +4.26] |


### T-J (post). Paired ablations on identical origins (variant vs base, % change of loss; negative = variant better)

| Variant | Base | n | Pinball (9 deciles) h=1 / 5 / 15 | QLIKE h=1 / 5 / 15 |
|---|---|---|---|---|
| timestral-209M_ctx8192_price_sub6 | timestral-209M_ctx2048_price | 1120 | +0.05 [-0.46, +0.56] / -0.24 [-0.65, +0.15] / -0.22 [-0.61, +0.15] | +0.52 [-0.54, +1.67] / -0.16 [-1.17, +0.98] / -0.59 [-2.22, +0.89] |
| timestral-28M_ctx8192_price_sub6 | timestral-28M_ctx2048_price | 1120 | +0.35 [-0.18, +0.84] / -0.29 [-0.74, +0.07] / +0.15 [-0.35, +0.63] | +0.13 [-0.84, +1.10] / -0.12 [-1.47, +1.36] / -2.60 [-6.72, +0.33] |
| timestral-28M_ctx32768_price_sub6 | timestral-28M_ctx2048_price | 1120 | +0.72 [-0.05, +1.49] / -0.24 [-0.77, +0.22] / -0.26 [-0.81, +0.24] | -0.13 [-2.03, +1.68] / -0.56 [-1.94, +0.80] / -2.57 [-7.09, +0.29] |
| timestral-28M_ctx2048_price_cov_sub3 | timestral-28M_ctx2048_price | 2239 | +0.48 [-0.12, +1.09] / **+0.88 [+0.23, +1.64]** / **+0.88 [+0.16, +1.59]** | +0.99 [-0.26, +2.38] / -0.00 [-0.79, +0.97] / +1.66 [-0.31, +4.53] |
| timestral-28M_ctx512_price_nosym_sub6 | timestral-28M_ctx512_price | 1120 | +0.23 [-0.19, +0.64] / -0.10 [-0.40, +0.17] / -0.06 [-0.44, +0.26] | **+0.79 [+0.19, +1.44]** / +0.13 [-0.32, +0.61] / -0.21 [-1.08, +0.52] |
| timestral-28M_ctx2048_logret_sub6 | timestral-28M_ctx2048_price | 1120 | -0.39 [-0.95, +0.23] / – / – | -0.67 [-2.05, +0.78] / – / – |
| official-chronos2_ctx8192_price_sub6 | official-chronos2_ctx2048_price | 1120 | **+6.09 [+2.93, +9.18]** / **+2.08 [+0.35, +3.83]** / +1.00 [-0.25, +2.51] | **+4.23 [+1.16, +7.18]** / -0.36 [-3.71, +2.69] / -1.72 [-5.39, +2.15] |
| timestral-209M_ctx2048_price | timestral-28M_ctx2048_price | 6715 | **-0.26 [-0.49, -0.00]** / -0.07 [-0.25, +0.13] / +0.01 [-0.32, +0.35] | +0.18 [-0.51, +0.94] / -0.59 [-1.61, +0.29] / +0.39 [-0.93, +1.79] |
| timestral-28M_ctx2048_price | timestral-28M_ctx512_price | 6715 | **-0.31 [-0.61, -0.04]** / -0.14 [-0.47, +0.14] / -0.40 [-0.99, +0.19] | -0.49 [-1.48, +0.26] / -0.73 [-1.62, +0.17] / **-2.27 [-4.24, -0.67]** |
| official-chronos2_ctx2048_logret | official-chronos2_ctx2048_price | 6715 | **-6.10 [-7.11, -5.01]** / – / – | +4.08 [-0.79, +10.54] / – / – |
| official-chronos2_ctx512_logret | official-chronos2_ctx512_price | 6715 | **-4.59 [-5.50, -3.60]** / – / – | +1.85 [-0.36, +4.76] / – / – |
| official-chronos2_ctx2048_logret | timestral-28M_ctx2048_price | 6715 | -0.03 [-0.29, +0.28] / – / – | **+6.61 [+2.64, +12.00]** / – / – |
| samay-timesfm25_ctx512_logret | samay-timesfm25_ctx512_price | 6715 | **-2.19 [-3.06, -1.31]** / – / – | **+31.34 [+23.63, +41.17]** / – / – |
| neural_trade-rolling1440m-seed0_ctx60_price | neural_trade-asis-seed0_ctx60_price | 6715 | **-2.76 [-3.85, -1.71]** / **-1.10 [-1.89, -0.34]** / **-3.13 [-4.13, -2.06]** | **+3.54 [+0.30, +6.87]** / **-4.70 [-7.62, -1.74]** / **-8.20 [-13.51, -1.94]** |
| neural_trade-rolling1440m-seed1_ctx60_price | neural_trade-asis-seed1_ctx60_price | 6715 | -0.37 [-1.02, +0.30] / +0.24 [-0.48, +0.93] / **-1.17 [-2.14, -0.24]** | **+8.49 [+3.46, +13.96]** / -0.28 [-4.40, +4.05] / -0.84 [-7.17, +6.28] |
| neural_trade-rolling1440m-seed2_ctx60_price | neural_trade-asis-seed2_ctx60_price | 6715 | +0.32 [-0.66, +1.21] / -0.28 [-0.99, +0.41] / -0.94 [-2.04, +0.06] | **+5.67 [+0.51, +11.07]** / +1.08 [-3.83, +6.65] / +4.51 [-4.05, +13.13] |
| official-chronos2_ctx512_aggret | timestral-28M_ctx512_aggret | 6715 | -0.21 [-0.48, +0.04] / -0.08 [-0.44, +0.24] / -0.11 [-0.68, +0.38] | **+3.46 [+1.45, +6.08]** / **+3.47 [+0.96, +6.40]** / -0.04 [-4.40, +4.78] |
| official-chronos2_ctx512_aggret | timestral-28M_ctx2048_price | 6715 | +0.26 [-0.05, +0.61] / **+0.90 [+0.52, +1.33]** / **+1.13 [+0.47, +1.82]** | **+5.75 [+3.16, +9.94]** / **+6.00 [+3.54, +8.41]** / +4.02 [-1.01, +8.63] |
| timestral-28M_ctx512_aggret | timestral-28M_ctx2048_price | 6715 | **+0.47 [+0.17, +0.84]** / **+0.98 [+0.60, +1.41]** / **+1.24 [+0.62, +1.90]** | **+2.21 [+0.77, +4.11]** / **+2.44 [+0.03, +4.47]** / +4.07 [-0.14, +7.91] |
| timestral-28M_ctx2048_aggret | timestral-28M_ctx2048_price | 6715 | -0.24 [-0.46, +0.01] / **+0.33 [+0.03, +0.65]** / +0.43 [-0.06, +0.94] | +0.70 [-0.15, +1.72] / +1.12 [-0.88, +3.05] / +3.04 [-0.31, +6.11] |
| official-chronos2_ctx2048_aggret_sub6 | timestral-28M_ctx2048_aggret | 1120 | +0.31 [-0.15, +0.81] / -0.24 [-0.84, +0.39] / +0.18 [-0.47, +0.94] | **+2.44 [+0.32, +4.63]** / +2.98 [-0.46, +6.29] / +4.75 [-3.80, +16.39] |
| official-chronos2_ctx2048_aggret_sub6 | timestral-28M_ctx2048_price | 1120 | -0.08 [-0.73, +0.68] / -0.18 [-0.93, +0.61] / +0.42 [-0.49, +1.33] | +1.75 [-0.88, +4.47] / +2.99 [-0.47, +7.18] / +5.59 [-4.21, +18.64] |
| official-chronos2_ctx2048_aggret_sub6 | official-chronos2_ctx512_aggret | 1120 | -0.33 [-1.01, +0.33] / -0.52 [-1.11, +0.06] / **-0.62 [-1.19, -0.01]** | -1.82 [-5.01, +1.09] / -2.99 [-6.63, +0.28] / +4.20 [-3.71, +14.45] |


### T-H. Single-series CPU latency (4 threads unless stated, fp32, package defaults)

| Model | Context | Symmetric avg. | p50 ms | p95 ms | p99 ms | Cold start ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|
| timestral-5M | 64 | True | 5.6 | 6.5 | 7.5 | 43.6 | 400.7 |
| timestral-5M | 64 | False | 2.8 | 3.3 | 3.6 | 43.6 | 400.7 |
| timestral-5M | 512 | True | 6.9 | 10.4 | 11.4 | 43.6 | 400.7 |
| timestral-5M | 512 | False | 3.7 | 4.9 | 6.0 | 43.6 | 400.7 |
| timestral-5M | 2048 | True | 11.6 | 13.2 | 14.6 | 43.6 | 400.9 |
| timestral-5M | 2048 | False | 5.7 | 6.5 | 7.2 | 43.6 | 400.9 |
| timestral-5M | 8192 | True | 26.5 | 30.4 | 34.9 | 43.6 | 406.6 |
| timestral-5M | 8192 | False | 13.3 | 15.2 | 17.1 | 43.6 | 406.6 |
| timestral-5M | 32768 | True | 127.0 | 145.3 | 154.1 | 43.6 | 428.1 |
| timestral-5M | 32768 | False | 67.3 | 77.3 | 97.6 | 43.6 | 428.1 |
| timestral-28M | 64 | True | 10.9 | 14.1 | 15.1 | 148.0 | 619.2 |
| timestral-28M | 64 | False | 5.3 | 6.1 | 7.1 | 148.0 | 619.2 |
| timestral-28M | 512 | True | 17.0 | 21.0 | 23.6 | 148.0 | 619.2 |
| timestral-28M | 512 | False | 8.4 | 9.8 | 10.6 | 148.0 | 619.2 |
| timestral-28M | 2048 | True | 29.3 | 32.6 | 36.0 | 148.0 | 619.2 |
| timestral-28M | 2048 | False | 14.7 | 16.0 | 20.7 | 148.0 | 619.2 |
| timestral-28M | 8192 | True | 76.2 | 91.7 | 116.9 | 148.0 | 619.2 |
| timestral-28M | 8192 | False | 40.8 | 50.4 | 59.8 | 148.0 | 619.2 |
| timestral-28M | 32768 | True | 318.4 | 355.5 | 381.8 | 148.0 | 619.2 |
| timestral-28M | 32768 | False | 159.9 | 183.5 | 202.8 | 148.0 | 619.2 |
| timestral-123M | 64 | True | 34.8 | 41.8 | 48.1 | 720.5 | 1435.1 |
| timestral-123M | 64 | False | 16.3 | 23.9 | 25.2 | 720.5 | 1435.1 |
| timestral-123M | 512 | True | 57.3 | 69.6 | 78.1 | 720.5 | 1435.1 |
| timestral-123M | 512 | False | 29.3 | 39.3 | 46.8 | 720.5 | 1435.1 |
| timestral-123M | 2048 | True | 103.1 | 122.1 | 143.9 | 720.5 | 1435.1 |
| timestral-123M | 2048 | False | 52.1 | 65.5 | 78.7 | 720.5 | 1435.1 |
| timestral-123M | 8192 | True | 272.0 | 302.7 | 337.7 | 720.5 | 1435.1 |
| timestral-123M | 8192 | False | 136.0 | 158.8 | 184.7 | 720.5 | 1435.1 |
| timestral-123M | 32768 | True | 1130.8 | 1218.7 | 1249.5 | 720.5 | 1435.1 |
| timestral-123M | 32768 | False | 547.8 | 612.2 | 657.1 | 720.5 | 1435.1 |
| timestral-209M | 64 | True | 50.6 | 59.6 | 71.9 | 950.9 | 2468.9 |
| timestral-209M | 64 | False | 25.0 | 27.3 | 29.4 | 950.9 | 2468.9 |
| timestral-209M | 512 | True | 97.4 | 118.8 | 144.3 | 950.9 | 2468.9 |
| timestral-209M | 512 | False | 46.9 | 50.7 | 55.7 | 950.9 | 2468.9 |
| timestral-209M | 2048 | True | 161.7 | 178.7 | 207.5 | 950.9 | 2468.9 |
| timestral-209M | 2048 | False | 80.0 | 94.6 | 101.0 | 950.9 | 2468.9 |
| timestral-209M | 8192 | True | 420.3 | 469.9 | 492.6 | 950.9 | 2468.9 |
| timestral-209M | 8192 | False | 209.9 | 238.1 | 262.1 | 950.9 | 2468.9 |
| timestral-209M | 32768 | True | 1759.0 | 1862.1 | 1870.4 | 950.9 | 2468.9 |
| timestral-209M | 32768 | False | 887.3 | 947.9 | 981.1 | 950.9 | 2468.9 |
| chronos-2 (official) | 64 | – | 23.5 | 27.7 | 30.7 | 1181.5 | 989.9 |
| chronos-2 (official) | 512 | – | 38.5 | 43.9 | 46.3 | 1181.5 | 990.1 |
| chronos-2 (official) | 2048 | – | 76.9 | 91.6 | 106.5 | 1181.5 | 990.1 |
| chronos-2 (official) | 8192 | – | 273.5 | 303.8 | 324.3 | 1181.5 | 1018.2 |
| neural_trade as-is (Model.predict, as in model.py, 4 thr) | 60 | – | 58.83 | 62.87 | 73.45 | – | – |
| neural_trade as-is (tf.function call, 4 thr) | 60 | – | 19.16 | 20.25 | 21.75 | – | – |
| neural_trade as-is (Model.predict, as in model.py, 1 thr) | 60 | – | 60.15 | 67.22 | 73.28 | – | – |
| neural_trade as-is (tf.function call, 1 thr) | 60 | – | 19.37 | 21.48 | 23.9 | – | – |


### T-I. Samay (locked env) vs official packages, identical origins

```json
{
 "samay-chronos2_ctx512_price_sub6": {
  "n": 483,
  "max_abs_diff_bps_all": 0.0007886046626250677,
  "max_abs_diff_bps_deciles": 0.0007886046626250677,
  "mean_abs_diff_bps": 1.927978906644475e-06
 },
 "samay-bolt_small_ctx512_price_sub6": {
  "n": 483,
  "max_abs_diff_bps_all": 0.0007895169206503125,
  "max_abs_diff_bps_deciles": 0.0007895169206503125,
  "mean_abs_diff_bps": 5.85801867837804e-05
 }
}
```

---

## Appendix B. Main sources

- Timestral:
  - code: https://github.com/mistralai/timestral (commit a534bdb)
  - models: https://huggingface.co/mistralai/Timestral-209M (and -123M, -28M, -5M), including LICENSE
  - blog: https://mistral.ai/news/timestral
  - technical report: arXiv:2610.37585
  - issues: https://github.com/mistralai/timestral/issues (#3, #7)
- Samay: https://github.com/AdityaLab/Samay (commit d23c90f); https://pypi.org/project/samay/ (the unrelated package)
- Samaya Sterling (out of scope): https://samaya.ai/blog/rl-post-trained-agentic-model
- Chronos-2 / Chronos-Bolt: https://github.com/amazon-science/chronos-forecasting; weights https://huggingface.co/amazon/chronos-2 (Apache-2.0)
- TimesFM-2.5: https://huggingface.co/google/timesfm-2.5-200m-pytorch (Apache-2.0)
- Literature: arXiv 2511.18578, 2508.02739, 2412.09880, 2505.11163, 2607.05291, 2410.11773, 2606.27100, 2608.03259, 2203.10716; Jaquart, Dann & Weinhardt, *J. Finance and Data Science* 7 (2021); Jheng et al., ECECSR 60(2) (2026).
- Market data: Binance public market-data mirror, https://data-api.binance.vision (BTCUSDT 1-minute klines).
- Fees: https://www.binance.com/en/fee/schedule
