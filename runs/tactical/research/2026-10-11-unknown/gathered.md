## project
# neural_trade tactical evidence: what failed, what is still open, and what can be reused

All paths are relative to `D:/nt/nt_tactical/`. The journal is `runs/tactical/LOG.md` (H5-H65), the plan is `runs/tactical/PLAN.md`, the audit is `runs/tactical/rebuild/AUDIT.md`, and the MVP summary is `runs/tactical/MVP_SUMMARY.md`.

**One warning applies to every number below.** Most lab AUCs enter at the signal bar's close. Bid-ask bounce makes them optimistic by about 0.03 for 2018-20 and about 0.00-0.01 for 2024-25 (H59-H61). The 24-slice lab mean of 0.570 overstates the present: 2024-25 is about 0.52-0.53 (H51, H59).

---

## 1. Negative space: tested and failed (do not re-propose)

### 1a. Direction ceiling and baselines
- **The ceiling.** A logistic regression on 13 scale-free features ("tb7") reaches it: 7 textbook indicators at the last bar (EMA20 distance, RSI14, MACD histogram, BB %b, ATR, Stoch, CCI) plus 6 trailing returns, all divided by window volatility.
  - Mean of 3 horizons AUC: 0.570 [0.552, 0.588] on 24 slices (H27, H28).
  - Most of it comes from trailing returns alone: 0.576 (H27).
  - The signal is short-term reversal. The univariate AUC of the 1-bar return was 0.38-0.42 in 2017-20 and 0.505 in 2025 (H52).
- **Decay.** Monthly walk-forward AUC by year: 2017 0.568, 2018 0.590, 2019 0.599, 2020 0.590, 2021 0.549, 2022 0.544, 2023 0.554, 2024 0.531, 2025 0.520 (H51).
  - With next-open fills: 0.56-0.57 in 2018-20, 0.533 in 2024, 0.517 in 2025 (H59).
  - It decays the same way at every bar size and horizon: 1m/5m/15m/1h × 1-30 bars, 0.58-0.65 then 0.50-0.54. The signal does not move to another scale (H52).
- **Larger bars are less predictable.** Next-bar AUC: 1h 0.557, 4h 0.543, 1d 0.525 (not significant). The owner's "70% daily" version does not exist (H29).
- **The recent market with everything built in** (H56, pre-registered, look period 2024-07..2025-09):
  - 1m: all features + L1 gives +0.0042 [+0.0012, +0.0072] AUC over tb7. With next-open fills it is +0.0040 [+0.0004, +0.0076]; it passes at 95% but not at the Bonferroni 98.3% (H61).
  - 1h: nothing beats tb7 (AUC about 0.547).
  - Training on 365 days beats 60 days everywhere.

### 1b. Network architectures for direction (none beats the regression)
- **The default `gru_attention` network** (316,751 parameters) scores 0.52-0.55, and the regression beats it by +0.036 [+0.006, +0.067] (H23, H27). Hill-climb Config switches change nothing:
  - Head constraints: `skip_only`, `shrink`, dropout (H5).
  - Epochs, learning rate, `LAMBDA_DIR` ×5, `LOOKBACK` 20 and 120, batch 1024, no physics, calibration by value or by gradient (H6, H12, H13). Calibration by gradient is worse.
  - The one adoption, the no-price-head network, fails on the FINAL slices (H30).
- **The rebuild ladder:**
  - The Keras linear model equals sklearn exactly (L1 vs L0: +0.0000).
  - None of these beats the regression: a rich linear model on 43 features (−0.0018), a residual MLP (−0.0008), GRU(32) (−0.0001) (H28).
  - The same holds with training from 11.5 days up to 3 years (H35).
- **The new network** (PatchTST-lite or TCN, about 20-27k parameters, a zero-init residual over the frozen regression):
  - 1 year: −0.0002 (H36).
  - 3 years, 3-seed ensemble: +0.0026 [+0.0001, +0.0052]; on the FINAL slices +0.0044 [+0.0012, +0.0075].
  - It misses the +0.005 criterion, and the residual stops at epoch 0 on 25-46% of slices (H44).
- **Long hourly context** (168 or 336 bars, patch transformer or TCN): −0.004 to −0.008. A planted-pattern check shows the nets can learn (0.49 → 0.79-0.83) (H65).
- **Head combinations:** averaging the 3 horizons or requiring all 3 to agree gives +0.005..+0.022 AUC in-sample (H15). The tail above the null did not hold out of sample (H19/H20 → H30: 59.6% → 54.3%, +4.5 → +0.3 bps).
- **The price head** carries no information: correlation 0.01-0.04 and skill below predicting zero, including a price-only network (H14, H31).

### 1c. Indicators (the project's core premise)
- Learned periods equal textbook periods (logreg vs logreg_frozen: −0.0006 [−0.006, +0.005]) (H25).
- No indicators at all equals 14 learned families on direction, with better log loss (H26 D).
- A thin linear readout: learned vs textbook periods −0.002 [−0.005, +0.001] (H26 A).
- Splitting the network by function (direction, confidence, price) gives no gain on any function (H26 B, H31).
- Starting periods do not matter (H37). An Optuna search of the periods adds +0.004 (H23).
- On 1-day blocks fewer families win only because 54 channels overfit about 1,440 windows; the advantage disappears on the long block (H21, H22, H37).
- Indicator gradients from the direction loss only: +0.001 (H24). Combined with geometry, nothing beyond geometry alone (H43).

### 1d. Geometry (exhaustively negative for direction)
- Slopes, distances, crossings, squeeze:
  - In the network: AUC +0.013, not significant; only log loss improves, which is calibration and not skill (H24, H43).
  - In the lab: no gain (H45).
  - On 22 channels (164 features) with logistic, boosting and MLP readouts and 11.5 days to 3 years of training: at best +0.0024, with the CI never above 0 (H47).
  - Squeeze is volatility persistence, not a breakout signal (H47).
  - In the triple barrier, geometry stands in for nonlinearity and adds nothing on top of boosting (H47).
- **Novel families N1-N7** (Frenet curvature and torsion of indicator trajectories, Lévy areas of price against volume/RSI/EMA/range, phase-portrait winding, convex-hull support and resistance, Wasserstein shift inside the window, Fisher-Rao path of (μ, σ), convexity): every difference is ≤ +0.0001, and 11 of 16 are significantly negative after Holm (H49).
- Excluded earlier by literature search: path signatures (negative) and TDA (mixed) (`novel/SPEC.md`).
- **Indicators as a loss on the predicted future path** (hypothesis E: a path head plus a shape loss): no effect (H32, H37).

### 1e. JEPA (four rounds, about 20 GPU-hours)
- **v1:** a 26.7k patch-transformer encoder with an EMA target on the next 15 bars, smooth-L1 + VICReg loss, pretrained on 2017-2019. Embedding probe 0.524 vs tb7 0.547; both together −0.007 (H34).
- **v2:** a longer future (60 bars in 3 segments), I-JEPA-style masked context, and both. Direction is below tb7 in every variant (−0.006..−0.009) (H41).
- **v3:** 120k steps × 2 seeds; the embedding as a meta-model trade filter gives nothing (−0.11 bps) (H46).
- **v4:** pretraining on 2017-21, mask ratios 30/50/70%, a 4× encoder (103k), a 120-bar context, and end-to-end fine-tuning (H55).
  - Fine-tuned direction equals the regression: −0.0001 [−0.0046, +0.0043].
  - **The volatility gains of JEPA (+0.10 Spearman over tb7) vanish once HAR log-variances are in the linear part** (−0.002..+0.002).
- No representation collapse anywhere (effective rank 28-30 of 32).

### 1f. Physics and loss terms
- The six physics terms (t_perp, Casimir, HD, IFE, vacuum, vacuum overflow):
  - No-physics run: +0.36 noise units, not significant (H12).
  - Removing HD: −0.006 (H24).
- Graph-mode gradient probe (H42, which corrects H16):
  - Direction carries 30-45% of the trunk gradient and NLL 40-50%.
  - Direction is nearly orthogonal to every other term on the trunk (|cos| ≤ 0.13).
  - NLL-CRPS (+0.5..+0.9) and NLL-t_perp (+0.3..+0.8) are aligned.
  - Reading: the physics terms are coupled to the variance path, not to direction.
- Shape loss on the predicted path (H32) and a raised direction weight (H6): nothing.
- The regime gate on the direction head improves log loss only (H24).

### 1g. Other inputs and effects
- **Order-flow proxies from OHLCV** (BVC imbalance at 5/15/60, VPIN-style, accumulation/distribution, VWAP-return gap, a Kyle-λ proxy): +0.0019 on selection, −0.0003 on the look (H64).
- **Calendar effects** (36 buckets: hour, weekday, funding hours, month ends, weekends): only hour 01 UTC passed on selection (+6.7 bps). It fails on the look (+2.6 bps), and it fades by year (H63).
- **Regimes** (volatility level and trend, trend strength, session, weekend): the US session gives +0.008 AUC, a real but tiny effect. No hourly regime filter survives the held-out period (H40).
- **Excursion asymmetry** (next max-high against min-low): 0.58-0.61 in 2017-20, 0.51-0.52 in 2021-25 (H54).
- **The 0.805-AUC slice** (Easter-weekend 2022-04-16 range): a regime, not skill. Fading the last 10 bars scores 0.756 there (H7, H9).
- **Context features beyond the window** (17 features: 1h/4h/1d/1w returns, ranges, time of day): add nothing to the regression (H36).

### 1h. Trading constructions (all fail with realistic fills or fees)
- **Triple barrier with meta-labelling, minute bars:**
  - The held-out pass (H48) was +0.54 bps per trade at zero cost.
  - With next-open fills it is +0.30 bps, the monthly CI [−0.20, +0.79] crosses 0, and it fails (H60).
  - Stricter selection or wider barriers: the best edge is about 1 bps over a random side (H50).
  - The same configuration earned +5.3 bps in 2017-19, so the edge has decayed about 4×.
- **Hourly barrier:**
  - Dev +32 bps per trade, held-out +9.8, failed (H38).
  - With next-open fills: +5.4 bps, below its 95th-percentile null (H61).
- **Equity against the owner's goal** (hit > 60%, drawdown < 5%): nothing reaches it.
  - Hit rates are 52-57% and maximum drawdowns 9-29% (H58, H61).
  - The closest is 1h rich L1: 57% hit, 9.2% drawdown, profitable up to 2 bps of cost.
- **Frozen 4h candidate:** a 2019-21 effect; 2022..2024-06 gives +1.9 bps (H62).
- **Magnitude filter:** it lifts the hit rate to 62%, but the bps stay under the null, because the filter selects larger moves on either side (H35).
- **Over-confidence of P(up):** Platt scaling brings it back to the constant; there is no average edge (H20).

### 1i. Volatility (positive but saturated)
- **HAR-RV + time of day** is near the ceiling for these inputs. Its Spearman with |r| grew: 0.29 (2018) → 0.40 (2023-25) (H53).
- Learned models beat it on ranking but not on QLIKE:
  - Minute: ridge +0.014, tower +0.016..+0.019 (H39).
  - Hourly: boosting on context +0.045 (H39).
- **volnet** (pre-registered): the ensemble gains +0.006 Spearman at 15 and 60 minutes, but QLIKE fails. It passes both only at 240 minutes (+0.009 Spearman, +0.026 QLIKE).
  - About half of that gain comes from volume and range features.
  - The transformer never left its HAR starting point (H57).
- tb7 adds about 0.005 to HAR (H53). Geometry adds about 0.02 over bare HAR and nothing once richer vectors are present (H47).

---

## 2. Open loose ends ("found, not done") that could seed new questions

1. **Why did volatility predictability grow while direction decayed?** (H53 vs H51/H59.) This is described but never explained or modelled. Candidates: the market-structure shift itself as the object of study, and the anti-correlation between the two as a signal.
2. **The physics terms were never judged on their natural axis.** H42 shows t_perp, NLL and CRPS are aligned and orthogonal to direction. No test asks whether a physics term improves volatility or calibration against the **HAR baseline**. They were only judged on direction and the composite metric. AUDIT C11 lists defects in the variance path: 1/v gradients, a zero-gradient floor, training-only noise, a dead output at inference.
3. **The scale of the variance head versus its ranking.** At 1 day the net ranks risk as well as at 7 days (Spearman 0.23 vs 0.24), but its intervals are overconfident (coverage 0.31-0.45) (H11). The screen never rescales on the calibration block. Ranking and scale separate, and that is unexplored.
4. **JEPA pretraining had not converged.** Masked-context losses were "still falling at 30k" (H41), and the volatility gain flattens at 120k (H46). Every JEPA target was a latent of raw future bars. No JEPA had a target defined on **HAR residuals**, a distributional target, or **volatility-change** targets, which matters given H55's "HAR absorbs it".
5. **The new-network residual stops at epoch 0** on 25-46% of slices (H44). The H65 nets leave their linear start but move only calibration. "The residual learns calibration, not ranking" is an observed but undissected phenomenon.
6. **The 1m L1 sliver of about +0.004 AUC** survives next-open fills at 95% (H56, H61). Its source among the 284 features was not attributed.
7. **Quiet-regime reversal.** Easter-weekend fade AUC 0.756 (H7). Low trend strength gave +0.014 but missed Holm (H40). PLAN §4 ("volatility-gated direction: reversion in quiet regimes, continuation in trends") was tested only as a network gate (H24) and as tercile cells, never as a conditional model.
8. **Agreement across horizons.** All-3-agree covers 56% of bars and gives +0.016..+0.022 AUC in-sample (H15). Its out-of-sample tail failed (H30). Multi-horizon consistency as a structural target or constraint, rather than a filter, was not tried.
9. **An anti-collapse guard in the aggregate metric.** Batch 1024 "wins" by doing less (H10). This was flagged and never fixed in `hc4_metric.py`.
10. **Rebuild ladder rungs not run** (H28 ran L0, L1, L3, L5, L6, L7):
    - L2: in-graph features and their soft/EWMA versions.
    - L4: learnable periods on the clean stack.
    - L8: separate towers for the other heads, then the physics terms added one at a time with direction held within ±0.003.
    - PLAN §8's "proof of dominance" (the indicator gradient share from their own loss ≥ 50%) was never met.
11. **A pending forward test.** The 4h candidate is frozen in `fwd/FORWARD_SPEC.md` but needs data after 2025-09-29, which is blocked by D-050 (H62).
12. **Never touched by any run** (grep over `runs/tactical` finds no skew, kurtosis, jump or Hawkes terms in the journal or plan):
    - forecasting higher moments, tails or jump probability;
    - the sign of the volatility change (Δ log RV) as a target;
    - volume forecasting;
    - time-reversal asymmetry;
    - barrier hit ordering as a target, as opposed to a trade rule;
    - hazard or point-process models of extreme bars.

---

## 3. Practical constraints and reusable infrastructure

### Data and protocol
- **Data:** BTC/USDT 1-minute OHLCV, 2017-01..2025-09-29 only (`D:/nt/neural_trade/Bitcoin_BTCUSDT.csv`). No new source (D-050). Trading costs are 0 by default (D-044), but report a 1/2/5 bps cost grid, as `equity/` does.
- **The tactical excerpt:** `Bitcoin_BTCUSDT_tactical.csv`, 1.74M rows covering the 12 slices plus the last 70 days (`runs/tactical/data/make_excerpt.py`).
- **What is spent or protected:**
  - The held-out period 2024-07..2025-09 is spent for direction hypotheses. H56, H57, H61, H63 and H64 each took their one look.
  - The 6 FINAL slices were used by H30, H44 and H48-era checks.
  - The protected span (`DATA_END_PROTECTED_DAYS`, the last 70 days) is never used.
- **Honest-protocol conventions to keep:**
  - Pre-register a SPEC and commit it before any result. This was done for H49, H56, H57, H63 and H64 (`novel/`, `recent/`, `volnet/`, `calendar/`, `flow/` `SPEC.md`).
  - Monthly walk-forward with a purge: a training label must close before the month starts (`recent/wf.py split_masks`).
  - Selection months 2021-01..2024-06, then exactly one look.
  - Next-open entry. Non-overlapping trades. A random-side null at the 95th percentile. Paired differences with t-intervals over months or slices, and Holm or Bonferroni corrections.
  - The honest split: fit thresholds on the first half of a validation block, test on the second (`probe/honest_split.py`).

### Code that can be reused
| What | Where |
|---|---|
| Lab cache: 24 slices × (`Wtr`, `Wva` [N,60,5] OHLCV; `rtr`, `rva` [N,3] returns; deadband), on the network's exact blocks | `runs/tactical/lab/cache/*.npz`, `cache.py`, `cache_final.py` (FINAL slices), `lab.py` (tb7, returns and rich-43 feature sets, logistic regression and boosting) |
| Decay and walk-forward library: bar caches for 1m/5m/15m/1h at `D:/nt/nt_decay_cache`, monthly walk-forward, yearly summaries | `lab/decay_lib.py`, `decay*.py`, `test_decay_walk.py` |
| Recent-market walk-forward, 7 feature sets (tb7, rich, ctx, gR geometry, novel, HAR, all) | `recent/wf.py`, `recent/feats.py` |
| Volatility benchmarks: RV15-1440, EWMA, GARCH(1,1) MLE, HAR, HAR+tod, Spearman and QLIKE with a train-fitted scale | `vol/volbench.py`; `volnet/volnet_lib.py`, `run.py` (MLP or transformer with a HAR skip at initialisation) |
| JEPA: patch-transformer encoder, EMA target, predictor, VICReg; pretraining, probes, fine-tuning; checkpoints | `jepa/model.py`, `pretrain.py`, `probe.py`, `jepa/v2..v4/` (`model4.py`, `finetune4.py`), `jepa/*/ckpt` |
| New network: frozen regression logit plus a zero-init PatchTST-lite or TCN residual; a separate volatility tower with Laplace NLL on log\|r\| | `newnet/data.py`, `model.py`, `train.py` |
| Rebuild ladder, long-span training | `rebuild/ladder.py`, `longtrain.py` |
| Geometry and novel feature libraries (with tests) | `geom2/geomlib.py`, `novel/features.py` |
| Triple barrier and meta-labelling | `lab/barrier.py`, `geom2/meta.py`, `meta_nextopen.py`, `holdout_min*.py` |
| Equity and forward test: non-overlapping trades, cost grid, random-side null | `equity/eqlib.py`, `fwd/fwdlib.py` |
| Graph-mode per-term gradient probe (2× faster) | `runs/tactical/probe/graph/` (H42) |
| Main network's loss terms, including physics; tactical switches `PRICE_HEAD`, `ACTIVE_HORIZONS`, `INDICATOR_GRAD_SOURCE`, geometry, `DATA_TAIL_TRIM` | `src/neural_trade/losses/functions.py:615-1048`; the switches are in `core/config.py` |

### Compute and operations
- **GPU:** one RTX 4070 Ti with 12 GB, TF 2.10 on Windows, 32 CPU threads.
  - Each tactical GPU run is limited to at most 2 minutes on ultra-short blocks (D-063). This was often exceeded with the owner's consent.
  - At most 2 heavy processes at once: three caused two NVIDIA driver resets (H43).
  - Parallel processes do not raise throughput (H10).
  - Run through the load-guard scripts at below-normal priority with `TF_FORCE_GPU_ALLOW_GROWTH` (`docs/TACTICAL.md`).
- **Disk and RAM:** disk D: was nearly full in the MVP status, and RAM limits caused stops (H19, H21).
- **Noise floor:**
  - One slice's AUC has an error of about 0.06-0.12, and the 6-slice screen sees only effects of about +0.05.
  - A 0.005 effect needs 24+ slices or about 40+ walk-forward months.
- **Rules:** no default changes from the tactical branch (D-025). The physics terms stay in the project (D-003): removing one goes to the owner. TF 2.10 / Keras 2 only (D-001).

## loss_physics
# Loss-function physics for 1-minute crypto forecasting: literature sweep, 2022-2026 focus

## Bottom line

**No source I found shows a physics-, geometry- or OT-based loss creating a direction edge on intraday crypto, or on any liquid asset, at 10-20 minutes.** Confidence: high that nothing like that is well known. Confidence: medium that nothing exists anywhere.

The published positive results fall into three groups:
- **Distributional and path generation:** signature-kernel scores, Sig-W1, energy score, diffusion models.
- **Risk:** jointly elicitable VaR/ES losses.
- **Instruments that carry a real no-arbitrage structure:** option surfaces and yield curves.

Spot prices at 1 minute have none of that structure. Two consequences:
1. The best open directions are **(a)** better density, volatility and risk forecasts, and **(b)** diagnostics that measure how much predictive information exists, or where it sits.
2. Hoping a new loss will push direction AUC above about 0.52 is not well supported.

A few ideas look genuinely unexplored for 1-minute BTC OHLCV:
- multi-horizon martingale / temporal-difference consistency;
- NEEP-style entropy production (the arrow of time) as a gate for when direction is predictable;
- a ceiling on predictive information;
- signature-kernel or energy scores on the joint 10/15/20-minute path;
- a Fissler–Ziegel VaR/ES loss against a HAR baseline.

## 1. Physics-informed (PDE / conservation-law) losses for prices

| | |
|---|---|
| **What exists** | PINNs put a PDE residual into the loss. In finance they are used almost only for **pricing PDEs**: Black–Scholes, and Heston through Feynman–Kac (Hainaut & Casas 2024, cited inside the PMC review below). For forecasting raw series: "PINNs with unknown PDEs … multivariate time series" (Mortezanejad et al., 2025, [PMC12294105](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12294105/)). Physics-inspired Newton's-law KAN for RL trading ([arXiv 2602.01388](https://arxiv.org/pdf/2602.01388)). ARTEMIS, a neuro-symbolic model with economic constraints ([arXiv 2603.18107](https://arxiv.org/pdf/2603.18107)). |
| **Financial / HF tests** | The forecasting papers use daily stock data or toy setups. I found none on intraday crypto with a strong baseline such as a logistic regression or HAR. |
| **Results** | Reported gains are against weak baselines (plain LSTM). No out-of-sample direction claim I would trust. |
| **Gap for us** | A spot price has no governing PDE. The only defensible "physics" is **probabilistic consistency** between horizons: Chapman–Kolmogorov / Fokker–Planck coherence of the predicted densities at 10, 15 and 20 minutes (see idea H4). This is not published for multi-horizon return heads, as far as I found. |
| **Confidence** | High that the literature here is thin and weak. |

## 2. Hamiltonian, Lagrangian and symplectic objectives

- **What exists:** HNN (Greydanus, Dzamba, Yosinski, NeurIPS 2019); LNN (Cranmer et al., 2020); symplectic-integrator HNets (overview: [deepai](https://deepai.org/publication/deep-hamiltonian-networks-based-on-symplectic-integrators)). Hamiltonian / Lagrangian views of learning itself: "Physics of Learning: a Lagrangian perspective" ([arXiv 2509.21049](https://arxiv.org/pdf/2509.21049)); Noether's learning dynamics ([arXiv 2105.02716](https://arxiv.org/pdf/2105.02716)).
- **Finance:** the only peer-style item I found is J. Marín, "Hamiltonian Neural Networks for Robust Out-of-Time Credit Scoring" (2024/25, [arXiv 2410.10182](https://arxiv.org/abs/2410.10182)): a Hamiltonian-inspired optimizer and loss on Freddie Mac loans, with a claim of more stable out-of-time AUC. It is not a price series. The Numerai forum has an informal "Lagrangian dynamics" risk loss ([forum](https://forum.numer.ai/t/teach-your-neural-net-to-minimize-risk-with-lagrangian-dynamics/2219)), not peer-reviewed.
- **Results on prices:** none found.
- **Gap and assessment:** an HNN conserves energy. Returns are dissipative and close to a martingale, so there is no conserved quantity to learn. The structurally right variants would be dissipative or metriplectic (port-Hamiltonian, GENERIC), but I found nothing on financial series. **I would not prioritise this:** the prior is low, and the project's own physics terms (T-perp, Casimir, HD, IFE, vacuum) already failed for direction.
- **Confidence:** medium-high.

## 3. Thermodynamic, free-energy, maximum-entropy and entropy-production objectives

- **What exists:**
  - Max-entropy / free-energy views of learning: Vanchurin, "Towards a theory of machine learning" ([arXiv 2004.09280](https://arxiv.org/abs/2004.09280)).
  - PIDL for entropy prediction, including S&P 500 daily returns: Sahoo & Patra, May 2026 ([alphaxiv 2606.01179](https://www.alphaxiv.org/abs/2606.01179.md)). It predicts the KDE entropy of daily returns with MSE 3.2e-3. **This is a weak paper: no economic test.**
  - Entropy of intraday BTC returns as a daily VaR predictor: "Using High-Frequency Entropy to Forecast Bitcoin's Daily Value at Risk" ([PMC7514585](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7514585/)).
  - "Price predictability at ultra-high frequency: entropy-based randomness test" ([arXiv 2312.16637](https://arxiv.org/pdf/2312.16637)).
- **The genuinely interesting physics: stochastic thermodynamics.**
  - **NEEP:** Kim, Bae, Lee, Jeong, "Learning Entropy Production via Neural Networks", PRL 125, 140604 (2020) ([arXiv 2003.04166](https://arxiv.org/abs/2003.04166)). Its objective trains a network to estimate entropy production from trajectories, which is an arrow-of-time / time-irreversibility estimator.
  - **Financial irreversibility:** Flanagan & Lacasa, "Irreversibility of financial time series: a graph-theoretical approach" (2016, [ar5iv 1601.01980](https://ar5iv.org/html/1601.01980)) found that irreversibility varies over time and relates to market state. It used daily data and visibility graphs, not a neural estimator.
- **Gap (open, as far as I found):**
  - No one has applied NEEP or another learned entropy-production estimator to 1-minute crypto.
  - No one has tested whether local irreversibility marks **when** direction is predictable, or whether its decline from 2018 to 2025 tracks the AUC decay already observed (0.60 to 0.52).
  - Under detailed balance (a reversible process) no direction edge is possible from the path alone. Irreversibility is therefore a physically motivated **upper-bound diagnostic and gate**, not a loss trick.
- **Confidence:** medium. The physics is solid. Whether it helps prediction is untested.

## 4. Optimal transport: Wasserstein, Sinkhorn, DTW-type losses

- **What exists:**
  - Shape and time losses: Soft-DTW (Cuturi & Blondel, ICML 2017) and DILATE (Le Guen & Thome, NeurIPS 2019). Both are from memory and were not re-verified this session.
  - Feature-aligned N-BEATS with a Sinkhorn divergence for domain generalisation (ICLR 2024, [arXiv 2305.15196](https://arxiv.org/abs/2305.15196v1)).
  - A. Chernov, "Fine-Tuning a Time Series Foundation Model with Wasserstein Loss" (NeurIPS 2024 TS workshop, [arXiv 2409.15367](https://arxiv.org/pdf/2409.15367)). It uses a W1 loss over **discretised value bins** instead of cross-entropy.
  - Learning with a Wasserstein loss over ordinal labels (Frogner et al., NeurIPS 2015; from memory).
- **Finance:**
  - Sig-Wasserstein GANs: Liao, Ni, Szpruch, Wiese, Sabate-Vidales, Xiao, *Mathematical Finance* 2023 ([arXiv 2006.05421](https://arxiv.org/abs/2006.05421v2)). Positive on generation and conditional forecasting realism, mostly with SPX/DJI-type data.
  - Quant GANs (Wiese et al., 2020; from memory).
- **Results:** these losses help with generation realism and with shape on smooth series. **No direction evidence.** DILATE-type losses assume a smooth target, and 1-minute returns are not smooth.
- **Gap:**
  - A cost-aware W1 loss over **binned forward returns** (10/15/20 minutes, bins in volatility units) has not been tested on HF crypto. It penalises predicting "+2σ" when the outcome is "−2σ" more than predicting "0". It is cheap. Expect a gain in density and CRPS, not in AUC.
  - Sinkhorn alignment across market regimes (the Feature-aligned N-BEATS idea) is aimed directly at the **non-stationarity** that killed the 2018-20 edge. This is untested in crypto.
- **Confidence:** medium.

## 5. Proper scoring rules beyond CRPS

| Rule | Key refs | Finance / HF use | Gap for us |
|---|---|---|---|
| Energy score (multivariate CRPS) | Gneiting & Raftery, JASA 2007; Pacchiardi et al., "Probabilistic forecasting with generative networks via scoring rule minimization" ([arXiv 2112.08217](https://arxiv.org/pdf/2112.08217)); engression, Shen & Meinshausen ([arXiv 2307.00835](https://arxiv.org/html/2307.00835v3)); energy-score-guided neural GMM ([arXiv 2603.27672](https://arxiv.org/pdf/2603.27672)) | Weather and electricity prices; the GMM paper includes financial series (daily) | A generative head trained with the energy score on the **joint** (r10, r15, r20) vector. Untested for BTC 1-minute. Engression's claimed extrapolation ability is relevant to regime shift. |
| Variogram score | Scheuerer & Hamill, MWR 2015 (from memory); electricity use: [arXiv 2311.14033](https://arxiv.org/pdf/2311.14033) | Evaluation mostly | Targets the dependence across horizons. Usable as a loss; not used in finance that I found. |
| MVG-CRPS | [arXiv 2410.09133](https://arxiv.org/pdf/2410.09133) | General multivariate forecasting | A robust Gaussian-copula alternative to the NLL. |
| **Signature-kernel scores** | Issa, Horvath, Lemercier, Salvi, NeurIPS 2023 ([arXiv 2305.16274](https://arxiv.org/abs/2305.16274)); Sig-MMD generative model ([arXiv 2407.19848](https://arxiv.org/html/2407.19848v3)); Sig-MMD / censored Sig-MMD metrics ([arXiv 2602.10182](https://arxiv.org/pdf/2602.10182)); pySigLib on GPU ([arXiv 2509.10613](https://arxiv.org/html/2509.10613v1)) | **Yes:** rough-vol simulation, **conditional probabilistic forecasts of real FX pairs**, LOB generation. Positive against GAN training. | The strongest "geometric loss" candidate: strictly proper on path space. Not applied to 1-minute crypto **conditional** forecasting against a HAR baseline. Cost: PDE-solver kernels; pySigLib exists for GPU, but TF 2.10 integration is work. |
| Log score on mixtures / MDN; Lévy-jump likelihoods | Wang & Rachev, "Neural Lévy SDE for state-dependent risk and density forecasting" (2025, [arXiv 2509.01041](https://arxiv.org/abs/2509.01041)): jumps, permutation-entropy features, quasi-ML with bipower weighting | Equities; better calibration than GARCH | A jump-diffusion head for 1-minute BTC against HAR: feasible and close to the volatility edge the project actually has. |
| Quantile / interval / **joint VaR-ES (Fissler–Ziegel)** | Fissler & Ziegel, Ann. Stat. 2016 (from memory); crypto FZ empirics ([IDEAS](https://ideas.repec.org/a/afj/journl/v27y2025i1p53-69.html)); extreme ES with neural networks on crypto ([HAL](https://hal.univ-grenoble-alpes.fr/hal-04347859v2)) | Daily crypto VaR/ES | **Intraday (10-20 minute) neural FZ against HAR-quantile is basically unpublished.** It fits the owner's separate risk-management branch (D-063). |
| Asymmetric scoring | [arXiv 2006.11265](https://arxiv.org/pdf/2006.11265) | Density forecasts | Minor. |

**Confidence:** high on what exists. Medium on whether these would add anything beyond HAR: the gain HAR leaves on the table is small (about +0.006 Spearman measured).

## 6. Martingale and no-arbitrage constraints as losses

- **What exists:**
  - Arbitrage-free neural-SDE market models: Cohen, Reisinger, Wang, *Applied Math. Finance* 2023 ([arXiv 2105.11053](https://arxiv.org/abs/2105.11053v2)), for option books.
  - HJM-constrained neural filters with an arbitrage-error regulariser: Gao & Hyndman, 2025 ([arXiv 2511.17892](https://arxiv.org/abs/2511.17892)). Positive: the largest gains are at short maturities, 5 days ahead.
  - Deep hedging (Buehler et al., 2019; from memory).
- **Why it does not transfer directly:** a spot price under the physical measure is not required to be a martingale. "No-arbitrage" places no constraint on the forecast itself.
- **What does transfer: the martingale property of rational forecasts.**
  - Augenblick & Rabin, "Belief Movement, Uncertainty Reduction, and Rational Updating", QJE 2021 ([pdf](https://faculty.haas.berkeley.edu/ned/AugenblickRabin_MovementUncertainty.pdf)). The expected squared belief movement must equal the expected reduction in uncertainty. Excess movement indicates noise or overreaction in the forecaster.
  - Forecast-stability metrics for multi-horizon forecasts: the "AC score" ([arXiv 2601.10863](https://arxiv.org/abs/2601.10863v2)).
  - Coherent temporal hierarchies: Rangapuram et al., AISTATS 2023 ([mlanthology](https://mlanthology.org/aistats/2023/rangapuram2023aistats-coherent)).
- **Gap (looks open):**
  - **(i)** Applying the Augenblick–Rabin test to a neural P(up) sequence. It is a diagnostic for overreaction to noise, and the project's calibrated P(up) is already worse than a constant 0.5 (NT-176).
  - **(ii)** A **tower-property / temporal-difference consistency loss** across the 10/15/20-minute heads: the model's 20-minute mean ≈ its 10-minute mean + E_t[its own 10-minute forecast made at t+10].
  - I found no financial-forecasting paper using either.
- **Confidence:** medium-high that this is open. Low-medium that it adds skill: it regularises noise and does not create information.

## 7. Entropy regularisation, information bottleneck, predictive information, SSL losses

- **What exists:**
  - Predictive information: Bialek, Nemenman, Tishby, "Predictability, complexity and learning" ([physics/0007070](https://arxiv.org/pdf/physics/0007070)).
  - Past-future information bottleneck (Creutzig, Globerson, Tishby, PRE 2009; from memory).
  - CPIC (Meng et al., 2022, [arXiv 2203.02051](https://arxiv.org/pdf/2203.02051)).
  - Conditional-MI contrastive loss for financial forecasting ([arXiv 2002.07638](https://arxiv.org/pdf/2002.07638)).
  - ReLaMix, an IB-style bottleneck for delay-robust financial forecasting ([arXiv 2603.20869](https://arxiv.org/pdf/2603.20869)).
- **JEPA:**
  - TS-JEPA, "Joint Embeddings Go Temporal" (NeurIPS-W 2024, [arXiv 2509.25449](https://arxiv.org/abs/2509.25449v1)). Gains are on ETT/Electricity, not finance.
  - LaT-PFN (JEPA plus PFN).
  - **LeJEPA / SIGReg:** Balestriero & LeCun, Nov 2025 ([arXiv 2511.08544](https://arxiv.org/abs/2511.08544), [code](https://github.com/rbalestr-lab/lejepa)). It replaces EMA teachers and stop-gradient with an isotropic-Gaussian regulariser on the embeddings, matched through characteristic functions along random 1-D projections. They prove it is the optimal embedding distribution for downstream risk.
- **Finance:** IB variants report gains only against weak baselines. I found no JEPA on HF crypto apart from this project's own negative result.
- **Gap:**
  - **(a)** Use an InfoNCE / CPIC bound on I(past window; future 10-20 minutes) as a **ceiling diagnostic**: is there predictive information beyond what tb7 captures, by year? If the bound collapses to tb7's level, the hunt for an architecture can stop with evidence.
  - **(b)** SIGReg is a cheap swap into the existing JEPA. The prior is low because JEPA already failed. It is worth one screen run only if collapse or anisotropy was seen in the old JEPA embeddings.
- **Confidence:** medium.

## 8. Adjacent, non-physics loss work worth knowing

- **FreDF** (Wang et al., ICLR 2025, [arXiv 2402.02399](https://arxiv.org/abs/2402.02399v2)). Direct multi-step losses are **biased when labels are autocorrelated**, and the 10/15/20-minute forward returns overlap heavily. A frequency-domain or decorrelated label loss is untested in finance. It is a cheap check on the multi-horizon heads.
- **Signal-to-decision losses:** Sharpe-ratio losses (Lim, Zohren, Roberts, 2019; from memory). The triple-barrier result already shows that edges vanish under realistic fills, so this has a low prior.
- **Diffusion / score-matching forecasters:** heteroskedastic-noise score models ([arXiv 2507.09916](https://arxiv.org/pdf/2507.09916v3)), the diffusion denoiser for financial series ([arXiv 2409.02138](https://arxiv.org/pdf/2409.02138)), and conditional diffusion for LOB forecasting (ICML 2026, [listing](https://icml.cc/virtual/2026/74197)). They capture stylised facts. I found no evidence of a direction edge.
- **HF BTC baselines:** Barez, Bilokon, Gervais, Lisitsyn (2023), Transformers vs LSTM on BTC-USDT HF log returns ([PwC](https://www.paperswithcode.com/author/nikita-lisitsyn)). Their protocols mostly lack next-open fills and decay-over-time analysis. Their positive claims probably include the bid-ask bounce this project found to be about half of the 2018-20 edge. This is my inference, labelled as such.

## Candidate hypotheses for a fresh experiment (unpublished as far as I found; all fit OHLCV only, D-050)

Ranked by (novelty × chance of an informative answer) / cost. "Pre-registered" means the criteria are fixed before the run.

| # | Hypothesis | Physics/geometry angle | Falsifiable test (existing infrastructure) | Expected | Cost |
|---|---|---|---|---|---|
| **H1** | A NEEP-style entropy-production estimate on 1-minute windows marks the periods when direction is predictable; its yearly mean falls 2018→2025 in step with the AUC decay. | Stochastic thermodynamics, arrow of time | Train NEEP on forward vs time-reversed 60-bar windows on train years. Spearman of local EP with \|tb7 AUC excess\| per day; tb7 AUC in the top vs bottom EP quintile on dev folds. | Informative either way. A gate might restore part of the edge in sub-periods. | Low (CPU / small GPU) |
| **H2** | An upper bound on the predictive information (InfoNCE/CPIC) between the 60-minute past and the 10-20 minute future is no larger than what tb7 already captures. | Information theory, predictive information | Bound per year and per bar size, compared with the MI implied by tb7's log loss. | A ceiling: tells whether any architecture search is worth continuing. | Low |
| **H3** | A tower-property / TD consistency loss across horizons plus the Augenblick–Rabin excess-movement penalty makes P(up) less noisy and improves the Brier score (fixes NT-176-type failures) without lowering AUC. | Martingale property of rational forecasts | Brier and log loss on dev folds against the current heads; the excess-movement statistic before and after. | Calibration gain; no AUC gain. | Low |
| **H4** | A Chapman–Kolmogorov coherence loss between the 10/15/20-minute predicted densities (variance additivity plus a convolution check) improves CRPS and interval coverage at 15/20 minutes. | Fokker–Planck / semigroup consistency | CRPSS against HAR-based densities, paired over 5 folds (D-025/D-046). | Small CRPS gain. | Low-medium |
| **H5** | An energy-score or signature-kernel-score head on the joint 10/15/20-minute path beats a marginal-CRPS / NLL head and HAR on joint-path scores and tail coverage. | Strictly proper scores on path space (rough paths) | Energy, variogram and Sig-MMD scores against HAR plus a Gaussian copula. | Possible gain in the risk branch. | Medium (signature kernels in TF 2.10 are work) |
| **H6** | A neural Fissler–Ziegel VaR/ES loss (10-20 minutes, 1% / 2.5%) beats HAR-quantile and GARCH-t on FZ loss and Christoffersen / Acerbi tests. | Elicitability (a geometric property of the loss) | The test fold is spent for direction only, but ES needs its own pre-registration. | Most likely to give a practical result (drawdown control). | Low-medium |
| H7 | A cost-aware W1 loss over volatility-scaled return bins beats cross-entropy / NLL on CRPS. | Optimal transport | Paired CRPS on 5 folds. | Small. | Low |
| H8 | SIGReg instead of the EMA teacher in the existing JEPA. | Isotropic-Gaussian embedding geometry | Linear-probe AUC and HAR-residual Spearman. | Probably null (JEPA already null). | Low |

**Not recommended:** HNN / LNN / symplectic losses for prices, which have no conserved quantity; DILATE / soft-DTW on 1-minute returns, whose target is not smooth; PINN "price PDEs".

## Confidence and limits of this sweep

- Searches were web-level, about 30 queries; I fetched 4 abstracts and pages in full. Items marked "from memory" are standard references that I did not re-verify this session.
- 2025-2026 arXiv items (2509.x, 2511.x, 2601.x, 2602.x, 2603.x, 2606.x) were confirmed only at the abstract or search-snippet level. Their reported results are the authors' claims and were not checked.
- "Unexplored" means that targeted searches found nothing. It does not prove that nothing exists, especially in practitioner or industry work.

## geometry_arch
# Geometric logic and original architectures for HF crypto direction and volatility: literature sweep 2022-2026

## Bottom line

- **I found no published, credible result on high-frequency financial data where any of these architectures beats a strong linear or HAR baseline by a margin that survives trading.** The serious HF studies are either negative or show tiny gains:
  - Ballarin et al. 2025, echo state network (ESN) on 10-minute stock returns: MSFE 0.88% lower than ridge, out-of-sample R² still negative.
  - Briola et al. 2025: limit-order-book (LOB) deep learning works only on large-tick stocks.
  - Rahimikia, Ni and Wang 2025: generic foundation models do poorly on financial returns.
  - Saly-Kaufmann et al. 2026, daily futures: Mamba does not lead; VSN+LSTM has the best Sharpe.

  This agrees with the tactical session's H5-H65 findings.
- **Papers that claim large gains (MambaStock, NODE "MAE −76%", ESN on daily BTC) predict the price level against a naive baseline or report RMSE in dollars.** That design inflates the numbers. Treat them as red flags, not as evidence.
- **Where real untested room remains, it is in losses and targets, not in encoders:**
  - (a) Proper scoring rules on path space (signature-kernel scores) and the Fisher-Rao geometry of the predictive distribution, applied to the distribution of the future path. This is wider than the variance alone.
  - (b) Closed-form, fast re-fitting heads against non-stationarity: next-generation reservoir computing (NG-RC), random features with rolling ridge or recursive least squares, test-time adaptation. Nagel's 2025 critique is a hazard here.
  - (c) Scale-equivariance, using the self-similarity of volatility across bar sizes.
- **Methodological constraint (my reading, not from the literature):** the direction held-out period 2024-07..2025-09 is spent. A new direction hypothesis has no clean evaluation until data after 2025-09 exists (open owner question 1). So the first fresh experiments should be about volatility, path shape or calibration, where the held-out period has not been used for direction.

## Per-direction table

Confidence is in the summary of the published evidence, not in the effect.

| # | Direction | Published on finance / HF (with result) | What is open | TF 2.10 / 12 GB | Prior for us | Confidence |
|---|---|---|---|---|---|---|
| 1 | **Signature kernels, signature-kernel scoring rules** | MMD-signature generator on S&P 500, beats a GAN on stylised facts ([2407.19848](https://arxiv.org/abs/2407.19848v3)). Non-adversarial Neural SDEs trained with signature-kernel scores (NeurIPS 2023, [Issa et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract.html)). A strictly proper signature-kernel score for path forecasts (weather, [2510.19110](https://arxiv.org/abs/2510.19110)). Signature-kernel tail-forecast metrics, ICML 2026 ([link](https://icml.cc/virtual/2026/71580)). | Nobody trains a **conditional HF price-path forecaster** with a path-space proper score and checks whether path shape (high before low, sign of the path's area, intra-horizon skew) is predictable beyond variance. | Truncated signatures up to level 3-4 of a 2-3-channel path are cheap tensor products, easy in TF. The PDE kernel (Goursat solver) is a double loop: on a 20-step path, feasible with `tf.while_loop`. [pySigLib](https://arxiv.org/html/2509.10613v1) is PyTorch/CPU only. | **Medium** (new target, not a new encoder) | medium |
| 2 | **Log-signatures, signature regression, expected signature** | Signature OLS/LASSO/logistic with minimax-optimal rates; FX realised-volatility forecasting improves over handcrafted features ([Horvath et al. 2607.17865](https://arxiv.org/abs/2607.17865), abstract only). Rough-volatility signature hybrids ([2508.07151](https://arxiv.org/abs/2508.07151)). Signature Trading: strategy linear in the signature, with drawdown control ([2308.15135](https://arxiv.org/abs/2308.15135v2)). Signature-MMD online regime detection ([Horvath and Issa 2306.15835](https://arxiv.org/abs/2306.15835)). Crypto: signatures of BTC/ETH + basis + funding on hourly data for volatility classification ([nhsjs 2026](https://nhsjs.com/wp-content/uploads/2026/08/Path-Signatures-for-Regime-Detection-in-Cryptocurrency-Markets-A-Rough-Path-Framework-Using-Spot-Perpetual-Basis-and-Funding-Rates.pdf); student-level, needs data D-050 forbids). | Signature regression for **intraday crypto RV against HAR** is not published as far as I found. Signature Trading has not been tested on 1-minute crypto with next-open fills. Level-2 Lévy areas were already negative in H5-H65. | Trivial (CPU, NumPy/TF) | Low-medium for direction (Lévy areas already failed); **medium for RV against HAR** | medium |
| 3 | **Path development (Lie-group developments)** | DevNet, finite-dimensional Lie-group (SO(n), Sp(n), U(n)) development layer, TMLR 2024; beats signature features on irregular series and in compactness ([2204.00740](https://arxiv.org/abs/2204.00740), [code](https://github.com/PDevNet/DevNet.git)). Path-Development GAN on financial generative models (Lou, Li, Ni, NeurIPS 2023). No direction or HF forecasting results. | A development layer as a **learned non-commutative "indicator"**: matrix exponentials of increments, a product along the path. That fits VISION's learned indicators. Never tested on HF crypto. | Feasible: `tf.linalg.expm` on small matrices (n=4-8) times 60 steps, ordered by a product scan. The gradient through expm exists in TF 2.10. | Low-medium (an order-information signal like Lévy areas, which were already negative) | medium |
| 4 | **Information geometry of predictive distributions** | InfoGeo Loss, ICML 2026: Fisher metric + Bregman + natural-gradient weighting, −6.8% MSE on 7 general datasets, **no finance** ([Fang et al.](https://proceedings.mlr.press/v306/fang26m.html)). NGBoost natural gradient ([1910.03225](https://arxiv.org/pdf/1910.03225)). Scoring rules for paths and the divergences they induce ([2111.06314](https://arxiv.org/abs/2111.06314)). | Not tested on HF finance. **Original-hypothesis lead:** the Gaussian family (μ, σ) under the Fisher metric **is the hyperbolic plane**, so the Fisher-Rao distance is closed-form and scale-invariant. A loss on the Fisher-Rao (hyperbolic) distance between predicted and realised distributions (or natural-gradient updates of the heads) is untested on volatility heads. "Fisher-Rao path" was tested in H5-H65 as a feature, not as a loss. | Trivial (a closed-form loss) | Low-medium (calibration and CRPS gain likely, direction gain unlikely) | medium on the maths, low on the effect |
| 5 | **Hyperbolic / Lie-group equivariant nets** | Finance: only cross-sectional hierarchies of stocks, Poincaré hypergraphs on S&P 500 ([THINK](https://dclibrary.mbzuai.ac.ae/nlpfp/26); [HIT](https://scholar.hit.edu.cn/en/publications/hypergraph-convolutional-stock-price-prediction-model-based-on-hy/)). General GL(n)/SO(n)-equivariant architectures ([2510.22984](https://arxiv.org/abs/2510.22984v2)). | A single asset has no hierarchy, so hyperbolic embeddings have nothing to work on. Only item 4 (the hyperbolic geometry of the Gaussian family) carries over. | Feasible but pointless as an encoder | **Low** | medium |
| 6 | **Scale-equivariance (time and amplitude)** | Weight-tied dilated convolutions are exactly equivariant to dyadic time rescaling, with Hurst-FiLM ([2605.17582](https://arxiv.org/abs/2605.17582), synthetic and self-similar series). Wavelet networks, TMLR 2024 ([2006.05259](https://arxiv.org/abs/2006.05259)). FTS-Diffusion, ICLR 2024 (scale-invariant patterns in finance, generative). | Never tested: **one network trained jointly on all bar sizes 1m..1h with tied weights**, as an inductive bias plus data augmentation by rescaling. The volatility target is self-similar (rough vol). Direction decayed at every bar size, so for direction it only pools noise. | Easy: Conv1D with shared kernels | Low for direction; **medium for volatility** | medium |
| 7 | **State-space models (S4/S5/Mamba)** | MambaStock ([2402.18959](https://arxiv.org/abs/2402.18959); price level, weak design). Large benchmark, daily futures 2010-25: Mamba not on top, VSN+LSTM best Sharpe, xLSTM most robust to costs ([Saly-Kaufmann et al. 2603.01820](https://arxiv.org/abs/2603.01820)). I found no credible minute-level LOB/crypto results. | Long context (days of minute bars), where an SSM is cheap. But H5-H65 found that long hourly context does not help. | S4D/S5 diagonal: a sequential scan or FFT convolution in TF works. Mamba's selective scan has no fused kernel, so it is slow but feasible at L ≤ 4k. | **Low** | medium-high |
| 8 | **Neural ODE / SDE / CDE / rough DE** | Neural RDE (log-signatures over sub-intervals), ICML 2021, up to 17k steps ([2009.08295](https://arxiv.org/abs/2009.08295)). Neural SDE as infinite-dimensional GANs ([Kidger et al.](https://www.maths.ox.ac.uk/system/files/attachments/2021-05%20Neural%20SDEs%20as%20Infinite-Dimensional%20GANs%20by%20Patrick%20Kidger%2C%20James%20Foster%2C%20Xuechen%20Li%2C%20Harald%20Oberhauser%20%26%20Terry%20Lyons.pdf)). Bayesian neural SDE calibration ([2409.06551](https://arxiv.org/abs/2409.06551)). Forecasting claims (NODE −76% MAE, ODE-LSTM +41.8% excess return, [AIMS](https://www.aimspress.com/article/doi/10.3934/math.2023712)) look methodologically doubtful. | A Neural RDE on 1-minute crypto is effectively a GRU on log-signature features of sub-windows; that should not differ from what was tested. Neural SDEs are generative and calibration tools, not a direction signal. | No torchcde/diffrax in TF. A fixed-step Euler/RK4 is easy to write by hand. | **Low** | medium |
| 9 | **Koopman operator learning** | Koopa (non-stationarity, general benchmarks, [2305.18803](https://arxiv.org/abs/2305.18803)). KNF for distribution shift ([2210.03675](https://arxiv.org/abs/2210.03675)). Finance: neural time-varying Markov operators, entropy of the transition rows correlates with RV at r = −0.62 ([2605.04690](https://arxiv.org/abs/2605.04690)). DMD for trading, old and weak ([1508.04487](https://arxiv.org/abs/1508.04487)). | Spectral, operator-based regime description, useful as an **abstention/regime gate**. Linear dynamics in a lifted space are, for returns, close to the AR baseline. | Easy (a linear layer on encoder features + eigen regularisation) | Low for direction; low-medium as a regime feature | medium |
| 10 | **Reservoir computing / ESN / NG-RC** | **The closest relevant result:** ESN on 500 stocks, 10-minute bars, 2013: MSFE −0.88% at 10 minutes against ridge (DM p < 0.01), R² from −0.0766 to −0.0675, gain falls to 0 by end of day, no trading results ([Ballarin, Capra, Dellaportas 2504.19623](https://arxiv.org/abs/2504.19623)). ESN on daily BTC: RMSE in dollars against naive, likely a weak design ([2508.05416](https://arxiv.org/abs/2508.05416)). NG-RC = nonlinear VAR, no random matrices (Nature Comm. 2021, [2106.07688](https://arxiv.org/abs/2106.07688)). Quantum RC for RV ([2505.13933](https://arxiv.org/abs/2505.13933v1)). | **Hazard:** random features with far more features than samples and short windows reduce to recency-weighted, volatility-timed momentum ([Nagel 2025, NBER w34104](https://www.nber.org/papers/w34104); vs [Kelly, Malamud, Zhou](https://www.nber.org/system/files/working_papers/w30217/w30217.pdf)). Open question: can a closed-form readout, re-fitted daily on the last N days (RLS), slow the AUC decay? | Trivial (NumPy ridge on CPU) | **Medium as a test of the "drift" hypothesis** (cheap, decisive) | high on the ESN numbers |
| 11 | **LTC / CfC (liquid networks)** | KOSPI daily with technical indicators (Korean journal); natural gas daily ([2604.24788](https://arxiv.org/abs/2604.24788), abstract gives no baseline numbers). CfC: 1-5 orders of magnitude faster than ODE solvers ([2106.13898](https://arxiv.org/abs/2106.13898)). | I found no credible HF results. | The `ncps` package has a tf.keras version (my recollection, not checked for TF 2.10). CfC is a few lines by hand. | **Low** (another RNN cell) | medium |
| 12 | **KAN** | Fair comparison on 128 datasets: KAN ≈ MLP, and the output is driven by the base (SiLU) component, not the B-splines ([2408.07314](https://arxiv.org/abs/2408.07314)). Temporal KAN transformer on Binance notional volume ([TKAT](https://www.themoonlight.io/es/review/a-temporal-kolmogorov-arnold-transformer-for-time-series-forecasting)). | Possibly an **interpretable learned transform of an indicator** (an RSI curve shape, for example) for the VISION product. No gain expected. | Easy (B-splines in TF) | **Low** for skill; low-medium as an interpretability tool | medium-high |
| 13 | **Hypernetworks** | HyperGPA (NeurIPS 2023 workshop, temporal drift), HN-MVTS (AAAI 2026, [2511.08340](https://arxiv.org/abs/2511.08340)), Hyper-Trees ([2405.07836](https://arxiv.org/abs/2405.07836)). Nothing on HF finance. | Weights of the tb7 head generated by a "regime state" (RV, trend, time). This is a regime-conditional logistic. It probably reduces to interactions that GBDT/MLP already tested. | Easy | Low-medium | medium |
| 14 | **Test-time training / adaptation** | Test-time adaptation (TTA) on SPY/QQQ/EURUSD daily: **updating BatchNorm statistics is a robust default**, entropy minimisation for direction ([2602.00073](https://arxiv.org/abs/2602.00073)). TAFAS (AAAI 2025), DynaTTA (ICML 2025). TTT layers (ICML 2025, [2407.04620](https://arxiv.org/abs/2407.04620); JAX/PyTorch). | Not tested on minute-level crypto. **This is the direct test against the main enemy, non-stationarity.** | TTA: trivial. TTT-Linear: the inner gradient step on a linear state can be written in closed form in TF. | **Medium** (cheap; the expected effect is small but it is a clean hypothesis) | medium |
| 15 | **Hamiltonian / symplectic** | No finance applications found. SympNets, HNN overviews ([2408.09821](https://arxiv.org/abs/2408.09821), [2106.11753](https://arxiv.org/abs/2106.11753v2)) assume conservation or dissipation of energy. | Log-price is a martingale-like process with no conserved quantity. The prior is that it does not fit. | Easy | **Very low** | medium (no evidence found; that is not proof of absence) |
| 16 | **JEPA** | TS-JEPA ≈ SOTA on ETT/Electricity ([2509.25449](https://arxiv.org/abs/2509.25449v1)). MTS-JEPA for anomalies ([2602.04643](https://arxiv.org/abs/2602.04643)). Nothing on finance. | Already tested in H5-H65 (negative). | - | Closed | high |
| 17 | **TS foundation models (context)** | Off-the-shelf TSFMs are poor zero-shot and fine-tuned on financial returns; pre-training from scratch on finance helps ([Rahimikia, Ni, Wang 2511.18578](https://arxiv.org/abs/2511.18578)). 2 of 10 tasks significant under DM ([Noguer i Alonso 2606.27100](https://arxiv.org/abs/2606.27100)). | - | PyTorch, not TF | Low | medium-high |

**Context from LOB studies.**
- Deep-learning predictability depends on microstructure. Large-tick stocks reach MCC 0.29; small-tick stocks only 0.11, and about 0.01 at long horizons. Accuracy does not translate into executable trades ([Briola, Bartolucci, Aste 2025, Quant. Finance](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)).
- LOBCAST: all 15 models drop sharply on new data ([2308.01915](https://arxiv.org/abs/2308.01915v1)).
- BTC/USDT has a very small tick relative to its price, so it is a small-tick asset. This agrees with H5-H65's finding that the edge is weak and half of it was bid-ask bounce. That is my reading.

## Hypotheses where we would be first (ranked by information per GPU-minute)

1. **H-drift (CPU, cheap, decisive).** The AUC decay from 0.60 to 0.52 comes partly from drift in the mapping, not only from its disappearance.
   - Test: tb7 with a logistic re-fitted daily or weekly on the last N days (RLS or rolling ridge), plus NG-RC (quadratic monomials of the tb7 lags), against a static fit.
   - Guard against Nagel's illusion: report the recency-weighted momentum baseline alongside.
   - Since the held-out period is spent for direction, judge it on the 2021-2023 dev folds and wait for data after 2025-09.
   - Sources: rows 10 and 14.
2. **H-pathshape (new target, GPU in seconds).** Beyond |r|, part of the shape of the future 10-20-minute path is predictable: the order of the high and low (MFE/MAE), the sign of the Lévy area of (t, price), intra-horizon skew.
   - Trained with a signature-kernel score, or with an energy score on the level-≤3 signature, against independent Gaussian heads.
   - This is the "physics of losses" in the form of proper scoring rules on path space. Untested on HF crypto. It also bears directly on the separate risk-management branch (stops and take-profit).
   - Source: row 1.
3. **H-FisherRao (a loss, minutes).** A variance or quantile head trained on the Fisher-Rao (hyperbolic) distance of Gaussians, or with natural-gradient preconditioning, gives better calibration and CRPS than NLL at the same HAR inputs.
   - Expected effect: small. Value: a clean geometric hypothesis with a pre-registered threshold.
   - Source: row 4.
4. **H-scale (volatility).** A tied-weight, scale-equivariant convolutional net trained jointly on 1/5/15/60-minute bars beats a separate HAR at each bar size on RV Spearman.
   - Pre-registration test: more than +0.006, which is the net's margin over HAR in H5-H65.
   - Source: row 6.
5. **H-signature-RV.** Signature-LASSO on the intra-bar path (OHLC as a 4-point path, lead-lag embedding) against HAR-RV for 10-20-minute RV.
   - This repeats the FX result of Horvath et al. on crypto.
   - Source: row 2.

**Not recommended** (low prior, or already covered by H5-H65): Mamba/S4 as an encoder, Neural CDE/RDE, LTC/CfC, KAN, Hamiltonian/symplectic, hyperbolic encoders, JEPA, foundation models.

## Caveats

- For 2603.01820, 2607.17865, 2604.24788, 2605.04690 and 2511.18578 I read only the abstract or summary page, not the full text. Their numbers are as stated there.
- For 2504.19623 and PMC12315853 the figures come from the full HTML text.
- The claim "no credible HF results" for SSM, LTC and Hamiltonian is based on about 25 searches. It is not a systematic review. Confidence: medium.
- The ranking of the hypotheses and the "prior for us" column are my estimates. They are not measured.

## jepa_targets
# Literature sweep: JEPA and self-supervised learning for time series, and changing the target when direction cannot be predicted (BTC 1-minute OHLCV)

Sweep date: 2026-10-11. Sources come from web search and arXiv abstract pages. I read the abstract or HTML of most cited papers. I did not read full PDFs, so the numbers are as the abstracts state them. Confidence labels: **H** means stated in the abstract or paper I read, **M** means taken from a search summary or secondary page, **L** means my inference or recalled from memory and not checked here.

## Main finding

1. **Part A (JEPA, self-supervised learning, foundation models).** No published work shows that a JEPA, contrastive or foundation model has direction skill on high-frequency crypto returns.
   - The best return-forecasting study of foundation models, on equities, finds no statistically reliable alpha.
   - On realized volatility, foundation models beat HAR at best by a narrow margin.
   - The only JEPA-for-time-series study finds that ordinary autoregressive training beats JEPA at short horizons.
   - Your negative JEPA result (H-series) matches the literature. Repeating it is not new territory.
   - The open and testable gap is a JEPA objective whose target encoder sees the **future path's geometry**, meaning the shape of the path rather than its return. No published work tests this, and it connects to Part B.
2. **Part B (change the question).** The most promising untested direction for this project has two layers:
   - First, **forecast path functionals as residuals over the volatility-implied analytic answer.** Under a driftless diffusion, the expected range, the probability of hitting a barrier, the first-passage time and the expected maximum drawdown are closed-form functions of σ. Since HAR already captures σ, any new signal must show up as the **residual from Brownian geometry**: the shape of the coming path that σ does not explain.
   - Second, **meta-forecasting**: predict *when* the direction model or the volatility model will work.
   - The literature supports the pieces (jump clustering, the within-hour periodicity of volatility, semivariance asymmetry, "pockets" of predictability, selective forecasting). I found no study that combines them on 1-minute crypto with a Brownian-residual baseline. Confidence that this gap exists: M. Confidence that it will produce a tradeable edge: L.

---

## A. JEPA, self-supervised learning and time-series foundation models

### A1. JEPA for time series

| Work | What it shows | Confidence |
|---|---|---|
| TS-JEPA, "Joint Embeddings Go Temporal" (Ennadir, Golkar, Sarra, 2025), [arXiv 2509.25449](https://arxiv.org/html/2509.25449v1) | 1D-CNN patch encoder, EMA target encoder (m = 0.998). Beats contrastive and autoregressive training on **classification**. On **short-horizon forecasting the autoregressive model wins** (ETT-Small MSE 0.009 against 0.017). Long-horizon forecasting is better on 2 of 3 datasets. No financial data. | H |
| MTS-JEPA / SC-JEPA (He et al., 2026), [arXiv 2602.04643](https://arxiv.org/abs/2602.04643v1) | Multi-resolution JEPA with a soft codebook against collapse, used for **anomaly and regime-shift early warning**. State of the art on industrial benchmarks. Not tested on finance. | H |
| Phys-JEPA (Nie et al., 2026), [arXiv 2606.16076](https://arxiv.org/abs/2606.16076) | Splits the latent state into a "physical" and a residual part and applies physics constraints **in latent space**. Gains of about 3% MSE on weather and traffic. Not tested on finance. | H |
| LeJEPA / SIGReg (Balestriero, LeCun, 2025), [arXiv 2511.08544](https://arxiv.org/abs/2511.08544) | Proves that an isotropic Gaussian embedding minimises downstream risk. SIGReg replaces EMA and stop-gradient and has one hyperparameter. | H (the theory is for images) |
| LaT-PFN (2024), [arXiv 2405.10093](https://arxiv.org/html/2405.10093v1) | JEPA combined with a prior-fitted network for in-context forecasting. | M |

**Still open:**
- No JEPA work evaluates financial returns, volatility or path functionals.
- No work uses a target encoder over the **future path's invariants**: range, signature, drawdown, the sequence of barrier hits.
- No work runs the anomaly or early-warning variant (MTS-JEPA) on volatility bursts.

**Honest test on BTC 1-minute data:**
- **JEPA-PF.** The context encoder reads the past 60–240 bars. The target encoder reads the next 10–60 bars, **normalised by the HAR-predicted σ**, so the encoder cannot reduce to a volatility estimate. Use SIGReg against collapse.
- **Probe.** A frozen linear probe on the embedding, for Part B targets only: range/σ, barrier-first-hit side, time to barrier, drawdown/σ.
- **Baseline.** The analytic Brownian answer plus HAR.
- **Acceptance.** Pre-register a minimum CRPS or log-loss gain of the probe over the baseline on ≥ 5 judgement folds (D-046).
- **Prior: L.** Earlier JEPA trials learned nothing about direction. They may still learn path shape. That is the only form of JEPA that is new here.

### A2. Contrastive and masked-model self-supervised learning on financial data

- Contrastive learning on 1-minute Bitcoin and limit-order-book data was used to **denoise labels**, not to forecast: [arXiv 2112.10139](https://arxiv.org/pdf/2112.10139) (M).
- Contrastive predictive coding (CPC) for trading around earnings seasons: [arXiv 2409.17392](https://arxiv.org/pdf/2409.17392) (M). Contrastive asset embeddings: [arXiv 2407.18645](https://arxiv.org/pdf/2407.18645) (M).
- None of these reports an out-of-sample direction edge on high-frequency crypto after costs.

### A3. Time-series foundation models on financial data

| Work | Result | Confidence |
|---|---|---|
| Noguer i Alonso and Franklin (2026), [arXiv 2606.27100](https://arxiv.org/abs/2606.27100). TimeGPT, TimesFM-2.5, Moirai-2.0, Chronos and Chronos-2 on 5 US stocks. | Foundation models win 8 of 10 tasks, but a Diebold-Mariano test rejects only 2 asset-model pairs against the benchmark. The authors call them "not universal engines for statistically reliable alpha". | H |
| Brini (2026), [arXiv 2607.05291](https://arxiv.org/abs/2607.05291). 9 zero-shot foundation models against Log-HAR on 50 assets (VOLARE). | Only Tiny Time Mixers (TTM) beats Log-HAR consistently, "by a narrow margin". An equal-weight average of TTM and Log-HAR is the most reliable. | H |
| Goel et al. (2025), [arXiv 2505.11163](https://arxiv.org/abs/2505.11163). TimesFM for realized volatility. | Zero-shot is weak. **Incremental fine-tuning** gives significant Diebold-Mariano and Giacomini-White gains over econometric benchmarks. Assets and frequency are not in the abstract. | H (the abstract) |
| A fine-tuned TimesFM against AR(1), [open-problem summary](https://www.emergentmind.com/open-problems/timesfm-fin-vs-ar1-consistent-superiority) | "Mixed relative to a simple AR(1) model, especially in currencies and crypto." | M |
| Kronos (2025), [arXiv 2508.02739](https://arxiv.org/html/2508.02739v1). Autoregressive model on tokenised OHLCV K-lines, 12 billion records from 45 exchanges. | Claims +93% RankIC over the best foundation model and 9% lower MAE on volatility. **The pretraining corpus very likely includes Binance BTC data, so zero-shot results on our period may leak.** | H for the claims, L for the leakage risk |
| FinCast (2025), [arXiv 2508.19609](https://arxiv.org/pdf/2508.19609) | Reports crypto results at 1 minute, 1 hour and 1 day, measured by MSE and MAE only. No directional or trading evaluation was visible. | M |
| Chronos-2 on the Optiver realized-volatility data, [search summary](https://arxiv.org/pdf/2603.18107) | Negative RankIC. | M–L |
| Audrino and Chassot, "HARd to Beat" (2024), [arXiv 2406.08041](https://arxiv.org/abs/2406.08041) | On 1,455 stocks, HAR fitted with proper rolling windows beats ML on realized volatility and VIX predictors. | H |
| Zhang, Zohren et al., intraday commonality (2022), [arXiv 2202.08962](https://arxiv.org/pdf/2202.08962) | Neural networks beat HAR when they are **pooled across stocks** and use market volatility. Intraday commonality is about 74% at 65 minutes. Our setup is a single asset, so this lever is not available. | H/M |

**Still open:**
- The edge of a foundation model in volatility comes from **fine-tuning** or **ensembling with HAR**, not from zero-shot use. This matches your finding that networks add about +0.006 Spearman.
- No clean study evaluates a foundation model on post-cutoff BTC 1-minute data.

**Honest test:**
- Only a fine-tuned small model (TTM or Chronos-Bolt) against HAR on the volatility residual.
- Kronos must not be used on any period inside its pretraining window.
- Expected gain about +0.01 Spearman (estimate). That is unlikely to be worth the GPU time against D-018 and D-050.

---

## B. Changing the target or the question

### B0. The key observation: Brownian geometry is the baseline

For driftless Brownian motion with volatility σ over a horizon T:
- E[range] = √(8/π)·σ√T.
- P(hit +a before −b) = b/(a+b).
- The first-passage time to a barrier has a closed-form distribution.
- The expected maximum drawdown has a known function of σ√T (Magdon-Ismail et al.; L, recalled from memory).
- The open, high, low and close bars cut the variance of the path's interpolation to **42%** of what the open and close alone give: [arXiv 1911.05280](https://arxiv.org/pdf/1911.05280) (H).

**Consequence:** a "path target" that a model predicts well may simply be HAR's σ in disguise. **Each Part B test must score the model against "HAR σ plugged into the Brownian formula", not against a constant.** This guard is missing from most published crypto ML papers, and it is our own way to separate new information from volatility.

### B1. Full distribution and quantiles (CRPS, value-at-risk, expected shortfall)

- Deep quantile estimators and composite quantile regression networks give value-at-risk gains over linear quantile regression and GARCH. Crypto value-at-risk and expected shortfall with a monotone composite quantile regression network, [Essex repository](https://repository.essex.ac.uk/35621/1/crk2023.pdf) (M).
- Fine-tuned TimesFM for value-at-risk performs comparably to the GAS model ([arXiv 2410.11773](https://arxiv.org/html/2410.11773v1), M).
- **Still open:** almost all of this is daily data. The 10–60 minute horizon on 1-minute crypto, scored by CRPS against a **σ-scaled empirical quantile baseline**, is barely covered.
- **Test:** a quantile head scored by the pinball loss and CRPS against HAR σ times the empirical standardised quantiles from the training block. Gains appear only if the **shape** (skewness or tails) varies predictably. **Prior: M** for a small gain in the tails, **L** for usefulness in trading.

### B2. Tails, jumps and their clustering

- Bitcoin jumps cluster. They follow rises in volume, order-flow imbalance and spikes in the bid-ask spread. Scaillet, Treccani and Trevisan, Mt. Gox tick data from 2011–2013, [arXiv 1704.08175](https://arxiv.org/abs/1704.08175) (H). We have no order flow; OHLCV volume is the only proxy.
- Self-exciting (Hawkes) jumps cluster for 2.5–6 hours. Models with jump clustering cut value-at-risk and expected-shortfall loss by up to 15%, mostly at horizons under 2 hours, [arXiv 2510.21297](https://arxiv.org/html/2510.21297v1) (M: the attribution comes from a search summary).
- A modulated renewal Hawkes model of extreme mid-price drops in crypto, [UNSW](https://unsworks.unsw.edu.au/entities/publication/05a52a6e-7993-40ea-b207-0a83ffb02692) (M).
- Time-varying, asymmetric intraday tails. Bitcoin alone has a heavier **positive** tail, [Huddersfield](https://pure.hud.ac.uk/en/publications/an-application-of-autoregressive-extreme-value-theory-to-cryptocu/) (M). Scaling of extreme price changes, [arXiv 1803.08405](https://arxiv.org/pdf/1803.08405) (M).
- **Still open:** on 1-minute OHLCV, does P(|r over the next 10–30 minutes| > kσ̂_HAR) depend on anything beyond σ̂, such as the age of the last jump, volume bursts or the signed semivariance? In other words, is the *tail-shape index* predictable, not only its scale?
- **Test:** the target is the exceedance of the σ̂-normalised return, so HAR is built into the denominator. Compare a Hawkes intensity estimated on past exceedances, a logistic model on tb7 features plus jump age, and a constant base rate. Use log-loss and the Brier skill score with a block bootstrap. **Prior: M.** Clustering of exceedances after normalisation is plausible: HAR uses fixed lags, Hawkes uses a decay since the last event.

### B3. Regime and change-point prediction

- Online change-point detection inside a deep momentum network raised the Sharpe ratio by a third (daily data, 1995–2020), [arXiv 2105.13727](https://arxiv.org/abs/2105.13727v2) (H).
- HMM regime switching for Bitcoin, [AIMS 2025](https://www.aimspress.com/aimspress-data/dsfe/2025/3/PDF/DSFE-05-03-017.pdf) (M).
- Signature plus MMD regime detection uses basis and funding data, which D-050 forbids, [NHSJS](https://nhsjs.com/wp-content/uploads/2026/08/Path-Signatures-for-Regime-Detection-in-Cryptocurrency-Markets-A-Rough-Path-Framework-Using-Spot-Perpetual-Basis-and-Funding-Rates.pdf) (M).
- **Still open:** most of this work *detects* a change after it happens. Predicting a change point *before* it happens on minute data is rarely shown out of sample.
- **Test:** label future change points with an offline detector (for example PELT on the volatility level) and predict "a change within the next 60 minutes". The baseline is the hazard from the time since the last change. **Prior: L–M.** It is also easy to leak through the labelling, so the detector must use only the future window.

### B4. Path functionals: range, barrier hits, first passage, drawdown

- Range models (CARR, ACARR, CARR-MIDAS) beat GARCH on volatility, Bitcoin included (VS-ACARR), [Zayed University](https://zuscholars.zu.ac.ae/works/5837) (M).
- The idea "predict the exit, not the close" ([blog](https://www.craftedbydaniel.com/blog/predict-the-exit-not-the-close), L-quality source) matches your triple-barrier work.
- **Still open, and our best candidate:** the residual of each path functional relative to its Brownian value given σ̂_HAR. Examples:
  - range/(σ̂√T) − √(8/π);
  - P(the up-barrier is hit first) − b/(a+b), which tests whether the *geometry* of the coming path is asymmetric even when the drift is zero;
  - the ratio of the time to the first barrier to its Brownian expectation;
  - the forward drawdown divided by σ̂√T.

  These are "direction-free" questions about mean reversion, trendiness and kurtosis inside the window. I found no paper that scores 1-minute crypto against this baseline.
- **Test:**
  - Compute each functional from 1-minute OHLC: high and low give the range exactly; barrier order is ambiguous only inside a single bar. Use the Brownian-bridge correction or a 1-bar tie rule.
  - Run on judgement folds outside 2024-07..2025-09, plus one final untouched check.
  - Report the Spearman correlation and CRPS of the residual.
  - Pre-register: the effect must exceed the block-bootstrap null at n_eff = N/horizon.
- **Prior: M** that the range residual is predictable through volume, time of day and recent jumps; it is partly mechanical, from the diurnal pattern. **L** that the hit-side residual is predictable: it is close to direction in disguise.

### B5. Realized skewness and higher moments

- Realized skewness *of the cross-section* predicts weekly crypto returns, with a negative sign: Lee and Wang, JFQA 2025, [IDEAS](https://ideas.repec.org/a/cup/jfinqa/v60y2025i4p1859-1890_9.html) (H for "higher variance gives lower returns, driven by positive-jump variance"). This is a cross-section and weekly result; it does not carry over to a single asset at 10 minutes.
- At high sampling frequency, realized skewness mainly measures the jump part, [arXiv 1612.08526](https://ar5iv.labs.arxiv.org/html/1612.08526) (M).
- Signed semivariance: past **negative** semivariance predicts future volatility much more strongly. Patton and Sheppard, [Duke](https://public.econ.duke.edu/~ap172/Patton_Sheppard_Realized_Semivariance_7oct11.pdf) (H for equities). In Bitcoin, jumps contribute less variance than in equities, [HU Berlin](https://www.wiwi.hu-berlin.de/de/forschung/irtg/results/discussion-papers/abstracts-1/irtg1792dp2019-024.txt) (M).
- **Test:** check whether HAR with RS⁺/RS⁻ (a semivariance HAR) beats plain HAR at 10–60 minutes. This is cheap and runs on the CPU. **Prior: M** for a small gain. Predicting the *sign* of the next window's realized skewness is mostly jump timing (see B2).

### B6. Roughness and the Hurst exponent of volatility

- In BTC 1-minute data from 2017–2024, the roughness index of realized volatility is about 0.05–0.09, below ½ in every finite estimate. It varies with the measurement design: Pontiggia, [arXiv 2507.00575](https://arxiv.org/abs/2507.00575) (H).
- Cont and Das: apparent roughness can come from **estimation error and microstructure noise**, not from the volatility process, [arXiv 2203.13820](https://arxiv.org/abs/2203.13820) (H).
- The Hurst exponent of BTC prices moved from anti-persistent before 2013 to about 0.5 afterwards. Efficiency grows with liquidity, [arXiv 1902.09253](https://arxiv.org/pdf/1902.09253) (M). This matches your decay of AUC from 0.60 to 0.52.
- **Still open:** whether rolling local roughness or the Hurst exponent *forecasts* anything, such as the next window's range residual or the AUC of the direction model. **Prior: L.** The estimator noise at 60–240 bars is very large.

### B7. Volatility of volatility

- HARQ (realized quarticity) and HAR with jumps and structural breaks beat HAR on Bitcoin at daily horizons, [Reading](https://reading-clone.eprints-hosting.org/87932) (M).
- **Still open:** a direct forecast of vol-of-vol at intraday horizons, as an uncertainty-of-σ target for position sizing. This fits the owner's separate risk-management branch (D-063).
- **Test:** the target is |log RV_{t+1} − log RV̂_{t+1}|. Baselines are a constant and the realized quarticity. **Prior: M.**

### B8. Intraday shape of volatility

- BTC volatility has strong day-of-week, hour-of-day and **within-hour** patterns: a burst in the first minute of each hour, smaller bursts at minutes 15, 30 and 45, and every 5 minutes. There is no U-shape. The patterns **grew stronger over time**, and are linked to algorithmic trading and funding times. Hansen, Kim and Kimbrough, [arXiv 2109.12142](https://arxiv.org/abs/2109.12142) (H).
- **Still open:** whether HAR with a learned periodic multiplier (a flexible Fourier form, FFF) closes the remaining gap between networks and HAR at 10–20 minutes. This could explain the +0.006 Spearman networks add.
- **Test:** CPU only. Compare HAR, HAR × the periodic component and the network. If the periodic HAR equals the network, the gap was the clock, not deep features. **Prior: H** that this explains part of the gap. It is cheap and settles a question.

### B9. Liquidity proxies from OHLCV

- For crypto, the Corwin-Schultz and Abdi-Ranaldo estimators track the time-series variation of liquidity best, at every frequency and venue: Brauneis et al., J. Banking & Finance 2021, [IDEAS](https://ideas.repec.org/a/eee/jbfina/v124y2021ics0378426620303022.html) (H).
- **Caveat:** on 1-minute Binance BTC data the spread is about one tick, so the estimators are dominated by noise. Trading costs are 0 by D-044.
- **Use:** as a *feature* that predicts the tail and range residuals (as in Scaillet et al.: spread spikes come before jumps). **Prior: L–M.**

### B10. Predicting predictability (meta-forecasting)

- Selective forecasting by meta-learning (Inácio et al., 2026, [arXiv 2606.23448](https://arxiv.org/abs/2606.23448)) predicts the percentile of the forecast error from features of recent lags. Abstaining on predicted-hard cases "consistently improves accuracy across coverage levels" (H). It is not tested on finance.
- Crypto high-frequency predictability is "time-varying … unpredictable most of the time", [CEEJ 2025](https://ceej.wne.uw.edu.pl/?p=2168) (M).
- "Pockets of predictability" (Farmer, Schmidt and Timmermann, J. Finance 2023) found local, temporary predictability in equities (L: recalled from memory, not checked in this sweep).
- Forecastability measures (spectral predictability, Lyapunov exponent, auto mutual information by horizon), [arXiv 2601.10006](https://arxiv.org/pdf/2601.10006) (M).
- **Still open, and the strongest framing for your result:** the AUC of tb7 decayed on average, but are there *predictable windows* where its local AUC is still about 0.58?
- **Test:**
  - Train tb7 as it is.
  - On a **separate** block, train a meta-model that predicts the realized local error of tb7: rolling log-loss or hit correctness over the next N windows. Its inputs are σ̂, the vol-of-vol, the hour, the age of the last jump, the local Hurst exponent and the confidence of tb7.
  - Evaluate the AUC of tb7 by meta-score decile on judgement folds.
  - **The acceptance criterion must be fixed before the run:** for example, a top-decile AUC of at least 0.55 with a lower block-bootstrap bound above 0.52 on ≥ 5 folds.
  - Abstention multiplies the effective search space, so count every threshold tried.
- **Constraint:** 2024-07..2025-09 is spent for direction hypotheses, and this is a direction hypothesis. The verdict needs earlier judgement folds (2021–2024) placed after every fold used for a choice, and a final check only if new data arrives (owner question 1).
- **Prior: M** that the decile spread exists; it is likely tied mainly to volatility. **L** that it survives next-open fills.

---

## Ranked candidate hypotheses (our own, falsifiable, within D-050)

| # | Hypothesis | Baseline that must be beaten | Compute | Prior |
|---|---|---|---|---|
| 1 | **The Brownian-residual path geometry is predictable.** range/(σ̂√T), the drawdown ratio and the first-passage ratio depend on volume, the within-hour clock, jump age and semivariance beyond what HAR σ̂ explains. | The HAR σ̂ plugged into the closed-form Brownian functionals | CPU, then a small network | M |
| 2 | **HAR with the within-hour periodic component closes the network-over-HAR gap.** | The network's +0.006 Spearman | CPU, minutes | H that it explains part of the gap |
| 3 | **Exceedances of the σ̂-normalised tails cluster (Hawkes decay); the tail-shape index is predictable.** | A constant base rate with σ̂ scaling | CPU | M |
| 4 | **Meta-forecasting: the AUC of tb7 differs by predicted-predictability decile.** | The unconditional AUC of tb7 | CPU | M for the spread, L for trading |
| 5 | **JEPA-PF: a JEPA whose target encoder reads the σ-normalised future path, with SIGReg, evaluated by a linear probe on targets 1 and 3.** | Ridge and logistic models on the hand-made features of hypothesis 1 | GPU, screen layout, at most 2 minutes per run (D-063) | L |
| 6 | Vol-of-vol forecasting for sizing (the risk-management branch). | Constant and realized-quarticity baselines | CPU | M |

**Rules that apply to all six:**
- Score every target against the volatility-implied analytic baseline, never against a constant.
- Use n_eff = N / horizon and a block bootstrap.
- Pre-register the threshold before running.
- Hypotheses 1–3 and 6 are not direction hypotheses. The direction hold-out 2024-07..2025-09 can serve as their final check once, but the earlier HAR analysis already looked at 2025 volatility, so that check is partly contaminated. State this in the SPEC.
- Do not use Kronos or another foundation model zero-shot on any period that may be in its pretraining data.

**Gaps in this sweep:**
- Full PDFs were not read for FinCast, Goel et al., Lee–Wang and the Hawkes value-at-risk paper; their numbers come from abstracts or search summaries.
- The "pockets of predictability" and maximum-drawdown-formula references are recalled from memory, not checked.
- I found no published study of 1-minute BTC with a Brownian-residual baseline. Absence from search is not proof that none exists.

## Sources

- https://arxiv.org/html/2509.25449v1
- https://arxiv.org/abs/2602.04643v1
- https://arxiv.org/abs/2606.16076
- https://arxiv.org/abs/2511.08544
- https://arxiv.org/html/2405.10093v1
- https://arxiv.org/pdf/2112.10139
- https://arxiv.org/pdf/2409.17392
- https://arxiv.org/pdf/2407.18645
- https://arxiv.org/abs/2606.27100
- https://arxiv.org/abs/2607.05291
- https://arxiv.org/abs/2505.11163
- https://www.emergentmind.com/open-problems/timesfm-fin-vs-ar1-consistent-superiority
- https://arxiv.org/html/2410.11773v1
- https://arxiv.org/html/2508.02739v1
- https://arxiv.org/pdf/2508.19609
- https://arxiv.org/pdf/2603.18107
- https://arxiv.org/abs/2406.08041
- https://arxiv.org/pdf/2202.08962
- https://arxiv.org/pdf/1911.05280
- https://repository.essex.ac.uk/35621/1/crk2023.pdf
- https://arxiv.org/abs/1704.08175
- https://arxiv.org/html/2510.21297v1
- https://unsworks.unsw.edu.au/entities/publication/05a52a6e-7993-40ea-b207-0a83ffb02692
- https://pure.hud.ac.uk/en/publications/an-application-of-autoregressive-extreme-value-theory-to-cryptocu/
- https://arxiv.org/pdf/1803.08405
- https://arxiv.org/abs/2105.13727v2
- https://www.aimspress.com/aimspress-data/dsfe/2025/3/PDF/DSFE-05-03-017.pdf
- https://nhsjs.com/wp-content/uploads/2026/08/Path-Signatures-for-Regime-Detection-in-Cryptocurrency-Markets-A-Rough-Path-Framework-Using-Spot-Perpetual-Basis-and-Funding-Rates.pdf
- https://zuscholars.zu.ac.ae/works/5837
- https://www.craftedbydaniel.com/blog/predict-the-exit-not-the-close
- https://ideas.repec.org/a/cup/jfinqa/v60y2025i4p1859-1890_9.html
- https://ar5iv.labs.arxiv.org/html/1612.08526
- https://public.econ.duke.edu/~ap172/Patton_Sheppard_Realized_Semivariance_7oct11.pdf
- https://www.wiwi.hu-berlin.de/de/forschung/irtg/results/discussion-papers/abstracts-1/irtg1792dp2019-024.txt
- https://arxiv.org/abs/2507.00575
- https://arxiv.org/abs/2203.13820
- https://arxiv.org/pdf/1902.09253
- https://reading-clone.eprints-hosting.org/87932
- https://arxiv.org/abs/2109.12142
- https://ideas.repec.org/a/eee/jbfina/v124y2021ics0378426620303022.html
- https://arxiv.org/abs/2606.23448
- https://ceej.wne.uw.edu.pl/?p=2168
- https://arxiv.org/pdf/2601.10006