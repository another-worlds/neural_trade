# Adversarial verification of the 15 proposed hypotheses (Designer A: A1-A8; Designer B: H-1 to H-7)

Method: about 30 web searches and page fetches, plus checks against the project files in `D:/nt/nt_tactical`. "Not found" means my targeted searches found nothing. That is weak evidence of novelty, not proof.

## 0. Cross-cutting findings (read first)

**0.1 Duplicates. The 15 hypotheses are 10 distinct studies.**

| Merge | What they share |
|---|---|
| A1 = H-4 | Neural estimate of entropy production / the arrow of time |
| A2 ≈ H-1 | A sign-free trendiness target (Z2, ρ, variance ratio) |
| A3 = H-5 | TD / semigroup estimation of the realised-variance term structure |
| A5 = H-2 | Omori / Hawkes hazard of HAR-normalised exceedances |
| A8 = H-3 | JEPA with a shape-only target, using SIGReg |

Each merged pair should run as one study under one SPEC. The selection counts should be pooled for Holm.

**0.2 Protocol breach in the existing record (needs the lead or the owner).**
- `docs/TACTICAL.md:72-73` says the long file's protected span (`DATA_END_PROTECTED_DAYS`) stays protected.
- Several pre-registered looks ran to the end of the data anyway:
  - `runs/tactical/recent/wf.py:14` sets `"look": ("2024-07","2025-09")`.
  - `calendar/SPEC.md:25` reads "One look: 2024-07-01 .. 2025-09-29 (the data end)".
  - `volnet/SPEC.md` and `flow/SPEC.md` also look through 2025-09.
- So H56, H57, H61, H63 and H64 used the 64-day protected span (`core/config.py:257`: 64.0, not the 70 the designers quote). They also used the newest ~32 days, which D-040 reserves as the MVP test fold, against D-063's "never the test fold".
- The designers' new confirmation window, 2024-07-01..2025-07-20, avoids this going forward. It is correct and should be kept.

**0.3 Mandate.**
- D-063 limits the tactical session's goal to direction skill. Risk management is "a separate independent branch, not started".
- A3/H-5, A7 and partly A5/H-2 are risk-branch work. They need the owner's go-ahead; this was question 2 in the previous status message.
- The direction-adjacent tests (A1-P3, H-4(ii), the adaptation gain in H-6, and H-7's trading verdict) have no clean confirmation until data after 2025-09 exists (question 1). The 2021-01..2024-06 months were already used for selection by H56, H63 and H64. Results there are exploratory only.

**0.4 The noise numbers quoted by the designers check out.**
- Spearman SE: 0.037 per month and 0.0042 over 78 months at T = 60; 0.005 over 18 months at T = 20.
- Exceedance counts: about 88 per year and 569 in total at α = 1%, h = 60; about 29 per month and about 1,900 in total for H-2.
- H-1's decay ratio: 0.03 / 0.095 = 0.316.
- A4's cost does not check out (see A4).

**0.5 Errors found in the research inputs.**
- *jepa_targets* says "No JEPA work evaluates financial returns". This is false. Mahns, Balestriero and Assran, ICML 2026, ran a JEPA on about 500 US stocks with daily data. Its embeddings track realised volatility and correlation, with "only weak association with market direction": https://icml.cc/virtual/2026/poster/65643. This matches the project's JEPA v1-v4 results.
- *loss_physics* §3 says "under detailed balance no direction edge is possible". This is false. Designer B's correction is right: a Gaussian ARMA process is time-reversible and still linearly predictable (Weiss 1975, J. Appl. Prob. 12).
- Petersen et al. 2010 is 531 US stocks at 1 minute, 2001-02, with 219 shocks. It is not "1-minute S&P data", as H-2 says: https://ideas.repec.org/p/arx/papers/1006.1882.html

**0.6 Citations I verified.**

| Citation | Status | Link |
|---|---|---|
| NEEP, PRL 125, 140604 | correct | https://arxiv.org/abs/2003.04166 |
| LeJEPA / SIGReg (Epps-Pulley, random 1-D projections) | correct | https://github.com/rbalestr-lab/lejepa |
| De Asis et al., AAAI 2020 | correct | https://ojs.aaai.org/index.php/AAAI/article/view/5784 |
| Patton, Ziegel, Chen, J. Econometrics 211(2) 2019 | correct | https://arxiv.org/pdf/1707.05108 |
| Issa et al., NeurIPS 2023 (conditional FX forecasts) | correct | https://arxiv.org/pdf/2305.16274 |
| Nagel, NBER w34104 (2025) | correct | https://ideas.repec.org/p/nbr/nberwo/34104.html |
| Hansen-Kim-Kimbrough, JFEC 2024 (minute-0, 15/30/45 and 5-minute bursts, strengthening over time) | correct | https://quantpedia.com/periodicity-in-cryptocurrencies-recurrent-patterns-in-volatility-and-volume/ |
| Pontiggia (roughness 0.054-0.086) | values correct, but the data are Bitstamp BTC/USD, not Binance | https://arxiv.org/abs/2507.00575 |
| Pinson & Tastu 2013 | discussed under A6 | |

## 1. Per-hypothesis verdicts

### A1 + H-4: arrow of time / entropy production as a "when" gate. Verdict: survives with corrections; partly novel

**Novelty: much of it is done.**
- Morales Herrera & Salgado-García (Physica A 2023), on minute data for BTC, ETH, XRP, LTC and BCH over 2015-2022: a 91-day rolling irreversibility index. BTC moved "toward reversibility" non-monotonically. The correlation of irreversibility with inefficiency ranged from −0.21 to 0.67: https://ar5iv.labs.arxiv.org/html/2307.08612
- Vodret, Pacini, Bongiorno 2025: a gradient-boosting classifier of forward against time-reversed financial paths, with a functional decomposition of irreversibility (daily equities): https://arxiv.org/html/2407.06063v2
- Kim, Cho, Lee 2026: high-frequency crypto 2020-25, with directional irreversibility concentrated around large moves: https://papers.cool/arxiv/2608.10852
- Zumbach / time-reversal asymmetry effects in BTC are already documented by period: https://arxiv.org/pdf/2408.07653

**What remains new:** a neural estimator on 1-minute BTC from 2017 to 2025, and its use as a gate on tb7 with a σ̂-gate control.

**Contradiction.** The 2026 paper ties irreversibility to large-move events, so entropy production is likely a proxy for volatility or jumps. This supports H-4's own bet.

**Corrections:**
1. Correlate detrended or first-differenced monthly series, or use within-year deviations. Both entropy production and AUC trend down over 2018-24, so a raw Spearman ≥ 0.4 (H-4(i)) or < 0.3 (A1-P2) mostly measures the shared trend.
2. Standardise each window by a time-symmetric scalar, such as the window's own realised volatility. Per-bar trailing EWMA standardisation is itself time-asymmetric and creates fake irreversibility.
3. EP_dir = EP_tot − EP_vol is valid only as a chain-rule conditional KL. Estimate it as the paired held-out log-loss gain of the signed-input classifier over the |r|-only classifier, with a CI. Do not subtract two separately biased estimates.
4. Sign asymmetry includes the leverage effect (sign → future |r|), which is not direction. Add a "sign → future sign" control.
5. A1-P1 and H-4(iii) are sanity checks of known stylised facts, not findings.
6. P3 and (ii) are exploratory only, for the reasons in 0.3.

### A2: forecasting the martingale defect (Z2). Verdict: survives with corrections; partly novel; low prior; merge into H-1

**Novelty.**
- Serial correlation conditional on volatility and volume is classical:
  - LeBaron 1992 (autocorrelation falls when volatility is high): https://ideas.repec.org/a/ucp/jnlbus/v65y1992i2p199-219.html
  - Campbell, Grossman, Wang 1993; Llorente, Michaely, Saar, Wang 2002 (RFS): https://img.nber.org/papers/w8312
- Time-varying efficiency of BTC under the adaptive market hypothesis: https://arxiv.org/pdf/1904.09403
- Machine-learning prediction of directional-change overshoot: https://repository.essex.ac.uk/43797/
- What remains new: an out-of-sample forecast of a statistic of the future window that is mean-zero under the martingale hypothesis, on 1-minute crypto.

**Contradiction.**
- "A spot trader cannot monetise it" is false. A forecastable conditional autocorrelation becomes a conditional direction signal once the in-window returns are seen, through a regime × recent-return interaction.
- That class (boosting and L1 over tb7 plus regime and geometry features; H47, H56; H40 regime gates) is in the negative space. This lowers the prior.
- The per-window noise is large. Z2 ≈ (VR − 1)/2 minus the lag-1 term, and VR has a standard deviation of about √2 under the null. So the attainable Spearman is about 0.03-0.05 even with real regime variation.
- In 2025, lag-1 reversal is gone (H52: 0.505), and lags of 2 or more are smaller still.

**Corrections:**
- The baseline must contain the LeBaron and Campbell-Grossman-Wang terms (σ̂, volatility surprise, volume) plus past Z2. Otherwise a "shape" gain is a known effect.
- Report Z3 (lags ≥ 3).

### H-1: trendiness in its own clock, ρ = |ΔP|/range. Verdict: survives with corrections; the best candidate; partly novel

**Novelty.**
- The target is close to well-known trendiness indicators: Kaufman's efficiency ratio, the choppiness index and the Vertical Horizontal Filter.
- DC/overshoot forecasting exists (link above).
- Peters & de Vilder 2006 could not reject a time-changed Brownian motion for S&P futures in quadratic-variation time: https://ideas.repec.org/a/bes/jnlbes/v24y2006p444-454.html
- Ané & Geman 2000 (normality in a transaction clock): https://ideas.repec.org/a/bla/jfinan/v55y2000i5p2259-2284.html
- An out-of-sample *conditional* forecast of the scale-free future shape against a Brownian/HAR null on high-frequency crypto: not found.
- It is new to the project. A grep of `runs/tactical` finds no variance-ratio, efficiency-ratio, Hurst or trendiness target.

**The theory checks:** ρ is clock-invariant for a continuous time-changed Brownian motion when the clock is independent of the path.

**Contradictions:**
- BTC's inverted leverage (https://pmc.ncbi.nlm.nih.gov/articles/PMC7850481) means the clock is not independent of the path.
- Jumps break the Dambis-Dubins-Schwarz argument: a martingale with jumps has a ρ distribution that depends on jump intensity.

**Corrections:**
1. Add jump-intensity state (time since the last ≥ 4σ bar, bipower vs RV) to the **baseline**. Otherwise "shape predictability" is just jump-intensity predictability.
2. Drop log(tick/σ̂√T). The BTC/USDT tick is $0.01, far below σ√60 (tens to hundreds of dollars), so the term is a constant.
3. Define the range including C_t: max(C_t, max H) − min(C_t, min L), so that ρ ≤ 1.
4. Add the LeBaron, Campbell-Grossman-Wang and volume terms to the baseline, as for A2.
5. The look window is clean for ρ itself, but the SPEC must state that related path functionals were already scored there: H54's excursion asymmetry (2021-25) and the triple-barrier outcomes of H48, H60 and H61.
6. Pre-register the power: SE ≈ 0.0046 over 66 months at T = 60, which makes ΔSpearman ≥ +0.02 detectable.
7. Run it as one study with A2's targets (ρ, VR, Z2, Z3; Holm over everything).

### H-7: sign-free OCO bracket on the H-1 forecast. Verdict: survives only as gated on H-1; low novelty; low prior

- **Novelty:** breakouts filtered by ADX or choppiness are standard technical-analysis practice.
- **Overlap:** squeeze breakouts (H47) and the triple barrier (H48, H60, H61: edges vanish with next-open fills).
- **Corrections:**
  - Use adverse stop fills: the worse of the stop and the open, plus a slippage grid that includes a fraction of the bar's range.
  - Count every (a, b, c) tried for Holm.
  - The confirmation is clean only for this strategy class; it is partly seen through the barrier outcomes.

### A3 + H-5: TD / semigroup estimation of the RV term structure. Verdict: survives with corrections; partly novel; risk branch

**Novelty.**
- No TD or LSTD estimation of the RV term structure found in finance (one targeted search, low confidence).
- The nearest literature is direct vs iterated vs MIDAS forecasting. Ghysels et al. (ARFE 2019), on 30 assets with 5-minute data: "direct RV is preferred to iterated RV; MIDAS most precise": https://www.sfi.ch/en/publications/n-19-02-direct-versus-iterated-multi-period-volatility-forecasts-why-midas-is-king-e.-ghysels-a.-plazzi-r.-i.-valkanov-a.-r.-serrano-a.-dossani-2019.

**Contradiction.** TD sits between direct and iterated forecasting, and that evidence favours direct. Expect TD bias to dominate wherever the features are not Markov-sufficient.

**Corrections:**
1. Tower additivity holds for variance **levels**. H-5's "target log RV" with bootstrapping is invalid (Jensen's inequality). Bootstrap in levels and evaluate QLIKE on levels.
2. The purge gap must cover 2T (the bootstrap target reads features at t+T).
3. Add direct HAR, iterated HAR and MIDAS as baselines, with the rolling-window choice of "HARd to beat".
4. A3's h = 60 at a 30-day refit has n_eff ≈ 720 for about 8 parameters, so a null result is almost guaranteed. Use H-5's 240/1440-minute horizons, where n_eff is 30-180.
5. The look is contaminated by H53 and H57, as both designers say. The gain-against-window curve is the main evidence.
6. Owner go-ahead (risk branch, see 0.3).

### A4: physics terms × a "Fisher-flat" variance head. Verdict: survives with corrections as a D-003 project verdict; refuted as a geometric novelty

- **Novelty:**
  - Log or natural parameterisation of heteroscedastic heads is established: Seitzer et al., ICLR 2022 (β-NLL), https://arxiv.org/abs/2203.09168; Immer et al., NeurIPS 2023 (natural parameterisation).
  - Designer B is right: for (μ, log σ) the Fisher matrix is diag(1/σ², 2), so the natural gradient in log σ is the NLL gradient divided by 2. P1 is an optimisation fix, not a hypothesis.
- **Project check:** the code confirms the premise.
  - The variance heads are `softplus` (`models/gru_attention.py:413,452,490`).
  - The floor is `tf.maximum(var, VAR_FLOOR)`, which has zero gradient below the floor (`training/custom_model.py:403`).
- **Overlap:** H12 already ran "no physics" on a composite that includes risk ranking: +0.36 [−0.51, +1.22].
- **Feasibility is wrong:**
  - The 24 lab slices are the hc4 long blocks (`lab/cache.py`: make_hc4.WEEK, about 11.5 days of training).
  - H12 measured about 7 minutes per network run on them. So 96 runs take about 11 GPU-hours, not 3.2, and each run breaks D-063's 2-minute rule; that needs the owner's consent.
  - The screen layout cannot judge the variance head: n_eff is 5-11 (D-064).
- **Corrections:**
  - Score QLIKE only after the D-008 calibration-block rescale, or lead with Spearman. The raw head is overconfident (H11 coverage 0.31-0.45), and scale error would swamp the effect.
  - Estimate the per-slice QLIKE noise from existing runs before fixing the 0.005 threshold.

### A5 + H-2: Omori / Hawkes hazard of HAR-normalised exceedances. Verdict: survives with corrections; partly novel; modest prior

**Novelty.**

| Prior work | Data | Link |
|---|---|---|
| Omori law in BTC volatility, descriptive | BTC | https://arxiv.org/pdf/1905.03211 |
| Hawkes-POT VaR (Chavez-Demoulin, Davison, McNeil 2005) | classical | |
| Hawkes-POT with volatility covariates (Herrera & Clements, JBF 2018) | daily | https://eprints.qut.edu.au/116240 |
| ETAS early warning beats volatility models (Gresnigt, Kole, Franses, JBF 2015) | daily S&P 500 | https://papers.tinbergen.nl/14067.pdf |
| Intraday crypto Hawkes for extremes (theses) | intraday crypto | https://unsworks.unsw.edu.au/entities/publication/05a52a6e-7993-40ea-b207-0a83ffb02692 |

What remains new: out-of-sample skill over HAR + clock on HAR-normalised exceedances at the minute scale.

**Contradiction.** Weber et al. 2007 show that Omori processes occur on all scales and are equivalent to volatility memory: https://arxiv.org/abs/physics/0611093. HAR's short lags already approximate that memory, so expect a log-loss gain well below 2%.

**Corrections:**
1. The baseline must include RV at 1/5/15 minutes **and** a plain "time since last exceedance" feature. Otherwise the Omori kernel beats a strawman.
2. Compare power-law and exponential kernels, as H-2 plans.
3. Fix the threshold (4σ vs 5σ, q98) by expected event counts before any run.
4. The look is partly contaminated (H57).

### A6: energy-score path generator with a scale / shape / endpoint split. Verdict: survives only with major corrections; demoted

- **Novelty:** conditional path forecasting with signature-kernel scores on real FX already exists (Issa et al.), as do engression-based multivariate forecasters (https://www.alphaxiv.org/abs/2603.11909). Partly novel.
- **Contradiction:** Pinson & Tastu 2013 show the energy score barely discriminates dependence structure. A correlation of −0.8 forecast as +0.3 changes the score by only about 5% (https://arxiv.org/pdf/2101.12693). P1's "≥ 0.5% better than FHS" would mostly reflect scale.
- **The design is confounded:**
  - CRPS is scale-sensitive, so endpoint CRPS improves with any σ refinement (H57: volume and range features add +0.006 Spearman). P2's "endpoint within ±0.1%" is violated for reasons that have nothing to do with shape.
  - range/(σ̂√T) mixes HAR's scale error into "shape".
- **Corrections:**
  - Normalise every functional by the future window's realised QV, as in H-1 / H-3.
  - Or give the FHS baseline the same σ model as the generator.
  - Run only if H-1 / A2 shows a signal; their CPU tests already cover the marginals.

### A7: intraday FZ0 VaR/ES. Verdict: survives with corrections; partly novel; risk branch

- **Novelty:**
  - FZ joint VaR/ES on crypto exists at daily horizons: https://ideas.repec.org/a/wly/jforec/v42y2023i4p989-1007.html
  - An industry framework (Talos, May 2026) refreshes 1-day VaR/CVaR hourly with FZ0 and Diebold-Mariano tests: https://www.talos.com/insights/backtesting-risk-measures-for-digital-assets
  - What remains new: 15/60-minute horizons against HAR-FHS.
- **Feasibility:** about 570 exceedances at α = 1%. A 1% FZ0 gain is probably underpowered under a Diebold-Mariano test with Bonferroni over 4 cells.
- **Corrections:**
  - The SPEC needs a power simulation, or a move to α = 2.5% and 5% as primary.
  - The look is contaminated.
  - It depends on A5's state.
  - Owner go-ahead (risk branch).

### A8 + H-3: JEPA on a shape-only target (business time, quotient by scale × reflection), with SIGReg. Verdict: survives with corrections; partly novel; low prior

- **Novelty:** the specific target is not found elsewhere.
- **Contradiction:** JEPA on financial data learns "risk weather", with weak direction (Mahns et al. 2026), and the business-time null is classically not rejected (Peters & de Vilder; Ané & Geman). Both point to a null. Under that null the JEPA training signal for the context encoder is nearly zero.
- **Corrections:**
  1. The synthetic negative control must include the empirical intraday clock: the minute-0 burst and the 15/30/45-minute bursts (Hansen et al.). Interpolation artefacts in the QV resampling are then **predictable from the time of day**, and would pass as "shape information" on real data while the control stays clean.
  2. Add a supervised upper bound: the same encoder trained directly on ρ / Z2. If that cannot beat ridge, JEPA cannot.
  3. InfoNCE at about 0.005 nats is a small-MI regime with known estimator limits (Poole et al. 2019; McAllester & Stratos 2020). Use held-out windows, a block bootstrap, and a permuted-context null.
  4. Run in resumable 2-minute chunks (D-063).
  5. Gate it on a positive H-1 / A2 result.

### H-6: rotation or shrinkage of the tb7 coefficient vector. Verdict: survives with corrections; low novelty; useful as a closure

- **Novelty:** the time-varying efficiency of BTC is well documented (Urquhart 2016, Sensoy 2019, and others: https://reference-global.com/article/10.2478/ceej-2025-0003). The norm/angle decomposition is a minor twist.
- **Prior:** H56 (365 days beat 60) and H35 already point to shrinkage.
- **Corrections:**
  - As ‖β‖ shrinks toward its standard error, the estimated angle grows even under pure shrinkage, so "rotation" can be an artefact.
  - Test the scaling model β_t = c_t·β_ref against free β_t with a held-out likelihood ratio, against a bootstrap null of the angle under pure shrinkage.
  - Keep Nagel's momentum guard.
  - The adaptation gain is exploratory (see 0.3).

## 2. Verified ranking (CPU first, gated)

| Step | Study | Status |
|---|---|---|
| 1 | H-1 + A2 merged | CPU, clean look, partly novel |
| 2 | A5 + H-2 merged | CPU |
| 3 | A1 + H-4 | CPU / small GPU, descriptive, partly known |
| 4 | H-6 | CPU, closure |
| 5 | H-5 / A3 | Owner go-ahead (risk branch) |
| 6 | A7 | Owner go-ahead (risk branch) + power simulation |
| 7 | A4 | Owner consent: about 11 GPU-hours, over D-063's 2-minute rule |
| 8 | A6, A8 + H-3, H-7 | Only after 1 shows a signal |

Nothing was refuted outright. The only refutation is A4's novelty claim, while A4 survives as a project verdict on the physics terms. A6, A8 + H-3 and H-7 survive only if gated on H-1.

Project files cited: `D:/nt/nt_tactical/runs/tactical/recent/wf.py`, `D:/nt/nt_tactical/runs/tactical/calendar/SPEC.md`, `D:/nt/nt_tactical/docs/TACTICAL.md`, `D:/nt/nt_tactical/src/neural_trade/core/config.py`, `D:/nt/nt_tactical/runs/tactical/lab/cache.py`, `D:/nt/nt_tactical/runs/tactical/LOG.md`.