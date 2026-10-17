# Multimodal breakthrough directions for neural_trade (BTC/USDT 1 min)

Date: 2026-10-17. Status: desk research plus verification. Nothing was installed or run on this machine. Numbers are the authors' own unless marked "ours" or "inference". Confidence is marked H/M/L.

## 1. One-paragraph answer

No pretrained model or image/graph encoding is shown, anywhere I could verify, to beat a HAR plus time-of-day baseline on 1-minute or 5-minute crypto. The best external evidence is small and about daily horizons. Brini (arXiv 2607.05291) finds that only TTM beats Log-HAR, by 1.3-1.8% QLIKE, and that an equal-weight TTM + Log-HAR average is as good as the best single model. So the honest direction is not a replacement. It is a **late-fusion stack on top of HAR/S3**. Each "view" of the data we already hold gives a scale forecast or quantile set: numeric foundation models (Chronos-2, TTM, TimesFM), tokenised candles (Kronos), rendered charts with geometry lines (CNN or CLIP), retrieved historical analogs (FAISS), and graph/wavelet statistics. One small meta-model adds them to HAR and keeps a view only if its measured increment survives controls. Direction stays closed (AUC about 0.52). The targets are log-RV level, next-hour vol change, H3 barrier-hit and HAR-scale intervals. The six tests flagged below need no new data (D-050) and each takes about 10-60 GPU or CPU minutes.

## 2. Concept map

| Concept | Views of our existing data | Tools (verified to exist) | Connects to our stack as |
|---|---|---|---|
| Numeric foundation models | 5-min log RV, range, volume series, with HAR/S3/ToD/geometry as covariates | Chronos-2, TTM r2, TimesFM 2.5, TiRex | Extra regressor or 50/50 average with HAR; quantiles feed T03/T05 interval comparison |
| Token / language of candles | OHLCV as discrete K-line tokens | Kronos | Sampled futures give range, vol and P(barrier hit) for H3; embeddings as a "candle-language" feature |
| Image | Rendered candles, volume panel, geometry lines (T13) as image channels | own CNN, CLIP, VisionTS | Geometry lines become an input modality; target is next-hour vol change (where T13 found lines help), not H3 |
| Text (auto-generated) | Templated regime caption (vol percentile, squeeze, stretch, session, line distance) | CLIP text, small encoder | Late fusion with numeric features; a caption is a rewrite of the same numbers, so test it as a leave-one-modality-out increment |
| Frequency | CWT band energies, wavelet RV | PyWavelets | Scale-free indicators next to HAR terms |
| Graph | Horizontal visibility graph statistics of 60-bar windows | ts2vg | Interpretable geometry features next to T13 lines |
| Memory / retrieval | Every past window indexed by regime descriptors | FAISS | Analog correction of the HAR residual and of H3 |
| Tabular in-context | Our existing T01/T10 feature tables | TabPFN (plain regressor/classifier) | Comparator for ridge/boosting on the same columns |
| Fusion and calibration | Outputs of all views | ridge/LightGBM meta-model, Venn-Abers, ACI | One interpretable meta-model on the HAR residual; calibration on a pooled window |
| New-data concepts | Funding, OI, book depth, news text, on-chain | data.binance.vision, text APIs | Section 5; owner decision |

Inference: every image, graph or frequency encoding is a deterministic function of the same OHLCV window. It can add nonlinear capacity, not information. So each needs an information-equivalent numeric control.

## 3. Top-12 ranked

Top-6 are flagged INSTALL AND TEST TONIGHT. Env plan: `D:/nt/nt_mm_env` (Python 3.11, torch 2.11.0+cu126) for most; `D:/nt/nt_mm_legacy` (Python 3.10, torch 2.5.1+cu124) for Kronos and Sundial pins. The TF 2.10 `nt` env is untouched.

| # | Name | Modality | Why it could help | Evidence and caveats | Install effort | GPU/disk | Build-on path | Expected value |
|---|---|---|---|---|---|---|---|---|
| 1 | **Chronos-2** (TONIGHT) | numeric + covariates, quantiles | In-context covariates (HAR, S3, ToD, line distances) give a correction and a quantile band for vol and range | arXiv 2510.15821; ranks 1st on GIFT-Eval (WQL/MASE win 81.9/83.8; TimesFM-2.5 77.5/77.7, TiRex 76.5/71.9; authors). No finance result in the paper. ARTEMIS 2603.18107 shows zero-shot embedding + linear head has RankIC -0.14 on Optiver (weak, single author). Covariate use on vol is untested (H on "exists", L on "helps") | Low (pure PyTorch) | 3 GB / 2 GB | Target log RV; HAR forecast as covariate; average with HAR | Medium, about 25-30% chance of a real increment |
| 2 | **TTM r2** (TONIGHT) | numeric, minutely/hourly, exogenous channels | Only model with a measured win over Log-HAR; fits "ensemble with HAR" | Brini: QLIKE ratio 0.982/0.986/0.987 at 1/5/22 days, 50 assets, 5-min RV; edge mostly scaling (MZ recalibration). 805k params, point forecast only. r2 pretraining list includes "Bitcoin" (contamination). Repo unmaintained | Medium (granite-tsfm 0.3.10 needs Py>=3.11, torch>=2.10) | 1 GB / 1 GB | 50/50 average with HAR/S3; fine-tune on 5-20% with HAR terms as exogenous | Medium |
| 3 | **Kronos** (TONIGHT) | OHLCV tokens, sampled paths | Sampled OHLC paths give P(hit), range quantiles for H3 and T05; embeddings as candle-shape view | arXiv 2508.02739, AAAI-26. Pretraining ends June 2024, so clean windows are 2024-07-01 onward only. Headline model not open; crypto tests were price-only at 5 min+; no independent BTC test; repo stale since 2026-04 | Medium (own legacy env; skip pyqlib fine-tune) | 4 GB / 2 GB | Features "Kronos-range", "Kronos-P(hit)" into H3 and S3 ridge; null = HAR-scaled Gaussian paths | Medium-low |
| 4 | **Regime card**: chart image + numeric + templated caption (TONIGHT) | image + text + numeric | Puts T13 geometry lines in the image; fusion head tested per modality | JKX, J. Finance 78(6) 2023 (daily US stocks). Haggett 2605.00875 AUC 0.892 is best of 8 runs, forward 7-day daily label, cross-asset AUC 0.49/0.46 (low confidence). T13: lines help next-hour vol change (+0.0195 on 2017-18, +0.0087 on Binance 2025-26) but not H3 | Low | 4 GB / 6 GB | Residual of S3 ridge (Spearman 0.4475); ladder numeric -> +image -> +caption -> both; shuffle and random-feature controls | Medium-low (about 25%) |
| 5 | **VisionTS / VisionTS++** (TONIGHT) | time series as image, MAE | Seasonal-profile forecaster for volume and range with a daily period | arXiv 2408.17253 (ICML 2025), 2508.04379. No finance result. PyPI `visionts` 1.0.1; repo stale (2025). Use 5-min bars (period 288) | Low | 3 GB / 2 GB | One extra column in HAR+ToD; VisionTS++ quantiles vs HAR-scale intervals | Low (15-20%), cheapest test |
| 6 | **Regime retrieval (FAISS analogs)** (TONIGHT) | memory of own history | Non-linear regime dependence without a new model; CPU | TimeRAF 2412.20810 (paper only, no code). No finance evidence. Neighbours overlap in time; time decay (H59) | Very low | 0 / 1 GB | Mean HAR residual of k=50 neighbours as a feature; keys include line distance, session, stretch | Medium-low (30% vol, 35% H3, own estimate) |
| 7 | TimesFM 2.5 | numeric, quantile head | Frozen feature + residual model (the pattern in 2608.08825) | Brini: no gain over Log-HAR. 2608.08825 is about stock returns in the opening hour, so its 0.597 cannot support the vol claim. v3.0 weights are non-commercial; `timesfm[torch]` now installs 3.0, pin 2.0.2 | Medium | 3 GB / 2 GB | Same ladder as 1 | Low-medium |
| 8 | **HAR-initialised network** (HARNet-style, arXiv 2205.07719) | numeric | Network that starts as the exact HAR forecast, so it cannot be worse at init; fits "build on HAR" | Daily indices only; no code verified, own implementation | Own code | small | Initialise at S3/HAR, add geometry, ToD, FM features as extra inputs | Medium, high consistency with our stack |
| 9 | Visibility-graph + wavelet statistics | graph, frequency | Scale-free interpretable features | ts2vg 1.2.4 (numpy only, Windows wheels). H47: squeeze already equals persistence, so overlap with HAR is likely | Very low | 0 / 1 GB | Add to T13 geometry set; shuffled-order control | Low |
| 10 | Sundial path samples | numeric, generative | Samples for first-passage / H3 | arXiv 2502.00816; Brini: no gain over Log-HAR; undisclosed corpus; needs transformers==4.40.1 and `trust_remote_code` | Medium | 3 GB / 1 GB | Same as Kronos, compare with H1 first-passage formula | Low |
| 11 | TabPFN (plain `tabpfn`) | tabular | Foundation comparator on our own feature tables | Weights licence is non-commercial with a browser click-through (owner account action). Do not install `tabpfn-time-series` (resolver fails, cloud client by default) | Low, but blocked by licence | 4-8 GB / 1 GB | Replace ridge/GBM in T01/T10 as a comparator | Low-medium |
| 12 | Venn-Abers / ACI on H3 with a pooled window | calibration | H3 already has skill (+15.4% Brier over B1); ECE 2.11/1.26 pp vs 1 pp bar | T12 already showed monthly isotonic calibration is 3.9% worse than H3 in all 9 cells; T05 already ran ACI (M5). Fit on 6-12 months pooled or a 2-parameter map; use the Venn-Abers [p0,p1] width as a feature | Very low | 0 / 0.1 GB | Judge ECE only | Medium for ECE, low for skill |

Watchlist, not ranked: Toto 2.0 (needs Py 3.12, Student-t head is only in 1.0), Moirai 2.0 (CC-BY-NC, heavy pins), TiRex and TiRex-2 (Linux-only kernels, no nvcc/MSVC here), FinCast (arXiv 2508.19609, tiny 1-min crypto differences, authors' own), TimeCMA/Time-LLM (LLM branch disputed, arXiv 2406.16964), MTF/GAF images, ChatTS/ChatTime (not realistic on 12 GB shared GPU), self-supervised chart MAE (only after item 4 shows an image increment).

## 4. Minimal test plans for the top-6

Common design (fixed before running, filter not engine):
- Data: cut file via `runs/tactical/common/timefix.py`, resampled to 5-min bars unless stated; Binance forward 2025-10..2026-09.
- Baselines: HAR + ToD, S3 ridge (Spearman 0.4475 on T01 target A), H3 logistic (T10/T12), HAR-scale intervals (T03/T05).
- Selection: 2019-01..2024-06, monthly walk-forward, one seed. Judgement: 2017-18 (Look J1) and Binance 2025-26 (J2), never used to choose. Block bootstrap CIs, n_eff = N // horizon (D-012). Purge gap per D-034.
- Always recalibrate (Mincer-Zarnowitz) before comparing, because Brini shows the edge is mostly scaling.
- Controls for every view: shuffled-in-time input, information-equivalent numeric control, random-feature control.
- Prospective freeze: commit code, arms and thresholds (sha256) before the judgement run. Score the months after 2026-10-02 once, together with H3 Look B.
- Success: a view stays only if its increment over S3/HAR on the judgement looks has a CI above zero after Holm, and survives the controls. A miss means one or two improvement iterations (longer context, different covariates, different rendering) before shelving.

**T-1 Chronos-2.** Inputs: 5-min log RV; context 2048 bars; covariates HAR (D/W/M), S3 forecast, ToD sin/cos, T13 line distances, with calendar as known-future. Target: log RV over the next 60 min, and the T01 delta (forecast minus known past RV). Arms: (a) HAR+ToD, (b) zero-shot univariate, (c) with covariates, (d) 50/50 of (a) and (c), (e) ridge of HAR + Chronos feature. Metrics: Spearman, QLIKE, pinball and 80% coverage vs T03/T05 HAR intervals. About 20 min GPU. Combine: use the quantile band as an extra interval candidate and the forecast as a ridge column.

**T-2 TTM r2.** Inputs: same channels, context 512, horizon 12 (confirm available branch; else truncate). Arms: HAR+ToD, zero-shot, fine-tuned on 10% (selection months only), 50/50 with HAR/S3. Metrics: QLIKE, Spearman after MZ recalibration. Under 15 min. Combine: equal-weight average first (the evidence-backed pattern), then ridge stack. Add a quantile loss on the head only if the point forecast shows a gain.

**T-3 Kronos.** Use Kronos-small (ctx 512) and mini (ctx 2048). Windows only from 2024-07-01..2025-07-20 in the cut file plus the Binance forward file (contamination rule); pre-2024-07 windows are labelled contaminated and not used for conclusions. Sample 30 paths, pred_len 60, 500 windows; check temperature and top_p sensitivity. Read off sampled high-low range, quantile coverage, P(barrier hit) for the T10/T12 barriers. Metrics: Brier/AUC vs H3, Spearman of path-implied vol vs HAR. Null: HAR-scaled Gaussian paths. About 30-60 min. Combine: add as features to H3 logistic and S3 ridge.

**T-4 Regime card.** Render 96x96 charts (numpy/PIL, no matplotlib) of the last 120 min: candles, volume panel, T13 lines as extra channels; caption from templates. Variants: frozen CLIP ViT-B/32 + PCA 32 (cheap), and a 4-layer CNN trained on a 6-hour-per-month layout (stronger). Target: S3 ridge residual of the next-hour vol change; H3 logit as a secondary target. Ladder: numeric, +image, +caption, +both; controls: shuffled embeddings, lines removed, transition-count/HAR-feature control. Rendering must be causal and per-window scaled. About 1 h.

**T-5 VisionTS.** 5-min log range and log volume, 7-day context, period 288; zero-shot forecast mean for the next hour as a column. Compare HAR+ToD vs +VisionTS (also VisionTS++ quantiles vs HAR intervals). Minutes of GPU. Control: the seasonal-profile (median by minute-of-day) alone. Expected result is small; the point is a fast kill-or-keep.

**T-6 Regime retrieval.** Embed each 5-min anchor by about 20 descriptors (vol percentile, squeeze, stretch 20-60 min, session, line distances, HAR residual of the last hours). FAISS index on months before t (with D-034 purge gap), k=50, recency weighting (H59). Features: mean realised HAR residual and mean barrier-hit of neighbours. Metrics: added Spearman and H3 Brier, block bootstrap. CPU, minutes. Control: random neighbours.

## 5. Concepts needing new data (owner decision, D-050)

| Concept | Cost | Note |
|---|---|---|
| Funding rate, open interest, long/short ratio | Free bulk zips from data.binance.vision (USD-M funding from 2020-01; metrics with OI from 2020-09-01) | Cheapest next step if D-050 is relaxed. Feeds the vol head and H3 as covariates. Pilot about 1-3 days, about 20 GB |
| Book depth snapshots | data.binance.vision lists daily `bookDepth`; start date not checked | D-044 refused tick order-book data; this is bulk snapshots, so the owner should say whether it counts |
| News/social text | APIs with restrictive terms; look-ahead and survivorship risk | Needs a finance text encoder; not verified |
| On-chain, cross-exchange spreads | Not assessed | Not verified |

Nothing was downloaded. Ask the owner one question with options A (funding/OI), B (book depth), C (news).

## 6. Strongest critiques and what they imply

- Foundation models are not direction predictors in finance (Rahimikia 2511.18578; Noguer i Alonso 2606.27100). Implication: no direction target in any test.
- The best volatility edge (TTM) is 1-2% QLIKE and mostly scaling (Brini). Implication: always MZ-recalibrate, always compare with a 50/50 average, expect small increments.
- Brini tests daily horizons on 5-min RV for equities/FX/futures, not next-hour crypto. Implication: our tests are the first direct evidence either way.
- Contamination: Kronos clean only after 2024-07; TTM lists Bitcoin in pretraining; other corpora undisclosed. Implication: the Binance forward file is the main judgement block, and a contamination label goes on every pre-2024-07 number.
- Images, MTF/GAF and graphs are functions of the same window. Implication: information-equivalent controls are mandatory.
- Tan et al. 2406.16964: removing the LLM branch does not hurt time-series LLMs. Implication: a random-LLM or no-LLM control, low priority.
- Internal conflicts: T13 shows lines add nothing to H3, so point image and geometry tests at vol change. T12 shows monthly isotonic calibration hurts, so calibration uses pooled windows. T05 already ran ACI on HAR scale.
- The earlier claim of a "0.597 correlation" lift from residual stacking is for stock returns, not volatility. Do not cite it as support.

## 7. Completeness critic

Unverified or missing:
- Nothing was run here; all Windows claims are inference. No nvcc/MSVC was found, so any kernel build (TiRex, flashrnn, pyqlib) is expected to fail.
- Chronos-2 and TTM prediction-length branches, exact class names and the GIFT-Eval numbers beyond Table 4 were checked in text, not in code.
- Kronos pins in `requirements.txt` resolved on a dry run but weren't installed together with Chronos-2; keep separate envs.
- TimesFM 2.5 `xreg` extra (jax[cuda]) has no Windows GPU build; XReg is CPU or skipped.
- Licences not fully checked: Chronos-2 model card, VisionTS++ weights, TiRex-2 HF weights, FinGPT/FinMA.
- Not reopened: Time-LLM, CALF, TEMPO; the HuggingFace Kronos tokenizer card; no independent reproduction of Kronos or JKX on BTC intraday.
- The Haggett and Hasriadi results are low confidence. The Journal of Forecasting 2025 Chronos-on-BTC claim is secondary and unchecked.
- I did not read the full project LOG rows H5-H96 here; internal cross-references (H41, H46, H47, H59, T10-T13) come from the gathered notes and should be rechecked against `runs/tactical/` before any claim is written into a decision.
- Text and sentiment modalities have no verified finance result in this pass.
- A last verification item ("chart/VLM route as a negative control") was cut off in the input, so it is not used.

## 8. References

- Chronos-2: https://arxiv.org/abs/2510.15821 , https://github.com/amazon-science/chronos-forecasting , https://huggingface.co/amazon/chronos-2
- Kronos: https://arxiv.org/abs/2508.02739 , https://github.com/shiyu-coder/Kronos , https://huggingface.co/NeoQuasar
- TimesFM: https://github.com/google-research/timesfm ; TTM: https://arxiv.org/abs/2401.03955 , https://github.com/ibm-granite/granite-tsfm
- TiRex: https://arxiv.org/abs/2505.23719 , https://github.com/NX-AI/tirex ; TiRex-2: https://arxiv.org/abs/2607.01204 , https://github.com/NX-AI/tirex-2
- Moirai 2.0: https://arxiv.org/abs/2511.11698 ; Sundial: https://arxiv.org/abs/2502.00816 , https://github.com/thuml/Sundial ; Toto: https://arxiv.org/abs/2407.07874 , https://github.com/DataDog/toto
- TabPFN-TS: https://arxiv.org/abs/2501.02945 ; FinCast: https://arxiv.org/abs/2508.19609
- Finance TSFM evidence: Brini https://arxiv.org/abs/2607.05291 ; Rahimikia https://arxiv.org/abs/2511.18578 ; Noguer i Alonso https://arxiv.org/abs/2606.27100 ; Goel https://arxiv.org/abs/2505.11163 ; Marconi https://arxiv.org/abs/2507.07296 ; ARTEMIS https://arxiv.org/abs/2603.18107 ; 2608.08825
- HARNet: https://arxiv.org/abs/2205.07719
- Images: Jiang-Kelly-Xiu, J. Finance 78(6), doi 10.1111/jofi.13268 ; Haggett https://arxiv.org/abs/2605.00875 ; Lensky-Hao https://arxiv.org/abs/2304.02472 ; VisionTS https://arxiv.org/abs/2408.17253 , https://github.com/Keytoyze/VisionTS ; VisionTS++ https://arxiv.org/abs/2508.04379 ; Time-VLM https://arxiv.org/abs/2502.04395 ; ViTST https://arxiv.org/abs/2303.12799 ; Huang-Sangiorgi-Urquhart, doi 10.1016/j.intfin.2024.102064
- Retrieval / LLM: TimeRAF https://arxiv.org/abs/2412.20810 ; TimeCMA https://arxiv.org/abs/2406.01638 ; Tan et al. https://arxiv.org/abs/2406.16964 ; ChatTS https://arxiv.org/abs/2412.03104
- Graph / frequency: https://arxiv.org/abs/1710.10980 , https://arxiv.org/abs/1202.1854 , ts2vg on PyPI
- Calibration: https://arxiv.org/abs/2502.05676 , https://arxiv.org/abs/2106.00170 , https://arxiv.org/abs/2511.04275 , https://github.com/scikit-learn-contrib/MAPIE
- Contamination: https://arxiv.org/abs/2605.26161 , https://arxiv.org/abs/2609.10357 , https://arxiv.org/abs/2410.10393
- Data: https://data.binance.vision ; FinMME https://arxiv.org/abs/2505.24714
