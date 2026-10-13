# What the 13-feature logistic regression captures on 1-minute BTC, and why its edge faded

**Scope.** This report covers the tactical-session evidence (LOG.md H51, H52, H53, H54, H59, H60, H72, H76, H82, H83), four research angles (mechanism, crypto literature, market events, test design with first CPU runs) and one adversarial verification pass. Findings the verification refuted are left out, and its corrections are applied. All numbers computed in this study are exploratory. They come from development data only (the cut file `D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv`, which ends 2025-07-20). There is no SPEC behind them.

**Correction to the task framing.** The long-history file `Bitcoin_BTCUSDT.csv` is not a Binance-like BTC/USDT feed. It is Bitstamp BTC/USD. Its timestamps are UTC+4 during US daylight saving and UTC+5 otherwise, up to a splice on 2024-09-13 08:00 (file time). This is the most consequential finding, and Section 2 gives the evidence. Every mechanism below is therefore about Bitstamp, a thinner USD venue.

---

## 1. Answer first

### 1a. What the regression captures, ranked by evidence

**1. The close-to-next-open gap, from bid-ask bounce or a stale close on a thin venue.**
- Confidence: high.
- Status: established on this data and reproduced by independent code.
- The 1-minute "reversal" sits almost entirely in the jump from `close_t` to `open_{t+1}`. The next bar's own move does not carry it.

Correlation of r_t with each target, by era:

| Target | 2017H2–2020 | 2021–23 | 2024–25 |
|---|---|---|---|
| next return, close to close | −0.079 | −0.003 | −0.011 |
| next gap, close_t to open_{t+1} | **−0.229** | −0.104 | −0.052 |
| next return, open to close | +0.038 | +0.041 | +0.009 |

- **It is concentrated in thin bars.** The slope of r_{t+1} on r_t goes from −0.242 in the lowest relative-volume quintile to −0.019 in the highest. The gap slope has the same ordering (−0.265 down to −0.060). Few trades means the close is printed on one side of the book.
- **The pooled Roll spread was large.** It was 10.4, 7.9, 6.4 and 6.4 bps in 2017, 2018, 2019 and 2020.
- **An independent study saw the same thing on the same venue.** De Nicola (2021) used Bitstamp 1-minute bars from 2015 to 2018. He reports a lag-1 autocorrelation of −0.145 and attributes autocorrelation at 15 minutes and below to bid-ask bounce. Under a Gaussian mapping, ρ = −0.14 corresponds to a 1-bar AUC of about 0.437 (simulated). That matches the project's 1-minute-ahead cell.
- **It cannot be captured at the next open.** This is the part H59 and H60 remove.
  - In AUC terms it is about one third of the 2018–20 edge: the excess over 0.5 falls from about 0.095 (0.59–0.60, H51) to about 0.065 (0.56–0.57, H59).
  - The top-10% hit rate falls from 63–65% to 59–60%, also about one third of the excess.
  - "About half" holds only for H48's bps per trade (+0.54 to +0.28/+0.30, H60).
- **The H59/H60 label "bid-ask bounce" stands.** The alternative (sub-minute transient impact after bursts of aggressive orders) is refuted for this data. It assumed a Binance one-tick spread, and it predicts the reversal is strongest in high-volume bars, which is the opposite of what we see.

**2. A small reversal beyond the gap, at 10–15 minutes, plus multi-bar features.**
- Confidence that it exists: medium. Confidence in any particular mechanism: low.
- Under next-open entry, the 1-bar return still has a univariate AUC below 0.5 in 2018–20:
  - h = 10: 0.490, 0.485 and 0.480 in 2018, 2019 and 2020.
  - h = 15: 0.487, 0.485 and 0.476.
  - With close entry the same cells were 0.446–0.464.
- So a small tradable reversal component existed, and longer features (BB %b, longer trailing returns) carried part of the remaining 0.56–0.57.
- Part of it is also venue-specific. On matched windows, Binance BTCUSDT had a real but smaller lag-1 reversal in 2018–19, about one third of the file's: mean AC1 −0.039 against −0.116 in 2018, and −0.063 against −0.186 in 2019.
- Candidate mechanisms, none identified yet:
  - inventory or liquidity-provision price pressure (Grossman-Miller 1988; Hendershott-Menkveld 2014; Nagel 2012);
  - overreaction to moderate and large moves;
  - slow propagation of stale prices from more liquid venues.

**3. Weak lag-1 continuation from the next open to the next close.**
- Confidence: low-medium.
- The correlation is +0.04 in 2017–23. It fits persistent order signs (Lillo-Farmer 2004).
- It is small, and we have not shown that the regression uses it.

**How the reversal depends on move size** (this corrects an earlier statement that it does not depend on size at all):
- It is absent for small moves and present for moderate and large ones.
- It does not grow further for the largest moves. Slope by |z| quintile: Q2 +0.046, Q3 −0.052, Q4 −0.098, Q5 −0.079. The correlation is strongest in Q5 at −0.122.
- De Nicola's 1-minute table shows the same shape: −0.156 for all moves, −0.200 for moves of at least 1σ, falling to −0.045 at 6σ.
- H52's AUC-based statement that large moves are not more predictable can still hold, because AUC measures sign skill, not the size of the reversal.

**Not supported or refuted as the mechanism:**

| Candidate | Verdict | Evidence |
|---|---|---|
| Tick discreteness, or a relative tick of 1/price | Refuted | The tick was $0.01 while the reversal was strong (a relative tick of 0.001–0.03 bps against a Roll spread of 6–10 bps). The reversal was already gone when the $1 tick arrived on 2022-08-10. |
| Sub-minute transient impact after aggressive bursts | Refuted as the main driver | The reversal is strongest in low-volume bars, and the premise assumed Binance. |
| High-volume liquidity provision (Campbell-Grossman-Wang 1993) as the main driver | Contradicted | The volume-quintile ordering runs the wrong way. |
| Round-number order clustering (Osler 2003) | Not supported (medium confidence) | Reversal is weaker within $2 or 2 bps of a $100 multiple (−0.043 against −0.082). Round $1000 levels and the direction of a crossing were not separated. |

### 1b. Why it faded, ranked

**1. The venue's effective spread and bounce shrank.** Confidence: high.
- The pooled Roll spread was 10.4 bps (2017), 7.9 (2018), 6.4 (2019), 6.4 (2020), 0 (2021), 2.4 (2022), 0 (2023), 1.9 (2024) and 0.6 (2025). The estimator is censored at 0, so the zero years are lower bounds.
- The monthly correlation between r_t and the next gap was −0.25 to −0.35 in 2017. It was mostly −0.15 to −0.30 in 2018–20, about −0.10 to −0.18 in 2021, −0.05 to −0.12 in 2022–23, −0.03 to −0.10 in 2024, and about 0 from 2025-04.
- The 1-bar univariate AUC moved from 0.38–0.42 to about 0.505 in the 1-minute-ahead cell. It moved from 0.434–0.464 to 0.498 for the 10–15 minute targets.
- Bitstamp's daily volume fell from about 11k BTC (2017) to about 1.3k (2025). So "liquidity improved" here means narrower effective spreads, not more volume.

**2. Short-horizon efficiency rose across the market.** Confidence: medium-high.
- On Binance, the reversal shrank in the same years: AC1 was −0.004 in 2020, −0.015 in 2021 and +0.020 in 2022.
- Published support:
  - Sensoy (2019): BTC more efficient intraday since 2016, with liquidity raising efficiency.
  - Brauneis-Mestel (2018): cryptocurrencies become less predictable as liquidity increases.
  - Tang et al. (2025): the short-scale Hurst exponent rose from 0.415 (2019–20) to 0.478 (2021–22).
  - Kim-Hansen (2026 preprint): little 1-minute autocorrelation on Binance perpetuals in 2021–24, except at quarter-hour marks.
- The equity precedent is Chordia-Roll-Subrahmanyam (2005, 2008): short-horizon predictability fell as the tick and spreads fell.
- The adaptive market hypothesis (Chu-Zhang-Chan 2019; Takaishi-Adachi 2020) allows this decline to come in episodes rather than as a smooth trend.

**3. Changes in how the data were built confound the later breaks.** Confidence: medium.
- On 2022-08-10 Bitstamp's tick changed to $1. Zero-return minutes rose from 3–4% to 12% (Oct 2022) and 31% (Dec 2022).
- On 2024-09-13 the file is spliced. About 4 hours of real data are missing, zero-volume minutes rise from 3.7% to 12.0%, and the share of bars with open equal to the previous close rises from 0.20 to 0.26.
- On 2025-03-15 there is a 1,160-minute gap. After it, zero-volume minutes are about 0.6–0.8% and the gap reversal is about 0.
- The ~2024-09 AUC break coincides with the splice within H72's coarse dating. It is confounded, but it has not been shown to be an artefact.

**4. Price discovery may have moved to perpetuals.** Confidence: low.
- This would be a slow mechanism, consistent with the gradual decay after 2021, but no single event lines up with a break.
- The derivative-share figures behind it (CCData about 70% in Oct 2024, FOW about 48% in 2021) were not verified.

**Not the explanation:**
- Binance-only events cannot move a Bitstamp series: the zero-fee period (2022-07-08 to 2023-03-22), the BNB discount steps and the launch of Binance perpetuals.
- Halvings, the FTX bankruptcy and the US spot-ETF launch do not line up with any of the breaks.

### 1c. Volatility predictability (HAR)

- **The level is real.** On clean Binance data in 2025–26, the HAR Spearman correlation is 0.448 (H83).
- **The rise from 0.29 to 0.40 (H53, H72) is partly confounded.** H72 dates the step in HAR Spearman to 2022-10, two months after the $1 tick, when stale 1-minute prices jumped.
- **The story is coherent but untested.** It says that reversal is a first-moment premium that directional traders compete away, while volatility clustering cannot be arbitraged by sign bets. Its supporting claim, that "the noise share fell over time" (attributed to Hansen-Lunde 2006), could not be verified.
- **Confidence that volatility predictability actually rose: low-medium.**

---

## 2. Dated market-structure events against our break dates

The breaks come from H72, which reports direction-AUC plateaus of about 0.57, 0.545 and 0.521 with sup-Chow p = 0.004. Only the 2018-03 break is tightly dated.

**Data-construction events.** These are not market events, and they belong in any re-dating of the breaks.

| Date | Event | 2018-03 | 2021-03 | 2024-09 | Source |
|---|---|---|---|---|---|
| 2018-03-11 | DST seam in the shifted file: one real hour missing, a 109 bps one-minute jump | Coincides, but too small to move monthly AUC | – | – | own check (adv_verify) |
| Every spring and autumn until 2024 | DST seams (for example 2019-11-03: a misaligned hour) | outlier features twice a year | – | – | own check |
| 2022-08-10, 10:00–11:00 CET | Bitstamp BTC/USD tick from $0.01 to $1. The file switches inside file hour 12, which is 08–09 UTC under the +4 h shift. | no match | no match | no match. It lines up with the HAR step of 2022-10. | Bitstamp blog 2022-08-02 |
| 2024-09-13 08:00 (file time) | Splice: offset UTC+4 to UTC, about 4 h missing, candle construction changes | – | – | **Match within H72's dating: confounded** | own check |
| 2025-03-15 | 1,160-minute gap; afterwards about 0.6–0.8% zero-volume minutes and a gap correlation of about 0 | – | – | Inside the post-2024-09 regime; a second change point | own check |

**Market events.**

| Date | Event | 2018-03 | 2021-03 | 2024-09 | Source |
|---|---|---|---|---|---|
| 2017-12-10 / 12-17 | Cboe and CME BTC futures launch | Near, not exact. Regime-level at most. | – | – | general knowledge, not re-verified |
| Jan–Feb 2018 | End of the 2017 bubble; record Bitstamp volume (about 15–21k BTC a day in the file) | Plausible regime shift, but no specific event | – | – | file data |
| ~2018-07 / 2019-07-14 | Binance BNB fee discount steps (50% to 25%, then kept at 25%) | no (Binance only) | no | no | Decrypt; Binance.US |
| Sept 2019 | Binance USDT-margined BTC perpetuals | no | no | no | Decrypt 2019 |
| 2020-05-11 | Halving | no | no | no | standard |
| 2020-09-10 | Bitstamp moves to a Nasdaq matching engine. This was most likely the only migration: the 2018 Cinnober contract was delivered as Nasdaq after Nasdaq bought Cinnober in Jan 2019. | – | Candidate. The file's bounce statistic falls gradually through late 2020 to early 2021. A causal link is unproven. | – | Forklog 2020-09-11; Finextra 2020 |
| Nov 2020 – Mar 2021 | Price rally from $16k to $47–57k; institutional inflow | – | Plausible, market-wide | – | price data |
| 2021-01 | Tang et al. period split (Hurst 0.415 to 0.478). The split is the authors' choice, not an estimated break. | – | Loose corroboration | – | arXiv 2402.11930 |
| 2022-02-15 | Binance USDⓈ-M BTCUSDT perpetual tick from 0.01 to 0.1 (futures, not spot) | – | – | no | MultiCharts forum |
| 2022-07-08 to 2023-03-22 | Binance zero-fee BTC spot promotion | no | no | no (and Binance only) | The Block; FXStreet; Bloomberg |
| 2022-11-11 | FTX bankruptcy filing | no | no | no | The Block |
| 2024-01-10 / 01-11 | US spot ETFs approved, then trading | – | – | no (8 months earlier) | standard |
| 2024-04-20 | Halving | – | – | no | standard |
| 2025-06-02 | Robinhood completes the Bitstamp acquisition | – | – | no | Robinhood newsroom |

**Summary.**
- **2018-03:** no specific exchange event. It is most likely a regime or volatility change after the bubble, or a purely statistical change point.
- **2021-03:** a plausible but unproven combination of the Nasdaq engine on Bitstamp, a rally that shrank the relative tick and market-wide efficiency gains.
- **2024-09:** confounded by the data splice.

---

## 3. Published crypto studies on minute-scale autocorrelation and its trend

| Study | Period | Venue / data | Horizon | Finding | Confidence |
|---|---|---|---|---|---|
| De Nicola 2021, *Ledger* 6:58–80 | 2015-03 to 2018-06 | **Bitstamp** BTC/USD, 1-minute | 1m to 1d | Lag-1 AC: 1m −0.145, 5m −0.102, 15m −0.058, 30m −0.044, 1h −0.056, 2h −0.086, 4h −0.056 (all p < 1e-5); 1d −0.007 (n.s.). At 1m the reversal is −0.200 after moves of at least 1σ and decays for larger thresholds; the author attributes ≤15m to bounce. Hourly volatility AC1 95% CI 0.66–0.72. | High (tables read). Same venue as our file. |
| Jaquart, Dann & Weinhardt 2021, *JFDS* 7:45–66 | 2019-03 to 2019-12 | Bloomberg composite BTC/USD, 1-minute | 1/5/15/60m | Accuracy, logistic regression: 0.511/0.518/0.520/0.539. GRU: 0.518/0.525/0.536/0.557. LSTM up to 0.560. Predictability grows with horizon; lagged returns are the top features. Negative after 30 bps round-trip costs. | High (table read) |
| Tang et al. 2025, *Fractal Fract.* 9(10):635 (arXiv 2402.11930) | 2019-04 to 2022-05 | Reuters BTC/USD, 10-minute | short scale | Short-scale Hurst 0.415 ± 0.006 (2019–20) rising to 0.478 ± 0.004 (2021–22): mean reversion fading. | High for the numbers; loose as dating of our break |
| Sensoy 2019, *FRL* 28:68–73 | ~2013–2018 | BTCUSD, BTCEUR, intraday | intraday | More efficient intraday since early 2016. Efficiency is lower at higher frequency; liquidity raises it and volatility lowers it. | High for the direction (abstract) |
| Takaishi & Adachi 2020, *APFM* 27:145–154 | early years to ~2019 | BTC | (not verified) | Hurst < 0.5 early, about 0.5 after ~2013; deteriorates intermittently later. | Medium |
| Zargar & Kumar 2019, *RIBF* 47:344–353 | (not verified) | BTC | 15/30/60/120m, 1d | Inefficient at high frequency, memoryless at daily; efficiency varies over time. | Medium-high (abstract) |
| Catania & Sandholdt 2019, *JRFM* 12(1):36 | (not verified) | BTC | up to 1d | Return predictability up to 6 h, none at 1 d. Realised-volatility predictability increases over time. | Medium-high (abstract) |
| Aslan & Sensoy 2020, *FRL* 35:101298 | (not verified) | crypto | intraday | U-shaped relation between efficiency and sampling frequency. | Medium |
| Brauneis & Mestel 2018, *Econ. Letters* 165:58–61 | (not verified) | many coins | daily | Less predictable as liquidity rises. ("Bitcoin least predictable" was not verified.) | Medium |
| Shen, Urquhart & Wang 2022, *Fin. Review* 57(2) | (not verified) | BTC | 30m sessions | First half-hour return predicts the last (momentum); strongest in high-volume and high-volatility sessions; attributed to liquidity provision. | High on the sign |
| Wen, Bouri, Xu & Zhao 2022, *NAJEF* 62:101733 | 2013-03 to 2020-05 | BTC and others, high frequency | 30m sessions | Both intraday momentum and reversal, depending on jumps, FOMC days, liquidity and COVID. | Low (paywalled; abstract only) |
| Miralles-Quirós & Miralles-Quirós 2022, *J. Appl. Econ.* 25(1):1294–1313 | (not verified) | BTC, hourly | 6–24h | Overreaction after negative hourly shocks, then reversal. | Medium |
| Borgards 2021, *NAJEF* 57:101428 | (not verified) | 20 coins | intraday and interday | Longer momentum periods than US stocks. The method is ex post, not a forecast test. | Medium |
| Kim & Hansen 2026, arXiv 2607.09426 (preprint) | 2021-01 to 2024-10 | **Binance perpetuals** (BTC and 5 others) | 1m, 10s | Little 1-minute correlation at most minutes; strong negative AC at quarter-hour marks. Quarter-hour order imbalance reverses over the first half hour. Out-of-sample R² 3.37% for the 10-second return after the boundary. | Medium (not peer reviewed) |
| Chu, Zhang & Chan 2019, *IRFA* 64:221–231 | (not verified) | BTC, ETH; high frequency | intraday | Time-varying efficiency (adaptive market hypothesis). Types of news and events **may not** be significant for efficiency. | Medium |
| Akyildirim, Goncu & Sensoy 2021, *Ann. Oper. Res.* 297:3–36 | (not verified) | 12 coins | minutes to daily | 55–65% accuracy, SVM best. | Low-medium (likely optimistic) |

**Gap in the literature.** We found no published estimate of break dates in BTC minute-return autocorrelation. We also found no year-by-year 1-minute spot series for 2022–25. On those points the project's measurements are new rather than replications.

---

## 4. Testable implications, mapped to T09 Q1–Q9

**Mechanisms**
- **M1:** bounce or stale close on the venue.
- **M2:** inventory or liquidity-provision reversal.
- **M3:** overreaction to moderate and large moves.
- **M4:** an artefact of feed or data construction.
- **M5:** market-wide efficiency gain.

**Constraints that apply to every test**
- Convert the file to UTC first: subtract 4 h during US DST and 5 h otherwise, up to 2024-09-13 08:00.
- Mask the DST seams.
- Treat 2022-08-10, 2024-09-13 and 2025-03-15 as known change points.
- Do not use the midrange (H+L)/2 as a "true mid" without a simulated null. For Brownian motion its lag-1 autocorrelation is 0.17–0.31, depending on the number of trades per bar.
- D-050 excludes new data sources. A clean Binance history needs the owner's approval (H82).

| Q | Test | Expected under M1 (bounce / stale) | Under M2 (inventory) | Under M3 (overreaction) | Under M4 / M5 |
|---|---|---|---|---|---|
| **Q1 features** | Per-feature univariate AUC and drop-one importance, close entry against next-open entry; add the close-location value CLV = (2C−H−L)/(H−L) | The 1-bar return and CLV dominate with close entry and collapse with next-open entry | Multi-bar returns and BB %b keep weight under next-open entry | Weight on returns scaled by size (\|z\|) and longer windows | M4: feature weights jump at the change points |
| **Q2 shape** | Partial dependence or splines on r_t/σ; slopes by \|z\| decile | Saturates: reversal at moderate moves, flat for the largest (observed) | Grows with size and with trailing RV | Grows with \|z\|, may be asymmetric (after negative shocks, Miralles-Quirós) | – |
| **Q3 lag / horizon** | Lo-MacKinlay VR(q) for q = 2..240 on close and next-open returns; gap decomposition (T1) | VR deficit fully explained by lag-1 autocovariance, gap target only | Deficit keeps growing over several lags (half-life of minutes or more), present in open-to-close | VR < 1 at 60–240 m on next-open returns (De Nicola: 1–4 h AC −0.056 to −0.086) | M5: VR → 1 at every q after 2021 |
| **Q4 microstructure** | Monthly Roll, Corwin-Schultz, Abdi-Ranaldo and EDGE spreads; regress the bounce premium (close AUC minus next-open AUC) on spread (T2) | Premium ∝ spread, about 0 where every estimator is 0; next-open AUC independent of spread | Next-open AUC rises with spread **and** volatility (Nagel) | No spread link | M4: estimator breaks at the feed change points |
| **Q5 conditions** | Slices by relative volume, RV, zero-volume or flat flags, time since the last trade, hour of week (in UTC), weekend; HAC errors (T4) | Strongest in low-volume, flat, quiet bars (observed, gap target) | Strongest in high-volume, high-RV bars, open-to-close target | Strongest after large \|z\| | – |
| **Q6 decay** | Bai-Perron on monthly AUC and on the Roll and gap series, with feed dummies (2022-08, 2024-09-13, 2025-03-15) | AUC breaks coincide with spread breaks (2018-03 and 2021-01..03 roughly do) | Decay with market-maker depth; no tie to the spread | – | M4: the 2024-09 break vanishes once the splice is modelled. M5: same timing appears on Binance history. |
| **Q7 economic size** | bps per trade and top-decile hit rate at next-open fill, zero costs (D-044) | About 0 net of the gap; the about-one-third AUC loss and about-half bps loss already observed (H59/H60) | Small but positive at next open, concentrated in stress months | Positive after large shocks, holding 30 m to 24 h | – |
| **Q8 hourly** | 1-hour-bar AUC (0.557 now) under next-open entry, by era; hour-of-day effects in UTC only | Little hourly skill from bounce (it is a 1-bar effect) | Fades over hours (Hendershott-Menkveld half-life ~1 day on NYSE) | Persists at 1–24 h (De Nicola; Miralles-Quirós); possibly session momentum (Shen et al.) | Hour-of-day results before 2024-09-13 are shifted 4–5 h unless converted |
| **Q9 present** | 2025–26 Binance (H83 data): next-open AUC, quarter-hour phase feature (minute 0/15/30/45) | About 0.5 unconditionally (gap correlation about 0 from 2025-04) | Small residual only in stress | Small residual after shocks | Residual structure is conditional (quarter-hour phase, Kim-Hansen), not unconditional lags |

**Most decisive next tests, in order:**
1. **T1, the gap decomposition with a per-lag AUC.** It is CPU-only and has partly run already.
2. **The Q6 break re-test with feed change points.** It decides whether the 2024-09 break is real.
3. **T2, spread against skill.**
4. **Q8 in UTC.**
5. **The Binance re-run of H51 and H59 for 2018–25.** Expected result, an inference: the 2018–20 AUC comes out materially below 0.59–0.60, and the "decay" looks smaller. This needs the owner's approval of the data source.

---

## 5. Completeness critic: still unverified or missing

1. **No direct candle match with Bitstamp.** The Bitstamp API was unreachable from this machine. The venue is identified by the tick-change date matching to the hour, by elimination (the file starts before Binance existed) and by volume level. Confidence high, but not proven.
2. **The Binance comparison is thin.** It uses one 1,000-minute window per month (about 8–12 a year). A 2018 Binance mean AC1 of +0.009 was not reproduced (−0.039 instead). Treat the yearly Binance values as ±0.03.
3. **The quintile betas are pooled.** They have no standard errors, and their quintiles are ranked over a whole era. Monthly betas with HAC bands (T4) are pending.
4. **The mechanism of the residual reversal beyond the gap** (1-bar AUC 0.476–0.490 at 10–15 minutes, next-open entry, 2018–20) is not identified.
5. **The cause of the 2018-03 break** is unknown.
6. **The 2021-03 attribution is unproven.** The Nasdaq engine on Bitstamp is a candidate, but the change looks gradual, not like a step.
7. **The 2024-09 break** is confounded by the splice. This is not resolved.
8. **The rise in HAR predictability** may be a $1-tick staleness artefact. It needs HAR on noise-robust realised volatility (subsampled or pre-averaged) and on Binance history.
9. **Unverifiable citations:**
   - Hansen-Lunde's "noise fell over time" claim. The synthesis that leans on it is unsupported.
   - The content of Wen et al. (paywalled).
   - A roughly 90% fall in Binance BTC volume after the zero-fee period ended.
   - The CCData and FOW derivative-share figures.
   - "Bitcoin least predictable" in Brauneis-Mestel.
   - The sample periods of several abstract-only papers (Section 3).
10. **The pre-2017 history and the Binance spot tick before today** (0.01 now) were not checked.
11. **Missing order-flow data.** Taker-buy volume and trade signs would separate bounce, stale print and inventory cleanly, but D-050 excludes them. Any attribution from OHLCV bars alone is provisional.
12. **Effect on the MVP.** Folds whose blocks straddle 2022-08-10, 2024-09-13 or 2025-03-15 (including D-046 judgement folds) are exposed to feed changes. The MVP lead should record this. Treat it as an inference until the Q6 test runs.

---

## 6. References

**Project and scripts**
- Project evidence: `D:/nt/nt_tactical/runs/tactical/LOG.md` rows H51–H54, H59, H60, H72, H76, H82, H83; `runs/tactical/decay/report.log`.
- Scripts from this study: `C:/Users/Step/AppData/Local/Temp/claude/d--nt-neural-trade/19204bbf-66e0-4407-bdcf-53181c7d4fa1/scratchpad/`
  - provenance: `peek.py`, `peek2.py`, `lag2.py`, `scan.py`, `scan.csv`, `daily.py`, `splice.py`, `conv.py`;
  - first CPU tests: `micro_diag.py`, `micro_cond.py`;
  - verification: `adv_verify.py`, `adv_verify2.py`, `adv_verify3.py`.

**Market microstructure theory**
- Roll 1984, *J. Finance* 39(4):1127–1139. https://doi.org/10.1111/j.1540-6261.1984.tb03897.x
- Harris 1990, *JFQA* 25(3):291–306. Harris 1991, *RFS* 4(3):389–415. https://doi.org/10.1093/rfs/4.3.389
- Grossman & Miller 1988, *J. Finance* 43(3):617–633.
- Campbell, Grossman & Wang 1993, *QJE* 108(4):905–939. https://doi.org/10.2307/2118454
- Llorente, Michaely, Saar & Wang 2002, *RFS* 15(4):1005–1047.
- Kyle 1985, *Econometrica* 53(6):1315–1335. Hasbrouck 1991, *J. Finance* 46(1):179–207.
- Kaul & Nimalendran 1990, *JFE* 28:67–93. https://doi.org/10.1016/0304-405X(90)90048-4
- Chordia, Roll & Subrahmanyam 2005, *JFE* 76(2):271–292. Chordia, Roll & Subrahmanyam 2008, *JFE* 87(2):249–268. https://ideas.repec.org/a/eee/jfinec/v87y2008i2p249-268.html
- Avramov, Chordia & Goyal 2006, *J. Finance* 61(5):2365–2394. https://doi.org/10.1111/j.1540-6261.2006.01060.x
- Nagel 2012, *RFS* 25(7):2005–2039. https://doi.org/10.1093/rfs/hhs066 (NBER w17653)
- Hendershott & Menkveld 2014, *JFE* 114(3):405–423. https://doi.org/10.1016/j.jfineco.2014.08.001
- Obizhaeva & Wang 2013, *JFM* 16(1):1–32.
- Bouchaud, Gefen, Potters & Wyart 2004, *Quant. Finance* 4(2):176–190. https://arxiv.org/abs/cond-mat/0307332
- Cont, Kukanov & Stoikov 2014, *JFEc* 12(1):47–88. https://arxiv.org/abs/1011.6402
- Lillo & Farmer 2004, *SNDE* 8(3). https://doi.org/10.2202/1558-3708.1226
- Osler 2003, *J. Finance* 58(5):1791–1819. https://doi.org/10.1111/1540-6261.00588
- Dayri & Rosenbaum 2015, *Market Microstructure and Liquidity* 1(1). https://doi.org/10.1142/S2382626615500033

**Noise, volatility and statistical methods**
- Aït-Sahalia, Mykland & Zhang 2005, *RFS* 18(2):351–416. https://doi.org/10.1093/rfs/hhi016
- Hansen & Lunde 2006, *JBES* 24(2):127–161.
- Corsi 2009, *JFEc* 7(2):174–196.
- Lo & MacKinlay 1988, *RFS* 1(1):41–66. https://doi.org/10.1093/rfs/1.1.41
- Working 1960, *Econometrica* 28(4):916–918. https://doi.org/10.2307/1907573
- Bai & Perron 2003, *J. Appl. Econometrics* 18(1):1–22. https://doi.org/10.1002/jae.659

**Spread estimators**
- Corwin & Schultz 2012, *J. Finance* 67(2):719–760. https://doi.org/10.1111/j.1540-6261.2012.01729.x
- Abdi & Ranaldo 2017, *RFS* 30(12):4437–4480. https://doi.org/10.1093/rfs/hhx084
- Ardia, Guidotti & Kroencke 2024, *JFE* 161:103916. https://doi.org/10.1016/j.jfineco.2024.103916
- Brauneis, Mestel, Riordan & Theissen 2021, *JBF* 124:106041. https://madoc.bib.uni-mannheim.de/62108/

**Crypto studies**
- Bianchi, Babiak & Dickerson 2022, *JBF* 142. https://econpapers.repec.org/RePEc:eee:jbfina:v:142:y:2022:i:c:s0378426622001418 ; Riksbank WP 413.
- De Nicola 2021, *Ledger* 6:58–80. https://ledger.pitt.edu/ojs/ledger/article/view/213
- Jaquart, Dann & Weinhardt 2021, *JFDS* 7:45–66. https://doi.org/10.1016/j.jfds.2021.03.001
- Tang, Arias-Calluari, Najafi, Harré & Alonso-Marroquin 2025, *Fractal Fract.* 9(10):635. https://doi.org/10.3390/fractalfract9100635 ; https://arxiv.org/abs/2402.11930
- Sensoy 2019, *FRL* 28:68–73. https://ideas.repec.org/a/eee/finlet/v28y2019icp68-73.html
- Takaishi & Adachi 2020, *APFM* 27:145–154. https://arxiv.org/abs/1902.09253
- Urquhart 2016, *Econ. Letters* 148:80–82. https://eprints.soton.ac.uk/400597
- Bariviera 2017, *Econ. Letters* 161:1–4. https://arxiv.org/pdf/1709.08090
- Tiwari, Jana, Das & Roubaud 2018, *Econ. Letters* 163:106–109.
- Zargar & Kumar 2019, *RIBF* 47:344–353. https://ideas.repec.org/a/eee/riibaf/v47y2019icp344-353.html
- Catania & Sandholdt 2019, *JRFM* 12(1):36. https://www.mdpi.com/1911-8074/12/1/36
- Aslan & Sensoy 2020, *FRL* 35:101298.
- Brauneis & Mestel 2018, *Econ. Letters* 165:58–61. https://ideas.repec.org/a/eee/ecolet/v165y2018icp58-61.html
- Dimpfl 2017, SSRN 2949807.
- Akyildirim, Goncu & Sensoy 2021, *Ann. Oper. Res.* 297:3–36.
- Shen, Urquhart & Wang 2022, *Fin. Review* 57(2). https://doi.org/10.1111/fire.12290
- Wen, Bouri, Xu & Zhao 2022, *NAJEF* 62:101733. https://doi.org/10.1016/j.najef.2022.101733
- Borgards 2021, *NAJEF* 57:101428.
- Miralles-Quirós & Miralles-Quirós 2022, *J. Appl. Econ.* 25(1):1294–1313. https://ideas.repec.org/a/taf/recsxx/v25y2022i1p1294-1313.html
- Chu, Zhang & Chan 2019, *IRFA* 64:221–231. https://ideas.repec.org/a/eee/finana/v64y2019icp221-231.html
- Kim & Hansen 2026, "The Quarter-Hour Effect", arXiv 2607.09426 (preprint). https://arxiv.org/abs/2607.09426
- Urquhart 2017, *Econ. Letters* 159:145–148. https://ideas.repec.org/a/eee/ecolet/v159y2017icp145-148.html

**Exchange events and press**
- Bitstamp tick change: https://blog.bitstamp.net/post/changes-to-tick-sizes/ (2022-08-02)
- Bitstamp matching engine: Forklog 2020-09-11; Finextra 2020 (https://www.finextra.com/pressarticle/84015/bitstamp-installs-nasdaq-matching-engine); The Block 2018 (https://www.theblock.co/post/2915/); Crowdfund Insider 2018-11.
- Binance zero-fee promotion:
  - https://www.theblock.co/post/156121/binance-set-to-drop-bitcoin-trading-fees-to-zero-on-select-pairs-globally
  - https://www.fxstreet.com/amp/cryptocurrencies/news/binance-market-share-could-drop-after-abolishing-most-zero-fee-trading-boosts-trueusd-stablecoin-202303230217
  - https://bitcoinmagazine.com/business/binance-removes-bitcoin-trading-fees
- Binance futures tick change: https://www.multicharts.com/discussion/viewtopic.php?p=139091
- BNB discount: https://support.binance.us/en/articles/9843485 ; https://decrypt.co/resources/binance-coin
- FTX filing: https://www.theblock.co/post/185083/ftx-files-for-chapter-11-bankruptcy
- Robinhood and Bitstamp: https://www.theblock.co/post/356701/robinhood-bitstamp-acquisition
- Binance perpetuals launch: https://decrypt.co/8757/binance-bitcoin-futures-trading-september
- Derivatives share (unverified): https://ccdata.io/reports/exchange-review-october-2024 ; https://www.fow.com/insights/3696519-derivatives-take-48-share-of-crypto-market-report