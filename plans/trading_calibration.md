# Trading rule calibrated on horizon alignment and confidence

| | |
|---|---|
| Date | 2026-11-01 |
| Task | Calibrate a direction-trading rule on horizon alignment and confidence. Where there is a choice, maximise accuracy at the expense of trading volume. |
| Models | Zero-shot Timestral-28M (A0), Timestral-LIG A1 and A2, neural_trade (as-is and daily-retrained, 3-seed ensembles), a price-only baseline |
| Data | Both Test sets of the benchmark: repo (6 days, 2,895 origins) and clean (14 days, 6,715 origins), origins every 3 minutes |
| Code and results | `research/timestral_lig/calibrate.py`, `calib_tables.py`, `calib_curves.py`; `research/timestral_lig/results/calibration.{json,md}`, `calibration_curves.md` |
| Prepared by | Claude Code (AI) |

---

## 1. Answer

**No calibrated rule reaches a reliable accuracy above 50% out of sample. The calibrated answer is "do not trade".**

- **Calibration says no.** We searched up to 270 rules per model. A rule was deployed only if at most 1 of 39 copies of the calibration data, with the labels circularly shifted, produced an equal or better best rule (p_sel ≤ 0.05). The gate rejected 37 of 44 calibrated selections.
- **The 7 deployed rules (6 distinct) fail on new data.** Their Test accuracy is 45.7–62.1% on 29–107 trades each. None is significant, even before correction (smallest one-sided p = 0.12; smallest Holm-adjusted p = 0.26). Pooled over the 6 distinct rules, they are right on 176 of 336 trades: 52.4% (p = 0.21).
- **Calibration accuracy shrinks on Test.** Rules that looked 59–75% accurate on calibration data scored about 46–62% on Test. Rules picked on Test itself score 51–61%, but the best of this many rules reaches that by chance (permutation p = 0.075–1.0).
- **Horizon alignment adds nothing.** Requiring the 1-, 5- and 15-minute forecasts to agree changes Test accuracy by at most 4.4 points, more often down than up (36 of 58 cells down). This test applies to A0, neural_trade and the price baseline only: A1 and A2 make no calls at 5 and 15 minutes.
- **Model confidence adds nothing consistent.**
  - Timestral's most confident 15-minute calls were right 57–61% of the time on the clean Test (89–181 trades, z or pm confidence).
  - Its most confident 1-minute calls there were right 54–56%.
  - On the repo Test the 15-minute filter gave 47–51%, and at 5 minutes on the clean Test it gave 44–48%.
- **The most consistent pattern is price-only, and it is not a model edge.**
  - After a large 5- or 15-minute move, the price tends to reverse. This held in calibration and on Test on both sets.
  - Fading the top 2% of moves was right 63–69% of the time in three of the four Test cells (41–95 trades) and 54.5% in the fourth (clean set, 5 minutes).
  - It made money after 4 bps only on the repo set (+97 bps at 5 minutes and +96 bps at 15 minutes, over 6 days). On the clean set it lost money after costs (−185 bps at 15 minutes, −812 bps at 5 minutes).
  - It is one of about 130 descriptive cells and does not survive correction. The calibrated versions of it scored 52–62% on Test.
- **Costs rule out every deployed rule.**
  - Assume wins and losses equal the average move. Then covering a 4 bps round trip needs about 61–67% accuracy at 15 minutes and 68–80% at 5 minutes.
  - At 1 minute it needs about 90% on the repo set; on the clean set it is not reachable.
  - Every deployed rule loses money at 4 bps (−14 to −658 bps) and at 10 bps.

**Recommendation.** Do not trade any of these signals. If the reversal pattern is worth pursuing, pre-register one rule now and test it on new data (Section 6).

---

## 2. Terms

| Term | Meaning |
|---|---|
| H | Trade horizon in minutes (1, 5 or 15). The rule holds one 1-unit position for H minutes; positions do not overlap. |
| Horizon alignment | `agree3`: the median forecasts at 1, 5 and 15 minutes have the same sign. `path15`: all 15 steps of the median path have the same sign (Timestral only). `none`: no filter. |
| Confidence | `z`: median / robust spread (IQR / 1.349). `pm`: P(up) − 0.5 from the quantiles. `dp`: neural_trade's direction head minus its calibration median. A confidence counts only when it agrees with the trade direction. |
| Keep k | Trade only when the confidence ranks in the top k among the agreeing calls of the last 480 origins. This rank is causal and uses no labels, so a rule carries over when the confidence scale drifts. |
| Follow / fade | Trade with the forecast, or against it. |
| Accuracy | Share of trades with the right sign. Trades with a zero return are left out. |
| W, R | Selection rules on calibration data. W maximises the 95% Wilson lower bound of accuracy (at least 30 trades). R maximises raw accuracy (at least 100 trades). |
| p_sel | Selection-adjusted p-value: (1 + the number of the 39 copies with circularly shifted labels in which the same search scores at least as high) / 40. The minimum is 0.025. |
| P-A, P-B | P-A: calibrate on data before Test, then evaluate on the full Test. P-B: calibrate on the first half of Test, then evaluate on the second half. P-B is the only option for neural_trade, which has no pre-Test forecasts. |
| bps | Basis points, 0.01% |

---

## 3. Method

1. **Rule grid.**
   - Parameters: H × alignment × confidence × keep (100, 50, 30, 20, 10, 5, 2, 1%) × side.
   - Size: 270 rules for the Timestral models, 264 for neural_trade and 96 for the price baseline. k = 100% ignores confidence and is counted once.
   - A1 and A2 shrink their median at 5 and 15 minutes to inside the no-call band (A2 at both horizons; A1 at 15 minutes, and at 5 minutes on the clean set). Their alignment and longer-horizon rules make no trades, so their effective grid is 30–60 rules.
   - Direction comes from the sign of the median forecast. A median within 0.01 bps of the last price is no call.
2. **Calibration data.**
   - A0 and the price baseline: about 19–20k pre-Test origins per set (Train every 2nd minute, V1 and V2).
   - A1 and A2: V1 + V2 (5,760 origins). These blocks steered their training, so this calibration is partly in-sample. The like-for-like reference is "A0 (V1+V2 calib)".
   - P-B: the first half of Test. Calibration labels that reach into the second half are purged.
3. **Selection and gate.**
   - Rules W and R pick the best rule on calibration data.
   - The same search then runs on 39 circular shifts of the labels.
   - The rule is deployed only if p_sel ≤ 0.05; otherwise the decision is "abstain".
4. **Evaluation.**
   - Each deployed rule runs once on the evaluation data.
   - We report accuracy with its Wilson 95% CI and a one-sided binomial p against 50%. Holm correction is applied within each protocol, across both sets and over distinct deployed rules: when W and R pick the same rule, it counts once.
   - We also report gross PnL (sum of log returns) and PnL net of 4 and 10 bps per round trip.
   - Rows with fewer than 30 decided trades are marked as too few.
5. **Descriptive outputs.**
   - The best rule picked on the evaluation data itself, with its permutation p.
   - Accuracy-versus-confidence curves for the follow side, calibration → Test.

---

## 4. Results

### 4.1 Deployed rules (out of sample)

| Set, protocol | Model | Rule (chosen on calibration) | Calib accuracy (p_sel) | Test accuracy [95% CI] | Trades | One-sided p (Holm) | Gross / net 4 bps |
|---|---|---|---|---|---|---|---|
| repo, P-A | A0 zero-shot | H=5, agree3, pm top 1%, follow | 60.4% (0.025) | 45.7% [30.5, 61.8] | 35 | 0.75 (1.00) | −16 / −156 |
| repo, P-A | A1 | H=5, none, z top 2%, follow | 66.2% (0.025) | 49.0% [35.6, 62.5] | 49 | 0.61 (1.00) | −54 / −250 |
| repo, P-A | price-only | H=1, agree3, z top 2%, fade | 60.2% (0.025) | 52.1% [38.3, 65.5] | 48 | 0.44 (1.00) | −57 / −249 |
| repo, P-B | price-only | H=5, none, z top 2%, fade | 75.0% (0.050) | 62.1% [44.0, 77.3], too few | 29 | 0.13 (0.26) | +102 / −14 |
| clean, P-A | price-only | H=5, agree3, z top 2%, fade | 58.9% (0.025) | 56.1% [46.6, 65.1] | 107 | 0.12 (0.49) | −226 / −658 |
| clean, P-B | A0 zero-shot | H=15, agree3, z top 5%, follow | 63.1% (0.050) | 48.5% [37.0, 60.2] | 68 | 0.64 (0.64) | −178 / −450 |

Rule R chose the same A0 rule on repo P-A as rule W, which makes 7 deployments and 6 distinct rules. Every other
selection abstained. For each abstained selection, `calibration.md` shows what its rule would have scored on Test
(34.6–61.3%), as information only.

### 4.2 What the model and price signals do with confidence and alignment (descriptive)

Accuracy of "follow" trades, calibration → Test, for the top 30% / 5% / 2% by confidence (decided Test trades in brackets).

| Signal | Repo | Clean |
|---|---|---|
| A0, H=1 | 51.7→49.4% / 50.6→52.3% / 48.4→44.8% (87) | 51.0→50.4% / 53.2→51.1% / 53.3→55.6% (160) |
| A0, H=15 | 51.2→50.2% / 53.0→50.0% / 52.6→51.1% (45) | 52.1→54.4% / 56.5→57.5% / 56.6→59.6% (94) |
| A0, H=15, agree3 | 51.2→46.0% / 53.2→47.1% / 54.1→48.8% (41) | 52.1→53.6% / 56.6→57.3% / 56.2→58.4% (89) |
| A0, H=5 | 52.5→50.9% / 54.1→52.8% / 57.1→49.1% (57) | 50.6→49.7% / 52.6→43.9% / 53.5→47.5% (122) |
| Price momentum, H=5 (fade = 100 − value) | 46.2→50.0% / 47.0→41.1% / 42.7→31.1% (61) | 46.9→48.2% / 47.8→47.2% / 42.9→45.5% (123) |
| Price momentum, H=15 (fade = 100 − value) | 47.9→50.2% / 48.3→40.2% / 43.1→36.6% (41) | 48.8→43.9% / 46.3→45.6% / 46.9→36.8% (95) |

- **Confidence helps Timestral mainly in one cell family.** At clean-set H=15, accuracy rose to 57–61% with z or pm confidence. It helped weakly at clean-set H=1, top 2% (54–56%). On repo, and at clean-set H=5, it did not.
- **Alignment (`agree3`) shifts accuracy by only a few points either way.**
- **Fading large price moves is the one direction that holds in calibration and on Test on both sets.**
  - For momentum, fewer than 50% of follow trades were right in 73 of 80 nested cells at H=5 and H=15 (calibration and Test, both sets).
  - The strongest Test cells are fade 69.6% (repo, H=5, agree3, top 2%, 56 trades), 68.9% (repo, H=5, top 2%, 61 trades), 63.4% (repo, H=15, top 2%, 41 trades) and 63.2% (clean, H=15, top 2%, 95 trades).
  - These cells are nested and selected after the fact, so they are not independent confirmations.

### 4.3 Rules picked on the evaluation data (descriptive)

The best rule picked on the evaluation data itself scores 51.5–61.0%. That is within the range the same search
reaches on label-shifted data (permutation p = 0.075–1.0). The high accuracies quoted for single cells measure selection bias, not skill.

---

## 5. Why no rule survives

- **No forecaster has direction skill at these horizons** (zero-shot report, Section 7.1, and `plans/timestral_lig.md` 4.8).
  - Filtering the noise more finely gives fewer trades, not better trades.
  - Where Timestral's confident calls look good, they largely repeat a reversal of the last 15-minute move.
  - Its most confident calls (top 5%) pointed against the last 15-minute move in 58–70% of cases, and in 55–66% of the trades actually taken. For all its calls the figure is 55–60%.
  - The tilt is largest where the calls looked good: clean-set H=15, 70% against 60%.
- **The reversal pattern is weak and unstable.**
  - For the top 10% of moves it held in the repo set only in the first half of Test (55–58%, then 48–50%).
  - For the top 2% it stayed above 50% in all eight half-Test cells (53–75%, 17–64 trades each), but these samples are small.
  - On the clean set, losing fade trades were larger than winning ones. The deployed 56% rule lost 226 bps gross. The top-2% 5-minute fade had an average loss of 14.5 bps against an average win of 7.4 bps.
  - The break-even accuracy at 4 bps, from the actual win and loss sizes of the top-2% fade, is 59–85%.
- **Costs dominate.**
  - The average absolute move is 3–5 bps at 1 minute and 7–11 bps at 5 minutes.
  - A 4 bps round trip needs about 61–67% accuracy at 15 minutes and 68–80% at 5 minutes.
  - Only cells picked after the fact on the 6-day repo set cleared that bar: the top-2% fades, +97 and +96 bps net.
  - The best deployed rule ended at −14 bps, and the same fades lost money on the clean set.

## 6. Changes during the work, and next step

- **A first version had flaws that an internal review found.**
  - The half-Test split had no purge.
  - neural_trade's direction head is centred at 0.31–0.44, so its confidence picked the most bearish moments.
  - Absolute thresholds did not carry over when the confidence scale drifted.
  - There was no "abstain" option.
  - Nominal p-values were shown for rules picked on Test.
- **The fixes:**
  - a purge at the split;
  - a centred, agreement-signed head confidence;
  - causal rank thresholds;
  - the selection-adjusted gate;
  - permutation p-values;
  - Holm correction;
  - seed ensembles for neural_trade.
- **The first version is kept only as a record:** `results/calibration_v1.md`.
- **Next step, only if wanted.** Pre-register one price-only rule now, exactly as scored here:
  - z = last 15-minute return / (EWMA σ x √15);
  - trade when its rank among the trailing 480 origins is in the top 2%;
  - no alignment filter; fade the move; hold 15 minutes; no overlapping positions.
  Score it once on at least 4 weeks of new data, gross and net of 4 bps.
  - On the two Test sets it scored 63.4% (+96 bps net at 4 bps) and 63.2% (−185 bps net).
  - It needs about 59–69% accuracy at 15 minutes to cover 4 bps, depending on the win and loss sizes.
  - Nothing here suggests it will get there reliably.
