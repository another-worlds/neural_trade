# Outlook session journal (tactical, started 2026-10-13)

Owner, 2026-10-13: "Я хочу позитивный аутлук насчет геометрии, физики, регрессии и синтетических интерпретируемых
индикаторов. Что тут получилось хорошо и где у нас перспективы. Этим займется следующая сессия ... У кого что получается -
тот то и делает".

Rules: [docs/TACTICAL.md](../../docs/TACTICAL.md), in particular:
- "Division of work" (the parallel sessions, folders, branches);
- "Report style" (the owner's favourite HTML report);
- the hard cutoff 2025-07-20;
- the forward data, a separate Binance file 2025-10..2026-10-12, already used by H82-H86;
- SPEC before results, next-open fills, thresholds from the past.

## Start here

1. **Read:**
   - [OUTLOOK_positive.md](OUTLOOK_positive.md): what worked in geometry, physics, regression and synthetic indicators, and five prospects with their deciding tests;
   - [REVIEW_2026-10-13.md](REVIEW_2026-10-13.md): what failed, why, and the untested variants U4-U11;
   - the hypotheses table of [LOG.md](LOG.md) (H5-H86) for the evidence behind each claim.
2. **Choose and pre-register.** Choose the first prospect, write its SPEC under `runs/tactical/outlook/<id>/` and commit it before any run.
3. **Remember what is spent.** The Binance forward months 2025-10..2026-09 have been seen by H82-H86. A new unseen test needs either untouched early history (2017-08..2018-12 for targets not used there) or the months after 2026-10 (prospective, as T04-B).

## Handoff

_Rewritten at the end of every outlook session._ **2026-10-13 (session 1).**

**State:**
- O1-O7 are done and recorded. Nothing is running.
- GPU used today: O2, 2.4 h wall (4.7 h process).
- Report: `runs/tactical/report/outlook_report.html`, built by `build_outlook_report.py`.

**What the evidence says (positive outlook, honestly bounded):**
- **Geometry, the one confirmed positive.** 15-minute EMA trend curvature adds to the forecast of the SIZE of the next 15-60 minutes:
  - over S3 (O1, QA PASS): look and forward both pass;
  - over the best linear base, BANK (O5, QA PASS): +0.0056 [+0.0024, +0.0088] at H = 60, +0.012 at H = 15, judged on 2018-07..2020-12.
- **Limits:**
  - the effect is small, about 0.5-1 Spearman points;
  - it ages within a forward year (O1 QA);
  - it does not hold over a boosted base on the forward months (O1);
  - hourly geometry HURTS (O6);
  - as 90% intervals it gains only 0.6% over S3 and does not beat T03's HAR form (O7).
- **Physics:** on the variance head it is a small early-training effect; 8 epochs = 4 epochs within noise, and the HAR gap stays (O2).
- **Synthetic learned kernels:** they learn nothing beyond HAR (O3).
- **BANK (a denser HAR):** not confirmed retrospectively (O4, a SPEC design error of the lead: 10.7 training days in 2017-09).
- **Direction:** no outlook study touched direction.

**Pending (score once, never refit):**

| study | `score_B.py` | months |
|---|---|---|
| O4 | `O4_bank_confirm/score_B.py` | 3 complete months from 2026-10 |
| O5 | `O5_geom_bestbase/score_B.py` | 3 complete months from 2026-10 |
| O6 | `O6_geom_hourly/score_B.py` | 3 complete months from 2026-10 (moot after the fail) |
| O7 | `O7_geom_intervals/score_B.py` | calibrated on 2026-10, scored on 2026-11..2027-01 |

They need the Binance file extended past 2026-12. That extension is the owner's forward-data permission (TACTICAL.md "Forward data").

**Next candidates (not started):**
1. The regression prospect: a readable per-period coefficient atlas of L1 (what geometry the market rewards, by year), descriptive only.
2. Geometry for H = 240 by another route: daily-cycle-aligned features, not geometry.
3. The combined readable model (regression side + L1 size) judged on CRPS, when a forward period is available.

None of them looks strong. Ask the owner before spending GPU.

**Housekeeping:**
- The merge of `origin/nt-tactical` into this branch was blocked by the permission classifier (5 commits behind, H87/T05). The owner was asked.
- The pinned worktree `D:/nt/nt_o2` (O2's code, nothing heavy) was left in place for the owner to remove.

## For the MVP lead (outlook)
- **2026-10-13 (O1, O5, O7):** geometry features of the last 60 bars of 15 minutes (EMA slopes and positions; `geom2/geomlib.geom_rich`) add about +0.005 Spearman (H = 60) and +0.012 (H = 15) to a linear HAR-type forecast of realised volatility. This held:
  - on the look;
  - on the Binance forward months;
  - on 2018-07..2020-12 over a denser HAR.

  Candidate (not urgent): the variance head / conformal scale (D-008) could take these as inputs. Needs a paired MVP test (D-025). As intervals the gain is 0.6% and is not better than HAR (O7), so the expected value for trading is small.

## Hypotheses

| # | Date | Hypothesis | Method | Cost | Result | Evidence |
|---|---|---|---|---|---|---|
| O1 | 2026-10-13 | Prospect 2: does indicator geometry (1-minute window, and new: 60 bars of 15 minutes) add to a FAIR volatility base (S3: HAR + volume/range + hourly RV lags), on the size of the next 15/60/240 minutes? | SPEC 8d9d20c: ridge R0 = S3, R1 = +G1, R2 = +G1+G15, shuffled control; monthly walk-forward, selection 2021-01..2024-06, one look 2024-07..2025-07-20, Binance forward 2025-10..2026-09 frozen | 26 min CPU | **HOLDS (narrowly, on the linear base).** G* = R2 (15-minute geometry adds). H = 60 gain over S3: look +0.0054 [+0.0032, +0.0076] 12/13, forward +0.0052 [+0.0029, +0.0075] 12/12, QLIKE better; H = 15 +0.008..+0.014 every month; H = 240 nothing; no gain on ln|r|. Top features: 15-minute EMA20/40 slopes and positions (trend curvature). **Caveat:** boosted S3 beats linear S3 by +0.020 forward, and against it geometry is -0.0030 forward: not yet a gain over the best base (O5). Sanity gate ok. **QA PASS** (exact recompute, no leakage); the size bar is a knife edge: forward gain decays from ~+0.009 to ~+0.0025 over the year, and without the partial 2025-10 it is +0.0046 [+0.0025, +0.0067] 11/11. | runs/tactical/outlook/O1_geom_vol/REPORT.md |
| O2 | 2026-10-13 | Prospect 3: do the physics terms help the variance head more when the net trains 8 epochs instead of H75/H77's 4? | SPEC 287bbc6: PRICE_HEAD none, physics on/off, 6 climb slices x seeds 0, 1, 8 epochs; phys7 scoring; d8 - d4 on the same seeds | 2.4 h GPU wall (4.7 h process; runs ~11-15 min, the owner's shrink exception) | **NO DETECTABLE EFFECT at 8 epochs.** On - off +0.0094 [-0.0068, +0.0257], 3/6 slices (4 epochs on the same seeds: +0.0157, 5/6); d8 - d4 -0.0062 [-0.0238, +0.0114]: same within noise, leaning to shrink. Training longer does not close the HAR+tod gap (-0.063 on, -0.073 off). The physics gain on the variance head is not strengthened by training; it looks like a small early-training effect. All 24 runs finite. | runs/tactical/outlook/O2_phys_var8/REPORT.md |
| O3 | 2026-10-13 | Prospect 4: synthetic interpretable volatility indicators = learned memory kernels (softmax mixture over an EWMA bank of r^2, range, volume); do they add over S3 and over a linear readout of the whole bank? | SPEC 37b58af; as O1's spans and criterion; controls BANK (ridge on S3 + 36 EWMA logs) and shuffled; numpy | 20 min CPU | **No; synthetic adds: no.** LK - R0 at H = 60: look +0.0004 [-0.0002, +0.0009], forward -0.0008; LK - BANK look -0.0065 (0/13). Kernels stayed near uniform (HAR's short+long tilt only). **Found on the way (not pre-registered):** the control BANK beats S3 at every span and horizon: H = 60 look +0.0068 [+0.0047, +0.0089] 13/13, forward +0.0073 [+0.0049, +0.0096] 12/12, QLIKE better; a denser HAR, not a synthetic indicator. Confirmation: O4. Sanity gate ok (-0.0006). | runs/tactical/outlook/O3_synth_volind/REPORT.md |
| O4 | 2026-10-13 | Is O3's BANK gain real? Test where no choice was made | SPEC d8712e6: A retrospective 2017-09..2018-12 (16 months), BANK - R0 at H = 60 >= +0.005 with CI > 0, >= 11/16 months, QLIKE not worse; B frozen on Binance 2025-10..2026-09 for months from 2026-10 | 1 min CPU | **A: not confirmed.** -0.0043 [-0.0266, +0.0181], 14/16 months positive, QLIKE -0.0007. One month sinks it: 2017-09 trained on 10.7 days (52 features) and lost 0.160; the SPEC's short-history rule was the lead's design error. Post hoc without it +0.0061 [+0.0030, +0.0093] 14/15 (not a verdict). **B frozen** (sha256 83dd3ad7...); score_B.py scores 3 complete months from 2026-10 once. | runs/tactical/outlook/O4_bank_confirm/REPORT.md |
| O5 | 2026-10-13 | Does geometry add over the BEST volatility base (BANK ridge or HGB on S3 + bank, chosen on 2021-01..2024-06), on months that chose nothing? | SPEC (this commit): judgement 2018-07..2020-12 (30 months, >= 318 training days each); pass = gain >= +0.003 with CI > 0, >= 20/30 months, QLIKE not worse; prospective freeze for 2026-10+ | 17 min CPU | **Geometry adds over the best base.** B* = BANK ridge (HGB lost on selection). L1 - L0 at H = 60 on 2018-07..2020-12: +0.0056 [+0.0024, +0.0088], 23/30, QLIKE -0.0116 (27/30); H = 15 +0.0122 30/30; H = 240 nothing. BANK itself beats S3 here (+0.0054). Weak stretch 2019-11..2020-02. **QA PASS** (exact recompute, no leakage; LOO min +0.0049; on the 24 months O4 never touched +0.0055 [+0.0018, +0.0091]). Prospective freeze sha 79f7f25d... (score_B from 2026-10). | runs/tactical/outlook/O5_geom_bestbase/REPORT.md |
| O6 | 2026-10-13 | The geometry ladder: does geometry on 60 hourly bars (G60) help at H = 240, where 1- and 15-minute geometry did not? | SPEC (this commit): L2 = O5's L1 + G60 vs L1, judged on 2019-01..2020-12 (second look at O5's span, new feature block; caveat), H = 240 gain >= +0.003 with CI > 0, >= 16/24, QLIKE not worse; shuffled control; prospective freeze | 36 min CPU | **Fail: the ladder stops at 15 minutes.** H = 240 L2 - L1 -0.0143 [-0.0231, -0.0055], 8/24; H = 60 -0.0039; negative on every span and horizon; shuffled control -0.0009 (gate ok). Hourly-EMA coefficients large and collinear. | runs/tactical/outlook/O6_geom_hourly/REPORT.md |
| O7 | 2026-10-13 | Prospect 5 (judge as a risk model): do geometry-enhanced vol forecasts (O5's L1) give sharper 90% return intervals than S3 at the same coverage? | SPEC (this commit): T03's cal-interval method, monthly walk-forward, judgement 2019-02..2020-12 (a further look at O5's months, downstream claim; caveat), T = 60 interval score L1 < R0 with CI < 0, >= 15/23 months, coverage in [0.88, 0.92]; prospective on O5's freeze | 4 min CPU | **Pass, tiny; not better than HAR.** T = 60 interval score L1 - R0 -1.75 bps [-2.64, -0.86] (0.6%), 18/23, coverage 0.8925; T = 15 -1.84 (23/23); T = 240 nothing. L1 vs T03's HAR form: +0.65 [-3.97, +5.28] at T = 60 (no difference). Prospective scoring from 2026-11. | runs/tactical/outlook/O7_geom_intervals/REPORT.md |
| O8 | 2026-10-13 | Owner: build a strategy on the new approach and backtest it on independent data. Vol-band orders at k*sigma_hat (L1 size forecast): reversion (REV) or breakout (BRK), chosen on 2021-01..2024-06 | SPEC (this commit): 24 configs, choice committed before tests; tests B 2024-07..2025-07-20 + C Binance 2025-10..2026-10-12 (frozen sigma), A 2018-07..2020-12 separately; pessimistic 1-minute fills, 2 bps; arms L1 / R0 (no geometry) / TRAIL / SHUF (null); works iff mean monthly return > 0 and > SHUF with CI > 0; geometry adds iff L1 > R0 with CI > 0 | CPU | SPEC committed | runs/tactical/outlook/O8_volband_strategy/ |
