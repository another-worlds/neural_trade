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

_Rewritten at the end of every outlook session._ 2026-10-13: started; O1 (CPU) and O2 (GPU) running, O3 SPEC committed.

## Hypotheses

| # | Date | Hypothesis | Method | Cost | Result | Evidence |
|---|---|---|---|---|---|---|
| O1 | 2026-10-13 | Prospect 2: does indicator geometry (1-minute window, and new: 60 bars of 15 minutes) add to a FAIR volatility base (S3: HAR + volume/range + hourly RV lags), on the size of the next 15/60/240 minutes? | SPEC 8d9d20c: ridge R0 = S3, R1 = +G1, R2 = +G1+G15, shuffled control; monthly walk-forward, selection 2021-01..2024-06, one look 2024-07..2025-07-20, Binance forward 2025-10..2026-09 frozen; pass = Spearman gain >= +0.005 with CI > 0 and QLIKE not worse at H = 60 | CPU | running (implementer) | runs/tactical/outlook/O1_geom_vol/ |
| O2 | 2026-10-13 | Prospect 3: do the physics terms help the variance head more when the net trains 8 epochs instead of H75/H77's 4? | SPEC 287bbc6: PRICE_HEAD none, physics on/off, 6 climb slices x seeds 0, 1, 8 epochs; phys7 scoring; d8 - d4 on the same seeds | ~3 h GPU wall (estimate) | running (experimenter) | runs/tactical/outlook/O2_phys_var8/ |
| O3 | 2026-10-13 | Prospect 4: synthetic interpretable volatility indicators = learned memory kernels (softmax mixture over an EWMA bank of r^2, range, volume); do they add over S3 and over a linear readout of the whole bank? | SPEC 37b58af; as O1's spans and criterion; controls BANK (ridge on S3 + 36 EWMA logs) and shuffled; numpy | 20 min CPU | **No; synthetic adds: no.** LK - R0 at H = 60: look +0.0004 [-0.0002, +0.0009], forward -0.0008; LK - BANK look -0.0065 (0/13). Kernels stayed near uniform (HAR's short+long tilt only). **Found on the way (not pre-registered):** the control BANK beats S3 at every span and horizon: H = 60 look +0.0068 [+0.0047, +0.0089] 13/13, forward +0.0073 [+0.0049, +0.0096] 12/12, QLIKE better; a denser HAR, not a synthetic indicator. Confirmation: O4. Sanity gate ok (-0.0006). | runs/tactical/outlook/O3_synth_volind/REPORT.md |
| O4 | 2026-10-13 | Is O3's BANK gain real? Test where no choice was made | SPEC (this commit): A retrospective 2017-09..2018-12 (16 months), criterion BANK - R0 at H = 60 >= +0.005 with CI > 0, >= 11/16 months, QLIKE not worse; B frozen on Binance 2025-10..2026-09 for months from 2026-10 | CPU | SPEC committed | runs/tactical/outlook/O4_bank_confirm/ |
