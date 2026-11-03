# Pre-registration: the frozen daily trend rule on new data

Written 2026-11-03, before any PnL on a new market or a new period was computed. Code: `nd_fetch.py` (data),
`nd_run.py` (engine H and scoring). `nd_run.py score` refuses to run unless this file is committed unchanged at HEAD,
research/pnl has no uncommitted or untracked file, and every data file matches the SHA-256 manifest committed with
this file (`results/newdata_manifest.json`). It runs once: the lock (`work/nd/score_lock.json`, copied to
`results/newdata_lock.json` with the HEAD commit) blocks a second run unless `ND_RESCORE=1` is set AND CHANGES.log has
a NEW 'newdata rescore' entry that gives the reason (one entry per re-run; the committed copy counts as a run).

Before this file, only one PnL computation was run: `nd_run.py validate`. It runs engine H on BTCUSDT over the
already-scored dev and sealed windows and compares with engine E. It reads no new market and no new period.

## 1. Question

Does T2-P, the study's Track-2 candidate, make money on data that the study never scored?
T2-P is used exactly as frozen in SPEC 'track2' (`rules.py`), with no change and no new parameter:
tau = mean over L in {7, 14, 28, 56} days of sign(ln C_d - ln C_{d-L}); x* = round(tau x min(1, 0.5 / sigma_ann), 1);
trade at the 00:00 UTC daily close if |x* - x| >= 0.2. Long or short, at most 1x.

## 2. New data

1. **New markets (primary).** Panel P: every USD-M USDT-margined perpetual in the data.binance.vision archive listing
   (delisted symbols included, so there is no survivorship filter) whose first 1d monthly archive is for a month
   <= 2020-12, except BTCUSDT. The rule uses listing dates only. N = 80:
   1INCH AAVE ADA ALGO ALPHA ATOM AVAX AXS BAL BAND BAT BCH BEL BLZ BNB BZRX COMP CRV CTK CVC DASH DEFI DOGE DOT EGLD
   ENJ EOS ETC ETH FIL FLM FTM GRT HNT ICX IOST IOTA KAVA KNC KSM LEND LINK LRC LTC MATIC MKR NEAR NEO OCEAN OMG ONT
   QTUM REN RLC RSR RUNE SKL SNX SOL SRM STORJ SUSHI SXP THETA TOMO TRB TRX UNI VET WAVES XLM XMR XRP XTZ YFII YFI ZEC
   ZEN ZIL ZRX (each + USDT).
   LENDUSDT stopped trading in 2020-10 (it migrated to AAVE), so it never enters the window: 79 markets can be active.
   Window: 2021-01-01 .. 2026-10-31 UTC (the study's dev and sealed windows plus 2026-10-31; the study used none of
   these markets).
2. **New periods for BTC (secondary).**
   - (a) BTC before the study: Binance spot BTCUSDT, 2017-08 .. 2019-12-31 (first decision on day 56 of the data). No
     funding: the perp did not exist for most of it. A price-path test only.
   - (b) BTC 2020: BTCUSDT perp with funding, 2020 (first decision on day 56). In the study's data range, but T2-P was
     never scored on it (dev OOS started 2021-01-01).
   - (c) BTC after the seal: 2026-10-31 .. 2026-11-02, the only complete days after the sealed year. The run starts
     2026-01-01, so the position carries over from the seal. Descriptive only (3 days cannot test anything).

## 3. Engine and costs

- **Engine H** (`nd_run.py`): hourly bars; decisions only at the 23:00 bar close (the 00:00 UTC daily close); funding
  per archive print, paid by the position held over the print; the 1x cap and trims of engine E (|x| > 1.10 at any bar
  close, > 1.02 at a decision bar where the rule does not trade; as in E, the trim acts first and the rule decides on
  the trimmed exposure); positions closed at the end of the run or of a data segment. Engine E checks trims every
  minute, engine H every hour, so H does not see wicks inside the hour: that helps a short on a spike that reverts
  inside the hour and hurts it on a squeeze that persists. The net direction is not known.
- **Data rule.** Hourly bars with zero quote volume are dropped: the archive keeps flat zero-volume bars after a
  delisting (seen before scoring, e.g. FTM after 2025-01, BLZ after 2024-12). So a market ends at its last traded
  hour, and its position is closed there at the last traded close (the settlement price is not in the data). A gap
  > 7 days splits a segment; a new segment needs 56 days of warm-up again. "Survivor" = last traded bar on or after
  2026-10-30. Renamed markets (EOS, MATIC, FTM and others whose ticker changed) count as delisted.
- **Archive holes.** The monthly 1h archives leave out some days that the daily archives contain (2022-02-26..28 and
  2022-04-01..02 for 30 panel markets); `nd_fetch.py` fills every short day from the daily archive. 2026-10-28 is in
  neither archive for any market: on that day there is no decision and no return, the position carries, and the move
  is booked on 10-29; funding prints inside a hole are paid by the position held across it. A daily-archive bar is
  merged only if no monthly bar lies within an hour of it. BTC (a): after the 2018-02-08 exchange outage, 43 spot bars
  are stamped at :28 (2018-02-09 09:28 .. 02-11 03:28), so 2018-02-08..10 have no 23:00 decision bar (3 decisions
  skipped).
- **Validation against engine E** (BTC, C7-B costs, maker 2 bps in / taker 5 bps out): see section 6.
- **Costs on the panel.** Primary: 5 bps per unit of |x| increased, 10 bps per unit reduced (engine E's maker 2 and
  taker 5 bps, plus 3 and 5 bps for wider spreads and chased fills on smaller markets). Sensitivity: BTC level (2 / 5)
  and stress (25 / 25). BTC sets use 2 / 5, as in the study.
- **Funding.** Archive prints. A month without a funding archive is charged 0, and the number of coin-days affected is
  reported.

## 4. Portfolio

Equal capital per active market, rebalanced daily: the portfolio's daily simple return is the mean of the active
markets' daily net simple returns. A market is active from its first decision day (cash on that day) to its last day
in the window or in its data segment. Days with no active market return 0.

## 5. Primary test and verdict (one test, no correction)

Statistic: the mean daily net log return of the panel's T2-P portfolio, primary costs, 2021-01-01 .. 2026-10-31.
Stationary bootstrap (`stats.stationary_bootstrap`), B = 10,000, mean block 20 days, seed 0.

- **REPLICATED**: the one-sided 95% lower bound of the mean is > 0.
- **CONSISTENT**: the mean is > 0, the lower bound is <= 0.
- **NOT REPLICATED**: the mean is <= 0.

Secondary verdicts with the same rule (reported, not gates): BTC (a) and (b); the panel's dev-window part
(2021-01 .. 2025-10) and sealed-window part (2025-11 .. 2026-10); the 20 most liquid markets (median daily quote volume
2020-07-01 .. 2020-12-31, among markets with at least 90 days of bars in that period, so that a few launch days
cannot rank a market; chosen before the window): ETH LINK YFI LTC SUSHI XRP UNI BCH BNB YFII ADA DOT EOS SXP TRX CRV
XTZ WAVES BAND XLM; the panel without the index contract DEFIUSDT. Subset statistics use only days on which at least
one of the subset's markets is active.

Descriptive: returns by year; long and short PnL, funding and fees (plus a write-off term on a -100% day); turnover;
share of markets with a positive compound net (a market with a -100% day counts as not positive); median market
Sharpe (of daily simple returns); the three cost levels; survivors against delisted markets; tsmom_unscaled, voltarget_long, B&H
and cash on the same panel; the correlation of the panel's T2-P with BTC's T2-P.

What each verdict means for the outlook (fixed now):
- REPLICATED: the rule's edge carries over to other crypto markets. It is still not proof for BTC alone, and the
  markets move together.
- CONSISTENT: the same state as on BTC: possible, not proven.
- NOT REPLICATED: the BTC result is likely specific to BTC or luck; the outlook goes down.

## 6. Engine-H validation (run before this file; BTC only)

`results/newdata_validation.json`. Engine H against engine E, BTCUSDT, C7-B costs (maker 2 bps in, taker 5 bps out):

| rule | dev 2021-01..2025-10, %/yr: H / E (gap) | sealed year, total log: H / E (gap) |
|---|---|---|
| T2-P | 12.75 / 12.69 (+0.06) | 0.336 / 0.332 (+0.004) |
| tsmom_unscaled | 17.16 / 17.20 (-0.05) | 0.349 / 0.344 (+0.004) |
| voltarget_long | 15.27 / 15.27 (0.00) | -0.444 / -0.445 (0.000) |
| B&H | 15.69 / 15.69 (0.00) | -0.387 / -0.386 (-0.001) |

The study's own reconciliation bound is 0.5% of equity per year; every gap is inside it. Sharpe agrees to 0.01.

## 7. Known limits, disclosed before the result

- **Not blind to the market path.** We know the broad 2021-2026 path: most of these coins fell far below their 2021
  highs, and several were delisted. A long/short trend rule earns on long falls. The rule was fixed before (SPEC,
  2026-11-01), and no parameter is chosen here, but the designers expected falls.
- **Correlated markets.** Daily returns of these coins move with BTC and with each other. The panel is not 80
  independent tests. The bootstrap resamples days of the portfolio, so it keeps the cross-market correlation.
- **Execution.** Hourly engine; assumed costs; no market impact; equal weights ignore the small capacity of thin
  markets; the delisting settlement price is not modelled (the last archive close is used).
- **Prior knowledge of the method.** Time-series momentum and trend following on crypto markets are documented; this
  test checks one fixed version, not the idea.
