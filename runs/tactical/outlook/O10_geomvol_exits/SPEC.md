# O10: geometry and volatility as the EXIT of a trade, not the entry

Outlook lead, 2026-10-14. Committed BEFORE any number of this study is computed. Owner (2026-10-14): "Возьми геометрию и волатильность
как завершение сделки а не открытие". Tactical, exploratory; CPU only.

## Idea
The direction source is the regression gate (O9: it carried the ROBUST candidate). Geometry and volatility forecast the SIZE and
the SHAPE of the next 15-60 minutes (O1, O5, H72/T01), which is what an exit needs: how much of the move is left, and when the
trend bends. So:
- **The entry is fixed:** ROBUST's entry from O9 (`choice_robust.json`, ca1bd23). That is a breakout stop at m(1 +- 1.0 sigma_60),
  gated by p_up >= 0.52 for longs and <= 0.48 for shorts, size 1x, one position at a time, valid 15 bars.
- **Only the exit varies.** The baseline E0 is ROBUST's own exit: stop 1 sigma, target 2 sigma, time stop 60 bars.

## Exit rules
Every rule keeps the protective stop at 1 sigma_60 from the entry (PESS-checked in the entry bar, as O8/O9) and a maximum holding time T.

| id | exit rule (in addition to the 1-sigma stop and the time stop T) |
|---|---|
| E0 | target at 2 sigma_60 (ROBUST, the baseline) |
| E1c | **vol trailing stop:** the stop trails the best price since entry at c * sigma_15(t), where sigma_15 is refreshed at every 15-minute decision time from L1's 15-minute forecast; c in {1, 2} |
| E2a | **1-minute geometry against the position:** close the position when the close crosses the 1-minute EMA20 against the side (the `pos` channel of `ema20` changes sign against the side) |
| E2b | **15-minute geometry against the position:** at a 15-minute decision time, close if the slope3 of the 15-minute EMA20 (O1's top feature `g15.ema20.slope3`) points against the side |
| E3 | **volatility contraction:** at a 15-minute decision time, close if dV (the H72/T01 forecast of the next-hour change of ln RV) < 0 |
| E4 | **learned exit (geometry + volatility):** at each 15-minute decision time while in a position, a logistic model predicts whether holding to the next decision time earns > 0. Features: O1's top-15 geometry features signed by the side, ln sigma_15, ln sigma_60, dV, the unrealised return in sigma_60 units, and minutes in trade. Close if P < 0.5. Monthly walk-forward on the past E0-entry positions' checkpoints whose next checkpoint closed before the month. |

- **T in {60, 240}** bars: dynamic exits may hold longer than ROBUST's 60.
- **Grid:** {E0, E1(c=1), E1(c=2), E2a, E2b, E3, E4} x T{60, 240} = 14 configurations.
- **Signal timing:** every exit signal is computed on CLOSED bars (bars <= t-1 at the check at bar t). The exit is a market exit at
  the open of the next bar. Stops are intrabar with PESS rules (as O9).
- **Costs:** 2 bps round trip is primary; 0 and 5 bps are reported.

## Spans and selection (as O9)
- **Selection:** 2021-01 .. 2024-06 (42 months), PESS fills, 2 bps. The rule is the highest t-statistic of the monthly net return.
  `choice.json` is committed alone before anything else, and a git gate (as O8/O9) refuses `describe` and `freeze` without it.
- **Descriptive only (already seen by O8/O9):**
  - A (2018-07..2020-12);
  - B (2024-07..2025-07-20);
  - C (Binance 2025-10-05..2026-10-12, models frozen to 2025-07-20).

  For the choice: E0, the choice, SHUF (the exit signals permuted within the month, the entries unchanged), and the paired monthly
  difference choice - E0 with its 95% t-interval. The figures also show mean holding time, average win and loss, and exit reasons.

## Prospective verdict (frozen now; scored once later, together with O9)
- **Freeze:** all models are fitted once on the Binance file to 2026-09-30 23:59 (as O9). E4's model is fitted once on the
  checkpoints of that year. The artefact is `frozen_P.json` + sha256.
- **Scoring:** `score_P.py` scores 2026-11, 2026-12 and 2027-01 once, from daily net returns with a 5-day block bootstrap (10,000
  draws, seed 0). It refuses short data and a second run.
- **Verdict "geometry/volatility exits add":** the choice minus E0 (paired daily) has a 95% lower bound > 0.
- **Verdict "the strategy works":** the choice's mean > 0 with its lower bound > 0, and the choice minus SHUF has its lower bound > 0.
- **Also reported:** the owner's target (hit > 60 %, drawdown < 5 %).

## Tests (seconds)
- No exit signal uses the bar it acts on or any later bar.
- Each exit rule behaves correctly on hand-made bars:
  - the trailing stop only moves in the position's favour;
  - an EMA cross against the side closes at the next open;
  - E2b, E3 and E4 act only at 15-minute decision times.
- E4 trains on checkpoints that ended before the month.
- The freeze reads no bar after 2026-09-30 23:59, and no cut-file bar of 2025-07-21..2025-09-30.
- E0 reproduces O9's ROBUST trades on span B exactly (same entries, exits and returns).
- The choice gate works, and `score_P.py` refuses on the current file.

## Outputs
`runs/tactical/outlook/O10_geomvol_exits/`: code, `selection.jsonl`, `choice.json`, `describe.jsonl`, `tables.txt`, `trades_*.csv.gz`,
`frozen_P.json` (+ sha256), `score_P.py`, figures (equity and drawdown; the distribution of holding times and of the exit R-multiples),
`windows.json` (real example trades with the exit signal drawn, from cut-file bars <= 2025-07-20), `run.log`, `tests.log`, `REPORT.md`.

## Addendum (lead, 2026-10-14, after describe and BEFORE any prospective number)
- **The selection chose E0_T240.** That is a price-level exit; only its time stop is longer than ROBUST's. So the verdict "geometry/volatility
  exits add" (choice - E0) would test the time stop, not geometry.
- **The geometry/volatility verdict is named now, before scoring:** GV - E0 at the same T (GV = E4_T60, the best non-E0 exit by
  selection t; E0 = E0_T60), paired daily, with a lower bound > 0.
- **Choice - E0 stays reported** under its own name: "longer time stop adds".
- `score_P.py` already computes both; this addendum only names which one carries which verdict.
