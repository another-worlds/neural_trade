# O9 report: an ensemble breakout strategy on the found strengths (phases 1 and 2)

SPEC `d721cfe` + addendum `4f3a9eb`. Choices: `choice.json` (38e08d6, PRIMARY, pre-registered rule), `choice_robust.json`
(ca1bd23, ROBUST, chosen AFTER the selection table). Code and results: a05c55e (selection), 1e37936 and 76a0bc4 (describe, freeze,
score_P). Tests: 27 passed. CPU only. Written by the outlook lead from the implementer's text (its Write tool refused the file).

## Status: no verdict yet
- **A, B, C are descriptive only.** O8 had already seen these spans, and the ensemble's design came from O8 and the selection table.
- **The verdict is prospective.** `score_P.py` scores 2026-11, 2026-12 and 2027-01 once, after the forward file reaches 2027-02-01 01:15.
  - Arms: both candidates, their SHUF arms and O8's breakout as the baseline.
  - Basis: PESS fills, 2 bps, daily net returns, a 5-day block bootstrap, Holm over the two candidates.
  - Today it refuses: no complete month exists.
- **Freeze:** `frozen_P.json`, sha256 `9c5259a60e0c2c6e966f530585bb354e881ee1d53f3a5098ff01b3d8bb055195`.
  - Data: Binance 2025-10-01..2026-09-30, no refit.
  - sigma_ref comes from 2026-10 at scoring time.

## The ensemble
At each 15-minute decision:
1. **Entry:** O8's breakout stop orders at m(1 +- k*sigma_60), with sigma_60 the L1 size forecast (geometry).
2. **Gates:**
   - a regression direction gate: long needs p_up >= 0.52, short needs p_up <= 0.48;
   - and/or a volatility-expansion gate (dV >= 0, the H72/T01 forecast).
3. **Exits:** stop 1 sigma, optional target 2 sigma, time stop 60 bars.
4. **Size:** 1x or vol-targeted (capped at 2).

| candidate | rule | selection 2021-01..2024-06 (PESS) |
|---|---|---|
| PRIMARY | k 1.0, dV gate, vol-targeted, no target | +3.76 %/mo at 2 bps (t 4.09), -1.08 % at 5 bps, DD 36 % |
| ROBUST | k 1.0, regression gate 0.02, target 2 sigma, 1x | +1.82 %/mo at 2 bps (t 3.20), +0.19 % at 5 bps, hit 46 %, DD 16 % |

## Descriptive results (PESS, 2 bps, mean monthly net %)
| span | PRIMARY | ROBUST | ROBUST SHUF | O8 baseline | buy-and-hold |
|---|---|---|---|---|---|
| B 2024-07..2025-07 | +1.54 [-1.74, +4.83] | **+2.12 [+0.41, +3.84]**, hit 49 %, DD 9 % | -3.13 | +1.18 | +63.9 % |
| C 2025-10..2026-10-12 | -0.66 [-3.97, +2.64] | +0.68 [-0.56, +1.91], hit 46 %, DD 6.1 % | -1.61 | -3.30 | -33.2 % |
| A 2018-07..2020-12 | +2.40 [-0.46, +5.26] | +0.21 [-2.14, +2.56], DD 43 % | -6.84 | -4.29 | +152 % |

## Lead's reading
- **ROBUST is the more interesting candidate:**
  - it is the only one that stays positive on B and C under pessimistic fills;
  - its drawdown is 6-9 % there;
  - shuffling its signals turns it negative on every span.
  - **The regression direction gate carries it:** at 5 bps it beats the same strategy without the gate on all three spans (B +0.61 vs -1.20, C -0.83 vs -9.18, A -1.34 vs -6.95).
  - **The weakness:** its interval includes 0 on C and A, and it was chosen after the table.
- **PRIMARY** is cost-fragile and flat-to-negative on C.
- **Geometry** in sigma adds nothing consistent here (ROBUST vs NO_GEOM: +0.33 on B, -0.19 on C, -0.10 on A).
- **The dV gate** hurt on B and was neutral elsewhere.
- **The owner's target** (hit > 60 %, DD < 5 %) is met by no row of any span. ROBUST on C comes closest on drawdown (6.1 %), but its hit rate is 46 %.
- **Realistic expectation for the prospective test:** a small positive result for ROBUST is possible. A clear pass over three months is unlikely: on C its monthly mean of +0.68 % is about 1.2 standard errors from zero over 13 months.

## Caveats
- **Spans:** A, B and C are not clean; ROBUST is a post-hoc choice; minute direction has decayed (H51).
- **Fills:** the gap between PESS and the original fills is large, so the truth lies between them.
- **First months:**
  - span C's first month uses w = 1;
  - span A's first sigma_ref comes from about 9.5 months of training.
- **Run history:** `describe` ran twice with identical code, the second run to collect example trades. An earlier attempt failed before producing any number.
- **Bootstrap:** criterion 1 adds a Holm-adjusted two-sided bootstrap p < 0.05 to "lower bound > 0", and both must hold. The bootstrap is not circular (minor).

## Files
`selection.jsonl`, `tables_selection.txt`, `choice.json`, `choice_robust.json`, `describe.jsonl`, `tables.txt`, `trades_{A,B,C}.csv.gz`,
`equity_dd.png` / `.plotly.json`, `windows.json` (8 real trades, cut file only), `frozen_P.json` (+ sha256), `score_P.py`, `run.log`, `tests.log`.
