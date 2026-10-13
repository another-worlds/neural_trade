# O10 report: geometry and volatility as the EXIT of a trade

SPEC `1b4b3c0` (+ addendum after describe, before any prospective number). Commits:
- code af6f167;
- choice b42d13b (alone);
- selection 75f1251;
- arms d4518f6;
- describe 2b42dfa;
- freeze 7d28920.

Tests: 32 passed. CPU only. Written by the outlook lead from the implementer's text (its Write tool refused the file). Selection rows checked against `tables_selection.txt`.

## Answer
**No geometry or volatility exit beats the plain price-level exit on this entry.**
- **E1, E2 and E3 lose.** On the selection months (PESS, 2 bps), every vol-trailing, EMA-cross, 15-minute-slope and vol-contraction exit is
  worse than ROBUST's own exit (stop 1 sigma, target 2 sigma) at the same holding limit, by 0.5-2.2 %/month. They cut the winners
  (average win 32-62 bps against 58-95) and do not cut the losses enough.
- **E4, the learned geometry + volatility exit, changes the shape of the trades, not the profit.**
  - Trades last 8-17 minutes and drawdown on B and C is 4-6 %.
  - It earns less than E0 on all three seen spans and no more than its own shuffle.
- **The choice is E0_T240 (pre-registered rule).** It is ROBUST's exit with a 240-minute time stop instead of 60: t 3.32, +2.56 %/month on selection.
  - On the seen spans it does not differ from ROBUST beyond noise.
  - It beats its shuffle on all three spans.
- **The entry carries the strategy:** the regression gate on the breakout (O9). The exit adds little either way.

## Selection (2021-01..2024-06, PESS, 2 bps; % of capital per month)
| config | trades/mo | m2 | m5 | t | hit % | avg win / loss bps | hold min | max DD % |
|---|---|---|---|---|---|---|---|---|
| **E0_T240 (choice)** | 50.5 | +2.56 | +1.05 | 3.32 | 42.1 | 94.9 / -60.1 | 102 | 21.9 |
| E0_T60 (= O9 ROBUST) | 54.5 | +1.82 | +0.19 | 3.20 | 46.0 | 57.9 / -43.2 | 46 | 16.0 |
| E4_T60 (learned) | 57.2 | +2.12 | +0.40 | 2.22 | 47.1 | 33.2 / -22.5 | 11 | 12.6 |
| E1 c=2, T60 (vol trail) | 55.1 | +1.34 | -0.32 | 2.06 | 41.2 | 53.3 / -33.2 | 34 | 17.3 |
| E2b T60 (15-min EMA20 slope) | 55.8 | +0.91 | -0.76 | 1.40 | 43.6 | 41.3 / -29.1 | 26 | 23.3 |
| E3 T60 (vol contraction) | 56.6 | +0.69 | -1.01 | 1.42 | 46.2 | 33.2 / -26.2 | 18 | 11.6 |
| E2a T60 (1-min EMA20 cross) | 55.9 | +0.74 | -0.94 | 1.51 | 36.8 | 45.7 / -24.5 | 19 | 14.2 |
| E1 c=1, T60 | 56.7 | +0.36 | -1.34 | 0.92 | 41.1 | 32.3 / -21.4 | 11 | 13.9 |

Full table (14 configurations, original fills too): `tables_selection.txt`.

## Seen spans (descriptive; PESS, 2 bps; mean monthly net %)
| span | CHOICE (E0_T240) | E0 (ROBUST) | GV (E4_T60) | CHOICE - E0 | GV - E0 | CHOICE - SHUF |
|---|---|---|---|---|---|---|
| B 2024-07..2025-07 | +3.01 [+0.33, +5.69], DD 14.8 % | +2.12, DD 9.2 % | +0.86, DD 5.0 % | +0.88 [-1.48, +3.24] | -1.27 [-2.75, +0.22] | +1.85 [+0.37, +3.33] |
| C 2025-10..2026-10-12 | +1.01 [-0.31, +2.33], DD 8.0 % | +0.68, DD 6.1 % | +0.03, DD 5.3 % | +0.33 [-0.71, +1.37] | -0.64 [-1.90, +0.62] | +2.31 [+1.06, +3.56] |
| A 2018-07..2020-12 | -0.65, DD 44.6 % | +0.21, DD 43.0 % | -1.95 | -0.86 [-2.45, +0.73] | -2.16 [-4.30, -0.02] | +2.32 [+0.17, +4.47] |

- **Owner's target** (hit > 60 %, DD < 5 %): met by no arm on any span. GV comes closest on drawdown (5.0-5.3 % on B and C), but its hit rate is 45-48 % and its profit is about 0.
- **Buy-and-hold** for comparison: B +63.9 %, C -33.2 %, A +152 %.

## Prospective (frozen; scored once with O9 on 2026-11..2027-01)
- **Frozen artefact:** `frozen_P.json`, sha256 `3f87ef099e8782cae547920a706bbc569ae225caf5ff6b4ffa774c8007509e3c`. It contains O9's frozen models (sha recorded), the 15-minute sigma, and the E4 models.
- **Scoring:** `score_P.py` refuses today (0 complete months). Verdicts as defined in the SPEC and its addendum:
  - "exits add (geometry/volatility)" = GV - E0, lower bound > 0;
  - "longer time stop adds" = CHOICE - E0;
  - "works" = CHOICE > 0 and > SHUF.
- **Expectation:** an "exits add" pass is unlikely. GV was below E0 on every seen span.

## Caveats and deviations
- **Spans:** A, B and C are not clean (O8/O9 saw them). The GV arm was added after the selection table, before describe.
- **SHUF:**
  - for price-level exits, SHUF permutes the sigma that sets the stop and target within each month;
  - for the signal exits, it permutes the signal series;
  - for E4, it draws probabilities from the true arm's monthly multiset.
- **Rule details:**
  - E1 trails from closed bars and never goes beyond the 1-sigma stop.
  - E2a uses a causal 1-minute EMA20 (span 20).
  - E2b, E3 and E4 act only at decision bars after the entry bar, exiting at that bar's open.
- **E4 training:**
  - it trains on gross checkpoint returns, from past checkpoints only;
  - on span C it is a single fit from 2025-07-21, so it ages over 13 months;
  - the frozen E4 is in-sample on the freeze year (mild optimism).
- **Selection:** no eligibility floor was needed (every configuration had 50-58 trades a month).
- **Original fills:** they give higher numbers for every arm; the truth lies between the two fill models.

## Files
`selection.jsonl`, `tables_selection.txt`, `choice.json`, `describe.jsonl`, `tables.txt`, `trades_{A,B,C}.csv.gz`, `equity_dd.png`, `hold_R.png`
(+ plotly JSON), `windows.json` (10 real trades, cut-file bars only), `frozen_P.json` (+ sha256), `score_P.py`, `run.log`, `tests.log`.
