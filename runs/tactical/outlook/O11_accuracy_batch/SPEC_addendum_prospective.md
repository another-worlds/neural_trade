# O11 SPEC addendum (lead, 2026-10-15): prospective criteria

Written after the J results and BEFORE any prospective month exists. It extends `SPEC.md` (21bb2c1); the SPEC file itself is unchanged.

The implementer's proposal in `o11_freeze.py` is ratified for the two frozen candidates:
- **Part a (volatility-change skill, candidate - base):** over the 3 scored months, the mean skill gain is >= +0.010 and it is positive in 3 of 3 months.
- **Part b (refit - frozen, L1, H = 15 Spearman):** the mean gain is >= +0.003 and it is positive in 3 of 3 months. The refit arm is refitted at
  scoring on the 365 calendar days before each month (`refit_cal`), as in the J test.
- **Scoring:** once, on the first 3 complete months after 2026-10. Each `score_P_*.py` refuses before that, and refuses a second run.

Part c failed and part d failed on J, so neither is frozen.
