# Owner Q&A: the window-free plan (2026-09-29)

The owner's decision on the plan that D-032 required (NT-053):
[research/2026-09-29-window-free-plan/](../research/2026-09-29-window-free-plan/README.md), revised
after its adversarial review ([REVIEW.md](../research/2026-09-29-window-free-plan/REVIEW.md)). Read
this before asking the owner anything about removing the window, indicator forms, period bounds or the
plan's GPU budgets.

## Round 1 (all four answers were the recommended options)

- **The plan:** APPROVED AS WRITTEN. The A/Bs judge probabilistic skill (per-fold retention of the CRPS
  edge over constant variance, and paired coverage); net Sharpe is reported beside the verdict but not
  judged.
- **The new families' forms (NT-047):** WINDOW MODE NOW, EXPONENTIAL / LEAKY FORMS LATER. NT-047 is
  built now in window mode against the Indicators registry interface; after A/B-1 the series forms are
  exponential and leaky (decayed soft max/min, leaky OBV, VWAP as a ratio of EWMAs), drawn next to the
  textbook values.
- **The period bound under "unlimited":** HOLD AT THE DATA BOUND. A learned period that would need more
  history or pass length than available is projected at the data-derived bound (defaults: a 2,048-bar
  burn-in on the bundled file, about 360 bars; a pass budget of 4 training blocks, about 5,300 bars on
  the long history), counted and reported. No configured ceiling.
- **GPU budgets:** THE CEILINGS ARE APPROVED: A/B-1 about 6 GPU-hours, A/B-1b about 6, A/B-2 about 5.6,
  the probe about 1.5 (worst cases, estimates), each re-checked from its dev runs' measured times
  before any judged run; a study whose recomputed worst case exceeds its ceiling goes back to the owner.
