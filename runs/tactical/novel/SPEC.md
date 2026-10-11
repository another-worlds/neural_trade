# Novel geometry hypotheses: SPEC (fixed before any run, 2026-10-11 ~02:15)

Owner, 2026-10-11: "Найди гипотезы которые не были тестированы в мире и не имеют отрицательных отзывов по геометрии,
реализуй и запусти. Работай автономно до 13 часов дня".

## Literature screen (2026-10-11, web search; "not found" means not found by this search, not proof of absence)

Excluded (already tested): path signatures (sub-daily direction: negative, arXiv 2608.26158); persistent homology / TDA
(tested, mixed); price-path convexity (tested on stock daily returns, positive: kept only as a labelled transfer control N7).
Candidates with no academic test on short-horizon direction found (some exist only as TradingView scripts, untested):
- **N1 Frenet geometry of the indicator trajectory:** the last 20 bars as a 3-D curve in (close-EMA20, RSI14, MACD hist) space,
  each axis z-scored on the window: mean and end curvature, end torsion, total turning angle, tortuosity (arc / chord).
- **N2 Levy areas between price and its own indicators:** signed area of the 2-D paths (cum return, cum signed volume),
  (cum return, RSI), (cum return, close-EMA20), (cum return, high-low range) over the last 15 / 30 / 60 bars, each divided by
  the product of the two paths' total variations (scale-free, in [-1, 1]).
- **N3 Phase portrait with winding:** velocity and acceleration of EMA5-smoothed close (z-scored on the window); the angle
  theta of (v, a), its angular velocity at the end, the total winding (radians turned) over the last 30 bars, the radius.
- **N4 Convex hull of the window:** upper and lower hulls of (t, high) and (t, low): the last close's position between the hull
  lines at the last bar (0 = lower, 1 = upper), slopes of the last upper and lower hull edges / sd (geometric resistance and
  support lines), the distance to each line / sd, hull area / (range x length), the number of hull vertices.
- **N5 Wasserstein shift inside the window:** W1 between 1-bar returns of the first and second halves and between the last 15
  and the previous 45 bars, / sd; the signed median shift / sd.
- **N6 Fisher-Rao path:** (mu, sigma) of 1-bar returns over 6 consecutive 10-bar sub-windows as points on the Poincare
  half-plane (the Gaussian family's Fisher-Rao geometry): hyperbolic path length, net hyperbolic displacement, and its mu
  component's sign x magnitude.
- **N7 (control, already tested on stocks):** convexity = the quadratic coefficient of a fit to the last 15 / 30 / 60 log
  closes, / sd.

## Protocol

- Data: the 24 lab slices (runs/tactical/lab/cache), the network's exact blocks; features at the window's last bar, causal
  (tests). Readouts: logistic regression (C 0.1) and HistGradientBoosting, each family alone and as tb7 + family; all
  novel families together; per horizon, mean of 3 (as lab.py).
- **Primary verdict:** the paired per-slice mean3 AUC difference vs tb7 logistic; a family "adds" only if its 95% t-CI over
  the 24 slices is above 0 after a Holm correction over all (family x readout) tests.
- Secondary: (a) the volatility read: ridge on log|r| with tb7 + family vs HAR-RV + time of day (runs/tactical/vol), Spearman
  paired diffs; (b) the honest top-10% tail vs the random null; (c) 1-year training spans (as runs/tactical/rebuild/longtrain.py)
  for any family whose boosting readout gains more than +0.003 at 11.5 days.
- A family that "adds" is checked ONCE on the 6 FINAL minute slices (cache.py --final logic). Nothing is chosen there. The
  hourly held-out period (2024-07 onward) is spent and not used.
