# Hill-climb v2: results (SPEC.md; tactical, exploratory; one lockbox look taken 2026-10-13)

Winner of the climb (best of 10 climbs; also the global dev optimum of the 1,944 configurations): **rich+ctx | HGB | 730 d | 4 h | top 20% | agree(1,2,4 h) | time exit | fixed**.
Baseline (frozen H62): rich | logreg | 730 d | 4 h | top 10% | follow | time | fixed. 2 bps round trip, fixed notional 1x.

| | dev (54 months) | | lockbox (12 months) | |
|---|---|---|---|---|
| | winner | baseline | winner | baseline |
| trades | 3432 | 2118 | 727 | 470 |
| hit | 0.595 | 0.592 | 0.574 | 0.598 |
| mean net bps / trade | 10.33 | 10.41 | 0.79 | 19.30 |
| cumulative return % | 354.4 | 220.5 | 5.8 | 90.7 |
| max drawdown % | 7.5 | 12.8 | 31.1 | 9.2 |
| Calmar | 10.53 | 3.84 | 0.19 | 9.79 |
| months hit > 60% | 25/54 | 23/54 | 4/12 | 6/12 |
| months with DD < 5% | 12/54 | 23/54 | 2/12 | 5/12 |
| weight for DD < 5% / return there | 0.55 / 194.9% | 0.22 / 48.5% | 0.11 / 0.6% | 0.37 / 33.6% |

Verdict on the lockbox: **NOT PASSED** (winner). (1) mean monthly net bps 48 with 95% t-interval [-901, 997]: lower bound not above 0; (2) Calmar 0.19 against the baseline's 9.79: not above;
(3) cumulative return 5.8% against its random-side null p95 58.4% (P(null >= actual) = 0.32): not above. The baseline on the same months: mean monthly 756 bps [-53, 1565] (lower bound just below 0),
return 90.7% against null p95 39.0% (P = 0.000), Calmar 9.79, hit 0.598, 6/12 months with hit > 60%, 5/12 with DD < 5%, DD < 5% needs weight 0.37.

Climb: 10 starts reached 3 distinct optima in 183 distinct evaluated configurations (of 1,944; 0.55 CPU-s). From the baseline: 3.84 -> 4.83 (q 20%) -> 8.25 (HGB) -> 8.51 (agree); the best start:
1.82 -> 5.06 -> 6.39 -> 6.94 -> 8.16 -> 8.88 -> 10.53. Null climbs (40 sign-flipped worlds): final Calmar median 0.65, p90 2.45, p95 2.63, max 4.28; best of all 1,944 per draw: median 0.81, p95 2.64, max 4.28.
So the dev Calmar of 10.5 is far above what selection alone gives (p95 2.6) - and still fell to 0.19 on the lockbox. The dev edge is therefore NOT pure selection noise of this search; it is specific to the dev months
(HGB with 60 features was chosen on months that earlier studies and the frozen features/window already saw, and 20% of the signals on the weakest-confidence side dilute the trade quality: 10 bps/trade on dev, 0.8 on the lockbox).
Caveats: 12 lockbox months; the lockbox is partly seen by earlier studies (the baseline's features/window/horizon were fixed on 2019-2024-06); one seed per model; the drawdown is one path.
Run times: predictions 6 workers ~5 min wall; paths 22 s per horizon; climbs 0.5 CPU-s; 40 null draws 41 s wall; lockbox look <1 min. Total well under the 6 CPU-hour budget (about 0.6 CPU-h).
