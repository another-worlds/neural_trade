# Adversarial review of the plan draft (2026-09-29)

Reviewed the draft of README.md (commit e7ce6c6) against its evidence (A/, B/, C/), the first round, NT-053's
acceptance and the project rules. CPU only; scripts and outputs in [review/](review/) (`r1_ab1_one_change`,
`r2_kernel_long`, `r3_common_target`, `r3b_self_target`). Transcribed by the lead from the reviewer's report.

**Verdict: not ready for owner approval.** 7 must-fix, 15 should-fix, 9 nits. The kernel, the D6b assembly
and D-034's leakage argument hold up.

## Must-fix

- **M1. The reach rule of the probe (spec 0) and A/B-2 is broken by seed noise (measured).** R = mean of arm
  A's min J~ + 5% of span. On the 42 v1 dev-fold runs the seed-to-seed SD of min J~ is 17% of span; against a
  target built from the other seeds of the same condition, 20/42 runs (48%) never reach it (18/42 even with
  themselves in the target). An equivalent B1024 is classified update-bound (rho = infinity) about 66% of the
  time, and half of equivalent A/B-2 pairs get t_B = infinity. C's "misclassification under 0.1%" modelled only
  the reach-epoch noise against each run's own minimum. Fix: a quality level set from the seed noise or the
  margin (R = T_A exp(delta), or 50-80% of A's improvement span, or time to the served checkpoint with quality
  judged by the guard-rail), then re-simulate with target misses included.
- **M2. The history bound is a de facto ceiling on the bundled file; nothing bounds the pass on the long
  history.** With the mask fixed at run start and the projection to the available history, a 1,024-bar mask
  caps the longest single-EWMA base period at 180 bars (2,048: 360); this governs CI, the golden run and the
  notebooks, and the plan does not say so. On the long history the pass grows to 68.5k bars at period 10,080
  with no stated budget. A (refuse or mask the fold) and B (project the logits) contradict each other. Fix:
  choose the data-start mask and state the bound it implies; define a pass budget (fail loudly or project, and
  call projection a ceiling); put the bounds to the owner.
- **M3. No stage removes the 60-bar ceiling.** A/B-1's arm B keeps it; the "later scenario" is in no stage or
  item, while the Answer claims A2 removes it. D-032's "unlimited" and acceptance (4) are delivered by
  nothing scheduled. Fix: a stage and item that remove the clip in series mode, with its gate, after A/B-1.
- **M4. A/B-1 is not a one-change comparison (measured, r1).** Series mode varies alpha per bar inside the
  former window; window mode applies the anchor's alpha to all 60 bars. RMS difference relative to today's
  feature at fold -1 val anchors: adaptation granularity alone 1.9-6.2%, memory alone 0.0-20.4%; for 5 of 6
  instances the adaptation change exceeds the memory change. Fix: call arm B the series engine, or judge a
  memory-only pair with adaptation off in both arms, or a third arm with a pre-stated deciding pair.
- **M5. The margin is not a fraction of A's edge on the judged folds.** delta = E_dev / 3 on the dev folds,
  but the edge varies 2.5-2.9x between folds: with the dev edge .040, delta = .0133 is 68-95% of fold -1's
  edge. Fix: judge retention per pair, e.g. the upper bound of mean(d_f - E_A,f / 3) < 0 (about +0.0033 SD in
  quadrature, an estimate).
- **M6. The GPU budget counts only the judged runs.** OPERATING_MODEL counts every run of the SPEC: dev runs,
  calibration and the pre-committed INCONCLUSIVE re-run. Worst case at F = 21: 130 runs, 5.6-13.7 GPU-hours
  (estimate); A/B-2 with A' 2.3-3.6. Fix: budget the SPEC total; ask the owner for A/B-1's budget.
- **M7. The placement breaks the pick order (acceptance 6).** Stages 3-6 in MVP-6 depend on NT-041 (MVP-4)
  and NT-032 (MVP-2); "between NT-046 and NT-048" delays NT-048 needlessly; "a stage starts when the one
  before passed" contradicts stage 1 before stage 0; stage 4 shares files with NT-047 and NT-041. Fix: an
  explicit dependency graph; split stage 4 into 4a (switch, warm-up, data-start mask, Predictor, reporting;
  needs NT-046 and stage 3) and 4b (gap policy, fold roles; with NT-041); A/B-1 as a research track after
  MVP-4; order stage 4 against NT-047; keep NT-048 free of the stages.

## Should-fix

1. An epoch cap can decide A/B-1 (EPOCHS 60 at ~40 steps per epoch; "more than 25% capped -> at most
   INCONCLUSIVE"). Set EPOCHS from the dev runs (e.g. 2x their largest served epoch) before judged runs.
2. The stage-4 D-018 gate is a single run within +5% (the first review flagged this); G-A2's timing has no
   threshold. Use at least 3 interleaved runs per side, epochs 1 and later, the GPU-free check, a named
   runner (implementers may not run GPU jobs) and a numeric threshold.
3. TF32: only the flag was measured; "runs in TF32 on this GPU" is an estimate. The meta-shift Dense (a 2 to
   18 MatMul) feeds alpha but sits outside the kernel census and precision gate; B's kernels use einsum and
   B's off switch routes to an einsum kernel. Census the whole indicator layer; write the shift Dense as
   elementwise multiply-add; the off switch is V1 with a zero shift. Turning TF32 off is D-018 (owner if slower).
4. "About 1.7x a fixed period" was measured on B's factored kernel, not V1 (A measured V1 costs the same for
   constant and per-bar alpha). Correct it and reject the factored form explicitly.
5. The Adam per-step bound: |d logit| can reach about 2.5-3.2 lr per step, not lr. Project each step to the
   allocated pass length as well as to the history bound.
6. The mask is M_run (plan), M + L - 1 (Predictor) or M + W (C); the reset threshold is 60 minutes (A) or 30
   (C). Use M_run + L - 1 everywhere and one threshold.
7. The probe measures the window model on the 30-day file (~119 steps per epoch); stage 7 uses 7-day blocks
   (~40) and the A/B-1 winner. Run it just before stage 7 on that setting.
8. The INCONCLUSIVE re-run with twice the folds does not say whether it pools with the first set (optional
   stopping). Pre-register group-sequential alpha spending or a standalone re-run.
9. Sizing uses test-fold numbers (sigma_plan pools folds -2 and -1; the planning edge .0195 is fold -1).
   Use fold -2 only (direct pair SD .0075) or record that sizing is exempt from D-020.
10. Cited evidence missing from the repo copy (A/q1_results.json, q1_summary.txt, det_strings_*.txt,
    C/q2_v1_per_run.csv). (Copied by the lead.)
11. The warm-up rule covers EWMAs only; NT-047's families (ADX's triple cascade, soft extremes, leaky OBV,
    VWAP) need their own M. Make M(eps) a per-family registry contract with an offset-invariance test.
12. Serving and masking costs unstated: after a gap over 60 minutes the Predictor refuses for M_run + L - 1
    minutes (about 40 days at period 10,080); resets mask up to 81% of some long-file blocks at M = 8,198.
13. Retracing on growth: +1,024 bars per retrace at about 13 s each is about 55 retraces (12 minutes) from 1,365
    to 57k bars. Grow geometrically or use a dynamic pass length.
14. Missing owner questions: the period bounds (M2), A/B-1's GPU budget (M6), that the verdict is on CRPS and
    not the net-Sharpe yardstick (every arm-A run loses money; pair SD about 20), TF32 off if slower, A/B-2's
    speed factor.
15. A/B-1's family set is unspecified; if NT-047 lands first in window mode, arm B needs other forms for the
    box-window families. Pin A/B-1 to today's four families.

## Nits

Chat preambles in B/ and C/ FINDINGS (removed by the lead); the W consumers omit HD and the new context-span
key; the tolerance normalised by max|state| is lax for spiky channels (var p2: 2.8e-5 of RMS, 1e-6 of max);
"judgement folds after every choice fold" should read as read-range disjointness (the defaults were chosen on
the bundled file, which lies after the long file's end); M recomputed per epoch (A) or per step (B);
`check_numerics` at data load, not per step (D-018); stage 1 only in `scripts/bench/`; joint power quoted
only at 20 pairs (at F = 10 P2's false-fail exceeds .146); A/B-2 lacks the contention rules, and lambdas
calibrated on arm A slightly favour A.

## Checked and found sound

- The V1 kernel beyond 43k bars (measured, r2): at 70,000 and 262,144 bars of the long file, per-bar alpha,
  periods 2 to 1e6 and logit -40, C = 16: worst error 1.1e-6 of max|state| (2.8e-5 of RMS).
- D-034's leakage argument (structure and cost arithmetic).
- D6b by argument (not run on a GPU): with distinct anchors its backward is gather plus reduce_sum only.
- C's pair counts follow from its noncentral-t formula.
- Most first-round must-fixes are addressed, except the one-change arm (M4), the margin (M5) and the single-run
  speed gate (should-fix 2).

| NT-053 acceptance | Status | Blocked by |
|---|---|---|
| (1) | partly met | M1, M4, M5, S1 |
| (2) | partly met | S4 |
| (3) | met | - |
| (4) | partly met | M2, M3, S5, S6 |
| (5) | mostly met | S3, S10 |
| (6) | not met | M7, S6 |
| (7) | pending | - |
