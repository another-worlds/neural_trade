# Owner Q&A 2026-09-29: training on a long history

Context: the NT-005 strategy study (runs/experiments/strategy_study_v1/REPORT.md) found no strategy that beats
costs. The lead listed the model's 7-day training block (about 500 effective samples at the 20-bar horizon)
as the first lever to test, with a learning-curve study as the proposal.

## Owner statements (verbatim)

1. "I think there are key factors thar contribute to the por performance of the strategies as a whole:
   1) extremely small training set for such a complex model 2) noisy data, btcusd is notorious for noise and
   was selected for exomplexity 3) asbsense of the PnL in the models' targets. While point 3 and 2 we should
   address later, we can eval how much in days we train now."
2. "launch 1 360 days training sample. Ensure GPU is running at full effectiveness. the full dataset is in
   Bitcoin_BTCUSDT.csv. Before training I want to see thr comparison between the old project structure and
   new project structure time estmatesin training and reasonign why"
3. "also ensure that we use as high batch size as possible"
4. (Russian) "после бенчмарка сам разберись как будет правильно поставить 360д трейн и запусти его сам
   автономно желательно через ноутбук, чтобы я могу трекаьть прогресс" (after the benchmark, work out
   yourself how to set up the 360-day training correctly and launch it yourself, autonomously, preferably
   through a notebook, so that I can track the progress).

## What follows (recorded in DECISIONS D-040)

- One training run on a 360-day training block of Bitcoin_BTCUSDT.csv (2017-01 to 2025-09, 1-minute bars),
  set up by the lead: layout, batch size, shuffling and learning rate are the lead's call.
- The old-against-new training-time comparison is shown before the run starts (the benchmark report).
- The run is launched and tracked through a notebook.
- Points 2 (noise) and 3 (P&L-aware targets) come later.

## Follow-up (2026-09-29)

5. (Russian) "убери коммиссию полностью" (remove the commission completely). The lead's reading: re-score the
   trained 360-day run with fee, half-spread and slippage all 0, beside the default costs, without changing the
   default costs (VISION "Honest trading numbers" requires them); whether the default changes is asked back.
   Result: runs/scenarios/long_360d/rescore/zero_cost_v1-20260929T123047Z/ (configs/strategy_studies/zero_cost_v1.yaml).
