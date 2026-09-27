# Owner Q&A: the indicator catalogue (2026-09-28)

The separate indicator Q&A promised in D-027, held on 2026-09-28 right after the vision Q&A
([2026-09-28-vision-mvp.md](2026-09-28-vision-mvp.md)). The owner's own words are quoted where they
went beyond the offered options. Read this before asking the owner anything about indicators.

## Context given to the owner

Today: four close-only families, all EWMA-based (MA 5/10/30; MACD 12/26/9, 5/35/5, 8/17/9; RSI
9/14/21; Bollinger 10/20/25): 18 learnable periods -> 31 channels, each period shifted per window by
a small `meta_adjust` network; combinations are mixed implicitly by the Bi-GRU and attention layers.

## Round A

- Families in the MVP catalogue: ALL FOUR GROUPS: today's four (MA/EMA, MACD, RSI, Bollinger);
  range / volatility (ATR, Stochastic, Williams %R, Keltner); volume (OBV, VWAP, MFI); trend strength
  / channels (ADX/DMI, CCI, Donchian). The model input therefore becomes OHLCV (not close-only).
- Learnability: owner: "All should be learnable" -> every indicator has parameters learnable by
  gradient descent; smooth (differentiable) versions are written where needed (e.g. soft max/min).
- Combinations: owner: "the combination are what the network does automatically, it's the core of
  it's mechanic. I don't think it needs any extra mechanism" -> no gate or selection layer; the
  network mixes the indicators itself.
- Per-window adaptive periods: KEEP, REPORT AS A RANGE (global value plus per-window range, both
  drawn; a switch turns adaptation off for the frozen twin and comparisons).

## Round B

- Registry entry: A FAMILY (e.g. `rsi` registered once; it declares its inputs (close / high / low /
  volume), learnable parameters with textbook defaults and bounds, output channels, and how it is
  drawn (on price or in its own panel); the config lists its instances, in wall-clock time).
- Warm-up / the window: owner: "I want a massive optimization of this process: currently the
  'window' is a huge limiting factor in the training. I think we can get rid off it completely.
  Research this." -> research launched on 2026-09-28 (removing the fixed input window); its result
  is recorded here and in DECISIONS when the owner decides.
- Defaults: ALL FAMILIES ON in the default configuration (the network decides what to use).
- Importance read-out: YES: grouped permutation importance after training (drop one family or
  instance at a time on validation data, loss of skill with noise bands), shown in the
  discovered-indicators notebook and the run report. A read-out only, not a model mechanism.

## Round C

- Instances per family: 3 PER FAMILY (today's default; spread-out starting periods, e.g. RSI 9/14/21),
  for every family; more or fewer is a config change.
- Delivery of the discovered set outside the notebooks: owner: "html report with extensive rich
  interactive visualizations" -> a self-contained HTML report per run (plotly, D-014-rich): the
  discovered indicators drawn on price against the textbook defaults, learned parameters in wall-clock
  time with their per-window ranges, the importance read-out with noise bands, how the parameters moved
  during training. Not a YAML or Pine export.

## Round D (the window research's questions, answered 2026-09-28)

The research record: [docs/research/2026-09-28-window-free/](../research/2026-09-28-window-free/README.md).

- Path: owner: "commit to this in another research and write it down in a plan." -> removing the
  fixed input window is committed to; the concrete path (stages, order, gates) and the corrected A/B
  specifications come from a second research round that writes a plan (NT-053), which the owner
  approves before its first implementation item is picked.
- Longest learnable period: owner: "unlimited" -> no configured period ceiling.
- Replacement for the per-window adaptive periods: owner: "research" -> part of NT-053.
- Purge rule for indicators with unbounded memory: owner: "research yourself" -> the lead researches
  and decides it (part of NT-053), recorded as a new DECISIONS entry with a test.
