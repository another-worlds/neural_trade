# Vision

Stable document. It changes only when the owner changes direction. Plans live in
[ROADMAP.md](ROADMAP.md) and [BACKLOG.md](BACKLOG.md); the current state lives in [STATUS.md](STATUS.md).

## Purpose

`neural_trade` asks one question and must answer it with evidence:

> Can one neural network that reads the last 60 one-minute BTC/USDT closes predict the next 10,
> 15 and 20 minutes (price change, direction, uncertainty) well enough that a strategy built on it
> beats realistic baselines **after trading costs**, on data it never saw?

The architecture under test: a GRU + attention network with learnable technical indicators (EMA,
MACD, RSI, Bollinger periods trained by gradient descent), nine output heads (delta, P(up) and
variance for each horizon), and six physics-inspired loss terms.

## End goal

A model and a strategy that, on out-of-sample walk-forward folds with next-open fills and
26 bps round-trip costs:

1. beat the pre-registered baselines (zero / mean delta, class prior, logistic regression on
   trailing returns, constant variance, buy-and-hold, random entries at the same frequency) by
   more than the noise, with intervals that account for overlapping targets;
2. are served through the `Predictor` API with calibrated probabilities and intervals;
3. come with the evidence to trust them: reports, notebooks executed on real runs, and every
   number linked to a run directory.

A clear, evidenced **negative** answer (the edge does not survive costs, or a term adds nothing)
is also a valid outcome. It is recorded, and the project moves on. It is not tuned away.

*Owner: this end goal is inferred from the remediation work and your decisions. Edit this
section if it is not what you want. Every later plan derives from it.*

## Principles

- **Evidence, not claims.** A result counts only if it links to a run directory, a test or an
  executed notebook. "It works" means a real run was executed and its outputs were inspected.
- **Real runs over toy runs.** Tests on synthetic data are necessary but not sufficient.
  Figures and notebooks are verified on the shipped defaults (notebook 01 trains the full
  `EPOCHS` on the GPU).
- **Noise-aware statistics.** Consecutive 1-minute samples share most of their target bars.
  Every interval and chance band uses effective samples (about `N // horizon bars`) or a block
  bootstrap. A number without its noise level is not reported as a finding.
- **Pre-registered criteria.** An experiment states its hypothesis, metrics and thresholds
  before it runs (as `configs/ablation_criteria.yaml` did). Test data is touched once, for the
  verdict, never for choices.
- **Honest trading numbers.** Next-open fills, fees, spread and slippage, stops on high/low,
  baselines in every report, and a random null at the same trade frequency.
- **Fast training.** Fast GPU training and solid optimisation are first-class requirements;
  inference speed is negligible (owner, D-018).
- **One visual system.** Every figure uses `visualization/theme.py`. Horizons keep their
  colours (h0 blue, h1 orange, h2 green); dotted lines mean training.

## Fixed decisions (owner)

These are settled. Do not reopen them without new evidence and the owner's agreement; see
[DECISIONS.md](DECISIONS.md).

- TensorFlow 2.10 / Keras 2 (the last release with native Windows GPU support). No Keras 3 migration.
- All nine component registries stay, and stay wired into the training path.
- The six physics terms stay. Fix their mathematics and iterate until they demonstrably provide
  value under the pre-registered ablation criteria, or until the evidence says they cannot.
- Git history is never rewritten. `master` changes only when the owner merges.

## Out of scope (until the owner says otherwise)

Keras 3 / newer TensorFlow; new data sources (exchange APIs, order book); multi-asset
evaluation; MLflow; architectures other than `gru_attention` (the Models registry is ready
for them, but none is planned).
