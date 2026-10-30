# Timestral-LIG: learnable indicators and trend geometry inside Timestral-28M

This folder holds the code and the small result files of the experiment. The write-up is in
[`plans/timestral_lig.md`](../../plans/timestral_lig.md).

## What it is

Zero-init adapters that feed 62 learnable indicator and geometry features into a frozen Timestral-28M:

| Site | Where | Starts at |
|---|---|---|
| E | added to the 64 encoder patch tokens | 0 (zero output layer) |
| D | added to the forecast-query token (x 0.1) | 0 |
| H | added to the raw quantile-head channels, before `monotone_quantiles` | 0 |
| G | 10-parameter recalibration of the quantile fan (median shrink, widths per side, step and tail) | identity |
| LoRA | rank 4 on time-attention `wQKV` / `wO`, blocks 8-11 | 0 (B = 0) |

At step 0 the model equals zero-shot Timestral (checked on every Test origin, `T0.json`).

- **Features (`indicators.py`).** 11 gated groups:
  - EMA distance, MACD, MACD pairs, RSI, Bollinger and momentum, with learnable periods initialised from neural_trade's Config.
  - Trend geometry: slope and curvature at 1, 5 and 15 bars.
  - Volatility geometry, path shape, OHLC range and calendar.
- **Losses (`losses.py`).**
  - σ-normalised pinball as the anchor, plus QLIKE.
  - Five geometry terms on the predicted fan: no shrinking widths, a straight term structure, tail thinning, no median bending, and a random-walk median.
- **Governance.**
  - Gate penalty, with groups made identifiable by group-weight-normalised first layers.
  - Loss weights set by gradient alignment with held-out pinball (V1).
  - Early stopping and rung promotion on V2.
  - A pre-registered pass rule, scored once on Test.

## Files

| File | Purpose |
|---|---|
| `data.py` | Train / V1 / V2 / Test origins with 76-bar gaps; batches |
| `indicators.py` | the feature bank; exact causal EMA scan (`seq`) and matmul (`last`) |
| `model.py` | `LIG` wrapper (sites E, D, H, G, LoRA), `CFHS` licence-free control |
| `losses.py` | objective, geometry terms, governed weights, evaluation metrics |
| `train.py` | `cache`, `t1`, `t2`, `select`, `export` commands |
| `checks.py` | automatic checks T0c-T8 (causality, parity, gradient flow, equivariance) |
| `score.py` | Test scoring against the pass rule P1-P7 (uses `../tsfm_eval/bench/metrics2.py`) |
| `results/` | constants, checks, per-run logs, candidate.json (V2 selection), score.json |

## Reproduce

Needs the `timestral` package and the `Timestral-28M` weights (see `../tsfm_eval/README.md`), the benchmark data and
the zero-shot result files of `../tsfm_eval`. On 4 CPU threads:

```
python train.py cache --set repo        # ~10 min; also runs T0
python checks.py repo
python train.py t1 --set repo --rungs A1,A2 --seeds 0,1,2,3,4
python train.py t1 --set repo --rungs A2-plc,A2-g0,A2-close,C-FHS --seeds 0,1,2
python train.py t2 --set repo --rung A3 --seed 0      # ~10 min per run
python train.py t2 --set repo --rung A4 --seed 0
python train.py select --set repo       # writes candidate.json before any Test scoring
python train.py export --set repo
# same for --set post, then:
python score.py
```

Work files go to `work/` (git-ignored). Adapter checkpoints are derivatives of Timestral under its Modified MIT
licence and are not committed.
