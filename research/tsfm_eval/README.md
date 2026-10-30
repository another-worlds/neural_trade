# tsfm_eval: zero-shot foundation-model benchmark for neural_trade

This directory reproduces the numbers in `plans/model_evaluation_timestral.md`.

## Layout

| Path | Content |
|---|---|
| `bench/common.py` | Data sets and forecast origins. `BENCH_DATA=repo` (default, repo CSV, last 20% of rows) or `BENCH_DATA=post` (Binance 2026-09-20..10-29; origins from 2026-10-16). Stride 3 min; horizons 1/5/15 min. |
| `bench/fetch_post.py` | Downloads the post-release set from the Binance public market-data mirror. |
| `bench/run_timestral.py` | Timestral forecasts. Args: `size ctx framing [--no-sym]`. Framing is `price`, `logret` or `price_cov`. |
| `bench/run_official.py` | Chronos-2 / Chronos-Bolt through the official `chronos-forecasting` package. |
| `bench/run_samay.py` | The same models plus TimesFM-2.5 through AdityaLab Samay's wrappers. Run it in Samay's locked environment. |
| `bench/run_vol.py` | Volatility role: forecasts the mean 1-min high-low range over the next h min. |
| `bench/metrics2.py` | Scores all forecasts against causal baselines, with paired block-bootstrap CIs against EWMA-Student-t. |
| `bench/metrics_vol.py` | Scores the volatility forecasts against EWMA, rolling means and HAR. |
| `bench/incremental2.py` | Feature role: walk-forward logistic regression, ΔAUC. |
| `bench/matched.py` | Paired ablations: each variant against its base config on identical origins (context, size, covariates, framing, symmetric averaging). |
| `bench/latency.py` | Single-series CPU latency. Runs only when `LATENCY_GO=1`. |
| `bench/make_tables.py` | Builds the report tables from the result JSON. |
| `neural_trade_h2h/` | Retrains neural_trade as-is (imports the repo `model.py` unmodified; redirects IO paths and seed only) and predicts on the same origins. `nt_latency.py` times its single-call inference. |
| `results/` | Committed result summaries (JSON/markdown). Forecast arrays (`*.npz`) are not committed; the scripts regenerate them. |

Working files go to `$TSFM_EVAL_WORK`, default `research/tsfm_eval/work/`, which is git-ignored.

## Environments

```bash
# 1) main: Timestral + official Chronos (Python 3.13 tested)
python3 -m venv .venv-tsfm && . .venv-tsfm/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/mistralai/timestral.git chronos-forecasting pandas scipy scikit-learn

# 2) Samay: only its exact lock file works (Python 3.11, torch 2.5.1, transformers 4.47.1)
git clone https://github.com/AdityaLab/Samay && cd Samay && git checkout d23c90f && uv sync --frozen --no-dev
# Do NOT run `pip install samay`: that name on PyPI is an unrelated package.

# 3) neural_trade head-to-head
python3 -m venv .venv-nt && .venv-nt/bin/pip install tensorflow-cpu==2.20.0 keras==3.11.3 pandas scikit-learn scipy joblib matplotlib tqdm
```

## Run order

```bash
cd research/tsfm_eval/bench
export TSFM_EVAL_WORK="$(cd .. && pwd)/work"            # default location; any path works
python fetch_post.py                                   # post-release data
python run_timestral.py 28M 2048 price                 # repeat per size / context / framing
python run_official.py chronos2 512
<samay-venv>/bin/python run_samay.py timesfm25 512
python run_vol.py timestral 28M 2048
BENCH_DATA=post python run_timestral.py 28M 2048 price # same commands with BENCH_DATA=post
BENCH_SUBSTRIDE=6 python run_timestral.py 209M 8192 price   # long context on every 6th origin
(cd ../neural_trade_h2h && <nt-venv>/bin/python make_csvs.py && \
   for d in repo post; do for s in 0 1 2; do <nt-venv>/bin/python run_nt.py --data $d --seed $s; done; done)
cp $TSFM_EVAL_WORK/nt_h2h/out/repo/*.npz $TSFM_EVAL_WORK/results/
cp $TSFM_EVAL_WORK/nt_h2h/out/post/*.npz $TSFM_EVAL_WORK/results_post/
python metrics2.py && BENCH_DATA=post python metrics2.py
python metrics_vol.py && BENCH_DATA=post python metrics_vol.py
python incremental2.py timestral-209M_ctx2048_price official-chronos2_ctx512_price
BENCH_DATA=post python incremental2.py timestral-28M_ctx512_price timestral-28M_ctx2048_price \
   timestral-209M_ctx2048_price official-chronos2_ctx512_price neural_trade-asis-seed0_ctx60_price
python matched.py && BENCH_DATA=post python matched.py
LATENCY_GO=1 python latency.py timestral && LATENCY_GO=1 python latency.py chronos2
(cd ../neural_trade_h2h && for t in 4 1; do <nt-venv>/bin/python nt_latency.py --threads $t; done) >> $TSFM_EVAL_WORK/nt_latency.jsonl
python make_tables.py > ../results/tables.md
```

On a 4-CPU machine the full grid takes about 4–5 hours. Most of that is Timestral-209M at long context and the covariate runs.

## Notes

- All targets are log returns in bps from the last observed close. No model sees data after the forecast origin.
- Baselines (EWMA λ and ν, HAR, logistic regression) are fitted only on data before the first scored origin.
- Timestral weights are under a "Modified MIT" licence with a $20M monthly-revenue limit. Read the LICENSE on Hugging Face before any use beyond research.
