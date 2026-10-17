# T14 stage 1: install and smoke test (2026-10-17)

Envs (both new, isolated; the `nt` env untouched): `D:/nt/nt_mm_env` (py3.11, torch 2.11.0+cu126, torchvision 0.26.0+cu126) and `D:/nt/nt_mm_legacy` (py3.10, torch 2.5.1+cu124).
Caches: `D:/nt/nt_mm_cache` (`HF_HOME=D:/nt/nt_mm_cache/hf`, `TORCH_HOME=.../torch`, `PIP_CACHE_DIR=.../pip`). Smoke data: tail of the cut file (last bar 2025-07-20 23:59, file-time/UTC part from 2024-09-13), via `common_data.py`.
Python paths: M1, M2, M4, M5, M6 -> `D:/nt/nt_mm_env/python.exe`; M3 Kronos -> `D:/nt/nt_mm_legacy/python.exe` (+ clone `D:/nt/nt_mm_cache/src_kronos`, commit 67b630e, on sys.path).

| candidate | env | installed | smoke | seconds (warm cache; cold) | peak GPU GB | versions | weights |
|---|---|---|---|---|---|---|---|
| M1 Chronos-2 (amazon/chronos-2, 120M) | mm_env | Y | Y: `predict_df`, 512 five-min log RV bars, horizon 12, q 0.1/0.5/0.9 | 1.5 (cold ~20) | 0.49 | chronos-forecasting 2.3.2, transformers 5.20.1 | model.safetensors 477,937,... B, sha256 ddcda3c7 |
| M2 TTM r2 (granite-timeseries-ttm-r2, 805k params) | mm_env | Y | Y: zero-shot, ctx 512 log RV, 96-step branch (12, 24, 48 branches also load) | 0.7 (cold 5.9) | 0.013 | granite-tsfm 0.3.10 | two snapshots (main + a branch): a706726a, b3cbaf0a |
| M3 Kronos-small + Tokenizer-base | mm_legacy | Y | Y: 400 one-min OHLCV bars, pred_len 60, sample_count 1 | 2.0 (cold 49.9) | 0.14 | torch 2.5.1+cu124, pandas 2.2.2, numpy 2.2.6, hf_hub 0.33.1 | small b082dfcb, tokenizer 59d85f6a |
| M4 regime card (96x96 numpy/PIL candle+volume image, CLIP ViT-B/32 OpenAI) | mm_env | Y | Y: embedding (1,512) | 4.5 (cold 90) | 0.62 | open_clip_torch 3.3.0, mplfinance 0.12.10b0, lightgbm 4.7.0, scikit-learn 1.7.2, pillow 13.0.0, timm 1.0.30 | e6d1bd77 |
| M5 VisionTS (MAE base, ImageNet) | mm_env | Y | Y: ctx 2016 five-min log RV, period 288, pred 12 | 1.4 (cold ~150 incl. 448 MB download) | 0.47 | visionts 1.0.1 (needed `pip install requests` extra) | c551c37b (mae_visualize_vit_base.pth) |
| M6 FAISS retrieval | mm_env | Y | Y: 1,000 windows x 20 descriptors, IndexFlatL2, k=50 | 0.03 (CPU) | 0 | faiss-cpu 1.15.1 | none |

Full sha256 values and sizes: `weights_sha256.txt`; pip freezes: `versions_mm_env.txt`, `versions_mm_legacy.txt`; commands: `install_log.txt`, `install_log_legacy.txt`. Loaders: `M?_*/load.py` (each runs the smoke: `HF_HOME=D:/nt/nt_mm_cache/hf <python> M?_*/load.py`). `M4_regime_card/sample_card.png` is the rendered card.

## Exact install commands that worked
- mm_env: `conda.exe create -y -p D:/nt/nt_mm_env python=3.11`; `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126`; `pip install chronos-forecasting "pandas[pyarrow]"`; `pip install granite-tsfm`; `pip install open_clip_torch mplfinance lightgbm scikit-learn pillow`; `pip install visionts requests`; `pip install faiss-cpu`; then repair (below): `pip install --force-reinstall --no-deps torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126`.
- mm_legacy: `conda.exe create -y -p D:/nt/nt_mm_legacy python=3.10`; `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`; `git clone https://github.com/shiyu-coder/Kronos D:/nt/nt_mm_cache/src_kronos`; `pip install -r src_kronos/requirements.txt`.
- Sundial / transformers 4.40.1 (rank 10, not tonight) was not installed.

## Gotchas
- **Never `pip install` a package into mm_env without re-checking torch.** `open_clip_torch` (via torchvision) pulled torch 2.14.1 (CPU build from PyPI) over the CUDA wheel, which also broke granite-tsfm's `torch<2.12` pin. Repaired once with the force-reinstall above; torch.cuda.is_available() True after it. Install any further package with `--no-deps` or add `torch==2.11.0+cu126 torchvision==0.26.0+cu126` to the command.
- Chronos-2: `predict_df` needs the column `item_id` (not `id`); output timestamps run into 2025-07-21 (forecast labels, no data read). Needs HF Hub on first use.
- TTM: use `tsfm_public.toolkit.get_model.get_model(repo, context_length=512, prediction_length=P)`; the r2 repo downloads a different branch snapshot per (context, prediction length). Point forecast only. Input must be normalised by the caller.
- Kronos: `KronosPredictor.predict(sample_count=N)` AVERAGES the N paths internally and returns one frame; for path quantiles or P(hit) call it with sample_count=1 in a loop (or use the model's sampling API directly: needs a stage-2 code read). pred index labels are the y_timestamp passed. Kronos needs `cwd`/`sys.path` = the clone. Its requirements pinned torch only as >=2.0 (installed 2.5.1+cu124 before, kept).
- VisionTS: weights come from dl.fbaipublicfiles.com (not HF); `visionts` imports `requests` without declaring it. Context must give >= 1 period after padding; `bars_5m(2016)` needs the extra row margin in `common_data.py` (fixed). On BTC the library docs advise `fp64=True` for overflow in some data; not needed here.
- open_clip: the OpenAI tag needs the `ViT-B-32-quickgelu` model name (plain `ViT-B-32` warns about a QuickGELU mismatch).
- Windows: the HF cache holds symlink-style files; hash with `cat f | sha256sum`.
- GPU peaks are `torch.cuda.max_memory_allocated` of one forward pass (well under 6 GB). Only one GPU process of mine ran at a time.

## Disk
mm_env 5.0 GB, mm_legacy 4.8 GB, mm_cache 6.7 GB (includes pip cache and the Kronos clone). D: free 43 GB after install (59 GB before).
