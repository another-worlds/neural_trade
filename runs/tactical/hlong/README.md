# hlong: does a longer hourly context add direction skill? (exploratory, dev only)

Monthly walk-forward on the 1h cell (1-bar label, 1e-3 deadband, `recent/wf.split_masks`), months 2019-01..2024-06 for 1-year
training windows (66 months) and 2020-03..2024-06 for 3-year windows (52; a 3-year window needs data from 2017-02). The hourly
period from 2024-07 is not read (`hwf.load` drops it; a test pins it). Cells: mean AUC with 95% t-interval over months.
Models: tb7 logistic (baseline), `ctx` logistic on multi-scale context (20 features), patch transformer and TCN (20-22k params)
over the last 168 / 336 hourly bars (7 scale-free channels) with a frozen linear skip at the logistic solution and a zero-init
residual, and ensembles. Early stopping on the last 15% of each training window (epoch 0 = the linear start).
Files: `hfeats.py` (causal features), `hmodel.py`, `hwf.py` (walk-forward), `analyze.py` -> `tables.txt`, `results_T365.json`,
`results_T1095.json`; raw per-month rows `walk_T*.jsonl`; `sanity_learns.py` (planted pattern: nets go 0.49 -> 0.79 / 0.83, so they can learn);
`test_hlong.py` (5 tests, seconds).

Result: no gain. Overall (T = 365): tb7 0.5755, ctx 0.5737, nets 0.5704-0.5709, ens_all 0.5718; paired diff vs tb7 is -0.004 to -0.005,
intervals include 0 for the ensembles and just exclude it for single nets (patch -0.0051 [-0.0099,-0.0003]). T = 1095: tb7 0.5700, nets
0.5618-0.5633, paired diff -0.007 to -0.008, intervals exclude 0 (worse). The nets leave the linear start in 76-89% of months
(best epoch > 0) but their AUC against their own start is within +-0.003 (0.0013 [-0.0002,+0.0028] at best): the residual moves
calibration, not ranking. See `tables.txt` for per-year tables and the best-epoch histograms.
