# NT-114 CPU step-time check: DETERMINISTIC_GRU off vs on

Script `time_step.py` (this folder), numbers in `raw.json`. CPU only (CUDA_VISIBLE_DEVICES=-1). Production input shape: BATCH 256 x LOOKBACK 60 x 5 channels. Per model and per seed (3 seeds): build off and on, copy the weights, 5 warm-up + 30 timed forward+backward steps (tf.function, training=True, mean square of every head as the loss), off/on alternated (order flipped on odd seeds); 90 timed steps per cell. This is the model fwd+bwd step (no loss terms, no optimizer), not the full CustomTrainModel step.

| model | off mean (s) | off median (s) | on mean (s) | on median (s) | on/off mean | on/off median |
|---|---|---|---|---|---|---|
| gru_attention | 1.763 | 1.727 | 1.976 | 1.789 | 1.121 | 1.036 |
| gru_small | 0.801 | 0.783 | 0.886 | 0.857 | 1.107 | 1.094 |

**Correction 2026-10-06 (QA, quiet machine, same script unchanged; `D:/nt/nt_qa/nt114_cpu_rerun/`): on/off ratio (mean / median) gru_attention 0.978 / 0.944, gru_small 0.959 / 0.983. The two runs disagree in direction, so the CPU cost of unrolling cannot be told apart from 0; the table below was taken under load and its 'costs 4-12 %' reading is withdrawn.**

Original reading (withdrawn): unrolling costs about 4-12 % (gru_attention) and 9-11 % (gru_small) on the CPU. Caveat (an estimate, not a clean measurement): a QA full-suite run was loading the machine, so absolute seconds are inflated and the mean/median spread shows the noise; only the ratios mean anything. The earlier recovered-WIP run (BATCH 64, quiet machine, no script kept) gave 1.209 / 1.116. On the CPU Keras has no fused GRU kernel to lose (the off path is already the generic while-loop), so this is a graph-size effect and says little about the GPU, where off is the fused cuDNN kernel. The GPU sec_per_step is the experimenter's measurement (NT-114 acceptance 4; D-018: an opt-in path, reported, not gated).
