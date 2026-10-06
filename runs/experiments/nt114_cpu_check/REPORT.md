# NT-114 CPU step-time check: DETERMINISTIC_GRU off vs on

Tiny setup: LOOKBACK=60, BATCH=64, 5 warm-up + 30 timed forward+backward steps (tf.function, training=True) per seed, 3 seeds, off/on alternated per seed. CPU only (CUDA_VISIBLE_DEVICES=-1). This is a CPU comparison, not the GPU sec_per_step the definition of done tracks (D-018) - that measurement is the experimenter's step, on the GPU, per the item's instructions.

| model | off mean (s) | off median (s) | on mean (s) | on median (s) | on/off ratio |
|---|---|---|---|---|---|
| gru_attention | 0.17705 | 0.16681 | 0.21398 | 0.20313 | 1.209 |
| gru_small | 0.07626 | 0.07847 | 0.08512 | 0.08814 | 1.116 |
