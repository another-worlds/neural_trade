# REPORT: window_free_kit_v1 (NT-060)

GPU run of `scripts/bench/window_free.py` on branch `nt-048` at `d1f6fa9` (the commit that records `fwd_sha256` and calls `enable_op_determinism` on `--device gpu`). TF32 was left at its default. Both runs used `--device gpu --reps 5`. Kit time is the script's own `seconds` field: 9.94 s then 9.76 s, about 20 s together, under the half-hour budget.

| file | device | TF | op determinism | TF32 execution | visible GPU | kit seconds |
|---|---|---|---|---|---|---|
| `run_a.json` | gpu | 2.10.0 | true | true | `/physical_device:GPU:0` | 9.94 |
| `run_b.json` | gpu | 2.10.0 | true | true | `/physical_device:GPU:0` | 9.76 |

## Gates

| clause | result |
|---|---|
| Runs with op determinism on, no error | PASS. Both processes exited 0. `g_a1.census.offending_ops` is empty on the A2 layer. |
| Precision within 1e-5 x max\|state\| and 3e-5 x RMS, TF32 at its default | PASS. `g_a1.precision.PASS` is true in both files, and the two precision blocks are equal. Worst max-relative error is 1.143e-6 (per-bar `bb_var_ewma_d2`). Worst RMS-relative error is 9.309e-7 (constant `bb_var_ewma_d2`). Limit is 1e-5 and 3e-5. T = 43,008. |
| Two runs bitwise identical | PASS on the ten window-free cases (kernel V1, assembly D6b, A2 layer): each `fwd_sha256` matches. FAIL on `today_LearnableIndicators`: the two forwards differ. Recorded below. No code change. |
| A2 forward+backward at most 1.10x today's layer | PASS at the 7-day span (N = 11,504): 0.715 and 0.726. PASS at N = 30,720 on run A (1.072). FAIL at N = 30,720 on run B (1.128). Recorded below. No code change. |

NT-061 (keep TF32 or turn it off) is not decided here. Today's layer census lists `Einsum` and `MatMul`. The A2 layer lists none.

## GPU-free check

RUNBOOK rule: busy if framebuffer is above 2000 MB or the median `sm` is above 30%. Target drive D:.

Before run A, `nvidia-smi dmon -s um -c 10`: framebuffer 514–528 MB; `sm` samples 24, 23, 27, 16, 19, 24, 26, 20, 23, 24 (median 23.5). Disk at that check: C: 18 GB free, D: 35 GB free.

Before run B, `nvidia-smi dmon -s um -c 8`: framebuffer 514 MB on every sample; `sm` samples 12, 24, 20, 19, 24, 25, 20, 24 (median 22).

## Hash of one forward

`fwd_sha256` is SHA-256 of one forward after the op census. Equal means the two files hold the same digest.

| case | equal | run A |
|---|---|---|
| `kernel_v1/N11504_K31` | yes | `c2ebe1b71eb25e9267b34465c9adc1dc11cebec8a9fe4251773891ef27e8d40c` |
| `kernel_v1/N11504_K87` | yes | `7b8803d38fa0efd0aad35a4d77ed13d37366c6667a4c7f814b4b6b39d03c004c` |
| `kernel_v1/N30720_K31` | yes | `a4ca765f0925137b9db5fb3397977acc84d705676234045cbaaa661358e91bc3` |
| `kernel_v1/N30720_K87` | yes | `c0c48cca5ac3dd78775ca05eb0b15171f4a704639ee19f076043af7ef21495bf` |
| `assembly_d6b/N11504_C31` | yes | `0e7febcc87a0e919ac4c15e4cfbffc36ea8ce1af55d14bc0fadcd62fdd83e3f9` |
| `assembly_d6b/N11504_C87` | yes | `f3521b0238db3ebf0665b87ec7f5d8b7c9515a6c92047d9c06ae77417f4a5188` |
| `assembly_d6b/N30720_C31` | yes | `7d20d5cedafc01966a3dc62969a6d4493db0cab00551088b7b720572252d2c5e` |
| `assembly_d6b/N30720_C87` | yes | `ec4d1557c52c293eadcfb0d37d1261498b5d3a44f6fde9ec66a9adf1736f4ae0` |
| `a2_layer/N11504` | yes | `85365a62781b0efabfc296cef1d562da72ea1fef881e6df0eaa66b6d59fa3299` |
| `a2_layer/N30720` | yes | `051040b2efba954e006efc3fc4201d8ef73ecfc7e51d0355a122f3fc27924fc9` |
| `today_layer/today_LearnableIndicators` | no | run A `c7c28fab1c3535f42e92ca01e366f83247e45f8e59e813d5c87755ea0e9f6dff`; run B `97674dc4f2f62a4e4d8239a2e7db358f7d77b3d972a489c9608de58b8886adc3` |

The window-free cases build their variables from `numpy` `Generator(0)`. Today's layer does that for the window indices and the loss weights, then builds `LearnableIndicators` and a `Dense(18, tanh)` with Keras initializers. The Dense kernel is a fresh Glorot draw in each process, so the forward bytes are not a repeat of the same weights. The ten matching hashes are the bitwise check of kernel V1, assembly D6b, and the A2 layer on this GPU.

## Timing

Each cell is the median of 5 interleaved forward+backward repeats (`cpu_fwdbwd` in the JSON; the run was on the GPU). The ratio is that median divided by today's layer in the same file. Today's layer is one batch of 256 windows of length 60. The A2 time is one pass of N bars assembled into 256 windows.

| run | case | median ms | min–max ms | ratio to today | 1.10 gate |
|---|---|---|---|---|---|
| A | A2 N=11504 | 11.298 | 9.861–13.603 | 0.715 | pass |
| A | A2 N=30720 | 16.950 | 16.560–20.498 | 1.072 | pass |
| A | today | 15.811 | 4.245–28.304 | 1 |  |
| B | A2 N=11504 | 11.464 | 10.781–11.939 | 0.726 | pass |
| B | A2 N=30720 | 17.802 | 16.807–20.619 | 1.128 | fail |
| B | today | 15.788 | 3.429–30.804 | 1 |  |

Today's layer spread is wide (IQR 13.4 ms on run A, 18.6 ms on run B). The A2 medians are tight (IQR under 1 ms). The 1.128 ratio is the failed gate. It is recorded here and does not change the kit.
