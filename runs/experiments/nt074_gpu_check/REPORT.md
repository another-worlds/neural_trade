# NT-074 GPU check: same-seed GPU runs differ at epoch 0 - source identified

Written by the experimenter, 2026-10-01. Pinned worktree D:/nt_exp_nt074_gpu, code sha d71e3b4
(origin/remediation/plan, the head that carries NT-074's merged CPU fix). Branch nt-074-gpu.

**Every process in this check ran with PYTHONPATH=D:/nt_exp_nt074_gpu/src, set explicitly before
each python invocation, in the same shell command that launched it.** Proof, not just the setting:

1. Every run's env fingerprint (stderr) records "git": "d71e3b4" - the pinned worktree's own
   checked-out commit, not fde4f30 (the stale nt-099 branch checked out in the main checkout
   D:/neural_trade at the time of the lead's note).
2. From the seeded_layers runs onward (after the lead's mid-task note), the harness itself prints
   neural_trade.__file__ to stderr before anything else imports: every run after that point
   recorded "neural_trade_file": "D:\\nt_exp_nt074_gpu\\src\\neural_trade\\__init__.py".
3. The three baseline runs finished before the harness had that print line added. They are not
   re-run, because the evidence already rules out the stale checkout: env_fingerprint.git is
   computed from the process's own working directory/git state (d71e3b4, the worktree's sha, not
   fde4f30); and an independent one-liner run with the exact same PYTHONPATH, issued immediately
   after, printed neural_trade.__file__ = D:\nt_exp_nt074_gpu\src\neural_trade\__init__.py -
   confirming the import resolution for this invocation pattern. No result in this report rests on
   code from the stale checkout.

GPU-free check before starting: nvidia-smi dmon -s um -c 10 read fb about 1110 MB, sm 0-2 percent
(idle; the owner's other project was not running). No other GPU job ran during this check (one GPU
job at a time).

## Setup

BTC/USDT 1-minute, configs/default.yaml (today's default: OHLCV, 14 indicator families, D-047),
window 60, horizons 10/15/20, batch 256, fold -3 (62 steps/epoch), 1 epoch (tiny-first, D-048: the
divergence NT-035 saw was already visible at epoch 0, so 1 epoch is enough to see it and keeps each
run to about 1-2 minutes). seed_everything(777, deterministic=True) called before train_and_evaluate
(same mechanism NT-035 used), TF_DETERMINISTIC_OPS=1 (set by import neural_trade). RTX 4070 Ti,
TF 2.10.0, cuDNN 8100, TF32 on (unchanged, D-001 scope).

Harness: runs/experiments/nt074_gpu_check/harness_train_once.py (a copy of NT-035's
harness_train_once.py, extended with --override KEY=VALUE, --force-gru-unroll and --cpu-only probe
flags - it still only imports and calls existing public functions; no edit to src/ or tests/).

## Results

Every run: separate process, one at a time, --epochs 1 --seed 777 --deterministic 1.
val_loss is epoch 0 validation loss from metrics.jsonl (the same number NT-035 reported as
"val_loss ... at epoch 0").

| Variant | Command (added flags) | run 1 | run 2 | run 3 | identical? |
|---|---|---|---|---|---|
| baseline (current merged code, OHLCV default) | (none) | 9.590570 | 9.705194 | 9.612173 | no - still diverges, same order of magnitude as NT-035 original (9.5673/9.6394/9.6603) |
| SEEDED_STOCHASTIC_LAYERS=True | --override SEEDED_STOCHASTIC_LAYERS=true | 9.534631 | 9.635226 | 9.559114 | no - rules out VacuumSaturationNoise unseeded tf.random.normal as the (sole) source |
| GRU forced off cuDNN (unroll=True, identical math) | --force-gru-unroll 1 | 9.572048 | 9.572048 | 9.572048 | yes - bit-for-bit identical |

Full-precision check on the unroll variant (val_loss as stored, float64 repr): 9.57204818725586
for all three runs, loss (train) also identical: 10.662149429321289.

No op raised in any run (all exit 0; stderr greps for "eterminis", "OpKernel", "UnimplementedError",
"Traceback" found nothing, matching NT-035 Part B finding).

## Source identified

The cuDNN-fused GRU kernel. models/gru_attention.py builds layers.GRU(64, ...) /
layers.Bidirectional(layers.GRU(64, ...)) and models/gru_small.py builds layers.GRU(32, ...) with
Keras cuDNN-eligible defaults (activation='tanh', recurrent_activation='sigmoid',
recurrent_dropout=0, reset_after=True, unroll=False) - so on this GPU, in graph mode, Keras picks
TF fused cuDNN RNN kernel. Forcing unroll=True (same math, same weights, same seed; it only changes
HOW the recurrence is computed - a Python-level unrolled loop of the identical GRU cell instead of
the single fused cuDNN op) makes three independent processes, same seed, same deterministic=True
mode, produce bit-identical val_loss after training. Reverting to the default (cuDNN path)
reproduces the divergence. This isolates the cuDNN GRU kernel as the source: TF 2.10
tf.config.experimental.enable_op_determinism() does not cover cuDNN fused RNN (GRU/LSTM) kernel.
This matches TensorFlow own documented exception list for enable_op_determinism (cuDNN RNN ops are
not guaranteed bit-exact across runs even with the global seed and op-determinism flag set - a
known TF/cuDNN limitation, not a neural_trade bug): cuDNN RNN implementation uses an internal
parallel-reduction strategy for its backward pass whose order is not pinned by TF determinism flag,
so gradients (and hence the trained weights, and hence epoch 0 val_loss) differ between runs even
though every Python-visible RNG is seeded identically.

Ruled out: VacuumSaturationNoise legacy stateful tf.random.normal (SEEDED_STOCHASTIC_LAYERS probe
still diverged, by about the same magnitude as baseline) and the CPU-identified Grappler
arithmetic-rewrite leak (already off by default for the OHLCV config via the merged NT-074 CPU fix,
confirmed by set_arithmetic_rewrite own rule - len(INPUT_SERIES) <= 1 is false for OHLCV - yet the
GPU still diverged until the cuDNN GRU was bypassed). Dropout legacy-stateful kernel (used by two
layers.Dropout(0.1) calls and MultiHeadAttention internal dropout) was not probed separately: the
unroll fix alone already gave bit-for-bit identity across all three runs, leaving no residual gap
for Dropout (or tf.data shuffle order, or CPU-pinned ops) to explain. tf.data shuffle/map-order and
CPU-pinned-ops candidates were therefore not run - the evidence already accounts for the entire
divergence.

## Criterion (4): is full GPU reproducibility possible in TF 2.10

Yes, but not with the cuDNN-fused GRU path. enable_op_determinism() plus a fixed seed is sufficient
for every op in this model graph except the fused cuDNN GRU/LSTM kernel, which TF 2.10 does not
cover. Bypassing it (unroll=True, or an equivalent non-cuDNN RNN implementation) restores
bit-for-bit reproducibility at the cost of training speed (this check is not a clean speed
benchmark - each run also pays calibration/evaluation overhead unrelated to the backbone - but the
raw wall times suggest roughly 1.2-1.4x: baseline runs 68.6-93.9 s vs. unroll runs 81.1-114.6 s for
the identical 1-epoch job; an implementer item should measure sec_per_step properly per D-018
before adopting it anywhere).

## Recommendation (implementer item, not done here - this session must not edit src/)

Add a Config switch (for example DETERMINISTIC_GRU: bool, default False, unchanged behaviour) that,
when True together with deterministic=True, builds the GRU layers with unroll=True instead of the
cuDNN-eligible defaults - mirroring how set_arithmetic_rewrite is already an explicit, per-config,
process-wide call from trainer.train_and_evaluate. This gives pre-registered comparison studies
(D-025) a genuinely reproducible GPU path without changing ordinary training defaults or speed.
The implementer should: (a) confirm unroll=True reproduces identical results for more than 1 epoch
(this check only trained 1 epoch per run); (b) measure the sec_per_step cost properly (D-018);
(c) decide whether Dropout also needs seeded_stochastic_layers()-style wrapping for the
deterministic mode to extend past 1 epoch without drifting (untested here - the 1-epoch bit-for-bit
match does not by itself prove later epochs stay identical if Dropout legacy-stateful draws differ
across processes; a longer run should check this before the fix is called complete).

## GPU time used

About 15.3 minutes wall (06:06:45Z to 06:22:00Z UTC), summing to about 0.26 GPU-hours of the
1-hour budget stated for this check (well under the 3-hour item cap too). Breakdown:
- baseline x3: 68.6 + 69.9 + 93.9 = 232.4 s
- SEEDED_STOCHASTIC_LAYERS x3: 56.8 + 54.3 + 85.9 = 197.0 s
- force-gru-unroll x3: 105.8 + 81.1 + 114.6 = 301.5 s
- (plus process start-up/data-load overhead outside wall_s, included in the 15.3-minute wall total)

One GPU job at a time throughout (sequential for loops, never parallel).

## Run ids

- runs/experiments/nt074_gpu_check/baseline/20261001T060645Z-d71e3b4-3afcfb38-base_r1
- runs/experiments/nt074_gpu_check/baseline/20261001T060800Z-d71e3b4-3afcfb38-base_r2
- runs/experiments/nt074_gpu_check/baseline/20261001T060922Z-d71e3b4-3afcfb38-base_r3
- runs/experiments/nt074_gpu_check/seeded_layers/20261001T061145Z-d71e3b4-6f0834ac-seeded_r1
- runs/experiments/nt074_gpu_check/seeded_layers/20261001T061251Z-d71e3b4-6f0834ac-seeded_r2
- runs/experiments/nt074_gpu_check/seeded_layers/20261001T061353Z-d71e3b4-6f0834ac-seeded_r3
- runs/experiments/nt074_gpu_check/gru_unroll/20261001T061641Z-d71e3b4-3afcfb38-unroll_r1
- runs/experiments/nt074_gpu_check/gru_unroll/20261001T061834Z-d71e3b4-3afcfb38-unroll_r2
- runs/experiments/nt074_gpu_check/gru_unroll/20261001T062005Z-d71e3b4-3afcfb38-unroll_r3

Logs: *.stdout / *.stderr next to each group directory under runs/experiments/nt074_gpu_check/.
Harness: runs/experiments/nt074_gpu_check/harness_train_once.py.

## Deviations from the task bisection plan

- tf.data shuffle/map-order and CPU-pinned-ops/XLA candidates were not run: the cuDNN-GRU probe
  alone already explained the full divergence (bit-for-bit match), so spending more GPU time ruling
  out already-explained candidates seemed wasteful against the 1-hour budget. If the implementer fix
  (above) does not hold past 1 epoch, those candidates are the next thing to check.
- Only 1 epoch was trained per run (NT-035 used 3); the divergence is visible at epoch 0 already, so
  1 epoch was enough to identify the source within the tiny-first budget (D-048).
