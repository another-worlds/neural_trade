# Runtime Backend Setup: CUDA vs DirectML

This project now includes backend-aware GRU construction in the model runtime:

- If TensorFlow is a CUDA build, GRU stays cuDNN-eligible for best performance.
- If TensorFlow is not a CUDA build (for example DirectML), GRU switches to a non-cuDNN-safe path to avoid CudnnRNN kernel errors.

## What Changed

The model recurrent block in model construction now routes through runtime-aware helpers:

- Build-info detection using TensorFlow runtime metadata.
- CUDA mode uses cuDNN-capable GRU settings.
- Non-CUDA mode uses a DirectML-safe GRU configuration.
- Compatibility fallback handles runtimes that do not expose the GRU use_cudnn argument.

## Why This Was Needed

On some Windows + DirectML setups, TensorFlow can list GPU devices but still report:

- is_cuda_build = False

In that case, Keras may still attempt a cuDNN RNN execution path unless explicitly guarded, which can fail with a CudnnRNN OpKernel registration error.

## Drawbacks and Trade-Offs

### DirectML path (non-CUDA)

- Lower throughput than native CUDA/cuDNN for recurrent layers in most cases.
- Non-cuDNN GRU path may increase training time.
- Slightly different numerical behavior may occur versus cuDNN kernels.
- The non-CUDA safeguard sets recurrent_dropout in GRU to force a generic kernel path, which can reduce raw training speed.

### CUDA path

- Fastest path for GRU remains available when TensorFlow is CUDA-enabled.
- Requires compatible NVIDIA driver + CUDA + cuDNN stack.

## Setup Instructions

### 1) Choose environment

### Option A: CUDA environment (preferred for speed)

1. Use a TensorFlow build that reports CUDA support.
2. Ensure NVIDIA driver and CUDA/cuDNN dependencies are compatible with that TensorFlow version.
3. Verify TensorFlow runtime metadata before training.

### Option B: DirectML environment (Windows fallback)

1. Use your DirectML-enabled TensorFlow environment.
2. Confirm TensorFlow is not a CUDA build.
3. Train normally; model runtime will use the non-cuDNN GRU path automatically.

### 2) Verify backend before training

Run this diagnostic cell:

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
print(tf.sysconfig.get_build_info().get("is_cuda_build"))
print(tf.sysconfig.get_build_info())
```

Interpretation:

- is_cuda_build == True: CUDA/cuDNN path is available.
- is_cuda_build == False: DirectML or CPU path; model will avoid cuDNN-only GRU execution.

### 3) Confirm model branch (optional)

The backend branch is selected automatically by model runtime helpers and does not require manual flags in notebook code.

## Troubleshooting

- If you still see CudnnRNN kernel errors in a non-CUDA build, reload the model module in the notebook and rebuild the model to ensure the updated helpers are active.
- If performance is too slow in DirectML mode, use a CUDA-enabled environment for recurrent workload acceleration.
