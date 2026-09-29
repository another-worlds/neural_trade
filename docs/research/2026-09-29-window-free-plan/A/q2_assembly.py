"""Q2: assembling the network's [B, L, C] windows from the in-graph series F [N, C] at the batch's anchors
without an op that has no deterministic GPU kernel in TF 2.10 (op determinism is on in every process).

Variants (X[b, l, :] = F[anchor_b - (L - 1) + l, :]):
  D0  tf.gather                               bwd: IndexedSlices -> UnsortedSegmentSum      (RAISES on GPU)
  D0b tf.gather_nd                            bwd: IndexedSlices -> UnsortedSegmentSum      (RAISES on GPU)
  D9  tf.signal.frame + gather                bwd: gather inside frame -> UnsortedSegmentSum (RAISES)
  D10 tf.image.extract_patches + gather       bwd: SparseTensorDenseMatMul                  (RAISES)
  D1  L strided slices (all windows of the block) + one-hot matmul select   (fwd matmul: TF32 on GPU)
  D1b L strided slices + custom anchor gather whose bwd is a gather (anchors distinct)
  D2  banded one-hot matmul [B, L, N] x [N, C]                              (fwd+bwd matmul: TF32)
  D3  custom gradient: fwd gather, bwd one-hot matmul                       (bwd matmul: TF32)
  D6a custom gradient: fwd gather, bwd "transpose by gather": inverse map bar -> batch row
      (in-graph Equal/Sum), one gather of the padded upstream gradient [N, L, C], reduce_sum over L
  D6b as D6a with the inverse map computed on the host (tf.data / numpy) and fed as an input
  D7  custom gradient: fwd gather, bwd UnsortedSegmentSum pinned to /CPU:0   (device round trip on GPU)
  D8  custom gradient: fwd gather, bwd tf.scatter_nd                        (GPU: DoScatterNdOnCpu + host sync)
Every variant also adds the price-relative term rel[b, l] * PRICE_TYPE (data only, no gradient).

Command: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q2_assembly.py
Output:  q2_results.json
"""
import time

from common import dump, env_info, graph_census, interleaved_timing, np, tf

tf.config.threading.set_inter_op_parallelism_threads(4)
B, L = 256, 60


def idx_of(anchors):
    return anchors[:, None] - (L - 1) + tf.range(L)[None, :]


def D0(F, anchors, inv):
    return tf.gather(F, idx_of(anchors))


def D0b(F, anchors, inv):
    return tf.gather_nd(F, idx_of(anchors)[..., None])


def D9(F, anchors, inv):
    frames = tf.signal.frame(F, L, 1, axis=0)                          # [N-L+1, L, C], frame i = bars i..i+L-1
    return tf.gather(frames, anchors - (L - 1))


def D10(F, anchors, inv):
    N, C = F.shape
    p = tf.image.extract_patches(tf.reshape(F, [1, N, 1, C]), sizes=[1, L, 1, 1], strides=[1, 1, 1, 1],
                                 rates=[1, 1, 1, 1], padding="VALID")
    return tf.gather(tf.reshape(p, [N - L + 1, L, C]), anchors - (L - 1))


def _all_windows(F):
    N = F.shape[0]
    S = N - L + 1
    return tf.stack([F[l:l + S] for l in range(L)], axis=1)            # [S, L, C]: row a = window of anchor a+L-1


def D1(F, anchors, inv):
    W = _all_windows(F)
    oh = tf.one_hot(anchors - (L - 1), W.shape[0], dtype=F.dtype)      # [B, S]
    return tf.einsum("bs,slc->blc", oh, W)


def D1b(F, anchors, inv):
    W = _all_windows(F)
    S = W.shape[0]
    rows = anchors - (L - 1)

    @tf.custom_gradient
    def pick(W):
        def grad(dy):                                                  # anchors distinct: row a gets dy[inv_a] or 0
            eq = tf.cast(tf.equal(tf.range(S)[:, None], rows[None, :]), tf.int32)
            inv_r = tf.reduce_sum(eq * tf.range(B)[None, :], 1) + B * (1 - tf.reduce_sum(eq, 1))
            dyp = tf.concat([dy, tf.zeros([1, L, dy.shape[-1]], dy.dtype)], 0)
            return tf.gather(dyp, inv_r)
        return tf.gather(W, rows), grad
    return pick(W)


def D2(F, anchors, inv):
    oh = tf.one_hot(idx_of(anchors), F.shape[0], dtype=F.dtype)         # [B, L, N]
    return tf.einsum("bln,nc->blc", oh, F)


def D3(F, anchors, inv):
    idx = idx_of(anchors)
    N, C = F.shape

    @tf.custom_gradient
    def g(F):
        def grad(dy):
            oh = tf.one_hot(tf.reshape(idx, [-1]), N, dtype=dy.dtype)   # [B*L, N]
            return tf.matmul(oh, tf.reshape(dy, [-1, C]), transpose_a=True)
        return tf.gather(F, idx), grad
    return g(F)


def _transpose_by_gather(dy, inv, N):
    """dF[n] = sum_l dy[inv[n + L - 1 - l], l] (inv = batch row of the anchor at each bar, B if none)."""
    C = dy.shape[-1]
    nl = tf.range(N)[:, None] + (L - 1) - tf.range(L)[None, :]         # [N, L]: anchor bar feeding bar n via l
    flat = tf.gather(inv, nl) * L + tf.range(L)[None, :]
    dyp = tf.concat([tf.reshape(dy, [B * L, C]), tf.zeros([L, C], dy.dtype)], 0)   # rows B*L.. are zeros
    return tf.reduce_sum(tf.gather(dyp, flat), axis=1)


def D6a(F, anchors, inv):
    idx = idx_of(anchors)
    N = F.shape[0]

    @tf.custom_gradient
    def g(F):
        def grad(dy):
            eq = tf.cast(tf.equal(tf.range(N + L - 1)[:, None], anchors[None, :]), tf.int32)   # [N+L-1, B]
            inv_ = tf.reduce_sum(eq * tf.range(B)[None, :], 1) + B * (1 - tf.reduce_sum(eq, 1))
            return _transpose_by_gather(dy, inv_, N)
        return tf.gather(F, idx), grad
    return g(F)


def D6b(F, anchors, inv):
    idx = idx_of(anchors)
    N = F.shape[0]

    @tf.custom_gradient
    def g(F):
        def grad(dy):
            return _transpose_by_gather(dy, inv, N)
        return tf.gather(F, idx), grad
    return g(F)


def D7(F, anchors, inv):
    idx = idx_of(anchors)
    N, C = F.shape

    @tf.custom_gradient
    def g(F):
        def grad(dy):
            with tf.device("/CPU:0"):
                return tf.math.unsorted_segment_sum(tf.reshape(dy, [-1, C]), tf.reshape(idx, [-1]), N)
        return tf.gather(F, idx), grad
    return g(F)


def D8(F, anchors, inv):
    idx = idx_of(anchors)
    N, C = F.shape

    @tf.custom_gradient
    def g(F):
        def grad(dy):
            return tf.scatter_nd(tf.reshape(idx, [-1, 1]), tf.reshape(dy, [-1, C]), [N, C])
        return tf.gather(F, idx), grad
    return g(F)


VARIANTS = {"D0_gather": D0, "D0b_gather_nd": D0b, "D9_signal_frame": D9, "D10_extract_patches": D10,
            "D1_shiftstack_onehot_select": D1, "D1b_shiftstack_gather_bwd": D1b, "D2_banded_onehot_matmul": D2,
            "D3_gather_fwd_onehot_matmul_bwd": D3, "D6a_gather_fwd_transpose_by_gather_bwd": D6a,
            "D6b_same_host_inverse_map": D6b, "D7_gather_fwd_cpu_segment_sum_bwd": D7,
            "D8_gather_fwd_scatter_nd_bwd": D8}
TF32_EXPOSED = {"D1_shiftstack_onehot_select": "forward (features rounded to TF32)",
                "D2_banded_onehot_matmul": "forward and backward",
                "D3_gather_fwd_onehot_matmul_bwd": "backward only (gradient rounded to TF32)"}


def price_type(C):
    p = np.zeros(C, np.float32)
    p[: max(1, C * 13 // 31)] = 1.0                                     # 13 of today's 31 channels are price-level
    return tf.constant(p)


def build(N, C, seed=0):
    rng = np.random.default_rng(seed)
    anchors = np.sort(rng.choice(np.arange(L - 1, N), size=B, replace=False)).astype(np.int32)
    rng.shuffle(anchors)
    inv = np.full(N + L - 1, B, np.int32)
    inv[anchors] = np.arange(B, dtype=np.int32)
    V = tf.Variable(rng.normal(size=(N, C)).astype(np.float32))
    cum = tf.constant(np.cumsum(rng.normal(size=N)).astype(np.float32))
    W = tf.constant(rng.normal(size=(B, L, C)).astype(np.float32))
    return anchors, inv, V, cum, W


def make_fns(fn, N, C, V, cum, W):
    pt = price_type(C)
    sig = [tf.TensorSpec([B], tf.int32), tf.TensorSpec([N + L - 1], tf.int32)]

    def fwd(anchors, inv):
        F = V * 1.0                                                     # dense producer, like the kernel output
        idx = idx_of(anchors)
        rel = tf.gather(cum, idx) - tf.gather(cum, anchors)[:, None]
        return fn(F, anchors, inv) + rel[..., None] * pt

    def fb(anchors, inv):
        with tf.GradientTape() as tape:
            X = fwd(anchors, inv)
            loss = tf.reduce_sum(X * W)
        return X, tape.gradient(loss, V)
    return tf.function(fwd, input_signature=sig), tf.function(fb, input_signature=sig)


def main():
    out = {"env": env_info(), "B": B, "L": L, "cases": {}}
    for N in (11520, 30720):
        for C in (31, 87):
            anchors, inv, V, cum, W = build(N, C)
            a_t, inv_t = tf.constant(anchors), tf.constant(inv)
            small = (N, C) == (11520, 31)          # the two library traps are only censused on the small case
            fns = {k: make_fns(f, N, C, V, cum, W) for k, f in VARIANTS.items()
                   if small or k not in ("D9_signal_frame", "D10_extract_patches")}
            case = {}
            X0, g0 = fns["D0_gather"][1](a_t, inv_t)
            X0, g0 = X0.numpy(), g0.numpy()
            for k, (fwd, fb) in fns.items():
                cf = graph_census(fwd.python_function, a_t, inv_t)
                cb = graph_census(fb.python_function, a_t, inv_t)
                bwd_types = {t: cb["types"][t] - cf["types"].get(t, 0) for t in cb["types"]
                             if cb["types"][t] - cf["types"].get(t, 0) > 0}
                X, g = fb(a_t, inv_t)
                X, g = X.numpy(), g.numpy()
                case[k] = {
                    "fwd_total_ops": cf["total_ops"], "fwd_compute_ops": cf["compute_ops"],
                    "fwdbwd_total_ops": cb["total_ops"], "fwdbwd_compute_ops": cb["compute_ops"],
                    "fwd_op_types": cf["types"], "bwd_only_op_types": bwd_types,
                    "RAISES_on_GPU_under_determinism": cb["raise_on_gpu"],
                    "host_round_trip_on_GPU": cb["host_round_trip_on_gpu"],
                    "tf32_exposed_matmul": TF32_EXPOSED.get(k),
                    "largest_intermediate_MB": cb["largest_output_bytes"] / 2 ** 20,
                    "X_bitwise_equal_to_gather": bool(np.array_equal(X.view(np.uint32), X0.view(np.uint32))),
                    "X_max_abs_diff": float(np.abs(X - X0).max()),
                    "dF_max_rel_diff_vs_gather_grad": float(np.abs(g - g0).max() / np.abs(g0).max()),
                }
                print(f"N={N} C={C} {k}: ops fwd+bwd {cb['total_ops']} ({cb['compute_ops']} compute) "
                      f"raises={cb['raise_on_gpu']} host={cb['host_round_trip_on_gpu']} "
                      f"X==gather {case[k]['X_bitwise_equal_to_gather']} dFrel {case[k]['dF_max_rel_diff_vs_gather_grad']:.1e}")
            timed = {k: (lambda fb=fb: fb(a_t, inv_t)) for k, (fwd, fb) in fns.items()}
            t0 = time.perf_counter()
            tim = interleaved_timing(timed, reps=7, warm=1)
            for k in case:
                case[k]["cpu_fwdbwd"] = tim[k]
            print(f"  timing {time.perf_counter() - t0:.1f}s:",
                  {k: round(v["median_s"] * 1e3, 2) for k, v in tim.items()})
            out["cases"][f"N{N}_C{C}"] = case
    dump("q2_results.json", out)


if __name__ == "__main__":
    main()
