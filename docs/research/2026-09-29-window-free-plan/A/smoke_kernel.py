"""Smoke test of kernel.linrec against the float64 recursion (small sizes, random data, resets, h0)."""
from common import np, tf
from kernel import NEG, linrec, ref_linrec

rng = np.random.default_rng(0)
for T in (1, 5, 64, 65, 200, 2048, 5000):
    K = 3
    la = -rng.uniform(0.001, 2.0, size=(K, T))
    if T > 10:
        la[:, T // 3] = NEG                         # a reset
    b = rng.normal(size=(K, T))
    h0 = rng.normal(size=K)
    ref = ref_linrec(la, b, h0)
    for C, G in ((64, 32), (128, 32), (4, 2), (64, 10 ** 9)):
        for seg in ("segsum", "segsub"):
            if seg == "segsub" and T > 10:
                continue                            # segsub cannot represent a reset
            for mode in ("mulsum", "einsum"):
                h, hl = linrec(tf.constant(la, tf.float32), tf.constant(b, tf.float32),
                               tf.constant(h0, tf.float32), C=C, G=G, seg=seg, contract=mode)
                h = h.numpy()
                err = np.abs(h - ref).max()
                ok = err < 1e-4 * max(1.0, np.abs(ref).max()) and np.allclose(hl.numpy(), h[:, -1])
                if not ok:
                    print("FAIL", T, C, G, seg, mode, err)
    # float64 kernel must match the recursion to ~1e-12
    h64, _ = linrec(tf.constant(la), tf.constant(b), tf.constant(h0), C=64, G=32)
    e64 = np.abs(h64.numpy() - ref).max()
    assert e64 < 1e-10, (T, e64)
print("smoke OK")
