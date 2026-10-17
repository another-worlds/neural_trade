"""Precision check (legacy env): the same 48 anchors, N=4, seed fixed, fp32 vs tf32 vs bf16 autocast. Reports tokenizer round-trip error and the path statistics."""
import sys, time, numpy as np, torch
sys.argv = ["x"]; import run_kronos as R
z = np.load(R.CACHE + "/inputs_cut.npz"); idx = np.arange(5000, 5000 + 48 * 100, 100)
xn, xs, ys, mean, std = R.prep(z["x"][idx], z["ts_ns"][idx]); model, tok = R.load(); c0 = z["x"][idx][:, -1, 3]
def stats(p):
    hi, lo, cl = p[..., 1], p[..., 2], p[..., 3]
    rng = (hi.max(2) - lo.min(2)) / c0[:, None]; lr = np.diff(np.log(np.concatenate([np.repeat(c0[:, None, None], p.shape[1], 1), cl[..., :]], 2)), axis=2)
    return float(np.median(rng)), float(np.sqrt((lr ** 2).mean())), float(np.mean(np.abs(p[..., 3] / c0[:, None, None] - 1)))
res = {}
for mode in ["none", "tf32", "bf16"]:
    torch.backends.cuda.matmul.allow_tf32 = mode != "none"; torch.manual_seed(1); t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode == "bf16"):
        x = torch.from_numpy(xn).cuda(); rt = tok.decode(tok.encode(x, half=True), half=True).float()
        err = float((rt - x).abs().mean()); p = R.sample_paths(model, tok, xn, xs, ys, mean, std, 4)
    torch.cuda.synchronize(); res[mode] = (err, stats(p), round(time.time() - t, 1), p)
    print(mode, "tokenizer roundtrip mean|err| (normalised)", round(err, 4), "median range, rms 5m ret, mean |close/c0-1|", stats(p), "secs", res[mode][2], flush=True)
# paired comparison of range median per anchor between modes (different RNG paths with different numerics: distribution only)
for m in ["tf32", "bf16"]:
    a = res["none"][3]; b = res[m][3]
    ra = np.median((a[..., 1].max(2) - a[..., 2].min(2)), 1); rb = np.median((b[..., 1].max(2) - b[..., 2].min(2)), 1)
    print(m, "corr of per-anchor median range vs fp32:", round(float(np.corrcoef(np.log(ra), np.log(rb))[0, 1]), 3))
