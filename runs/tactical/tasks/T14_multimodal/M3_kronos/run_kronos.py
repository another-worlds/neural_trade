"""M3 step 2 (legacy env, py3.10): Kronos-small sampled paths, N paths per anchor, individual (NOT averaged) paths.
Same preprocessing and sampling loop as KronosPredictor.predict / auto_regressive_inference of the clone (commit 67b630e), which only differs in
returning every sampled path instead of their mean. Context = 400 five-minute bars, pred_len 12, T=1.0, top_p=0.9 (predictor defaults), top_k=0.
Resumable chunks of CH anchors -> D:/nt/nt_t14_cache/m3/paths_{tag}_{source}/chunk_xxxxx.npz (paths [CH,N,12,4] open high low close, USD).
usage: run_kronos.py --source cut --n 4 --tag n4 [--stride k] [--limit n] [--max-hours h]"""
import os, sys, time, argparse
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
sys.path.insert(0, "D:/nt/nt_mm_cache/src_kronos")
import numpy as np, pandas as pd, torch
from model import Kronos, KronosTokenizer
from model.kronos import sample_from_logits, calc_time_stamps

CACHE = "D:/nt/nt_t14_cache/m3"; CTXMAX = 512; PRED = 12; CLIP = 5


def load():
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"); m = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return m.to("cuda:0").eval(), tok.to("cuda:0").eval()


def prep(x5, ts_ns):
    """x5 [B,L,5] ohlcv float64, ts_ns [B,L] -> normalised x [B,L,6], x_stamp [B,L,5], y_stamp [B,PRED,5], mean, std (per anchor, 6)."""
    B, L, _ = x5.shape
    amt = x5[:, :, 4] * x5[:, :, :4].mean(axis=2)
    x = np.concatenate([x5, amt[:, :, None]], axis=2).astype(np.float32)
    mean = x.mean(axis=1, keepdims=True); std = x.std(axis=1, keepdims=True)
    xn = np.clip((x - mean) / (std + 1e-5), -CLIP, CLIP)
    xs, ys = [], []
    for b in range(B):
        xt = pd.Series(pd.to_datetime(ts_ns[b])); yt = pd.Series(pd.date_range(xt.iloc[-1] + pd.Timedelta(minutes=5), periods=PRED, freq="5min"))
        xs.append(calc_time_stamps(xt).values.astype(np.float32)); ys.append(calc_time_stamps(yt).values.astype(np.float32))
    return xn, np.stack(xs), np.stack(ys), mean, std


@torch.no_grad()
def sample_paths(model, tok, xn, xs, ys, mean, std, n, T=1.0, top_k=0, top_p=0.9):
    dev = "cuda:0"
    x = torch.from_numpy(xn).to(dev); x_stamp = torch.from_numpy(xs).to(dev); y_stamp = torch.from_numpy(ys).to(dev)
    B = x.size(0)
    x = x.unsqueeze(1).repeat(1, n, 1, 1).reshape(-1, x.size(1), x.size(2))
    x_stamp = x_stamp.unsqueeze(1).repeat(1, n, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
    y_stamp = y_stamp.unsqueeze(1).repeat(1, n, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2))
    x_token = tok.encode(x, half=True)
    L = x.size(1); bs = x_token[0].size(0); total = L + PRED
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
    gpre = x_token[0].new_empty(bs, PRED); gpost = x_token[1].new_empty(bs, PRED)
    pre_buf = x_token[0].new_zeros(bs, CTXMAX); post_buf = x_token[1].new_zeros(bs, CTXMAX)
    blen = min(L, CTXMAX); st = max(0, L - CTXMAX)
    pre_buf[:, :blen] = x_token[0][:, st:st + blen]; post_buf[:, :blen] = x_token[1][:, st:st + blen]
    for i in range(PRED):
        cur = L + i; wl = min(cur, CTXMAX)
        inp = [pre_buf[:, :wl], post_buf[:, :wl]] if cur <= CTXMAX else [pre_buf, post_buf]
        cs = max(0, cur - CTXMAX)
        s1_logits, ctx = model.decode_s1(inp[0], inp[1], full_stamp[:, cs:cur, :].contiguous())
        sp = sample_from_logits(s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        s2_logits = model.decode_s2(ctx, sp)
        sq = sample_from_logits(s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
        gpre[:, i] = sp.squeeze(-1); gpost[:, i] = sq.squeeze(-1)
        if cur < CTXMAX:
            pre_buf[:, cur] = sp.squeeze(-1); post_buf[:, cur] = sq.squeeze(-1)
        else:
            pre_buf.copy_(torch.roll(pre_buf, -1, 1)); post_buf.copy_(torch.roll(post_buf, -1, 1)); pre_buf[:, -1] = sp.squeeze(-1); post_buf[:, -1] = sq.squeeze(-1)
    fp = torch.cat([x_token[0], gpre], 1); fq = torch.cat([x_token[1], gpost], 1)
    cst = max(0, total - CTXMAX)
    z = tok.decode([fp[:, cst:total].contiguous(), fq[:, cst:total].contiguous()], half=True)
    z = z.reshape(B, n, z.size(1), z.size(2))[:, :, -PRED:, :].float().cpu().numpy()          # [B,n,PRED,6] normalised
    z = z * (std[:, None, :, :] + 1e-5) + mean[:, None, :, :]
    return z[..., :4].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--n", type=int, default=4); ap.add_argument("--tag", default="n4")
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-hours", type=float, default=99.0); ap.add_argument("--seed", type=int, default=20261017); ap.add_argument("--bench", action="store_true"); ap.add_argument("--amp", default="none")
    a = ap.parse_args()
    try:
        import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL
    except Exception: pass
    z = np.load(os.path.join(CACHE, f"inputs_{a.source}.npz")); ta, X, TS = z["ta_ns"], z["x"], z["ts_ns"]
    sel = np.arange(0, len(ta), a.stride)
    if a.limit: sel = sel[:a.limit]
    out = os.path.join(CACHE, f"paths_{a.tag}_{a.source}"); os.makedirs(out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = a.amp in ("tf32", "bf16"); torch.backends.cudnn.allow_tf32 = a.amp in ("tf32", "bf16")
    model, tok = load(); t0 = time.time(); done = 0; ngp = 0.0
    for ci, s in enumerate(range(0, len(sel), a.chunk)):
        fn = os.path.join(out, f"chunk_{ci:05d}.npz")
        if os.path.exists(fn): continue
        if (time.time() - t0) / 3600 > a.max_hours: print("time budget reached"); break
        idx = sel[s:s + a.chunk]; xn, xs, ys, mean, std = prep(X[idx], TS[idx])
        torch.manual_seed(a.seed + ci)
        tt = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.amp == "bf16")):
            p = sample_paths(model, tok, xn, xs, ys, mean, std, a.n)
        torch.cuda.synchronize(); dt = time.time() - tt
        if not a.bench: np.savez(fn, ta_ns=ta[idx], paths=p)
        done += len(idx); ngp += dt
        if ci % 10 == 0 or a.bench: print(f"{a.source} chunk {ci} anchors {done}/{len(sel)} gpu-sec/chunk {dt:.2f} elapsed {(time.time()-t0)/60:.1f} min peakGB {torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
        if a.bench and ci >= 2: break
    print("DONE", a.source, "anchors", done, "gpu-sec", round(ngp, 1), "wall-sec", round(time.time() - t0, 1))

if __name__ == "__main__":
    main()
