"""Build M2 views. D:/nt/nt_mm_env/python.exe build_view.py --variant M2_ttm|M2b_ttm_ft [--limit N]
Zero-shot (M2_ttm) or fine-tuned (M2b_ttm_ft, post hoc) TTM r2, context 512, prediction length 12, point forecasts. One GPU process, below normal priority."""
import argparse, ctypes, json, os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN, view_cache as VC
import m2_lib as L
CACHE = "D:/nt/nt_t14_cache"; FT_PATH = f"{CACHE}/m2_ft_weights.pt"


def FT(n):
    import torch
    return torch.full((n,), 3, dtype=torch.long).cuda()   # DEFAULT_FREQUENCY_MAPPING['5min'] = 3 (the r2.1 branch for pred 12 is the 48-step fine-tuned one, which needs a frequency token)


def lower_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception as e:
        print("priority not set", e)


def load_model():
    from tsfm_public.toolkit.get_model import get_model
    return get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=L.CTX, prediction_length=L.HOR).cuda()


def predict(model, X, batch=512):
    import torch
    model.eval(); outs = []
    z, mu, sd = L.normalise(X)
    with torch.no_grad():
        for i in range(0, len(z), batch):
            inp = torch.tensor(z[i:i + batch], dtype=torch.float32)[:, :, None].cuda()
            outs.append(model(past_values=inp, freq_token=FT(len(inp))).prediction_outputs[:, :L.HOR, 0].float().cpu().numpy())
    return np.concatenate(outs) * sd + mu


def finetune(model, log):
    """Fine-tune on the first 20% of the 2023-07..2024-06 cut anchors' windows (5-minute stride, targets = the next 12 bars of ln rv5);
    the last 15% of those windows (after a 512-window purge gap) pick the epoch. Every label ends long before 2024-07-01."""
    import torch
    frame = L.IN.make_frame(AN.get_store("cut")); idx = frame.index
    t_lo, t_hi = pd.Timestamp("2023-07-01"), pd.Timestamp("2024-07-01")
    t_cut = t_lo + 0.20 * (t_hi - t_lo)
    tgt = frame["target"].to_numpy(float)
    ends = np.where((idx >= t_lo + pd.Timedelta(minutes=5 * 12)) & (idx + pd.Timedelta(minutes=5 * L.HOR) <= t_cut))[0]
    ends = ends[ends >= L.CTX]
    Xs, Ys = [], []
    for e in ends:
        c = tgt[e - L.CTX:e]; y = tgt[e:e + L.HOR]
        if np.isfinite(c).sum() < 256 or len(y) < L.HOR or not np.isfinite(y).all():
            continue
        Xs.append(pd.Series(c).ffill().bfill().to_numpy()); Ys.append(y)
    X = np.array(Xs); Y = np.array(Ys); n = len(X); nva0 = int(n * 0.85); ntr = nva0 - L.CTX
    z, mu, sd = L.normalise(X); zy = (Y - mu) / sd
    Xtr, Ytr, Xva, Yva = z[:ntr], zy[:ntr], z[nva0:], zy[nva0:]
    log(f"fine-tune windows {n}, train {len(Xtr)}, val {len(Xva)}, span to {t_cut}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    def vloss():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Xva), 512):
                p = model(past_values=torch.tensor(Xva[i:i + 512], dtype=torch.float32)[:, :, None].cuda(), freq_token=FT(len(Xva[i:i + 512]))).prediction_outputs[:, :L.HOR, 0].cpu()
                tot += float(((p - torch.tensor(Yva[i:i + 512], dtype=torch.float32)) ** 2).sum())
        return tot / Yva.size
    best = vloss(); best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; rng = np.random.default_rng(0)
    log(f"epoch 0 (zero-shot) val mse {best:.4f}"); hist = [("0", best)]
    for ep in range(1, 6):
        model.train(); perm = rng.permutation(len(Xtr)); tl = 0.0
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = torch.tensor(Xtr[b], dtype=torch.float32)[:, :, None].cuda(); yb = torch.tensor(Ytr[b], dtype=torch.float32)[:, :, None].cuda()
            loss = model(past_values=xb, future_values=yb, freq_token=FT(len(xb))).loss; opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss) * len(b)
        v = vloss(); hist.append((str(ep), v)); log(f"epoch {ep} train mse {tl / len(Xtr):.4f} val mse {v:.4f}")
        if v < best:
            best = v; best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save(best_state, FT_PATH)
    return {"n_windows": int(n), "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "span_end": str(t_cut), "val_mse_by_epoch": hist, "best_val_mse": best}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); lower_priority()
    log = {"variant": a.variant, "sources": {}}; t_all = time.time()

    def pr(s):
        print(s, flush=True)
    model = load_model()
    if a.variant == "M2b_ttm_ft":
        t0 = time.time(); log["finetune"] = finetune(model, pr); log["finetune"]["seconds"] = time.time() - t0
    for src in ("cut", "fwd"):
        anc = AN.load_anchors(src); ta_all = anc.index if not a.limit else anc.index[:a.limit]
        frame = L.IN.make_frame(AN.get_store(src))
        items = [(t, L.context(frame, t)) for t in ta_all]; good = [(t, c[1]) for t, c in items if c is not None]
        X = np.array([c for _, c in good]); done_t = pd.DatetimeIndex([t for t, _ in good]); pr(f"{src} anchors {len(ta_all)} with input {len(good)}")
        t0 = time.time(); fc = predict(model, X); dt = time.time() - t0
        log["sources"][src] = {"n": len(good), "seconds": dt, "anchors_per_s": len(good) / dt}; pr(f"{src} {dt:.1f}s {len(good) / dt:.0f} anchors/s")
        base = AN.load_base(src); y = base.A["T"]["A"] + base.A["LV0"]
        pos = pd.Series(np.arange(len(base.ta)), index=pd.DatetimeIndex(base.ta)).reindex(done_t).to_numpy()
        yy = y[pos]; okk = base.ok[pos]
        raw = L.hourly_from_bars(fc)
        cap = "2025-11-01" if src == "fwd" else None
        c, n_c, span = L.debias_constant(done_t, raw, yy, okk, cap)
        log["sources"][src].update({"debias": c, "debias_n": n_c, "debias_span": span}); pr(f"{src} debias {c:.4f} from {n_c} training anchors {span}")
        df = pd.DataFrame(L.to_columns(fc, X[:, -1], c), index=done_t.rename("ta"))
        if not a.limit:
            kind = "fine-tuned on the first 20% of 2023-07..2024-06 cut windows, post hoc" if a.variant != "M2_ttm" else "zero-shot"
            meta = VC.write_view(a.variant, src, df, note=f"TTM r2 ({kind}), ctx 512 five-min ln rv5, pred 12, per-window standardisation; point_log_rv = implied hourly log RV + constant {c:.4f} (training anchors only)")
            pr(json.dumps(meta))
    log["total_seconds"] = time.time() - t_all; log["gpu_hours"] = log["total_seconds"] / 3600
    if not a.limit:
        json.dump(log, open(os.path.join(HERE, f"build_log_{a.variant}.json"), "w"), indent=1, default=str)
    pr(json.dumps(log, default=str))


if __name__ == "__main__":
    main()
