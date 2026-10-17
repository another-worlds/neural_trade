"""Build the M5 views. D:/nt/nt_mm_env/python.exe build_view.py [--limit N] [--thin K]
Zero-shot VisionTS (MAE base), two series per anchor in one pass (nvars = 2), 5-min bars, period 288, context 2016, pred 12. Writes M5_visionts and
M5b_visionts_seasonal (post hoc: debiased hourly log RV forecast minus the 7-day same-hour mean of hourly log RV, ONE column). One GPU process, below-normal priority."""
import argparse, ctypes, json, os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN, view_cache as VC
import m5_lib as L
CKPT = "D:/nt/nt_mm_cache/visionts_ckpt/"


def lower_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception as e:
        print("priority not set", e)


def load_model():
    from visionts import VisionTS
    m = VisionTS(arch="mae_base", finetune_type="ln", ckpt_dir=CKPT, load_ckpt=True)
    m.update_config(context_len=L.CTX, pred_len=L.HOR, periodicity=L.PERIOD)
    return m.cuda().eval()


def predict(model, X, batch=64, pr=print):
    import torch
    outs = []; t0 = time.time()
    with torch.no_grad():
        for j, i in enumerate(range(0, len(X), batch)):
            x = torch.tensor(X[i:i + batch], dtype=torch.float32).cuda()          # [B, 2016, 2]
            outs.append(model(x, fp64=False).float().cpu().numpy())               # [B, 12, 2]
            if j % 50 == 0:
                pr(f"  {i}/{len(X)} {time.time() - t0:.0f}s")
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--thin", type=int, default=1)
    a = ap.parse_args(); lower_priority()
    log = {"thin": a.thin, "sources": {}}; t_all = time.time(); pr = lambda s: print(s, flush=True)
    model = load_model()
    for src in ("cut", "fwd"):
        anc = AN.load_anchors(src); ta_all = anc.index
        if a.thin > 1:
            ta_all = ta_all[::a.thin]
        if a.limit:
            ta_all = ta_all[-a.limit:]
        store = AN.get_store(src); frame = L.make_frame(store)
        items = [(t, L.context(frame, t)) for t in ta_all]; good = [(t, c[1]) for t, c in items if c is not None]
        X = np.stack([c for _, c in good]); done_t = pd.DatetimeIndex([t for t, _ in good]); pr(f"{src} anchors {len(ta_all)} with input {len(good)}")
        t0 = time.time(); Y = predict(model, X, pr=pr); dt = time.time() - t0
        log["sources"][src] = {"n_anchors_grid": int(len(anc)), "n_requested": int(len(ta_all)), "n": len(good), "seconds": dt, "anchors_per_s": len(good) / dt}
        pr(f"{src} {dt:.1f}s {len(good) / dt:.1f} anchors/s")
        if a.limit:
            continue
        np.savez_compressed(f"{AN.CACHE_DIR}/m5_raw_{src}.npz", ta=done_t.values, Y=Y.astype(np.float32))
        base = AN.load_base(src); y = base.A["T"]["A"] + base.A["LV0"]
        pos = pd.Series(np.arange(len(base.ta)), index=pd.DatetimeIndex(base.ta)).reindex(done_t).to_numpy()
        yy, okk = y[pos], base.ok[pos]
        raw = L.hourly_from_bars(Y[:, :, 0])
        c, n_c, span = L.debias_constant(done_t, raw, yy, okk, "2025-11-01" if src == "fwd" else None)
        log["sources"][src].update({"debias": c, "debias_n": n_c, "debias_span": span}); pr(f"{src} debias {c:.4f} from {n_c} training anchors {span}")
        cols = L.to_columns(Y[:, :, 0], Y[:, :, 1], X[:, -1, 0], c)
        df = pd.DataFrame(cols, index=done_t.rename("ta"))
        meta = VC.write_view("M5_visionts", src, df, note=f"VisionTS MAE-base zero-shot, 2 series (ln rv5, ln(1+vol)), 5-min bars, period 288, ctx 2016, pred 12; thin {a.thin}; point_log_rv = implied hourly log RV + constant {c:.4f} (training anchors only)")
        pr(json.dumps(meta))
        seas = L.seasonal7(store, done_t)
        d2 = pd.DataFrame({"vts_seasonal_resid": cols["point_log_rv"] - seas}, index=done_t.rename("ta"))
        log["sources"][src]["seasonal_nan"] = int(np.isnan(seas).sum())
        meta = VC.write_view("M5b_visionts_seasonal", src, d2, note="post hoc: VisionTS hourly log RV forecast (debiased) minus the 7-day same-hour-of-day mean of hourly log RV; one feature column, no point_log_rv (so no 50/50 average)")
        pr(json.dumps(meta))
    log["total_seconds"] = time.time() - t_all; log["gpu_hours"] = log["total_seconds"] / 3600
    if not a.limit:
        json.dump(log, open(os.path.join(HERE, "build_log_M5_visionts.json"), "w"), indent=1, default=str)
    pr(json.dumps(log, default=str))


if __name__ == "__main__":
    main()
