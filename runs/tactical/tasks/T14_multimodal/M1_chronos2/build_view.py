"""Build the M1 / M1b views. Usage (from anywhere): HF_HOME=D:/nt/nt_mm_cache/hf D:/nt/nt_mm_env/python.exe build_view.py --variant M1_chronos2|M1b_chronos2_har [--limit N]
One GPU process, below-normal priority, raw forecasts cached to D:/nt/nt_t14_cache/m1_raw_{variant}_{source}.npz (resumable per source)."""
import argparse, ctypes, json, os, sys, time
os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN, view_cache as VC
import inputs as IN

def lower_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception as e:
        print("priority not set", e)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args(); lower_priority(); har = a.variant == "M1b_chronos2_har"
    import torch
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    log = {"variant": a.variant, "sources": {}}; t_all = time.time()
    for src in ("cut", "fwd"):
        anc = AN.load_anchors(src); ta_all = anc.index if not a.limit else anc.index[:a.limit]
        frame = IN.make_frame(AN.get_store(src), har=har)
        raw_p = f"D:/nt/nt_t14_cache/m1_raw_{a.variant}_{src}.npz"
        wins = [(t, IN.window(frame, t)) for t in ta_all]; good = [(t, w) for t, w in wins if w is not None]
        print(src, "anchors", len(ta_all), "with input", len(good), flush=True)
        if os.path.exists(raw_p) and not a.limit:
            z = np.load(raw_p); done_t = pd.DatetimeIndex(z["ta_ns"].astype("datetime64[ns]")); raw = z["q"]
        else:
            t0 = time.time(); outs = []
            names = list(good[0][1]["past"].keys())
            for i in range(0, len(good), a.batch):
                chunk = good[i:i + a.batch]
                inp = [{"target": w["target"], "past_covariates": w["past"], "future_covariates": w["future"]} for _, w in chunk]
                q, _ = pipe.predict_quantiles(inp, prediction_length=IN.HOR, quantile_levels=IN.QS, batch_size=a.batch)
                outs.extend([x.float().cpu().numpy()[0] for x in q])   # [12, 3]
                if (i // a.batch) % 20 == 0:
                    print(src, i, len(good), f"{time.time() - t0:.0f}s", flush=True)
            raw = np.stack(outs).transpose(0, 2, 1)                  # [n, 3, 12]
            done_t = pd.DatetimeIndex([t for t, _ in good])
            if not a.limit:
                np.savez_compressed(raw_p, ta_ns=np.asarray(done_t).astype("datetime64[ns]").astype(np.int64), q=raw)
            dt = time.time() - t0
            log["sources"][src] = {"n": len(good), "seconds": dt, "anchors_per_s": len(good) / dt}
            print(src, "done", f"{dt:.0f}s", f"{len(good) / dt:.1f} anchors/s", flush=True)
        cols = IN.to_columns(IN.hourly_from_bars(raw))
        df = pd.DataFrame(cols, index=pd.DatetimeIndex(done_t, name="ta"))
        if not a.limit:
            meta = VC.write_view(a.variant, src, df, note=f"Chronos-2 120M zero-shot, ctx {IN.CTX} 5-min logRV bars, horizon {IN.HOR}, q .1/.5/.9; covariates lrange, lvol, tod" + ("; + HAR components rm_day, rm_week (post hoc variant)" if har else ""))
            print(json.dumps(meta), flush=True)
    log["total_seconds"] = time.time() - t_all; log["gpu_hours"] = log["total_seconds"] / 3600
    if not a.limit:
        json.dump(log, open(os.path.join(HERE, f"build_log_{a.variant}.json"), "w"), indent=1)
    print(json.dumps(log))

if __name__ == "__main__":
    main()
