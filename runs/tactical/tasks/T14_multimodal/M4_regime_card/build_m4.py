"""Build the M4 views (both sources) with view_cache.write_view. Run: D:/nt/nt_mm_env/python.exe build_m4.py [--stage emb|views|gbm|all]
GPU: one process (CLIP ViT-B/32 fp16, batch 256), below-normal priority. CPU rendering is single process.

Views:  M4_img (img_pc1..16), M4_cap (cap_pc1..16), M4_both (both), M4_num (num_pc1..16: PCA of the 60x5 raw OHLCV window, the information-equivalent
numeric analogue, used as --c2-view), and POST HOC M4b_img_gbm (column img_gbm_dA: LightGBM on the image PCs, monthly walk-forward, pooled over both sources).
PCA: fitted on the cut-file ok anchors of 2023-07-01..2024-06-30 only (unsupervised, before Look 1), applied to both sources. Rows with a missing piece are dropped.
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "D:/nt/nt_mm_cache/hf"); os.environ.setdefault("TORCH_HOME", "D:/nt/nt_mm_cache/torch")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m4_lib as L  # noqa: E402
import anchors as AN  # noqa: E402
import view_cache as VC  # noqa: E402

SOURCES = ("cut", "fwd")
CACHE = AN.CACHE_DIR


def log(s):
    print(s, flush=True)


def prep(source):
    """base, store arrays and the anchor positions of the rows we build (ok and a clean 240-bar window)."""
    base = AN.load_base(source); store = AN.get_store(source); m1 = store.m1; ta = base.ta
    sigma = L.S.sigma_blend(base.arr["A_S3"])
    idx = m1.index; pos = idx.searchsorted(ta, side="left"); pl = np.maximum(pos - 1, 0)
    last_ok = (pos >= 1) & np.asarray(idx[pl] == (ta - pd.Timedelta(minutes=1)))
    tmin = idx.values.astype("datetime64[m]").astype(np.int64)
    clean = L.S.clean_windows(tmin, 1, L.NB)(pos) & last_ok & np.isfinite(sigma) & (sigma > 0)
    rows = np.where(base.ok & clean)[0]
    return base, m1, ta, sigma, pos, rows


def clip_model():
    import open_clip
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    m, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai", cache_dir="D:/nt/nt_mm_cache/hf")
    return m.half().cuda().eval(), open_clip.get_tokenizer("ViT-B-32-quickgelu")


MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def embed_images(model, imgs, bs=256):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).cuda().permute(0, 3, 1, 2).float().div_(255.0)
            x = ((x - torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)) / torch.tensor(STD, device="cuda").view(1, 3, 1, 1)).half()
            out.append(model.encode_image(x).float().cpu().numpy())
    return np.concatenate(out)


def embed_text(model, tok, texts, bs=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            out.append(model.encode_text(tok(texts[i:i + bs]).cuda()).float().cpu().numpy())
    return np.concatenate(out)


def stage_emb(info):
    L.below_normal(); model, tok = clip_model()
    import torch
    for source in SOURCES:
        t0 = time.time(); base, m1, ta, sigma, pos, rows = prep(source)
        o, h, l, c, v = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        torch.cuda.reset_peak_memory_stats()
        emb = np.full((len(ta), 512), np.nan, np.float32); t_render = 0.0; t_gpu = 0.0
        for sl, imgs in _timed(L.render_images(o, h, l, c, v, pos[rows], sigma[rows])):
            tr = time.time(); e = embed_images(model, imgs); t_gpu += time.time() - tr
            emb[rows[sl]] = e
        tot = time.time() - t0
        log(f"{source}: {len(rows)} images embedded in {tot:.0f}s ({len(rows) / tot:.0f}/s), gpu part {t_gpu:.0f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        np.savez_compressed(os.path.join(CACHE, f"m4_img_emb_{source}.npz"), emb=emb)
        info.setdefault("emb_img", {})[source] = {"n_images": int(len(rows)), "seconds": round(tot, 1), "images_per_s": round(len(rows) / tot, 1),
                                                  "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
        # captions
        t1 = time.time()
        srv = L.S.sr_block(m1["high"].to_numpy(float), m1["low"].to_numpy(float), c, pos[rows], sigma[rows], 1, L.NB)
        cols = {nm: np.full(len(ta), np.nan) for nm in ("s2_dR", "s2_dS", "s2_missR", "s2_missS")}
        for nm in cols:
            cols[nm][rows] = srv[:, L.S.SR_NAMES.index(nm)]
        cf = L.caption_frame(m1, ta, sigma, cols)
        caps = np.array([L.caption(r) if (i in set_rows) else None for i, r in enumerate(cf.itertuples())]) if False else None
        set_rows = np.zeros(len(ta), bool); set_rows[rows] = True
        texts = [(L.caption(r) if set_rows[i] else None) for i, r in enumerate(cf.itertuples())]
        have = np.array([t is not None for t in texts])
        uniq = sorted(set(t for t in texts if t is not None)); lut = {t: i for i, t in enumerate(uniq)}
        ue = embed_text(model, tok, uniq)
        temb = np.full((len(ta), 512), np.nan, np.float32)
        for i in np.where(have)[0]:
            temb[i] = ue[lut[texts[i]]]
        np.savez_compressed(os.path.join(CACHE, f"m4_cap_emb_{source}.npz"), emb=temb)
        pd.DataFrame({"caption": [t if t is not None else "" for t in texts]}, index=ta).join(cf).to_parquet(os.path.join(CACHE, f"m4_captions_{source}.parquet"))
        info["emb_img"][source].update({"n_captions": int(have.sum()), "n_unique_captions": len(uniq), "caption_seconds": round(time.time() - t1, 1)})
        log(f"{source}: {int(have.sum())} captions ({len(uniq)} unique) in {time.time() - t1:.0f}s")


def _timed(gen):
    for x in gen:
        yield x


def stage_views(info):
    emb = {s: {k: np.load(os.path.join(CACHE, f"m4_{k}_emb_{s}.npz"))["emb"] for k in ("img", "cap")} for s in SOURCES}
    bases = {s: AN.load_base(s) for s in SOURCES}
    pca = {}
    for k in ("img", "cap"):
        mu, comps, evr, n = L.fit_pca(emb["cut"][k], bases["cut"].ta, bases["cut"].ok)
        pca[k] = (mu, comps); info.setdefault("pca", {})[k] = {"fit_anchors": n, "explained_ratio_16": [round(float(x), 5) for x in evr], "sum": round(float(evr.sum()), 4),
                                                              "range": [str(L.PCA_LO), str(L.PCA_HI)]}
    # numeric analogue: standardise then PCA, both fitted on the same anchors
    num = {}
    for s in SOURCES:
        base, m1, ta, sigma, pos, rows = prep(s)
        o, h, l, c, v = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        X = np.full((len(ta), 5 * L.NUM_BARS), np.nan)
        X[rows] = L.num_window(o, h, l, c, v, pos[rows]); num[s] = X
    mu0 = np.nanmean(num["cut"][_fit_mask(bases["cut"])], 0); sd0 = np.nanstd(num["cut"][_fit_mask(bases["cut"])], 0) + 1e-9
    std = {s: (num[s] - mu0) / sd0 for s in SOURCES}
    mun, cn, evn, nn = L.fit_pca(std["cut"], bases["cut"].ta, bases["cut"].ok)
    info["pca"]["num"] = {"fit_anchors": nn, "explained_ratio_16": [round(float(x), 5) for x in evn], "sum": round(float(evn.sum()), 4)}
    names = {"img": [f"img_pc{i + 1}" for i in range(16)], "cap": [f"cap_pc{i + 1}" for i in range(16)], "num": [f"num_pc{i + 1}" for i in range(16)]}
    pcs = {}
    for s in SOURCES:
        ta = bases[s].ta
        pcs[s] = {"img": L.apply_pca(emb[s]["img"].astype(np.float64), *pca["img"]), "cap": L.apply_pca(emb[s]["cap"].astype(np.float64), *pca["cap"]),
                  "num": L.apply_pca(std[s], mun, cn)}
    np.savez_compressed(os.path.join(CACHE, "m4_pca_params.npz"), img_mu=pca["img"][0], img_c=pca["img"][1], cap_mu=pca["cap"][0], cap_c=pca["cap"][1],
                        num_mu0=mu0, num_sd0=sd0, num_mu=mun, num_c=cn)
    for s in SOURCES:
        ta = bases[s].ta
        out = {"M4_img": pd.DataFrame(pcs[s]["img"], index=ta, columns=names["img"]), "M4_cap": pd.DataFrame(pcs[s]["cap"], index=ta, columns=names["cap"]),
               "M4_num": pd.DataFrame(pcs[s]["num"], index=ta, columns=names["num"])}
        out["M4_both"] = pd.concat([out["M4_img"], out["M4_cap"]], axis=1)
        for nm, df in out.items():
            df = df[df.notna().all(axis=1)]
            meta = VC.write_view(nm, s, df, note=f"M4 {nm}; PCA 16 fitted on cut ok anchors {L.PCA_LO.date()}..{L.PCA_HI.date()}")
            info.setdefault("views", {}).setdefault(nm, {})[s] = {"n_rows": meta["n_rows"], "n_ok_anchors_covered": meta["n_ok_anchors_covered"], "sha256": meta["sha256"]}
            log(f"  wrote {nm} {s}: {meta['n_rows']} rows, covered {meta['n_ok_anchors_covered']}/{meta['n_ok_anchors']}")


def _fit_mask(base):
    ta = base.ta
    return np.asarray((ta >= L.PCA_LO) & (ta <= L.PCA_HI)) & base.ok


def stage_gbm(info):
    L.below_normal(); bases = {s: AN.load_base(s) for s in SOURCES}
    ta = bases["cut"].ta.append(bases["fwd"].ta); n = len(bases["cut"].ta)
    X = np.vstack([read_pc(s) for s in SOURCES]); y = np.concatenate([bases[s].arr["A_dA"].astype(float) for s in SOURCES])
    ok = np.concatenate([bases[s].ok for s in SOURCES]) & np.isfinite(X).all(1) & np.isfinite(y)
    t0 = time.time(); pred, mi = L.monthly_gbm(ta, X, y, ok, log=log)
    info["gbm"] = {"months": mi, "seconds": round(time.time() - t0, 1), "pool_rows": int(ok.sum())}
    for s, sl in (("cut", slice(0, n)), ("fwd", slice(n, None))):
        df = pd.DataFrame({"img_gbm_dA": pred[sl]}, index=ta[sl]); df = df[df.notna().all(axis=1)]
        meta = VC.write_view("M4b_img_gbm", s, df, note="POST HOC: LightGBM on M4_img PCs -> next-hour change in log RV; monthly walk-forward, pool = both sources, 365 d, labels ended before the month")
        info.setdefault("views", {}).setdefault("M4b_img_gbm", {})[s] = {"n_rows": meta["n_rows"], "n_ok_anchors_covered": meta["n_ok_anchors_covered"], "sha256": meta["sha256"]}
        log(f"  wrote M4b_img_gbm {s}: {meta['n_rows']} rows, covered {meta['n_ok_anchors_covered']}/{meta['n_ok_anchors']}")


def read_pc(s):
    df = VC.read_view("M4_img", s).reindex(AN.load_anchors(s).index)
    return df.to_numpy(float)


if __name__ == "__main__":
    stage = (sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all")
    p = os.path.join(HERE, "build_info.json"); info = json.load(open(p)) if os.path.exists(p) else {}
    t0 = time.time()
    if stage in ("emb", "all"):
        stage_emb(info)
    if stage in ("views", "all"):
        stage_views(info)
    if stage in ("gbm", "all"):
        stage_gbm(info)
    info.setdefault("stage_seconds", {})[stage] = round(time.time() - t0, 1)
    json.dump(info, open(p, "w"), indent=1, default=str)
