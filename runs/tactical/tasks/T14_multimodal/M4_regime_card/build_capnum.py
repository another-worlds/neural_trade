"""POST HOC diagnostic control: the numbers the caption is built from (vp, bbp, z20, dR, dS, mR, mS, wkend), as raw columns, no CLIP. Tells whether the CLIP text
embedding adds anything beyond the numbers it encodes. Run: D:/nt/nt_mm_env/python.exe build_capnum.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m4_lib  # noqa: F401  (sets sys.path for the core)
import pandas as pd, anchors as AN, view_cache as VC
for s in ("cut", "fwd"):
    d = pd.read_parquet(os.path.join(AN.CACHE_DIR, f"m4_captions_{s}.parquet"))
    d = d[d.caption != ""][["vp", "bbp", "z20", "dR", "dS", "mR", "mS", "wkend"]]; d = d[d.notna().all(axis=1)]; d.columns = ["cn_" + c for c in d.columns]
    m = VC.write_view("M4c_capnum", s, d, note="POST HOC control: raw caption numbers"); print(s, m["n_rows"], m["n_ok_anchors_covered"])
